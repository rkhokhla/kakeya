package crr

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Reader handles cross-region WAL segment replay (Phase 5 WP1)
// Downloads segments from remote region and applies them idempotently
type Reader struct {
	mu             sync.RWMutex
	remoteBucket   string        // S3/GCS bucket for CRR
	sourceRegion   string        // Source region identifier
	localRegion    string        // Local/target region identifier
	downloader     StorageDownloader
	stagingDir     string        // Local directory for downloaded segments
	watermarkFile  string        // Tracks last replayed segment
	lastReplayed   string        // Last successfully replayed segment name
	replayInterval time.Duration // How often to check for new segments
	replayHandler  ReplayHandler // Callback to apply entries to local backend
	metrics        *ReaderMetrics
	stopCh         chan struct{}
	wg             sync.WaitGroup
}

// ReaderMetrics tracks replay operations (Phase 5 WP1)
type ReaderMetrics struct {
	mu                  sync.RWMutex
	SegmentsReplayed    int64
	EntriesApplied      int64
	EntriesSkipped      int64 // Duplicates (idempotency guard)
	ReplayErrors        int64
	LastReplayTimestamp time.Time
	LagSeconds          float64
}

// StorageDownloader abstracts S3/GCS download operations
type StorageDownloader interface {
	Download(ctx context.Context, remotePath, localPath string) error
	ListObjects(ctx context.Context, prefix string) ([]string, error)
}

// ReplayHandler applies a single PCS entry to the local backend
// Returns nil if entry was applied successfully, or if entry was skipped due to idempotency
type ReplayHandler interface {
	Apply(ctx context.Context, entry []byte) error
	IsIdempotent(ctx context.Context, pcsID string) (bool, error) // Check if pcs_id already exists
}

// ReaderConfig holds reader configuration
type ReaderConfig struct {
	RemoteBucket   string
	SourceRegion   string
	LocalRegion    string
	Downloader     StorageDownloader
	StagingDir     string
	ReplayHandler  ReplayHandler
	ReplayInterval time.Duration
}

// NewReader creates a new WAL reader for cross-region replication
func NewReader(config ReaderConfig) (*Reader, error) {
	if config.RemoteBucket == "" {
		return nil, fmt.Errorf("RemoteBucket is required")
	}
	if config.Downloader == nil {
		return nil, fmt.Errorf("Downloader is required")
	}
	if config.ReplayHandler == nil {
		return nil, fmt.Errorf("ReplayHandler is required")
	}

	if config.StagingDir == "" {
		config.StagingDir = "./staging-crr"
	}
	if config.ReplayInterval == 0 {
		config.ReplayInterval = 30 * time.Second // Default: check every 30s
	}

	// Ensure staging directory exists
	if err := os.MkdirAll(config.StagingDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create staging dir: %w", err)
	}

	watermarkFile := filepath.Join(config.StagingDir, ".replay-watermark")

	reader := &Reader{
		remoteBucket:   config.RemoteBucket,
		sourceRegion:   config.SourceRegion,
		localRegion:    config.LocalRegion,
		downloader:     config.Downloader,
		stagingDir:     config.StagingDir,
		watermarkFile:  watermarkFile,
		replayInterval: config.ReplayInterval,
		replayHandler:  config.ReplayHandler,
		metrics:        &ReaderMetrics{},
		stopCh:         make(chan struct{}),
	}

	// Load watermark from disk (resume from last replayed)
	if err := reader.loadWatermark(); err != nil {
		fmt.Printf("WAL CRR Reader: could not load watermark: %v (starting fresh)\n", err)
	}

	return reader, nil
}

// Start begins the replay loop (runs in background)
func (r *Reader) Start(ctx context.Context) {
	r.wg.Add(1)
	go r.replayLoop(ctx)
	fmt.Printf("WAL CRR Reader: started (region %s ← %s)\n", r.localRegion, r.sourceRegion)
}

// Stop gracefully stops the reader
func (r *Reader) Stop() {
	close(r.stopCh)
	r.wg.Wait()
	fmt.Printf("WAL CRR Reader: stopped\n")
}

// replayLoop continuously checks for new segments and replays them
func (r *Reader) replayLoop(ctx context.Context) {
	defer r.wg.Done()

	ticker := time.NewTicker(r.replayInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-r.stopCh:
			return
		case <-ticker.C:
			if err := r.replayPendingSegments(ctx); err != nil {
				fmt.Printf("WAL CRR Reader: error replaying segments: %v\n", err)
				r.metrics.mu.Lock()
				r.metrics.ReplayErrors++
				r.metrics.mu.Unlock()
			}
		}
	}
}

// replayPendingSegments finds and replays all segments after last watermark
func (r *Reader) replayPendingSegments(ctx context.Context) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// List all remote segments for this region pair
	prefix := fmt.Sprintf("crr/%s/%s/", r.sourceRegion, r.localRegion)
	remoteSegments, err := r.downloader.ListObjects(ctx, prefix)
	if err != nil {
		return fmt.Errorf("failed to list remote segments: %w", err)
	}

	// Filter out manifests (we only want .jsonl files)
	var segments []string
	for _, seg := range remoteSegments {
		if filepath.Ext(seg) == ".jsonl" {
			segments = append(segments, filepath.Base(seg))
		}
	}

	// Filter segments after last replayed
	pending := r.filterPendingSegments(segments)
	if len(pending) == 0 {
		return nil // Nothing to replay
	}

	// Replay segments in order
	for _, segment := range pending {
		if err := r.replaySegment(ctx, segment, prefix); err != nil {
			return fmt.Errorf("failed to replay segment %s: %w", segment, err)
		}

		// Update watermark after successful replay
		r.lastReplayed = segment
		if err := r.saveWatermark(); err != nil {
			fmt.Printf("WAL CRR Reader: warning: failed to save watermark: %v\n", err)
		}

		r.metrics.mu.Lock()
		r.metrics.SegmentsReplayed++
		r.metrics.LastReplayTimestamp = time.Now()
		r.metrics.mu.Unlock()
	}

	return nil
}

// replaySegment downloads and applies a single WAL segment
func (r *Reader) replaySegment(ctx context.Context, segmentName, prefix string) error {
	remotePath := filepath.Join(prefix, segmentName)
	localPath := filepath.Join(r.stagingDir, segmentName)

	// Download segment
	if err := r.downloader.Download(ctx, remotePath, localPath); err != nil {
		return fmt.Errorf("failed to download segment: %w", err)
	}
	defer os.Remove(localPath) // Clean up after replay

	// Verify checksum from manifest (if available)
	manifestPath := remotePath + ".manifest"
	manifestLocal := localPath + ".manifest"
	if err := r.downloader.Download(ctx, manifestPath, manifestLocal); err == nil {
		defer os.Remove(manifestLocal)
		if err := r.verifyChecksum(localPath, manifestLocal); err != nil {
			return fmt.Errorf("checksum verification failed: %w", err)
		}
	} else {
		fmt.Printf("WAL CRR Reader: warning: manifest not found for %s (skipping checksum)\n", segmentName)
	}

	// Replay entries from segment
	entriesApplied := 0
	entriesSkipped := 0

	f, err := os.Open(localPath)
	if err != nil {
		return fmt.Errorf("failed to open segment: %w", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		// Extract pcs_id for idempotency check
		var entry map[string]interface{}
		if err := json.Unmarshal(line, &entry); err != nil {
			fmt.Printf("WAL CRR Reader: warning: failed to parse entry: %v (skipping)\n", err)
			continue
		}

		pcsID, ok := entry["pcs_id"].(string)
		if !ok {
			fmt.Printf("WAL CRR Reader: warning: entry missing pcs_id (skipping)\n")
			continue
		}

		// Idempotency guard: check if already processed
		exists, err := r.replayHandler.IsIdempotent(ctx, pcsID)
		if err != nil {
			return fmt.Errorf("idempotency check failed for %s: %w", pcsID, err)
		}
		if exists {
			entriesSkipped++
			continue // Skip duplicate
		}

		// Apply entry through verify→dedup path (Phase 1 invariant preserved)
		if err := r.replayHandler.Apply(ctx, line); err != nil {
			return fmt.Errorf("failed to apply entry %s: %w", pcsID, err)
		}
		entriesApplied++
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading segment: %w", err)
	}

	r.metrics.mu.Lock()
	r.metrics.EntriesApplied += int64(entriesApplied)
	r.metrics.EntriesSkipped += int64(entriesSkipped)
	r.metrics.mu.Unlock()

	fmt.Printf("WAL CRR Reader: replayed %s (%d applied, %d skipped)\n",
		segmentName, entriesApplied, entriesSkipped)

	return nil
}

// filterPendingSegments returns segments after last watermark
func (r *Reader) filterPendingSegments(all []string) []string {
	if r.lastReplayed == "" {
		return all // Replay everything if no watermark
	}

	var pending []string
	found := false
	for _, seg := range all {
		if found {
			pending = append(pending, seg)
		}
		if seg == r.lastReplayed {
			found = true
		}
	}

	return pending
}

// verifyChecksum verifies segment against manifest
func (r *Reader) verifyChecksum(segmentPath, manifestPath string) error {
	// Read manifest
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		return err
	}

	var manifest ShipManifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return err
	}

	// Compute checksum of downloaded segment
	f, err := os.Open(segmentPath)
	if err != nil {
		return err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	checksum := hex.EncodeToString(h.Sum(nil))

	// Compare
	if checksum != manifest.Checksum {
		return fmt.Errorf("checksum mismatch: expected %s, got %s", manifest.Checksum, checksum)
	}

	return nil
}

// loadWatermark loads last replayed segment from disk
func (r *Reader) loadWatermark() error {
	data, err := os.ReadFile(r.watermarkFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No watermark yet, start fresh
		}
		return err
	}

	r.lastReplayed = string(data)
	return nil
}

// saveWatermark persists last replayed segment to disk
func (r *Reader) saveWatermark() error {
	return os.WriteFile(r.watermarkFile, []byte(r.lastReplayed), 0644)
}

// GetMetrics returns current reader metrics
func (r *Reader) GetMetrics() ReaderMetrics {
	r.metrics.mu.RLock()
	defer r.metrics.mu.RUnlock()

	// Compute lag (time since last segment was shipped vs last replay)
	// Note: This is an approximation based on segment modification times
	// For accurate lag, compare with shipper's LastShipTimestamp via shared metric store

	return *r.metrics
}
