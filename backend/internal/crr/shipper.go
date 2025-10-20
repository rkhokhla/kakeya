package crr

import (
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

// Shipper handles cross-region replication of WAL segments (Phase 5 WP1)
// Tails append-only WAL directory, ships segments to remote region with ordering guarantees
type Shipper struct {
	mu              sync.RWMutex
	walDir          string
	remoteBucket    string        // S3/GCS bucket for CRR
	regionID        string        // Source region identifier
	targetRegionID  string        // Target region identifier
	uploader        StorageUploader // S3/GCS uploader interface
	watermarkFile   string        // Local file tracking last shipped segment
	lastShipped     string        // Last successfully shipped segment name
	shipInterval    time.Duration // How often to check for new segments
	retryBackoff    time.Duration // Exponential backoff for retries
	maxRetries      int
	metrics         *ShipperMetrics
	stopCh          chan struct{}
	wg              sync.WaitGroup
}

// ShipperMetrics tracks shipping operations (Phase 5 WP1)
type ShipperMetrics struct {
	mu                sync.RWMutex
	SegmentsShipped   int64
	BytesShipped      int64
	ShipErrors        int64
	LastShipTimestamp time.Time
	LagSeconds        float64
}

// StorageUploader abstracts S3/GCS operations
type StorageUploader interface {
	Upload(ctx context.Context, localPath, remotePath string) error
	Exists(ctx context.Context, remotePath string) (bool, error)
}

// ShipperConfig holds shipper configuration
type ShipperConfig struct {
	WALDir         string
	RemoteBucket   string
	RegionID       string
	TargetRegionID string
	Uploader       StorageUploader
	ShipInterval   time.Duration
	MaxRetries     int
}

// NewShipper creates a new WAL shipper for cross-region replication
func NewShipper(config ShipperConfig) (*Shipper, error) {
	if config.WALDir == "" {
		return nil, fmt.Errorf("WALDir is required")
	}
	if config.RemoteBucket == "" {
		return nil, fmt.Errorf("RemoteBucket is required")
	}
	if config.Uploader == nil {
		return nil, fmt.Errorf("Uploader is required")
	}

	if config.ShipInterval == 0 {
		config.ShipInterval = 30 * time.Second // Default: check every 30s
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}

	watermarkFile := filepath.Join(config.WALDir, ".crr-watermark")

	shipper := &Shipper{
		walDir:         config.WALDir,
		remoteBucket:   config.RemoteBucket,
		regionID:       config.RegionID,
		targetRegionID: config.TargetRegionID,
		uploader:       config.Uploader,
		watermarkFile:  watermarkFile,
		shipInterval:   config.ShipInterval,
		retryBackoff:   1 * time.Second,
		maxRetries:     config.MaxRetries,
		metrics:        &ShipperMetrics{},
		stopCh:         make(chan struct{}),
	}

	// Load watermark from disk (resume from last shipped)
	if err := shipper.loadWatermark(); err != nil {
		// Not fatal, we'll start from the beginning
		fmt.Printf("WAL CRR Shipper: could not load watermark: %v (starting fresh)\n", err)
	}

	return shipper, nil
}

// Start begins the shipping loop (runs in background)
func (s *Shipper) Start(ctx context.Context) {
	s.wg.Add(1)
	go s.shipLoop(ctx)
	fmt.Printf("WAL CRR Shipper: started (region %s → %s)\n", s.regionID, s.targetRegionID)
}

// Stop gracefully stops the shipper
func (s *Shipper) Stop() {
	close(s.stopCh)
	s.wg.Wait()
	fmt.Printf("WAL CRR Shipper: stopped\n")
}

// shipLoop continuously tails WAL directory and ships new segments
func (s *Shipper) shipLoop(ctx context.Context) {
	defer s.wg.Done()

	ticker := time.NewTicker(s.shipInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-s.stopCh:
			return
		case <-ticker.C:
			if err := s.shipPendingSegments(ctx); err != nil {
				fmt.Printf("WAL CRR Shipper: error shipping segments: %v\n", err)
				s.metrics.mu.Lock()
				s.metrics.ShipErrors++
				s.metrics.mu.Unlock()
			}
		}
	}
}

// shipPendingSegments finds and ships all segments after last watermark
func (s *Shipper) shipPendingSegments(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// List all WAL segments in directory
	segments, err := s.listSegments()
	if err != nil {
		return fmt.Errorf("failed to list segments: %w", err)
	}

	// Filter segments after last shipped
	pending := s.filterPendingSegments(segments)
	if len(pending) == 0 {
		return nil // Nothing to ship
	}

	// Ship segments in order
	for _, segment := range pending {
		if err := s.shipSegment(ctx, segment); err != nil {
			return fmt.Errorf("failed to ship segment %s: %w", segment, err)
		}

		// Update watermark after successful ship
		s.lastShipped = segment
		if err := s.saveWatermark(); err != nil {
			fmt.Printf("WAL CRR Shipper: warning: failed to save watermark: %v\n", err)
		}

		s.metrics.mu.Lock()
		s.metrics.SegmentsShipped++
		s.metrics.LastShipTimestamp = time.Now()
		s.metrics.mu.Unlock()
	}

	return nil
}

// shipSegment ships a single WAL segment with retries
func (s *Shipper) shipSegment(ctx context.Context, segmentName string) error {
	localPath := filepath.Join(s.walDir, segmentName)
	remotePath := fmt.Sprintf("crr/%s/%s/%s", s.regionID, s.targetRegionID, segmentName)

	// Compute checksum for integrity
	checksum, size, err := s.computeChecksum(localPath)
	if err != nil {
		return fmt.Errorf("failed to compute checksum: %w", err)
	}

	// Upload with retries
	backoff := s.retryBackoff
	for attempt := 0; attempt < s.maxRetries; attempt++ {
		// Check if already exists (idempotent)
		exists, err := s.uploader.Exists(ctx, remotePath)
		if err == nil && exists {
			fmt.Printf("WAL CRR Shipper: segment %s already exists in remote (skipping)\n", segmentName)
			return nil
		}

		// Upload
		if err := s.uploader.Upload(ctx, localPath, remotePath); err != nil {
			if attempt < s.maxRetries-1 {
				fmt.Printf("WAL CRR Shipper: upload failed (attempt %d/%d): %v, retrying in %v\n",
					attempt+1, s.maxRetries, err, backoff)
				time.Sleep(backoff)
				backoff *= 2 // Exponential backoff
				continue
			}
			return fmt.Errorf("upload failed after %d attempts: %w", s.maxRetries, err)
		}

		// Upload manifest (metadata: checksum, size, timestamp)
		manifest := ShipManifest{
			SegmentName:    segmentName,
			Checksum:       checksum,
			Size:           size,
			SourceRegion:   s.regionID,
			TargetRegion:   s.targetRegionID,
			ShippedAt:      time.Now(),
		}
		if err := s.uploadManifest(ctx, remotePath+".manifest", manifest); err != nil {
			fmt.Printf("WAL CRR Shipper: warning: failed to upload manifest: %v\n", err)
		}

		s.metrics.mu.Lock()
		s.metrics.BytesShipped += size
		s.metrics.mu.Unlock()

		fmt.Printf("WAL CRR Shipper: shipped %s (%d bytes, checksum %s)\n", segmentName, size, checksum)
		return nil
	}

	return fmt.Errorf("unreachable")
}

// listSegments returns all WAL segment files in directory (sorted)
func (s *Shipper) listSegments() ([]string, error) {
	entries, err := os.ReadDir(s.walDir)
	if err != nil {
		return nil, err
	}

	var segments []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		// Only ship .jsonl files (skip watermark and other metadata)
		if filepath.Ext(entry.Name()) == ".jsonl" {
			segments = append(segments, entry.Name())
		}
	}

	return segments, nil
}

// filterPendingSegments returns segments after last watermark
func (s *Shipper) filterPendingSegments(all []string) []string {
	if s.lastShipped == "" {
		return all // Ship everything if no watermark
	}

	var pending []string
	found := false
	for _, seg := range all {
		if found {
			pending = append(pending, seg)
		}
		if seg == s.lastShipped {
			found = true
		}
	}

	return pending
}

// computeChecksum computes SHA-256 checksum of file
func (s *Shipper) computeChecksum(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()

	h := sha256.New()
	size, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}

	return hex.EncodeToString(h.Sum(nil)), size, nil
}

// loadWatermark loads last shipped segment from disk
func (s *Shipper) loadWatermark() error {
	data, err := os.ReadFile(s.watermarkFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No watermark yet, start fresh
		}
		return err
	}

	s.lastShipped = string(data)
	return nil
}

// saveWatermark persists last shipped segment to disk
func (s *Shipper) saveWatermark() error {
	return os.WriteFile(s.watermarkFile, []byte(s.lastShipped), 0644)
}

// uploadManifest uploads segment metadata
func (s *Shipper) uploadManifest(ctx context.Context, remotePath string, manifest ShipManifest) error {
	// Serialize manifest to JSON
	data, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	// Write to temp file
	tmpFile := filepath.Join(s.walDir, ".manifest.tmp")
	if err := os.WriteFile(tmpFile, data, 0644); err != nil {
		return err
	}
	defer os.Remove(tmpFile)

	// Upload
	return s.uploader.Upload(ctx, tmpFile, remotePath)
}

// GetMetrics returns current shipper metrics
func (s *Shipper) GetMetrics() ShipperMetrics {
	s.metrics.mu.RLock()
	defer s.metrics.mu.RUnlock()

	// Compute lag (time since last segment was created vs last ship)
	latestSegment, err := s.getLatestSegmentTime()
	if err == nil && !latestSegment.IsZero() && !s.metrics.LastShipTimestamp.IsZero() {
		s.metrics.LagSeconds = time.Since(latestSegment).Seconds()
	}

	return *s.metrics
}

// getLatestSegmentTime returns timestamp of most recent segment
func (s *Shipper) getLatestSegmentTime() (time.Time, error) {
	segments, err := s.listSegments()
	if err != nil || len(segments) == 0 {
		return time.Time{}, err
	}

	// Get modification time of latest segment
	latestPath := filepath.Join(s.walDir, segments[len(segments)-1])
	info, err := os.Stat(latestPath)
	if err != nil {
		return time.Time{}, err
	}

	return info.ModTime(), nil
}

// ShipManifest contains metadata about shipped segment
type ShipManifest struct {
	SegmentName  string    `json:"segment_name"`
	Checksum     string    `json:"checksum"`
	Size         int64     `json:"size"`
	SourceRegion string    `json:"source_region"`
	TargetRegion string    `json:"target_region"`
	ShippedAt    time.Time `json:"shipped_at"`
}
