package audit

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// BatchAnchoring handles batch anchoring of WORM audit segments (Phase 5 WP3)
// Builds Merkle roots from segments and writes external attestations
type BatchAnchoring struct {
	mu              sync.RWMutex
	wormStore       WORMStore
	attestationStore AttestationStore
	batchSize       int           // Number of segments per batch
	batchInterval   time.Duration // How often to anchor batches
	metrics         *AnchoringMetrics
	stopCh          chan struct{}
	wg              sync.WaitGroup
}

// AnchoringMetrics tracks anchoring operations
type AnchoringMetrics struct {
	mu                    sync.RWMutex
	BatchesAnchored       int64
	SegmentsAnchored      int64
	AttestationsWritten   int64
	AnchoringErrors       int64
	LastAnchoredAt        time.Time
	AvgBatchProcessingMs  float64
}

// WORMStore provides access to WORM audit logs
type WORMStore interface {
	ListSegments(ctx context.Context, since time.Time) ([]string, error)
	ReadSegment(ctx context.Context, segmentPath string) ([]WORMEntry, error)
	GetSegmentRoot(ctx context.Context, segmentPath string) (string, error) // Merkle root of segment
}

// WORMEntry is defined in worm.go

// AttestationStore writes external attestations (blockchain, timestamping service, etc.)
type AttestationStore interface {
	WriteAttestation(ctx context.Context, attestation Attestation) error
	ReadAttestation(ctx context.Context, batchID string) (*Attestation, error)
	ListAttestations(ctx context.Context, since time.Time) ([]Attestation, error)
}

// Attestation represents an external attestation of a batch
type Attestation struct {
	BatchID         string    `json:"batch_id"`
	SegmentPaths    []string  `json:"segment_paths"`
	SegmentRoots    []string  `json:"segment_roots"` // Merkle root per segment
	BatchRoot       string    `json:"batch_root"`    // Merkle root of segment roots
	AncoredAt       time.Time `json:"anchored_at"`
	AttestationType string    `json:"attestation_type"` // "blockchain", "timestamp", "internal"
	AttestationData string    `json:"attestation_data"` // Blockchain TX hash, timestamp token, etc.
}

// BatchAnchoringConfig holds configuration
type BatchAnchoringConfig struct {
	WORMStore        WORMStore
	AttestationStore AttestationStore
	BatchSize        int           // Default: 100 segments per batch
	BatchInterval    time.Duration // Default: 1 hour
}

// NewBatchAnchoring creates a new batch anchoring manager
func NewBatchAnchoring(config BatchAnchoringConfig) (*BatchAnchoring, error) {
	if config.WORMStore == nil {
		return nil, fmt.Errorf("WORMStore is required")
	}
	if config.AttestationStore == nil {
		return nil, fmt.Errorf("AttestationStore is required")
	}

	if config.BatchSize == 0 {
		config.BatchSize = 100 // Default: 100 segments per batch
	}
	if config.BatchInterval == 0 {
		config.BatchInterval = 1 * time.Hour // Default: anchor every hour
	}

	anchoring := &BatchAnchoring{
		wormStore:        config.WORMStore,
		attestationStore: config.AttestationStore,
		batchSize:        config.BatchSize,
		batchInterval:    config.BatchInterval,
		metrics:          &AnchoringMetrics{},
		stopCh:           make(chan struct{}),
	}

	return anchoring, nil
}

// Start begins the batch anchoring loop (runs in background)
func (b *BatchAnchoring) Start(ctx context.Context) {
	b.wg.Add(1)
	go b.anchoringLoop(ctx)
	fmt.Printf("Batch Anchoring: started (batch size %d, interval %v)\n",
		b.batchSize, b.batchInterval)
}

// Stop gracefully stops the batch anchoring
func (b *BatchAnchoring) Stop() {
	close(b.stopCh)
	b.wg.Wait()
	fmt.Printf("Batch Anchoring: stopped\n")
}

// anchoringLoop continuously anchors batches of segments
func (b *BatchAnchoring) anchoringLoop(ctx context.Context) {
	defer b.wg.Done()

	ticker := time.NewTicker(b.batchInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-b.stopCh:
			return
		case <-ticker.C:
			if err := b.anchorBatch(ctx); err != nil {
				fmt.Printf("Batch Anchoring: error anchoring batch: %v\n", err)
				b.metrics.mu.Lock()
				b.metrics.AnchoringErrors++
				b.metrics.mu.Unlock()
			}
		}
	}
}

// anchorBatch anchors a single batch of segments
func (b *BatchAnchoring) anchorBatch(ctx context.Context) error {
	start := time.Now()

	b.mu.Lock()
	defer b.mu.Unlock()

	// Get segments since last anchor (or last 1 hour if first run)
	since := b.metrics.LastAnchoredAt
	if since.IsZero() {
		since = time.Now().Add(-b.batchInterval)
	}

	segments, err := b.wormStore.ListSegments(ctx, since)
	if err != nil {
		return fmt.Errorf("failed to list segments: %w", err)
	}

	if len(segments) == 0 {
		return nil // No segments to anchor
	}

	// Process in batches
	for i := 0; i < len(segments); i += b.batchSize {
		end := i + b.batchSize
		if end > len(segments) {
			end = len(segments)
		}

		batch := segments[i:end]
		if err := b.processBatch(ctx, batch); err != nil {
			return fmt.Errorf("failed to process batch: %w", err)
		}
	}

	// Update metrics
	duration := time.Since(start).Milliseconds()

	b.metrics.mu.Lock()
	b.metrics.BatchesAnchored++
	b.metrics.SegmentsAnchored += int64(len(segments))
	b.metrics.LastAnchoredAt = time.Now()

	// Update average batch processing time (exponential moving average)
	if b.metrics.AvgBatchProcessingMs == 0 {
		b.metrics.AvgBatchProcessingMs = float64(duration)
	} else {
		b.metrics.AvgBatchProcessingMs = 0.9*b.metrics.AvgBatchProcessingMs + 0.1*float64(duration)
	}
	b.metrics.mu.Unlock()

	fmt.Printf("Batch Anchoring: anchored %d segments (%dms)\n", len(segments), duration)

	return nil
}

// processBatch processes a single batch of segments
func (b *BatchAnchoring) processBatch(ctx context.Context, segments []string) error {
	// Build segment roots
	segmentRoots := make([]string, 0, len(segments))
	for _, segmentPath := range segments {
		root, err := b.wormStore.GetSegmentRoot(ctx, segmentPath)
		if err != nil {
			return fmt.Errorf("failed to get segment root for %s: %w", segmentPath, err)
		}
		segmentRoots = append(segmentRoots, root)
	}

	// Build batch root (Merkle root of segment roots)
	batchRoot := computeBatchRoot(segmentRoots)

	// Create attestation
	batchID := fmt.Sprintf("batch-%d", time.Now().UnixNano())
	attestation := Attestation{
		BatchID:         batchID,
		SegmentPaths:    segments,
		SegmentRoots:    segmentRoots,
		BatchRoot:       batchRoot,
		AncoredAt:       time.Now(),
		AttestationType: "internal", // In production, this would be "blockchain", "timestamp", etc.
		AttestationData: fmt.Sprintf("batch-root=%s", batchRoot),
	}

	// Write attestation with retries
	maxRetries := 3
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		if err := b.attestationStore.WriteAttestation(ctx, attestation); err != nil {
			lastErr = err
			fmt.Printf("Batch Anchoring: attestation write failed (attempt %d/%d): %v\n",
				attempt+1, maxRetries, err)
			time.Sleep(time.Duration(attempt+1) * time.Second)
			continue
		}

		// Success
		b.metrics.mu.Lock()
		b.metrics.AttestationsWritten++
		b.metrics.mu.Unlock()

		fmt.Printf("Batch Anchoring: attestation written (batch %s, root %s)\n",
			batchID, batchRoot)
		return nil
	}

	return fmt.Errorf("failed to write attestation after %d attempts: %w", maxRetries, lastErr)
}

// computeBatchRoot computes the Merkle root of a list of segment roots
func computeBatchRoot(segmentRoots []string) string {
	if len(segmentRoots) == 0 {
		return ""
	}

	// Simple implementation: hash concatenation
	// In production, you'd use a proper Merkle tree with intermediate nodes
	h := sha256.New()
	for _, root := range segmentRoots {
		h.Write([]byte(root))
	}
	return hex.EncodeToString(h.Sum(nil))
}

// GetMetrics returns current anchoring metrics
func (b *BatchAnchoring) GetMetrics() AnchoringMetrics {
	b.metrics.mu.RLock()
	defer b.metrics.mu.RUnlock()
	return *b.metrics
}

// ForceSingleBatch performs an immediate batch anchor (for testing/ops)
func (b *BatchAnchoring) ForceSingleBatch(ctx context.Context) error {
	return b.anchorBatch(ctx)
}

// --- Mock Implementations for Testing ---

// MockWORMStore implements WORMStore for testing
type MockWORMStore struct {
	segments map[string][]WORMEntry
	roots    map[string]string
	mu       sync.RWMutex
}

func NewMockWORMStore() *MockWORMStore {
	return &MockWORMStore{
		segments: make(map[string][]WORMEntry),
		roots:    make(map[string]string),
	}
}

func (m *MockWORMStore) AddSegment(path string, entries []WORMEntry, root string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.segments[path] = entries
	m.roots[path] = root
}

func (m *MockWORMStore) ListSegments(ctx context.Context, since time.Time) ([]string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	paths := make([]string, 0, len(m.segments))
	for path := range m.segments {
		paths = append(paths, path)
	}
	return paths, nil
}

func (m *MockWORMStore) ReadSegment(ctx context.Context, segmentPath string) ([]WORMEntry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	entries, ok := m.segments[segmentPath]
	if !ok {
		return nil, fmt.Errorf("segment not found: %s", segmentPath)
	}
	return entries, nil
}

func (m *MockWORMStore) GetSegmentRoot(ctx context.Context, segmentPath string) (string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	root, ok := m.roots[segmentPath]
	if !ok {
		return "", fmt.Errorf("segment root not found: %s", segmentPath)
	}
	return root, nil
}

// MockAttestationStore implements AttestationStore for testing
type MockAttestationStore struct {
	attestations map[string]Attestation
	mu           sync.RWMutex
}

func NewMockAttestationStore() *MockAttestationStore {
	return &MockAttestationStore{
		attestations: make(map[string]Attestation),
	}
}

func (m *MockAttestationStore) WriteAttestation(ctx context.Context, attestation Attestation) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.attestations[attestation.BatchID] = attestation
	return nil
}

func (m *MockAttestationStore) ReadAttestation(ctx context.Context, batchID string) (*Attestation, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	attestation, ok := m.attestations[batchID]
	if !ok {
		return nil, fmt.Errorf("attestation not found: %s", batchID)
	}
	return &attestation, nil
}

func (m *MockAttestationStore) ListAttestations(ctx context.Context, since time.Time) ([]Attestation, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	attestations := make([]Attestation, 0, len(m.attestations))
	for _, a := range m.attestations {
		attestations = append(attestations, a)
	}
	return attestations, nil
}
