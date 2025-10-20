package crr

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// DivergenceDetector monitors dedup state across regions and detects split-brain (Phase 5 WP1)
// Compares key counts and samples values to detect geo-replication divergence
type DivergenceDetector struct {
	mu              sync.RWMutex
	regions         map[string]DedupStore // region_id → dedup store
	sampleSize      int                   // Number of keys to sample for comparison
	countThreshold  float64               // Max allowed key count divergence (%)
	sampleThreshold float64               // Max allowed sample mismatch (%)
	checkInterval   time.Duration
	alertHandler    AlertHandler // Callback to raise alerts
	metrics         *DivergenceMetrics
	stopCh          chan struct{}
	wg              sync.WaitGroup
}

// DivergenceMetrics tracks divergence detection operations
type DivergenceMetrics struct {
	mu                   sync.RWMutex
	ChecksPerformed      int64
	DivergencesDetected  int64
	LastCheckTimestamp   time.Time
	MaxCountDivergence   float64 // Max observed count divergence (%)
	MaxSampleDivergence  float64 // Max observed sample mismatch (%)
	AffectedRegionPairs  []string
}

// DedupStore abstracts dedup store operations for divergence checking
type DedupStore interface {
	Count(ctx context.Context) (int64, error)                    // Total key count
	SampleKeys(ctx context.Context, n int) ([]string, error)     // Random sample of n keys
	Get(ctx context.Context, key string) (interface{}, error)    // Get value for key
	Exists(ctx context.Context, key string) (bool, error)        // Check if key exists
}

// AlertHandler handles divergence alerts
type AlertHandler interface {
	RaiseAlert(ctx context.Context, alert DivergenceAlert) error
}

// DivergenceAlert represents a detected divergence event
type DivergenceAlert struct {
	Severity        string    // "warning" or "critical"
	Region1         string
	Region2         string
	CountDivergence float64   // Key count divergence (%)
	SampleMismatch  float64   // Sample mismatch (%)
	Description     string
	RunbookLink     string
	DetectedAt      time.Time
}

// DivergenceDetectorConfig holds configuration
type DivergenceDetectorConfig struct {
	Regions         map[string]DedupStore
	SampleSize      int
	CountThreshold  float64       // Default: 5.0 (5% divergence triggers warning)
	SampleThreshold float64       // Default: 10.0 (10% mismatch triggers warning)
	CheckInterval   time.Duration // Default: 5 minutes
	AlertHandler    AlertHandler
}

// NewDivergenceDetector creates a new divergence detector
func NewDivergenceDetector(config DivergenceDetectorConfig) (*DivergenceDetector, error) {
	if len(config.Regions) < 2 {
		return nil, fmt.Errorf("at least 2 regions required for divergence detection")
	}
	if config.AlertHandler == nil {
		return nil, fmt.Errorf("AlertHandler is required")
	}

	if config.SampleSize == 0 {
		config.SampleSize = 100 // Default: sample 100 keys
	}
	if config.CountThreshold == 0 {
		config.CountThreshold = 5.0 // Default: 5% divergence
	}
	if config.SampleThreshold == 0 {
		config.SampleThreshold = 10.0 // Default: 10% mismatch
	}
	if config.CheckInterval == 0 {
		config.CheckInterval = 5 * time.Minute // Default: check every 5 minutes
	}

	detector := &DivergenceDetector{
		regions:         config.Regions,
		sampleSize:      config.SampleSize,
		countThreshold:  config.CountThreshold,
		sampleThreshold: config.SampleThreshold,
		checkInterval:   config.CheckInterval,
		alertHandler:    config.AlertHandler,
		metrics:         &DivergenceMetrics{},
		stopCh:          make(chan struct{}),
	}

	return detector, nil
}

// Start begins the divergence detection loop (runs in background)
func (d *DivergenceDetector) Start(ctx context.Context) {
	d.wg.Add(1)
	go d.checkLoop(ctx)
	fmt.Printf("Geo Divergence Detector: started (monitoring %d regions)\n", len(d.regions))
}

// Stop gracefully stops the detector
func (d *DivergenceDetector) Stop() {
	close(d.stopCh)
	d.wg.Wait()
	fmt.Printf("Geo Divergence Detector: stopped\n")
}

// checkLoop continuously checks for divergence across region pairs
func (d *DivergenceDetector) checkLoop(ctx context.Context) {
	defer d.wg.Done()

	ticker := time.NewTicker(d.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-d.stopCh:
			return
		case <-ticker.C:
			if err := d.checkAllRegionPairs(ctx); err != nil {
				fmt.Printf("Geo Divergence Detector: error checking divergence: %v\n", err)
			}
		}
	}
}

// checkAllRegionPairs checks divergence between all region pairs
func (d *DivergenceDetector) checkAllRegionPairs(ctx context.Context) error {
	d.mu.RLock()
	defer d.mu.RUnlock()

	regionIDs := make([]string, 0, len(d.regions))
	for id := range d.regions {
		regionIDs = append(regionIDs, id)
	}

	// Check all pairs
	for i := 0; i < len(regionIDs); i++ {
		for j := i + 1; j < len(regionIDs); j++ {
			region1 := regionIDs[i]
			region2 := regionIDs[j]

			if err := d.checkRegionPair(ctx, region1, region2); err != nil {
				fmt.Printf("Geo Divergence Detector: error checking %s vs %s: %v\n",
					region1, region2, err)
			}
		}
	}

	d.metrics.mu.Lock()
	d.metrics.ChecksPerformed++
	d.metrics.LastCheckTimestamp = time.Now()
	d.metrics.mu.Unlock()

	return nil
}

// checkRegionPair checks divergence between two specific regions
func (d *DivergenceDetector) checkRegionPair(ctx context.Context, region1, region2 string) error {
	store1 := d.regions[region1]
	store2 := d.regions[region2]

	// 1. Compare key counts
	count1, err := store1.Count(ctx)
	if err != nil {
		return fmt.Errorf("failed to get count for %s: %w", region1, err)
	}

	count2, err := store2.Count(ctx)
	if err != nil {
		return fmt.Errorf("failed to get count for %s: %w", region2, err)
	}

	countDivergence := computeDivergence(count1, count2)

	// 2. Sample keys and compare values
	sampleMismatch := 0.0
	if count1 > 0 && count2 > 0 {
		mismatch, err := d.compareSamples(ctx, store1, store2, region1, region2)
		if err != nil {
			fmt.Printf("Geo Divergence Detector: warning: sample comparison failed: %v\n", err)
		} else {
			sampleMismatch = mismatch
		}
	}

	// 3. Update metrics
	d.metrics.mu.Lock()
	if countDivergence > d.metrics.MaxCountDivergence {
		d.metrics.MaxCountDivergence = countDivergence
	}
	if sampleMismatch > d.metrics.MaxSampleDivergence {
		d.metrics.MaxSampleDivergence = sampleMismatch
	}
	d.metrics.mu.Unlock()

	// 4. Raise alert if thresholds exceeded
	if countDivergence > d.countThreshold || sampleMismatch > d.sampleThreshold {
		severity := "warning"
		if countDivergence > d.countThreshold*2 || sampleMismatch > d.sampleThreshold*2 {
			severity = "critical"
		}

		alert := DivergenceAlert{
			Severity:        severity,
			Region1:         region1,
			Region2:         region2,
			CountDivergence: countDivergence,
			SampleMismatch:  sampleMismatch,
			Description: fmt.Sprintf(
				"Geo-replication divergence detected between %s and %s: "+
					"key count divergence %.2f%%, sample mismatch %.2f%%",
				region1, region2, countDivergence, sampleMismatch),
			RunbookLink: "docs/runbooks/geo-split-brain.md",
			DetectedAt:  time.Now(),
		}

		if err := d.alertHandler.RaiseAlert(ctx, alert); err != nil {
			return fmt.Errorf("failed to raise alert: %w", err)
		}

		d.metrics.mu.Lock()
		d.metrics.DivergencesDetected++
		d.metrics.AffectedRegionPairs = append(d.metrics.AffectedRegionPairs,
			fmt.Sprintf("%s-%s", region1, region2))
		d.metrics.mu.Unlock()

		fmt.Printf("Geo Divergence Detector: ALERT (%s) - %s\n", severity, alert.Description)
	}

	return nil
}

// compareSamples samples keys from both stores and compares values
// Returns mismatch percentage (0-100)
func (d *DivergenceDetector) compareSamples(ctx context.Context,
	store1, store2 DedupStore, region1, region2 string) (float64, error) {

	// Sample keys from region1
	keys1, err := store1.SampleKeys(ctx, d.sampleSize)
	if err != nil {
		return 0, fmt.Errorf("failed to sample keys from %s: %w", region1, err)
	}

	if len(keys1) == 0 {
		return 0, nil // No keys to compare
	}

	// For each sampled key, check if it exists in region2 and if values match
	mismatches := 0
	for _, key := range keys1 {
		exists2, err := store2.Exists(ctx, key)
		if err != nil {
			continue // Skip on error
		}

		if !exists2 {
			mismatches++
			continue
		}

		// Get values from both stores
		val1, err1 := store1.Get(ctx, key)
		val2, err2 := store2.Get(ctx, key)

		if err1 != nil || err2 != nil {
			continue // Skip on error
		}

		// Compare values (simple equality check for now)
		// In production, you'd compare specific fields (e.g., accepted, D_hat, budget)
		if !valuesEqual(val1, val2) {
			mismatches++
		}
	}

	mismatchPct := float64(mismatches) / float64(len(keys1)) * 100.0
	return mismatchPct, nil
}

// computeDivergence calculates percentage divergence between two counts
func computeDivergence(count1, count2 int64) float64 {
	if count1 == 0 && count2 == 0 {
		return 0.0
	}

	avg := float64(count1+count2) / 2.0
	diff := math.Abs(float64(count1 - count2))
	return (diff / avg) * 100.0
}

// valuesEqual compares two dedup store values for equality
// This is a simplified comparison - in production, you'd want to compare
// specific fields (accepted, D_hat, budget, etc.) with tolerance
func valuesEqual(v1, v2 interface{}) bool {
	// Type assertion to map (assuming VerifyResult serializes to map)
	m1, ok1 := v1.(map[string]interface{})
	m2, ok2 := v2.(map[string]interface{})

	if !ok1 || !ok2 {
		return false
	}

	// Compare key fields
	if m1["accepted"] != m2["accepted"] {
		return false
	}

	// Compare numeric fields with tolerance (9-decimal rounding)
	if !floatsEqual(m1["D_hat"], m2["D_hat"]) {
		return false
	}
	if !floatsEqual(m1["coh_star"], m2["coh_star"]) {
		return false
	}
	if !floatsEqual(m1["r"], m2["r"]) {
		return false
	}
	if !floatsEqual(m1["budget"], m2["budget"]) {
		return false
	}

	return true
}

// floatsEqual compares two floats with 9-decimal precision (Phase 1 rounding)
func floatsEqual(v1, v2 interface{}) bool {
	f1, ok1 := v1.(float64)
	f2, ok2 := v2.(float64)

	if !ok1 || !ok2 {
		return false
	}

	// Round to 9 decimals (Phase 1 canonical rounding)
	r1 := math.Round(f1*1e9) / 1e9
	r2 := math.Round(f2*1e9) / 1e9

	return r1 == r2
}

// GetMetrics returns current divergence metrics
func (d *DivergenceDetector) GetMetrics() DivergenceMetrics {
	d.metrics.mu.RLock()
	defer d.metrics.mu.RUnlock()

	return *d.metrics
}

// ForceSingleCheck performs an immediate divergence check (for testing/ops)
func (d *DivergenceDetector) ForceSingleCheck(ctx context.Context) error {
	return d.checkAllRegionPairs(ctx)
}

// --- Mock implementations for testing ---

// MockDedupStore implements DedupStore for testing
type MockDedupStore struct {
	data map[string]interface{}
	mu   sync.RWMutex
}

func NewMockDedupStore() *MockDedupStore {
	return &MockDedupStore{
		data: make(map[string]interface{}),
	}
}

func (m *MockDedupStore) Count(ctx context.Context) (int64, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return int64(len(m.data)), nil
}

func (m *MockDedupStore) SampleKeys(ctx context.Context, n int) ([]string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	keys := make([]string, 0, len(m.data))
	for k := range m.data {
		keys = append(keys, k)
	}

	// Shuffle and take first n
	rand.Shuffle(len(keys), func(i, j int) {
		keys[i], keys[j] = keys[j], keys[i]
	})

	if n > len(keys) {
		n = len(keys)
	}

	return keys[:n], nil
}

func (m *MockDedupStore) Get(ctx context.Context, key string) (interface{}, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	val, ok := m.data[key]
	if !ok {
		return nil, fmt.Errorf("key not found: %s", key)
	}
	return val, nil
}

func (m *MockDedupStore) Exists(ctx context.Context, key string) (bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	_, ok := m.data[key]
	return ok, nil
}

func (m *MockDedupStore) Set(key string, value interface{}) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.data[key] = value
}
