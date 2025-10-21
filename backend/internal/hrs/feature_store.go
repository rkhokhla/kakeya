package hrs

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// FeatureStore provides online feature vectors for HRS (Phase 7 WP1)
// Strict latency budget: ≤5ms p95 for feature extraction
type FeatureStore struct {
	mu             sync.RWMutex
	recentPCS      map[string]*PCSFeatures // pcs_id → features (bounded cache)
	tenantStats    map[string]*TenantStats // tenant_id → rolling stats
	cacheSize      int
	cacheTTL       time.Duration
	driftDetector  *FeatureDriftDetector
	metrics        *FeatureStoreMetrics
}

// PCSFeatures are online features derived from PCS
type PCSFeatures struct {
	// Signal features
	DHat         float64
	CohStar      float64
	R            float64
	Budget       float64

	// Timing features
	VerifyLatencyMs  float64
	ArrivalTimestamp time.Time

	// Metadata features
	TenantID     string
	ShardID      string
	Regime       string

	// Derived features
	SignalEntropy    float64 // -Σ p log p over normalized signals
	CoherenceDelta   float64 // coh★ - recent_avg_coh★
	CompressibilityZ float64 // (r - mean_r) / std_r (Z-score)

	// Computed at extraction time
	ExtractedAt  time.Time
}

// TenantStats tracks rolling statistics per tenant
type TenantStats struct {
	mu                sync.RWMutex
	TenantID          string
	WindowDuration    time.Duration

	// Rolling averages (exponential moving average, α=0.3)
	AvgDHat           float64
	AvgCohStar        float64
	AvgR              float64
	AvgVerifyLatencyMs float64

	// Rolling std dev (for Z-scores)
	StdR              float64

	// Counts
	TotalPCS          int64
	EscalatedPCS      int64
	LastUpdate        time.Time
}

// FeatureDriftDetector monitors feature distribution drift
type FeatureDriftDetector struct {
	mu                sync.RWMutex
	baselineStats     map[string]float64 // feature_name → baseline_mean
	currentStats      map[string]float64 // feature_name → current_mean
	driftThreshold    float64 // Max acceptable drift (e.g., 0.2 = 20%)
	alertsTriggered   int64
}

// FeatureStoreMetrics tracks feature store performance
type FeatureStoreMetrics struct {
	mu                    sync.RWMutex
	FeaturesExtracted     int64
	CacheHits             int64
	CacheMisses           int64
	ExtractionLatencyMs   []float64 // Rolling window for p95
	DriftAlertsTriggered  int64
}

// NewFeatureStore creates a new feature store
func NewFeatureStore(cacheSize int, cacheTTL time.Duration) *FeatureStore {
	return &FeatureStore{
		recentPCS:     make(map[string]*PCSFeatures),
		tenantStats:   make(map[string]*TenantStats),
		cacheSize:     cacheSize,
		cacheTTL:      cacheTTL,
		driftDetector: NewFeatureDriftDetector(0.2), // 20% drift threshold
		metrics:       &FeatureStoreMetrics{},
	}
}

// NewFeatureDriftDetector creates a new drift detector
func NewFeatureDriftDetector(threshold float64) *FeatureDriftDetector {
	return &FeatureDriftDetector{
		baselineStats:  make(map[string]float64),
		currentStats:   make(map[string]float64),
		driftThreshold: threshold,
	}
}

// ExtractFeatures extracts online features from PCS
// Latency budget: ≤5ms p95
func (fs *FeatureStore) ExtractFeatures(ctx context.Context, pcsID, tenantID string, dHat, cohStar, r, budget float64, verifyLatencyMs float64, regime string) (*PCSFeatures, error) {
	startTime := time.Now()
	defer func() {
		latency := time.Since(startTime).Milliseconds()
		fs.recordLatency(float64(latency))
	}()

	// Get tenant stats
	stats := fs.getOrCreateTenantStats(tenantID)

	// Compute derived features
	features := &PCSFeatures{
		DHat:              dHat,
		CohStar:           cohStar,
		R:                 r,
		Budget:            budget,
		VerifyLatencyMs:   verifyLatencyMs,
		ArrivalTimestamp:  time.Now(),
		TenantID:          tenantID,
		Regime:            regime,

		// Derived features
		SignalEntropy:     fs.computeSignalEntropy(dHat, cohStar, r),
		CoherenceDelta:    cohStar - stats.AvgCohStar,
		CompressibilityZ:  fs.computeZScore(r, stats.AvgR, stats.StdR),

		ExtractedAt:       time.Now(),
	}

	// Update tenant stats
	fs.updateTenantStats(tenantID, dHat, cohStar, r, verifyLatencyMs)

	// Cache features
	fs.cacheFeatures(pcsID, features)

	// Check for drift
	fs.checkFeatureDrift(features)

	fs.metrics.mu.Lock()
	fs.metrics.FeaturesExtracted++
	fs.metrics.mu.Unlock()

	return features, nil
}

// computeSignalEntropy computes entropy of normalized signal vector
func (fs *FeatureStore) computeSignalEntropy(dHat, cohStar, r float64) float64 {
	// Normalize signals to [0, 1] (assuming typical ranges)
	normDHat := dHat / 3.5           // D̂ ∈ [0, 3.5]
	normCohStar := cohStar           // coh★ ∈ [0, 1]
	normR := r                       // r ∈ [0, 1]

	// Compute probabilities (softmax-like normalization)
	sum := normDHat + normCohStar + normR
	if sum == 0 {
		return 0
	}

	pDHat := normDHat / sum
	pCohStar := normCohStar / sum
	pR := normR / sum

	// Entropy: -Σ p log p
	entropy := 0.0
	if pDHat > 0 {
		entropy -= pDHat * math.Log(pDHat)
	}
	if pCohStar > 0 {
		entropy -= pCohStar * math.Log(pCohStar)
	}
	if pR > 0 {
		entropy -= pR * math.Log(pR)
	}

	return entropy
}

// computeZScore computes Z-score for a value
func (fs *FeatureStore) computeZScore(value, mean, stdDev float64) float64 {
	if stdDev == 0 {
		return 0
	}
	return (value - mean) / stdDev
}

// getOrCreateTenantStats gets or creates tenant stats
func (fs *FeatureStore) getOrCreateTenantStats(tenantID string) *TenantStats {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	stats, ok := fs.tenantStats[tenantID]
	if !ok {
		stats = &TenantStats{
			TenantID:       tenantID,
			WindowDuration: 24 * time.Hour,
			AvgDHat:        1.5, // Reasonable defaults
			AvgCohStar:     0.7,
			AvgR:           0.8,
			StdR:           0.1,
			LastUpdate:     time.Now(),
		}
		fs.tenantStats[tenantID] = stats
	}

	return stats
}

// updateTenantStats updates rolling statistics for a tenant
func (fs *FeatureStore) updateTenantStats(tenantID string, dHat, cohStar, r, verifyLatencyMs float64) {
	stats := fs.getOrCreateTenantStats(tenantID)

	stats.mu.Lock()
	defer stats.mu.Unlock()

	// Exponential moving average (α=0.3)
	alpha := 0.3
	stats.AvgDHat = alpha*dHat + (1-alpha)*stats.AvgDHat
	stats.AvgCohStar = alpha*cohStar + (1-alpha)*stats.AvgCohStar
	stats.AvgR = alpha*r + (1-alpha)*stats.AvgR
	stats.AvgVerifyLatencyMs = alpha*verifyLatencyMs + (1-alpha)*stats.AvgVerifyLatencyMs

	// Update std dev for r (simple running estimate)
	delta := r - stats.AvgR
	stats.StdR = math.Sqrt(alpha*delta*delta + (1-alpha)*stats.StdR*stats.StdR)

	stats.TotalPCS++
	stats.LastUpdate = time.Now()
}

// cacheFeatures caches features (bounded LRU-like)
func (fs *FeatureStore) cacheFeatures(pcsID string, features *PCSFeatures) {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	// Simple bounded cache (evict oldest if full)
	if len(fs.recentPCS) >= fs.cacheSize {
		// Find and remove oldest
		var oldestID string
		var oldestTime time.Time
		for id, feat := range fs.recentPCS {
			if oldestTime.IsZero() || feat.ExtractedAt.Before(oldestTime) {
				oldestID = id
				oldestTime = feat.ExtractedAt
			}
		}
		delete(fs.recentPCS, oldestID)
	}

	fs.recentPCS[pcsID] = features
}

// GetCachedFeatures retrieves cached features
func (fs *FeatureStore) GetCachedFeatures(pcsID string) (*PCSFeatures, bool) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	features, ok := fs.recentPCS[pcsID]
	if !ok {
		fs.metrics.mu.Lock()
		fs.metrics.CacheMisses++
		fs.metrics.mu.Unlock()
		return nil, false
	}

	// Check TTL
	if time.Since(features.ExtractedAt) > fs.cacheTTL {
		fs.mu.RUnlock()
		fs.mu.Lock()
		delete(fs.recentPCS, pcsID)
		fs.mu.Unlock()
		fs.mu.RLock()

		fs.metrics.mu.Lock()
		fs.metrics.CacheMisses++
		fs.metrics.mu.Unlock()
		return nil, false
	}

	fs.metrics.mu.Lock()
	fs.metrics.CacheHits++
	fs.metrics.mu.Unlock()

	return features, true
}

// checkFeatureDrift checks for feature distribution drift
func (fs *FeatureStore) checkFeatureDrift(features *PCSFeatures) {
	fs.driftDetector.mu.Lock()
	defer fs.driftDetector.mu.Unlock()

	// Update current stats
	fs.driftDetector.currentStats["D_hat"] = features.DHat
	fs.driftDetector.currentStats["coh_star"] = features.CohStar
	fs.driftDetector.currentStats["r"] = features.R

	// Check drift (if baseline exists)
	for featureName, currentValue := range fs.driftDetector.currentStats {
		if baselineValue, ok := fs.driftDetector.baselineStats[featureName]; ok {
			drift := math.Abs(currentValue-baselineValue) / baselineValue
			if drift > fs.driftDetector.driftThreshold {
				fs.driftDetector.alertsTriggered++
				fmt.Printf("Feature drift detected: %s (baseline: %.3f, current: %.3f, drift: %.1f%%)\n",
					featureName, baselineValue, currentValue, drift*100)
			}
		}
	}
}

// SetBaseline sets baseline statistics for drift detection
func (fs *FeatureStore) SetBaseline(dHat, cohStar, r float64) {
	fs.driftDetector.mu.Lock()
	defer fs.driftDetector.mu.Unlock()

	fs.driftDetector.baselineStats["D_hat"] = dHat
	fs.driftDetector.baselineStats["coh_star"] = cohStar
	fs.driftDetector.baselineStats["r"] = r

	fmt.Printf("Baseline set: D̂=%.3f, coh★=%.3f, r=%.3f\n", dHat, cohStar, r)
}

// recordLatency records feature extraction latency
func (fs *FeatureStore) recordLatency(latencyMs float64) {
	fs.metrics.mu.Lock()
	defer fs.metrics.mu.Unlock()

	// Keep rolling window of last 1000 latencies
	fs.metrics.ExtractionLatencyMs = append(fs.metrics.ExtractionLatencyMs, latencyMs)
	if len(fs.metrics.ExtractionLatencyMs) > 1000 {
		fs.metrics.ExtractionLatencyMs = fs.metrics.ExtractionLatencyMs[1:]
	}
}

// GetMetrics returns feature store metrics
func (fs *FeatureStore) GetMetrics() FeatureStoreMetrics {
	fs.metrics.mu.RLock()
	defer fs.metrics.mu.RUnlock()
	return *fs.metrics
}

// GetP95Latency returns p95 feature extraction latency
func (fs *FeatureStore) GetP95Latency() float64 {
	fs.metrics.mu.RLock()
	defer fs.metrics.mu.RUnlock()

	if len(fs.metrics.ExtractionLatencyMs) == 0 {
		return 0
	}

	// Sort and find p95
	sorted := make([]float64, len(fs.metrics.ExtractionLatencyMs))
	copy(sorted, fs.metrics.ExtractionLatencyMs)

	// Simple bubble sort (good enough for 1000 elements)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	p95Index := int(float64(len(sorted)) * 0.95)
	return sorted[p95Index]
}
