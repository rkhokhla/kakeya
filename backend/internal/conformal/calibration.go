package conformal

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/fractal-lba/kakeya/internal/api"
)

// NonconformityScore represents a single calibration data point.
// It stores the input PCS, the computed nonconformity score, and metadata.
type NonconformityScore struct {
	PCSID     string    `json:"pcs_id"`
	Score     float64   `json:"score"`
	TrueLabel bool      `json:"true_label"` // True if known to be benign
	Timestamp time.Time `json:"timestamp"`
	TenantID  string    `json:"tenant_id"`
}

// CalibrationSet manages a collection of nonconformity scores for quantile computation.
type CalibrationSet struct {
	mu       sync.RWMutex
	scores   []NonconformityScore
	maxSize  int
	window   time.Duration // Time window for recency-based pruning
	tenantID string        // Empty string = global calibration
}

// NewCalibrationSet creates a new calibration set with specified capacity.
// maxSize: maximum number of scores to retain (FIFO eviction)
// window: time window for score retention (0 = no time-based pruning)
func NewCalibrationSet(maxSize int, window time.Duration, tenantID string) *CalibrationSet {
	if maxSize <= 0 {
		maxSize = 1000 // Default to 1000 calibration points
	}
	return &CalibrationSet{
		scores:   make([]NonconformityScore, 0, maxSize),
		maxSize:  maxSize,
		window:   window,
		tenantID: tenantID,
	}
}

// Add appends a nonconformity score to the calibration set.
// Evicts oldest entry if maxSize is reached.
func (cs *CalibrationSet) Add(score NonconformityScore) {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	// Filter by tenant if specified
	if cs.tenantID != "" && score.TenantID != cs.tenantID {
		return
	}

	// Add score
	cs.scores = append(cs.scores, score)

	// Evict oldest if over capacity (FIFO)
	if len(cs.scores) > cs.maxSize {
		cs.scores = cs.scores[1:]
	}

	// Prune old scores outside time window
	if cs.window > 0 {
		cs.pruneOld()
	}
}

// pruneOld removes scores outside the time window.
// Caller must hold cs.mu lock.
func (cs *CalibrationSet) pruneOld() {
	cutoff := time.Now().Add(-cs.window)
	kept := cs.scores[:0]
	for _, s := range cs.scores {
		if s.Timestamp.After(cutoff) {
			kept = append(kept, s)
		}
	}
	cs.scores = kept
}

// Quantile computes the (1-delta) quantile of nonconformity scores.
// Returns the quantile value and number of scores used.
// Uses linear interpolation between order statistics for fractional quantiles.
func (cs *CalibrationSet) Quantile(delta float64) (float64, int, error) {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	n := len(cs.scores)
	if n == 0 {
		return 0, 0, fmt.Errorf("calibration set is empty")
	}

	if delta <= 0 || delta >= 1 {
		return 0, n, fmt.Errorf("delta must be in (0, 1), got: %.3f", delta)
	}

	// Copy scores for sorting
	sorted := make([]float64, n)
	for i, s := range cs.scores {
		sorted[i] = s.Score
	}
	sort.Float64s(sorted)

	// Compute (1-delta) quantile with linear interpolation
	// For split conformal: want ceiling((n+1)*(1-delta))/n quantile
	// Simpler: use (1-delta)*(n+1) position with interpolation
	pos := (1 - delta) * float64(n+1)
	idx := int(math.Floor(pos)) - 1 // 0-indexed
	frac := pos - math.Floor(pos)

	// Boundary cases
	if idx < 0 {
		return sorted[0], n, nil
	}
	if idx >= n-1 {
		return sorted[n-1], n, nil
	}

	// Linear interpolation
	q := sorted[idx] + frac*(sorted[idx+1]-sorted[idx])
	return q, n, nil
}

// Size returns the current number of scores in the calibration set.
func (cs *CalibrationSet) Size() int {
	cs.mu.RLock()
	defer cs.mu.RUnlock()
	return len(cs.scores)
}

// ComputeScore computes the nonconformity score for a PCS.
// Higher scores indicate more anomalous/non-conforming inputs.
// This is a simple weighted combination of signals; production should train a model.
func ComputeScore(pcs *api.PCS, params *api.VerifyParams) float64 {
	// Normalize signals to [0, 1] ranges
	dhatNorm := normalizeD(pcs.DHat)
	cohNorm := normalizeCoh(pcs.CohStar, params)
	rNorm := normalizeR(pcs.R)

	// Combine with weights (tunable hyperparameters)
	// Higher D_hat = more complex = lower anomaly score (inverted)
	// Lower coh_star = less coherent = higher anomaly score
	// Extreme r (very low or very high) = higher anomaly score
	w1, w2, w3 := 0.35, 0.40, 0.25 // Weights sum to 1.0

	score := w1*(1-dhatNorm) + w2*(1-cohNorm) + w3*rAnomalyScore(rNorm)

	return score
}

// normalizeD maps D_hat to [0, 1] range.
// D in [0, 3.5] → norm in [0, 1]
func normalizeD(dhat float64) float64 {
	const maxD = 3.5
	if dhat < 0 {
		return 0
	}
	if dhat > maxD {
		return 1
	}
	return dhat / maxD
}

// normalizeCoh maps coh_star to [0, 1] range with tolerance.
// coh in [0, 1+tolCoh] → norm in [0, 1]
func normalizeCoh(coh float64, params *api.VerifyParams) float64 {
	max := 1.0 + params.TolCoh
	if coh < 0 {
		return 0
	}
	if coh > max {
		return 1
	}
	return coh / max
}

// normalizeR maps compressibility to [0, 1] range.
func normalizeR(r float64) float64 {
	if r < 0 {
		return 0
	}
	if r > 1 {
		return 1
	}
	return r
}

// rAnomalyScore computes anomaly contribution from compressibility.
// U-shaped: low r (repetitive) and high r (noisy) both anomalous.
func rAnomalyScore(r float64) float64 {
	// Parabola with minimum at r=0.65 (normal range)
	optimal := 0.65
	deviation := math.Abs(r - optimal)
	// Map [0, 0.65] to [0, 1] for deviation
	return math.Min(1.0, deviation/optimal)
}

// Decision represents the conformal prediction decision.
type Decision string

const (
	DecisionAccept   Decision = "ACCEPT"   // Score ≤ quantile
	DecisionEscalate Decision = "ESCALATE" // Score near quantile (ambiguous)
	DecisionReject   Decision = "REJECT"   // Score >> quantile
)

// PredictionResult contains the conformal prediction outcome.
type PredictionResult struct {
	Decision      Decision  `json:"decision"`
	Score         float64   `json:"score"`
	Quantile      float64   `json:"quantile"`
	Delta         float64   `json:"delta"`           // Target miscoverage
	CalibrationN  int       `json:"calibration_n"`   // Calibration set size
	Margin        float64   `json:"margin"`          // score - quantile
	Confidence    float64   `json:"confidence"`      // 1 - score (for ACCEPT)
	Timestamp     time.Time `json:"timestamp"`
	CalibrationID string    `json:"calibration_id"`  // Hash of calibration set
}

// Predict makes a conformal prediction decision for a PCS.
// Returns ACCEPT if score ≤ quantile, ESCALATE if near threshold, REJECT if far above.
func (cs *CalibrationSet) Predict(pcs *api.PCS, params *api.VerifyParams, delta float64) (*PredictionResult, error) {
	// Compute nonconformity score
	score := ComputeScore(pcs, params)

	// Get quantile from calibration set
	quantile, n, err := cs.Quantile(delta)
	if err != nil {
		return nil, fmt.Errorf("failed to compute quantile: %w", err)
	}

	// Compute margin and decision
	margin := score - quantile

	var decision Decision
	const escalateThreshold = 0.05 // Within 5% of quantile → escalate

	if score <= quantile {
		decision = DecisionAccept
	} else if margin <= escalateThreshold {
		decision = DecisionEscalate // Ambiguous region
	} else {
		decision = DecisionReject
	}

	// Confidence for accepted predictions (1 - score, higher is better)
	confidence := 1.0 - score

	result := &PredictionResult{
		Decision:      decision,
		Score:         score,
		Quantile:      quantile,
		Delta:         delta,
		CalibrationN:  n,
		Margin:        margin,
		Confidence:    confidence,
		Timestamp:     time.Now(),
		CalibrationID: cs.Hash(),
	}

	return result, nil
}

// Hash computes a hash of the calibration set for tracking.
// Returns first 8 chars of SHA-256 hash of (tenantID, maxSize, n, timestamp_range).
func (cs *CalibrationSet) Hash() string {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	if len(cs.scores) == 0 {
		return "empty"
	}

	// Simple hash: tenant + size + count + time range
	minTime := cs.scores[0].Timestamp
	maxTime := cs.scores[len(cs.scores)-1].Timestamp

	return fmt.Sprintf("%s_%d_%d_%d", cs.tenantID, cs.maxSize, len(cs.scores),
		maxTime.Unix()-minTime.Unix())
}

// Stats returns calibration set statistics for monitoring.
type Stats struct {
	Size         int       `json:"size"`
	MaxSize      int       `json:"max_size"`
	TenantID     string    `json:"tenant_id"`
	WindowHours  float64   `json:"window_hours"`
	OldestScore  time.Time `json:"oldest_score"`
	NewestScore  time.Time `json:"newest_score"`
	MeanScore    float64   `json:"mean_score"`
	MedianScore  float64   `json:"median_score"`
	StdDevScore  float64   `json:"stddev_score"`
}

// GetStats returns statistics about the calibration set.
func (cs *CalibrationSet) GetStats() Stats {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	stats := Stats{
		Size:        len(cs.scores),
		MaxSize:     cs.maxSize,
		TenantID:    cs.tenantID,
		WindowHours: cs.window.Hours(),
	}

	if len(cs.scores) == 0 {
		return stats
	}

	// Time range
	stats.OldestScore = cs.scores[0].Timestamp
	stats.NewestScore = cs.scores[len(cs.scores)-1].Timestamp

	// Compute mean, median, stddev
	var sum float64
	sorted := make([]float64, len(cs.scores))
	for i, s := range cs.scores {
		sorted[i] = s.Score
		sum += s.Score
	}
	stats.MeanScore = sum / float64(len(cs.scores))

	sort.Float64s(sorted)
	stats.MedianScore = sorted[len(sorted)/2]

	// Stddev
	var variance float64
	for _, s := range cs.scores {
		diff := s.Score - stats.MeanScore
		variance += diff * diff
	}
	stats.StdDevScore = math.Sqrt(variance / float64(len(cs.scores)))

	return stats
}
