package conformal

import (
	"math"
	"testing"
	"time"

	"github.com/fractal-lba/kakeya/internal/api"
)

func TestCalibrationSet_AddAndQuantile(t *testing.T) {
	cs := NewCalibrationSet(100, 0, "")

	// Add 100 scores uniformly from 0 to 1
	for i := 0; i < 100; i++ {
		score := float64(i) / 99.0
		cs.Add(NonconformityScore{
			PCSID:     "test",
			Score:     score,
			TrueLabel: true,
			Timestamp: time.Now(),
		})
	}

	// Test quantile computation
	tests := []struct {
		delta    float64
		expected float64
		tolerance float64
	}{
		{delta: 0.05, expected: 0.95, tolerance: 0.05}, // 95th percentile
		{delta: 0.10, expected: 0.90, tolerance: 0.05}, // 90th percentile
		{delta: 0.50, expected: 0.50, tolerance: 0.05}, // median
	}

	for _, tt := range tests {
		q, n, err := cs.Quantile(tt.delta)
		if err != nil {
			t.Fatalf("Quantile(%.2f) failed: %v", tt.delta, err)
		}
		if n != 100 {
			t.Errorf("Expected n=100, got %d", n)
		}
		if math.Abs(q-tt.expected) > tt.tolerance {
			t.Errorf("Quantile(%.2f): got %.3f, want %.3f ± %.2f", tt.delta, q, tt.expected, tt.tolerance)
		}
	}
}

func TestCalibrationSet_FIFO_Eviction(t *testing.T) {
	cs := NewCalibrationSet(10, 0, "") // Max 10 scores

	// Add 20 scores
	for i := 0; i < 20; i++ {
		cs.Add(NonconformityScore{
			PCSID:     "test",
			Score:     float64(i),
			Timestamp: time.Now(),
		})
	}

	// Should have only last 10
	if cs.Size() != 10 {
		t.Errorf("Expected size 10, got %d", cs.Size())
	}

	// First score should be 10 (0-9 evicted)
	cs.mu.RLock()
	firstScore := cs.scores[0].Score
	cs.mu.RUnlock()

	if firstScore != 10.0 {
		t.Errorf("Expected first score 10.0, got %.1f", firstScore)
	}
}

func TestCalibrationSet_TimeWindowPruning(t *testing.T) {
	window := 1 * time.Hour
	cs := NewCalibrationSet(100, window, "")

	// Add old scores (2 hours ago)
	oldTime := time.Now().Add(-2 * time.Hour)
	for i := 0; i < 10; i++ {
		cs.Add(NonconformityScore{
			PCSID:     "old",
			Score:     float64(i),
			Timestamp: oldTime,
		})
	}

	// Add recent scores
	recentTime := time.Now()
	for i := 0; i < 5; i++ {
		cs.Add(NonconformityScore{
			PCSID:     "recent",
			Score:     float64(i),
			Timestamp: recentTime,
		})
	}

	// After adding recent scores, pruneOld() should remove old ones
	// Size should be 5 (only recent retained)
	if cs.Size() != 5 {
		t.Errorf("Expected size 5 after pruning, got %d", cs.Size())
	}
}

func TestComputeScore(t *testing.T) {
	params := api.DefaultVerifyParams()

	tests := []struct {
		name     string
		pcs      *api.PCS
		wantLow  bool // True if score should be low (benign)
	}{
		{
			name: "normal_text",
			pcs: &api.PCS{
				DHat:    1.8,  // Normal range
				CohStar: 0.75, // Coherent
				R:       0.65, // Optimal compressibility
			},
			wantLow: true,
		},
		{
			name: "repetitive_hallucination",
			pcs: &api.PCS{
				DHat:    0.8,  // Low dimension
				CohStar: 0.85, // High coherence
				R:       0.25, // Highly compressible
			},
			wantLow: false, // Should have high anomaly score
		},
		{
			name: "semantic_drift",
			pcs: &api.PCS{
				DHat:    2.8,  // High dimension
				CohStar: 0.45, // Low coherence
				R:       0.70,
			},
			wantLow: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := ComputeScore(tt.pcs, params)

			if tt.wantLow && score > 0.5 {
				t.Errorf("Expected low score for %s, got %.3f", tt.name, score)
			}
			if !tt.wantLow && score < 0.3 {
				t.Errorf("Expected high score for %s, got %.3f", tt.name, score)
			}

			t.Logf("%s: score=%.3f", tt.name, score)
		})
	}
}

func TestCalibrationSet_Predict(t *testing.T) {
	cs := NewCalibrationSet(100, 0, "")
	params := api.DefaultVerifyParams()

	// Add calibration scores: 50 benign (low scores) + 50 anomalous (high scores)
	for i := 0; i < 50; i++ {
		cs.Add(NonconformityScore{
			Score:     float64(i) / 100.0, // 0.0 to 0.5
			TrueLabel: true,
			Timestamp: time.Now(),
		})
	}
	for i := 50; i < 100; i++ {
		cs.Add(NonconformityScore{
			Score:     float64(i) / 100.0, // 0.5 to 1.0
			TrueLabel: false,
			Timestamp: time.Now(),
		})
	}

	// Test benign PCS (should accept)
	benignPCS := &api.PCS{
		DHat:    1.8,
		CohStar: 0.75,
		R:       0.65,
	}
	result, err := cs.Predict(benignPCS, params, 0.05)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	if result.Decision != DecisionAccept {
		t.Errorf("Expected ACCEPT for benign PCS, got %s (score=%.3f, quantile=%.3f)",
			result.Decision, result.Score, result.Quantile)
	}

	// Test anomalous PCS (should reject or escalate)
	anomalousPCS := &api.PCS{
		DHat:    0.8,
		CohStar: 0.85,
		R:       0.25,
	}
	result, err = cs.Predict(anomalousPCS, params, 0.05)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	if result.Decision == DecisionAccept {
		t.Errorf("Expected REJECT/ESCALATE for anomalous PCS, got %s (score=%.3f, quantile=%.3f)",
			result.Decision, result.Score, result.Quantile)
	}

	t.Logf("Benign: score=%.3f, decision=%s", ComputeScore(benignPCS, params), result.Decision)
	t.Logf("Anomalous: score=%.3f, decision=%s", ComputeScore(anomalousPCS, params), result.Decision)
}

func TestCalibrationSet_Stats(t *testing.T) {
	cs := NewCalibrationSet(100, 1*time.Hour, "tenant1")

	// Add scores
	for i := 0; i < 50; i++ {
		cs.Add(NonconformityScore{
			Score:     float64(i) / 49.0,
			TenantID:  "tenant1",
			Timestamp: time.Now(),
		})
	}

	stats := cs.GetStats()

	if stats.Size != 50 {
		t.Errorf("Expected size 50, got %d", stats.Size)
	}
	if stats.TenantID != "tenant1" {
		t.Errorf("Expected tenant1, got %s", stats.TenantID)
	}
	if math.Abs(stats.MeanScore-0.5) > 0.05 {
		t.Errorf("Expected mean ~0.5, got %.3f", stats.MeanScore)
	}
	if math.Abs(stats.MedianScore-0.5) > 0.05 {
		t.Errorf("Expected median ~0.5, got %.3f", stats.MedianScore)
	}

	t.Logf("Stats: %+v", stats)
}

func TestCalibrationSet_TenantFiltering(t *testing.T) {
	cs := NewCalibrationSet(100, 0, "tenant1")

	// Add scores from different tenants
	cs.Add(NonconformityScore{Score: 0.1, TenantID: "tenant1", Timestamp: time.Now()})
	cs.Add(NonconformityScore{Score: 0.2, TenantID: "tenant2", Timestamp: time.Now()}) // Should be filtered
	cs.Add(NonconformityScore{Score: 0.3, TenantID: "tenant1", Timestamp: time.Now()})

	// Should only have 2 scores (tenant1)
	if cs.Size() != 2 {
		t.Errorf("Expected size 2 (tenant1 only), got %d", cs.Size())
	}
}
