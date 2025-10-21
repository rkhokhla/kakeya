package routing

import (
	"testing"
)

func TestShouldSkipEnsemble_Disabled(t *testing.T) {
	policy := DefaultRiskRoutingPolicy()
	policy.Enabled = false

	hrs := &HRSPrediction{
		Risk:            0.05,
		ConfidenceLower: 0.90,
		ConfidenceUpper: 0.92,
	}

	if policy.ShouldSkipEnsemble(hrs) {
		t.Error("Expected standard path when disabled")
	}

	metrics := policy.GetMetrics()
	if metrics.FastPathCount != 0 {
		t.Errorf("Expected 0 fast path decisions, got %d", metrics.FastPathCount)
	}
	if metrics.StandardPathCount != 1 {
		t.Errorf("Expected 1 standard path decision, got %d", metrics.StandardPathCount)
	}
}

func TestShouldSkipEnsemble_LowRiskNarrowCI(t *testing.T) {
	policy := DefaultRiskRoutingPolicy()
	policy.Enabled = true
	policy.SkipEnsembleBelow = 0.10
	policy.MaxCIWidth = 0.05

	hrs := &HRSPrediction{
		Risk:            0.05,
		ConfidenceLower: 0.90,
		ConfidenceUpper: 0.93, // CI width = 0.03
	}

	if !policy.ShouldSkipEnsemble(hrs) {
		t.Error("Expected fast path for low risk and narrow CI")
	}

	metrics := policy.GetMetrics()
	if metrics.FastPathCount != 1 {
		t.Errorf("Expected 1 fast path decision, got %d", metrics.FastPathCount)
	}
}

func TestShouldSkipEnsemble_HighRisk(t *testing.T) {
	policy := DefaultRiskRoutingPolicy()
	policy.Enabled = true
	policy.SkipEnsembleBelow = 0.10

	hrs := &HRSPrediction{
		Risk:            0.15,
		ConfidenceLower: 0.90,
		ConfidenceUpper: 0.92,
	}

	if policy.ShouldSkipEnsemble(hrs) {
		t.Error("Expected standard path for high risk")
	}

	metrics := policy.GetMetrics()
	if metrics.FastPathCount != 0 {
		t.Errorf("Expected 0 fast path decisions, got %d", metrics.FastPathCount)
	}
}

func TestShouldSkipEnsemble_WideCI(t *testing.T) {
	policy := DefaultRiskRoutingPolicy()
	policy.Enabled = true
	policy.RequireNarrowCI = true
	policy.SkipEnsembleBelow = 0.10
	policy.MaxCIWidth = 0.05

	hrs := &HRSPrediction{
		Risk:            0.05,
		ConfidenceLower: 0.80,
		ConfidenceUpper: 0.90, // CI width = 0.10 (too wide)
	}

	if policy.ShouldSkipEnsemble(hrs) {
		t.Error("Expected standard path for wide CI")
	}
}

func TestShouldSkipEnsemble_BoundaryCase(t *testing.T) {
	policy := DefaultRiskRoutingPolicy()
	policy.Enabled = true
	policy.SkipEnsembleBelow = 0.10
	policy.MaxCIWidth = 0.05

	// Risk exactly at threshold (should reject - not strictly below)
	hrs := &HRSPrediction{
		Risk:            0.10,
		ConfidenceLower: 0.90,
		ConfidenceUpper: 0.92,
	}

	if policy.ShouldSkipEnsemble(hrs) {
		t.Error("Expected standard path when risk equals threshold (not below)")
	}

	// CI width at threshold (should reject - not strictly below)
	// Using 0.051 to avoid floating point precision issues
	hrs2 := &HRSPrediction{
		Risk:            0.05,
		ConfidenceLower: 0.90,
		ConfidenceUpper: 0.951, // CI width = 0.051 (at/above max)
	}

	if policy.ShouldSkipEnsemble(hrs2) {
		t.Error("Expected standard path when CI width at/above threshold")
	}

	// Just below threshold (should accept fast path)
	hrs3 := &HRSPrediction{
		Risk:            0.09,
		ConfidenceLower: 0.90,
		ConfidenceUpper: 0.94, // CI width = 0.04 (below max)
	}

	if !policy.ShouldSkipEnsemble(hrs3) {
		t.Error("Expected fast path when risk and CI both below thresholds")
	}
}

func TestGetRouteDecision(t *testing.T) {
	tests := []struct {
		name           string
		enabled        bool
		risk           float64
		ciLower        float64
		ciUpper        float64
		expectedRoute  string
		expectedReason string
	}{
		{
			name:           "disabled",
			enabled:        false,
			risk:           0.05,
			ciLower:        0.90,
			ciUpper:        0.92,
			expectedRoute:  "standard",
			expectedReason: "risk_routing_disabled",
		},
		{
			name:           "fast path",
			enabled:        true,
			risk:           0.05,
			ciLower:        0.90,
			ciUpper:        0.93,
			expectedRoute:  "fast_path",
			expectedReason: "low_risk_narrow_ci",
		},
		{
			name:           "high risk",
			enabled:        true,
			risk:           0.15,
			ciLower:        0.90,
			ciUpper:        0.92,
			expectedRoute:  "standard",
			expectedReason: "high_risk",
		},
		{
			name:           "wide CI",
			enabled:        true,
			risk:           0.05,
			ciLower:        0.80,
			ciUpper:        0.90,
			expectedRoute:  "standard",
			expectedReason: "wide_confidence_interval",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			policy := DefaultRiskRoutingPolicy()
			policy.Enabled = tt.enabled
			policy.SkipEnsembleBelow = 0.10
			policy.MaxCIWidth = 0.05

			hrs := &HRSPrediction{
				Risk:            tt.risk,
				ConfidenceLower: tt.ciLower,
				ConfidenceUpper: tt.ciUpper,
			}

			decision := policy.GetRouteDecision(hrs)

			if decision.Route != tt.expectedRoute {
				t.Errorf("Expected route %q, got %q", tt.expectedRoute, decision.Route)
			}

			if decision.Reason != tt.expectedReason {
				t.Errorf("Expected reason %q, got %q", tt.expectedReason, decision.Reason)
			}

			if decision.HRSRisk != tt.risk {
				t.Errorf("Expected HRSRisk %.2f, got %.2f", tt.risk, decision.HRSRisk)
			}
		})
	}
}

func TestGetMetrics(t *testing.T) {
	policy := DefaultRiskRoutingPolicy()
	policy.Enabled = true

	// Simulate routing decisions
	policy.ShouldSkipEnsemble(&HRSPrediction{Risk: 0.05, ConfidenceLower: 0.90, ConfidenceUpper: 0.92}) // Fast path
	policy.ShouldSkipEnsemble(&HRSPrediction{Risk: 0.15, ConfidenceLower: 0.90, ConfidenceUpper: 0.92}) // Standard
	policy.ShouldSkipEnsemble(&HRSPrediction{Risk: 0.05, ConfidenceLower: 0.90, ConfidenceUpper: 0.93}) // Fast path
	policy.ShouldSkipEnsemble(&HRSPrediction{Risk: 0.20, ConfidenceLower: 0.90, ConfidenceUpper: 0.92}) // Standard

	metrics := policy.GetMetrics()

	if metrics.TotalDecisions != 4 {
		t.Errorf("Expected 4 total decisions, got %d", metrics.TotalDecisions)
	}

	if metrics.FastPathCount != 2 {
		t.Errorf("Expected 2 fast path decisions, got %d", metrics.FastPathCount)
	}

	if metrics.StandardPathCount != 2 {
		t.Errorf("Expected 2 standard path decisions, got %d", metrics.StandardPathCount)
	}

	expectedPercentage := 50.0
	if metrics.FastPathPercentage < expectedPercentage-0.1 || metrics.FastPathPercentage > expectedPercentage+0.1 {
		t.Errorf("Expected fast path percentage ~%.1f%%, got %.1f%%",
			expectedPercentage, metrics.FastPathPercentage)
	}
}

func TestResetMetrics(t *testing.T) {
	policy := DefaultRiskRoutingPolicy()
	policy.Enabled = true

	// Make some decisions
	policy.ShouldSkipEnsemble(&HRSPrediction{Risk: 0.05, ConfidenceLower: 0.90, ConfidenceUpper: 0.92})
	policy.ShouldSkipEnsemble(&HRSPrediction{Risk: 0.05, ConfidenceLower: 0.90, ConfidenceUpper: 0.92})

	metrics := policy.GetMetrics()
	if metrics.TotalDecisions != 2 {
		t.Errorf("Expected 2 decisions before reset, got %d", metrics.TotalDecisions)
	}

	// Reset
	policy.ResetMetrics()

	metrics = policy.GetMetrics()
	if metrics.TotalDecisions != 0 {
		t.Errorf("Expected 0 decisions after reset, got %d", metrics.TotalDecisions)
	}
	if metrics.FastPathCount != 0 {
		t.Errorf("Expected 0 fast path decisions after reset, got %d", metrics.FastPathCount)
	}
	if metrics.StandardPathCount != 0 {
		t.Errorf("Expected 0 standard path decisions after reset, got %d", metrics.StandardPathCount)
	}
}

func TestEnableDisable(t *testing.T) {
	policy := DefaultRiskRoutingPolicy()

	// Initially disabled
	if policy.Enabled {
		t.Error("Expected policy to be disabled by default")
	}

	// Enable with custom thresholds
	policy.Enable(0.15, 0.08)

	if !policy.Enabled {
		t.Error("Expected policy to be enabled")
	}
	if policy.SkipEnsembleBelow != 0.15 {
		t.Errorf("Expected SkipEnsembleBelow=0.15, got %.2f", policy.SkipEnsembleBelow)
	}
	if policy.MaxCIWidth != 0.08 {
		t.Errorf("Expected MaxCIWidth=0.08, got %.2f", policy.MaxCIWidth)
	}

	// Disable
	policy.Disable()

	if policy.Enabled {
		t.Error("Expected policy to be disabled")
	}
}

func TestConcurrentAccess(t *testing.T) {
	policy := DefaultRiskRoutingPolicy()
	policy.Enabled = true

	done := make(chan bool)

	// Simulate concurrent routing decisions
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				hrs := &HRSPrediction{
					Risk:            0.05,
					ConfidenceLower: 0.90,
					ConfidenceUpper: 0.92,
				}
				policy.ShouldSkipEnsemble(hrs)
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	metrics := policy.GetMetrics()
	if metrics.TotalDecisions != 1000 {
		t.Errorf("Expected 1000 decisions with concurrent access, got %d", metrics.TotalDecisions)
	}
}
