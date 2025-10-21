package routing

import (
	"sync"
	"time"
)

// Phase 11 WP4: Risk-Based Routing (Fast Path)
// Skip ensemble for low-risk PCS (HRS confidence >0.90, narrow CI) to cut
// p95 latency 25-40% while preserving containment within ±0.5%

// HRSPrediction represents risk score from HRS (Hallucination Risk Scorer)
type HRSPrediction struct {
	Risk            float64 // Risk score in [0, 1]
	ConfidenceLower float64 // 95% CI lower bound
	ConfidenceUpper float64 // 95% CI upper bound
	ModelVersion    string  // HRS model version
	Features        map[string]float64
}

// RiskRoutingPolicy defines fast path routing rules
type RiskRoutingPolicy struct {
	Enabled           bool
	SkipEnsembleBelow float64 // Default: 0.10 (skip if risk <10%)
	RequireNarrowCI   bool    // Default: true
	MaxCIWidth        float64 // Default: 0.05 (5pp CI width)

	// Metrics
	mu                    sync.RWMutex
	totalDecisions        int64
	fastPathCount         int64
	standardPathCount     int64
	lastDecisionTimestamp time.Time
}

// DefaultRiskRoutingPolicy returns production defaults
func DefaultRiskRoutingPolicy() *RiskRoutingPolicy {
	return &RiskRoutingPolicy{
		Enabled:           false, // Disabled by default (feature flag)
		SkipEnsembleBelow: 0.10,
		RequireNarrowCI:   true,
		MaxCIWidth:        0.05,
	}
}

// ShouldSkipEnsemble determines if a PCS should take the fast path
// Returns true if risk is low enough to skip ensemble verification
func (p *RiskRoutingPolicy) ShouldSkipEnsemble(hrs *HRSPrediction) bool {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.totalDecisions++
	p.lastDecisionTimestamp = time.Now()

	// Feature disabled - always use standard path
	if !p.Enabled {
		p.standardPathCount++
		return false
	}

	// Check risk threshold (must be strictly below)
	if hrs.Risk >= p.SkipEnsembleBelow {
		// Risk too high or at threshold - standard path
		p.standardPathCount++
		return false
	}

	// Check CI width if required (must be strictly below)
	if p.RequireNarrowCI {
		ciWidth := hrs.ConfidenceUpper - hrs.ConfidenceLower
		if ciWidth >= p.MaxCIWidth {
			// CI too wide or at threshold - standard path (low confidence)
			p.standardPathCount++
			return false
		}
	}

	// Fast path criteria met
	p.fastPathCount++
	return true
}

// GetRouteDecision returns detailed routing decision with reasons
func (p *RiskRoutingPolicy) GetRouteDecision(hrs *HRSPrediction) RouteDecision {
	if !p.Enabled {
		return RouteDecision{
			UseFastPath: false,
			Route:       "standard",
			Reason:      "risk_routing_disabled",
			HRSRisk:     hrs.Risk,
			CIWidth:     hrs.ConfidenceUpper - hrs.ConfidenceLower,
		}
	}

	ciWidth := hrs.ConfidenceUpper - hrs.ConfidenceLower

	// Check risk threshold (must be strictly below)
	if hrs.Risk >= p.SkipEnsembleBelow {
		return RouteDecision{
			UseFastPath: false,
			Route:       "standard",
			Reason:      "high_risk",
			HRSRisk:     hrs.Risk,
			CIWidth:     ciWidth,
			Threshold:   p.SkipEnsembleBelow,
		}
	}

	// Check CI width (must be strictly below)
	if p.RequireNarrowCI && ciWidth >= p.MaxCIWidth {
		return RouteDecision{
			UseFastPath: false,
			Route:       "standard",
			Reason:      "wide_confidence_interval",
			HRSRisk:     hrs.Risk,
			CIWidth:     ciWidth,
			Threshold:   p.MaxCIWidth,
		}
	}

	// Fast path approved
	return RouteDecision{
		UseFastPath: true,
		Route:       "fast_path",
		Reason:      "low_risk_narrow_ci",
		HRSRisk:     hrs.Risk,
		CIWidth:     ciWidth,
	}
}

// RouteDecision contains detailed routing decision information
type RouteDecision struct {
	UseFastPath bool
	Route       string  // "fast_path" or "standard"
	Reason      string  // Decision rationale
	HRSRisk     float64 // Risk score
	CIWidth     float64 // Confidence interval width
	Threshold   float64 // Threshold that was checked
}

// GetMetrics returns current routing metrics
func (p *RiskRoutingPolicy) GetMetrics() RiskRoutingMetrics {
	p.mu.RLock()
	defer p.mu.RUnlock()

	var fastPathPercentage float64
	if p.totalDecisions > 0 {
		fastPathPercentage = float64(p.fastPathCount) / float64(p.totalDecisions) * 100.0
	}

	return RiskRoutingMetrics{
		Enabled:               p.Enabled,
		TotalDecisions:        p.totalDecisions,
		FastPathCount:         p.fastPathCount,
		StandardPathCount:     p.standardPathCount,
		FastPathPercentage:    fastPathPercentage,
		LastDecisionTimestamp: p.lastDecisionTimestamp,
	}
}

// RiskRoutingMetrics holds routing statistics
type RiskRoutingMetrics struct {
	Enabled               bool
	TotalDecisions        int64
	FastPathCount         int64
	StandardPathCount     int64
	FastPathPercentage    float64
	LastDecisionTimestamp time.Time
}

// ResetMetrics clears routing metrics (for testing/ops)
func (p *RiskRoutingPolicy) ResetMetrics() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.totalDecisions = 0
	p.fastPathCount = 0
	p.standardPathCount = 0
}

// Enable turns on risk routing with optional custom thresholds
func (p *RiskRoutingPolicy) Enable(skipBelow, maxCIWidth float64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.Enabled = true
	if skipBelow > 0 {
		p.SkipEnsembleBelow = skipBelow
	}
	if maxCIWidth > 0 {
		p.MaxCIWidth = maxCIWidth
	}
}

// Disable turns off risk routing (all traffic to standard path)
func (p *RiskRoutingPolicy) Disable() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.Enabled = false
}
