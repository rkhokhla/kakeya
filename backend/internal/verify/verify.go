package verify

import (
	"fmt"
	"math"
	"sort"

	"github.com/fractal-lba/kakeya/internal/api"
)

// Engine performs PCS verification according to CLAUDE.md invariants
type Engine struct {
	params api.VerifyParams
}

// NewEngine creates a verification engine with given parameters
func NewEngine(params api.VerifyParams) *Engine {
	return &Engine{params: params}
}

// Verify performs complete PCS verification
func (e *Engine) Verify(pcs *api.PCS) (*api.VerifyResult, error) {
	// 1. Validate structure
	if err := pcs.Validate(); err != nil {
		return &api.VerifyResult{
			Accepted:  false,
			Escalated: true,
			Reason:    fmt.Sprintf("validation failed: %v", err),
		}, nil
	}

	// 2. Recompute D_hat from scales and N_j
	recomputedDHat, err := e.RecomputeDHat(pcs.Scales, pcs.Nj)
	if err != nil {
		return &api.VerifyResult{
			Accepted:  false,
			Escalated: true,
			Reason:    fmt.Sprintf("D_hat recomputation failed: %v", err),
		}, nil
	}

	// 3. Check D_hat tolerance
	dhatDiff := math.Abs(pcs.DHat - recomputedDHat)
	if dhatDiff > e.params.TolD {
		return &api.VerifyResult{
			Accepted:       false,
			Escalated:      true,
			RecomputedDHat: recomputedDHat,
			Reason:         fmt.Sprintf("D_hat out of tolerance: claimed=%.4f, recomputed=%.4f, diff=%.4f > tol=%.4f", pcs.DHat, recomputedDHat, dhatDiff, e.params.TolD),
		}, nil
	}

	// 4. Check coherence bounds
	if pcs.CohStar < 0 || pcs.CohStar > 1+e.params.TolCoh {
		return &api.VerifyResult{
			Accepted:  false,
			Escalated: true,
			Reason:    fmt.Sprintf("coh_star out of bounds: %.4f not in [0, %.4f]", pcs.CohStar, 1+e.params.TolCoh),
		}, nil
	}

	// 5. Verify regime classification
	expectedRegime := e.ClassifyRegime(pcs.DHat, pcs.CohStar)
	if pcs.Regime != expectedRegime {
		// This is a soft warning - accept but escalate
		return &api.VerifyResult{
			Accepted:         true,
			Escalated:        true,
			RecomputedDHat:   recomputedDHat,
			RecomputedBudget: e.ComputeBudget(pcs.DHat, pcs.CohStar, pcs.R),
			Reason:           fmt.Sprintf("regime mismatch: claimed=%s, expected=%s", pcs.Regime, expectedRegime),
		}, nil
	}

	// 6. Recompute and check budget
	recomputedBudget := e.ComputeBudget(pcs.DHat, pcs.CohStar, pcs.R)
	budgetDiff := math.Abs(pcs.Budget - recomputedBudget)
	if budgetDiff > 0.01 { // 1% tolerance
		return &api.VerifyResult{
			Accepted:         true,
			Escalated:        true,
			RecomputedDHat:   recomputedDHat,
			RecomputedBudget: recomputedBudget,
			Reason:           fmt.Sprintf("budget deviation: claimed=%.4f, recomputed=%.4f", pcs.Budget, recomputedBudget),
		}, nil
	}

	// All checks passed
	return &api.VerifyResult{
		Accepted:         true,
		Escalated:        false,
		RecomputedDHat:   recomputedDHat,
		RecomputedBudget: recomputedBudget,
	}, nil
}

// RecomputeDHat computes the fractal dimension using Theil-Sen estimator
func (e *Engine) RecomputeDHat(scales []int, nj map[string]int) (float64, error) {
	if len(scales) < 2 {
		return 0, fmt.Errorf("need at least 2 scales for regression")
	}

	// Build (log2(scale), log2(N_j)) pairs
	type point struct {
		x, y float64
	}
	var points []point

	for _, scale := range scales {
		key := fmt.Sprintf("%d", scale)
		n, ok := nj[key]
		if !ok {
			return 0, fmt.Errorf("missing N_j entry for scale %d", scale)
		}
		if n <= 0 {
			return 0, fmt.Errorf("N_j must be positive for scale %d", scale)
		}

		points = append(points, point{
			x: math.Log2(float64(scale)),
			y: math.Log2(float64(n)),
		})
	}

	// Theil-Sen: median slope of all pairwise slopes
	var slopes []float64
	for i := 0; i < len(points); i++ {
		for j := i + 1; j < len(points); j++ {
			dx := points[j].x - points[i].x
			if math.Abs(dx) < 1e-9 {
				continue
			}
			dy := points[j].y - points[i].y
			slope := dy / dx
			slopes = append(slopes, slope)
		}
	}

	if len(slopes) == 0 {
		return 0, fmt.Errorf("no valid slopes computed")
	}

	sort.Float64s(slopes)
	medianSlope := slopes[len(slopes)/2]

	return api.Round9(medianSlope), nil
}

// ClassifyRegime determines sticky/mixed/non_sticky based on D_hat and coh_star
func (e *Engine) ClassifyRegime(dhat, cohStar float64) string {
	if cohStar >= 0.70 && dhat <= 1.5 {
		return "sticky"
	}
	if dhat >= 2.6 {
		return "non_sticky"
	}
	return "mixed"
}

// ComputeBudget calculates budget according to CLAUDE.md formula
func (e *Engine) ComputeBudget(dhat, cohStar, r float64) float64 {
	budget := e.params.Base +
		e.params.Alpha*(1-r) +
		e.params.Beta*math.Max(0, dhat-e.params.D0) +
		e.params.Gamma*cohStar

	// Clamp to [0, 1]
	if budget < 0 {
		budget = 0
	}
	if budget > 1 {
		budget = 1
	}

	return api.Round9(budget)
}

// Params returns the verification parameters
func (e *Engine) Params() api.VerifyParams {
	return e.params
}
