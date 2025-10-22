package verify

import (
	"testing"

	"github.com/fractal-lba/kakeya/internal/api"
)

// Phase 11 WP5: Fuzz testing for PCS verification
// Tests resilience against malformed inputs

// FuzzRecomputeDHat fuzzes D̂ computation
func FuzzRecomputeDHat(f *testing.F) {
	// Seed with valid inputs (individual ints, will build array in fuzz function)
	f.Add(2, 4, 8, 16)

	f.Fuzz(func(t *testing.T, s1, s2, s3, s4 int) {
		// Build scales array from individual values
		scales := []int{}
		if s1 > 0 && s1 < 10000 {
			scales = append(scales, s1)
		}
		if s2 > 0 && s2 < 10000 {
			scales = append(scales, s2)
		}
		if s3 > 0 && s3 < 10000 {
			scales = append(scales, s3)
		}
		if s4 > 0 && s4 < 10000 {
			scales = append(scales, s4)
		}

		// Need at least 2 scales for meaningful test
		if len(scales) < 2 {
			return
		}

		// Create valid N_j map
		nj := make(map[string]int)
		for _, scale := range scales {
			nj[string(rune(scale))] = scale + 1
		}

		// Create engine and test - should not crash
		engine := NewEngine(api.VerifyParams{TolD: 0.15, TolCoh: 0.05})
		_, _ = engine.RecomputeDHat(scales, nj)
	})
}
