package verify

import (
	"testing"
)

// Phase 11 WP5: Fuzz testing for PCS verification
// Tests resilience against malformed inputs

// FuzzRecomputeDHat fuzzes D̂ computation
func FuzzRecomputeDHat(f *testing.F) {
	// Seed with valid inputs
	f.Add([]int{2, 4, 8, 16})

	f.Fuzz(func(t *testing.T, scales []int) {
		// Should not crash on any input combination
		if len(scales) == 0 || len(scales) > 100 {
			return
		}

		// Create valid N_j map
		nj := make(map[string]int)
		for _, scale := range scales {
			if scale > 0 && scale < 10000 {
				nj[string(rune(scale))] = scale + 1
			}
		}

		// Should not crash
		_ = RecomputeDHat(scales, nj)
	})
}
