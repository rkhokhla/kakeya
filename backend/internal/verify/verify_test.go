package verify

import (
	"testing"

	"github.com/fractal-lba/kakeya/internal/api"
)

func TestRecomputeDHat(t *testing.T) {
	engine := NewEngine(api.DefaultVerifyParams())

	scales := []int{2, 4, 8, 16, 32}
	Nj := map[string]int{
		"2":  3,
		"4":  5,
		"8":  9,
		"16": 17,
		"32": 31,
	}

	dhat, err := engine.RecomputeDHat(scales, Nj)
	if err != nil {
		t.Fatalf("RecomputeDHat failed: %v", err)
	}

	// Expected slope should be close to 1 (log2(N_j) ≈ log2(scale))
	if dhat < 0.8 || dhat > 1.2 {
		t.Errorf("Expected D_hat near 1.0, got %.4f", dhat)
	}
}

func TestClassifyRegime(t *testing.T) {
	engine := NewEngine(api.DefaultVerifyParams())

	tests := []struct {
		name     string
		dhat     float64
		cohStar  float64
		expected string
	}{
		{"sticky", 1.3, 0.75, "sticky"},
		{"non_sticky", 2.7, 0.50, "non_sticky"},
		{"mixed", 2.0, 0.60, "mixed"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := engine.ClassifyRegime(tt.dhat, tt.cohStar)
			if result != tt.expected {
				t.Errorf("ClassifyRegime(%f, %f) = %s, want %s", tt.dhat, tt.cohStar, result, tt.expected)
			}
		})
	}
}

func TestComputeBudget(t *testing.T) {
	engine := NewEngine(api.DefaultVerifyParams())

	// Test budget clamping
	budget := engine.ComputeBudget(3.0, 1.0, 0.1)

	if budget < 0 || budget > 1 {
		t.Errorf("Budget out of bounds [0, 1]: %.4f", budget)
	}
}

func TestVerify(t *testing.T) {
	engine := NewEngine(api.DefaultVerifyParams())

	pcs := &api.PCS{
		PCSID:      api.ComputePCSID("abc123", 1, "shard-001"),
		Schema:     "fractal-lba-kakeya",
		Version:    "0.1",
		ShardID:    "shard-001",
		Epoch:      1,
		Attempt:    1,
		Scales:     []int{2, 4, 8},
		Nj:         map[string]int{"2": 3, "4": 5, "8": 9},
		DHat:       0.96,
		CohStar:    0.75,
		R:          0.50,
		Regime:     "sticky",
		Budget:     0.35,
		MerkleRoot: "abc123",
	}

	result, err := engine.Verify(pcs)
	if err != nil {
		t.Fatalf("Verify failed: %v", err)
	}

	if !result.Accepted {
		t.Errorf("Expected PCS to be accepted, got rejected: %s", result.Reason)
	}
}
