// Package canonical provides canonical JSON utilities for FLK/PCS signature generation.
// Phase 10 WP1: Ensures cross-language signature compatibility (Python/Go/TS).
//
// Key requirements:
// - Floats formatted to exactly 9 decimal places
// - Stable field ordering for signature subset
// - No whitespace in JSON output
package canonical

import (
	"encoding/json"
	"fmt"
	"strconv"
)

// SignatureFields defines the 8-field subset used for signing (Phase 1 specification)
var SignatureFields = []string{
	"D_hat",
	"budget",
	"coh_star",
	"epoch",
	"merkle_root",
	"pcs_id",
	"r",
	"shard_id",
}

// SignatureSubset represents the minimal PCS fields used for signing
type SignatureSubset struct {
	PCSID      string  `json:"pcs_id"`
	MerkleRoot string  `json:"merkle_root"`
	Epoch      int     `json:"epoch"`
	ShardID    string  `json:"shard_id"`
	DHat       float64 `json:"D_hat"`
	CohStar    float64 `json:"coh_star"`
	R          float64 `json:"r"`
	Budget     float64 `json:"budget"`
}

// F9 formats a float64 to exactly 9 decimal places.
//
// This ensures stable signatures across languages and prevents
// floating-point drift.
//
// Example:
//
//	F9(1.23456789012345) // returns "1.234567890"
//	F9(0.5)              // returns "0.500000000"
func F9(x float64) string {
	return strconv.FormatFloat(x, 'f', 9, 64)
}

// Round9 rounds a float64 to 9 decimal places.
//
// Used for normalizing floats before formatting to ensure
// consistent behavior across operations.
func Round9(x float64) float64 {
	// Multiply by 10^9, round, divide by 10^9
	const factor = 1e9
	return float64(int64(x*factor+0.5)) / factor
}

// CanonicalJSONBytes generates canonical JSON bytes for signing.
//
// Rules:
//   - Signature fields only (8 fields)
//   - Floats formatted to 9 decimal places
//   - Keys sorted alphabetically
//   - No whitespace (compact JSON)
//   - UTF-8 encoded
//
// Args:
//
//	subset: SignatureSubset with required fields
//
// Returns:
//
//	Canonical JSON as bytes, ready for signing
func CanonicalJSONBytes(subset *SignatureSubset) ([]byte, error) {
	// Normalize floats to 9 decimal places
	normalized := map[string]interface{}{
		"D_hat":       Round9(subset.DHat),
		"budget":      Round9(subset.Budget),
		"coh_star":    Round9(subset.CohStar),
		"epoch":       subset.Epoch,
		"merkle_root": subset.MerkleRoot,
		"pcs_id":      subset.PCSID,
		"r":           Round9(subset.R),
		"shard_id":    subset.ShardID,
	}

	// Marshal with no indentation (compact)
	// Go's json.Marshal already sorts keys alphabetically
	return json.Marshal(normalized)
}

// SignaturePayload extracts signature subset and generates canonical payload.
//
// This is the main entry point for generating bytes to sign.
//
// Example:
//
//	subset := &SignatureSubset{
//	    PCSID:      "test",
//	    DHat:       1.23456789,
//	    CohStar:    0.75,
//	    R:          0.5,
//	    Budget:     0.35,
//	    MerkleRoot: "abc",
//	    Epoch:      1,
//	    ShardID:    "s1",
//	}
//	payload, err := SignaturePayload(subset)
func SignaturePayload(subset *SignatureSubset) ([]byte, error) {
	if subset.PCSID == "" {
		return nil, fmt.Errorf("missing required field: pcs_id")
	}
	if subset.MerkleRoot == "" {
		return nil, fmt.Errorf("missing required field: merkle_root")
	}
	if subset.ShardID == "" {
		return nil, fmt.Errorf("missing required field: shard_id")
	}

	return CanonicalJSONBytes(subset)
}

// ExtractSignatureSubset extracts signature fields from a PCS map.
//
// This helper function is useful when working with dynamic PCS structures.
func ExtractSignatureSubset(pcs map[string]interface{}) (*SignatureSubset, error) {
	subset := &SignatureSubset{}

	// Extract required fields
	if v, ok := pcs["pcs_id"].(string); ok {
		subset.PCSID = v
	} else {
		return nil, fmt.Errorf("missing or invalid pcs_id")
	}

	if v, ok := pcs["merkle_root"].(string); ok {
		subset.MerkleRoot = v
	} else {
		return nil, fmt.Errorf("missing or invalid merkle_root")
	}

	if v, ok := pcs["epoch"].(float64); ok {
		subset.Epoch = int(v)
	} else if v, ok := pcs["epoch"].(int); ok {
		subset.Epoch = v
	} else {
		return nil, fmt.Errorf("missing or invalid epoch")
	}

	if v, ok := pcs["shard_id"].(string); ok {
		subset.ShardID = v
	} else {
		return nil, fmt.Errorf("missing or invalid shard_id")
	}

	if v, ok := pcs["D_hat"].(float64); ok {
		subset.DHat = v
	} else {
		return nil, fmt.Errorf("missing or invalid D_hat")
	}

	if v, ok := pcs["coh_star"].(float64); ok {
		subset.CohStar = v
	} else {
		return nil, fmt.Errorf("missing or invalid coh_star")
	}

	if v, ok := pcs["r"].(float64); ok {
		subset.R = v
	} else {
		return nil, fmt.Errorf("missing or invalid r")
	}

	if v, ok := pcs["budget"].(float64); ok {
		subset.Budget = v
	} else {
		return nil, fmt.Errorf("missing or invalid budget")
	}

	return subset, nil
}
