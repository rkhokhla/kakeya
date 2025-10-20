package signing

import (
	"crypto/sha256"
	"encoding/json"
	"math"

	"github.com/fractal-lba/kakeya/internal/api"
)

// SignatureSubset represents the canonical subset of PCS fields used for signing.
// Field order in the struct determines JSON marshaling order.
type SignatureSubset struct {
	Budget     float64 `json:"budget"`
	CohStar    float64 `json:"coh_star"`
	DHat       float64 `json:"D_hat"`
	Epoch      int     `json:"epoch"`
	MerkleRoot string  `json:"merkle_root"`
	PCSID      string  `json:"pcs_id"`
	R          float64 `json:"r"`
	ShardID    string  `json:"shard_id"`
}

// Round9 rounds a float64 to 9 decimal places for signature stability.
func Round9(x float64) float64 {
	return math.Round(x*1e9) / 1e9
}

// SignaturePayload generates the canonical signature payload from a PCS.
//
// Per CLAUDE.md §2.1:
// 1. Extract signature subset (pcs_id, merkle_root, epoch, shard_id, D_hat, coh_star, r, budget)
// 2. Round numeric fields to 9 decimals
// 3. Serialize to JSON with sorted keys (struct field order is alphabetical by json tag)
// 4. Return UTF-8 bytes
func SignaturePayload(pcs *api.PCS) ([]byte, error) {
	subset := SignatureSubset{
		Budget:     Round9(pcs.Budget),
		CohStar:    Round9(pcs.CohStar),
		DHat:       Round9(pcs.DHat),
		Epoch:      pcs.Epoch,
		MerkleRoot: pcs.MerkleRoot,
		PCSID:      pcs.PCSID,
		R:          Round9(pcs.R),
		ShardID:    pcs.ShardID,
	}

	// Marshal with no indentation or extra spaces
	return json.Marshal(subset)
}

// SignatureDigest computes the SHA-256 digest of the canonical signature payload.
//
// This is the value that gets signed/verified.
func SignatureDigest(pcs *api.PCS) ([]byte, error) {
	payload, err := SignaturePayload(pcs)
	if err != nil {
		return nil, err
	}

	hash := sha256.Sum256(payload)
	return hash[:], nil
}
