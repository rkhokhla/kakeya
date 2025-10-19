package api

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// PCS represents a Proof-of-Computation Summary
type PCS struct {
	PCSID      string             `json:"pcs_id"`
	Schema     string             `json:"schema"`
	Version    string             `json:"version"`
	ShardID    string             `json:"shard_id"`
	Epoch      int                `json:"epoch"`
	Attempt    int                `json:"attempt"`
	SentAt     time.Time          `json:"sent_at"`
	Seed       int64              `json:"seed"`
	Scales     []int              `json:"scales"`
	Nj         map[string]int     `json:"N_j"`
	CohStar    float64            `json:"coh_star"`
	VStar      []float64          `json:"v_star"`
	DHat       float64            `json:"D_hat"`
	R          float64            `json:"r"`
	Regime     string             `json:"regime"`
	Budget     float64            `json:"budget"`
	MerkleRoot string             `json:"merkle_root"`
	Sig        string             `json:"sig"`
	FT         FaultToleranceInfo `json:"ft"`
}

// FaultToleranceInfo contains metadata about delivery and system state
type FaultToleranceInfo struct {
	OutboxSeq   int64    `json:"outbox_seq"`
	Degraded    bool     `json:"degraded"`
	Fallbacks   []string `json:"fallbacks"`
	ClockSkewMs int      `json:"clock_skew_ms"`
}

// VerifyResult contains the outcome of PCS verification
type VerifyResult struct {
	Accepted        bool    `json:"accepted"`
	RecomputedDHat  float64 `json:"recomputed_D_hat,omitempty"`
	RecomputedBudget float64 `json:"recomputed_budget,omitempty"`
	Reason          string  `json:"reason,omitempty"`
	Escalated       bool    `json:"escalated"`
}

// VerifyParams contains tolerances and budget calculation parameters
type VerifyParams struct {
	TolD      float64 `json:"tol_D"`
	TolCoh    float64 `json:"tol_coh"`
	Alpha     float64 `json:"alpha"`
	Beta      float64 `json:"beta"`
	Gamma     float64 `json:"gamma"`
	Base      float64 `json:"base"`
	D0        float64 `json:"D0"`
	DedupTTL  time.Duration `json:"dedup_ttl"`
}

// DefaultVerifyParams returns the standard parameters from CLAUDE.md
func DefaultVerifyParams() VerifyParams {
	return VerifyParams{
		TolD:     0.15,
		TolCoh:   0.05,
		Alpha:    0.30,
		Beta:     0.50,
		Gamma:    0.20,
		Base:     0.10,
		D0:       2.2,
		DedupTTL: 14 * 24 * time.Hour,
	}
}

// Round9 rounds a float64 to 9 decimal places for signature stability
func Round9(x float64) float64 {
	return float64(int64(x*1e9+0.5)) / 1e9
}

// ComputePCSID computes the canonical pcs_id = sha256(merkle_root|epoch|shard_id)
func ComputePCSID(merkleRoot string, epoch int, shardID string) string {
	data := fmt.Sprintf("%s|%d|%s", merkleRoot, epoch, shardID)
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

// SigningPayload returns the canonical JSON subset used for signing
func (p *PCS) SigningPayload() ([]byte, error) {
	payload := map[string]interface{}{
		"pcs_id":      p.PCSID,
		"merkle_root": p.MerkleRoot,
		"epoch":       p.Epoch,
		"shard_id":    p.ShardID,
		"D_hat":       Round9(p.DHat),
		"coh_star":    Round9(p.CohStar),
		"r":           Round9(p.R),
		"budget":      Round9(p.Budget),
	}

	// Serialize with sorted keys, no spaces
	keys := make([]string, 0, len(payload))
	for k := range payload {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var sb strings.Builder
	sb.WriteString("{")
	for i, k := range keys {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(fmt.Sprintf("\"%s\":", k))

		v := payload[k]
		switch val := v.(type) {
		case string:
			sb.WriteString(fmt.Sprintf("\"%s\"", val))
		case int:
			sb.WriteString(fmt.Sprintf("%d", val))
		case float64:
			sb.WriteString(fmt.Sprintf("%.9f", val))
		}
	}
	sb.WriteString("}")

	return []byte(sb.String()), nil
}

// Validate performs basic structural validation
func (p *PCS) Validate() error {
	if p.PCSID == "" {
		return fmt.Errorf("pcs_id is required")
	}
	if p.Schema != "fractal-lba-kakeya" {
		return fmt.Errorf("invalid schema: expected fractal-lba-kakeya, got %s", p.Schema)
	}
	if p.Version == "" {
		return fmt.Errorf("version is required")
	}
	if p.ShardID == "" {
		return fmt.Errorf("shard_id is required")
	}
	if p.Epoch < 0 {
		return fmt.Errorf("epoch must be non-negative")
	}
	if len(p.Scales) == 0 {
		return fmt.Errorf("scales cannot be empty")
	}
	if p.MerkleRoot == "" {
		return fmt.Errorf("merkle_root is required")
	}

	// Verify pcs_id matches computed value
	expectedID := ComputePCSID(p.MerkleRoot, p.Epoch, p.ShardID)
	if p.PCSID != expectedID {
		return fmt.Errorf("pcs_id mismatch: expected %s, got %s", expectedID, p.PCSID)
	}

	return nil
}
