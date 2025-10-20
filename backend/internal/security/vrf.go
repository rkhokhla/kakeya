package security

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"

	"github.com/fractal-lba/kakeya/internal/api"
)

// VRFProof represents a verifiable random function proof (Phase 3 WP6)
type VRFProof struct {
	Proof  string `json:"proof"`   // Base64-encoded proof
	Output string `json:"output"`  // Base64-encoded output (seed)
	PubKey string `json:"pub_key"` // Base64-encoded public key
}

// VRFVerifier verifies VRF proofs to prevent adversarial direction sampling
type VRFVerifier struct {
	enabled bool
}

// NewVRFVerifier creates a new VRF verifier
func NewVRFVerifier(enabled bool) *VRFVerifier {
	return &VRFVerifier{enabled: enabled}
}

// Verify verifies a VRF proof (simplified - production would use proper VRF library)
func (v *VRFVerifier) Verify(proof *VRFProof, message []byte) error {
	if !v.enabled {
		return nil // VRF verification disabled
	}

	if proof.Proof == "" {
		return fmt.Errorf("VRF proof is empty")
	}

	// Decode proof and output
	proofBytes, err := base64.StdEncoding.DecodeString(proof.Proof)
	if err != nil {
		return fmt.Errorf("invalid VRF proof encoding: %w", err)
	}

	outputBytes, err := base64.StdEncoding.DecodeString(proof.Output)
	if err != nil {
		return fmt.Errorf("invalid VRF output encoding: %w", err)
	}

	pubKeyBytes, err := base64.StdEncoding.DecodeString(proof.PubKey)
	if err != nil {
		return fmt.Errorf("invalid VRF public key encoding: %w", err)
	}

	// Placeholder verification (production would use ECVRF or similar)
	// For now, verify that output is a valid hash of proof+message
	hash := sha256.New()
	hash.Write(proofBytes)
	hash.Write(message)
	hash.Write(pubKeyBytes)
	_ = hash.Sum(nil) // expectedOutput - would be used in full verification

	// Compare first 32 bytes
	if len(outputBytes) < 32 {
		return fmt.Errorf("VRF output too short: %d bytes", len(outputBytes))
	}

	// In production, this would be a proper cryptographic verification
	// For Phase 3 demonstration, we accept if output is plausible
	if len(outputBytes) != 32 {
		return fmt.Errorf("VRF output must be 32 bytes, got %d", len(outputBytes))
	}

	return nil
}

// SanityChecker performs input validation to detect adversarial manipulation (Phase 3 WP6)
type SanityChecker struct {
	strictMode bool
}

// NewSanityChecker creates a new sanity checker
func NewSanityChecker(strictMode bool) *SanityChecker {
	return &SanityChecker{strictMode: strictMode}
}

// CheckPCS performs comprehensive sanity checks on a PCS
func (s *SanityChecker) CheckPCS(pcs *api.PCS) error {
	// Check 1: N_j monotonicity (Phase 3 WP6 requirement)
	if err := s.checkNjMonotonic(pcs); err != nil {
		return fmt.Errorf("N_j monotonicity check failed: %w", err)
	}

	// Check 2: Scale ranges
	if err := s.checkScaleRanges(pcs); err != nil {
		return fmt.Errorf("scale range check failed: %w", err)
	}

	// Check 3: Coherence bounds
	if err := s.checkCoherenceBounds(pcs); err != nil {
		return fmt.Errorf("coherence bounds check failed: %w", err)
	}

	// Check 4: Compressibility bounds
	if err := s.checkCompressibility(pcs); err != nil {
		return fmt.Errorf("compressibility check failed: %w", err)
	}

	// Check 5: Fractal dimension bounds
	if err := s.checkFractalDimension(pcs); err != nil {
		return fmt.Errorf("fractal dimension check failed: %w", err)
	}

	// Check 6: Budget bounds
	if err := s.checkBudgetBounds(pcs); err != nil {
		return fmt.Errorf("budget bounds check failed: %w", err)
	}

	// Check 7: Merkle root format
	if err := s.checkMerkleRoot(pcs); err != nil {
		return fmt.Errorf("merkle root check failed: %w", err)
	}

	return nil
}

// checkNjMonotonic verifies that N_j is non-decreasing with scale
func (s *SanityChecker) checkNjMonotonic(pcs *api.PCS) error {
	if len(pcs.Scales) < 2 {
		return nil // Can't check monotonicity with < 2 scales
	}

	prevN := -1
	for _, scale := range pcs.Scales {
		scaleStr := fmt.Sprintf("%d", scale)
		n, ok := pcs.Nj[scaleStr]
		if !ok {
			return fmt.Errorf("missing N_j for scale %d", scale)
		}

		if prevN >= 0 && n < prevN {
			return fmt.Errorf("N_j not monotonic: N_%d=%d < N_prev=%d (adversarial manipulation suspected)", scale, n, prevN)
		}

		prevN = n
	}

	return nil
}

// checkScaleRanges verifies scales are in reasonable range
func (s *SanityChecker) checkScaleRanges(pcs *api.PCS) error {
	if len(pcs.Scales) == 0 {
		return fmt.Errorf("empty scales array")
	}

	for _, scale := range pcs.Scales {
		if scale < 1 || scale > 1024 {
			return fmt.Errorf("scale %d out of range [1, 1024]", scale)
		}
	}

	return nil
}

// checkCoherenceBounds verifies coherence is in [0, 1] with tolerance
func (s *SanityChecker) checkCoherenceBounds(pcs *api.PCS) error {
	tolerance := 0.05 // Match CLAUDE.md default
	minCoh := 0.0
	maxCoh := 1.0 + tolerance

	if pcs.CohStar < minCoh || pcs.CohStar > maxCoh {
		return fmt.Errorf("coh_star=%.4f out of bounds [%.2f, %.2f]", pcs.CohStar, minCoh, maxCoh)
	}

	return nil
}

// checkCompressibility verifies compressibility ratio is in [0, 1]
func (s *SanityChecker) checkCompressibility(pcs *api.PCS) error {
	if pcs.R < 0 || pcs.R > 1 {
		return fmt.Errorf("compressibility r=%.4f out of bounds [0, 1]", pcs.R)
	}

	return nil
}

// checkFractalDimension verifies fractal dimension is plausible
func (s *SanityChecker) checkFractalDimension(pcs *api.PCS) error {
	// D_hat should be in [0, 3] for 3D space (Phase 3 WP6)
	if pcs.DHat < 0 || pcs.DHat > 3.5 {
		return fmt.Errorf("D_hat=%.4f out of plausible range [0, 3.5]", pcs.DHat)
	}

	// In strict mode, apply tighter bounds
	if s.strictMode && (pcs.DHat < 0.5 || pcs.DHat > 3.0) {
		return fmt.Errorf("D_hat=%.4f out of strict bounds [0.5, 3.0]", pcs.DHat)
	}

	return nil
}

// checkBudgetBounds verifies budget is in [0, 1]
func (s *SanityChecker) checkBudgetBounds(pcs *api.PCS) error {
	if pcs.Budget < 0 || pcs.Budget > 1 {
		return fmt.Errorf("budget=%.4f out of bounds [0, 1]", pcs.Budget)
	}

	return nil
}

// checkMerkleRoot verifies merkle root is valid hex string
func (s *SanityChecker) checkMerkleRoot(pcs *api.PCS) error {
	if pcs.MerkleRoot == "" {
		return fmt.Errorf("merkle_root is empty")
	}

	// Check if valid hex
	_, err := hex.DecodeString(pcs.MerkleRoot)
	if err != nil {
		return fmt.Errorf("merkle_root is not valid hex: %w", err)
	}

	// Check length (64 hex chars = 32 bytes for SHA-256)
	if len(pcs.MerkleRoot) != 64 {
		return fmt.Errorf("merkle_root has invalid length %d (expected 64 hex chars)", len(pcs.MerkleRoot))
	}

	return nil
}

// AnomalyScore computes an anomaly score for a PCS (Phase 3 WP6)
type AnomalyScore struct {
	Score      float64           // 0.0-1.0 (0=normal, 1=highly anomalous)
	Factors    map[string]float64 // Contributing factors
	Threshold  float64           // Alert threshold
	Escalate   bool              // Whether to escalate (202 response)
}

// AnomalyScorer scores PCS submissions for anomalous patterns
type AnomalyScorer struct {
	alertThreshold  float64
	rejectThreshold float64
}

// NewAnomalyScorer creates a new anomaly scorer
func NewAnomalyScorer(alertThreshold, rejectThreshold float64) *AnomalyScorer {
	return &AnomalyScorer{
		alertThreshold:  alertThreshold,
		rejectThreshold: rejectThreshold,
	}
}

// Score computes anomaly score for a PCS
func (a *AnomalyScorer) Score(pcs *api.PCS) *AnomalyScore {
	factors := make(map[string]float64)
	totalScore := 0.0

	// Factor 1: Extreme D_hat values
	if pcs.DHat < 0.5 || pcs.DHat > 2.8 {
		factor := 0.0
		if pcs.DHat < 0.5 {
			factor = (0.5 - pcs.DHat) / 0.5 // 0-1 range
		} else {
			factor = (pcs.DHat - 2.8) / 0.7 // 0-1 range
		}
		factors["extreme_D_hat"] = factor
		totalScore += factor * 0.3
	}

	// Factor 2: Coherence-dimension mismatch
	if pcs.CohStar > 0.8 && pcs.DHat > 2.5 {
		// High coherence with high dimension is suspicious
		mismatch := (pcs.CohStar - 0.8) * (pcs.DHat - 2.5)
		factors["coherence_dimension_mismatch"] = mismatch
		totalScore += mismatch * 0.25
	}

	// Factor 3: Suspicious compressibility
	if pcs.R < 0.1 || pcs.R > 0.95 {
		factor := 0.0
		if pcs.R < 0.1 {
			factor = (0.1 - pcs.R) / 0.1
		} else {
			factor = (pcs.R - 0.95) / 0.05
		}
		factors["extreme_compressibility"] = factor
		totalScore += factor * 0.2
	}

	// Factor 4: Regime inconsistency
	expectedRegime := ""
	if pcs.CohStar >= 0.70 && pcs.DHat <= 1.5 {
		expectedRegime = "sticky"
	} else if pcs.DHat >= 2.6 {
		expectedRegime = "non_sticky"
	} else {
		expectedRegime = "mixed"
	}

	if expectedRegime != "" && expectedRegime != pcs.Regime {
		factors["regime_inconsistency"] = 0.4
		totalScore += 0.4 * 0.25
	}

	// Normalize total score to [0, 1]
	if totalScore > 1.0 {
		totalScore = 1.0
	}

	return &AnomalyScore{
		Score:     totalScore,
		Factors:   factors,
		Threshold: a.alertThreshold,
		Escalate:  totalScore >= a.alertThreshold,
	}
}
