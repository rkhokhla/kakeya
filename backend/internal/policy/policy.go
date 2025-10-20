package policy

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"
)

// Policy represents verification parameters and rules (Phase 3 WP4)
type Policy struct {
	Version     string    `json:"version"`      // Semantic version
	Name        string    `json:"name"`
	Description string    `json:"description"`
	CreatedAt   time.Time `json:"created_at"`
	Active      bool      `json:"active"`

	// Verification parameters
	TolD      float64 `json:"tol_D"`
	TolCoh    float64 `json:"tol_coh"`
	Alpha     float64 `json:"alpha"`
	Beta      float64 `json:"beta"`
	Gamma     float64 `json:"gamma"`
	Base      float64 `json:"base"`
	D0        float64 `json:"D0"`

	// Regime thresholds
	StickyDHatMax        float64 `json:"sticky_D_hat_max"`
	StickyCoherenceMin   float64 `json:"sticky_coherence_min"`
	NonStickyDHatMin     float64 `json:"non_sticky_D_hat_min"`

	// Feature flags
	Flags map[string]bool `json:"flags,omitempty"`

	// Signature for tamper-evidence
	Signature string `json:"signature,omitempty"`
	SignedBy  string `json:"signed_by,omitempty"`
}

// ValidationError represents a policy validation error
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("policy validation error [%s]: %s", e.Field, e.Message)
}

// Validate performs compile-time validation of policy (Phase 3 WP4)
func (p *Policy) Validate() error {
	// Check version
	if p.Version == "" {
		return &ValidationError{Field: "version", Message: "version is required"}
	}

	// Validate tolerance bounds
	if p.TolD < 0 || p.TolD > 1 {
		return &ValidationError{Field: "tol_D", Message: "must be in [0, 1]"}
	}
	if p.TolCoh < 0 || p.TolCoh > 0.2 {
		return &ValidationError{Field: "tol_coh", Message: "must be in [0, 0.2]"}
	}

	// Validate coherence bounds (0 ≤ coh★ ≤ 1+ε)
	if p.StickyCoherenceMin < 0 || p.StickyCoherenceMin > 1+p.TolCoh {
		return &ValidationError{
			Field:   "sticky_coherence_min",
			Message: fmt.Sprintf("must be in [0, 1+tol_coh=%.2f]", 1+p.TolCoh),
		}
	}

	// Validate budget weights (must sum to ≤ 1.0 for normalized contribution)
	weightSum := p.Alpha + p.Beta + p.Gamma
	if weightSum > 1.0 {
		return &ValidationError{
			Field:   "weights",
			Message: fmt.Sprintf("alpha+beta+gamma = %.2f exceeds 1.0 (non-normalized)", weightSum),
		}
	}
	if p.Alpha < 0 || p.Beta < 0 || p.Gamma < 0 {
		return &ValidationError{Field: "weights", Message: "alpha, beta, gamma must be non-negative"}
	}

	// Validate base budget
	if p.Base < 0 || p.Base > 1 {
		return &ValidationError{Field: "base", Message: "base budget must be in [0, 1]"}
	}

	// Validate D0 (fractal dimension reference)
	if p.D0 < 0 || p.D0 > 3 {
		return &ValidationError{Field: "D0", Message: "D0 must be in [0, 3] (dimension range)"}
	}

	// Validate regime thresholds
	if p.StickyDHatMax >= p.NonStickyDHatMin {
		return &ValidationError{
			Field:   "regime_thresholds",
			Message: "sticky_D_hat_max must be < non_sticky_D_hat_min (gap required for 'mixed' regime)",
		}
	}

	// Validate no dangerous operations (Phase 3 security requirement)
	if p.Flags != nil {
		if val, ok := p.Flags["disable_wal"]; ok && val {
			return &ValidationError{Field: "flags.disable_wal", Message: "disabling WAL is forbidden (safety invariant)"}
		}
		if val, ok := p.Flags["skip_signature"]; ok && val {
			return &ValidationError{Field: "flags.skip_signature", Message: "skipping signature verification is forbidden"}
		}
	}

	return nil
}

// Hash computes a stable hash of the policy for lineage tracking (Phase 3 WP3)
func (p *Policy) Hash() (string, error) {
	// Create canonical representation (exclude signature fields)
	canonical := map[string]interface{}{
		"version":                p.Version,
		"tol_D":                  p.TolD,
		"tol_coh":                p.TolCoh,
		"alpha":                  p.Alpha,
		"beta":                   p.Beta,
		"gamma":                  p.Gamma,
		"base":                   p.Base,
		"D0":                     p.D0,
		"sticky_D_hat_max":       p.StickyDHatMax,
		"sticky_coherence_min":   p.StickyCoherenceMin,
		"non_sticky_D_hat_min":   p.NonStickyDHatMin,
	}

	jsonBytes, err := json.Marshal(canonical)
	if err != nil {
		return "", fmt.Errorf("failed to marshal policy for hashing: %w", err)
	}

	hash := sha256.Sum256(jsonBytes)
	return hex.EncodeToString(hash[:]), nil
}

// Registry manages versioned policies (Phase 3 WP4)
type Registry struct {
	policies map[string]*Policy // version -> policy
	active   string             // active policy version
}

// NewRegistry creates a new policy registry
func NewRegistry() *Registry {
	return &Registry{
		policies: make(map[string]*Policy),
	}
}

// Register adds a policy to the registry after validation
func (r *Registry) Register(p *Policy) error {
	if err := p.Validate(); err != nil {
		return fmt.Errorf("policy validation failed: %w", err)
	}

	if _, exists := r.policies[p.Version]; exists {
		return fmt.Errorf("policy version %s already exists", p.Version)
	}

	r.policies[p.Version] = p
	return nil
}

// Promote activates a policy version (canary → production)
func (r *Registry) Promote(version string) error {
	p, exists := r.policies[version]
	if !exists {
		return fmt.Errorf("policy version %s not found", version)
	}

	if !p.Active {
		return fmt.Errorf("policy version %s is not active", version)
	}

	r.active = version
	return nil
}

// GetActive returns the currently active policy
func (r *Registry) GetActive() (*Policy, error) {
	if r.active == "" {
		return nil, fmt.Errorf("no active policy")
	}

	p, exists := r.policies[r.active]
	if !exists {
		return nil, fmt.Errorf("active policy %s not found", r.active)
	}

	return p, nil
}

// Get retrieves a policy by version
func (r *Registry) Get(version string) (*Policy, error) {
	p, exists := r.policies[version]
	if !exists {
		return nil, fmt.Errorf("policy version %s not found", version)
	}
	return p, nil
}

// ListVersions returns all policy versions
func (r *Registry) ListVersions() []string {
	versions := make([]string, 0, len(r.policies))
	for v := range r.policies {
		versions = append(versions, v)
	}
	return versions
}

// DefaultPolicy returns the default policy from CLAUDE.md (Phase 1/2 compatibility)
func DefaultPolicy() *Policy {
	return &Policy{
		Version:              "1.0.0",
		Name:                 "default",
		Description:          "Default verification policy from CLAUDE.md",
		CreatedAt:            time.Now(),
		Active:               true,
		TolD:                 0.15,
		TolCoh:               0.05,
		Alpha:                0.30,
		Beta:                 0.50,
		Gamma:                0.20,
		Base:                 0.10,
		D0:                   2.2,
		StickyDHatMax:        1.5,
		StickyCoherenceMin:   0.70,
		NonStickyDHatMin:     2.6,
		Flags:                make(map[string]bool),
	}
}
