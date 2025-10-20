package signing

import (
	"fmt"
	"sync"

	"github.com/fractal-lba/kakeya/internal/api"
)

// MultiTenantVerifier wraps multiple verifiers for different tenants
type MultiTenantVerifier struct {
	mu        sync.RWMutex
	verifiers map[string]Verifier // tenantID -> verifier
	fallback  Verifier            // Default verifier for backward compatibility
}

// NewMultiTenantVerifier creates a new multi-tenant verifier
func NewMultiTenantVerifier(fallback Verifier) *MultiTenantVerifier {
	return &MultiTenantVerifier{
		verifiers: make(map[string]Verifier),
		fallback:  fallback,
	}
}

// RegisterTenant adds a verifier for a specific tenant
func (m *MultiTenantVerifier) RegisterTenant(tenantID string, alg string, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var verifier Verifier
	var err error

	switch alg {
	case "hmac":
		if key == "" {
			return fmt.Errorf("HMAC key required for tenant %s", tenantID)
		}
		verifier = NewHMACVerifier(key)
	case "ed25519":
		if key == "" {
			return fmt.Errorf("Ed25519 public key required for tenant %s", tenantID)
		}
		verifier, err = NewEd25519Verifier(key)
		if err != nil {
			return fmt.Errorf("failed to create Ed25519 verifier for tenant %s: %w", tenantID, err)
		}
	case "none":
		verifier = &NoOpVerifier{}
	default:
		return fmt.Errorf("unknown signature algorithm for tenant %s: %s", tenantID, alg)
	}

	m.verifiers[tenantID] = verifier
	return nil
}

// VerifyForTenant verifies a PCS using the tenant's specific verifier
func (m *MultiTenantVerifier) VerifyForTenant(tenantID string, pcs *api.PCS) error {
	m.mu.RLock()
	verifier, ok := m.verifiers[tenantID]
	m.mu.RUnlock()

	if ok {
		return verifier.Verify(pcs)
	}

	// Fall back to default verifier for backward compatibility
	if m.fallback != nil {
		return m.fallback.Verify(pcs)
	}

	return fmt.Errorf("no verifier configured for tenant %s and no fallback available", tenantID)
}

// Verify implements the Verifier interface for compatibility
// Uses fallback verifier when no tenant context is available
func (m *MultiTenantVerifier) Verify(pcs *api.PCS) error {
	if m.fallback != nil {
		return m.fallback.Verify(pcs)
	}
	return fmt.Errorf("no fallback verifier configured")
}

// RemoveTenant removes a tenant's verifier
func (m *MultiTenantVerifier) RemoveTenant(tenantID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.verifiers, tenantID)
}

// ListTenants returns all registered tenant IDs
func (m *MultiTenantVerifier) ListTenants() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	tenants := make([]string, 0, len(m.verifiers))
	for id := range m.verifiers {
		tenants = append(tenants, id)
	}
	return tenants
}
