package tenant

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"golang.org/x/time/rate"
)

var (
	ErrTenantNotFound   = errors.New("tenant not found")
	ErrQuotaExceeded    = errors.New("tenant quota exceeded")
	ErrInvalidTenantID  = errors.New("invalid tenant ID")
)

// Tenant represents a multi-tenant isolation unit
type Tenant struct {
	ID          string
	DisplayName string
	SigningKey  string // HMAC key or Ed25519 public key (base64)
	SigningAlg  string // "hmac" or "ed25519"

	// Quotas
	TokenRate   int     // requests/second
	BurstRate   int     // burst capacity
	DailyQuota  int64   // max requests per day (0 = unlimited)

	// Verification params can be customized per tenant
	CustomParams bool
	TolD         float64
	TolCoh       float64

	// Metadata
	CreatedAt   time.Time
	Active      bool
	Metadata    map[string]string
}

// Manager handles tenant lifecycle and quota enforcement
type Manager struct {
	mu       sync.RWMutex
	tenants  map[string]*Tenant
	limiters map[string]*rate.Limiter
	usage    map[string]*UsageCounter
}

// UsageCounter tracks daily request counts for quota enforcement
type UsageCounter struct {
	mu         sync.Mutex
	count      int64
	resetAt    time.Time
}

// NewManager creates a new tenant manager
func NewManager() *Manager {
	return &Manager{
		tenants:  make(map[string]*Tenant),
		limiters: make(map[string]*rate.Limiter),
		usage:    make(map[string]*UsageCounter),
	}
}

// RegisterTenant adds a new tenant to the system
func (m *Manager) RegisterTenant(t *Tenant) error {
	if t.ID == "" {
		return ErrInvalidTenantID
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Create rate limiter
	limiter := rate.NewLimiter(rate.Limit(t.TokenRate), t.BurstRate)

	// Create usage counter
	usage := &UsageCounter{
		count:   0,
		resetAt: time.Now().Add(24 * time.Hour),
	}

	m.tenants[t.ID] = t
	m.limiters[t.ID] = limiter
	m.usage[t.ID] = usage

	return nil
}

// GetTenant retrieves a tenant by ID
func (m *Manager) GetTenant(tenantID string) (*Tenant, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	tenant, ok := m.tenants[tenantID]
	if !ok {
		return nil, ErrTenantNotFound
	}

	if !tenant.Active {
		return nil, fmt.Errorf("tenant %s is not active", tenantID)
	}

	return tenant, nil
}

// Allow checks if a request is allowed under the tenant's rate limit and quota
func (m *Manager) Allow(ctx context.Context, tenantID string) error {
	m.mu.RLock()
	tenant, ok := m.tenants[tenantID]
	limiter, limiterOK := m.limiters[tenantID]
	usage, usageOK := m.usage[tenantID]
	m.mu.RUnlock()

	if !ok || !limiterOK || !usageOK {
		return ErrTenantNotFound
	}

	if !tenant.Active {
		return fmt.Errorf("tenant %s is not active", tenantID)
	}

	// Check rate limit (requests/second)
	if !limiter.Allow() {
		return ErrQuotaExceeded
	}

	// Check daily quota if enabled
	if tenant.DailyQuota > 0 {
		usage.mu.Lock()
		defer usage.mu.Unlock()

		// Reset counter if new day
		if time.Now().After(usage.resetAt) {
			usage.count = 0
			usage.resetAt = time.Now().Add(24 * time.Hour)
		}

		if usage.count >= tenant.DailyQuota {
			return ErrQuotaExceeded
		}

		usage.count++
	}

	return nil
}

// GetUsage returns current usage for a tenant
func (m *Manager) GetUsage(tenantID string) (int64, error) {
	m.mu.RLock()
	usage, ok := m.usage[tenantID]
	m.mu.RUnlock()

	if !ok {
		return 0, ErrTenantNotFound
	}

	usage.mu.Lock()
	defer usage.mu.Unlock()

	// Reset if new day
	if time.Now().After(usage.resetAt) {
		usage.count = 0
		usage.resetAt = time.Now().Add(24 * time.Hour)
	}

	return usage.count, nil
}

// ListTenants returns all registered tenants
func (m *Manager) ListTenants() []*Tenant {
	m.mu.RLock()
	defer m.mu.RUnlock()

	tenants := make([]*Tenant, 0, len(m.tenants))
	for _, t := range m.tenants {
		tenants = append(tenants, t)
	}

	return tenants
}

// UpdateTenant updates an existing tenant's configuration
func (m *Manager) UpdateTenant(tenantID string, update func(*Tenant)) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	tenant, ok := m.tenants[tenantID]
	if !ok {
		return ErrTenantNotFound
	}

	update(tenant)

	// Update rate limiter if token rate changed
	if limiter, ok := m.limiters[tenantID]; ok {
		limiter.SetLimit(rate.Limit(tenant.TokenRate))
		limiter.SetBurst(tenant.BurstRate)
	}

	return nil
}

// RemoveTenant removes a tenant from the system
func (m *Manager) RemoveTenant(tenantID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.tenants, tenantID)
	delete(m.limiters, tenantID)
	delete(m.usage, tenantID)

	return nil
}

// DefaultTenant returns a default tenant for backward compatibility
func DefaultTenant() *Tenant {
	return &Tenant{
		ID:          "default",
		DisplayName: "Default Tenant",
		SigningKey:  "", // Will use server-wide key
		SigningAlg:  "",
		TokenRate:   100,
		BurstRate:   200,
		DailyQuota:  0, // Unlimited
		CustomParams: false,
		CreatedAt:   time.Now(),
		Active:      true,
		Metadata:    make(map[string]string),
	}
}
