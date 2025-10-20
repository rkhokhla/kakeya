package tiering

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/fractal-lba/kakeya/internal/api"
)

// Tier represents storage tiers (Phase 4 WP3)
type Tier string

const (
	TierHot  Tier = "hot"  // Redis: <1h, fast access
	TierWarm Tier = "warm" // Postgres: <7d, medium latency
	TierCold Tier = "cold" // Object storage: >7d, slow but cheap
)

// TierPolicy defines TTL and promotion rules per tier
type TierPolicy struct {
	HotTTL    time.Duration // How long to keep in hot tier
	WarmTTL   time.Duration // How long to keep in warm tier
	ColdTTL   time.Duration // How long to keep in cold tier (0 = forever)
	PromoteOn string        // Condition to promote (e.g., "access_count > 3")
}

// TierConfig holds per-tenant tier policies
type TierConfig struct {
	Default TierPolicy
	Tenants map[string]TierPolicy
}

// StorageDriver abstracts tier-specific storage operations
type StorageDriver interface {
	Get(ctx context.Context, key string) (*api.VerifyResult, error)
	Set(ctx context.Context, key string, value *api.VerifyResult, ttl time.Duration) error
	Delete(ctx context.Context, key string) error
	Exists(ctx context.Context, key string) (bool, error)
}

// TieredStore manages hot/warm/cold storage tiers (Phase 4 WP3)
type TieredStore struct {
	mu      sync.RWMutex
	hot     StorageDriver       // Redis
	warm    StorageDriver       // Postgres
	cold    StorageDriver       // Object storage (S3/GCS)
	config  *TierConfig
	metrics *TierMetrics
}

// TierMetrics tracks tier operations
type TierMetrics struct {
	HotHits     int64
	WarmHits    int64
	ColdHits    int64
	Promotions  int64
	Demotions   int64
	Evictions   int64
}

// NewTieredStore creates a new tiered storage manager
func NewTieredStore(hot, warm, cold StorageDriver, config *TierConfig) *TieredStore {
	return &TieredStore{
		hot:     hot,
		warm:    warm,
		cold:    cold,
		config:  config,
		metrics: &TierMetrics{},
	}
}

// Get retrieves a value from the appropriate tier
func (ts *TieredStore) Get(ctx context.Context, key string) (*api.VerifyResult, error) {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	// Try hot tier first
	if ts.hot != nil {
		value, err := ts.hot.Get(ctx, key)
		if err == nil && value != nil {
			ts.metrics.HotHits++
			return value, nil
		}
	}

	// Try warm tier
	if ts.warm != nil {
		value, err := ts.warm.Get(ctx, key)
		if err == nil && value != nil {
			ts.metrics.WarmHits++
			// Promote to hot tier (lazy promotion)
			go ts.promote(ctx, key, value, TierWarm, TierHot)
			return value, nil
		}
	}

	// Try cold tier
	if ts.cold != nil {
		value, err := ts.cold.Get(ctx, key)
		if err == nil && value != nil {
			ts.metrics.ColdHits++
			// Promote to warm tier (lazy promotion)
			go ts.promote(ctx, key, value, TierCold, TierWarm)
			return value, nil
		}
	}

	return nil, fmt.Errorf("key not found in any tier: %s", key)
}

// Set writes a value to the appropriate tier based on policy
func (ts *TieredStore) Set(ctx context.Context, key string, value *api.VerifyResult, tenantID string) error {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	// Get policy for tenant
	policy := ts.getPolicyForTenant(tenantID)

	// Write to hot tier
	if ts.hot != nil {
		if err := ts.hot.Set(ctx, key, value, policy.HotTTL); err != nil {
			return fmt.Errorf("failed to write to hot tier: %w", err)
		}
	}

	// Optionally replicate to warm tier immediately (depends on policy)
	// For now, we rely on demotion from hot → warm on TTL expiry

	return nil
}

// promote moves a value up one tier
func (ts *TieredStore) promote(ctx context.Context, key string, value *api.VerifyResult, from, to Tier) error {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	var targetDriver StorageDriver
	var ttl time.Duration

	switch to {
	case TierHot:
		targetDriver = ts.hot
		ttl = ts.config.Default.HotTTL
	case TierWarm:
		targetDriver = ts.warm
		ttl = ts.config.Default.WarmTTL
	default:
		return fmt.Errorf("cannot promote to tier: %s", to)
	}

	if targetDriver != nil {
		if err := targetDriver.Set(ctx, key, value, ttl); err != nil {
			return fmt.Errorf("failed to promote from %s to %s: %w", from, to, err)
		}
		ts.metrics.Promotions++
	}

	return nil
}

// Demote moves a value down one tier (called by background worker)
func (ts *TieredStore) Demote(ctx context.Context, key string, value *api.VerifyResult, from, to Tier) error {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	var sourceDriver, targetDriver StorageDriver
	var ttl time.Duration

	// Get source driver
	switch from {
	case TierHot:
		sourceDriver = ts.hot
	case TierWarm:
		sourceDriver = ts.warm
	}

	// Get target driver
	switch to {
	case TierWarm:
		targetDriver = ts.warm
		ttl = ts.config.Default.WarmTTL
	case TierCold:
		targetDriver = ts.cold
		ttl = ts.config.Default.ColdTTL
	}

	// Write to target tier
	if targetDriver != nil {
		if err := targetDriver.Set(ctx, key, value, ttl); err != nil {
			return fmt.Errorf("failed to demote from %s to %s: %w", from, to, err)
		}
		ts.metrics.Demotions++
	}

	// Delete from source tier
	if sourceDriver != nil {
		if err := sourceDriver.Delete(ctx, key); err != nil {
			// Log but don't fail (source will expire naturally)
			fmt.Printf("Warning: failed to delete from %s after demotion: %v\n", from, err)
		}
	}

	return nil
}

// Evict removes a value from all tiers
func (ts *TieredStore) Evict(ctx context.Context, key string) error {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	// Delete from all tiers
	drivers := []StorageDriver{ts.hot, ts.warm, ts.cold}
	for _, driver := range drivers {
		if driver != nil {
			_ = driver.Delete(ctx, key) // Best effort
		}
	}

	ts.metrics.Evictions++
	return nil
}

// getPolicyForTenant returns the appropriate tier policy
func (ts *TieredStore) getPolicyForTenant(tenantID string) TierPolicy {
	if policy, ok := ts.config.Tenants[tenantID]; ok {
		return policy
	}
	return ts.config.Default
}

// GetMetrics returns current tier metrics
func (ts *TieredStore) GetMetrics() *TierMetrics {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	return &TierMetrics{
		HotHits:    ts.metrics.HotHits,
		WarmHits:   ts.metrics.WarmHits,
		ColdHits:   ts.metrics.ColdHits,
		Promotions: ts.metrics.Promotions,
		Demotions:  ts.metrics.Demotions,
		Evictions:  ts.metrics.Evictions,
	}
}

// DefaultTierConfig returns default tier configuration
func DefaultTierConfig() *TierConfig {
	return &TierConfig{
		Default: TierPolicy{
			HotTTL:  1 * time.Hour,      // 1 hour in Redis
			WarmTTL: 7 * 24 * time.Hour, // 7 days in Postgres
			ColdTTL: 0,                  // Forever in object storage
		},
		Tenants: make(map[string]TierPolicy),
	}
}
