package tiering

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Demoter handles background TTL-based demotion of entries across tiers (Phase 5 WP2)
// Moves entries from hot→warm→cold based on TTL expiry
type Demoter struct {
	mu              sync.RWMutex
	tieredStore     *TieredStore
	demoteInterval  time.Duration // How often to check for demotions
	batchSize       int           // Max entries to process per cycle
	metrics         *DemoterMetrics
	stopCh          chan struct{}
	wg              sync.WaitGroup
}

// DemoterMetrics tracks demotion operations
type DemoterMetrics struct {
	mu                     sync.RWMutex
	CyclesPerformed        int64
	HotToWarmDemotions     int64
	WarmToColdDemotions    int64
	ColdEvictions          int64
	DemotionErrors         int64
	LastCycleTimestamp     time.Time
	AvgCycleDurationMs     float64
}

// DemoterConfig holds configuration
type DemoterConfig struct {
	TieredStore    *TieredStore
	DemoteInterval time.Duration // Default: 5 minutes
	BatchSize      int           // Default: 1000 entries per cycle
}

// NewDemoter creates a new background demotion worker
func NewDemoter(config DemoterConfig) (*Demoter, error) {
	if config.TieredStore == nil {
		return nil, fmt.Errorf("TieredStore is required")
	}

	if config.DemoteInterval == 0 {
		config.DemoteInterval = 5 * time.Minute // Default: check every 5 minutes
	}
	if config.BatchSize == 0 {
		config.BatchSize = 1000 // Default: process 1000 entries per cycle
	}

	demoter := &Demoter{
		tieredStore:    config.TieredStore,
		demoteInterval: config.DemoteInterval,
		batchSize:      config.BatchSize,
		metrics:        &DemoterMetrics{},
		stopCh:         make(chan struct{}),
	}

	return demoter, nil
}

// Start begins the demotion loop (runs in background)
func (d *Demoter) Start(ctx context.Context) {
	d.wg.Add(1)
	go d.demoteLoop(ctx)
	fmt.Printf("Tiering Demoter: started (interval %v, batch size %d)\n",
		d.demoteInterval, d.batchSize)
}

// Stop gracefully stops the demoter
func (d *Demoter) Stop() {
	close(d.stopCh)
	d.wg.Wait()
	fmt.Printf("Tiering Demoter: stopped\n")
}

// demoteLoop continuously checks for expired entries and demotes them
func (d *Demoter) demoteLoop(ctx context.Context) {
	defer d.wg.Done()

	ticker := time.NewTicker(d.demoteInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-d.stopCh:
			return
		case <-ticker.C:
			if err := d.demoteCycle(ctx); err != nil {
				fmt.Printf("Tiering Demoter: error in demotion cycle: %v\n", err)
			}
		}
	}
}

// demoteCycle performs a single demotion cycle across all tiers
func (d *Demoter) demoteCycle(ctx context.Context) error {
	start := time.Now()

	d.mu.Lock()
	defer d.mu.Unlock()

	// Phase 1: Demote hot→warm (entries past hot TTL)
	hotDemoted, err := d.demoteHotToWarm(ctx)
	if err != nil {
		d.metrics.mu.Lock()
		d.metrics.DemotionErrors++
		d.metrics.mu.Unlock()
		return fmt.Errorf("hot→warm demotion failed: %w", err)
	}

	// Phase 2: Demote warm→cold (entries past warm TTL)
	warmDemoted, err := d.demoteWarmToCold(ctx)
	if err != nil {
		d.metrics.mu.Lock()
		d.metrics.DemotionErrors++
		d.metrics.mu.Unlock()
		return fmt.Errorf("warm→cold demotion failed: %w", err)
	}

	// Phase 3: Evict cold entries (past cold TTL, if lifecycle policy doesn't handle it)
	// Note: In production, S3/GCS lifecycle policies handle this automatically
	// This is a backup/optional path for explicit cold eviction
	coldEvicted, err := d.evictColdExpired(ctx)
	if err != nil {
		fmt.Printf("Tiering Demoter: warning: cold eviction failed: %v\n", err)
		// Non-fatal: lifecycle policies should handle this
	}

	// Update metrics
	duration := time.Since(start).Milliseconds()

	d.metrics.mu.Lock()
	d.metrics.CyclesPerformed++
	d.metrics.HotToWarmDemotions += int64(hotDemoted)
	d.metrics.WarmToColdDemotions += int64(warmDemoted)
	d.metrics.ColdEvictions += int64(coldEvicted)
	d.metrics.LastCycleTimestamp = time.Now()

	// Update average cycle duration (exponential moving average)
	if d.metrics.AvgCycleDurationMs == 0 {
		d.metrics.AvgCycleDurationMs = float64(duration)
	} else {
		d.metrics.AvgCycleDurationMs = 0.9*d.metrics.AvgCycleDurationMs + 0.1*float64(duration)
	}
	d.metrics.mu.Unlock()

	fmt.Printf("Tiering Demoter: cycle complete (hot→warm: %d, warm→cold: %d, cold evicted: %d, duration: %dms)\n",
		hotDemoted, warmDemoted, coldEvicted, duration)

	return nil
}

// demoteHotToWarm demotes expired entries from hot to warm tier
func (d *Demoter) demoteHotToWarm(ctx context.Context) (int, error) {
	if d.tieredStore.hot == nil || d.tieredStore.warm == nil {
		return 0, nil // No demotion if tiers not configured
	}

	// Get expired keys from hot tier
	// Note: This requires the hot tier (Redis) to expose a method to list keys by TTL
	// In production, you'd use Redis SCAN + TTL checks, or maintain a separate expiry index
	// For now, we'll use a simplified approach via the TieredStore

	expiredKeys, err := d.getExpiredKeys(ctx, TierHot, d.tieredStore.config.HotTTL)
	if err != nil {
		return 0, err
	}

	demoted := 0
	for _, key := range expiredKeys {
		// Call TieredStore.Demote to move hot→warm
		if err := d.tieredStore.Demote(ctx, key, TierHot, TierWarm); err != nil {
			fmt.Printf("Tiering Demoter: warning: failed to demote %s (hot→warm): %v\n", key, err)
			continue
		}
		demoted++

		// Batch size limit
		if demoted >= d.batchSize {
			break
		}
	}

	return demoted, nil
}

// demoteWarmToCold demotes expired entries from warm to cold tier
func (d *Demoter) demoteWarmToCold(ctx context.Context) (int, error) {
	if d.tieredStore.warm == nil || d.tieredStore.cold == nil {
		return 0, nil // No demotion if tiers not configured
	}

	expiredKeys, err := d.getExpiredKeys(ctx, TierWarm, d.tieredStore.config.WarmTTL)
	if err != nil {
		return 0, err
	}

	demoted := 0
	for _, key := range expiredKeys {
		// Call TieredStore.Demote to move warm→cold
		if err := d.tieredStore.Demote(ctx, key, TierWarm, TierCold); err != nil {
			fmt.Printf("Tiering Demoter: warning: failed to demote %s (warm→cold): %v\n", key, err)
			continue
		}
		demoted++

		// Batch size limit
		if demoted >= d.batchSize {
			break
		}
	}

	return demoted, nil
}

// evictColdExpired evicts expired entries from cold tier (optional, lifecycle handles this)
func (d *Demoter) evictColdExpired(ctx context.Context) (int, error) {
	if d.tieredStore.cold == nil {
		return 0, nil
	}

	// In production, S3/GCS lifecycle policies handle this automatically
	// This is a backup path for explicit deletion if needed

	expiredKeys, err := d.getExpiredKeys(ctx, TierCold, d.tieredStore.config.ColdTTL)
	if err != nil {
		return 0, err
	}

	evicted := 0
	for _, key := range expiredKeys {
		if err := d.tieredStore.cold.Delete(ctx, key); err != nil {
			fmt.Printf("Tiering Demoter: warning: failed to evict %s (cold): %v\n", key, err)
			continue
		}
		evicted++

		// Batch size limit
		if evicted >= d.batchSize {
			break
		}
	}

	return evicted, nil
}

// getExpiredKeys returns keys from a tier that are past their TTL
// Note: This is a simplified implementation. In production, you'd need to:
// - For Redis: use SCAN + TTL commands, or maintain a sorted set by expiry time
// - For Postgres: query entries with created_at + TTL < now()
// - For S3/GCS: rely on lifecycle policies, or maintain metadata index
func (d *Demoter) getExpiredKeys(ctx context.Context, tier Tier, ttl time.Duration) ([]string, error) {
	// This is a placeholder implementation
	// In Phase 5 full implementation, you'd need to:
	// 1. Add a ListExpiredKeys() method to StorageDriver interface
	// 2. Implement it for each driver (Redis, Postgres, S3/GCS)
	// 3. For Redis: SCAN + TTL checks
	// 4. For Postgres: SELECT pcs_id FROM dedup WHERE created_at + interval '$ttl' < now()
	// 5. For S3/GCS: Query metadata or use lifecycle events

	// For now, return empty list (lifecycle policies will handle cold tier)
	// Hot/warm tier demotion would be implemented in actual driver implementations
	return []string{}, nil
}

// GetMetrics returns current demoter metrics
func (d *Demoter) GetMetrics() DemoterMetrics {
	d.metrics.mu.RLock()
	defer d.metrics.mu.RUnlock()
	return *d.metrics
}

// ForceSingleCycle performs an immediate demotion cycle (for testing/ops)
func (d *Demoter) ForceSingleCycle(ctx context.Context) error {
	return d.demoteCycle(ctx)
}

// --- Per-Tenant TTL Policies ---

// TenantTTLPolicy defines TTL overrides for a specific tenant
type TenantTTLPolicy struct {
	TenantID string
	HotTTL   time.Duration
	WarmTTL  time.Duration
	ColdTTL  time.Duration
}

// TenantTTLManager manages per-tenant TTL policies (Phase 5 WP2)
type TenantTTLManager struct {
	mu       sync.RWMutex
	policies map[string]*TenantTTLPolicy // tenant_id → policy
	defaults *TenantTTLPolicy              // Default policy for tenants without custom policy
}

// NewTenantTTLManager creates a new per-tenant TTL manager
func NewTenantTTLManager(defaults *TenantTTLPolicy) *TenantTTLManager {
	return &TenantTTLManager{
		policies: make(map[string]*TenantTTLPolicy),
		defaults: defaults,
	}
}

// SetPolicy sets a custom TTL policy for a tenant
func (m *TenantTTLManager) SetPolicy(tenantID string, policy *TenantTTLPolicy) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.policies[tenantID] = policy
}

// GetPolicy retrieves the TTL policy for a tenant (returns defaults if not set)
func (m *TenantTTLManager) GetPolicy(tenantID string) *TenantTTLPolicy {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if policy, ok := m.policies[tenantID]; ok {
		return policy
	}
	return m.defaults
}

// RemovePolicy removes a custom TTL policy for a tenant (reverts to defaults)
func (m *TenantTTLManager) RemovePolicy(tenantID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.policies, tenantID)
}
