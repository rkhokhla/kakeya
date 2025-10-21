package sharding

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// CrossShardQuery provides read-only operations across all shards (Phase 5 WP4)
// Used for diagnostics, monitoring, and operational queries
type CrossShardQuery struct {
	mu    sync.RWMutex
	ring  *Ring
	stats *CrossShardStats
}

// CrossShardStats tracks cross-shard query statistics
type CrossShardStats struct {
	mu                  sync.RWMutex
	TotalQueries        int64
	TotalKeysScanned    int64
	AvgQueryLatencyMs   float64
	ShardHealthChecks   int64
	LastQueryTimestamp  time.Time
}

// ShardDistribution shows key distribution across shards
type ShardDistribution struct {
	ShardID         string  `json:"shard_id"`
	KeyCount        int64   `json:"key_count"`
	EstimatedBytes  int64   `json:"estimated_bytes"`
	LoadPercentage  float64 `json:"load_percentage"`
	Healthy         bool    `json:"healthy"`
	Lag             float64 `json:"lag_ms"` // Replication lag (if applicable)
}

// ShardHealth represents health status of a shard
type ShardHealth struct {
	ShardID       string    `json:"shard_id"`
	Healthy       bool      `json:"healthy"`
	Endpoint      string    `json:"endpoint"`
	Latency       float64   `json:"latency_ms"`
	LastCheck     time.Time `json:"last_check"`
	ErrorMessage  string    `json:"error_message,omitempty"`
}

// KeySample represents a sample of keys from a shard
type KeySample struct {
	ShardID    string   `json:"shard_id"`
	Keys       []string `json:"keys"`
	SampleSize int      `json:"sample_size"`
}

// NewCrossShardQuery creates a new cross-shard query interface
func NewCrossShardQuery(ring *Ring) *CrossShardQuery {
	return &CrossShardQuery{
		ring:  ring,
		stats: &CrossShardStats{},
	}
}

// GetDistribution returns key distribution across all shards
func (q *CrossShardQuery) GetDistribution(ctx context.Context) ([]ShardDistribution, error) {
	start := time.Now()
	defer func() {
		latency := time.Since(start).Milliseconds()
		q.updateQueryStats(float64(latency))
	}()

	q.mu.RLock()
	shards := q.ring.shards
	q.mu.RUnlock()

	// Query each shard in parallel
	type result struct {
		dist ShardDistribution
		err  error
	}

	results := make(chan result, len(shards))
	var wg sync.WaitGroup

	for _, shard := range shards {
		wg.Add(1)
		go func(s *Shard) {
			defer wg.Done()

			dist, err := q.getShardDistribution(ctx, s)
			results <- result{dist: dist, err: err}
		}(shard)
	}

	wg.Wait()
	close(results)

	// Collect results
	distributions := []ShardDistribution{}
	totalKeys := int64(0)

	for r := range results {
		if r.err != nil {
			fmt.Printf("CrossShardQuery: warning: failed to get distribution for shard: %v\n", r.err)
			continue
		}
		distributions = append(distributions, r.dist)
		totalKeys += r.dist.KeyCount
	}

	// Compute load percentages
	for i := range distributions {
		if totalKeys > 0 {
			distributions[i].LoadPercentage = float64(distributions[i].KeyCount) / float64(totalKeys) * 100.0
		}
	}

	return distributions, nil
}

// GetHealth returns health status of all shards
func (q *CrossShardQuery) GetHealth(ctx context.Context) ([]ShardHealth, error) {
	start := time.Now()
	defer func() {
		latency := time.Since(start).Milliseconds()
		q.updateQueryStats(float64(latency))
	}()

	q.mu.RLock()
	shards := q.ring.shards
	q.mu.RUnlock()

	// Query each shard in parallel
	type result struct {
		health ShardHealth
		err    error
	}

	results := make(chan result, len(shards))
	var wg sync.WaitGroup

	for _, shard := range shards {
		wg.Add(1)
		go func(s *Shard) {
			defer wg.Done()

			health, err := q.checkShardHealth(ctx, s)
			results <- result{health: health, err: err}
		}(shard)
	}

	wg.Wait()
	close(results)

	// Collect results
	healthStatuses := []ShardHealth{}
	for r := range results {
		if r.err != nil {
			// Include failed shard with error
			healthStatuses = append(healthStatuses, ShardHealth{
				Healthy:      false,
				ErrorMessage: r.err.Error(),
				LastCheck:    time.Now(),
			})
			continue
		}
		healthStatuses = append(healthStatuses, r.health)
	}

	q.stats.mu.Lock()
	q.stats.ShardHealthChecks++
	q.stats.mu.Unlock()

	return healthStatuses, nil
}

// SampleKeys returns a random sample of keys from each shard
func (q *CrossShardQuery) SampleKeys(ctx context.Context, sampleSize int) ([]KeySample, error) {
	start := time.Now()
	defer func() {
		latency := time.Since(start).Milliseconds()
		q.updateQueryStats(float64(latency))
	}()

	q.mu.RLock()
	shards := q.ring.shards
	q.mu.RUnlock()

	// Query each shard in parallel
	type result struct {
		sample KeySample
		err    error
	}

	results := make(chan result, len(shards))
	var wg sync.WaitGroup

	for _, shard := range shards {
		wg.Add(1)
		go func(s *Shard) {
			defer wg.Done()

			sample, err := q.sampleShardKeys(ctx, s, sampleSize)
			results <- result{sample: sample, err: err}
		}(shard)
	}

	wg.Wait()
	close(results)

	// Collect results
	samples := []KeySample{}
	for r := range results {
		if r.err != nil {
			fmt.Printf("CrossShardQuery: warning: failed to sample shard: %v\n", r.err)
			continue
		}
		samples = append(samples, r.sample)

		q.stats.mu.Lock()
		q.stats.TotalKeysScanned += int64(r.sample.SampleSize)
		q.stats.mu.Unlock()
	}

	return samples, nil
}

// FindKey locates which shard(s) a key belongs to
// Returns both current shard and predicted shard (useful during migration)
func (q *CrossShardQuery) FindKey(ctx context.Context, key string) (*ShardLocation, error) {
	start := time.Now()
	defer func() {
		latency := time.Since(start).Milliseconds()
		q.updateQueryStats(float64(latency))
	}()

	// Determine shard by consistent hash
	shard, err := q.ring.Pick([]byte(key))
	if err != nil {
		return nil, fmt.Errorf("failed to pick shard: %w", err)
	}

	location := &ShardLocation{
		Key:          key,
		CurrentShard: shard.Name,
		Endpoint:     shard.Addr,
		Healthy:      shard.Healthy,
	}

	return location, nil
}

// ShardLocation represents the location of a key
type ShardLocation struct {
	Key          string `json:"key"`
	CurrentShard string `json:"current_shard"`
	Endpoint     string `json:"endpoint"`
	Healthy      bool   `json:"healthy"`
}

// getShardDistribution queries a single shard for its distribution
func (q *CrossShardQuery) getShardDistribution(ctx context.Context, shard *Shard) (ShardDistribution, error) {
	// In production, this would query the actual dedup store (Redis, Postgres, etc.)
	// For now, return mock data

	// Placeholder: In production, you'd call:
	// - Redis: DBSIZE command for key count
	// - Postgres: SELECT COUNT(*) FROM dedup WHERE shard_id = ?
	// - Estimate bytes: MEMORY USAGE (Redis) or pg_relation_size (Postgres)

	dist := ShardDistribution{
		ShardID:        shard.Name,
		KeyCount:       100000, // Placeholder
		EstimatedBytes: 50 * 1024 * 1024, // 50MB
		Healthy:        shard.Healthy,
		Lag:            0,
	}

	return dist, nil
}

// checkShardHealth performs a health check on a single shard
func (q *CrossShardQuery) checkShardHealth(ctx context.Context, shard *Shard) (ShardHealth, error) {
	start := time.Now()

	// In production, this would:
	// - Ping the shard (Redis: PING, Postgres: SELECT 1)
	// - Measure latency
	// - Check replication lag (if applicable)

	// Placeholder
	latency := time.Since(start).Milliseconds()

	health := ShardHealth{
		ShardID:   shard.Name,
		Healthy:   shard.Healthy,
		Endpoint:  shard.Addr,
		Latency:   float64(latency),
		LastCheck: time.Now(),
	}

	return health, nil
}

// sampleShardKeys samples random keys from a single shard
func (q *CrossShardQuery) sampleShardKeys(ctx context.Context, shard *Shard, sampleSize int) (KeySample, error) {
	// In production, this would:
	// - Redis: RANDOMKEY command (multiple calls)
	// - Postgres: SELECT pcs_id FROM dedup ORDER BY RANDOM() LIMIT N

	// Placeholder
	keys := make([]string, 0, sampleSize)
	for i := 0; i < sampleSize; i++ {
		keys = append(keys, fmt.Sprintf("key-%s-%d", shard.Name, i))
	}

	sample := KeySample{
		ShardID:    shard.Name,
		Keys:       keys,
		SampleSize: len(keys),
	}

	return sample, nil
}

// updateQueryStats updates query statistics
func (q *CrossShardQuery) updateQueryStats(latencyMs float64) {
	q.stats.mu.Lock()
	defer q.stats.mu.Unlock()

	q.stats.TotalQueries++
	q.stats.LastQueryTimestamp = time.Now()

	// Exponential moving average for latency
	if q.stats.AvgQueryLatencyMs == 0 {
		q.stats.AvgQueryLatencyMs = latencyMs
	} else {
		q.stats.AvgQueryLatencyMs = 0.9*q.stats.AvgQueryLatencyMs + 0.1*latencyMs
	}
}

// GetStats returns cross-shard query statistics
func (q *CrossShardQuery) GetStats() CrossShardStats {
	q.stats.mu.RLock()
	defer q.stats.mu.RUnlock()
	return *q.stats
}

// --- HTTP Handler for Cross-Shard API ---

// CrossShardAPIHandler provides HTTP endpoints for cross-shard operations
type CrossShardAPIHandler struct {
	query *CrossShardQuery
}

// NewCrossShardAPIHandler creates a new API handler
func NewCrossShardAPIHandler(query *CrossShardQuery) *CrossShardAPIHandler {
	return &CrossShardAPIHandler{
		query: query,
	}
}

// HandleDistribution handles GET /api/shards/distribution
func (h *CrossShardAPIHandler) HandleDistribution(ctx context.Context) (interface{}, error) {
	return h.query.GetDistribution(ctx)
}

// HandleHealth handles GET /api/shards/health
func (h *CrossShardAPIHandler) HandleHealth(ctx context.Context) (interface{}, error) {
	return h.query.GetHealth(ctx)
}

// HandleSample handles GET /api/shards/sample?size=N
func (h *CrossShardAPIHandler) HandleSample(ctx context.Context, sampleSize int) (interface{}, error) {
	return h.query.SampleKeys(ctx, sampleSize)
}

// HandleFindKey handles GET /api/shards/find?key=xxx
func (h *CrossShardAPIHandler) HandleFindKey(ctx context.Context, key string) (interface{}, error) {
	return h.query.FindKey(ctx, key)
}

// HandleStats handles GET /api/shards/stats
func (h *CrossShardAPIHandler) HandleStats(ctx context.Context) (interface{}, error) {
	return h.query.GetStats(), nil
}
