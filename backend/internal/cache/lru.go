package cache

import (
	"sync"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
)

// LRUWithTTL provides a thread-safe LRU cache with TTL expiration.
//
// Phase 10 WP6: Replaces unbounded maps in feature stores and ensemble caches
// to prevent memory growth under high cardinality workloads.
//
// Key features:
//   - Size-bounded (evicts least recently used when full)
//   - TTL expiration (entries expire after configured duration)
//   - Thread-safe (safe for concurrent access)
//   - Metrics (hit/miss counters for observability)
type LRUWithTTL[K comparable, V any] struct {
	cache   *lru.Cache[K, *ttlEntry[V]]
	ttl     time.Duration
	mu      sync.RWMutex
	hits    uint64
	misses  uint64
	evicted uint64
}

type ttlEntry[V any] struct {
	value     V
	expiresAt time.Time
}

// NewLRUWithTTL creates a new LRU cache with TTL.
//
// Args:
//   - size: Maximum number of entries (LRU eviction when full)
//   - ttl: Time-to-live for entries (0 means no expiration)
//
// Returns:
//   - *LRUWithTTL or error if size is invalid
//
// Example:
//
//	cache, err := NewLRUWithTTL[string, int](1000, 5*time.Minute)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer cache.Close()
//
//	cache.Set("key1", 42)
//	if val, ok := cache.Get("key1"); ok {
//	    fmt.Println("Found:", val)
//	}
func NewLRUWithTTL[K comparable, V any](size int, ttl time.Duration) (*LRUWithTTL[K, V], error) {
	cache, err := lru.New[K, *ttlEntry[V]](size)
	if err != nil {
		return nil, err
	}

	return &LRUWithTTL[K, V]{
		cache: cache,
		ttl:   ttl,
	}, nil
}

// Get retrieves a value from the cache.
//
// Returns:
//   - value and true if found and not expired
//   - zero value and false if not found or expired
func (c *LRUWithTTL[K, V]) Get(key K) (V, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, ok := c.cache.Get(key)
	if !ok {
		c.misses++
		var zero V
		return zero, false
	}

	// Check TTL expiration
	if c.ttl > 0 && time.Now().After(entry.expiresAt) {
		c.misses++
		var zero V
		return zero, false
	}

	c.hits++
	return entry.value, true
}

// Set stores a value in the cache with TTL.
//
// If the cache is full, the least recently used entry is evicted.
func (c *LRUWithTTL[K, V]) Set(key K, value V) {
	c.mu.Lock()
	defer c.mu.Unlock()

	expiresAt := time.Time{} // no expiration
	if c.ttl > 0 {
		expiresAt = time.Now().Add(c.ttl)
	}

	evicted := c.cache.Add(key, &ttlEntry[V]{
		value:     value,
		expiresAt: expiresAt,
	})

	if evicted {
		c.evicted++
	}
}

// Delete removes a key from the cache.
func (c *LRUWithTTL[K, V]) Delete(key K) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache.Remove(key)
}

// Len returns the number of entries in the cache.
func (c *LRUWithTTL[K, V]) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.cache.Len()
}

// Clear removes all entries from the cache.
func (c *LRUWithTTL[K, V]) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache.Purge()
}

// Stats returns cache statistics for observability.
type Stats struct {
	Hits    uint64  `json:"hits"`
	Misses  uint64  `json:"misses"`
	Evicted uint64  `json:"evicted"`
	Size    int     `json:"size"`
	HitRate float64 `json:"hit_rate"`
}

// Stats returns current cache statistics.
func (c *LRUWithTTL[K, V]) Stats() Stats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := c.hits + c.misses
	hitRate := 0.0
	if total > 0 {
		hitRate = float64(c.hits) / float64(total)
	}

	return Stats{
		Hits:    c.hits,
		Misses:  c.misses,
		Evicted: c.evicted,
		Size:    c.cache.Len(),
		HitRate: hitRate,
	}
}

// ResetStats resets hit/miss/evicted counters to zero.
func (c *LRUWithTTL[K, V]) ResetStats() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.hits = 0
	c.misses = 0
	c.evicted = 0
}

// Close releases cache resources (currently a no-op, for future cleanup hooks).
func (c *LRUWithTTL[K, V]) Close() error {
	c.Clear()
	return nil
}

// CleanupExpired removes all expired entries from the cache.
//
// This should be called periodically by a background goroutine if TTL is enabled.
//
// Returns:
//   - Number of entries removed
func (c *LRUWithTTL[K, V]) CleanupExpired() int {
	if c.ttl == 0 {
		return 0 // no expiration
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	removed := 0

	// Iterate over all keys and remove expired entries
	// Note: This is O(n), so should be run infrequently
	keys := c.cache.Keys()
	for _, key := range keys {
		if entry, ok := c.cache.Peek(key); ok {
			if now.After(entry.expiresAt) {
				c.cache.Remove(key)
				removed++
			}
		}
	}

	return removed
}
