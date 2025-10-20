package sharding

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"sort"
	"sync"
)

// Shard represents a single dedup store shard (Phase 4 WP2)
type Shard struct {
	Name   string // Shard identifier (e.g., "shard-0")
	Addr   string // Connection address (e.g., "redis://shard-0:6379")
	Weight int    // Weight for virtual node distribution
	Healthy bool   // Health status
}

// Ring implements consistent hashing for shard selection (Phase 4 WP2)
type Ring struct {
	mu      sync.RWMutex
	shards  []*Shard
	vnodes  int                // Virtual nodes per physical shard
	ring    []uint32           // Sorted hash ring
	hashMap map[uint32]*Shard  // Hash → shard mapping
}

// NewRing creates a new consistent hash ring
func NewRing(vnodes int) *Ring {
	return &Ring{
		shards:  make([]*Shard, 0),
		vnodes:  vnodes,
		ring:    make([]uint32, 0),
		hashMap: make(map[uint32]*Shard),
	}
}

// AddShard adds a shard to the ring
func (r *Ring) AddShard(shard *Shard) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Add physical shard
	r.shards = append(r.shards, shard)

	// Add virtual nodes to ring
	for i := 0; i < r.vnodes; i++ {
		vkey := fmt.Sprintf("%s#%d", shard.Name, i)
		hash := r.hash([]byte(vkey))
		r.ring = append(r.ring, hash)
		r.hashMap[hash] = shard
	}

	// Keep ring sorted
	sort.Slice(r.ring, func(i, j int) bool {
		return r.ring[i] < r.ring[j]
	})

	return nil
}

// RemoveShard removes a shard from the ring
func (r *Ring) RemoveShard(shardName string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Remove from shards list
	for i, shard := range r.shards {
		if shard.Name == shardName {
			r.shards = append(r.shards[:i], r.shards[i+1:]...)
			break
		}
	}

	// Remove virtual nodes from ring
	newRing := make([]uint32, 0)
	for _, h := range r.ring {
		if shard := r.hashMap[h]; shard.Name != shardName {
			newRing = append(newRing, h)
		} else {
			delete(r.hashMap, h)
		}
	}
	r.ring = newRing

	return nil
}

// Pick selects a shard for the given key using consistent hashing
func (r *Ring) Pick(key []byte) (*Shard, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.ring) == 0 {
		return nil, fmt.Errorf("no shards available")
	}

	// Hash the key
	h := r.hash(key)

	// Binary search for closest hash on ring
	idx := sort.Search(len(r.ring), func(i int) bool {
		return r.ring[i] >= h
	})

	// Wrap around if necessary
	if idx == len(r.ring) {
		idx = 0
	}

	shard := r.hashMap[r.ring[idx]]

	// Check if shard is healthy
	if !shard.Healthy {
		// Try next shard (simple fallback - production would have more sophisticated logic)
		for i := 1; i < len(r.ring); i++ {
			nextIdx := (idx + i) % len(r.ring)
			nextShard := r.hashMap[r.ring[nextIdx]]
			if nextShard.Healthy {
				return nextShard, nil
			}
		}
		return nil, fmt.Errorf("no healthy shards available")
	}

	return shard, nil
}

// PickN selects N shards for replication
func (r *Ring) PickN(key []byte, n int) ([]*Shard, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.shards) < n {
		return nil, fmt.Errorf("not enough shards: have %d, need %d", len(r.shards), n)
	}

	// Hash the key
	h := r.hash(key)

	// Binary search for starting position
	idx := sort.Search(len(r.ring), func(i int) bool {
		return r.ring[i] >= h
	})
	if idx == len(r.ring) {
		idx = 0
	}

	// Collect unique shards
	seen := make(map[string]bool)
	shards := make([]*Shard, 0, n)

	for i := 0; i < len(r.ring) && len(shards) < n; i++ {
		pos := (idx + i) % len(r.ring)
		shard := r.hashMap[r.ring[pos]]

		if !seen[shard.Name] && shard.Healthy {
			seen[shard.Name] = true
			shards = append(shards, shard)
		}
	}

	if len(shards) < n {
		return nil, fmt.Errorf("not enough healthy shards: found %d, need %d", len(shards), n)
	}

	return shards, nil
}

// hash computes a 32-bit hash for consistent hashing
func (r *Ring) hash(key []byte) uint32 {
	h := sha256.Sum256(key)
	return binary.BigEndian.Uint32(h[:4])
}

// GetShards returns all registered shards
func (r *Ring) GetShards() []*Shard {
	r.mu.RLock()
	defer r.mu.RUnlock()

	shards := make([]*Shard, len(r.shards))
	copy(shards, r.shards)
	return shards
}

// MarkHealthy marks a shard as healthy/unhealthy
func (r *Ring) MarkHealthy(shardName string, healthy bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for _, shard := range r.shards {
		if shard.Name == shardName {
			shard.Healthy = healthy
			return nil
		}
	}

	return fmt.Errorf("shard %s not found", shardName)
}

// Stats returns statistics about the ring
func (r *Ring) Stats() map[string]interface{} {
	r.mu.RLock()
	defer r.mu.RUnlock()

	healthy := 0
	for _, shard := range r.shards {
		if shard.Healthy {
			healthy++
		}
	}

	return map[string]interface{}{
		"total_shards":   len(r.shards),
		"healthy_shards": healthy,
		"virtual_nodes":  len(r.ring),
		"vnodes_per_shard": r.vnodes,
	}
}
