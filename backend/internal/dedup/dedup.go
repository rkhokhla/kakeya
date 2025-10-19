package dedup

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/fractal-lba/kakeya/internal/api"
)

// Store provides idempotent deduplication for PCS submissions
type Store interface {
	// Get retrieves a stored result by pcs_id. Returns nil if not found.
	Get(ctx context.Context, pcsID string) (*api.VerifyResult, error)

	// Set stores a verify result with TTL. First write wins.
	Set(ctx context.Context, pcsID string, result *api.VerifyResult, ttl time.Duration) error

	// Close releases resources
	Close() error
}

// MemoryStore is an in-memory dedup store with optional file snapshot
type MemoryStore struct {
	mu       sync.RWMutex
	store    map[string]*entry
	snapshot string // optional file path for persistence
}

type entry struct {
	Result    *api.VerifyResult
	ExpiresAt time.Time
}

// NewMemoryStore creates an in-memory dedup store
func NewMemoryStore(snapshotPath string) *MemoryStore {
	ms := &MemoryStore{
		store:    make(map[string]*entry),
		snapshot: snapshotPath,
	}

	// Load from snapshot if exists
	if snapshotPath != "" {
		ms.loadSnapshot()
	}

	return ms
}

func (m *MemoryStore) Get(ctx context.Context, pcsID string) (*api.VerifyResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	e, ok := m.store[pcsID]
	if !ok {
		return nil, nil
	}

	if time.Now().After(e.ExpiresAt) {
		return nil, nil // expired
	}

	return e.Result, nil
}

func (m *MemoryStore) Set(ctx context.Context, pcsID string, result *api.VerifyResult, ttl time.Duration) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// First write wins
	if e, exists := m.store[pcsID]; exists {
		if time.Now().Before(e.ExpiresAt) {
			return nil // already exists and not expired
		}
	}

	m.store[pcsID] = &entry{
		Result:    result,
		ExpiresAt: time.Now().Add(ttl),
	}

	// Persist snapshot if configured
	if m.snapshot != "" {
		go m.saveSnapshot() // async to avoid blocking
	}

	return nil
}

func (m *MemoryStore) Close() error {
	if m.snapshot != "" {
		return m.saveSnapshot()
	}
	return nil
}

func (m *MemoryStore) loadSnapshot() error {
	data, err := os.ReadFile(m.snapshot)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // no snapshot yet
		}
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	var snapshot map[string]*entry
	if err := json.Unmarshal(data, &snapshot); err != nil {
		return fmt.Errorf("failed to unmarshal snapshot: %w", err)
	}

	// Only load non-expired entries
	now := time.Now()
	for k, v := range snapshot {
		if now.Before(v.ExpiresAt) {
			m.store[k] = v
		}
	}

	return nil
}

func (m *MemoryStore) saveSnapshot() error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Only persist non-expired entries
	now := time.Now()
	toSave := make(map[string]*entry)
	for k, v := range m.store {
		if now.Before(v.ExpiresAt) {
			toSave[k] = v
		}
	}

	data, err := json.MarshalIndent(toSave, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(m.snapshot, data, 0600)
}

// RedisStore wraps Redis for deduplication (placeholder - requires redis client)
type RedisStore struct {
	// TODO: implement with go-redis
	addr string
}

func NewRedisStore(addr string) (*RedisStore, error) {
	// Placeholder: would initialize redis.Client here
	return &RedisStore{addr: addr}, nil
}

func (r *RedisStore) Get(ctx context.Context, pcsID string) (*api.VerifyResult, error) {
	// TODO: GET key, unmarshal JSON
	return nil, fmt.Errorf("redis store not yet implemented")
}

func (r *RedisStore) Set(ctx context.Context, pcsID string, result *api.VerifyResult, ttl time.Duration) error {
	// TODO: SETNX with TTL
	return fmt.Errorf("redis store not yet implemented")
}

func (r *RedisStore) Close() error {
	return nil
}

// PostgresStore uses Postgres for deduplication (placeholder)
type PostgresStore struct {
	connStr string
}

func NewPostgresStore(connStr string) (*PostgresStore, error) {
	// Placeholder: would initialize pgx pool here
	return &PostgresStore{connStr: connStr}, nil
}

func (p *PostgresStore) Get(ctx context.Context, pcsID string) (*api.VerifyResult, error) {
	// TODO: SELECT from dedup table
	return nil, fmt.Errorf("postgres store not yet implemented")
}

func (p *PostgresStore) Set(ctx context.Context, pcsID string, result *api.VerifyResult, ttl time.Duration) error {
	// TODO: INSERT ON CONFLICT DO NOTHING
	return fmt.Errorf("postgres store not yet implemented")
}

func (p *PostgresStore) Close() error {
	return nil
}
