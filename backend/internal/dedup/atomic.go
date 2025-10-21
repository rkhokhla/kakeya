package dedup

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/fractal-lba/kakeya/internal/api"
	"github.com/go-redis/redis/v8"
	"github.com/jackc/pgx/v5/pgxpool"
)

// AtomicRedisStore implements Store using Redis SETNX for atomic first-write-wins.
//
// WP3: Ensures no race conditions even under concurrent writes to the same pcs_id.
type AtomicRedisStore struct {
	client *redis.Client
}

// NewAtomicRedisStore creates a Redis-backed dedup store with atomic guarantees.
//
// Args:
//   - addr: Redis address (e.g., "localhost:6379")
//   - password: Redis password (empty string if none)
//   - db: Redis database number (0-15, typically 0)
//
// Returns:
//   - *AtomicRedisStore or error if connection fails
func NewAtomicRedisStore(addr, password string, db int) (*AtomicRedisStore, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis connection failed: %w", err)
	}

	return &AtomicRedisStore{client: client}, nil
}

func (r *AtomicRedisStore) Get(ctx context.Context, pcsID string) (*api.VerifyResult, error) {
	key := fmt.Sprintf("pcs:%s", pcsID)

	data, err := r.client.Get(ctx, key).Result()
	if err == redis.Nil {
		return nil, nil // not found
	}
	if err != nil {
		return nil, fmt.Errorf("redis GET failed: %w", err)
	}

	var result api.VerifyResult
	if err := json.Unmarshal([]byte(data), &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal result: %w", err)
	}

	return &result, nil
}

func (r *AtomicRedisStore) Set(ctx context.Context, pcsID string, result *api.VerifyResult, ttl time.Duration) error {
	key := fmt.Sprintf("pcs:%s", pcsID)

	data, err := json.Marshal(result)
	if err != nil {
		return fmt.Errorf("failed to marshal result: %w", err)
	}

	// SETNX with TTL: atomic first-write-wins
	// Returns true if key was set (first write), false if already exists
	wasSet, err := r.client.SetNX(ctx, key, data, ttl).Result()
	if err != nil {
		return fmt.Errorf("redis SETNX failed: %w", err)
	}

	// If not set, key already existed (concurrent write won)
	// This is not an error - just means we lost the race
	_ = wasSet

	return nil
}

func (r *AtomicRedisStore) Close() error {
	return r.client.Close()
}

// AtomicPostgresStore implements Store using Postgres ON CONFLICT for atomic first-write-wins.
//
// WP3: Ensures no race conditions via unique constraint + ON CONFLICT DO NOTHING.
//
// Schema:
//
//	CREATE TABLE pcs_dedup (
//	  pcs_id VARCHAR(255) PRIMARY KEY,
//	  result JSONB NOT NULL,
//	  expires_at TIMESTAMP NOT NULL,
//	  created_at TIMESTAMP DEFAULT NOW()
//	);
//	CREATE INDEX idx_pcs_dedup_expires ON pcs_dedup(expires_at);
type AtomicPostgresStore struct {
	pool *pgxpool.Pool
}

// NewAtomicPostgresStore creates a Postgres-backed dedup store with atomic guarantees.
//
// Args:
//   - connStr: Postgres connection string (e.g., "postgres://user:pass@localhost:5432/dbname")
//
// Returns:
//   - *AtomicPostgresStore or error if connection fails
func NewAtomicPostgresStore(connStr string) (*AtomicPostgresStore, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	pool, err := pgxpool.New(ctx, connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to create postgres pool: %w", err)
	}

	// Test connection
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("postgres ping failed: %w", err)
	}

	return &AtomicPostgresStore{pool: pool}, nil
}

func (p *AtomicPostgresStore) Get(ctx context.Context, pcsID string) (*api.VerifyResult, error) {
	query := `
		SELECT result
		FROM pcs_dedup
		WHERE pcs_id = $1 AND expires_at > NOW()
	`

	var resultJSON []byte
	err := p.pool.QueryRow(ctx, query, pcsID).Scan(&resultJSON)
	if err != nil {
		if err.Error() == "no rows in result set" {
			return nil, nil // not found or expired
		}
		return nil, fmt.Errorf("postgres query failed: %w", err)
	}

	var result api.VerifyResult
	if err := json.Unmarshal(resultJSON, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal result: %w", err)
	}

	return &result, nil
}

func (p *AtomicPostgresStore) Set(ctx context.Context, pcsID string, result *api.VerifyResult, ttl time.Duration) error {
	resultJSON, err := json.Marshal(result)
	if err != nil {
		return fmt.Errorf("failed to marshal result: %w", err)
	}

	expiresAt := time.Now().Add(ttl)

	// ON CONFLICT DO NOTHING: atomic first-write-wins via primary key constraint
	query := `
		INSERT INTO pcs_dedup (pcs_id, result, expires_at)
		VALUES ($1, $2, $3)
		ON CONFLICT (pcs_id) DO NOTHING
	`

	_, err = p.pool.Exec(ctx, query, pcsID, resultJSON, expiresAt)
	if err != nil {
		return fmt.Errorf("postgres insert failed: %w", err)
	}

	// Even if insert was skipped (DO NOTHING), this is success
	// The first write won, which is the desired behavior
	return nil
}

func (p *AtomicPostgresStore) Close() error {
	p.pool.Close()
	return nil
}

// CleanupExpired removes expired entries from Postgres (for maintenance cron job).
//
// This should be run periodically to prevent table bloat.
//
// Returns:
//   - Number of deleted rows
func (p *AtomicPostgresStore) CleanupExpired(ctx context.Context) (int64, error) {
	query := `DELETE FROM pcs_dedup WHERE expires_at <= NOW()`

	result, err := p.pool.Exec(ctx, query)
	if err != nil {
		return 0, fmt.Errorf("cleanup failed: %w", err)
	}

	return result.RowsAffected(), nil
}
