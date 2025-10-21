-- Phase 10 WP3: Atomic dedup table with first-write-wins guarantee
--
-- This migration creates the pcs_dedup table for Postgres-based deduplication.
-- The PRIMARY KEY constraint on pcs_id ensures atomicity when combined with
-- ON CONFLICT DO NOTHING in the application code.

CREATE TABLE IF NOT EXISTS pcs_dedup (
    pcs_id VARCHAR(255) PRIMARY KEY,
    result JSONB NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for efficient cleanup of expired entries
CREATE INDEX IF NOT EXISTS idx_pcs_dedup_expires ON pcs_dedup(expires_at);

-- Comment for documentation
COMMENT ON TABLE pcs_dedup IS 'Deduplication store for PCS submissions with TTL-based expiration';
COMMENT ON COLUMN pcs_dedup.pcs_id IS 'Unique PCS identifier (sha256 of merkle_root|epoch|shard_id)';
COMMENT ON COLUMN pcs_dedup.result IS 'Serialized VerifyResult (JSON)';
COMMENT ON COLUMN pcs_dedup.expires_at IS 'TTL expiration timestamp';
COMMENT ON COLUMN pcs_dedup.created_at IS 'First-write timestamp for audit';
