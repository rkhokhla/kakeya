-- Fractal LBA + Kakeya FT Stack - Database Schema

-- Deduplication table (optional, if using Postgres backend)
CREATE TABLE IF NOT EXISTS dedup_store (
    pcs_id VARCHAR(64) PRIMARY KEY,
    result JSONB NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for expiry cleanup
CREATE INDEX IF NOT EXISTS idx_dedup_expires ON dedup_store(expires_at);

-- Optional: Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    pcs_id VARCHAR(64) NOT NULL,
    shard_id VARCHAR(255),
    epoch INTEGER,
    accepted BOOLEAN,
    escalated BOOLEAN,
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_received ON audit_log(received_at);
CREATE INDEX IF NOT EXISTS idx_audit_pcs_id ON audit_log(pcs_id);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO flk;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO flk;
