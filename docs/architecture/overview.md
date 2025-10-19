# Architecture Overview

## System Vision

The Fractal LBA + Kakeya FT Stack is designed to provide **verifiable computation summaries** for distributed event streams with **guaranteed delivery** and **cryptographic authenticity**.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Event Sources                             │
│  (IoT Sensors, Financial Tx, Network Traffic, Blockchain)       │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Python Agent                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Signals     │→ │   Merkle     │→ │   Signing    │         │
│  │  (D̂,coh★,r)  │  │   Tree       │  │  (HMAC/Ed25519)│        │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│           │                                   │                  │
│           ▼                                   ▼                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │           Outbox WAL (fsync)                     │          │
│  └──────────────────────────────────────────────────┘          │
└────────────────┬────────────────────────────────────────────────┘
                 │ HTTP POST /v1/pcs/submit
                 │ (exponential backoff + jitter)
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Go Backend                                  │
│  ┌────────────────────────────────────────────────┐            │
│  │  Inbox WAL (fsync) → Parse → Sig Verify       │            │
│  └────────────────────────────────────────────────┘            │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────────────────────────────────┐            │
│  │  Dedup Check (Memory/Redis/Postgres)           │            │
│  │  - First-write wins                             │            │
│  │  - TTL = 14 days                                │            │
│  └────────────────────────────────────────────────┘            │
│           │                                                      │
│           ▼ (if new)                                            │
│  ┌────────────────────────────────────────────────┐            │
│  │  Verification Engine                            │            │
│  │  - Recompute D̂ (Theil-Sen)                     │            │
│  │  - Validate bounds (coh★, r)                    │            │
│  │  - Check regime classification                  │            │
│  │  - Compute budget                               │            │
│  └────────────────────────────────────────────────┘            │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────────────────────────────────┐            │
│  │  Store Result + Update Metrics                 │            │
│  │  → 200 OK (accepted) or 202 (escalated)        │            │
│  └────────────────────────────────────────────────┘            │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Observability Stack                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Prometheus  │→ │   Grafana    │  │  Audit Log   │         │
│  │  (metrics)   │  │  (dashboard) │  │  (Postgres)  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Python Agent

**Purpose**: Compute signals from raw event data and generate signed PCS.

**Key Modules**:
- **Signals** (`signals.py`): Computes D̂, coh★, r from event streams
- **Merkle Tree** (`merkle.py`): Builds data integrity proofs
- **Outbox WAL** (`outbox.py`): Ensures at-least-once delivery
- **Client** (`client.py`): HTTP submission with retry logic
- **Agent** (`agent.py`): Orchestrates the pipeline

**Fault Tolerance**:
- WAL with fsync guarantees durability
- Exponential backoff with jitter for retries
- Dead letter queue (DLQ) for exhausted retries

### 2. Go Backend

**Purpose**: Verify PCS submissions with strict idempotency and fault tolerance.

**Key Modules**:
- **API Handler** (`api/`): HTTP endpoint `/v1/pcs/submit`
- **Verification Engine** (`verify/`): Recomputes D̂ and validates signals
- **Deduplication** (`dedup/`): Prevents duplicate processing
- **WAL** (`wal/`): Inbox write-ahead log
- **Signing** (`signing/`): HMAC-SHA256 and Ed25519 verification
- **Metrics** (`metrics/`): Prometheus counters

**Processing Pipeline**:
1. Append request to Inbox WAL (fsync)
2. Parse JSON
3. Check dedup cache (return cached result if duplicate)
4. Verify signature
5. Run verification engine
6. Store result in dedup cache
7. Update metrics
8. Return HTTP response

### 3. Storage Backends

**Deduplication**:
- **Memory**: In-process map with optional file snapshot
- **Redis**: Distributed cache with TTL support
- **Postgres**: Relational database with indexes

**WAL**:
- **Agent Outbox**: Local file with fsync, compaction job removes acked entries
- **Backend Inbox**: Daily rotation, retention policy

### 4. Observability

**Prometheus Metrics**:
- `flk_ingest_total`: Total submissions
- `flk_accepted`: Verified successfully
- `flk_escalated`: Uncertain/failed checks
- `flk_dedup_hits`: Duplicates served from cache
- `flk_signature_errors`: Signature verification failures
- `flk_wal_errors`: WAL write errors

**Grafana Dashboard**:
- Real-time ingest rate
- Accept vs escalate ratio
- Dedup hit ratio
- Error rate tracking
- SLO monitoring (escalation rate ≤2%)

## Design Principles

### 1. Idempotency First

Every operation is designed to be safely retryable:
- PCS ID is deterministic: `sha256(merkle_root|epoch|shard_id)`
- Dedup store implements "first-write wins"
- WAL enables replay without side effects

### 2. Fault Tolerance by Default

System survives crashes at any point:
- Write-ahead logs with fsync on both agent and backend
- At-least-once delivery semantics
- Graceful degradation with `ft.degraded` flag

### 3. Zero Trust Verification

Backend trusts nothing from agents:
- Recomputes D̂ from scales and N_j
- Validates all bounds (coh★ ∈ [0,1], r ∈ [0,1])
- Verifies cryptographic signatures
- Checks regime classification consistency

### 4. Production Ready

Built for real-world operations:
- Prometheus metrics exposed
- Health check endpoints
- Graceful shutdown with connection draining
- Resource limits and security contexts (K8s)
- Horizontal auto-scaling (HPA)

## Communication Protocols

### Agent → Backend

**Protocol**: HTTPS POST
**Endpoint**: `/v1/pcs/submit`
**Content-Type**: `application/json`

**Retry Strategy**:
- Base delay: 1s
- Max delay: 60s
- Exponential backoff: delay × 2^attempt
- Jitter: ±50%
- Max retries: 5

**Status Handling**:
- `200`: Success, mark outbox entry as acked
- `202`: Escalated, mark as acked (server accepted)
- `401`: Signature failure, do not retry
- `429`: Rate limited, respect `Retry-After` header
- `5xx`: Server error, retry with backoff

### Backend → Storage

**Deduplication**:
- Memory: Direct in-process access
- Redis: `SETNX` with TTL
- Postgres: `INSERT ... ON CONFLICT DO NOTHING`

**Metrics**:
- Push model: Backend increments counters
- Pull model: Prometheus scrapes `/metrics` every 15s

## Security Model

### Transport Security

- **TLS**: Required for production deployments
- **mTLS**: Optional for internal backend-to-backend
- **Certificate Management**: cert-manager (K8s) or Caddy (Docker)

### Authentication

- **Agent → Backend**: Cryptographic signatures (HMAC or Ed25519)
- **User → Metrics**: Basic Auth (optional)

### Data Integrity

- **Merkle Trees**: Prove data chunks haven't been tampered
- **Signature Verification**: Ensures PCS comes from trusted agent
- **Recomputation**: Backend independently verifies D̂

## Scalability

### Horizontal Scaling

**Backend**:
- Stateless design (dedup in external store)
- Kubernetes HPA targets 70% CPU utilization
- Pod Disruption Budget ensures availability during rolling updates

**Redis**:
- Single instance for dev
- Redis Cluster or Sentinel for production

**Postgres**:
- Read replicas for dedup lookups (optional)
- Partitioning by time for audit log

### Vertical Scaling

**Backend Resource Limits**:
- Requests: 100m CPU, 128Mi RAM
- Limits: 500m CPU, 512Mi RAM

**Agent Resource Considerations**:
- Depends on event rate and computation complexity
- Typical: 200m CPU, 256Mi RAM per agent

## Performance Characteristics

### Latency

- **p50 Verify**: ~50ms (in-memory dedup)
- **p95 Verify**: ~200ms (in-memory dedup)
- **p99 Verify**: ~500ms (Redis dedup)

### Throughput

- **Single Replica**: ~500 req/s (in-memory dedup, no verification)
- **Single Replica**: ~200 req/s (full verification pipeline)
- **Scaled (10 replicas)**: ~2000 req/s

### Storage

- **Dedup Entry**: ~1KB per PCS
- **WAL Entry**: ~2KB per PCS
- **14-day retention**: ~2GB per 1M PCS

## Next Steps

- [System Architecture](system-architecture.md) - Detailed component design
- [Data Flow](data-flow.md) - Request lifecycle walkthrough
- [Fault Tolerance](fault-tolerance.md) - Deep dive on reliability patterns
- [Signal Computation](signal-computation.md) - Mathematical foundations
