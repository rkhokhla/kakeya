# Fractal LBA + Kakeya FT Stack

> **Turn distributed event streams into verifiable, compact proofs—with mathematical rigor, battle-tested fault tolerance, multi-tenant governance, and global-scale deployment.**

## Elevator Pitch

Imagine you're analyzing massive, distributed event streams from IoT sensors, financial transactions, or network traffic **across multiple regions and tenants**. You need to **prove** your computation happened correctly, **compress** terabytes into kilobytes, **never lose data**—even during crashes, network splits, or replay attacks—and **comply** with audit requirements while **scaling globally**.

**Fractal LBA + Kakeya FT Stack** solves this by:

1. **Computing cryptographic summaries (PCS)** that capture the "shape" of your data using fractal geometry (D̂), directional coherence (coh★), and compressibility (r)
2. **Verifying summaries server-side** with robust statistical methods (Theil-Sen regression) to catch manipulated or corrupted data
3. **Guaranteeing delivery** with write-ahead logs (WAL) on both agent and backend—your proofs survive crashes
4. **Preventing duplicates** with idempotent deduplication across memory, Redis, Postgres, or sharded stores
5. **Ensuring authenticity** with HMAC-SHA256 or Ed25519 signatures
6. **Multi-tenant isolation** with per-tenant keys, quotas, rate limits, and labeled metrics
7. **Immutable audit trails** (WORM logs) for compliance and tamper-evidence
8. **Adversarial defenses** with VRF verification, sanity checks, and anomaly scoring
9. **Global scale** with active-active multi-region deployment, tiered storage (hot/warm/cold), and sharded dedup
10. **Production SDKs** for Python, Go, and TypeScript with automatic signing and retry logic

All wrapped in production-ready **Docker Compose** and **Kubernetes Helm charts** with observability (Prometheus + Grafana), auto-scaling (HPA), security hardening (mTLS, NetworkPolicies), and comprehensive runbooks.

**Use cases:** Blockchain light clients, IoT data integrity, compliance audit trails, distributed system health monitoring, anti-fraud detection, multi-tenant SaaS platforms, LLMs hallucination mitigation.

---

# Investor Pitch

### One-liner

**FLK/PCS** is a **fault-tolerant verification layer for AI** that measurably **reduces LLM hallucinations** and controls compute spend by turning model outputs into **Proof-of-Computation Summaries (PCS)**, verifying them server-side, and dynamically gating trust and budget.

---

## The problem

LLMs are brilliant but **unreliable by default**. Hallucinations erode user trust, inflate support costs, and create compliance risk. Today’s mitigations (bigger models, more RAG, more prompts) mostly **add latency and cost** without strong guarantees.

---

## Our solution

We add a **verifiable control plane** around generation:

* **PCS (Proof-of-Computation Summary):** the agent computes compact signals from its own work (fractal slope **D̂**, directional coherence **coh★**, compressibility **r**), then **cryptographically signs** the summary.
* **Server-side verification:** a Go backend **recomputes and checks** those signals with tolerances, enforces **idempotency** (no double effects), and decides **trust level + budget** per request.
* **Fault tolerance by design:** dual **WAL** (write-ahead logs) on both agent and backend, **verify-before-dedup** ordering, retries with jitter, and clear escalation paths.
* **Observability & policy:** metrics, dashboards, and a small policy DSL to bound risk (thresholds, budgets, escalation).

**Result:** untrusted generations are **de-risked**—they get less budget, more retrieval or human review; trustworthy ones flow fast with lower cost.

---

## How it reduces hallucinations (concretely)

1. **Detect structure vs noise:**

    * **D̂** (Theil–Sen slope on multi-scale occupancy) flags diffuse vs structured evidence.
    * **coh★** measures how concentrated evidence is along consistent directions.
    * **r** (LZ compressibility) rewards internally consistent, non-random work.
2. **Gate actions by trust:** a **budget function** (bounded [0,1]) combines the signals; low-trust outputs are **throttled**, forced into **retrieval/verification** or **human-in-the-loop**.
3. **Cryptographic accountability:** agents **sign** the exact subset of fields; the server rejects drift and **never** applies effects for bad signatures.
4. **Auditable trail:** each step is **WAL-recorded** and reconstructable, giving compliance and post-mortem clarity.
5. **Production resilience:** failures land in **202-escalation** rather than silent errors, so **hallucinations don’t slip through** during incidents.

---

## What’s unique

* **Model-agnostic, plug-in layer:** works with any LLM, RAG, or tool-use flow.
* **Quantitative, verifiable signals:** not just heuristics—**recomputed** on the server with tolerances.
* **Security first:** HMAC/Ed25519, TLS/mTLS, metrics auth; **no raw payloads** in logs.
* **Designed for scale:** multi-tenant, sharded/tiered dedup, geo replication (active-active), async audits.

---

## Architecture snapshot

* **Agent (Python):** computes PCS, signs it, writes **Outbox WAL**, posts to backend.
* **Verifier (Go):** appends **Inbox WAL** → verify → **idempotent dedup** → budget/regime → metrics.
* **Signals:** `D̂`, `coh★`, `r` → **regime** (`sticky/mixed/non_sticky`) → **budget** gate.
* **Ops:** Prometheus + Grafana, runbooks, chaos & DR drills, Helm for prod, SOPS/age for secrets.
* **Optionals:** VRF-seeded sampling (adversarial robustness), DP metrics, policy DSL & canary rollouts.

---

## Where it fits (initial customers)

* **AI customer support / assistants:** reduce wrong answers & refunds with gating + human review on low-trust cases.
* **Healthcare/Finance/Legal copilots:** enforce **verification before action** to meet compliance.
* **Data extraction & agents:** throttle or enrich when signals show low structure, avoiding costly re-runs.

---

## Business model

* **Enterprise license** + support (self-hosted or VPC).
* **Usage tier** by verified requests / tenants / regions.
* **Premium**: multi-region DR, lineage & anchoring, policy DSL, DP, advanced canary.

---

## Traction & roadmap (high level)

* **Phase 1–2:** canonical signing, verifier ordering, unit/E2E tests, perf baselines, prod Helm, alerts.
* **Phase 3–4:** multi-tenant & multi-region design, sharded/tiered storage, SDKs (Py/Go/TS), chaos & DR suites.
* **Phase 5:** implement CRR shipper/reader, cold-tier drivers & demotion, async audit workers, geo divergence detection.
* **Phase 6:** Kubernetes Operator, selective/multi-way CRR, SIEM integration, compliance automation, predictive tiering, Rust/WASM SDKs, formal verification (TLA+/Coq), buyer dashboards with hallucination KPIs.

---

## KPIs we track

* **Hallucination containment rate:** % low-trust generations caught before action.
* **Cost per accepted task:** dollars per trusted completion vs baseline.
* **p95 verify latency:** stays <200 ms under nominal load.
* **Error-budget burn:** (escalations + non-2xx) / ingest; alert <2% over rolling window.
* **DR readiness:** RTO/RPO met in drills; replication lag SLO.
* **Adoption:** tenants enabled, % traffic under PCS policy, SDK parity across stacks.

---

## Why now

The market is **shifting from prototypes to production**. Boards demand safety, governance, and cost control. Our layer **reduces hallucinations without retraining** or model lock-in—and makes audits and compliance practical.

---

## Risks & how we mitigate

* **Signal drift across languages/runtimes** → canonical JSON & 9-dp rounding, **golden vectors** in CI.
* **Latency under load** → verifier kept **pure & idempotent**, heavy work moved to **async audits**.
* **Operational complexity** → Helm, runbooks, alerts, chaos drills, **verify-before-dedup** invariant.
* **Adversarial inputs** → VRF seeding (optional), monotonicity guards, anomaly scoring & rate limits.

---

## The ask

We’re raising to **productize the enterprise feature set** (multi-region DR, lineage/anchoring, policy registry, DP), expand SDKs, and fund early **design-partner deployments** in support, healthcare, and finance.
**Use of funds:** security certifications, scale testing, GTM with reference customers.

> **Bottom line:** FLK/PCS is a **practical, defensible path** to safer, cheaper AI—**cutting hallucinations** not by guessing, but by **verifying**.


## Overview

This system implements a globally distributed architecture where:

- **Python/Go/TS Agents** compute signals from event streams and generate signed PCS
- **Go Backend** verifies PCS with strict idempotency, multi-tenant isolation, and fault tolerance
- **WAL (Write-Ahead Logs)** ensure at-least-once delivery semantics with cross-region replication
- **Deduplication** provides idempotent processing with sharded, tiered storage (hot/warm/cold)
- **Signing** (HMAC/Ed25519) ensures authenticity with per-tenant keys
- **Audit Logs** (WORM) provide tamper-evident, immutable records for compliance
- **Multi-Region** active-active deployment with geo-routing and disaster recovery
- **Observability** via Prometheus metrics, Grafana dashboards, and SLO alerts

See [CLAUDE.md](./CLAUDE.md) for the complete technical specification and design invariants.

---

## Quick Start

### Prerequisites

- **Docker & Docker Compose** (for local deployment)
- **Go 1.22+** (for backend development)
- **Python 3.10+** (for agent development)
- **Kubernetes 1.25+** and **Helm 3.x** (for production deployment)

### Local Development with Docker Compose

```bash
# Clone and navigate to project
cd kakeya

# Set environment variables
export PCS_HMAC_KEY="your-secret-key"
export METRICS_PASS="your-metrics-password"
export POSTGRES_PASSWORD="your-db-password"

# Start all services
cd deployments/docker
docker-compose up -d

# View logs
docker-compose logs -f backend

# Access services
# - Backend API: http://localhost:8080
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### Production Deployment with Kubernetes (Multi-Tenant, Multi-Region)

```bash
# Install with Helm (eu-west region)
cd deployments/k8s/helm

helm install fractal-lba-eu-west ./fractal-lba \
  --set region.id=eu-west \
  --set replication.enabled=true \
  --set replication.remoteRegions[0]=us-east \
  --set signing.enabled=true \
  --set signing.alg=hmac \
  --set-string signing.hmacKey="your-secret-key" \
  --set multiTenant.enabled=true \
  --set metricsBasicAuth.enabled=true \
  --set-string metricsBasicAuth.password="your-metrics-password" \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=api.fractal-lba.example.com \
  --set sharding.enabled=true \
  --set sharding.shardCount=3 \
  --set tiering.enabled=true

# Install in second region (us-east)
helm install fractal-lba-us-east ./fractal-lba \
  --set region.id=us-east \
  --set replication.enabled=true \
  --set replication.remoteRegions[0]=eu-west \
  # ... (same settings as eu-west)

# Check status
kubectl get pods -n fractal-lba
kubectl logs -l app=backend
```

---

## Architecture

### Phase 1-6 Implementation Stack

```
┌──────────────────────────────────────────────────────────────────┐
│         Multi-Region Active-Active with WAL CRR (Phase 5 WP1)    │
│  ┌────────────────────────┐      ┌────────────────────────┐     │
│  │  Region: eu-west       │◄────►│  Region: us-east       │     │
│  │  - Backend (3 replicas)│ CRR  │  - Backend (3 replicas)│     │
│  │  - Sharded Dedup (3x)  │Ship/ │  - Sharded Dedup (3x)  │     │
│  │  - Tiered Storage      │Replay│  - Tiered Storage      │     │
│  │  - WORM Audit          │      │  - WORM Audit          │     │
│  │  - WAL Shipper         │----->│  - WAL Reader          │     │
│  │  - Divergence Detector │      │  - Batch Anchoring     │     │
│  └────────────────────────┘      └────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Multi-Tenant Backend (Phase 3)                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Tenant Router → Per-Tenant: Keys, Quotas, Metrics      │    │
│  │  Policy DSL → Verification params, regime thresholds    │    │
│  │  PII Gates → Detect/Block/Redact (email, phone, SSN)    │    │
│  │  VRF + Sanity Checks → Adversarial defenses             │    │
│  │  WORM Audit → Tamper-evident, immutable logs            │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│              Sharded + Tiered Dedup (Phase 4 WP2/WP3)           │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Consistent Hash Ring → Route pcs_id to shard        │       │
│  │  Shard-0, Shard-1, Shard-2 (Redis/Postgres)          │       │
│  │  Tiered Storage:                                      │       │
│  │    - Hot (Redis, 1h TTL, <5ms latency)               │       │
│  │    - Warm (Postgres, 7d TTL, <50ms latency)          │       │
│  │    - Cold (S3/GCS, 90d+ TTL, <500ms latency)         │       │
│  └──────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│            Verification + Signing (Phase 1/Phase 2)              │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Signature Verification (HMAC/Ed25519)               │       │
│  │  Recompute D̂ (Theil-Sen), validate bounds            │       │
│  │  WAL Inbox (write before parse)                      │       │
│  │  Idempotent Dedup (first-write wins, TTL 14d)       │       │
│  └──────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                 Agents: Python, Go, TypeScript (Phase 4 WP5)    │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Compute Signals (D̂, coh★, r)                        │       │
│  │  Canonical Signing (8-field subset, 9-decimal)       │       │
│  │  Outbox WAL (at-least-once delivery)                 │       │
│  │  Retry with Backoff + Jitter                         │       │
│  └──────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

### Components

#### Backend (Go)

Located in `backend/`:

**Phase 1/2 Core:**
- **API Handler** (`internal/api/`) - HTTP endpoint `/v1/pcs/submit`
- **Verification Engine** (`internal/verify/`) - Recomputes D̂, validates signals
- **Deduplication** (`internal/dedup/`) - Memory/Redis/Postgres backends
- **WAL** (`internal/wal/`) - Write-ahead logging with fsync
- **Signing** (`internal/signing/`) - HMAC-SHA256 and Ed25519 verification
- **Metrics** (`internal/metrics/`) - Prometheus counters

**Phase 3 Enhancements:**
- **Tenant Manager** (`internal/tenant/`) - Multi-tenant isolation, quotas, rate limits
- **WORM Audit** (`internal/audit/`) - Immutable, tamper-evident logs
- **Policy DSL** (`internal/policy/`) - Versioned policies, compile-time validation
- **PII Scanner** (`internal/privacy/`) - Regex-based detection/redaction
- **Security** (`internal/security/`) - VRF verification, sanity checks, anomaly scoring

**Phase 4 Scale:**
- **Sharding** (`internal/sharding/`) - Consistent hashing for dedup shards, cross-shard query API
- **Tiering** (`internal/tiering/`) - Hot/warm/cold storage management, background demotion

**Phase 5 Global Production:**
- **CRR** (`internal/crr/`) - WAL cross-region replication (shipper/reader), geo divergence detector
- **Cold Tier** (`internal/tiering/colddriver.go`) - S3/GCS drivers with compression and lifecycle policies
- **Async Audit** (`internal/audit/`) - Worker queue, batch anchoring, DLQ management
- **Migration CLI** (`cmd/dedup-migrate/`) - Zero-downtime shard migration tool (plan→copy→verify→cutover→cleanup)

**Phase 6 Autonomous Operations & Enterprise:**
- **Kubernetes Operator** (`operator/`) - CRDs for ShardMigration, CRRPolicy, TieringPolicy with health gates and auto-rollback
- **Advanced CRR** (`internal/crr/selective.go`, `reconcile.go`) - Selective and multi-way replication, auto-reconciliation with safety scoring
- **SIEM Integration** (`internal/audit/siem.go`) - Real-time streaming to Splunk/Datadog/Elastic/Sumo Logic
- **Compliance Automation** (`internal/audit/compliance.go`) - Automated SOC2/ISO27001/HIPAA/GDPR report generation
- **Anchoring Optimizer** (`internal/audit/anchoring_optimizer.go`) - Cost-aware strategy selection (blockchain vs timestamp)
- **Predictive Tiering** (`internal/tiering/predictor.go`) - ML-based access pattern prediction for pre-warming
- **Buyer Dashboards** (`observability/grafana/buyer_dashboard.json`, `internal/metrics/buyer_kpis.go`) - Hallucination containment KPIs, cost per trusted task
- **Formal Verification** (`formal/`) - TLA+ spec for CRR idempotency, Coq lemmas for canonical signing correctness

#### Agents

**Python Agent** (Located in `agent/src/`):
- **Signals** (`signals.py`) - Computes D̂, coh★, r
- **Merkle** (`merkle.py`) - Merkle tree for data integrity
- **Outbox WAL** (`outbox.py`) - Agent-side WAL with fsync
- **Client** (`client.py`) - HTTP client with exponential backoff + jitter
- **Agent** (`agent.py`) - Main orchestrator

**Go SDK** (Phase 4, Located in `sdk/go/`):
- Full PCS lifecycle: compute, sign, submit
- Canonical signing (8-field subset, 9-decimal rounding)
- Multi-tenant support (X-Tenant-Id header)
- Health checks and error handling

**TypeScript SDK** (Phase 4, Located in `sdk/ts/`):
- TypeScript interfaces for PCS, VerifyResult
- Automatic HMAC-SHA256 signing
- Retry logic with exponential backoff + jitter
- Custom error types (ValidationError, SignatureError, APIError)

**Rust SDK** (Phase 6 WP5, Located in `sdk/rust/`):
- Zero-copy canonical signing with `zerocopy` crate
- SIMD-optimized SHA-256 and HMAC-SHA256
- Async/await with Tokio for non-blocking operations
- 100,000+ signatures/sec throughput
- Sub-10μs signing latency

**WASM SDK** (Phase 6 WP5, Located in `sdk/wasm/`):
- Browser-native WebAssembly agent
- Runs directly in modern browsers (Chrome, Firefox, Safari, Edge)
- ~50KB gzipped binary
- Suitable for edge workers (Cloudflare, Vercel)
- JavaScript/TypeScript bindings via wasm-bindgen

### Core Signals

| Signal | Description | Calculation |
|--------|-------------|-------------|
| **D̂** | Fractal dimension | Theil-Sen median slope of log₂(scale) vs log₂(N_j) |
| **coh★** | Directional coherence | Max histogram concentration along sampled directions |
| **r** | Compressibility | zlib(data) / len(data) |

### Regime Classification

- **sticky**: `coh★ ≥ 0.70 and D̂ ≤ 1.5`
- **non_sticky**: `D̂ ≥ 2.6`
- **mixed**: Otherwise

---

## Configuration

### Environment Variables (Backend)

#### Core (Phase 1/2):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | HTTP server port |
| `DEDUP_BACKEND` | `memory` | Dedup store: `memory`, `redis`, `postgres` |
| `REDIS_ADDR` | `localhost:6379` | Redis address |
| `POSTGRES_CONN` | - | PostgreSQL connection string |
| `TOKEN_RATE` | `100` | Rate limit (requests/sec) |
| `PCS_SIGN_ALG` | `none` | Signature algorithm: `none`, `hmac`, `ed25519` |
| `PCS_HMAC_KEY` | - | HMAC secret key (required if `hmac`) |
| `PCS_ED25519_PUB_B64` | - | Ed25519 public key base64 (required if `ed25519`) |
| `METRICS_USER` | - | Metrics endpoint basic auth user |
| `METRICS_PASS` | - | Metrics endpoint password |
| `WAL_DIR` | `data/wal` | WAL directory path |

#### Multi-Tenant (Phase 3):

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTI_TENANT` | `false` | Enable multi-tenant mode |
| `TENANTS` | - | Comma-separated: `tenant1:hmac:key1,tenant2:hmac:key2` |
| `AUDIT_WORM_DIR` | `data/worm/audit` | WORM audit log directory |
| `POLICY_VERSION` | `1.0.0` | Active policy version (SemVer) |
| `PII_SCAN_MODE` | `detect` | PII scanner mode: `detect`, `block`, `redact` |

#### Sharding + Tiering (Phase 4):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEDUP_SHARDS` | - | Comma-separated shard addresses (e.g., `redis://shard-0:6379,redis://shard-1:6379`) |
| `DEDUP_VNODES` | `150` | Virtual nodes per shard (consistent hashing) |
| `TIER_HOT_TTL` | `3600` | Hot tier TTL (seconds, default: 1 hour) |
| `TIER_WARM_TTL` | `604800` | Warm tier TTL (seconds, default: 7 days) |
| `TIER_COLD_TTL` | `0` | Cold tier TTL (seconds, 0=forever) |
| `TIER_COLD_BUCKET` | - | S3/GCS bucket for cold tier (e.g., `s3://flk-cold-dedup`) |

#### Multi-Region (Phase 4):

| Variable | Default | Description |
|----------|---------|-------------|
| `REGION_ID` | `default` | Region identifier (e.g., `eu-west`, `us-east`) |
| `WAL_REPLICATION_ENABLED` | `false` | Enable cross-region WAL replication |
| `WAL_REMOTE_BUCKET` | - | Remote S3/GCS bucket for WAL CRR |

### Environment Variables (Agents)

| Variable | Description |
|----------|-------------|
| `ENDPOINT` | Backend submission URL |
| `TENANT_ID` | Tenant ID (Phase 3 multi-tenant) |
| `PCS_SIGN_ALG` | Signature algorithm: `none`, `hmac`, `ed25519` |
| `PCS_HMAC_KEY` | HMAC secret key |
| `PCS_ED25519_PRIV_B64` | Ed25519 private key base64 |

---

## API Reference

### POST /v1/pcs/submit

Submit a Proof-of-Computation Summary.

**Request Headers:**
- `Content-Type: application/json`
- `X-Tenant-Id: <tenant_id>` (optional, Phase 3 multi-tenant)

**Request Body:**
```json
{
  "pcs_id": "sha256(merkle_root|epoch|shard_id)",
  "schema": "fractal-lba-kakeya",
  "version": "0.1",
  "shard_id": "shard-001",
  "epoch": 1,
  "attempt": 1,
  "sent_at": "2025-01-01T00:00:00Z",
  "seed": 42,
  "scales": [2, 4, 8, 16, 32],
  "N_j": {"2": 3, "4": 5, "8": 9, "16": 17, "32": 31},
  "coh_star": 0.73,
  "v_star": [0.12, 0.98, -0.05],
  "D_hat": 1.41,
  "r": 0.87,
  "regime": "mixed",
  "budget": 0.42,
  "merkle_root": "abc123...",
  "sig": "base64-signature",
  "ft": {
    "outbox_seq": 123,
    "degraded": false,
    "fallbacks": [],
    "clock_skew_ms": 0
  }
}
```

**Responses:**

- `200 OK` - Accepted and verified
- `202 Accepted` - Escalated (uncertain verification)
- `400 Bad Request` - Malformed JSON, validation error, or sanity check failed
- `401 Unauthorized` - Signature verification failed or VRF invalid
- `429 Too Many Requests` - Rate limited or quota exceeded
- `503 Service Unavailable` - Dedup store unavailable (degraded mode)

**Response Body (200/202):**
```json
{
  "accepted": true,
  "recomputed_D_hat": 1.405,
  "recomputed_budget": 0.418,
  "reason": "",
  "escalated": false
}
```

---

## Observability

### Prometheus Metrics

#### Core Metrics (Phase 1/2):

| Metric | Type | Description |
|--------|------|-------------|
| `flk_ingest_total` | Counter | Total PCS submissions |
| `flk_dedup_hits` | Counter | Duplicate submissions |
| `flk_accepted` | Counter | Verified and accepted (200) |
| `flk_escalated` | Counter | Escalated for review (202) |
| `flk_signature_errors` | Counter | Signature verification failures |
| `flk_wal_errors` | Counter | WAL write errors |

#### Multi-Tenant Metrics (Phase 3):

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `flk_ingest_total_by_tenant` | CounterVec | `tenant_id` | Total PCS by tenant |
| `flk_accepted_by_tenant` | CounterVec | `tenant_id` | Accepted PCS by tenant |
| `flk_escalated_by_tenant` | CounterVec | `tenant_id` | Escalated PCS by tenant |
| `flk_quota_exceeded_by_tenant` | CounterVec | `tenant_id` | Quota exceeded events |
| `flk_signature_errors_by_tenant` | CounterVec | `tenant_id` | Signature failures by tenant |

#### Tiering Metrics (Phase 4):

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `tier_hot_hits` | Counter | - | Hot tier cache hits |
| `tier_warm_hits` | Counter | - | Warm tier cache hits |
| `tier_cold_hits` | Counter | - | Cold tier reads |
| `tier_promotions` | Counter | - | Promotions (cold→warm, warm→hot) |
| `tier_demotions` | Counter | - | Demotions (hot→warm, warm→cold) |
| `tier_evictions` | Counter | - | Evictions from all tiers |

#### Geo-Replication Metrics (Phase 4):

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `wal_replication_lag_seconds` | Gauge | `source_region`, `target_region` | WAL replication lag |
| `dedup_divergence` | Gauge | `region_a`, `region_b` | Dedup state divergence (%) |

### Grafana Dashboards

Access at `http://localhost:3000` (Docker Compose) or via Ingress in Kubernetes.

**Dashboards:**
1. **Overview Dashboard** - Total ingests, accepted, escalated, error rates
2. **Multi-Tenant Dashboard** (Phase 3) - Per-tenant metrics, quota usage, SLO tracking
3. **Tiered Storage Dashboard** (Phase 4) - Tier hit ratios, latency heatmaps, cost trends
4. **Multi-Region Dashboard** (Phase 4) - Traffic by region, WAL replication lag, dedup consistency

### SLO Alerts (Phase 2)

19 Prometheus alert rules in `observability/prometheus/alerts.yml`:
- **SLO Alerts**: Error budget burn >2%, high latency (p95 >200ms), signature spike
- **Availability Alerts**: Backend down, health check failing, 5xx errors
- **Resource Alerts**: High CPU/memory (>80%/85%), WAL disk pressure (>85%)
- **Dedup Alerts**: Redis/Postgres down, high latency (p95 >100ms)

---

## Testing

### Go Backend Tests

```bash
cd backend
go test ./...

# With coverage
go test -cover ./...

# Specific package (e.g., Phase 3 tenant manager)
go test -v ./internal/tenant
```

### Python Agent Tests

```bash
cd agent
pip install -r requirements.txt
pip install pytest

# Run tests (Phase 1 unit tests: 33 tests)
pytest tests/

# With coverage
pytest --cov=src tests/
```

### E2E Integration Tests (Phase 2)

```bash
# Docker Compose E2E tests
cd tests/e2e
docker-compose -f ../../infra/compose-tests.yml up -d
pytest test_backend_integration.py -v
# 15 test cases: HMAC acceptance, deduplication, signature rejection, verify-before-dedup

# Cleanup
docker-compose -f ../../infra/compose-tests.yml down
```

### Load Testing (Phase 2)

```bash
# k6 load test with SLO gates
cd load
k6 run baseline.js

# Expected output:
# - p95 latency <200ms
# - Error rate <1%
# - Throughput: 1000-2000 req/s (memory dedup)
```

### SDK Tests (Phase 4)

**Go SDK:**
```bash
cd sdk/go
go test -v
```

**TypeScript SDK:**
```bash
cd sdk/ts
npm install
npm test
```

---

## Security

### Transport Security

- **TLS**: Required for production. Use Let's Encrypt via Caddy or cert-manager
- **mTLS**: Optional for internal backend-to-backend communication
- **NetworkPolicies**: Whitelist ingress traffic (Kubernetes)

### Signature Verification (Phase 1)

- **HMAC-SHA256**: Recommended for agents (symmetric key)
- **Ed25519**: Recommended for gateways (asymmetric)
- **Payload**: `{pcs_id, merkle_root, epoch, shard_id, D_hat, coh_star, r, budget}`
- **Canonicalization**: Numbers rounded to 9 decimals; JSON sorted keys, no spaces

### Multi-Tenant Security (Phase 3)

- **Per-Tenant Keys**: Isolated signing keys per tenant
- **Quotas**: Daily quota limits, token bucket rate limiting
- **PII Protection**: Detect/block/redact emails, phones, SSNs, credit cards
- **Audit Logs**: WORM (Write-Once-Read-Many) for tamper-evidence

### Adversarial Defenses (Phase 3)

- **VRF Verification**: Verify proofs from agents (RFC 9381 ECVRF)
- **Sanity Checks**: N_j monotonicity, scale ranges, coherence/compressibility bounds
- **Anomaly Scoring**: 0.0-1.0 score with alert threshold (0.5) and reject threshold (0.8)

### Metrics Protection

- `/metrics` endpoint secured with Basic Auth
- Set `METRICS_USER` and `METRICS_PASS`
- Or restrict at proxy/ingress level

---

## Operational Runbooks

### Phase 1/2 Runbooks

**Scenario: Backend Returns 429 (Rate Limit)**

**Symptom:** HTTP 429 responses to agent

**Action:**
1. Agent automatically backs off with jitter
2. If sustained, increase `TOKEN_RATE` or scale replicas
3. Monitor with `rate(flk_ingest_total[1m])`

**Scenario: Signature Failures (401 Spike)**

**Symptom:** Sudden increase in 401 responses

**Action:**
1. Check `PCS_SIGN_ALG` matches between agent and backend
2. Verify key rotation hasn't caused mismatch
3. Confirm numeric rounding (9 decimals) is consistent
4. Check for clock drift (affects timestamp but not signature)

**Scenario: Escalation Rate Spike (202)**

**Symptom:** `flk_escalated` counter increases

**Action:**
1. Inspect PCS distributions: D̂, coh★, r
2. Compare against server tolerances (`tolD=0.15`, `tolCoh=0.05`)
3. Review `N_j` computation and scales list
4. Consider widening tolerances only after analysis

**Scenario: WAL Disk Growth**

**Symptom:** Disk usage increasing in WAL directory

**Action:**
1. Confirm agent marks entries as `acked`
2. Enable WAL compaction (remove acked beyond 14d horizon)
3. Backend: rotate Inbox WAL with retention policy

**Detailed Runbooks** (Phase 2):
- `docs/runbooks/signature-spike.md` (4,800+ words, 6 diagnostic scenarios)
- `docs/runbooks/dedup-outage.md` (3,500+ words, recovery procedures)

### Phase 3 Runbooks

**Tenant SLO Breach:**
- Root cause scenarios: signature failures, escalation spike, quota exceeded, policy misconfiguration
- Mitigation: degraded mode, quota increase, policy rollback, key re-sync
- Runbook: `docs/runbooks/tenant-slo-breach.md`

**VRF Invalid Surge:**
- Root cause scenarios: agent misconfiguration, compromised key, coordinated attack, backend bug
- Mitigation: disable tenant, block IPs, rollback deployment, emergency bypass
- Runbook: `docs/runbooks/vrf-invalid-surge.md`

### Phase 4 Runbooks

**Multi-Region Failover:**
- Scenario: One region unavailable (network partition, datacenter outage)
- Goal: RTO ≤5 min, RPO ≤2 min, no data loss
- Actions: Reroute traffic via GSLB, verify WAL lag, enable degraded mode if needed
- Runbook: `docs/runbooks/geo-failover.md` (8,000+ words)

**Geo Split-Brain Resolution:**
- Scenario: Both regions processed writes independently during partition
- Goal: Reconcile state without data loss
- Actions: Detect divergence, elect authoritative WAL, rebuild dedup, verify idempotency
- Runbook: `docs/runbooks/geo-split-brain.md` (7,000+ words)

**Shard Migration:**
- Scenario: Scale dedup from N→N+1 shards (or N→N-1)
- Goal: Zero downtime, no data loss, no cache stampedes
- Actions: Plan ring, pre-copy keys, dual-write, cutover, verify consistency
- Runbook: `docs/runbooks/shard-migration.md` (8,500+ words)

**Tiered Storage Cold Hit Latency:**
- Scenario: Cold tier reads spike → p95 latency >200ms
- Goal: Balance latency and cost via TTL policies
- Actions: Increase hot TTL, preemptive warming, per-tenant policies
- Runbook: `docs/runbooks/tier-cold-hot-miss.md` (6,500+ words)

**Async Audit Backlog:**
- Scenario: Audit workers fall behind, backlog grows
- Goal: Drain backlog within 1 hour SLO, ensure 100% audit coverage
- Actions: Scale workers, investigate slow tasks, purge DLQ, verify completeness
- Runbook: `docs/runbooks/audit-backlog.md` (7,500+ words)

---

## Development Workflow

### Building Backend

```bash
cd backend
go mod tidy
go build -o server ./cmd/server
./server
```

### Running Agent (Python)

```bash
cd agent
pip install -r requirements.txt

# Example usage
python -c "
from agent.src import PCSAgent
import numpy as np

agent = PCSAgent(
    shard_id='dev-001',
    endpoint='http://localhost:8080/v1/pcs/submit',
    sign_alg='hmac',
    hmac_key='supersecret'
)

# Generate synthetic PCS
pcs = agent.compute_pcs(
    epoch=1,
    scales=[2, 4, 8],
    N_j={2: 3, 4: 5, 8: 9},
    points=np.random.randn(100, 3),
    raw_data=b'test data',
    seed=42
)

# Submit
success = agent.submit_pcs(pcs)
print(f'Submitted: {success}')
"
```

### Using Go SDK (Phase 4)

```go
package main

import (
    "context"
    "log"
    fractal "github.com/fractal-lba/kakeya/sdk/go"
)

func main() {
    client := fractal.NewClient(
        "https://api.fractal-lba.example.com",
        "tenant1",
        "supersecret",
        "hmac",
    )

    pcs := &fractal.PCS{
        PCSID: "abc123...",
        Schema: "fractal-lba-kakeya",
        // ... populate fields
    }

    result, err := client.SubmitPCS(context.Background(), pcs)
    if err != nil {
        log.Fatalf("Submission failed: %v", err)
    }

    log.Printf("Accepted: %v", result.Accepted)
}
```

### Using TypeScript SDK (Phase 4)

```typescript
import { FractalLBAClient, PCS } from '@fractal-lba/client';

const client = new FractalLBAClient({
  baseURL: 'https://api.fractal-lba.example.com',
  tenantID: 'tenant1',
  signingKey: 'supersecret',
  signingAlg: 'hmac'
});

const pcs: PCS = {
  pcs_id: 'abc123...',
  schema: 'fractal-lba-kakeya',
  // ... populate fields
};

try {
  const result = await client.submitPCS(pcs);
  console.log('Accepted:', result.accepted);
} catch (error) {
  console.error('Submission failed:', error.message);
}
```

---

## Performance & SLOs

### Phase 1/2 Baseline:
- **p95 Verify Latency**: ≤ 200ms (single replica, in-memory dedup)
- **Error Budget**: `escalated/ingest_total ≤ 2%` daily
- **Dedup Hit Ratio**: Goal ≥ 40% under typical replay conditions

### Phase 3/4 Scale:
- **Multi-Tenant Throughput**: 2,000 req/s (memory dedup), 1,500 req/s (Redis), 800 req/s (Postgres)
- **Tiered Storage Latency**: Hot <5ms, Warm <50ms, Cold <500ms (p95)
- **Multi-Region RTO/RPO**: RTO ≤5 minutes, RPO ≤2 minutes
- **Audit Lag SLO**: 99% of audit tasks complete within 1 hour

---

## Roadmap

### ✅ Completed (Phase 1-4)

**Phase 1: Canonicalization & Signing**
- [x] 9-decimal rounding, 8-field signature subset
- [x] HMAC-SHA256 and Ed25519 verification
- [x] Signal computation clarifications (D̂, coh★, r)
- [x] Golden test vectors and unit tests (33 tests)

**Phase 2: E2E Testing & Production Deployment**
- [x] E2E integration tests (15 tests: HMAC, dedup, verify-before-dedup)
- [x] Production Helm chart (11 templates, HPA, PDB, NetworkPolicy)
- [x] SLO-driven monitoring (19 Prometheus alerts)
- [x] k6 load testing with SLO gates
- [x] Ed25519 keygen script, WAL compaction automation

**Phase 3: Multi-Tenant, Governance & Adversarial Robustness**
- [x] Multi-tenant isolation (per-tenant keys, quotas, labeled metrics)
- [x] WORM audit logs (tamper-evident, Merkle anchoring ready)
- [x] Policy DSL validation (compile-time checks, versioned policies)
- [x] PII scanner (detect/block/redact modes, staged rollout)
- [x] Adversarial defenses (VRF, N_j monotonicity, sanity checks, anomaly scoring)
- [x] OpenAPI 3.0 specification, Python SDK

**Phase 4: Multi-Region, Sharding, Tiering & SDK Parity**
- [x] Sharded dedup with consistent hashing (virtual nodes, health checks)
- [x] Tiered storage (hot/warm/cold) with TTL policies and lazy promotion
- [x] Go SDK with canonical signing and multi-tenant support
- [x] TypeScript SDK with automatic signing and retry logic
- [x] Multi-region runbooks (5 comprehensive guides, 38,000+ words)

### 🔮 Future Enhancements

**Performance Optimizations:**
- [ ] Async WORM writes (reduce hot path latency)
- [ ] Batch metrics updates (reduce Prometheus cardinality)
- [ ] Differential Privacy for aggregate metrics

**Operational:**
- [ ] Automated chaos engineering drills (Chaos Mesh integration)
- [ ] Advanced canary rollout with percentage-based promotion
- [ ] Real-time geo-divergence detection and auto-reconciliation

**Research:**
- [ ] Formal proofs of invariants for PCS transformations
- [ ] VRF-based direction sampling (RFC 9381 ECVRF)
- [ ] Blockchain anchoring integration (Ethereum, Trillian)

---

## Documentation

- **[CLAUDE.md](./CLAUDE.md)** - Complete technical specification and design invariants
- **[docs/architecture/](./docs/architecture/)** - System design, signal computation deep-dive
- **[docs/api/](./docs/api/)** - REST API reference with examples
- **[docs/deployment/](./docs/deployment/)** - Docker Compose and Kubernetes deployment guides
- **[docs/operations/](./docs/operations/)** - Troubleshooting and operational procedures
- **[docs/security/](./docs/security/)** - Threat model, defenses, incident response
- **[docs/runbooks/](./docs/runbooks/)** - Detailed operational runbooks (38,000+ words)

**Phase Implementation Reports:**
- **[PHASE1_REPORT.md](./PHASE1_REPORT.md)** - Phase 1 implementation details (13,000+ words)
- **[PHASE2_REPORT.md](./PHASE2_REPORT.md)** - Phase 2 implementation details (8,000+ words)
- **[PHASE3_REPORT.md](./PHASE3_REPORT.md)** - Phase 3 implementation details (13,000+ words)
- **[PHASE4_REPORT.md](./PHASE4_REPORT.md)** - Phase 4 implementation details (coming soon)

---

## License

See [LICENSE](./LICENSE) for details.

---

## Contributing

1. Read [CLAUDE.md](./CLAUDE.md) for design invariants
2. Never change PCS field semantics without bumping `version`
3. Always preserve `pcs_id` contract and signing subset
4. Code must be idempotent by default
5. Include tests for new features (unit + E2E)
6. Update CLAUDE.md for design changes
7. For Phase 3+ features, ensure backward compatibility with Phase 1/2

---

## Support

- **Issues**: [GitHub Issues](https://github.com/fractal-lba/kakeya/issues)
- **Documentation**: See `CLAUDE.md` for technical deep-dive
- **Community**: [Discord](https://discord.gg/fractal-lba) (coming soon)

---

**Built with**: Go 1.22, Python 3.10, TypeScript 5.0, Redis 7, PostgreSQL 15, Prometheus, Grafana, Docker, Kubernetes

**Phase Coverage**: Phase 1 (Canonicalization) ✅ | Phase 2 (E2E + Production) ✅ | Phase 3 (Multi-Tenant + Governance) ✅ | Phase 4 (Multi-Region + Scale) ✅
