# Fractal LBA + Kakeya FT Stack

> **Turn distributed event streams into verifiable, compact proofs—with mathematical rigor and battle-tested fault tolerance.**

## Elevator Pitch

Imagine you're analyzing massive, distributed event streams from IoT sensors, financial transactions, or network traffic. You need to **prove** your computation happened correctly, **compress** terabytes into kilobytes, and **never lose data**—even during crashes, network splits, or replay attacks.

**Fractal LBA + Kakeya FT Stack** solves this by:

1. **Computing cryptographic summaries (PCS)** that capture the "shape" of your data using fractal geometry (D̂), directional coherence (coh★), and compressibility (r)
2. **Verifying summaries server-side** with robust statistical methods (Theil-Sen regression) to catch manipulated or corrupted data
3. **Guaranteeing delivery** with write-ahead logs (WAL) on both agent and backend—your proofs survive crashes
4. **Preventing duplicates** with idempotent deduplication across memory, Redis, or Postgres
5. **Ensuring authenticity** with HMAC-SHA256 or Ed25519 signatures

All wrapped in production-ready **Docker Compose** and **Kubernetes Helm charts** with observability (Prometheus + Grafana), auto-scaling (HPA), and security hardening (mTLS, NetworkPolicies).

**Use cases:** Blockchain light clients, IoT data integrity, compliance audit trails, distributed system health monitoring, anti-fraud detection.

---

## Overview

This system implements a distributed architecture where:

- **Python Agent** computes signals from event streams and generates signed PCS
- **Go Backend** verifies PCS with strict idempotency and fault tolerance
- **WAL (Write-Ahead Logs)** ensure at-least-once delivery semantics
- **Deduplication** provides idempotent processing with configurable storage backends
- **Signing** (HMAC/Ed25519) ensures authenticity
- **Observability** via Prometheus metrics and Grafana dashboards

See [CLAUDE.md](./CLAUDE.md) for the complete technical specification and design invariants.

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

### Production Deployment with Kubernetes

```bash
# Install with Helm
cd deployments/k8s/helm

helm install fractal-lba ./fractal-lba \
  --set signing.enabled=true \
  --set signing.alg=hmac \
  --set-string signing.hmacKey="your-secret-key" \
  --set metricsBasicAuth.enabled=true \
  --set-string metricsBasicAuth.password="your-metrics-password" \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=api.fractal-lba.example.com

# Check status
kubectl get pods
kubectl logs -l app=backend
```

## Architecture

### Components

#### Backend (Go)

Located in `backend/`:

- **API Handler** (`internal/api/`) - HTTP endpoint `/v1/pcs/submit`
- **Verification Engine** (`internal/verify/`) - Recomputes D̂, validates signals
- **Deduplication** (`internal/dedup/`) - Memory/Redis/Postgres backends
- **WAL** (`internal/wal/`) - Write-ahead logging with fsync
- **Signing** (`internal/signing/`) - HMAC-SHA256 and Ed25519 verification
- **Metrics** (`internal/metrics/`) - Prometheus counters

#### Agent (Python)

Located in `agent/src/`:

- **Signals** (`signals.py`) - Computes D̂, coh★, r
- **Merkle** (`merkle.py`) - Merkle tree for data integrity
- **Outbox WAL** (`outbox.py`) - Agent-side WAL with fsync
- **Client** (`client.py`) - HTTP client with exponential backoff + jitter
- **Agent** (`agent.py`) - Main orchestrator

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

## Configuration

### Environment Variables (Backend)

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

### Environment Variables (Agent)

| Variable | Description |
|----------|-------------|
| `ENDPOINT` | Backend submission URL |
| `PCS_SIGN_ALG` | Signature algorithm: `none`, `hmac`, `ed25519` |
| `PCS_HMAC_KEY` | HMAC secret key |
| `PCS_ED25519_PRIV_B64` | Ed25519 private key base64 |

## API Reference

### POST /v1/pcs/submit

Submit a Proof-of-Computation Summary.

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
- `400 Bad Request` - Malformed JSON or validation error
- `401 Unauthorized` - Signature verification failed
- `429 Too Many Requests` - Rate limited

## Observability

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `flk_ingest_total` | Counter | Total PCS submissions |
| `flk_dedup_hits` | Counter | Duplicate submissions |
| `flk_accepted` | Counter | Verified and accepted (200) |
| `flk_escalated` | Counter | Escalated for review (202) |
| `flk_signature_errors` | Counter | Signature verification failures |
| `flk_wal_errors` | Counter | WAL write errors |

### Grafana Dashboard

Access at `http://localhost:3000` (Docker Compose) with:
- Total ingests, accepted, escalated stats
- Ingest rate over time
- Dedup hit ratio
- Error rates
- Escalation rate gauge (SLO: ≤2%)

## Testing

### Go Backend Tests

```bash
cd backend
go test ./...

# With coverage
go test -cover ./...

# Specific package
go test -v ./internal/verify
```

### Python Agent Tests

```bash
cd agent
pip install -r requirements.txt
pip install pytest

# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

## Security

### Transport Security

- **TLS**: Required for production. Use Let's Encrypt via Caddy or cert-manager
- **mTLS**: Optional for internal backend-to-backend communication

### Signature Verification

- **HMAC-SHA256**: Recommended for agents (symmetric key)
- **Ed25519**: Recommended for gateways (asymmetric)
- Payload: `{pcs_id, merkle_root, epoch, shard_id, D_hat, coh_star, r, budget}`
- Numbers rounded to 9 decimals; JSON sorted keys, no spaces

### Metrics Protection

- `/metrics` endpoint secured with Basic Auth
- Set `METRICS_USER` and `METRICS_PASS`
- Or restrict at proxy/ingress level

## Operational Runbooks

### Scenario: Backend Returns 429 (Rate Limit)

**Symptom:** HTTP 429 responses to agent

**Action:**
1. Agent automatically backs off with jitter
2. If sustained, increase `TOKEN_RATE` or scale replicas
3. Monitor with `rate(flk_ingest_total[1m])`

### Scenario: Signature Failures (401 Spike)

**Symptom:** Sudden increase in 401 responses

**Action:**
1. Check `PCS_SIGN_ALG` matches between agent and backend
2. Verify key rotation hasn't caused mismatch
3. Confirm numeric rounding (9 decimals) is consistent
4. Check for clock drift (affects timestamp but not signature)

### Scenario: Escalation Rate Spike (202)

**Symptom:** `flk_escalated` counter increases

**Action:**
1. Inspect PCS distributions: D̂, coh★, r
2. Compare against server tolerances (`tolD=0.15`, `tolCoh=0.05`)
3. Review `N_j` computation and scales list
4. Consider widening tolerances only after analysis

### Scenario: WAL Disk Growth

**Symptom:** Disk usage increasing in WAL directory

**Action:**
1. Confirm agent marks entries as `acked`
2. Enable WAL compaction (remove acked beyond 14d horizon)
3. Backend: rotate Inbox WAL with retention policy

## Development Workflow

### Building Backend

```bash
cd backend
go mod tidy
go build -o server ./cmd/server
./server
```

### Running Agent

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

## Performance & SLOs

- **p95 Verify Latency**: ≤ 200ms (single replica, in-memory dedup)
- **Error Budget**: `escalated/ingest_total ≤ 2%` daily
- **Dedup Hit Ratio**: Goal ≥ 40% under typical replay conditions

## Roadmap

### Short-term
- [ ] Add verify latency histogram
- [ ] Helm: resources, HPA, PDB, NetworkPolicy defaults
- [ ] Redis/Postgres integration tests in CI

### Mid-term
- [ ] SOPS/age for secrets management
- [ ] Canary deploy hooks with error-budget gates

### Long-term
- [ ] Formal proofs of invariants
- [ ] VRF-based direction sampling

## License

See [LICENSE](./LICENSE) for details.

## Contributing

1. Read [CLAUDE.md](./CLAUDE.md) for design invariants
2. Never change PCS field semantics without bumping `version`
3. Always preserve `pcs_id` contract and signing subset
4. Code must be idempotent by default
5. Include tests for new features
6. Update CLAUDE.md for design changes

## Support

- Issues: [GitHub Issues](https://github.com/fractal-lba/kakeya/issues)
- Documentation: See `CLAUDE.md` for technical deep-dive

---

**Built with**: Go 1.22, Python 3.10, Redis, PostgreSQL, Prometheus, Grafana, Docker, Kubernetes
