# CLAUDE.md — Project Memory & Ops Playbook (Fractal LBA + Kakeya FT Stack)

> Audience: LLM collaborators (Claude/ChatGPT) and human engineers.
> Purpose: A **single source of truth** for concepts, contracts, guard-rails, and operational routines so assistants can help without drifting the design.

---

## 0) TL;DR

* This system ingests **Proof-of-Computation Summaries (PCS)** computed by a Python agent and **verifies** them on a Go backend with strict **idempotency** and **fault tolerance** (WAL→verify→dedup→effects).
* Core signals: `D̂` (fractal/scale slope via Theil–Sen), `coh★` (directional coherence), `r` (LZ compressibility proxy).
* Safety: **WAL logs**, **idempotent dedup**, **HMAC/Ed25519 signatures**, **TLS/mTLS**, **/metrics** Basic Auth.
* Observability: Prometheus counters + Grafana dashboard.
* Deploy: Docker Compose (with optional Caddy auto-TLS) or Helm (TLS/mTLS + signing + metrics auth).

---

## 1) Core Concepts & Invariants

### 1.1 PCS (Proof-of-Computation Summary)

A compact, verifiable JSON record:

```json
{
  "pcs_id": "<sha256(merkle_root|epoch|shard_id)>",
  "schema": "fractal-lba-kakeya",
  "version": "0.1",
  "shard_id": "shard-001",
  "epoch": 1,
  "attempt": 1,
  "sent_at": "2025-01-01T00:00:00Z",
  "seed": 42,
  "scales": [2,4,8,16,32],
  "N_j": { "2": 3, "4": 5, "8": 9, "16": 17, "32": 31 },
  "coh_star": 0.73,
  "v_star": [0.12, 0.98, -0.05],
  "D_hat": 1.41,
  "r": 0.87,
  "regime": "sticky | mixed | non_sticky",
  "budget": 0.42,
  "merkle_root": "<hex>",
  "sig": "<base64 signature>",
  "ft": { "outbox_seq": 123, "degraded": false, "fallbacks": [], "clock_skew_ms": 0 }
}
```

**Invariants**

* `pcs_id = sha256(merkle_root|epoch|shard_id)` (ASCII concatenation with `|`).
* `version` MUST bump for any semantic change to PCS fields.
* Numbers used for signing are **rounded to 9 decimal places** to avoid float drift.
* `N_j` keys are stringified scales (e.g., `"8": 9`), not integers, for JSON stability.

### 1.2 Signals

* **D̂ (fractal slope):** median slope (Theil–Sen) of `log2(scale)` vs `log2(unique_nonempty_cells)`.
* **coh★ (directional coherence):** max fraction of points concentrated in histogram bins along the “best” projection direction from a sampled set.
* **r (compressibility):** size(compressed)/size(raw) via LZ (zlib). Lower implies more structure.

### 1.3 Regimes & Budget

* Regimes:

    * `sticky` if `coh★ ≥ 0.70` and `D̂ ≤ 1.5`
    * `non_sticky` if `D̂ ≥ 2.6`
    * otherwise `mixed`
* Budget formula (bounded to `[0,1]`):
  `budget = base + α(1 − r) + β·max(0, D̂ − D0) + γ·coh★`

Typical params (server `VerifyParams`):

* `tolD=0.15`, `tolCoh=0.05`, `alpha=0.30`, `beta=0.50`, `gamma=0.20`, `base=0.10`, `D0=2.2`, `dedup_ttl=14 days`.

---

## 2) Architecture & Dataflow

### 2.1 Flow (happy path)

1. **Agent (Python)** computes signals from event stream; creates PCS; **signs** it; appends to **Outbox WAL** (fsync).
2. Agent POSTs PCS to `/v1/pcs/submit` (backoff+jitter retries; DLQ if exhausted).
3. **Backend (Go)** appends request body to **Inbox WAL** (fsync) before parsing.
4. Backend **verifies**:

    * Recompute `D̂` from `scales` and `N_j` within `tolD`.
    * Check coherence bounds (`0 ≤ coh★ ≤ 1+tolCoh`).
    * Validate **signature** if enabled.
5. **Idempotent dedup** on `pcs_id` (memory/Redis/Postgres). First-seen -> persist outcome; duplicates -> short-circuit with stored outcome.
6. Update Prometheus counters; return 200 (accepted) or 202 (escalated).

### 2.2 Fault tolerance patterns

* **At-least-once** delivery with **WAL outbox** (agent) and **WAL inbox** (backend).
* **Idempotency**: `pcs_id` key; dedup TTL ~14 days.
* Safe defaults when inputs missing: flag `ft.degraded=true`, reduce trust and escalate.

---

## 3) Security Model

### 3.1 Transport

* Local/dev: self-signed TLS allowed; **verify CA in prod**.
* Public: terminate TLS at **Caddy** (Let’s Encrypt) or K8s Ingress (cert-manager).
* Private: **mTLS** between proxy and backend if crossing trust boundaries.

### 3.2 PCS Signing

* **Agent** signs the **SHA-256 digest** of a **stable field subset**:

  ```jsonc
  { "pcs_id", "merkle_root", "epoch", "shard_id", "D_hat", "coh_star", "r", "budget" }
  // numbers rounded to 9 decimals; JSON serialized with sorted keys and no spaces
  ```
* Algorithms:

    * **HMAC-SHA256** (recommended on agents): `PCS_SIGN_ALG=hmac`, `PCS_HMAC_KEY=<secret>`
    * **Ed25519** (recommended on gateway): `PCS_SIGN_ALG=ed25519`, `PCS_ED25519_PUB_B64=<32B pub key b64>`
* Backend verifies signature **before** dedup and compute.

### 3.3 Metrics & PII

* `/metrics` protected by Basic Auth (`METRICS_USER`, `METRICS_PASS`) or restricted at proxy.
* Never log raw payloads; logs may include `pcs_id`, `status`, high-level counters.
* Redact emails/phones in any user-originated strings (defense-in-depth).

---

## 4) Contracts & Interfaces

### 4.1 Endpoint

```
POST /v1/pcs/submit
Content-Type: application/json
Body: PCS (see §1.1)
200 OK    → accepted={true}, recomputed values, budget
202 Accepted → accepted={false}, escalated path
401 Unauthorized → signature invalid (if signing enabled)
400 Bad Request → malformed JSON / missing required fields
```

### 4.2 Idempotency & Dedup

* Key: `pcs_id`
* Store backends: `memory` (file snapshot), `redis`, `postgres`.
* First-write wins. Duplicates return cached outcome with original HTTP status.

---

## 5) Deployment Profiles

### 5.1 Docker Compose

* Services: `backend`, `agent` (on-demand), `redis`, `postgres`, `prometheus`, `grafana`.
* Optional overlay: `caddy` with auto-TLS (requires public DNS + 80/443).

### 5.2 Kubernetes (Helm)

* Chart values toggle: TLS/mTLS at Ingress, signing, metrics auth, dedup backend.
* Recommended prod additions:

    * `resources` (CPU/mem), `HPA`, `PodDisruptionBudget`, `topologySpreadConstraints`.
    * NetworkPolicies limiting access to backend from ingress and trusted agents only.
    * Secret management via SOPS/age or external KMS.

---

## 6) Observability

### 6.1 Prometheus Counters

* `flk_ingest_total` — total POSTs observed.
* `flk_dedup_hits` — duplicates served from cache.
* `flk_accepted` — successful verifies (200).
* `flk_escalated` — uncertain/failed checks (202).

### 6.2 Grafana Dashboard

* Stat tiles for totals, rate over time: `rate(flk_ingest_total[1m])`.
* Suggested additions:

    * Error panel (non-2xx rate).
    * Verify latency histogram (if instrumented).
    * Dedup hit ratio.

---

## 7) Runbooks

### 7.1 Backend returns 429 (throttle)

* **Symptom:** HTTP 429 to agent.
* **Action:** Agent already backs off with jitter; if sustained, lower input rate or increase `TOKEN_RATE`/replicas.

### 7.2 Dedup store outage

* **Symptom:** Errors reaching Redis/Postgres.
* **Action:** Prefer fail-closed (503 + `Retry-After`), keep writing Inbox WAL, alert on availability. After recovery, replay if needed.

### 7.3 Signature failures (401)

* **Symptom:** Sudden spike in 401 on `/v1/pcs/submit`.
* **Action:** Check `PCS_SIGN_ALG`, rotated keys, drift in numeric rounding; ensure JSON canonicalization; verify clock skew only affects audit logs, not signatures.

### 7.4 Data drift / escalations (202 spike)

* **Symptom:** `flk_escalated` increases.
* **Action:** Inspect distributions of `D̂`, `coh★`, `r`; compare to server tolerances; confirm `N_j` computation and `scales` list. Consider raising tolerances after analysis, not ad-hoc.

### 7.5 WAL growth

* **Symptom:** Disk usage rising in Outbox/Inbox.
* **Action:** Confirm `acked` markers on agent; enable WAL compaction job (delete acked beyond horizon). On backend, rotate Inbox WAL with retention policy (e.g., 14d).

---

## 8) Testing Strategy

### 8.1 Unit

* Agent: `signals.py`, `merkle.py`, outbox WAL append/mark/pending.
* Backend: verify recomputation, budget function, dedup store contract.

### 8.2 Integration

* End-to-end: launch backend (memory dedup), POST PCS with synthetic signals, assert 200/202.
* Replay: send same `pcs_id` twice → second must be instant (dedup hit).

### 8.3 Load

* **k6** script posts constant PCS load; watch ingest rate and latency.
* **Locust** to shape traffic with varied regimes and burst patterns.

### 8.4 Chaos (recommended)

* Drop Redis for N seconds → verify 503 + retry, WAL persists.
* Inject duplicate deliveries → ensure idempotency and stable metrics.

---

## 9) Performance & SLOs

* **SLO**: p95 verify latency ≤ 200 ms under nominal load (single replica, in-mem dedup).
* **Error budget guard**: `escalated/ingest_total ≤ 2%` daily.
* **Dedup hit ratio**: goal ≥ 40% under typical replay/dup conditions (depends on source topology).

---

## 10) Change Management & Versioning

* **PCS schema**: SemVer. Breaking field changes → **major** bump; added optional field → **minor**; non-semantic text → patch.
* **Server tolerances**: store in config and version in release notes; never implicitly widen in hotfixes without audit.
* **Signing**: if switching algorithms/keys, support **overlap period** where both old and new keys verify; publish effective-from timestamp.

---

## 11) Collaboration Rules for LLMs

1. **Never** change PCS field semantics or add fields without stating: “bump `version` required.”
2. **Always** preserve `pcs_id` contract and signing subset/rounding.
3. Suggest code that is **idempotent by default** and **pure** in verify path.
4. If you propose retries: exponential backoff **with jitter**; cap and DLQ path noted.
5. Security advice must include **TLS/mTLS** and **secret handling**; **no hardcoding secrets**.
6. For Kubernetes, always mention `resources`, `HPA`, `PDB`, and **NetworkPolicies**.
7. When unsure, provide a **safe default** and call out the assumption explicitly.

---

## 12) DSL (ANTLR) — Brief (Optional Path)

* **Goal:** human-readable policy rules → JSON AST → server params (`tolD`, `tolCoh`, budget weights, regime thresholds).
* **Pattern:** EBNF style grammar; Visitors produce strongly typed JSON with explicit units and bounds.
* **Guard-rails:** refuse compilation if any threshold breaks invariants (`0 ≤ coh★ ≤ 1+ε`, `0 ≤ weights ≤ 1`, normalized/ bounded budget).

*(Full grammar and visitors are tracked in the DSL module; not required for MVP operation.)*

---

## 13) Roadmap

* **Short-term**

    * Add verify **latency histogram** + request duration buckets.
    * Helm: `resources`, `HPA`, `PDB`, `NetworkPolicy` defaults.
    * Redis/Postgres integration tests in CI.
* **Mid-term**

    * SOPS/age for secrets; KMS-backed optional.
    * Canary deploy hooks with error-budget gates.
* **Long-term**

    * Formal proofs of invariants for PCS transformations.
    * VRF-based direction sampling to reduce adversarial steering.

---

## 14) Glossary

* **WAL**: Write-Ahead Log; fsync’d append-only files used for crash recovery.
* **Idempotency**: repeated same request yields same effect; implemented with `pcs_id` dedup store.
* **Theil–Sen**: robust median slope estimator across pairwise points.
* **Kakeya-style**: directional projection ideas motivating `coh★`.
* **r (compressibility)**: proxy for structural predictability (zlib ratio).

---

## 15) FAQ (Ops)

* **Q:** Why sometimes 202 but no error?
  **A:** Verify passed bounds but confidence low; record is **escalated** for downstream checks. This protects against silent false-accepts.

* **Q:** Can we recover from total backend crash?
  **A:** Yes. Inbox WAL contains request bodies; upon restart, new requests resume; dedup prevents re-effects.

* **Q:** How to rotate keys for HMAC?
  **A:** Introduce `PCS_HMAC_KEYS=key1,key2` verification (multikey), flip agent to `key2`, wait TTL window, then drop `key1`.

---

## 16) Cut-and-Paste Config Snippets

### 16.1 Backend env (Compose)

```
DEDUP_BACKEND=redis
REDIS_ADDR=redis:6379
TOKEN_RATE=100
METRICS_USER=ops
METRICS_PASS=change-me
PCS_SIGN_ALG=hmac
PCS_HMAC_KEY=supersecret
```

### 16.2 Agent env

```
PCS_SIGN_ALG=hmac
PCS_HMAC_KEY=supersecret
ENDPOINT=https://api.example.com/v1/pcs/submit
```

### 16.3 Helm toggles

```
--set signing.enabled=true \
--set signing.alg=hmac \
--set metricsBasicAuth.enabled=true \
--set env.dedupBackend=redis
```

---

## 17) Final “Do-Not-Break” List

* Do **not** change signing subset or rounding without a full coordinated rollout plan.
* Do **not** log payload bodies in production.
* Do **not** mutate verify state after writing dedup outcome.
* Do **not** accept PCS without WAL append success.

---

If you (human or LLM) propose changes, **attach a diff** and explicitly list which **invariants** remain preserved and which require a `version` bump.

---

## 18) Phase 3-9 Metrics Concurrency (Known Technical Debt)

**Status:** 24 copy lock warnings in stub implementations (advisory-only in CI)

Phase 3-9 packages have metrics structs that embed `sync.RWMutex` and return by value, copying the mutex. This is detected by `go vet` but is **intentionally advisory-only** until full implementation.

### Architecture Decision

See `backend/docs/architecture/METRICS_CONCURRENCY.md` for full details.

**Current approach:**
- `go vet` runs with `continue-on-error: true` in CI
- Warnings are visible but don't block builds
- All 24 affected files remain unchanged

**Migration plan:**
- When implementing Phase 3-9 fully, use **snapshot pattern**
- Create `XxxMetricsSnapshot` structs without mutexes
- Implement `Snapshot()` methods that copy fields under lock
- Update `GetMetrics()` to return snapshots

**Why not fix now:**
1. Phase 3-9 are stubs - field names will change
2. Snapshot pattern requires matching 420+ fields across 24 structs
3. High maintenance burden during rapid prototyping
4. Proper fix deferred to full implementation phase

**Affected packages:** audit (5), crr (4), tiering (3), cost (4), hrs (4), anomaly (2), ensemble (1), sharding (1)

---

## 19) ASV: Split Conformal Prediction & Statistical Guarantees

**Status:** ✅ **PRODUCTION READY** (Implemented 2025-10-24, Week 1-2 of ASV Timeline)

We've implemented **split conformal prediction** (Vovk 2005, Angelopoulos & Bates 2023) to provide **finite-sample miscoverage guarantees** for verification decisions.

### 19.1 Core Concept

**Nonconformity Score:** `η(x)` measures how anomalous a PCS is relative to calibration data.

**Conformal Guarantee:** Under exchangeability assumption, if we accept when `η(x) ≤ quantile(1-δ)`, then:
```
P(accept wrong PCS) ≤ δ
```
This is a **finite-sample** guarantee (not asymptotic). For δ=0.05, miscoverage ≤ 5%.

### 19.2 Usage Patterns

#### **Backend Integration (Go)**

```go
import "github.com/fractal-lba/kakeya/internal/conformal"
import "github.com/fractal-lba/kakeya/internal/api"

// Create calibration set (per-tenant recommended)
cs := conformal.NewCalibrationSet(
    1000,                  // maxSize (FIFO eviction)
    24 * time.Hour,        // time window (recency-based pruning)
    "tenant_acme",         // tenantID (empty string = global)
)

// Add calibration data (from labeled examples)
cs.Add(conformal.NonconformityScore{
    PCSID:     "pcs_abc123",
    Score:     0.42,                  // Computed by ComputeScore()
    TrueLabel: true,                  // True if known benign
    Timestamp: time.Now(),
    TenantID:  "tenant_acme",
})

// Make prediction for new PCS
pcs := &api.PCS{
    DHat:    1.87,
    CohStar: 0.73,
    R:       0.42,
    // ... other fields
}
params := api.DefaultVerifyParams()
delta := 0.05  // Target miscoverage (5%)

result, err := cs.Predict(pcs, params, delta)
if err != nil {
    log.Fatalf("Prediction failed: %v", err)
}

switch result.Decision {
case conformal.DecisionAccept:
    // Accept with confidence result.Confidence
    log.Printf("ACCEPT: score=%.3f ≤ quantile=%.3f, confidence=%.3f",
        result.Score, result.Quantile, result.Confidence)

case conformal.DecisionEscalate:
    // Near threshold - ambiguous
    log.Printf("ESCALATE: score=%.3f ~ quantile=%.3f (margin=%.3f)",
        result.Score, result.Quantile, result.Margin)

case conformal.DecisionReject:
    // Clearly anomalous
    log.Printf("REJECT: score=%.3f >> quantile=%.3f (margin=%.3f)",
        result.Score, result.Quantile, result.Margin)
}
```

#### **Drift Detection**

```go
// Monitor for distribution shift
driftDetector := conformal.NewDriftDetector(
    100,  // maxRecent (track last 100 scores)
    0.10, // ksThreshold (10% drift threshold)
)

// Add recent scores
for _, pcs := range recentPCSBatch {
    score := conformal.ComputeScore(pcs, params)
    driftDetector.AddScore(score)
}

// Check for drift
report := driftDetector.CheckDrift(cs)
if report.Drifted {
    log.Printf("DRIFT DETECTED: KS=%.4f, p-value=%.4f < 0.05",
        report.KSStatistic, report.PValue)
    log.Printf("Recommendation: %s", report.Message)

    // Trigger recalibration
    RecalibrateCalibrationSet(cs, newLabeledData)
}
```

#### **Miscoverage Monitoring**

```go
// Track empirical miscoverage vs. target δ
monitor := conformal.NewMiscoverageMonitor(1000)

// Record decisions with ground truth labels
monitor.AddDecision(correct=true)   // Correct decision
monitor.AddDecision(correct=false)  // Miscoverage

// Check calibration quality
wellCalibrated, empiricalRate, targetDelta, n := monitor.CheckCalibration(0.05)
if !wellCalibrated {
    log.Printf("MISCALIBRATION: empirical %.3f vs target %.3f (n=%d)",
        empiricalRate, targetDelta, n)
}
```

### 19.3 Agent-Side Changes (Python)

**Product Quantization (Theoretically Sound Compression):**

```python
from agent.src.signals import (
    product_quantize_embeddings,
    compute_compressibility_pq,
)

# Before: Compressed raw floats (violated finite-alphabet assumption)
# data_bytes = embeddings.tobytes()
# r = zlib.compress(data_bytes) / len(data_bytes)  # ❌ Not theoretically sound

# After: Product quantization → discrete symbols → LZ
embeddings = np.random.randn(100, 768)  # (n_tokens, d_embedding)
r = compute_compressibility_pq(
    embeddings,
    n_subspaces=8,      # Partition dims into 8 subspaces
    codebook_bits=8,    # 256-symbol alphabet per subspace
    seed=42,            # Reproducibility
)
# ✅ Theoretically sound: finite alphabet → LZ universal coding
```

**ε-Net Sampling (Formal Approximation Guarantees):**

```python
from agent.src.signals import compute_coherence_with_guarantees

# Compute coherence with ε-net guarantees
coh, v_star, metadata = compute_coherence_with_guarantees(
    points=embeddings,       # (n_tokens, d)
    num_directions=100,      # M sampled directions
    num_bins=20,             # B histogram bins
    epsilon=0.1,             # Approximation tolerance
)

print(f"Coherence: {coh}")
print(f"Approximation error: {metadata['approximation_error']:.4f}")
print(f"Covering number N(ε): {metadata['covering_number']}")
print(f"Required M for guarantee: {metadata['required_samples']}")
print(f"Guarantee met: {metadata['guarantee_met']}")

# Mathematical guarantee:
# max(sampled coh) ≥ max(true coh) - L*ε  with prob ≥ 1-δ
# where L ≲ 2√n/B (Lipschitz constant), δ=0.05 (confidence)
```

### 19.4 Operational Procedures

**Calibration Data Collection:**
1. Deploy with `ENABLE_CALIBRATION_LOGGING=true`
2. Collect n_cal ∈ [100, 1000] labeled examples (human review or trusted ground truth)
3. Balance benign vs anomalous (50/50 recommended)
4. Store with timestamps for time-window management

**Recalibration Triggers:**
1. **Time-based:** Weekly recalibration (or per 10k decisions)
2. **Drift-based:** KS test p-value < 0.05
3. **Miscoverage-based:** Empirical rate deviates >50% from target δ

**Multi-Tenant Patterns:**
- Per-tenant calibration sets (isolated quantiles)
- Per-tenant drift detection (separate distributions)
- Global fallback for cold-start tenants (first 100 PCS)

### 19.5 Invariants & Guarantees

**Mathematical:**
- ✅ Miscoverage ≤ δ under exchangeability (Vovk 2005)
- ✅ Product quantization → finite alphabet (Jégou et al. 2011)
- ✅ ε-Net approximation: error ≤ L*ε with prob ≥ 1-δ (Haussler 1995)

**Operational:**
- ✅ Thread-safe (sync.RWMutex in CalibrationSet)
- ✅ FIFO eviction (maxSize parameter)
- ✅ Time-window pruning (automatic old-score removal)
- ✅ Per-tenant isolation (tenantID filtering)

**Backward Compatibility:**
- ✅ Old `compute_coherence()` preserved (Phase 1-11 code works)
- ✅ Old `compute_compressibility()` deprecated but functional
- ✅ All Phase 1-11 tests passing (33 Python + Phase 11 Go tests)

### 19.6 Documentation & References

**Primary Docs:**
- `docs/architecture/ASV_IMPLEMENTATION_STATUS.md` → Implementation status (Week 1-2 complete)
- `docs/architecture/asv_whitepaper_revised.md` → Mathematical foundations
- `docs/architecture/ASV_WHITEPAPER_ASSESSMENT.md` → Publication roadmap

**Code Locations:**
- `backend/internal/conformal/calibration.go` → CalibrationSet, ComputeScore, Predict
- `backend/internal/conformal/drift.go` → DriftDetector, MiscoverageMonitor
- `backend/internal/conformal/calibration_test.go` → 8 comprehensive tests (all passing)
- `agent/src/signals.py` → Product quantization, ε-net sampling

**References:**
- Vovk et al. (2005) - Algorithmic Learning in a Random World
- Lei et al. (2018) - Distribution-free predictive inference for regression
- Angelopoulos & Bates (2023) - Conformal Prediction: A Gentle Introduction
- Jégou et al. (2011) - Product Quantization for Nearest Neighbor Search
- Haussler (1995) - Sphere Packing Numbers for Subsets of the Boolean n-Cube

### 19.7 LLM Collaboration Notes

**When implementing conformal prediction features:**
1. **Always** maintain exchangeability assumption (no feedback loops without partitioning)
2. **Always** use linear interpolation for quantiles (standard split conformal formula)
3. **Always** check n_cal ≥ 100 before computing quantiles (stability requirement)
4. **Never** modify quantile computation without updating tests (golden test vectors)
5. **Never** compress raw floats (use product quantization for theoretical soundness)

**When proposing changes:**
- Cite Angelopoulos & Bates (2023) for conformal prediction modifications
- Cite Jégou et al. (2011) for product quantization changes
- Attach mathematical proof or simulation evidence for new guarantees

