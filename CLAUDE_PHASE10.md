# CLAUDE.md — Audit Remediation Plan for `kakeya` (Claude / claude-code)

**Audience:** `claude-code` (autonomous coder), maintainers
**Objective:** Implement the audit fixes across code, security, performance, CI/CD, Helm, and docs—without breaking Phase 1–8 invariants (verify-before-dedup, WAL-first, idempotency, SLOs).

---

## 0) TL;DR (What to ship)

* **Canonical JSON parity (9dp floats) across Python/Go/TS** + cross-language golden tests
* **Simpler HMAC** (HMAC over payload) + consistent verify
* **Atomic dedup** writes (SETNX / DB unique) + parallel submit test
* **Real VRF verification** (ECVRF) and tests
* **Better RAG overlap** (tokenized Jaccard)
* **Replace ad-hoc caches with LRU (thread-safe)** + cache metrics
* **AuthN hardening** (tenant-bound API key/JWT at gateway)
* **Risk-based routing** (skip heavy ensemble when risk is low)
* **CI/CD hardening** (lint/type/security/perf/chaos; golden vectors)
* **Helm updates** (new toggles/resources/RBAC/egress)
* **Docs refresh** (README, quickstarts, examples, OpenAPI, runbooks sync)

---

## 1) Scope & Non-Goals

**In scope:** code changes, tests, CI, Helm values/RBAC, docs, examples.
**Out of scope (track separately):** external pen-test, formal proofs expansion, net-new features beyond audit fixes.

---

## 2) Repo assumptions (paths)

* Agent SDK & tools (Python): `agent/`
* Verifier & services (Go): `backend/`
* SDKs: `sdk/python/`, `sdk/go/`, `sdk/ts/`
* Operator/CRDs (Go): `operator/`
* Helm: `helm/`
* CI: `.github/workflows/`
* Docs: `docs/`

Adjust paths if different.

---

## 3) Work Packages (WP) — step-by-step tasks for Claude

### WP1 — Canonical JSON parity & Golden Vectors

**Why:** prevent cross-language signature drift.

**Changes**

* Create **canonicalization utilities**:

    * Python: `agent/flk_canonical.py`
    * Go: `backend/pkg/canonical/canonical.go`
    * TS: `sdk/ts/src/canonical.ts`
* Force **floats to 9 decimal places** in the **signature subset** only.
* Build **golden vectors** consumed by all SDKs & backend.

**Tasks**

1. Implement `format_float_9dp(value) -> string` per language.
2. Build `signature_payload(pcs)` using explicit string formatting for floats; stable field order.
3. Add test corpus: `tests/golden/signature/`

    * `input.json` (PCS subset), `payload.bytes`, `sig_hmac_base64.txt`, `sig_ed25519_base64.txt`.
4. Add tests:

    * Python/Go/TS: assert **byte-for-byte payload equality** and equal signatures.
5. Wire into existing signing/verify paths.

**Acceptance**

* All languages produce **identical payload bytes** and signatures for the corpus.
* Backwards-compat note added if signature scheme changes (HMAC simplification in WP2).

**Example (Go)**

```go
// canonical.go
func F9(x float64) string { return strconv.FormatFloat(x, 'f', 9, 64) }
```

---

### WP2 — HMAC simplification (payload, not pre-hash)

**Why:** standard, less complexity.

**Changes**

* Sign: `HMAC(key, canonical_payload_bytes)`
* Verify: the same

**Tasks**

1. Update Python/Go signing and verify functions.
2. Update tests to **expect new HMAC** outputs (golden vectors).
3. Migration note: no data corruption (WORM stays), but signature checks **must** use new scheme. Keep a **compat window** env flag `SIGN_HMAC_MODE=legacy|standard` default `standard`.

**Acceptance**

* Golden tests pass; legacy mode still supported behind flag for rollback.

**Patch (Python)**

```diff
- digest = hashlib.sha256(payload).digest()
- sig = hmac.new(key, digest, hashlib.sha256).digest()
+ sig = hmac.new(key, payload, hashlib.sha256).digest()
```

---

### WP3 — Atomic dedup (first-write-wins under concurrency)

**Why:** prevent dual inserts in race.

**Changes**

* Redis: `SETNX pcs_id result_json` with TTL or
* Postgres: `UNIQUE(pcs_id)` + `INSERT ... ON CONFLICT DO NOTHING`

**Tasks**

1. Implement atomic check-and-set in `backend/internal/dedup/store.go`.
2. Add **parallel submit test** (spawn N=50 goroutines on same payload). Expect **1 write, N-1 hits**.
3. Emit metric: `dedup_first_write_total`, `dedup_duplicate_hits_total`.

**Acceptance**

* Parallel test stable; no double write observed.

---

### WP4 — Real VRF verification (ECVRF)

**Why:** close security gap.

**Changes**

* Integrate ECVRF lib (Go), e.g., `github.com/oasisprotocol/curve25519-voi/ecvrf` or similar.
* Add `vrf_verify(pubkey, alpha, pi) -> beta` and prove seed derivation.

**Tasks**

1. Add VRF verifier in `backend/pkg/crypto/vrf/`.
2. Validate with **known vectors** in tests.
3. Enforce policy: if tenant requires VRF, reject invalid proofs with 401/202-escalation.
4. Document config: `tenant.vrf_required`.

**Acceptance**

* Vector tests pass; invalid VRF requests rejected.

---

### WP5 — RAG overlap improvement (tokenized)

**Why:** robust grounding check.

**Changes**

* Replace char-Jaccard with **word/token Jaccard** + optional stemming/stopwords.

**Tasks**

1. Add tokenizer util (Go): split on Unicode word boundaries; lowercase; stopword filter.
2. Compute Jaccard over shingles (n=2 or 3).
3. Thresholds from config: `rag.overlap_min=0.35`.
4. Unit tests for edge cases (mixed punctuation, URLs).

**Acceptance**

* Tests pass; false positives down on synthetic cases.

---

### WP6 — Caches: thread-safe LRU + metrics

**Why:** stability & bounded memory.

**Changes**

* Replace ad-hoc caches with **LRU w/ TTL**:

    * Micro-vote embeddings cache
    * HRS feature store (if not already thread-safe)

**Tasks**

1. Use `hashicorp/golang-lru/v2` or equivalent.
2. Wrap with mutex if needed; add TTL sweep.
3. Metrics: `cache_hits_total`, `cache_misses_total`, `cache_size`.

**Acceptance**

* No data races under `go test -race`; hit rate reported.

---

### WP7 — AuthN hardening (tenant-bound API keys/JWT)

**Why:** defense-in-depth beyond X-Tenant-Id header.

**Changes**

* Introduce **tenant API key / JWT**. Gateway validates token and **binds tenant_id** claim.

**Tasks**

1. Add `auth/` middleware in backend: verify gateway header `X-Auth-Verified: true` and `X-Tenant-Id` from gateway only.
2. Provide example **Envoy/NGINX** config in `deploy/gateway/`.
3. Docs: onboarding flow for tenants; rotation SOP.

**Acceptance**

* Requests without verified identity **rejected**; signature checks remain.

---

### WP8 — Risk-based routing (save CPU)

**Why:** avoid heavy ensemble when risk is low.

**Changes**

* If HRS `risk < low_threshold` & `ci_narrow == true`: **skip ensemble**, fast-path accept (still WORM).

**Tasks**

1. Add policy flag `policy.risk.skip_ensemble_below`.
2. Implement guard in verifier hot-path.
3. Add E2E: mix of low/high risk traffic; ensure p95 improves while containment stays.

**Acceptance**

* p95 latency ↓ with no regression in escaped hallucinations (A/B in tests).

---

### WP9 — CI/CD: lint, type, security, perf, chaos, golden

**Why:** catch regressions early.

**Changes & Tasks**

1. **Python**: `ruff`/`flake8`, `mypy`, `bandit`
2. **Go**: `golangci-lint`, `go vet`, `go test -race`, `gosec`
3. **TS**: `tsc --noEmit`, `eslint`
4. **Supply chain**: `trivy` scan images & deps
5. **Perf**: run `k6` thresholds (p95, p99, error rate) on PR label `perf`
6. **Chaos**: nightly job to inject WAL lag / shard loss in staging
7. **Golden**: run cross-language signature suite on every PR

**Acceptance**

* CI green gates; artifacts (k6 HTML, chaos logs) attached.

---

### WP10 — Helm values, RBAC, egress

**Why:** production toggles & least privilege.

**Changes**

* Add toggles: `vrf.required`, `risk.skipEnsembleBelow`, `ensemble.enabled`, cache sizes, SIEM egress.
* Update **operator RBAC** for CRDs only.
* Increase default backend resources (CPU for ensemble/HRS).

**Tasks**

1. Edit `helm/values.yaml` + README.
2. Add NetworkPolicies:

    * allow SIEM egress
    * operator ↔ API server
3. PodSecurityContext: run as non-root.

**Acceptance**

* `helm template` validates; `helm lint` passes; deploy tested in kind.

---

### WP11 — Docs refresh (+ examples & OpenAPI)

**Why:** clarity for devs/investors.

**Changes & Tasks**

1. **README**: new intro, feature list, KPIs, quickstart.
2. **Examples**: `examples/basic/` E2E:

    * Start dev stack
    * Build PCS & sign
    * Submit & inspect verified result
3. **OpenAPI**: ensure spec current; publish in `docs/api/openapi.yaml`
4. **Runbooks sync**: update for new metrics/flags (VRF, risk skip)
5. **Diagrams**: Phase-8 architecture overview (`docs/architecture/overview.md`)

**Acceptance**

* A new dev can run end-to-end in <15 min following README.

---

### WP12 — Fuzz & input validation

**Why:** robustness against weird inputs.

**Tasks**

1. Add fuzz tests to JSON/PCS parsers (Go 1.18+ fuzzing).
2. Validate fields (epoch ≥ 0, finite floats, bounded sizes).
3. Unit tests for NaN/Inf rejection.

**Acceptance**

* Fuzz suite finds no panics; invalid inputs rejected 400.

---

## 4) Pull Request breakdown (suggested)

* **PR-001**: Canonical JSON + golden vectors (all langs)
* **PR-002**: HMAC simplification + compat flag
* **PR-003**: Atomic dedup + parallel test + metrics
* **PR-004**: VRF verification + vectors + docs
* **PR-005**: RAG tokenized overlap + tests
* **PR-006**: LRU caches + metrics + race tests
* **PR-007**: AuthN hardening (gateway sample, backend middleware)
* **PR-008**: Risk-based routing + E2E A/B
* **PR-009**: CI/CD hardening (lint, type, security, perf, chaos)
* **PR-010**: Helm values/RBAC/egress/resources
* **PR-011**: Docs + examples + OpenAPI + diagrams
* **PR-012**: Fuzz & input validation

Each PR: unit + e2e tests, dashboards/alerts updated if needed, rollback plan.

---

## 5) Acceptance Criteria (global)

* Cross-language **payload bytes identical**; golden vectors pass in Python/Go/TS.
* HMAC standard mode live; legacy toggle present.
* Dedup first-write-wins under concurrent submits; zero duplicate writes.
* VRF proof verify enforced when enabled.
* Ensemble RAG check uses tokenized overlap; tests green.
* No data races; caches bounded; cache hit metrics visible.
* Auth path rejects spoofed tenant; gateway integration doc exists.
* Risk-based skip reduces p95 w/o containment regression (in test env).
* CI includes lint/type/security/perf/chaos and is GREEN.
* Helm renders with new toggles; operator RBAC least privilege.
* README + examples enable end-to-end in <15 min.
* Fuzz tests find no panics; invalid inputs validated.

---

## 6) CI/CD: new/updated workflows

* `.github/workflows/ci.yml`: lint/type/unit, golden vectors
* `.github/workflows/security.yml`: gosec/bandit/trivy
* `.github/workflows/perf.yml`: k6 (conditional)
* `.github/workflows/chaos-nightly.yml`: staging chaos suite
* `.github/workflows/docs.yml`: OpenAPI validate + publish

---

## 7) Timeline (aggressive, ~2–3 weeks)

* **Week 1:** PR-001…004 (canonical, HMAC, dedup, VRF)
* **Week 2:** PR-005…008 (RAG, caches, AuthN, risk skip)
* **Week 3:** PR-009…012 (CI/CD, Helm, docs/examples/OpenAPI, fuzz)

Parallelize where safe; land in small, reviewable PRs.

---

## 8) Risks & Mitigations

* **Signature drift** → golden vectors on every PR; lockstep SDK updates.
* **Latency creep** → risk-skip policy; perf gates in CI.
* **Over-blocking (VRF/RAG)** → canary flags; 202-escalation not 4xx initially.
* **Operator/RBAC misconfig** → kind e2e + `helm template` checks.
* **Cache races** → `go test -race` mandatory gate.

---

## 9) Developer notes for Claude

* Prefer **small PRs** with exhaustive tests.
* When adding configs, **document in `helm/values.yaml`** and **README**.
* For new metrics, **extend Grafana JSON** under `docs/observability/`.
* Keep **verify → dedup** ordering and **WAL-first** writes intact.
* Always add **rollback notes** in PR descriptions.

---

## 10) Quick command snippets

**Run tests (Go)**

```bash
go test ./... -race
```

**Run golden vectors (Python)**

```bash
pytest tests/golden -q
```

**Lint & type (Python)**

```bash
ruff check .
mypy agent/ sdk/python/
```

**Helm lint**

```bash
helm lint helm/
```

**k6 perf (example)**

```bash
k6 run tests/perf/verify.js
```

---

**End of CLAUDE.md**
