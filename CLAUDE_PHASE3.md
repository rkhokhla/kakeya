# CLAUDE_PHASE3.md — Multi-Tenant Scale, Governance, and Adversarial Robustness

> **Audience:** `claude-code`, maintainers, SRE, security, and data governance
> **Goal:** Build on **Phase 2** (E2E tests, perf baselines, prod Helm, alerts) to deliver **multi-tenant, multi-region, policy-governed** operations with auditable lineage, formalized contracts, privacy controls, and adversarial robustness—without breaking Phase 1/2 invariants (canonical signing, verify-before-dedup, WAL safety, SLOs).

---

## 0) TL;DR (What Phase 3 delivers)

* ✅ **Multi-tenant** isolation & quotas: per-tenant keys, limits, budgets, dashboards
* ✅ **Multi-region & disaster recovery**: active-active ingest, CRR for WAL & dedup state, RTO/RPO targets
* ✅ **Governance & lineage**: immutable **audit trail** (WORM), PCS lineage graph, tamper-evident logs
* ✅ **Policy DSL → runtime**: compile-time validation, versioned policies, safe rollouts & feature flags
* ✅ **Privacy**: dataset classification, PII scanners at edges, redaction gates, optional **differential privacy** for metrics
* ✅ **Adversarial robustness**: VRF-seeded direction sampling, input sanity guards, anomaly scoring, rate-limit hardening
* ✅ **SDKs & contracts**: stable client SDKs (Python/Go/TypeScript), OpenAPI, backward-compatible schema evolution
* ✅ **Sustained throughput**: sharded dedup, Redis→Postgres→(optional) tiered store, async pipelines for cold-path audits

> All of the above **extends** Phase 2’s Helm/alerts/perf gates; do **not** weaken: signature-before-dedup, idempotency key, and SLO alerts.

---

## 1) Scope & Non-Goals

### In (Phase 3)

1. Tenant model (isolation, quotas, keys, dashboards)
2. Multi-region topology & DR (failover, replication, chaos drills)
3. Auditability & lineage (WORM logs, Merkle-anchoring, query tooling)
4. Policy DSL pipeline (validate → version → promote/rollback)
5. Privacy & security posture (PII gates, DP opt-in, key lifecycle)
6. Adversarial defenses (VRF, sanity constraints, anomaly scoring)
7. API/SDK contracts (OpenAPI, clients, semver policy)
8. Throughput scaling (sharding, tiered dedup, async audits)

### Out (future work)

* Formal proof artifacts beyond safety lemmas; deep ML fairness research; billing/monetization service (prepare hooks only)

---

## 2) Deliverables (Definition-of-Done)

1. **Tenant isolation**

    * Per-tenant credentials & signing keys; tenant-scoped quotas/rate limits; tenant label in metrics & logs
    * Per-tenant **SLO dashboards** & alert routes
2. **Geo & DR**

    * Dual-region deploy with health-based routing; CRR for Inbox/WAL & dedup store; RTO/RPO docs & drills
3. **Audit trail & lineage**

    * Append-only WORM log of PCS & verifier outcomes; lineage graph (PCS → verify params → outcome → policy version)
    * Tamper-evidence: periodic Merkle anchoring to an external, append-only medium
4. **Policy DSL to runtime**

    * Parser → static checks → signed, versioned bundle; canary rollout with feature flags; instant rollback
    * Policy registry with audit history and diff views
5. **Privacy**

    * Edge PII scanners & redaction; dataset classification labels; optional **DP** noise on aggregate metrics
6. **Adversarial robustness**

    * **VRF-seeded** direction sampling; scale/`N_j` sanity checks; request anomaly scoring & defense actions
7. **SDKs & OpenAPI**

    * OpenAPI spec (versioned); SDKs for Python/Go/TypeScript; **golden tests** for signature & canonicalization
8. **Throughput**

    * Sharded dedup stores (consistent hashing); write-through cache; async audit pipeline for heavy checks

> Keep Phase 2 gates (perf/Helm/alerts) green while adding the above.

---

## 3) Work Packages & Tasks

### WP1 — Multi-Tenant Core

* [ ] **Tenant model**: `tenant_id`, per-tenant keys (HMAC/Ed25519), quotas (tokens/s), policies
* [ ] **Isolation**: namespace metrics & logs; NetworkPolicies scoped by namespace; secret per tenant
* [ ] **Per-tenant SLOs**: dashboard templating; alert routes mapping tenant→pager/slack

### WP2 — Multi-Region & DR

* [ ] **Topology**: active-active ingest; DNS/Geo-LB; region labels in metrics
* [ ] **State**: cross-region WAL replication; dedup store replication/CRR; **idempotent replay** validated
* [ ] **DR drills**: failover runbooks; quarterly game-days with success criteria

### WP3 — Auditability & Lineage

* [ ] **WORM log**: append-only, time-boxed segments; retention policy & query tool
* [ ] **Merkle anchoring**: periodic root of segment to external anchor; integrity check CLI
* [ ] **Lineage graph**: store `(pcs_id, policy_version, verify_params_hash, outcome)`; UI/CLI to query lineage

### WP4 — Policy DSL → Runtime

* [ ] **Static checks**: bounds (0≤coh★≤1+ε), weight normalization, budget clamp; no dangerous ops
* [ ] **Versioned bundles**: signed policy artifacts; registry with promotion gates (dev→canary→prod)
* [ ] **Feature flags**: flip per-tenant/percent; automatic rollback on SLO burn

### WP5 — Privacy & Security

* [ ] **PII gates**: edge scanners; structured redaction; deny-by-default on detection
* [ ] **DP option**: add Laplace/Gaussian noise for exported aggregates; toggle per tenant
* [ ] **Key lifecycle**: rotation schedule; multi-key verify window; key escrow SOPs

### WP6 — Adversarial Robustness

* [ ] **VRF seeding**: verify VRF proofs from agents; seed projection sampling; reject if invalid
* [ ] **Sanity checks**: monotonic `N_j` guards; scale ranges; tolerance fences; penalty → 202 + audit
* [ ] **Anomaly scoring**: rate spikes, signature failures, cohort deviations; throttle/escalate

### WP7 — SDKs & API Contracts

* [ ] **OpenAPI**: describe `/v1/pcs/submit`, `/v1/verify/…`, `/v1/tenants/*` admin endpoints (if any)
* [ ] **SDKs**: Python/Go/TS clients with canonical signing helpers; **golden tests** vs server
* [ ] **SemVer & deprecation**: headers for `schema-version`, sunset dates, compat matrix

### WP8 — Throughput & Storage

* [ ] **Sharded dedup**: consistent hashing; migration tool; cross-shard query API
* [ ] **Tiering**: Redis (hot) → Postgres (warm) → object storage (cold) for audit; TTL by tenant
* [ ] **Async audit**: queue + workers; backpressure; observable SLAs

---

## 4) Acceptance Criteria

* **Isolation**: noisy neighbor cannot degrade others; quotas enforced per tenant
* **DR**: region failover < target RTO; **no data loss** beyond RPO; replay is idempotent
* **Audit**: WORM segments verifiably intact; lineage query returns complete chain for any `pcs_id`
* **Policies**: any policy change is signed, versioned, canaried, and rollback-able in < X min
* **Privacy**: PII detected → blocked or redacted; DP toggle works & documented
* **Robustness**: invalid VRF or sanity breach → rejected/202; anomaly actions visible in metrics
* **SDKs**: pass golden tests; interop with Phase 1 canonicalization rules
* **Perf**: p95 within Phase 2 SLOs under shard scale; alerts green under steady & burst loads

---

## 5) CI/CD Plan (Phase 3 jobs)

* `unit`: keep Phase 1/2 suites green; add DSL static checks & SDK tests
* `e2e-mt`: multi-tenant matrix (2–3 tenants, distinct keys, policies)
* `geo-dr`: chaos test simulating region outage + replay validation
* `sec-privacy`: PII scanner tests; DP metric sanity; secret scans
* `dsl-gate`: compile policies → sign → promote canary; rollback on SLO burn (simulated)
* `sdk-compat`: run golden vectors across Python/Go/TS
* `perf-sharded`: k6 against sharded dedup; threshold gates preserved from Phase 2

---

## 6) Milestones & Suggested Timeline

1. **M1 (Week 1–2)**: WP1 (tenants), WP7 (OpenAPI + SDK skeletons)
2. **M2 (Week 3–4)**: WP2 (geo/DR topology), WP8 (sharded dedup prototype)
3. **M3 (Week 5–6)**: WP3 (audit/lineage), WP4 (policy pipeline & flags)
4. **M4 (Week 7–8)**: WP5 (privacy), WP6 (adversarial), CI hardening, docs & runbooks

---

## 7) Implementation Notes & Interfaces

### 7.1 Tenant headers & quotas

* **Headers**: `X-Tenant-Id`, `X-PCS-Schema-Version` (server validates)
* **Quotas**: token bucket per tenant + burst caps; 429 with `Retry-After` (keeps Phase 2 backoff logic)

### 7.2 VRF-seeded sampling (agent → verifier)

* Agent includes `vrf_proof`, `vrf_output`; verifier checks proof against tenant pubkey; derive RNG seed → direction set
* On failure: 401 or 202 with `escalated: reason=vrf_invalid`

### 7.3 Policy bundle format

* TAR/JSON with `policy.json`, `checksums.txt`, `signature.sig`
* Registry: `POST /v1/policies`, `GET /v1/policies/:version`, `PATCH /v1/policies/promote`

### 7.4 WORM & anchoring

* Segment files `YYYY/MM/DD/HHiiss.jsonl` (append-only); compute segment Merkle root; periodically anchor root externally
* CLI: `worm verify --from 2025-01-01 --to 2025-01-31`

---

## 8) Risks & Mitigations

* **Complexity creep** → phase gates & feature flags; strong defaults
* **Key sprawl** → per-tenant key registry + rotation SOP; automation tests
* **False positives (PII/anomaly)** → staged “report-only” mode before blocking
* **Cross-region consistency** → idempotent replay + eventual consistency docs
* **Policy misconfig** → static checks + canary + auto-rollback on SLO burn

---

## 9) Runbooks to add (`/docs/runbooks/`)

* `tenant-slo-breach.md`, `region-failover.md`, `policy-rollback.md`, `vrf-invalid-surge.md`, `pii-detection.md`, `worm-verify.md`

---

## 10) PR Checklist (each WP)

* [ ] Does not break Phase 1/2 invariants (signature→dedup; WAL first; SLOs)
* [ ] Tests: unit + e2e + chaos for the WP
* [ ] Dashboards & alerts updated; runbooks linked
* [ ] Secrets via SOPS/age; no plaintext in repo/CI
* [ ] Docs: API (OpenAPI), ops, governance, and migration notes
* [ ] Rollback plan documented and tested

---

## 11) References

* **Phase 2 Plan** — E2E, perf baselines, prod Helm, alerts/chaos; these remain the operational floor for Phase 3.

---

**End of PHASE 3 plan**
