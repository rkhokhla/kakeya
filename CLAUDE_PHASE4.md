# CLAUDE_PHASE4.md — Multi-Region Active-Active, Sharded/Tiered Storage, and SDK Parity

> **Audience:** `claude-code`, maintainers, SRE, security, data governance
> **Goal:** Implement **Phase 4** capabilities prioritized in the Phase 3 report: **multi-region active-active topology, cross-region WAL replication, sharded & tiered dedup storage, async audit pipeline, SDK parity (Go/TS), and expanded E2E/chaos tests**—while preserving Phase 1/2 invariants and Phase 3 controls.

---

## 0) TL;DR (What Phase 4 delivers)

* ✅ **Active-Active Multi-Region** ingest with health-based global routing, **CRR** (cross-region replication) for Inbox/WAL & dedup state, documented **RTO/RPO** and disaster drills.
* ✅ **Sharded Dedup** with consistent hashing + live migration tooling; **tiered storage** (Redis→Postgres→Object) with TTL policies and observability.
* ✅ **Async Audit Pipeline** (queue + workers) for cold-path checks and anchoring throughput.
* ✅ **SDK Parity**: production-ready **Go** and **TypeScript** SDKs with golden-test compatibility against Phase 1 canonicalization & Phase 3 API.
* ✅ **E2E/Chaos/Geo-DR tests** and perf baselines at shard & region scale.

Non-goals: formal proofs, advanced ML anomaly models, blockchain anchoring (tracked as long-term).

---

## 1) Scope (from Phase 3 → Phase 4)

**Must ship (blocked in Phase 3):**

1. Multi-region active-active + CRR for WAL & dedup store; chaos drills and runbooks.
2. Sharded dedup + migration tool; tiered storage + TTL automation; cross-shard query.
3. Async audit workers for heavy checks & anchoring.
4. SDK parity (Go, TypeScript) + golden tests; expanded E2E suites.

**Should ship (medium priority from Phase 3):**

* Differential Privacy toggle for aggregate metrics; advanced canary percentage rollout; perf optimizations (async WORM, batch metrics).

**Keep intact:** Phase 1/2 invariants (WAL before parse, **signature-before-dedup**, idempotency), Phase 3 security/privacy (VRF, PII gates, sanity checks, WORM, policy DSL).

---

## 2) Work Packages (WPs)

### WP1 — Multi-Region Active-Active & DR

**Objectives**

* Deploy **two regions** (e.g., `eu-west` + `us-east`) as active-active; **global DNS/GSLB** with health probes.
* **CRR for Inbox/WAL** (e.g., S3/GCS cross-region) and **dedup state** (Redis/Postgres with async replication or dual-write chaperone).
* Enforce **idempotent replay** across regions (pcs_id first-write wins).
* Define **RTO/RPO** targets and run quarterly **DR game-days**.

**Deliverables**

* Region-aware Helm values (Region IDs, seed lists, replication endpoints).
* WAL writer → **append + fsync + ship** (eventual CRR); WAL reader in failover region → **replay idempotently**.
* **Geo-tags in metrics** and dashboards panels by region.
* **Runbooks**: `region-failover.md`, `geo-split-brain.md`.

**Acceptance**

* Forced outage of one region keeps p95 < SLO and **no data loss beyond RPO**.
* Replay produces **byte-for-byte** identical verified outcomes and dedup state.

---

### WP2 — Sharded Dedup Store + Live Migration

**Objectives**

* Introduce **consistent hashing** on `pcs_id` across N shards (Redis/Postgres).
* Provide **migration tool** to rehash & rebalance with zero downtime; throttled copy + cutover.
* **Cross-shard query** API for ops (read-only) & observability.

**Deliverables**

* Shard router library (Go) with stable hash ring; shard health & lag metrics.
* `dedup-migrate` CLI: plan → copy → verify → cutover; resumable checkpoints.
* Backfill job for **existing keys**; online consistency verifier.

**Acceptance**

* Migration of 1→3 shards without write errors; dedup hit ratio stable; **no cache stampedes**.

---

### WP3 — Tiered Storage (Hot→Warm→Cold)

**Objectives**

* **Hot**: Redis (<1h TTL), **Warm**: Postgres (<7d), **Cold**: Object store (>7d) with lifecycle.
* Automated **TTL policies**; efficient promote/demote between tiers.

**Deliverables**

* Storage manager with policy DSL: `class: hot|warm|cold`, `ttl`, `tenant overrides`.
* Background compactor & mover; metrics on tier sizes, promotions, evictions.
* Cost & latency dashboard panels.

**Acceptance**

* Tier transitions preserve idempotency & API latency targets; cold reads audited via WORM lineage.

---

### WP4 — Async Audit & Anchoring Pipeline

**Objectives**

* Offload heavy checks (e.g., extended anomaly analysis, batch anchoring) to a **queue + worker** model.
* Maintain WORM append sync for minimal fields; extended artifacts via async pipeline.

**Deliverables**

* Queue schema (`audit.enqueue`, `audit.work`), backoff & DLQ; worker autoscaling (HPA).
* Anchoring batcher: compute segment/root & push to external anchor; retry & attestations.

**Acceptance**

* Ingest path stable under load spikes (no p95 regression); async backlog drains within SLO.

---

### WP5 — SDK Parity (Go & TypeScript)

**Objectives**

* Implement **Go** and **TypeScript** SDKs mirroring Python SDK features: canonical signing (Phase 1), retries, multi-tenant headers, response models.
* Add **golden compatibility tests** SDK↔backend for signature bytes & responses.

**Deliverables**

* `sdk/go` and `sdk/ts` packages; CI publishing (Go module, npm).
* Conformance suite using Phase 1/3 golden vectors & OpenAPI.
* Examples & quickstarts.

**Acceptance**

* All SDKs pass golden tests and **interoperate** with Phase 3 server & policies.

---

### WP6 — E2E, Chaos, and Geo-DR Testing

**Objectives**

* Extend E2E to **multi-tenant × multi-region** matrices; simulate failovers, partition, clock skew.
* Add **chaos** injections: shard loss, WAL lag, CRR delays, hot-tier failures.

**Deliverables**

* `tests/e2e-geo/…` with compose/kind multi-cluster orchestration.
* k6 geo profiles; SLO burn-rate alert tests.
* Synthetic **geo replay** assertions (no divergence).

**Acceptance**

* E2E & chaos jobs **green**; artifacts published (logs, metrics, k6 reports).

---

### WP7 — DP Metrics & Advanced Canary (Optional in P4)

**Objectives**

* Add **Differential Privacy** option for exported aggregates; per-tenant ε/δ budgets.
* Introduce **percentage-based canary** promotion with auto-rollback on SLO burn.

**Deliverables**

* DP noise injectors (Laplace/Gaussian); audit of privacy budget use.
* Policy registry: staged `% rollout` with guardrails; runbook updates.

**Acceptance**

* DP’d metrics meet utility tests; canary promotion safely escalates 10→50→100% with rollback proof.

---

## 3) Technical Design Notes

* **CRR for WAL**: keep existing **append + fsync** semantics (Phase 3), then **ship** segments to remote region/bucket with **order guarantee** per segment; remote reader applies idempotent replay before dedup.
* **Sharded Dedup**: consistent hash ring with virtual nodes; write path chooses shard; **first-write wins** invariant preserved; shard health gates writes. Migration tool computes new ring, **dual-writes during cutover**, reconciles via read-repair.
* **Tiered Store**: driver abstraction with **Get→maybe promote**; **Warm** stores authoritative copy for TTL window; **Cold** is authoritative beyond; all reads audit via WORM lineage entry (Phase 3).
* **Async Audit**: immediate WORM entry persists minimal outcome; heavy tasks (e.g., segment Merkle anchoring) done async with durable retries; alerts on backlog size/time.
* **SDKs**: reuse canonical 8-field subset & 9-dp rounding; ensure **exact byte equivalence** of signature payloads across languages; wire multi-tenant headers (`X-Tenant-Id`).

---

## 4) Definition of Done (per WP)

* **No regressions** of Phase 1/2/3 invariants; all suites pass.
* **Geo failover**: verified RTO/RPO; replay correctness; dashboards & alerts.
* **Shard migration**: safe cutover, correctness proofs, perf reports.
* **Tiering**: policy-driven TTLs, measurable cost/latency impact.
* **Async audit**: ingestion latency stable; backlog SLO green.
* **SDK parity**: golden tests across Python/Go/TS.

---

## 5) CI/CD Plan

Pipelines extend Phase 2/3:

* `unit`: existing + shard router, CRR shipper/reader, tiered manager, queue workers.
* `e2e-geo`: multi-region topology spin-up; failover tests; replay diff checks.
* `chaos`: shard kill, WAL lag, CRR delay, cold store outage.
* `perf-sharded`: k6 against 1→3 shards; publish p50/p95/p99.
* `sdk-compat`: golden vectors for Python/Go/TS; signature byte-for-byte checks.
* `helm-lint/kind`: cluster per region; NetworkPolicies; NOTES verification.

Artifacts: k6 HTML, shard migration logs, replay audits, policy bundles, Grafana JSON, Prometheus rules.

---

## 6) Risks & Mitigations

* **Split-brain / divergence** across regions → single-writer per pcs_id via idempotent dedup; WAL total order per segment; **reconciliation checks** and lineage audits.
* **Migration errors** during sharding → dual-write & checksums; incremental cutover; automatic rollback.
* **Tiering latency spikes** → lazy promote with cache; budgeted cold-read limits; alerting on cold-hit rate.
* **SDK drift** → golden tests pinned to OpenAPI & canonicalization; semver gates.
* **CRR lag** → backpressure + alerting on replication delay; RPO policy docs and tenant comms.

---

## 7) Milestones & Timeline (suggested 8–10 weeks)

1. **M1 (Weeks 1–2)**: WP5 SDK parity; WP2 shard router prototype; CI golden tests.
2. **M2 (Weeks 3–4)**: WP1 geo topology (dev); WP2 migration tooling; WP6 e2e-geo harness.
3. **M3 (Weeks 5–6)**: WP3 tiered storage; WP4 async audit workers; perf gates.
4. **M4 (Weeks 7–8)**: DR drills, chaos suite, runbooks; hardening & docs; optional WP7 DP/canary.

---

## 8) Implementation Skeletons

### 8.1 Shard Router (Go)

```go
type Shard struct{ Name, Addr string }
type Ring struct{ vnodes int; shards []Shard; hash func([]byte) uint32 }

func (r *Ring) Pick(key []byte) Shard {
  h := r.hash(key)
  idx := int(h) % (len(r.shards)*r.vnodes)
  return r.shards[idx/r.vnodes]
}
```

### 8.2 WAL CRR Shipper

```go
// tail segments → upload → record watermark
// remote reader applies in order; idempotent replay on pcs_id
```

### 8.3 Tiered Storage Policy

```yaml
tiering:
  default:
    hot_ttl: 3600
    warm_ttl: 604800
    cold_bucket: "flk-cold"
  tenants:
    tenantA: { hot_ttl: 7200 }
```

### 8.4 SDK Golden Test (TS)

```ts
// build signature payload → sha256 → hmac → base64; compare to golden.json
```

---

## 9) Runbooks to Add

* `geo-failover.md`, `geo-replay-verify.md`, `shard-migration.md`, `tier-cold-hot-miss.md`, `audit-backlog.md`.

---

## 10) PR Checklist (each WP)

* [ ] Preserves Phase 1/2/3 invariants; adds tests proving it.
* [ ] Region/Shard metrics and dashboards updated; alerts wired.
* [ ] Migration/replay tools documented & idempotent.
* [ ] SDKs pass golden tests; examples added.
* [ ] Runbooks and SLOs updated (RTO/RPO, backlog SLO).

---

## References

* **Phase 3 Implementation Report** — scope finished vs deferred (multi-region DR, sharded/tiered storage, async audit, SDK parity), and testing/ops baselines used here.

**End of PHASE 4 plan**
