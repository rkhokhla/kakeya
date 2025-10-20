# CLAUDE_PHASE5.md — CRR Implementation, Cold-Tier Integration, Async Audit Workers, and Automated Geo Consistency

> **Audience:** `claude-code`, maintainers, SRE, security, data governance
> **Goal:** Turn Phase-4’s architecture + runbooks into **fully-implemented** production features: **Cross-Region Replication (CRR) shipper/reader**, **cold-tier drivers + background demotion + compression**, **async audit workers + batch anchoring**, **dedup migration CLI + cross-shard ops API**, and **automated geo-divergence detection**—while preserving all Phase 1–4 invariants and SLOs. The scope is derived directly from Phase 4’s “Known Limitations” and “Phase 5 Roadmap.”

---

## 0) TL;DR (What Phase 5 ships)

* ✅ **WAL CRR implemented**: durable shipper (append→fsync→ship), ordered reader with **idempotent replay** across regions; **lag metrics + alerts**; automated **geo-divergence detector**.
* ✅ **Tiering completion**: production **S3/GCS cold drivers**, **background demotion** workers, and **(optional) compression**; per-tenant TTL policies operational.
* ✅ **Async audit pipeline**: queue + workers with HPA, **batch anchoring** & external attestation, DLQ + runbooks wired.
* ✅ **Sharded dedup ops**: `dedup-migrate` CLI (plan→copy→dual-write→cutover→cleanup) and **cross-shard read API** for ops.
* ✅ **SDK + tests hardening**: golden-vector compat for Py/Go/TS, publish pipelines, geo/E2E/chaos suites expanded per Phase-4 guidance.
* ✅ **Optional**: Differential-privacy (DP) export path & advanced canary rollout.

> Phase-4 delivered sharding, tiering manager, SDK parity, and extensive runbooks; Phase-5 completes deferred implementations and automates geo reliability. Keep Phase 1/2/3/4 invariants (signature-before-dedup, WAL-first, idempotency, SLOs).

---

## 1) Scope & Non-Goals

### In (Phase-5)

1. **Implement WAL CRR shipper/reader** + geo divergence detection, alerts, and drills (automation of Phase-4 runbooks).
2. **Cold tier drivers** (S3/GCS), **background demotion**, and **compression** in tiered storage manager.
3. **Async audit queue** + **workers** + **batch anchoring** + external attestations; DLQ management.
4. **`dedup-migrate` CLI** + **cross-shard query API** to operationalize consistent-hash sharding in prod.
5. **Test & publish pipelines** for Go/TS SDKs; golden tests across SDKs; E2E geo-DR + chaos suites.
6. **Optional**: DP metrics export; advanced percentage canary.

### Out (Phase-6+)

* Automated blockchain anchoring and formal proofs beyond safety lemmas (tracked in long-term research).

---

## 2) Deliverables (Definition of Done)

1. **CRR Shipper/Reader (WAL)**

    * Shipper tails **append-only** segments, enforces per-segment order, persists watermark; Reader replays idempotently (**first-write wins**) into verify→dedup path.
    * Metrics: `wal_crr_lag_seconds`, `wal_ship_errors_total`, `wal_replay_applied_total`; Alerts: **WalReplicationLag**, **GeoRegionDown**; dashboards by region.
    * **Automated geo-divergence detection** (dedup key count & sample diff) with runbook link (split-brain).

2. **Tiered Storage Completion**

    * **Cold tier drivers** (S3/GCS) wired into manager; **background demotion**; **optional compression** (e.g., zstd) in hot/warm tiers.
    * Per-tenant TTL enforcement; cost/latency panels; lifecycle policies operational (delete/Glacier).

3. **Async Audit & Anchoring**

    * **Queue schema** (task types: enrich_worm, anchor_batch, attest_external) + workers w/ HPA; **batch anchoring** implemented; DLQ with tooling and runbooks.
    * Metrics/alerts: backlog size/age, DLQ overflow, task failure rate; backlog SLO (99% < 1h) green.

4. **Sharded Dedup Operations**

    * **`dedup-migrate` CLI** (plan→copy→verify→dual-write→cutover→cleanup) per Phase-4 runbook; resumable checkpoints.
    * **Cross-shard read API** for ops & diagnostics; shard health & distribution panels.

5. **SDK + Tests**

    * **Golden vectors** validated across Python/Go/TS; **publish** Go module & npm package; CI badges & examples updated.
    * **E2E geo-DR** suite + **chaos** (shard kill, WAL lag, CRR delay, cold outage) pass with artifacts.

6. **Optional Privacy/Canary**

    * **DP** injectors (Laplace/Gaussian) for aggregates w/ ε/δ budgets; **advanced canary** percent rollout & auto-rollback.

---

## 3) Work Packages & Tasks

### WP1 — WAL CRR Implementation & Geo Consistency

* [ ] **Shipper**: tail segment dir → sign/manifest → upload (S3/GCS CRR) → watermark persisted; retry with exponential backoff.
* [ ] **Reader**: ordered apply with **idempotent replay** guard; verifies Phase-1 order: **verify → (accept) → dedup**.
* [ ] **Divergence detector**: compare dedup key counts & sample keys across regions; raise **GeoDedupDivergence** alert; runbook auto-link (split-brain).
* [ ] **Drills**: codify failover + replay scenarios from runbooks.

### WP2 — Tiering: Cold Integration, Demotion, Compression

* [ ] **Drivers**: S3/GCS `Get/Set/Exists/Delete` with server-side encryption; lifecycle policies applied.
* [ ] **Background demoter**: hot→warm→cold on TTL expiry; **lazy promote** preserved (Phase-4 behavior).
* [ ] **Compression (opt-in)**: compress warm or hot entries; measure latency/CPU; fallback on error.

### WP3 — Async Audit Queue & Batch Anchoring

* [ ] **Queue + workers**: task schema finalized; at-least-once processing; idempotency checks.
* [ ] **Batch anchoring**: build segment/Merkle roots; write external attestation; retries + DLQ; attest reports.
* [ ] **SLOs & alerts**: backlog age, DLQ size, task p95; runbooks wired (audit-backlog).

### WP4 — Sharded Dedup: Migration CLI + Cross-Shard API

* [ ] **`dedup-migrate`**: `plan`, `copy`, `verify`, `dual-write`, `cutover`, `cleanup`; throttling; rollback plan.
* [ ] **Ops API**: read-only cross-shard queries (sample keys, distribution, health, lag).

### WP5 — SDK Publish & Compat

* [ ] **Golden tests**: server-side reference vectors; CI checks Py/Go/TS equality.
* [ ] **Publish**: Go module + npm release; docs/quickstarts; semver policy; deprecation headers.

### WP6 — E2E Geo-DR & Chaos

* [ ] **Harness**: multi-region kind/compose; synthetic traffic; geo replay assertions.
* [ ] **Chaos**: shard loss, WAL lag injection, CRR delay, cold bucket deny; SLO burn tests.

### WP7 — (Optional) DP & Advanced Canary

* [ ] **DP**: ε/δ budgets per tenant; utility checks; audit trail of budget spend.
* [ ] **Canary %**: progressive 10→50→100 with auto-rollback on SLO burn; policy registry integration.

---

## 4) Acceptance Criteria

* **CRR**: `wal_crr_lag_seconds` within target; forced region outage → **no data loss beyond RPO**; replay yields identical dedup state; divergence detector alerts with actionable runbook.
* **Tiering**: cold reads functional; demotion/promotion metrics stable; compression does not violate latency SLOs; cost dashboards reflect lifecycle wins.
* **Async audit**: backlog SLO (99% < 1h) met under 2× normal ingest; DLQ managed with runbook.
* **Sharded ops**: `dedup-migrate` runs 1→N without downtime; dedup hit ratio recovers within runbook bounds.
* **SDKs**: golden vectors match across Py/Go/TS; publish jobs succeed; examples pass.
* **Geo-DR & chaos**: suites green; artifacts (logs, k6, dashboards) archived.

---

## 5) CI/CD Plan

Pipelines extend Phase-4:

* `unit`: CRR shipper/reader, tiering demoter, cold drivers, workers, migrate CLI.
* `e2e-geo`: failover + replay correctness; divergence detector; RTO/RPO checks.
* `chaos`: shard kill, CRR delay, cold outage; verify SLO burn alerts + runbooks.
* `perf`: k6 with shard & region scale; publish p50/p95/p99.
* `sdk-compat`: golden vectors Py/Go/TS; signature bytes identical.
* `publish`: Go module + npm; tag & changelog automation.

Artifacts: k6 HTML, CRR watermarks, replay audit reports, migration logs, Grafana JSON, Prometheus rules.

---

## 6) Milestones (6–8 weeks suggested)

1. **M1 (Week 1–2)**: WP1 CRR shipper/reader MVP + metrics/alerts; WP5 golden tests.
2. **M2 (Week 3–4)**: WP2 cold drivers + demoter + compression; WP4 migrate CLI (plan/copy/verify).
3. **M3 (Week 5–6)**: WP3 async audit workers + batch anchoring; WP6 geo-DR E2E + chaos harness.
4. **M4 (Week 7–8)**: Hardening + docs/runbooks; SDK publish; optional WP7 DP/canary.

---

## 7) Technical Notes (aligning with Phase-4 design)

* **CRR** follows Phase-4 WAL semantics (append→fsync), with ordered shipping & idempotent apply; divergence detector automates split-brain checks described in runbooks.
* **Tiering** extends Phase-4 manager: lazy promote preserved; demoter adds TTL-driven moves; cold path integrates S3/GCS and lifecycle.
* **Async audit** operationalizes Phase-4 queue plan: backlog/age metrics & DLQ alerts from runbook are now enforced in code.
* **Sharded ops** implement the documented migration phases into tooling, reducing manual steps.

---

## 8) Risks & Mitigations

* **CRR lag / backpressure** → shipper rate control, alert on lag, increase WAL TTL window; documented RPO.
* **Cold-tier latency spikes** → lazy promote + prewarm; compression opt-in; alerts on cold hit rate.
* **Migration errors** → dual-write & checksums; resumable checkpoints; rollback plan per runbook.
* **Audit backlog growth** → HPA autoscale, DLQ hygiene, circuit-breaker for failing externals.
* **SDK drift** → golden vectors in CI; semver & deprecation policy; publish gates.

---

## 9) Runbooks to Update / Add

* Update: **geo-failover**, **geo-split-brain** with automation hooks & detector steps.
* Add: **wal-crr-ops.md**, **cold-driver-ops.md**, **anchoring-attestation.md**, **migrate-cli-howto.md** (maps to WP1–WP4).

---

## 10) PR Checklist (per Work Package)

* [ ] Preserves Phase 1–4 invariants; unit/E2E/chaos tests added.
* [ ] Metrics, alerts, dashboards and **runbooks** wired for the feature.
* [ ] Security: secrets via SOPS/KMS; no plaintext keys in repo/CI; TLS/mTLS as applicable.
* [ ] Observability: logs include region/shard/tenant labels; trace spans for shipper/reader/workers.
* [ ] Rollback: documented and **tested** for each feature (CRR off, migrate rollback, worker disable).

---

## References

* **Phase 4 Implementation Report** — achievements, known limitations, and Phase-5 roadmap items that this plan operationalizes (CRR shipper/reader, cold drivers & demotion/compression, async audit workers & anchoring, dedup-migrate CLI, cross-shard ops API, geo-DR/chaos tests, DP/canary).

**End of PHASE 5 plan**
