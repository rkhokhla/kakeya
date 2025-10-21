# CLAUDE_PHASE7.md — Real-Time Hallucination Prediction, Ensemble Defenses, and Cost Attribution

> **Audience:** `claude-code`, maintainers, SRE, Security/Compliance, Product & GTM
> **Goal:** Build on **Phase 6** (operator-driven autonomy, advanced CRR, SIEM/compliance automation, predictive tiering, Rust/WASM SDKs, formal proofs, buyer KPIs) to deliver real-time **hallucination prediction**, **ensemble verification**, and **per-tenant/model cost attribution**—turning our verification layer into a measurable business lever. Preserve all Phase 1–6 invariants (verify-before-dedup, WAL-first, SLOs, CRR safety) and reuse Phase-6 automation and dashboards as the operational floor.

---

## 0) TL;DR (What Phase 7 ships)

* **Hallucination Risk Scorer (HRS)**: low-latency, per-request risk estimate with confidence intervals; feeds policy gates & budgets in real time.
* **Ensemble defenses**: multi-signal, multi-model strategy combining PCS signals, retrieval checks, and lightweight cross-model votes; integrates with existing 202-escalation lanes.
* **Cost attribution**: per-tenant / per-model / per-task cost tracing (compute, storage, network, anchoring) wired into Buyer KPI dashboards and SIEM.
* **Operator playbooks as policies**: encode risk-aware routing, geo/topology selection, and pre-warm decisions into the Operator with safe rollbacks.
* **Anomaly detection**: autoencoder-based outlier scoring on PCS distributions to surface novel failure modes without sacrificing SLOs.

---

## 1) Why now (ties to investor pitch)

Our pitch promises **measurable hallucination reduction** and **economic control**. Phase 6 gave us the automation, proofs, and buyer dashboards; Phase 7 adds **proactive prediction** and **clear $-impact per tenant/model**, so customers and investors see *exactly how trust gating lowers error rates and costs in production*.

---

## 2) Scope & Non-Goals

### In

1. **Real-time Hallucination Risk Scorer (HRS)** w/ confidence intervals and latency budget ≤10 ms.
2. **Ensemble verification**: configurable N-of-M strategy (PCS, retrieval sanity, lightweight cross-model check).
3. **Cost attribution & budgets**: precise cost tracing per tenant/model/task; policy hooks to throttle/route by budget.
4. **Operator policy encoding**: risk-aware CRR selectivity, tier pre-warm, and canary % rollouts as CRDs (no hand-runs).
5. **Unsupervised anomaly detection** on PCS vectors (D̂, coh★, r + timing) with 202-escalation integration.
6. **Buyer dashboards v2**: add prediction quality, ensemble agreement, and cost-per-trusted-task deltas.

### Out (Phase 8+)

* Global query federation and adaptive sharding at runtime; deep model retraining pipelines (shadow mode only in P7).

---

## 3) Deliverables (Definition of Done)

1. **HRS microservice** (Go or Rust): online features from PCS + runtime stats; returns `{risk, ci_low, ci_high}` in ≤10 ms p95; exported metric `flk_hrs_latency_ms`. Feeds budget function and policy DSL.
2. **Ensemble module** in verifier path: pluggable checks (PCS consistency, retrieval overlap, micro-vote from a small auxiliary model) with **N-of-M** acceptance rules and **202-escalation** on disagreement. Builds on existing escalation lanes and WORM logging.
3. **Cost tracer**: per-request spans attribute compute/storage/network/anchoring using Phase-6 cost models; emits `flk_cost_*` by tenant/model/task; appears in Buyer KPI dashboard.
4. **Operator CRDs**:

    * `RiskRoutingPolicy`: bind HRS bands to actions (RAG required, human-review, reduce budget, alternate region).
    * `CostBudgetPolicy`: tenant/model daily budgets with soft/hard caps → auto-throttle / plan rollover.
    * `EnsemblePolicy`: N-of-M thresholds, timeouts, fail-open/close setting.
      Encoded with the same guardrails & rollback semantics introduced in Phase 6.
5. **Anomaly detector** (unsupervised): autoencoder on PCS/time features; metrics `flk_anomaly_rate`, alert → 202 & SIEM stream with redacted context (PII gates remain).
6. **Dashboards v2**: prediction ROC/PR curves (shadow), ensemble-agreement %, **cost-per-trusted-task** trend, and *savings vs control* panels; SOC/ISO evidence hooks extend Phase-6 compliance generator.

---

## 4) Work Packages & Key Tasks

### WP1 — Real-time HRS

* Feature store: online vectors from PCS + timing (no raw content); bounded feature drift checks.
* Model: start with calibrated logistic regression / gradient boosting; export **confidence intervals**.
* Pathing: called in verify hot-path with **strict 10 ms p95** budget; fails closed to “unknown” (policy decides).

### WP2 — Ensemble Verification

* Implement **N-of-M** policy: PCS recompute, retrieval overlap (shingle/Jaccard on citations), micro-vote model.
* Wire to **202-escalation** already present; WORM audit entries for disagreements; SIEM event “ensemble_disagree”.

### WP3 — Cost Attribution & Budgeting

* Extend Phase-6 buyer KPI codepath to tag spans with tenant/model/task and propagate cost meters.
* `CostBudgetPolicy` CRD: soft cap → degrade to cheaper strategy (more retrieval, fewer tool calls), hard cap → queue/deny with customer-visible reason.

### WP4 — Operator Policies

* Controllers for `RiskRoutingPolicy`, `CostBudgetPolicy`, `EnsemblePolicy`; dry-run + canary % rollout; rollback on SLO burn (reuse Phase-6 health gates).

### WP5 — Anomaly Detection

* Train compact autoencoder on PCS/timing; run **shadow** first; alerting hooks; add “anomaly feature” to HRS as auxiliary input after shadow wins.

### WP6 — Dashboards & Compliance

* Add HRS/AED quality (AUC/PR), ensemble agreement, and **cost per trusted task** diffs into Grafana; extend compliance report generator to include P7 controls & evidence.

---

## 5) Acceptance Criteria

* **Latency:** HRS call ≤10 ms p95; verify p95 ≤200 ms remains green; cold-read p95 ≤500 ms unchanged.
* **Quality:** In shadow, HRS AUC ≥0.85 and ensemble reduces **escaped hallucinations** ≥30% at ≤5% cost increase (tunable). KPIs land in Buyer dashboard.
* **Economics:** Cost tracer reconciles with monthly cloud bill within ±5%; **cost-per-trusted-task** shows ≥15% improvement vs baseline cohorts.
* **Safety:** 100% of ensemble disagreements and high-risk events written to WORM and streamed to SIEM; PII gates still enforced.
* **Ops:** New CRDs can be applied/rolled back by the Operator with zero downtime; SLO burn triggers rollback automatically.

---

## 6) CI/CD Plan

* `hrs-unit` & `hrs-latency`: model correctness + tight p95 timing checks.
* `ensemble-e2e`: N-of-M flows, timeouts, and 202-escalation path; WORM/SIEM events present.
* `cost-attr-tests`: golden accounting across scenarios; reconciliation tests with synthetic “bill”.
* `operator-policy-e2e`: apply Risk/Cost/Ensemble CRDs → canary → rollback on health-gate fail (reuse Phase-6 operator harness).
* `anomaly-shadow`: drift & false-positive guardrails; no policy effect until A/B win.
* `perf-regression`: maintain Phase-6 SLO gates (verify p95≤200 ms, CRR lag≤60 s, backlog p99<1 h).

Artifacts: HRS model cards, ROC/PR curves, ensemble reports, cost reconciliation logs, CRD yaml examples, Grafana JSON, compliance PDF addendum.

---

## 7) Milestones (6–8 weeks)

1. **M1 (Week 1–2):** HRS MVP (shadow) + cost tracer plumbing + CRD schemas for Risk/Cost/Ensemble.
2. **M2 (Week 3–4):** Ensemble path (shadow), dashboards v2, operator controllers; CI suites green.
3. **M3 (Week 5–6):** Anomaly detector (shadow) + A/B + budget/policy integration; compliance generator updates.
4. **M4 (Week 7–8):** Hardening; flip-on canaries for HRS/ensemble; rollback rehearsals; docs & runbooks.

---

## 8) Risks & Mitigations

* **Latency creep** in verify path → strict 10 ms budget for HRS; pre-computed features; fallback to “unknown” without blocking.
* **Over-blocking** by ensemble → staged shadow + canary; N-of-M tuned per tenant; fail-open for non-critical flows.
* **Cost mis-attribution** → span-level meters + reconciliation tests; surface “unallocated” bucket transparently.
* **Model drift** → periodic recalibration; alarms on AUC drop; feature store with schema versioning.
* **Operator mis-reconcile** → same dry-run & rollback gates as Phase-6 migrations; human approval for strict policies.

---

## 9) How Phase 7 advances the hallucination promise

* **Predict** likely hallucinations before they act, not just detect after: HRS gives a calibrated risk score per request.
* **Defend** with ensembles that turn disagreement into **202-escalations** and human review, logged in **WORM** and streamed to **SIEM**.
* **Prove** the business value with **cost-per-trusted-task** and containment-rate deltas on the Buyer dashboard—aligned with the investor pitch.

---

## 10) References

* **Phase 6 Implementation Report** — operator autonomy, advanced CRR with auto-reconcile, SIEM/compliance automation, predictive tiering, Rust/WASM SDKs, formal verification, and Buyer KPIs that Phase 7 builds upon.

**End of PHASE 7 plan**
