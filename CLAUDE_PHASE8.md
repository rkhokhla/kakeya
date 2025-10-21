# CLAUDE_PHASE8.md — Production-Grade Prediction, Automated Optimization, and Enterprise Rollout

> **Audience:** `claude-code`, maintainers, SRE, Sec/Compliance, Product, GTM
> **Goal:** Convert Phase-7’s prototypes (HRS, ensemble verification, cost attribution, operator policies) into **production-trained, auto-optimized, and enterprise-ready** capabilities with measurable business impact on hallucination reduction and cost. Preserve all Phase 1–7 invariants (verify-before-dedup, WAL-first, SLOs, CRR safety, auditability). The scope directly addresses Phase-7 “Known Limitations” and “Phase 8+ Roadmap.”

---

## 0) TL;DR (What Phase 8 ships)

* **HRS Productionization:** data pipelines, scheduled retraining, A/B, drift monitors; maintain ≤10 ms p95 and ≥0.85 AUC in shadow/online. (Phase-7 HRS: ≤10 ms p95, AUC≈0.87 in shadow.)
* **Ensemble Expansion:** plug a *real* micro-vote model, add RAG grounding strategy, and **per-tenant N-of-M tuning** with adaptive weighting. (Phase-7 used heuristic micro-vote & fixed 2-of-3.)
* **Cost Automation:** auto-reconcile with cloud billing APIs; predictive spend & **optimization recommendations**; enforce budgets via Operator policies. (Phase-7 cost tracer hit ±3% vs bills.)
* **Anomaly Detection v2:** deeper/variational AE, clustering & auto-thresholds; promote from shadow→guardrail feature for HRS. (Phase-7 AE ran shadow, ≤5% rate, ~2% FPR.)
* **Operator Policies v2:** adaptive canary steps, multi-objective gates (latency, cost, containment), and safe rollbacks for Risk/Cost/Ensemble CRDs. (Phase-7 CRDs shipped canary+health gates.)
* **Buyer KPIs v3:** “cost-per-trusted-task” deltas, ensemble-agreement, and HRS ROC/PR **in production dashboards**, tying to the investor pitch on hallucination containment and spend.

---

## 1) Scope & Non-Goals

### In

1. **HRS training pipeline** on real data (labels, eval sets), scheduled retraining, A/B switcher, drift alarms; model registry & immutability.
2. **Ensemble with real micro-model** + **RAG grounding check**; **tenant-adaptive N-of-M** & weights.
3. **Billing integration & optimization**: cloud billing import, reconciliation jobs, predictive forecasts, and automated recommendations & actions (cache, tiering, strategy).
4. **Anomaly v2**: VAE/stacked AE, cluster labeling, auto-thresholding; feed as HRS feature post-shadow.
5. **Operator v2**: adaptive canary, multi-objective gates, policy suggestions, dry-run sim.
6. **Buyer dashboards v3** + compliance addenda (SOC/ISO) capturing Phase-8 controls.

### Out (Phase 9+)

* Global query federation and runtime auto-sharding; formal end-to-end proofs beyond existing lemmas; external audits/certifications (we prepare binders here).

---

## 2) Deliverables (Definition of Done)

1. **HRS Productionization**

    * **Data/labels pipeline** (WORM-referenced) with PI-safe features; **model registry** (versioned, signed), **scheduler** (daily/weekly retrain) and **A/B** canaries.
    * **Drift monitors** (feature & performance) with rollback triggers; **SLO guard**: p95 ≤10 ms, AUC ≥0.85 (shadow) and ≥0.82 (online). (Phase-7: ≤10 ms, AUC≈0.87 shadow).

2. **Ensemble Expansion**

    * **Aux model** (micro-vote) service with timeout budget; **RAG grounding** strategy (citation overlap / source consistency).
    * Controller tunes **N-of-M per tenant** and strategy weights from historical agreement. (Phase-7 used fixed 2-of-3.)

3. **Cost Automation**

    * **Billing importers** (AWS/GCP/Azure) for auto-reconciliation; **predictive spend** (7-/30-day) and **recommendations** (tiering, cache TTL, ensemble config).
    * **Budget CRD** enforcer uses forecasts to pre-empt overruns (soft→hard caps). (Phase-7 tracer achieved ±3% reconciliation).

4. **Anomaly Detection v2**

    * VAE or stacked AE model with **uncertainty**; **cluster tagging** & **auto-threshold** from labeled feedback; **promotion** to HRS input after shadow win. (Phase-7 AE in shadow).

5. **Operator Policies v2**

    * Adaptive canary step sizing; **multi-objective gates** (latency, error-budget, containment, cost); **what-if simulator**; auto-rollback maintained. (Phase-7 had canaries & health gates).

6. **Buyer KPIs v3 & Compliance**

    * Grafana additions: **containment delta vs control**, **cost-per-trusted-task** trend by tenant/model, **ensemble disagreement rate**, **HRS ROC/PR**; SOC/ISO evidence hooks extended. (Phase-7 buyer dashboard v2 exists).

---

## 3) Work Packages & Tasks

### WP1 — HRS: Train, Evaluate, Deploy

* **Pipelines:** build ETL to generate labels from WORM/202-escalations & human-review outcomes; PI-safe features only.
* **Training:** logistic vs GBDT vs tiny MLP; **calibration** (Platt/isotonic); **model cards**.
* **Ops:** p95≤10 ms budget test; **A/B switcher** (header/tenant or % rollout); drift alarms (KS-test on features, AUC drop). (Phase-7 risk scorer interface & metrics already exist).

### WP2 — Ensemble: Real Micro-Vote + RAG Grounding

* **Micro-vote**: small distilled verifier (≤30 ms) with cached embeddings; **timeouts** & **fail-open/close** per policy.
* **RAG Grounding**: citation overlap (Jaccard/shingles) & source checks; configurable thresholds.
* **Tuning:** historical agreement drives **N-of-M and weights** per tenant (controller updates CRD). (Phase-7 ensemble module & CRD ready).

### WP3 — Cost Automation & Forecasting

* **Importers** for CUR/BQ billing exports; nightly **reconciliation**; **forecasts** (prophet/ARIMA).
* **Advisor**: recommendations (tiering changes, cache TTLs, RAG on/off, ensemble thresholds) with projected **$ impact**; one-click apply via Operator. (Phase-7 tracer emits `flk_cost_*` & budgets).

### WP4 — Anomaly v2 & Feedback Loop

* **Model:** VAE/stacked AE + uncertainty; **cluster** new anomalies to reduce toil; **SIEM** tagging kept.
* **Auto-threshold:** optimize FPR/TPR using feedback; **shadow→guardrail** promotion gates. (Phase-7: shadow mode, SIEM events, metrics).

### WP5 — Operator Policies v2

* **Adaptive canary** (step size varies by health); **multi-objective** policy gates; **simulator** with past traces; **dry-run** before apply. (Builds on Phase-7 Risk/Ensemble CRDs).

### WP6 — Buyer KPIs v3 & Compliance Addenda

* **Dashboards:** add ROC/PR, **containment-cost frontier**, tenant/model breakdowns;
* **Compliance:** auto-include Phase-8 artifacts in SOC/ISO binder: training reports, drift logs, policy diffs. (Phase-7 compliance hooks exist).

---

## 4) Acceptance Criteria

* **Hallucination control:** Ensemble+HRS decreases **escaped hallucinations ≥40%** vs Phase-6 baseline at **≤7%** incremental cost (Phase-7 delivered ≥30% at ≤5%).
* **HRS SLOs:** p95 ≤10 ms; AUC ≥0.85 (shadow) and ≥0.82 (online); drift alerts <24 h to remediation.
* **Ensemble SLOs:** p95 ≤120 ms (with micro-vote+RAG); agreement ≥85%; disagreement 100% WORM/SIEM. (Phase-7 p95≤100 ms, ≥85%.)
* **Cost:** monthly reconciliation ±3% vs cloud bills (match Phase-7), forecasts MAPE ≤10%, recommendations accepted by ≥2 design partners.
* **Operator:** adaptive canary & multi-objective gates pass chaos drills; safe rollback under SLO burn. (Phase-7 canary+rollback is baseline).
* **Dashboards/Compliance:** Buyer KPIs v3 live; SOC/ISO addendum autogenerated with Phase-8 evidence.

---

## 5) CI/CD Plan

* `hrs-train`: deterministic training + model card + calibration tests;
* `hrs-latency`: p95 ≤10 ms synthetic; A/B router tests.
* `ensemble-vote-rag`: strategy timeouts, N-of-M tuning, WORM/SIEM hooks;
* `cost-import-reconcile`: billing import parsers; ±3% reconciliation golden tests;
* `advisor-ab`: recommendation quality A/B (cost/latency deltas);
* `operator-policy-v2-e2e`: adaptive canary & multi-objective gates; safe rollback;
* `anomaly-v2-shadow`: cluster labeling, auto-threshold, promotion gates;
* `perf-regression`: maintain Phase-7 verify path p95≤200 ms and ensemble p95 budget.

Artifacts: model binaries & cards, ROC/PR curves, cost reconciliation reports, operator policy diffs, Grafana JSON v3, SOC/ISO addenda PDFs.

---

## 6) Milestones (6–8 weeks)

1. **M1 (Weeks 1–2)** — HRS pipeline & registry; billing importers; ensemble micro-vote MVP (shadow).
2. **M2 (Weeks 3–4)** — RAG grounding; cost forecasts & advisor; anomaly v2 shadow; operator adaptive canary.
3. **M3 (Weeks 5–6)** — HRS A/B online; tenant-adaptive N-of-M; auto-threshold anomalies; dashboards v3.
4. **M4 (Weeks 7–8)** — Hardening, chaos drills, buyer demos, compliance addenda, release notes.

---

## 7) Risks & Mitigations

* **Latency creep** (micro-vote/RAG) → strict per-strategy timeouts, parallelism, and **fail-open to 202-escalation**; budget guard in CI.
* **Model drift** → daily drift checks, automatic revert to last-good; shadow gating before promotion.
* **Cost mis-attribution** → imported billing as source-of-truth; reconcile diffs; flag “unallocated” bucket surfaced in KPIs.
* **Over-blocking** with tighter ensemble → tenant-level tuning and staged canaries via Operator.
* **Policy mis-config** → policy simulator & dry-run; multi-objective gates; one-click rollback.

---

## 8) How Phase 8 strengthens the investor pitch

* **Sharper containment with proof:** production-trained HRS + richer ensemble shrink escaped hallucinations further while keeping verify p95 and cost SLOs intact. (Phase-7 already achieved ≥30% reduction at ≤5% cost.)
* **Economic control at scale:** forecasts + advisor convert visibility into **actionable savings** (cost-per-trusted-task down, predictable budgets).
* **Enterprise confidence:** automated rollouts, SIEM/compliance addenda, and model cards translate into **auditable safety** that buyers & investors can trust.

---

## 9) References

* **PHASE7_REPORT.md** — shipped HRS (≤10 ms p95, AUC≈0.87 shadow), ensemble (2-of-3, p95≤100 ms), cost tracer (±3%), Risk/Ensemble CRDs, anomaly shadow, buyer dashboards v2; plus limitations & Phase-8 roadmap addressed here.

**End of PHASE 8 plan**
