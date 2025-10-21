# CLAUDE_PHASE9.md — Explainable Risk, Bandit-Tuned Ensembles, and Enterprise-Grade Cost Governance

> **Audience:** `claude-code`, maintainers, SRE, Security/Compliance, Product & GTM
> **Goal:** Build on **Phase 8** (production HRS with AUC≥0.82 online, ≤10 ms p95; ensemble agreement 88%; anomaly v2 FPR=1.8%/TPR=96.5%; cost reconciliation ±2.8% with MAPE 8%; operator v2 with adaptive canary; dashboards v3; SOC/ISO extensions) to deliver **explainable risk**, **bandit-tuned ensembles**, **multi-cloud cost governance**, and **blocking-mode anomaly defenses**—while preserving all Phase 1–8 invariants/SLOs.

---

## 0) TL;DR (what Phase 9 ships)

* **Explainable HRS**: per-request SHAP/LIME attributions (PI-safe), model cards v2, and fairness/drift audits with rollback hooks.
* **Bandit-tuned Ensemble**: Thompson-sampling/UCΒ controller to optimize **N-of-M** & strategy weights per tenant, improving agreement/containment at fixed latency budget (≤120 ms p95).
* **Blocking-mode Anomaly v2**: promote VAE detector from guardrail→**blocking** for high-confidence anomalies, with active-learning loop.
* **Cost Governance**: ARIMA/Prophet forecasts, BigQuery/Azure importers, budget policy automation, and a **Cost Advisor** that can apply operator policies with one-click reviews.
* **Operator Simulator v2**: causal-impact style “what-if” on historical traces for multi-objective gates (latency, cost, containment), safer than Phase-8 replay.
* **Buyer KPIs v4**: attribution of savings to specific policies (ensemble, HRS, tiering) and **cost-per-trusted-task** by tenant/model/region.

**Keep green the Phase 8 floors:** verify p95≤200 ms; HRS p95≤10 ms & AUC≥0.82 online; ensemble agreement≥85% (now 88%); cost reconciliation ±3% (now ±2.8%); anomaly FPR≤2%, TPR≥95%—and all Phase 1–7 invariants (WAL-first, verify-before-dedup, idempotency).

---

## 1) Scope & Non-Goals

### In

1. **HRS Explainability & Governance** (SHAP/LIME, model cards v2, fairness tests, drift + rollback gates).
2. **Bandit-tuned Ensemble** (tenant-adaptive exploration/exploitation for N-of-M & weights; keeps ≤120 ms p95).
3. **Anomaly v2 Blocking + Active Learning** (human-in-the-loop labeling, confidence thresholds, SIEM hooks).
4. **Cost Governance v2** (ARIMA/Prophet, full GCP/Azure importers, auto budgets via Operator policies).
5. **Operator Simulator v2** (causal impact; confidence bands for prediction).
6. **Buyer KPIs v4** (policy-level ROI attribution).

### Out (Phase 10+)

* Online learning for HRS; federated/continual learning; global query federation & runtime auto-sharding. (Tracked in Phase 10/11 roadmap.)

---

## 2) Deliverables (Definition of Done)

1. **Explainable HRS**

    * Per-prediction **attribution vector** with SHAP/LIME over PI-safe features (e.g., D̂, coh★, r, timing) and **model cards v2** with limitations & bias checks.
    * **Fairness/Drift audit job**: alerts on AUC drop >0.05 or subgroup performance gap >5 pts; **auto-revert** to last-good model.

2. **Bandit-tuned Ensemble**

    * Controller running **Thompson sampling/UCΒ** to select N-of-M & weights (PCS/RAG/micro-vote) per tenant, subject to latency/cost constraints; **CRD** `EnsembleBanditPolicy`.
    * Evidence of improved agreement/containment without breaching **≤120 ms p95**.

3. **Anomaly v2 Blocking Mode**

    * **Dual-threshold** policy: block when score≥τ_block and uncertainty≤u_max; else guardrail→HRS; include **active-learning queue** for ops labelling; all decisions WORM/SIEM.

4. **Cost Governance v2**

    * **Forecasting** adds ARIMA/Prophet; **importers**: GCP BigQuery (live) and Azure billing.
    * `CostBudgetPolicy` automation (soft→hard caps) and **Advisor** that opens one-click PRs/Operator policies with projected **$ impact**.

5. **Operator Simulator v2**

    * Causal-impact engine on historical traces; outputs **counterfactual diffs** (latency, containment, cost) with 95% CI; ties into adaptive canary gates.

6. **Buyer KPIs v4**

    * New panels: **policy-level ROI**, **containment-cost frontier**, **per-tenant/model/region CPTT**, and **savings attribution** (realized vs projected).

---

## 3) Work Packages (WPs) & Key Tasks

### WP1 — HRS Explainability & Governance

* Implement **SHAP/LIME** for the Phase-8 HRS feature set; redact PI/PII by design.
* Extend **model registry** with explainability artifacts and **model cards v2**; publish to dashboards.
* Add **fairness tests** (subgroup deltas, calibration), **drift sentinels**, and **auto-revert** integration with the operator.

### WP2 — Bandit-tuned Ensembles

* Build **bandit controller** using tenant reward signals (containment↑, agreement↑, latency/cost within SLO).
* Integrate with `TenantEnsemblePolicy` from Phase 8; guard with **≤120 ms p95** and cost ≤ Phase-8 +7%.
* Offline replay (simulator v2) → canary → rollout.

### WP3 — Anomaly Blocking + Active Learning

* Promote detector to **blocking** with conservative thresholds; instrument **uncertainty** to avoid over-blocking.
* Create **labeling UI/API**; prioritized sampling of edge cases; **threshold optimizer** uses new labels.

### WP4 — Cost Governance v2

* Add **ARIMA/Prophet** pipelines; wire **GCP BigQuery** & **Azure** importers (Phase-8 had placeholders).
* Auto-enforce **budget policies** via Operator; expose **advisor recommendations** as PRs with projected/realized savings.

### WP5 — Operator Simulator v2 (Causal Impact)

* Implement counterfactual estimation from traces to predict policy effects with **confidence bands**; feed health gates.
* Replace Phase-8 simple replay with causal scoring for **approve/reject** recommendations.

### WP6 — Buyer KPIs v4

* Attribute savings to **specific policies** (ensemble/HRS/tiering); add CPTT breakdowns; persist **policy→ROI** links in reports.

---

## 4) Acceptance Criteria

* **Explainability:** Every HRS decision returns **attribution** and appears in model card v2; fairness drift alerts fire <24 h; **auto-revert** works. (Builds on Phase-8 drift gates.)
* **Bandit gains:** +2–5 pp **agreement** or +5–10 % **containment** at **≤120 ms p95** (no worse than Phase-8) and cost delta ≤+7%. (Phase-8 ensemble at 88% agreement.)
* **Blocking anomaly:** Escape rate from anomaly class ↓ by ≥50% at **FPR ≤2%** (Phase-8 baseline 1.8%) and TPR ≥95%.
* **Cost governance:** Reconciliation stays within **±3%** (Phase-8: ±2.8%); forecasts **MAPE ≤8%** or better; budgets enforced automatically with rollback on SLO burn.
* **Simulator v2:** Prediction error (latency/containment/cost deltas) **≤±10%** vs canary outcomes (Phase-8 sim ≈92% accurate).
* **KPIs v4:** Dashboards show **policy-level ROI** and **CPTT** trends per tenant/model/region; exports included in compliance bundle.

---

## 5) CI/CD Plan

* `hrs-explain-unit`: attribution correctness, perf budget **≤2 ms** extra per call.
* `hrs-fairness-audit`: subgroup metrics; drift triggers → auto-revert test.
* `ensemble-bandit-replay`: offline evaluation vs logged traces; guard **p95≤120 ms**.
* `anomaly-blocking-e2e`: dual-threshold logic; FPR/TPR gates; SIEM/WORM events present.
* `cost-forecast-arima-prophet`: backtests with 30/60/90-day windows; MAPE guard.
* `operator-sim-v2`: causal impact unit + integration; **±10%** predictive accuracy gate.
* `perf-regression`: preserve Phase-8 SLOs (verify p95≤200 ms; HRS p95≤10 ms; ensemble p95≤120 ms).

Artifacts: model cards v2, SHAP plots (sampled), bandit policies & rewards, anomaly threshold reports, cost forecasts & reconciliation logs, simulator accuracy reports, Grafana v4 JSON, compliance addenda.

---

## 6) Milestones (6–8 weeks)

1. **M1 (Weeks 1–2):** HRS explainability MVP + fairness audits; cost importers (GCP/Azure) online; simulator v2 prototype.
2. **M2 (Weeks 3–4):** Bandit controller shadow tests; anomaly blocking in shadow + active-learning loop; ARIMA/Prophet backtests.
3. **M3 (Weeks 5–6):** Canary of bandit-tuned ensemble; anomaly blocking on small %; budget automation; KPIs v4 panels.
4. **M4 (Weeks 7–8):** Hardening, chaos drills, compliance reports with explainability evidence; broad rollout.

---

## 7) Risks & Mitigations

* **Latency creep from explainability** → compute SHAP/LIME **asynchronously** with cached summaries; inline path returns compact attributions only.
* **Bandit over-exploration** → tenant-level **caps** and simulator v2 gating; rollback on SLO burn.
* **Over-blocking anomalies** → uncertainty-aware thresholds; staged rollout; active learning to recalibrate.
* **Forecast drift** → ensemble of ARIMA/Prophet + exponential smoothing fallback; alert on MAPE↑.
* **Policy misconfig** → causal simulator + canary + auto-rollback (Phase-8 proven).

---

## 8) How Phase 9 strengthens the investor pitch

* **Trust you can see:** explainable scores & fairness audits make risk gating **auditable**.
* **Containment that optimizes itself:** bandit-tuned ensembles improve containment **without extra cost/latency**.
* **Spend you can steer:** budget automation and policy-level ROI connect governance directly to **CPTT** savings.
* **Enterprise-ready safety:** blocking anomalies + SIEM/WORM + compliance addenda turn “safer AI” into a **defensible platform**.

---

## References

* **Phase 8 Implementation Report** — HRS production (AUC≥0.82 online, ≤10 ms p95), ensemble 88% agreement, anomaly v2 (FPR=1.8%, TPR=96.5%), cost recon ±2.8% & MAPE 8%, operator v2, dashboards v3, SOC/ISO extensions—these are the floors Phase 9 builds upon.

**End of PHASE 9 plan**
