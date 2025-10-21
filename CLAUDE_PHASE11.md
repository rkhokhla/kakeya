# CLAUDE.md — PHASE 11 Plan for `kakeya` (Claude / claude-code)

**Audience:** `claude-code` (autonomous coder), maintainers, SRE
**Goal:** Turn Phase 9 designs into production code, finish Phase 10’s remaining WPs, and lock in observability/compliance so we can scale pilots confidently—all without breaking Phases 1–10 invariants.

---

## 0) TL;DR (what to ship in Phase 11)

* **Finish Phase 9 code** (fix stubs, compile errors, add tests) and gate it behind flags.
* **Implement Phase 10 WPs left “documented-only”**: VRF verification (ECVRF), tokenized RAG overlap, JWT gateway auth, risk-based routing, CI/CD hardening, Helm production toggles, docs refresh, fuzz + input validation.
* **Add first-class OpenTelemetry** traces/metrics/logs and tighten runbooks/alerts to investor-visible KPIs (containment, CPTT).
* **Ship per-tenant canaries + rollback** for every new control (VRF/JWT/RAG/routing).

---

## 1) Context & non-goals

* **Phase 10 is production-ready** for core parity/dedup/caching; deploy and keep it green.
* **Phase 9 code exists but had stubs/compilation gaps**; PHASE 11 is where we finish it and add tests.
* README already positions “Next (Phase 11+)” as VRF, tokenized RAG, JWT, risk routing, fuzz, CI/CD, Helm; we will deliver those.

**Non-goals:** new ML models beyond the documented HRS/ensemble; multi-region topology changes; external pen-test (track separately).

---

## 2) Deliverables (Definition of Done)

1. **Phase 9 code complete, typed, tested** (≥80% for new pkgs); zero compilation errors; E2E passing.
2. **ECVRF verify path** (RFC 9381) on the backend, feature-flagged per tenant; runbooks + KMS-ready key mgmt.
3. **Tokenized RAG overlap** (word-level Jaccard with n-gram shingles) configurable; perf impact ≤10ms p95.
4. **Gateway JWT auth** (Envoy jwt_authn) binding tenant_id; backend trusts only gateway-stamped payload.
5. **Risk-based routing** toggle with SLO guardrails and A/B harness; containment within ±0.5% of baseline.
6. **CI/CD hardening**: lint/type/security/perf/chaos/golden + nightly fuzz; PRs blocked on gates.
7. **Helm production values** (VRF/JWT/risk/RAG/cache sizes/RBAC/egress) + `helm lint` + kind smoke tests.
8. **Observability**: OTel traces/metrics/logs with dashboards and runbooks mapped to investor KPIs.
9. **Docs refresh** (README, quickstart, OpenAPI, runbooks) reflecting new controls and rollback steps.

---

## 3) Work Packages (WP) — tasks for `claude-code`

### WP0 — Phase 9 completion (fix + tests)

**Why:** Verification report shows missing types/methods and no tests.

**Tasks**

* Add `internal/cost/{tracer.go,types.go}` with `Tracer`, `BillingRecord`; fix `CostForecast` fields per report.
* Implement `ModelRegistry` methods (`GetActiveModel`, `GetPreviousActiveModel`, `PromoteModel`) and `TrainingMetrics` on `ModelCard`; fix `PCSFeatures` conversions.
* Repair tiering config fields (`HotTTL`, `WarmTTL`, `ColdTTL`) and `Demote()` signature (Phase 4/5 leftovers).
* Tests: `hrs_explainability_test.go`, `hrs_fairness_audit_test.go`, `ensemble_bandit_controller_test.go`, `cost_forecast_v2_test.go`, `anomaly_blocking_test.go`.
* E2E: HRS prediction + fairness audit; bandit loop; cost import + forecast; anomaly blocking WORM/SIEM.

**Acceptance:** `go build ./...` clean; new suites ≥80% coverage for changed code; E2E green.

---

### WP1 — ECVRF verification (per-tenant)

**Why:** Close the proof gap for seed derivation and strengthen defense-in-depth. Planned in Phase 10, not fully implemented.

**Tasks**

* Add `backend/pkg/crypto/vrf/` with ECVRF-ED25519-SHA512-TAI verify per **RFC 9381** test vectors; expose `Verify(pub, alpha, proof)`.
* Policy: `vrf.required`, `vrf.pubkey`, `vrf.mode=shadow|enforce`.
* Helm values + runbook `vrf-policy-mismatch.md` (key rotation, overlap period).
* KMS-ready note for pubkey rotation; roadmap item for managed keys.

**Acceptance:** Vectors pass; 401/202 behavior correct; p95 verify <5ms; feature-flag rollout.

---

### WP2 — Tokenized RAG overlap

**Why:** Robust grounding vs punctuation/formatting; documented plan.

**Tasks**

* Unicode-aware tokenization + optional stopwords/stemming; n-gram shingles; Jaccard.
* Config: `rag.enabled`, `rag.minOverlap` (default 0.35), `rag.shingleSize` (2), `rag.stopwords=true`, `rag.stemming=false`.
* Unit tests for URLs/code/mixed punctuation; performance benchmark (10KB text target ≤10ms p95).

**Acceptance:** False positives ↓ on synthetic cases; perf gate passes.

---

### WP3 — Gateway JWT auth (tenant binding)

**Why:** Prevent spoofing of tenant headers; declarative, well-supported in Envoy.

**Tasks**

* Provide Envoy `jwt_authn` config example (issuer, audiences, JWKS).
* Backend middleware: require `X-Auth-Verified: true`, parse payload, bind `tenant_id`, reject header mismatch.
* Runbook `auth-gateway-down.md` (JWKS outage, fallback), examples for NGINX/Contour.

**Acceptance:** Unauthorized requests rejected; per-tenant isolation enforced; end-to-end test with gateway passes.

---

### WP4 — Risk-based routing (fast path with guardrails)

**Why:** Cut p95 while preserving containment; planned and measured.

**Tasks**

* Policy: `riskRouting.enabled`, `skipEnsembleBelow`, `requireNarrowCI`, `maxCIWidth`.
* Implement decision in hot path; always WORM log; A/B harness and dashboards.
* Alerts: p95 regression, fast-path share, containment delta runbooks.

**Acceptance:** p95 ↓ 25–40% vs control; containment within ±0.5%; canary 10%→50%→100%.

---

### WP5 — CI/CD hardening + fuzz

**Why:** Block regressions and malformed inputs before they ship.

**Tasks**

* Add/enable: lint/type/security (ruff/mypy, golangci-lint/govet/gosec, eslint/tsc), supply-chain scan (Trivy).
* Perf gates (k6) by label; nightly chaos; Go 1.18+ fuzz for parsers/PCS/JSON.
* Golden vectors job for cross-language signatures (already present—expand vectors to ≥50).

**Acceptance:** PRs blocked on gates; nightly fuzz runs; artifacts uploaded.

---

### WP6 — Helm production toggles & RBAC

**Why:** One-click on/off for new controls; least privilege.

**Tasks**

* Values for `vrf`, `riskRouting`, `rag`, cache sizes, SIEM egress; RBAC tighten; NetworkPolicies.
* `helm lint` + kind “smoke” job in CI; NOTES.txt with validation cmds.

**Acceptance:** `helm template` clean; kind deploy green; toggles work.

---

### WP7 — Observability: OpenTelemetry first-class

**Why:** Correlate traces, metrics, and logs across controls (VRF/JWT/RAG/routing) and expose investor KPIs.

**Tasks**

* Wire **OTel** tracing/metrics/logs in backend; Collector example; exemplars for correlation.
* Dashboards: Containment vs CPTT vs latency; fast-path %.
* Runbooks: latency regressions, cache hit-rate, JWT/VRF failures.

**Acceptance:** Traces stitched to metrics; dashboards populated; alerts link to runbooks.

---

### WP8 — Docs refresh (investor-grade)

**Why:** README claims and investor pitch must match the shipping system.

**Tasks**

* Update README quickstart and feature matrix; link **signal computation**, **overview**, **policies**, **security**, **dashboards**, **runbooks**.
* OpenAPI sync; examples for JWT/VRF/risk/RAG flows; CHANGELOG.

**Acceptance:** Clean setup in <15 minutes; docs pass link-check; pitch reflects reality.

---

## 4) Implementation notes & code footprints

* **VRF**: use ECVRF-ED25519-SHA512-TAI as per RFC 9381; include golden vectors and key-rotation SOP.
* **JWT**: Envoy `jwt_authn` filter handles signature/issuer/audience/expiry; backend trusts only gateway-stamped identity.
* **RAG**: tokenizer/shingles/Jaccard helpers under `backend/pkg/text/`.
* **Risk routing**: guard narrow CI; always write WORM; expose metrics.
* **Dedup** remains atomic (Redis `SETNX` or Postgres `ON CONFLICT DO NOTHING`)—keep as is. ([Redis][1])

---

## 5) PR breakdown (small, reviewable)

1. **PR-P9-Fixes** — Phase 9 compile + tests + tiering fix.
2. **PR-VRF** — ECVRF verify + policy + Helm + runbook.
3. **PR-RAG** — Tokenized overlap + thresholds + tests.
4. **PR-JWT** — Gateway config samples + backend middleware + e2e.
5. **PR-RiskRoute** — Fast path + A/B + alerts/runbooks.
6. **PR-CI/Fuzz** — Lint/type/security/perf/chaos/golden + fuzz.
7. **PR-Helm** — Values/RBAC/egress + kind smoke.
8. **PR-Docs** — README/OpenAPI/examples/runbooks sync.

Each PR: tests + rollback plan + CHANGELOG.

---

## 6) Acceptance criteria (phase-level)

* All PRs merged; **e2e suite green**; **canary rollouts** complete with no SLO regressions.
* **Containment**, **CPTT**, **p95 latency** reported on dashboards with OTel correlation.
* **Docs** and runbooks reflect VRF/JWT/RAG/risk routing and how to toggle/rollback.

---

## 7) Suggested timeline (fast but safe)

* **Week 1:** WP0 (P9 fixes) → PR-P9-Fixes; start PR-VRF.
* **Week 2:** PR-RAG, PR-JWT; begin A/B harness for risk routing.
* **Week 3:** PR-RiskRoute canary; PR-CI/Fuzz; PR-Helm; PR-Docs polish.

---

## 8) Risks & mitigations

* **Signature/seed drift** with VRF → golden vectors + shadow mode first.
* **Auth outages** (JWKS down) → gateway HA + runbook + temporary legacy fallback.
* **Perf regressions** from RAG/routing → perf gates (k6) and canaries.
* **Observability gaps** → OTel collector baseline, exemplars linking traces/metrics/logs.

---

## 9) Quick commands & checks

* **Build & tests**

  ```bash
  go build ./... && go test ./... -race
  python -m pytest -q
  ```
* **Golden vectors**

  ```bash
  pytest tests/golden -q && npm test -- tests/golden/ && go test ./tests/golden/ -v
  ```
* **Helm smoke**

  ```bash
  helm lint helm/flk && kind create cluster && helm upgrade --install flk helm/flk -f helm/values.dev.yaml
  ```

---

## 10) Why this matters for investors (short)

Phase 11 closes the last verification gaps (crypto, grounding, auth, routing) and makes the platform **explainable, auditable, and cheap to operate at scale**. It directly supports the pitch that we **reduce hallucinations without retraining** and prove it with dashboards (containment ↑, CPTT ↓).

---

### References (standards/tools cited)

* **ECVRF / RFC 9381** (industry standard VRF verification). ([IETF Datatracker][2])
* **Envoy `jwt_authn` filter** (JWT verification at gateway). ([envoyproxy.io][3])
* **OpenTelemetry** (traces/metrics/logs correlation, Collector). ([OpenTelemetry][4])
* **Atomicity references** (already used in earlier phases but retained): Redis `SETNX`, Postgres `INSERT … ON CONFLICT`. ([Redis][1])

---

**Appendix — Status sources**

* Phase 9/10 verification status and action items.
* Phase 10 WP list and “documented-only” items to implement.
* README “Next (Phase 11+)” promises to fulfill.

**End of CLAUDE.md**

[1]: https://redis.io/docs/latest/commands/setnx/?utm_source=chatgpt.com "SETNX | Docs"
[2]: https://datatracker.ietf.org/doc/rfc9381/?utm_source=chatgpt.com "RFC 9381 - Verifiable Random Functions (VRFs)"
[3]: https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/jwt_authn_filter?utm_source=chatgpt.com "JWT Authentication"
[4]: https://opentelemetry.io/docs/specs/otel/overview/?utm_source=chatgpt.com "Overview"
