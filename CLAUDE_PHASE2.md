# CLAUDE_PHASE2.md — Integration, Performance, Prod Helm & Alerts

> **Audience:** `claude-code`, maintainers, and contributors
> **Goal:** Build on **Phase 1** (canonicalization + signing + unit tests + baseline DevOps) to deliver an end-to-end, production-grade system with repeatable **integration tests**, **performance baselines**, **prod-ready Helm**, and **SLO-driven alerting**. Phase 1 achievements and constraints are the source of truth and MUST remain intact.

---

## 0) TL;DR (What Phase 2 delivers)

* ✅ Black-box **E2E & contract tests** (HTTP→WAL→verify→dedup→metrics) across HMAC and Ed25519 paths
* ✅ **Performance baselines** (k6) with p50/p95/p99, CPU/RAM profiles, and regression gates
* ✅ **Production Helm**: HPA, PDB, NetworkPolicies, TLS/mTLS, Secrets, topology spread, resource limits
* ✅ **Observability & alerts**: Prometheus rules + Grafana panels + runbooks tied to SLOs
* ✅ **Security & ops hardening**: secret management (SOPS/age), key rotation with overlap, WAL retention & compaction
* ✅ **Chaos & failure drills**: dedup outage, duplicate floods, signature failures, replay safety

> Everything here **extends** Phase 1’s canonical signing and verifier ordering (signature→dedup), “golden files,” and dev configs—do not break those contracts.

---

## 1) Scope & Non-Goals

### In

1. End-to-end integration tests (Compose + GitHub Actions matrix)
2. Ed25519 signing **tests** (asymmetric path) + key-rotation overlap
3. Load & performance baselines with **k6**; latency SLOs and error budget
4. Prod Helm chart features (HPA, PDB, NetworkPolicy, TLS/mTLS, Secrets, topology)
5. Prometheus **alerts**, Grafana dashboards, and **runbooks**
6. Chaos scenarios: store outages, double-delivery, invalid signatures, WAL pressure
7. Ops hygiene: SOPS/age, WAL lifecycle, backpressure & rate-limits tuning

### Out (Phase 3+)

* Advanced audit pipelines, VRF seeding, formal proofs, multi-tenancy sharding strategies

---

## 2) Deliverables (DoD-driven)

1. **E2E test suite** (`tests/e2e/…`) that spins **backend+store** and posts real PCS:

    * HMAC valid → `200` accepted; duplicate → cached with same status
    * HMAC tampered → `401` (verify-before-dedup rule enforced)
    * Ed25519 valid/invalid matrix
    * Verify that **Inbox WAL** always gets written prior to parse/verify (crash safety)
2. **k6** scenarios & HTML report committed (`load/`), with CI artifact upload
3. **Prod Helm chart** (`helm/fractal-lba/`) incl.:

    * `Deployment`, `Service`, `Ingress` (TLS), `HPA`, `PDB`, `NetworkPolicy`, `ServiceAccount`/PSA
    * Values for **signing**, **metrics auth**, **dedup store** (Redis/Postgres), **TLS/mTLS**
    * TOP TIP: default safe limits/requests; topology spread across zones
4. **Alerts & dashboards**:

    * Prometheus rules for availability and error budget; Grafana JSON dashboards
    * Runbooks under `docs/runbooks/*.md` referenced by alert annotations
5. **Security & ops**:

    * SOPS/age secrets how-to & example; **key rotation** overlap (old+new) tests
    * WAL retention policy + compaction script + CI smoke check
6. **Docs** updated: architecture, operations, and **PHASE2 Summary**
7. **All tests green** in CI: unit + golden + E2E + k6 sanity; Go builds; Helm lint

> Phase 1 already delivered canonicalization, HMAC+Ed25519 verification libs, 33 unit tests, golden vectors, and dev Helm/Compose. Preserve and reuse.

---

## 3) Work Packages & Tasks

### WP1 — E2E Integration Tests

* [ ] **Harness**: docker-compose file (`infra/compose-tests.yml`) with `backend`, `redis` and a minimal agent stub
* [ ] **Fixtures**: reuse Phase 1 golden CSVs; generate PCS via CLI (HMAC & Ed25519)
* [ ] **Cases**:

    1. `200 OK`: valid HMAC PCS; assert dedup **miss→put** and metrics increment
    2. `401 Unauthorized`: tamper `D_hat`; assert no dedup write (verify-before-dedup)
    3. Duplicate submit: second response returns cached body; `flk_dedup_hits`++
    4. Ed25519 valid & invalid; key missing → hard fail
    5. WAL presence: file appears in Inbox on every POST
* [ ] **GitHub Actions**: E2E job (Linux) with compose up/down + logs as artifacts

### WP2 — Ed25519 Path & Rotation

* [ ] **Keygen** script (Python) to emit base64 pub/priv; docs on storage (Secrets)
* [ ] **Verifier** supports **multi-key overlap** window (old+new)
* [ ] Tests:

    * PCS signed with **old** key verifies while both keys present; fails after window
    * Wrong key fails; invalid base64 fails quietly with `401`

### WP3 — Performance & Load (k6)

* [ ] Scenarios: **baseline**, **burst**, **steady 1k rps (5m)**
* [ ] Metrics captured: p50/p95/p99 latency, non-2xx rate, CPU/RAM of backend
* [ ] Threshold gates (CI): e.g., p95 < **200 ms**, error rate < **1%** (tunable)
* [ ] Markdown report stored under `load/REPORT.md`; artifacts published in CI

### WP4 — Production Helm

* [ ] Create `helm/fractal-lba/` with:

    * Deploy/Service/Ingress, **HPA**, **PDB**, **NetworkPolicy**, PSA/SCC
    * Values for **signing**, `/metrics` auth, Redis/Postgres, TLS/mTLS
    * Topology spread (zone anti-affinity), resource requests/limits
* [ ] `helm lint`, **kind/minikube** smoke deployment job in CI
* [ ] `NOTES.txt` with `kubectl` commands and validation probes
* [ ] Integrate Phase 1 toggles (signing, metrics auth, dedup), now with prod defaults

### WP5 — Alerts, Dashboards, Runbooks

* [ ] Prometheus rules:

    * **Availability**: `probe_success` or HTTP 2xx ratio
    * **Error budget**: `(flk_escalated + non2xx)/ingest_total > 2% (5m)` ⇒ page
    * **Dedup anomaly**: dedup hit ratio out of band (+/− sigma)
    * **Signature errors**: spike in `401` rate
* [ ] Grafana: extend Phase 1 dashboard with latency histograms & SLO burn panels
* [ ] Runbooks in `docs/runbooks/`:

    * `signature-spike.md`, `dedup-outage.md`, `latency-surge.md`, `wal-disk-pressure.md`

### WP6 — Security & Ops Hardening

* [ ] SOPS/age guide + sample encrypted Secret for HMAC/Ed25519 keys
* [ ] Key rotation procedure: overlap period, CI test, rollback steps
* [ ] WAL retention & compaction: CLI or cronjob; disaster-recovery notes
* [ ] Backpressure tuning: rate-limit & Retry-After strategy (429), jitter & caps

### WP7 — Chaos & Failure Drills

* [ ] **Dedup outage**: Redis down ⇒ `503 + Retry-After`; WAL continues; post-recovery sanity
* [ ] **Duplicate floods**: repeated PCS; verify idempotency & stable outcomes
* [ ] **Signature invalidity**: high rate invalid → ensure **no** dedup writes
* [ ] **Slow disk**: simulate WAL latency; ensure timeouts and alerts

---

## 4) Acceptance Criteria (Definition of Done)

* **E2E**: all cases pass locally and in CI; Inbox WAL written pre-verify for each POST
* **Security**: invalid signatures **never** reach dedup; verify-before-dedup invariant holds (asserted in tests)
* **Performance**: p95 latency under target with baseline load; published report & thresholds in CI
* **Helm**: installs cleanly on kind/minikube; `helm lint` passes; NetworkPolicies applied
* **Alerts**: rules loaded; synthetic alert fire/resolve tested; runbooks linked
* **Ops**: SOPS/age demo works; rotation overlap test passes; WAL compaction documented
* **Docs**: updated architecture & operations; Phase 2 summary committed

---

## 5) Milestones & Timeline (suggested)

1. **M1 (Week 1):** WP1 (E2E) + WP2 (Ed25519 + rotation tests)
2. **M2 (Week 2):** WP3 (k6 baseline) + WP7 (chaos drills)
3. **M3 (Week 3):** WP4 (Helm prod) + WP5 (alerts, dashboards, runbooks)
4. **M4 (Week 4):** WP6 (secrets, WAL lifecycle, backpressure) + Docs polish + CI hardening

---

## 6) CI/CD Plan

* Jobs:

    * `unit`: Phase 1 tests (Python/Go) remain green (do not regress)
    * `e2e`: compose up → run tests → collect logs/artifacts
    * `perf-sanity`: short k6 smoke with thresholds
    * `helm-lint` + `kind-deploy`: install & ping health
    * `security`: secret-scan, lint, SOPS decryption check (no plaintext keys)
* Artifacts: k6 HTML, backend logs, Grafana JSON, Prometheus rules, WAL snapshots (trimmed)

---

## 7) Implementation Notes & Pseudocode

### 7.1 E2E test skeleton (pytest)

```python
def test_hmac_accept_and_dedup():
    pcs = make_pcs_hmac(seed=42, key="testsecret")
    r1 = post("/v1/pcs/submit", pcs); assert r1.status_code == 200
    r2 = post("/v1/pcs/submit", pcs); assert r2.status_code in (200, 202)
    assert prom("flk_dedup_hits") >= 1

def test_hmac_tamper_rejected_and_not_cached():
    pcs = make_pcs_hmac(seed=42, key="testsecret")
    pcs["D_hat"] += 0.12345  # tamper
    r = post("/v1/pcs/submit", pcs)
    assert r.status_code == 401
    assert not dedup_contains(pcs["pcs_id"])
```

### 7.2 k6 threshold gate

```js
export const options = {
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<200'], // ms
  },
};
```

### 7.3 Prometheus alert (sample)

```yaml
groups:
- name: flk-slo
  rules:
  - alert: FLKErrorBudgetBurn
    expr: (rate(flk_escalated[5m]) + rate(http_requests_errors_total[5m]))
          / rate(flk_ingest_total[5m]) > 0.02
    for: 10m
    labels: {severity: page}
    annotations:
      summary: "Error budget burn >2% (10m)"
      runbook_url: https://…/docs/runbooks/latency-surge.md
```

---

## 8) Risks & Mitigations

* **Signature drift**: Changing canonicalization → breakage
  → Mitigate with golden vectors & E2E gates; document any bump as **breaking** (future version)
* **Perf regressions**: New features add latency
  → k6 thresholds in CI, diff reports on PRs
* **Secrets leakage**: Misconfigured repo or CI
  → SOPS/age, secret scanning, deny plaintext env in CI
* **NetworkPolicy lockouts**: Too strict rules
  → Staged rollout, smoke checks, fallbacks with logs

---

## 9) Operability Runbooks (to add in `/docs/runbooks/`)

* `signature-spike.md` — investigate 401 spikes (clock skew? key rollover? seed drift?)
* `dedup-outage.md` — Redis down, WAL growth, when to shed load
* `latency-surge.md` — CPU/RAM saturation, profiling steps, scaling HPA
* `wal-disk-pressure.md` — compaction, retention settings, SLO tradeoffs

---

## 10) PR Checklist (each work package)

* [ ] Unit tests pass (including Phase 1’s 33 tests)
* [ ] E2E tests added/green; artifacts attached
* [ ] k6 report committed; thresholds enforced
* [ ] Helm installs on kind; `helm lint` passes
* [ ] Prometheus rules validated; test alerts fired/resolved
* [ ] Runbooks linked from alert annotations
* [ ] Secrets handled via SOPS/age; no plaintext in repo/CI logs
* [ ] Docs updated (CHANGELOG + PHASE2 summary)

---

## 11) References

* **PHASE 1 Implementation Report** — canonicalization, signing, unit tests, golden vectors, dev Helm/Compose, and verify-before-dedup ordering (MUST keep)

---

**End of PHASE 2 plan**
