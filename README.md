# FLK / PCS — A Verification Layer that Tames LLM Hallucinations

### TL;DR

**FLK/PCS** turns model outputs into **Proof-of-Computation Summaries (PCS)** that we **verify server-side**. Low-trust generations are automatically **contained** (extra retrieval, review, or budget caps). High-trust ones **flow fast**. Result: **fewer hallucinations at lower cost**—without retraining or vendor lock-in.

---

## Elevator Pitch (60 seconds)

LLMs are brilliant but unreliable. Throwing bigger models or more prompts at the problem just burns cash. **FLK/PCS** plugs into any agent or RAG stack and makes each generation **provably safer**:

* The agent emits a compact **PCS**—signals like fractal slope **D̂**, **directional coherence** `coh★`, and **compressibility** `r`—and **cryptographically signs** it.
* Our Go verifier **recomputes** those signals with tolerances, enforces **verify-before-dedup**, and decides a **trust-tier + budget** for the request.
* Low-trust outputs are **gated** into safer lanes; good outputs sail through.

It’s **model-agnostic**, **auditable**, and **production-ready**.

---

## Investor Pitch (short)

**The problem:** Hallucinations erode trust, trigger refunds, and create compliance risk. Today’s “fixes” (bigger LLMs, extra RAG) raise latency and spend—without guarantees.

**Our approach:** A **verification control plane** around generation. We score structure vs noise (D̂ / coh★ / r), **verify signatures**, and route traffic by **quantified trust**. The system is fault-tolerant (WAL, idempotency), observable (Prometheus/Grafana), and secure (HMAC/Ed25519, TLS/mTLS).

**Why now:** Enterprises are moving from pilots to production. They need **governance and cost control**—not just accuracy anecdotes. We reduce hallucinations **without retraining**, prove it with dashboards, and keep options open across model vendors.

**Moat:** Verifiable signals + server recomputation, strict invariants (verify → dedup), audit lineage, multi-region DR, SDK parity across Python/Go/TS (Rust/WASM deployed), **explainable risk scores** (SHAP/LIME), **self-optimizing ensembles** (bandit-tuned), **blocking anomaly detection**, and **policy-level ROI attribution**. It's a **drop-in layer** that standardizes safety and spend.

---

## What This Repo Contains

* **Agent side (Python):** Computes PCS, writes **Outbox WAL**, signs, and posts to the verifier.
* **Verifier (Go):** Appends **Inbox WAL** → verifies canonicalized PCS → **idempotent dedup** → decides budget/regime → emits metrics.
* **Signals:** `D̂`, `coh★`, `r` → **regime** (`sticky / mixed / non_sticky`) → **budget** (bounded [0,1]) that gates actions.
* **Ops:** Helm charts, SOPS/age secrets, Prometheus rules, Grafana dashboards, chaos & DR drills.

Dive deeper:

* 📄 **Signal computation details:** [docs/architecture/signal-computation.md](docs/architecture/signal-computation.md)
* 🧭 **System overview:** [docs/architecture/overview.md](docs/architecture/overview.md)
* ⚙️ **Operator & policies:** [docs/operations/policies.md](docs/operations/policies.md)
* 🔐 **Security & keys:** [docs/security/overview.md](docs/security/overview.md)
* 📊 **Dashboards & SLOs:** [docs/observability/dashboards.md](docs/observability/dashboards.md)
* 🚑 **Runbooks (incidents):** [docs/runbooks/](docs/runbooks/) *(start with [signature-spike.md](docs/runbooks/signature-spike.md), [dedup-outage.md](docs/runbooks/dedup-outage.md), [geo-failover.md](docs/runbooks/geo-failover.md))*

---

## Why It Reduces Hallucinations (and Spend)

1. **Detect** structure vs noise:

    * **D̂** (multi-scale Theil–Sen slope) highlights “signal-shaped” work.
    * **coh★** measures directional consistency of evidence.
    * **r** rewards compressible, internally consistent traces.

2. **Verify** claims before effects:

    * Signed PCS → **server recomputation** with tight tolerances.
    * **Verify-before-dedup** invariant ensures bad signatures can’t “stick.”

3. **Gate** actions by trust & budget:

    * Low trust ⇒ extra retrieval, smaller tool budgets, or human review.
    * High trust ⇒ fewer hops, lower latency, lower cost.

4. **Audit** everything:

    * Dual **WALs**, WORM audit trail, lineage of policy versions, and clear runbooks.

---

## Quickstart

1. **Agent SDK (Python)**

   ```bash
   pip install flk-pcs  # or local editable install
   ```

   Minimal send:

   ```python
   pcs = agent.compute_pcs(payload)
   signed = agent.sign(pcs)
   client.submit(signed)
   ```

2. **Verifier (Go)**

   ```bash
   make build
   make run-dev
   ```

   Health & metrics on `/healthz` and `/metrics`.

3. **Helm (dev cluster)**

   ```bash
   helm upgrade --install flk ./helm/flk -f ./helm/values.dev.yaml
   ```

Helpful docs:

* 🧪 **E2E tests:** [docs/testing/e2e.md](docs/testing/e2e.md)
* 📦 **Helm usage:** [docs/deploy/helm.md](docs/deploy/helm.md)
* 🧰 **Local dev (Compose/kind):** [docs/deploy/local.md](docs/deploy/local.md)

---

## Production Features (snapshots)

* **Fault tolerance:** inbox/outbox WAL, idempotent dedup, orderly crash-safe verification.
* **Security:** HMAC/Ed25519 signatures, TLS/mTLS, metrics Basic Auth, secret hygiene via SOPS/age.
* **Scale:** multi-tenant quotas, sharded & tiered dedup (Redis → Postgres → object storage), cross-region replication.
* **Observability:** Prometheus rules, Grafana dashboards, SLO burn alerts, SIEM streaming.
* **Governance:** policy DSL, canary rollouts, divergence detection, WORM lineage.

See:

* 🧱 **Invariants & guarantees:** [docs/architecture/invariants.md](docs/architecture/invariants.md)
* 🌍 **Multi-region & DR:** [docs/architecture/geo-dr.md](docs/architecture/geo-dr.md)
* 🧯 **SLOs & alerts:** [docs/observability/slos.md](docs/observability/slos.md)

---

## Roadmap (high level)

* **✅ Completed (Phases 1-10):** Core verification, E2E testing, multi-tenant governance, global scale, CRR/tiering, autonomous ops, real-time HRS, production ML, explainable risk, bandit ensembles, blocking anomalies, multi-cloud cost governance, **cross-language signature parity** (Phase 10), **atomic dedup** with first-write-wins, **thread-safe LRU caches**, HMAC simplification (sign payload directly).
* **Next (Phase 11+):** Real VRF verification (ECVRF), tokenized RAG overlap, JWT authentication, risk-based routing, fuzz testing, CI/CD hardening, Helm production readiness.

Progress notes live in:

* 🛣️ [docs/roadmap/phases.md](docs/roadmap/phases.md)
* 🔁 [docs/roadmap/changelog.md](docs/roadmap/changelog.md)

---

## Contributing

We love minimal, safe changes:

* Keep **verify → dedup** ordering.
* Don’t relax signature tolerances without updating **golden vectors**.
* Add tests for every new policy or signal path.

Start here:

* 🤝 [docs/contributing/guide.md](docs/contributing/guide.md)
* ✅ [docs/contributing/checklist.md](docs/contributing/checklist.md)

---

## License & Contact

* License: see [LICENSE](LICENSE).
* Security disclosures: [docs/security/overview.md](docs/security/overview.md) → **Responsible Disclosure**.
* Questions? Open a GitHub Discussion or ping maintainers listed in [CODEOWNERS](CODEOWNERS).

---

### One more line for the curious

We don’t make models behave. We **make their behavior accountable**. Plug it in, measure the drop in hallucinations, and watch your **cost-per-trusted-task** fall.
