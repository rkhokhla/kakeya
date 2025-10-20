# CLAUDE.md — Implementation Playbook for Proposed Improvements

**Audience:** `claude-code`, codegen assistants, and contributors.
**Goal:** Implement a production-grade, reproducible signal/PCS pipeline and docs, as proposed in review. This file is the **single source of truth** for scope, contracts, and tasks.

---

## 1) Scope of this change (high level)

1. **Canonicalization for PCS signature**: define a stable subset of fields, JSON serialization, numeric rounding, and `pcs_id` rule.
2. **Signal computation clarifications**:

    * D̂ (Theil–Sen): log-log transform, non-decreasing `N_j`, median-of-slopes.
    * coh★ (directional coherence): direction sampling, binning, zero-width handling, reproducibility.
    * r (LZ compressibility): canonical row formatting and encoding.
3. **Regime thresholds & tolerances**: document clearly where they apply (verifier), and expose defaults.
4. **Test vectors & “golden files”**: add small CSVs, expected PCS JSON with valid HMAC signatures.
5. **Unit tests & property checks**: monotonic `N_j` vs scale, deterministic r, Theil–Sen properties, signature verification.
6. **Docs updates**: `docs/architecture/signal-computation.md` and cross-links to security (/metrics auth, signing).
7. **DevOps touch-ups**: minimal Compose/Helm snippets to wire HMAC + optional mTLS, with examples.

---

## 2) Contracts & invariants (DO NOT BREAK)

### 2.1 PCS identity & signature

* `pcs_id = sha256( merkle_root | "|" | epoch | "|" | shard_id )` (ASCII concat).
* Signature covers **only** this subset (exact keys, exact names):

  ```jsonc
  {
    "pcs_id": "...",
    "merkle_root": "...",
    "epoch": <int>,
    "shard_id": "...",
    "D_hat": <float>,     // round(., 9)
    "coh_star": <float>,  // round(., 9)
    "r": <float>,         // round(., 9)
    "budget": <float>     // round(., 9)
  }
  ```
* **Serialization for signing**:

    * JSON with **sorted keys**, **no spaces** (`separators=(',', ':')`),
    * numbers pre-rounded to **9 decimal places** (round half away from zero not required; standard round is fine as long as both sides match),
    * UTF-8 bytes; take **SHA-256** digest of that exact byte sequence.
* Algorithms:

    * **HMAC-SHA256** on agents (recommended).
    * **Ed25519** verification on backend/gateway (optional).
* Verifier **rejects** signature before any stateful side effects or dedup write.

### 2.2 Signals

* **D̂ (Theil–Sen)**:

    * For each scale `s ∈ scales`, compute `N_j[s] = unique non-empty voxel count`.
    * Regress over points ( (x_i, y_i) ) where `x_i = log2(s)`, `y_i = log2(max(1, N_j[s]))`.
    * Slope = **median** of all pairwise slopes ((y_j - y_i)/(x_j - x_i)), (j>i).
* **coh★**:

    * Sample M random unit directions uniformly on the sphere (or normal vectors normalized).
    * Project points, histogram into `bins_per_level` linearly between min/max.
    * For each direction, `coh = max_bin_count / n_points`; coh★ = max over directions.
    * If `pmax==pmin`: use width=1.0 to avoid div-by-zero → single-bin behavior.
* **r (LZ ratio)**:

    * Build **canonical rows** (e.g., `"t,k,v"` using `.` as decimal separator), join with `\n`, encode UTF-8.
    * Compute `r = len(zlib.compress(raw, level=6)) / len(raw)`, with guard: if `len(raw)==0` then `r=1.0`.

### 2.3 Regimes, tolerances & budget

* **Regimes**:

    * `sticky` if `coh★ ≥ 0.70` **and** `D̂ ≤ 1.5`
    * `non_sticky` if `D̂ ≥ 2.6`
    * else `mixed`
* **Budget** (clamped to [0,1]):
  `budget = base + α(1 − r) + β·max(0, D̂ − D0) + γ·coh★`
* **Tolerances** apply **only** in verifier:

    * `tolD = 0.15`, `tolCoh = 0.05` (defaults)
    * Accept if `|D̂_recomputed − D̂| ≤ tolD` and `0 ≤ coh★ ≤ 1 + tolCoh`.

---

## 3) Directory and file changes

> Adjust to match the repo’s actual structure. Paths shown are typical.

```
docs/
  architecture/
    signal-computation.md           # UPDATE (sections listed below)
    signing-canonicalization.md     # NEW (optional split, or inline into signal-computation.md)
observability/
  tests/
    data/
      tiny_case_1.csv               # NEW
      tiny_case_2.csv               # NEW
    golden/
      pcs_tiny_case_1.json          # NEW (HMAC-signed)
      pcs_tiny_case_2.json          # NEW (HMAC-signed)
    test_signals.py                 # NEW (unit tests)
    test_signing.py                 # NEW (signature tests)
src/agent/                           # adjust to Python dirs used in repo
  utils/
    canonical_json.py               # NEW (stable dumps, 9-digit rounding)
    signing.py                      # NEW (HMAC signing helper)
  cli/
    build_pcs.py                    # UPDATE (use canonicalization, produce sig)
src/verifier/                        # adjust to Go dirs used in repo
  verify/
    canonical.go                    # NEW (stable JSON + rounding to compare/sign)
    signverify.go                   # NEW (HMAC/Ed25519 verify)
  handlers/
    submit.go                       # UPDATE (verify sig before dedup/effects)
infra/
  compose-examples/
    docker-compose.hmac.yml         # NEW (wiring HMAC env)
  helm/
    values-snippets.md              # NEW (flags for signing + metrics auth)
```

---

## 4) Tasks for `claude-code` (step-by-step)

### T1 — Canonicalization utilities

**Python** (`src/agent/utils/canonical_json.py`)

* Implement `dumps_canonical(obj: dict) -> bytes`

    * Round `D_hat`, `coh_star`, `r`, `budget` **only when** serializing the signature subset.
    * Use `json.dumps(..., sort_keys=True, separators=(',', ':'))`.
    * Return UTF-8 bytes.

**Python** (`src/agent/utils/signing.py`)

* Implement:

    * `signature_payload(pcs: dict) -> bytes` → extract subset, pre-round floats (9 dp), canonical dump.
    * `sign_hmac(payload: bytes, key: bytes) -> str` → base64 std encoding of HMAC-SHA256.
* Unit tests for both.

**Go** (`src/verifier/verify/canonical.go`)

* Implement `SignaturePayload(pcs PCS) ([]byte, error)`.

    * Build a map with the exact subset (rounded to 9 decimals).
    * Marshal with sorted keys (use a fixed struct or manual ordering → marshal as bytes).
    * SHA-256 digest is computed **elsewhere**.

**Go** (`src/verifier/verify/signverify.go`)

* Implement:

    * `VerifyHMAC(digest []byte, sigB64 string, key []byte) error`
    * `VerifyEd25519(digest []byte, sigB64 string, pubKey []byte) error`
* Return strict errors; no logging of payload.

### T2 — Update signal computation doc

**`docs/architecture/signal-computation.md`**:

* Add a **“Canonicalization & Signing”** section:

    * subset keys, rounding to 9dp, sorted keys JSON, ASCII concatenation for `pcs_id`.
* Clarify D̂ computation: log2 transform of `max(1, N_j[s])`, median of pairwise slopes.
* Clarify coh★: direction sampling method, `bins_per_level` defaults (e.g., 64), zero-range behavior, reproducible RNG seed.
* Clarify r: canonical row format `"t,k,v"`, newline `\n`, UTF-8 encoding, `level=6`, empty stream rule.
* Move regime thresholds here or cross-link to a regimes doc.
* State explicitly: **tolerances** are enforced on **verifier**.

### T3 — Test vectors & golden files

* Create two tiny CSVs:

    * `tiny_case_1.csv`: uniform-ish growth.
    * `tiny_case_2.csv`: more coherent, to produce higher coh★.
* Create Python script (if not present) to generate PCS from CSV and **sign via HMAC** with a sample `PCS_HMAC_KEY="testsecret"`.
* Compute expected PCS (`golden/*.json`) with **exact** sig values.
* Add `Makefile` targets:

  ```make
  .PHONY: golden verify-golden
  golden:
    python -m src.agent.cli.build_pcs --in observability/tests/data/tiny_case_1.csv \
      --out observability/tests/golden/pcs_tiny_case_1.json --key testsecret
    # repeat for case_2
  verify-golden:
    python scripts/verify_golden.py  # compares byte-for-byte canonical subset + signature
  ```

### T4 — Unit tests (pytest)

* `test_signals.py`:

    * **Monotonicity**: assert `N_j[s]` is non-decreasing with `s`.
    * **Determinism r**: repeated computation yields identical `r` for same input.
    * **Theil–Sen properties** on synthetic data (increasing trend yields positive slope).
    * **coh★ stability**: with fixed seed, values are stable within ε for small data.
* `test_signing.py`:

    * Canonicalize subset → digest → HMAC → compare to expected base64 (for a small fixture PCS).
    * Negative test: change one digit, signature must fail.

### T5 — Verifier path (Go)

* In `handlers/submit.go` (or equivalent):

    1. Append **Inbox WAL** (if present) before parsing.
    2. Parse PCS JSON.
    3. If signing enabled, compute `payload` & SHA-256 → verify HMAC/Ed25519.
    4. Dedup by `pcs_id`; return cached outcome on hit.
    5. Recompute D̂; check tolerances; recompute budget; classify regime (for observability only if needed).
    6. Store outcome with TTL; emit metrics; respond **200** (accepted) or **202** (escalated).

### T6 — DevOps snippets

* **Compose example** `infra/compose-examples/docker-compose.hmac.yml`:

    * Backend env: `PCS_SIGN_ALG=hmac`, `PCS_HMAC_KEY=${PCS_HMAC_KEY}`
    * Agent env: `PCS_SIGN_ALG=hmac`, `PCS_HMAC_KEY=${PCS_HMAC_KEY}`
    * Note: Use `.env` for secrets in dev; **do not commit real secrets**.
* **Helm** `infra/helm/values-snippets.md`:

    * Show toggles:

      ```yaml
      signing:
        enabled: true
        alg: hmac
        hmacKeySecretName: flk-sign-hmac
      metricsBasicAuth:
        enabled: true
        secretName: flk-metrics-auth
      ```
    * `kubectl create secret generic flk-sign-hmac --from-literal=key='supersecret'`

---

## 5) Acceptance criteria (Definition of Done)

* [ ] `signal-computation.md` updated with canonicalization, D̂/coh★/r clarifications, regimes, tolerances.
* [ ] New canonicalization/signing helpers in Python & Go with tests.
* [ ] Two CSV fixtures + two golden PCS JSON (with HMAC) committed.
* [ ] `pytest` passes locally and in CI; includes monotonicity/determinism/signature tests.
* [ ] Verifier rejects invalid signatures and never writes dedup on signature failure.
* [ ] Example Compose/Helm snippets compile/apply (syntax-only check).
* [ ] No personally identifiable payload data printed in logs; metrics still functional.
* [ ] Linting & formatting pass.

---

## 6) Commands (local dev)

```bash
# Python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q

# Generate & verify golden files
make golden
make verify-golden

# Go (if present)
go mod tidy
go build ./...

# Docker Compose (example)
export PCS_HMAC_KEY=testsecret
docker compose -f infra/compose-examples/docker-compose.hmac.yml up -d --build
```

---

## 7) Code skeletons

### 7.1 Python canonicalization & signing

```python
# src/agent/utils/canonical_json.py
import json
from decimal import Decimal, ROUND_HALF_UP

def round9(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.000000001"), rounding=ROUND_HALF_UP))

SIGN_KEYS = ("pcs_id","merkle_root","epoch","shard_id","D_hat","coh_star","r","budget")

def signature_subset(pcs: dict) -> dict:
    sub = {k: pcs[k] for k in SIGN_KEYS}
    for k in ("D_hat","coh_star","r","budget"):
        sub[k] = round9(sub[k])
    return sub

def dumps_canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",",":")).encode("utf-8")
```

```python
# src/agent/utils/signing.py
import base64, hmac, hashlib
from .canonical_json import signature_subset, dumps_canonical

def signature_payload(pcs: dict) -> bytes:
    return dumps_canonical(signature_subset(pcs))

def sign_hmac(pcs: dict, key: bytes) -> str:
    payload = signature_payload(pcs)
    digest = hashlib.sha256(payload).digest()
    sig = hmac.new(key, digest, hashlib.sha256).digest()
    return base64.b64encode(sig).decode()
```

### 7.2 Go verification (HMAC)

```go
// verify/signverify.go
package verify

import (
  "crypto/hmac"
  "crypto/sha256"
  "encoding/base64"
  "errors"
)

var ErrBadHMAC = errors.New("bad hmac")

func VerifyHMAC(digest []byte, sigB64 string, key []byte) error {
  mac := hmac.New(sha256.New, key)
  mac.Write(digest)
  exp := mac.Sum(nil)
  got, err := base64.StdEncoding.DecodeString(sigB64)
  if err != nil { return err }
  if !hmac.Equal(exp, got) { return ErrBadHMAC }
  return nil
}
```

```go
// verify/canonical.go (subset assembly + SHA-256)
package verify

import (
  "crypto/sha256"
  "encoding/json"
  "math"
  "sort"
)

type Subset struct {
  Budget   float64 `json:"budget"`
  CohStar  float64 `json:"coh_star"`
  DHat     float64 `json:"D_hat"`
  Epoch    int     `json:"epoch"`
  Merkle   string  `json:"merkle_root"`
  PCSId    string  `json:"pcs_id"`
  R        float64 `json:"r"`
  ShardID  string  `json:"shard_id"`
}

func round9(x float64) float64 { return math.Round(x*1e9) / 1e9 }

func SignatureDigest(p PCS) ([]byte, error) {
  sub := Subset{
    Budget:  round9(p.Budget),
    CohStar: round9(p.CohStar),
    DHat:    round9(p.DHat),
    Epoch:   p.Epoch,
    Merkle:  p.MerkleRoot,
    PCSId:   p.PCSId,
    R:       round9(p.R),
    ShardID: p.ShardId,
  }
  // Marshal with stable order: struct field order is already fixed.
  b, err := json.Marshal(sub)
  if err != nil { return nil, err }
  h := sha256.Sum256(b)
  return h[:], nil
}
```

---

## 8) CI additions

* Run `pytest` with coverage threshold (e.g., 80% for new modules).
* Optional: `pre-commit` hooks for black/ruff/isort (Python) and `gofmt`/`golangci-lint` (Go).
* Ensure `golden` verification runs in CI to detect canonicalization drift.

---

## 9) Security & privacy checklist

* [ ] No raw PCS payloads logged.
* [ ] Secrets come from env/Secrets, not committed.
* [ ] Signature verified **before** dedup/outcome write.
* [ ] `/metrics` guarded (Basic Auth or network filtering).
* [ ] Test vectors use **non-sensitive** synthetic data.

---

## 10) PR description template

**Title:** Canonical PCS signing, clarified signal math, tests & golden files

**Summary:**

* Define canonical signature subset (9dp rounding, sorted keys JSON).
* Clarify D̂, coh★, r computation; document regimes and tolerances on verifier.
* Add fixtures + golden PCS with HMAC; unit tests for monotonicity, determinism, and signing.
* Provide Compose/Helm snippets for HMAC + metrics auth.

**Why:**
Ensure reproducibility across languages/runtimes and make verification cryptographically robust.

**How tested:**

* `pytest` for signals and signing.
* Golden verification matches byte-for-byte.
* Manual run via Compose HMAC overlay.

**Breaking changes:**
None to runtime API; **documentation and internal helpers only**. PCS schema version unchanged. (If any field semantics changed, bump `version` and call it out.)

---

## 11) Future follow-ups (not in this PR)

* Latency histograms for verifier; more Prometheus metrics.
* Ed25519 signing on gateway, multikey verification window.
* SOPS/age for secret management; KMS integration.
* Chaos tests (dedup outage, duplicate floods).

---

### Final notes for `claude-code`

* Be meticulous with **numeric rounding** and **JSON canonicalization**—drift here breaks signatures.
* Keep changes **minimal and composable**; prefer new helpers over inlining logic everywhere.
* Include concise doc snippets and comments where ambiguity is likely (e.g., zero-width projections).
* If repo structure deviates, adapt paths but preserve **contract** and **tests**.
