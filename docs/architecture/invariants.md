# Architecture: Invariants & Guarantees

## Overview

This document catalogs the safety invariants and behavioral guarantees of the Fractal LBA verification layer. These invariants MUST be preserved across all phases and changes.

## Phase 1 Core Invariants

### 1. PCS Identity

**Invariant:** `pcs_id = SHA256(merkle_root || epoch || shard_id)`

- ASCII concatenation with `|` separator
- Immutable after creation
- Used as deduplication key

**Violation Impact:** Deduplication breaks, duplicate effects possible

### 2. Canonical Signing (Phase 1+)

**Invariant:** Signature computed over 8-field subset with 9-decimal rounding

**Signature Subset:**
1. `pcs_id`
2. `merkle_root`
3. `epoch`
4. `shard_id`
5. `D_hat` (rounded to 9 decimals)
6. `coh_star` (rounded to 9 decimals)
7. `r` (rounded to 9 decimals)
8. `budget` (rounded to 9 decimals)

**Serialization:**
- JSON with sorted keys
- No whitespace
- Example: `{"D_hat":1.410000000,"budget":0.420000000,...}`

**Violation Impact:** Signature verification fails, agent/backend divergence

### 3. WAL-First Write Ordering

**Invariant:** Inbox WAL append happens BEFORE request parsing

```
1. Receive HTTP POST
2. Append raw body to Inbox WAL (fsync)
3. Parse JSON
4. Verify signature
5. Compute dedup outcome
```

**Rationale:** Crash recovery, audit trail, replay capability

**Violation Impact:** Lost requests on crash, incomplete audit

### 4. Verify-Before-Dedup Contract

**Invariant:** Signature verification happens BEFORE dedup write

```
1. Parse PCS
2. Verify signature (401 if invalid)
3. Check dedup store
4. If miss: compute verification, write outcome to dedup
5. If hit: return cached outcome
```

**Rationale:** Invalid signatures never cached, prevents pollution

**Violation Impact:** Invalid PCS cached, security bypass

### 5. First-Write Wins Idempotency

**Invariant:** First PCS with given `pcs_id` determines outcome

- Subsequent requests return cached outcome with same HTTP status
- TTL-based expiration (default: 14 days)

**Violation Impact:** Non-deterministic outcomes, audit confusion

## Phase 3 Multi-Tenant Invariants

### 6. Per-Tenant Isolation

**Invariant:** Tenant resources never cross boundaries

- Separate signing keys per tenant
- Per-tenant quotas (token bucket + daily cap)
- Per-tenant metrics labels

**Violation Impact:** Quota leakage, cross-tenant data exposure

### 7. WORM Audit Integrity

**Invariant:** Audit logs are write-once, append-only

- File permissions: 0444 (read-only after write)
- Entry hash: SHA256 of entry JSON
- Segment root: Merkle root for external anchoring

**Violation Impact:** Audit tampering, compliance violations

## Phase 4 Geo-Replication Invariants

### 8. CRR Idempotent Replay

**Invariant:** WAL segments replay idempotently across regions

- First-write wins applies during replay
- Duplicate deliveries safe
- Out-of-order segments safe (within ordering constraints)

**Violation Impact:** Divergent outcomes, split-brain

### 9. At-Least-Once Delivery

**Invariant:** Every PCS eventually replicated to all target regions

- Outbox WAL persists before HTTP POST
- Exponential backoff + jitter on retry
- DLQ for exhausted retries

**Violation Impact:** Data loss, RPO violations

## Phase 7+ ML/HRS Invariants

### 10. HRS Model Immutability

**Invariant:** Model binaries are immutable once registered

- SHA-256 hash computed on registration
- Files written as read-only (0444)
- Version changes require new registration

**Violation Impact:** Non-reproducible predictions, audit trails break

### 11. Feature PI-Safety

**Invariant:** HRS features contain no PII or raw content

- Only derived signals (D̂, coh★, r, etc.)
- No text, embeddings, or user identifiers

**Violation Impact:** Privacy violations, GDPR non-compliance

## Behavioral Guarantees

### Latency

- **p95 verify latency ≤ 200ms** (Phase 1-8 maintained)
- **HRS prediction p95 ≤ 10ms** (Phase 7+)
- **Ensemble p95 ≤ 120ms** (Phase 8)

### Correctness

- **Escalation rate ≤ 2%** (error budget)
- **Hallucination containment ≥ 98%** (Phase 6+)
- **Ensemble agreement ≥ 85%** (Phase 7+)

### Cost

- **Cost per trusted task** tracked per tenant (Phase 7+)
- **Billing reconciliation within ±3%** (Phase 8)
- **Forecast MAPE ≤ 10%** (Phase 8)

### Availability

- **RTO ≤ 5 minutes** (geo-failover, Phase 4+)
- **RPO ≤ 2 minutes** (CRR lag SLO, Phase 4+)

## Invariant Validation

### Unit Tests

All invariants have dedicated unit tests:
- `test_pcs_id_stability` (Phase 1)
- `test_canonical_signing_subset` (Phase 1)
- `test_verify_before_dedup_contract` (Phase 2 E2E)
- `test_first_write_wins` (Phase 2 E2E)
- `test_worm_immutability` (Phase 3)
- `test_crr_idempotent_replay` (Phase 5)

### Integration Tests

E2E tests validate end-to-end invariant preservation:
- `test_backend_integration.py`: 15 test cases
- `test_geo_dr.py`: 5 geo scenarios
- `test_chaos.py`: 6 failure scenarios

### Formal Verification (Phase 6)

- **TLA+ specification:** `formal/crr_idempotency.tla`
  - Models CRR, dedup, WAL
  - Proves idempotency and first-write wins

- **Coq proofs:** `formal/canonical_signing.v`
  - 8 lemmas on canonical signing
  - Main theorem: signature protocol soundness

## Monitoring & Alerts

### Invariant Violation Alerts

**Prometheus alerts fire on invariant violations:**

- `InvariantViolation_PCSIdCollision`: Different merkle_roots with same pcs_id
- `InvariantViolation_VerifyAfterDedup`: Dedup hit before signature check (bug)
- `InvariantViolation_WORMModified`: Audit file mtime changed after creation
- `InvariantViolation_CRRDivergence`: Regions have conflicting outcomes for pcs_id

### Runbooks

Each alert links to runbook:
- [PCS ID Collision](../runbooks/pcs-id-collision.md)
- [Verify Order Bug](../runbooks/verify-order-bug.md)
- [WORM Tampering](../runbooks/worm-tampering.md)
- [Geo Split-Brain](../runbooks/geo-split-brain.md)

## Change Management

### Breaking Changes

**Any change violating an invariant requires:**
1. Architecture review
2. Formal proof update (if applicable)
3. Version bump (MAJOR)
4. Migration plan
5. Backward compatibility period

### Adding Invariants

**Process for new invariants:**
1. Document in this file
2. Add unit tests
3. Add integration tests
4. Add monitoring/alerts
5. Update CLAUDE.md

## Related Documentation

- [Architecture Overview](./overview.md)
- [Signal Computation](./signal-computation.md)
- [Security Model](../security/overview.md)
- [Testing Strategy](../testing/e2e.md)
