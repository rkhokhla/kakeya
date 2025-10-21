# Testing: End-to-End (E2E)

## Overview

This document describes the end-to-end testing strategy for the Fractal LBA verification layer, covering Phase 2 integration tests, Phase 5 geo-DR tests, and Phase 6 chaos engineering tests.

## Test Suites

### 1. Backend Integration Tests (Phase 2)

**Location:** `tests/e2e/test_backend_integration.py`

**Test Coverage (15 test cases):**

#### HMAC Acceptance Tests
- `test_hmac_acceptance_golden_tiny_case_1`: Verify golden PCS from test vector 1
- `test_hmac_acceptance_golden_tiny_case_2`: Verify golden PCS from test vector 2
- Tests canonical signing (8-field subset, 9-decimal rounding)

#### Deduplication Tests
- `test_first_write_wins`: Send same PCS twice, verify idempotent response
- `test_dedup_cache_hit`: Verify cache hit metrics increment
- `test_ttl_expiration`: Verify dedup entries expire after TTL

#### Signature Rejection Tests
- `test_tampered_d_hat_rejected`: Modify D̂ after signing → 401 Unauthorized
- `test_tampered_merkle_root_rejected`: Modify merkle_root → 401
- `test_missing_signature_rejected`: Remove sig field → 401
- `test_invalid_signature_format`: Malformed base64 → 401

#### Verify-Before-Dedup Contract
- `test_verify_before_dedup_contract`: Invalid sig never written to dedup store
- Ensures Phase 1 invariant: signature verification happens before dedup write

#### WAL Integrity Tests
- `test_wal_written_before_parse`: Inbox WAL contains entry even if parse fails
- `test_wal_survives_crash`: Restart backend, verify WAL replay

#### Metrics Tests
- `test_metrics_flk_ingest_total`: Counter increments on each POST
- `test_metrics_flk_dedup_hits`: Counter increments on duplicate PCS
- `test_metrics_flk_accepted`: Counter increments on 200 response
- `test_metrics_flk_signature_errors`: Counter increments on 401

#### Health Checks
- `test_health_endpoint`: GET /health returns 200 OK
- `test_readiness_endpoint`: GET /ready checks dedup store connectivity

#### Ed25519 Path (Placeholder)
- `test_ed25519_acceptance`: Skipped pending Phase 2 keygen implementation
- `test_ed25519_rejection`: Skipped

**Run Tests:**

```bash
# Start test stack
docker-compose -f infra/compose-tests.yml up -d

# Run E2E tests
cd tests/e2e
pytest test_backend_integration.py -v

# Teardown
docker-compose -f infra/compose-tests.yml down -v
```

### 2. Geo-DR Tests (Phase 5)

**Location:** `tests/e2e/test_geo_dr.py`

**Test Coverage (5 scenarios):**

#### Normal CRR Operation
- `test_crr_normal_replication`: Write PCS to region A, verify replica in region B within 60s
- Validates CRR shipper/reader implementation

#### Region Failover
- `test_region_failover_rto`: Kill region A, verify traffic switches to region B
- Target RTO: ≤5 minutes
- Verifies health probe detection and DNS/load balancer failover

#### WAL Replay Idempotency
- `test_wal_replay_idempotent`: Replay same WAL segment twice, verify no duplicate effects
- Validates first-write-wins during recovery

#### Split-Brain Detection
- `test_split_brain_detection`: Partition regions, verify divergence alerts fire
- Ensures geo-split-brain runbook triggers correctly

#### RPO Compliance
- `test_rpo_compliance`: Kill region A mid-write, verify data loss ≤2 minutes
- Target RPO: ≤2 minutes
- Validates WAL durability and CRR lag SLO

**Run Tests:**

```bash
# Requires multi-region Docker Compose or Kubernetes
cd tests/e2e
pytest test_geo_dr.py -v --regions us-east-1,us-west-2
```

### 3. Chaos Engineering Tests (Phase 6)

**Location:** `tests/e2e/test_chaos.py`

**Test Coverage (6 scenarios):**

#### Shard Loss
- `test_shard_loss_failover`: Kill Redis shard, verify consistent hash routes to next shard
- Validates Phase 4 sharding implementation

#### WAL Lag Alert
- `test_wal_lag_alert`: Delay CRR shipper, verify alert fires when lag >60s
- Validates Phase 5 CRR monitoring

#### CRR Delay
- `test_crr_delay_recovery`: Introduce network delay, verify eventual convergence
- Validates retry logic and backoff

#### Cold Tier Outage
- `test_cold_tier_outage`: Disable S3, verify fallback to warm tier
- Validates Phase 4 tiering degraded mode

#### Dedup Overload
- `test_dedup_overload_503`: Flood dedup store, verify 503 with Retry-After
- Validates rate limiting and degraded mode

#### Dual-Write Failure
- `test_dual_write_failure_rollback`: Fail dual-write phase during shard migration
- Validates Phase 5 migration rollback

**Run Tests:**

```bash
# Requires chaos tooling (Chaos Mesh, Toxiproxy, or manual)
cd tests/e2e
pytest test_chaos.py -v --chaos-mode manual
```

## Test Infrastructure

### Docker Compose Test Stack

**Location:** `infra/compose-tests.yml`

Minimal stack for CI/E2E:
- **Backend:** Memory dedup (fast, disposable)
- **Redis:** For external dedup store tests
- **Prometheus:** Metrics collection
- **Grafana:** Optional dashboard access

**Optimizations:**
- Health checks: 2s interval (vs 10s in prod)
- No persistent volumes (ephemeral state)
- Single replica (no HA overhead)

### Test Data

#### Golden Files
**Location:** `tests/golden/`

- `pcs_tiny_case_1.json`: 15-row CSV with uniform-ish growth
- `pcs_tiny_case_2.json`: 20-row CSV with high x-axis coherence

Generated with:
```bash
python3 agent/src/cli/build_pcs.py \
  --in tests/data/tiny_case_1.csv \
  --out tests/golden/pcs_tiny_case_1.json \
  --key testsecret
```

#### Test Vectors
**Location:** `tests/data/`

- `tiny_case_1.csv`: Minimal PCS test case
- `tiny_case_2.csv`: High coherence test case
- Future: Add edge cases (empty, single row, extreme D̂)

## CI/CD Integration

### GitHub Actions Workflow

**Location:** `.github/workflows/ci.yml`

**Jobs (5 total):**

1. **unit-tests-python:**
   ```bash
   pytest tests/ -v  # 33 Phase 1 tests
   ```

2. **build-go:**
   ```bash
   cd backend && go build ./... && go test ./...
   ```

3. **e2e-tests:**
   ```bash
   docker-compose -f infra/compose-tests.yml up -d
   pytest tests/e2e/test_backend_integration.py -v
   docker-compose -f infra/compose-tests.yml down -v
   ```

4. **helm-lint:**
   ```bash
   helm lint helm/fractal-lba
   ```

5. **security-scan:**
   ```bash
   trufflehog filesystem . --json
   grep -r "PCS_HMAC_KEY=.*[^{]" . && exit 1  # Detect plaintext secrets
   ```

**Triggers:**
- Push to `main` branch
- Pull requests
- Manual workflow dispatch

**Artifacts:**
- E2E test logs (on failure)
- Helm lint output
- Security scan reports

## Test Best Practices

### 1. Idempotency

All tests must be idempotent:
- Clean up state between tests
- Use unique PCS IDs per test (vary `epoch` or `shard_id`)
- Avoid test order dependencies

### 2. Timeouts

Set reasonable timeouts:
- Unit tests: 5s per test
- Integration tests: 30s per test
- Geo-DR tests: 10m per test (failover takes time)

### 3. Assertions

Use specific assertions:
```python
# Good
assert response.status_code == 200
assert response.json()["accepted"] is True

# Bad
assert response.ok  # Too vague
```

### 4. Logging

Log context on failures:
```python
try:
    response = client.post("/v1/pcs/submit", json=pcs)
    assert response.status_code == 200
except AssertionError:
    print(f"Failed PCS: {json.dumps(pcs, indent=2)}")
    print(f"Response: {response.text}")
    raise
```

## Troubleshooting

### E2E Tests Fail with Connection Refused

**Symptom:** `requests.exceptions.ConnectionError: ('Connection aborted.', ConnectionRefusedError(111, 'Connection refused'))`

**Solution:**
1. Verify backend is running: `docker-compose -f infra/compose-tests.yml ps`
2. Check backend logs: `docker-compose -f infra/compose-tests.yml logs backend`
3. Verify health check: `curl http://localhost:8080/health`

### Dedup Tests Fail Intermittently

**Symptom:** `test_dedup_cache_hit` passes sometimes, fails others

**Solution:**
- Use unique PCS IDs per test run
- Add explicit cache flush between tests
- Increase dedup TTL to prevent premature expiration

### WAL Tests Fail on CI

**Symptom:** `test_wal_survives_crash` fails in GitHub Actions but passes locally

**Solution:**
- Increase fsync timeout (CI disk I/O slower)
- Use volume mounts for WAL directory
- Add retry logic for crash recovery

## Related Documentation

- [Unit Tests](./unit-tests.md)
- [Load Testing](./load-testing.md)
- [CI/CD Pipeline](./cicd.md)
- [Troubleshooting](../operations/troubleshooting.md)

## Runbooks

- [Test Failure Triage](../runbooks/test-failure-triage.md)
- [CI Pipeline Broken](../runbooks/ci-pipeline-broken.md)
