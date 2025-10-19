# Troubleshooting Guide

Comprehensive guide for diagnosing and resolving common issues.

## Diagnostic Checklist

When encountering issues, systematically check:

1. ✅ **Service Health**: All containers/pods running?
2. ✅ **Connectivity**: Can agent reach backend?
3. ✅ **Logs**: Any error messages?
4. ✅ **Metrics**: Anomalies in Prometheus?
5. ✅ **Configuration**: Environment variables correct?
6. ✅ **Resources**: CPU/memory/disk sufficient?

---

## Common Issues

### 1. Backend Returns 429 (Rate Limited)

**Symptom**:
```bash
curl -X POST http://localhost:8080/v1/pcs/submit -d @pcs.json
# Returns: HTTP 429 Too Many Requests
# Headers: Retry-After: 10
```

**Metrics**:
- High request rate in Grafana
- Agent sees repeated 429 responses

**Root Cause**:
Token bucket exhausted due to:
- Burst traffic exceeding `TOKEN_RATE`
- Too many agents submitting concurrently
- Replay attack or runaway loop

**Solution**:

1. **Check current rate**:
```bash
# Prometheus query
rate(flk_ingest_total[1m]) * 60
```

2. **Increase rate limit** (temporary):
```bash
# Docker Compose
export TOKEN_RATE=500
docker-compose up -d backend

# Kubernetes
kubectl set env deployment/backend TOKEN_RATE=500
```

3. **Scale horizontally**:
```bash
# Kubernetes
kubectl scale deployment/backend --replicas=5
```

4. **Fix agent behavior**:
- Check for retry loops
- Ensure exponential backoff is working
- Verify deduplication on agent side

**Prevention**:
- Set `TOKEN_RATE` based on expected load
- Configure HPA in Kubernetes
- Monitor `rate(flk_ingest_total[1m])`

---

### 2. Signature Verification Failures (401)

**Symptom**:
```
HTTP 401 Unauthorized
{"error": "Signature verification failed"}
```

**Metrics**:
```prometheus
flk_signature_errors > 0
```

**Root Causes**:

#### A. Algorithm Mismatch

**Check**:
```bash
# Backend
echo $PCS_SIGN_ALG  # Should be: hmac or ed25519

# Agent
grep PCS_SIGN_ALG config.env
```

**Solution**: Ensure both use same algorithm.

#### B. Key Mismatch

**Check**:
```bash
# Backend HMAC key
echo $PCS_HMAC_KEY

# Agent HMAC key
grep PCS_HMAC_KEY agent/.env
```

**Solution**: Keys must match exactly (case-sensitive).

#### C. Numeric Rounding Inconsistency

**Root Cause**: Agent uses wrong precision for D̂, coh★, r, budget.

**Verify**:
```python
# Must round to 9 decimals
def round_9(x):
    return round(x, 9)

# Example
D_hat = 1.4123456789012  # Too precise
D_hat = round_9(D_hat)   # Correct: 1.412345679
```

**Solution**: Update agent to use 9-decimal rounding.

#### D. JSON Serialization Mismatch

**Requirements**:
- Sorted keys
- No whitespace
- Consistent encoding

**Verify**:
```python
import json

payload = {
    "pcs_id": "abc123",
    "merkle_root": "def456",
    "epoch": 1,
    "shard_id": "shard-001",
    "D_hat": 1.412345679,
    "coh_star": 0.734567890,
    "r": 0.871234567,
    "budget": 0.421234567
}

# Correct serialization
canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
print(canonical)
```

**Solution**: Use exact JSON format from CLAUDE.md.

---

### 3. High Escalation Rate (>2%)

**Symptom**:
```prometheus
(flk_escalated / flk_ingest_total) * 100 > 2
```

**Metrics**:
- Grafana SLO gauge shows yellow/red
- Many 202 responses

**Root Causes**:

#### A. D̂ Out of Tolerance

**Error Message**:
```
"reason": "D_hat out of tolerance: claimed=1.45, recomputed=1.20, diff=0.25 > tol=0.15"
```

**Investigation**:
```bash
# Check scales and N_j
cat pcs.json | jq '.scales, .N_j'
```

**Common Issues**:
- Missing scale in N_j map
- N_j computation bug (zero or negative values)
- Integer overflow in large scales

**Solution**:
- Verify Theil-Sen implementation
- Check N_j keys are strings (`"8"` not `8`)
- Increase tolerance (last resort):
  ```go
  params := api.VerifyParams{
      TolD: 0.20,  // Increased from 0.15
  }
  ```

#### B. Regime Classification Mismatch

**Error Message**:
```
"reason": "regime mismatch: claimed=mixed, expected=sticky"
```

**Investigation**:
```bash
# Check D̂ and coh★ values
cat pcs.json | jq '{D_hat, coh_star, regime}'
```

**Regime Rules**:
- `sticky`: coh★ ≥ 0.70 AND D̂ ≤ 1.5
- `non_sticky`: D̂ ≥ 2.6
- `mixed`: otherwise

**Solution**:
- Verify regime logic in agent matches backend
- Check for floating-point precision issues

#### C. Budget Deviation

**Error Message**:
```
"reason": "budget deviation: claimed=0.42, recomputed=0.35"
```

**Investigation**:
```go
// Budget formula
budget = base + alpha*(1-r) + beta*max(0, D̂-D0) + gamma*coh★
budget = 0.10 + 0.30*(1-r) + 0.50*max(0, D̂-2.2) + 0.20*coh★
```

**Solution**:
- Verify agent uses same formula
- Check all parameters (alpha, beta, gamma, base, D0)
- Ensure clamping to [0, 1]

---

### 4. WAL Disk Growth

**Symptom**:
```bash
df -h /data/wal
# Shows 90%+ usage
```

**Metrics**:
- Disk usage alerts
- Slow write performance

**Root Causes**:

#### A. No WAL Compaction (Agent)

**Check**:
```bash
# Agent outbox
ls -lh data/outbox.wal
```

**Solution**:
```python
from agent.src.outbox import OutboxWAL

wal = OutboxWAL('data/outbox.wal')
wal.compact(horizon_days=14)  # Remove acked entries >14d old
```

**Automate** (cron job):
```bash
# Daily compaction
0 2 * * * python -c "from agent.src.outbox import OutboxWAL; OutboxWAL('data/outbox.wal').compact()"
```

#### B. No WAL Rotation (Backend)

**Check**:
```bash
# Backend inbox
ls -lh /data/wal/
```

**Solution**:
```bash
# Manual rotation
mv inbox-20250101.wal inbox-20250101.wal.old
```

**Automate** (logrotate):
```
/data/wal/*.wal {
    daily
    rotate 14
    compress
    missingok
    notifempty
}
```

---

### 5. Dedup Store Outage

**Symptom**:
```
HTTP 503 Service Unavailable
{"error": "dedup store unavailable"}
```

**Logs**:
```
Error reaching Redis: connection refused
```

**Root Cause**:
- Redis/Postgres down
- Network partition
- Credentials invalid

**Immediate Action**:

1. **Check store health**:
```bash
# Redis
redis-cli ping

# Postgres
psql -U flk -d fractal_lba -c "SELECT 1"
```

2. **Restart store** (if down):
```bash
docker-compose restart redis
# or
kubectl rollout restart statefulset/redis
```

3. **Verify connectivity**:
```bash
# From backend pod
kubectl exec -it backend-xxx -- sh
nc -zv redis 6379
```

**Recovery**:
- Inbox WAL preserves requests during outage
- After recovery, backend resumes processing
- Dedup cache rebuilds from TTL'd entries

**Prevention**:
- Monitor store health
- Set up Redis Sentinel/Cluster for HA
- Configure circuit breakers

---

### 6. Memory Leaks

**Symptom**:
```bash
# Backend memory grows unbounded
kubectl top pods
# Shows: backend-xxx  500Mi/512Mi  (97%)
```

**Investigation**:

1. **Check Go runtime**:
```bash
curl http://localhost:8080/debug/pprof/heap > heap.prof
go tool pprof heap.prof
```

2. **Check dedup cache** (if memory backend):
```bash
# Cache should respect TTL
# Expired entries should be garbage collected
```

**Root Causes**:
- Memory dedup backend not expiring entries
- WAL not being closed/rotated
- Goroutine leaks

**Solution**:

1. **Switch to external dedup**:
```yaml
environment:
  - DEDUP_BACKEND=redis
```

2. **Increase memory limits**:
```yaml
resources:
  limits:
    memory: 1Gi
```

3. **Enable profiling**:
```go
import _ "net/http/pprof"

http.ListenAndServe(":6060", nil)
```

---

### 7. Clock Skew Issues

**Symptom**:
```
PCS rejected due to timestamp in future
```

**Investigation**:
```bash
# Check agent time
date -u

# Check backend time
kubectl exec backend-xxx -- date -u

# Diff should be <1 second
```

**Root Cause**:
- Agent/backend clocks out of sync
- No NTP service

**Solution**:

1. **Enable NTP** (agent):
```bash
sudo systemctl enable chronyd
sudo systemctl start chronyd
```

2. **Sync time** (manual):
```bash
sudo ntpdate pool.ntp.org
```

3. **K8s node time sync**:
```bash
# Ensure node clocks are synced
kubectl get nodes -o wide
```

**Note**: Signature verification is **time-independent**, but `sent_at` timestamps may be logged for audit purposes.

---

## Log Analysis

### Backend Logs

**Key Patterns**:

```bash
# Signature failures
grep "Signature verification failed" backend.log

# Dedup hits (good)
grep "duplicate" backend.log

# WAL errors (critical)
grep "WAL append error" backend.log

# Verification failures
grep "verification error" backend.log
```

**Example**:
```
2025-01-19T10:30:45Z INFO PCS abc123 accepted (D̂=1.41, coh★=0.73)
2025-01-19T10:30:46Z WARN PCS def456 escalated: regime mismatch
2025-01-19T10:30:47Z ERROR Signature verification failed for ghi789
```

### Agent Logs

**Key Patterns**:

```python
# Submission success
"PCS abc123 submitted successfully (status=200)"

# Retries
"Backing off for 2.5s before retry"

# DLQ entries (bad)
"Added pcs_id to DLQ: all retries exhausted"
```

---

## Prometheus Queries

### Request Rate

```prometheus
rate(flk_ingest_total[5m]) * 60
```

### Error Rate

```prometheus
rate(flk_signature_errors[5m]) / rate(flk_ingest_total[5m])
```

### Escalation Rate (SLO)

```prometheus
(flk_escalated / flk_ingest_total) * 100
```

### Dedup Hit Ratio

```prometheus
(flk_dedup_hits / flk_ingest_total) * 100
```

---

## Getting Help

If none of these solutions work:

1. **Gather diagnostics**:
```bash
# Backend
kubectl logs backend-xxx > backend.log
kubectl describe pod backend-xxx > backend-describe.txt
curl http://localhost:8080/metrics > metrics.txt

# Agent
cat data/outbox.wal > outbox-wal.txt
cat data/dlq.jsonl > dlq.txt
```

2. **Open GitHub issue** with:
- Description of problem
- Logs (redact secrets!)
- Configuration (env vars, Helm values)
- Metrics snapshot
- Steps to reproduce

3. **Check CLAUDE.md** for design invariants:
- Are you violating any contracts?
- Is the PCS schema correct?

---

## Prevention

### Monitoring Alerts

Set up alerts for:

```yaml
- alert: HighEscalationRate
  expr: (flk_escalated / flk_ingest_total) * 100 > 2
  for: 5m
  annotations:
    summary: "Escalation rate exceeds SLO"

- alert: SignatureFailures
  expr: rate(flk_signature_errors[5m]) > 0.1
  for: 1m
  annotations:
    summary: "Signature verification failing"

- alert: WALErrors
  expr: rate(flk_wal_errors[5m]) > 0
  for: 1m
  annotations:
    summary: "WAL writes failing"
```

### Chaos Testing

Regularly test failure scenarios:

```bash
# Kill Redis
docker-compose stop redis
# Expect: 503 responses, requests in WAL

# Restart after 60s
sleep 60
docker-compose start redis
# Expect: Processing resumes, no data loss
```

---

## Next Steps

- [Operations Guide](operations-guide.md) - Routine management
- [Runbooks](runbooks.md) - Incident response procedures
- [Monitoring](monitoring.md) - Observability deep-dive
