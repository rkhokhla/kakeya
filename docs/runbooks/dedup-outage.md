# Runbook: Deduplication Store Outage

**Alert:** `FLKRedisDown`, `FLKPostgresDown`, `FLKDedupAnomalyLow`
**Severity:** Critical
**Component:** Dedup store (Redis/PostgreSQL)

## Symptoms

- Backend returning 503 Service Unavailable
- `flk_dedup_hits` counter stopped incrementing
- Logs showing "Dedup store error: connection refused"
- WAL growing rapidly (no dedup writes succeeding)

## Impact

- **High:** All PCS submissions failing or returning 503
- WAL disk filling up (risk of disk full → complete outage)
- No idempotency enforcement (duplicate PCS processed multiple times)
- Potential data inconsistency

## Immediate Actions

### 1. Assess Severity

```bash
# Check if backend is still accepting requests
curl -I https://api.fractal-lba.example.com/health

# Check error rate
kubectl logs -n fractal-lba -l app=fractal-lba-backend --tail=100 | grep -c "503"

# Check WAL disk usage
kubectl exec -n fractal-lba deployment/fractal-lba-backend -- df -h /data/wal
```

### 2. Enable Degraded Mode (If Available)

```bash
# Temporarily disable dedup requirement (if backend supports)
kubectl set env deployment/fractal-lba-backend DEDUP_REQUIRED=false -n fractal-lba

# Or: Switch to in-memory dedup
kubectl set env deployment/fractal-lba-backend DEDUP_BACKEND=memory -n fractal-lba
```

**Note:** This loses idempotency guarantees but keeps system available.

### 3. Check Dedup Store Health

**For Redis:**
```bash
# Check Redis pod status
kubectl get pods -n fractal-lba -l app=redis

# Check Redis logs
kubectl logs -n fractal-lba -l app=redis --tail=100

# Test Redis connectivity
kubectl run redis-test --rm -i --tty --image=redis:7-alpine -n fractal-lba -- \
  redis-cli -h redis-master ping
```

**For PostgreSQL:**
```bash
# Check Postgres pod status
kubectl get pods -n fractal-lba -l app=postgresql

# Check Postgres logs
kubectl logs -n fractal-lba -l app=postgresql --tail=100

# Test Postgres connectivity
kubectl run pg-test --rm -i --tty --image=postgres:15-alpine -n fractal-lba -- \
  psql -h postgres-postgresql -U fractal -d fractal_lba -c "SELECT 1"
```

## Root Cause Diagnosis

### Scenario A: Redis/Postgres Pod Crashed

```bash
# Check recent pod events
kubectl describe pod -n fractal-lba -l app=redis

# Look for OOMKilled, CrashLoopBackOff
kubectl get events -n fractal-lba --sort-by='.lastTimestamp' | grep -i redis
```

**Resolution:**
```bash
# If OOMKilled: Increase memory limits
kubectl set resources deployment/redis -n fractal-lba \
  --limits=memory=2Gi --requests=memory=1Gi

# If CrashLoopBackOff: Check logs and restart
kubectl rollout restart deployment/redis -n fractal-lba
```

### Scenario B: Network Partition

```bash
# Check NetworkPolicy
kubectl get networkpolicy -n fractal-lba

# Test connectivity from backend to Redis
kubectl exec -n fractal-lba deployment/fractal-lba-backend -- \
  nc -zv redis-master 6379
```

**Resolution:**
```bash
# Temporarily disable NetworkPolicy if blocking
kubectl delete networkpolicy fractal-lba-backend -n fractal-lba

# Fix and reapply correct policy
kubectl apply -f helm/fractal-lba/templates/networkpolicy.yaml
```

### Scenario C: Disk Full (Persistence)

```bash
# Check PVC usage
kubectl exec -n fractal-lba -c redis deployment/redis -- df -h

# Check for large keys
kubectl exec -n fractal-lba deployment/redis -- \
  redis-cli --bigkeys
```

**Resolution:**
```bash
# Expand PVC (if storage class supports)
kubectl patch pvc redis-data -n fractal-lba -p \
  '{"spec":{"resources":{"requests":{"storage":"50Gi"}}}}'

# Or: Manually expire old keys
kubectl exec -n fractal-lba deployment/redis -- \
  redis-cli --scan --pattern "pcs:*" | head -1000 | xargs redis-cli DEL
```

### Scenario D: Too Many Connections

```bash
# Check active connections
kubectl exec -n fractal-lba deployment/redis -- \
  redis-cli CLIENT LIST | wc -l

# Check max connections
kubectl exec -n fractal-lba deployment/redis -- \
  redis-cli CONFIG GET maxclients
```

**Resolution:**
```bash
# Increase max connections
kubectl exec -n fractal-lba deployment/redis -- \
  redis-cli CONFIG SET maxclients 10000

# Or: Restart backend to clear stale connections
kubectl rollout restart deployment/fractal-lba-backend -n fractal-lba
```

## Recovery Procedure

### 1. Restore Dedup Store

**For Redis:**
```bash
# If Redis is down, restart it
kubectl rollout restart statefulset/redis-master -n fractal-lba

# Wait for readiness
kubectl wait --for=condition=ready pod -l app=redis -n fractal-lba --timeout=5m

# Verify data persistence
kubectl exec -n fractal-lba deployment/redis -- redis-cli DBSIZE
```

**For PostgreSQL:**
```bash
# Restart Postgres
kubectl rollout restart statefulset/postgresql -n fractal-lba

# Verify table exists
kubectl exec -n fractal-lba deployment/postgresql -- \
  psql -U fractal -d fractal_lba -c "SELECT COUNT(*) FROM dedup;"
```

### 2. Switch Backend Back to Normal Mode

```bash
# Re-enable external dedup store
kubectl set env deployment/fractal-lba-backend \
  DEDUP_BACKEND=redis \
  DEDUP_REQUIRED=true \
  -n fractal-lba

# Monitor for errors
kubectl logs -n fractal-lba -l app=fractal-lba-backend -f | grep "Dedup"
```

### 3. Process WAL Backlog (If Applicable)

```bash
# Check WAL size
kubectl exec -n fractal-lba deployment/fractal-lba-backend -- \
  du -sh /data/wal

# If needed, manually process WAL entries
# (Backend should auto-process on startup)
```

### 4. Verify System Health

```bash
# Check /health endpoint
curl https://api.fractal-lba.example.com/health

# Submit test PCS
curl -X POST https://api.fractal-lba.example.com/v1/pcs/submit \
  -H "Content-Type: application/json" \
  -d @tests/golden/pcs_tiny_case_1.json

# Check dedup is working (submit same PCS twice)
# Second submission should return cached result
```

## Prevention

1. **Monitoring**
   - Alert on dedup store latency (p95 > 100ms)
   - Alert on connection failures
   - Alert on disk usage >80%

2. **Capacity Planning**
   - Size Redis/Postgres for 2x expected load
   - Enable persistence with appropriate retention
   - Plan for 14-day dedup TTL (size accordingly)

3. **High Availability**
   - Run Redis in replication mode (master + 2 replicas)
   - Enable Redis Sentinel for automatic failover
   - Use PodDisruptionBudget (minAvailable: 1)

4. **Backup & Recovery**
   - Enable automated Redis RDB snapshots
   - Configure Postgres WAL archiving
   - Test restore procedure quarterly

## Degraded Mode Trade-offs

If dedup store is unavailable for extended period:

### Option 1: In-Memory Dedup
- **Pro:** Fast, no external dependency
- **Con:** Lost on pod restart, no cross-pod dedup
- **Use When:** Temporary outage (<1 hour)

### Option 2: No Dedup (Accept All)
- **Pro:** Maximum availability
- **Con:** Duplicate PCS processed, wastes resources
- **Use When:** Critical traffic, store unrecoverable

### Option 3: Shed Load (503)
- **Pro:** Preserves data integrity
- **Con:** Service unavailable
- **Use When:** Store recovery imminent, data quality critical

## Communication Template

**Subject:** [P1] Dedup Store Outage - Service Degraded

**Body:**
```
The Fractal LBA deduplication store is currently unavailable.

Status: [Investigating/Mitigating/Resolved]
Impact: PCS submissions may fail or be processed multiple times
Started: [Timestamp]
Expected Resolution: [Timestamp + 1 hour]

Current Mitigation:
- [e.g., "Switched to in-memory dedup, idempotency within pod only"]
- [e.g., "Returning 503 with Retry-After header"]

Root Cause:
- [TBD/Identified: ...]

Next Steps:
- [e.g., "Restoring Redis from snapshot"]
- [e.g., "Scaling up Postgres resources"]

We apologize for the disruption. Updates every 15 minutes.
```

## Related Runbooks

- [WAL Disk Pressure](./wal-disk-pressure.md)
- [Error Budget Burn](./error-budget-burn.md)
- [Latency Surge](./latency-surge.md)

---

**Last Updated:** 2025-01-20
**Owner:** SRE Team
**Escalation:** Page database team if store not recoverable in 1 hour
