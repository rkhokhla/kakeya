# Runbook: Shard Migration (Phase 4 WP2)

**Audience:** SRE, Database Administrators, Platform Engineers
**Scope:** Safe migration and rebalancing of sharded dedup stores with zero downtime
**Related:** CLAUDE_PHASE4.md WP2, backend/internal/sharding/router.go, dedup-outage.md

---

## 0) TL;DR

**Scenario:** Scale dedup store from N shards to M shards (typically N→N+1 for growth, or N→N-1 for consolidation).
**Goal:** Rebalance keys across new shard topology with **zero downtime**, **no data loss**, **no cache stampedes**.
**Key Actions:** Plan new ring, dual-write during cutover, verify consistency, gradual traffic shift.

---

## 1) When to Migrate Shards

### 1.1 Scale-Out Scenarios (N → N+1)

**Triggers:**
- **Storage pressure**: Redis/Postgres shard >80% capacity
- **Hot shard**: One shard receives >60% of traffic (imbalanced hash distribution)
- **Throughput bottleneck**: Shard p95 latency >50ms, CPU >70%
- **Regional expansion**: Adding new region with dedicated shards

**Example:** 3 shards → 4 shards
- Expected rebalancing: ~25% of keys move to new shard
- Minimal disruption to existing shards

### 1.2 Scale-In Scenarios (N → N-1)

**Triggers:**
- **Cost optimization**: Traffic dropped, over-provisioned shards
- **Shard failure**: Permanent loss, migrate keys to remaining shards
- **Consolidation**: Merging regions or tenants

**Example:** 4 shards → 3 shards
- Expected rebalancing: ~33% of keys move (all keys from removed shard + rebalancing)
- Higher risk of cache stampedes

### 1.3 Pre-Migration Checklist

- [ ] **Baseline metrics**: Record current dedup hit ratio, p95 latency, throughput
- [ ] **Capacity planning**: New shards have >50% headroom
- [ ] **Backup**: Snapshot all shards (RDB/Postgres dump)
- [ ] **Maintenance window**: Schedule during low-traffic period (if possible)
- [ ] **Runbook ready**: This document, practiced in staging

---

## 2) Architecture Overview

### 2.1 Consistent Hashing

**Current Implementation** (backend/internal/sharding/router.go):
```go
type Ring struct {
    shards  []*Shard
    vnodes  int  // Virtual nodes per shard (default: 150)
    ring    []uint32  // Sorted hash ring
    hashMap map[uint32]*Shard
}

func (r *Ring) Pick(key []byte) (*Shard, error) {
    h := r.hash(key)  // SHA-256, first 4 bytes as uint32
    idx := sort.Search(len(r.ring), func(i int) bool {
        return r.ring[i] >= h
    })
    if idx == len(r.ring) { idx = 0 }
    return r.hashMap[r.ring[idx]], nil
}
```

**Key Properties:**
- **Virtual nodes (vnodes)**: 150 per shard → smooth distribution
- **SHA-256 hash**: Uniform distribution of `pcs_id` across ring
- **Minimal migration**: Only ~1/N keys move when adding Nth shard

### 2.2 Migration Phases

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Plan (1 hour)                                      │
│ - Compute new ring topology                                 │
│ - Identify keys to migrate (old shard → new shard mapping) │
│ - Capacity check (new shards have headroom)                │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Pre-Copy (2-4 hours)                               │
│ - Copy keys from old shards to new shards (background)      │
│ - Throttled to avoid impacting live traffic                │
│ - Resumable checkpoints                                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Dual-Write (30 minutes)                            │
│ - Write to BOTH old and new shards                          │
│ - Catch up any keys written during pre-copy                │
│ - Verify consistency (sample checks)                        │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Cutover (5 minutes)                                │
│ - Update Ring with new topology                             │
│ - Route reads to new shards                                 │
│ - Monitor error rates and latency                           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 5: Cleanup (1 hour)                                   │
│ - Verify dedup hit ratio stable                             │
│ - Delete migrated keys from old shards (after TTL window)  │
│ - Update monitoring dashboards                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3) Migration Tool: `dedup-migrate`

### 3.1 Installation

```bash
# Build from source
cd backend/cmd/dedup-migrate
go build -o dedup-migrate

# Or download prebuilt binary
curl -LO https://github.com/fractal-lba/kakeya/releases/latest/download/dedup-migrate-linux-amd64
chmod +x dedup-migrate-linux-amd64
mv dedup-migrate-linux-amd64 /usr/local/bin/dedup-migrate
```

### 3.2 Commands

**`dedup-migrate plan`**: Compute migration plan
```bash
dedup-migrate plan \
  --old-shards redis://shard-0:6379,redis://shard-1:6379,redis://shard-2:6379 \
  --new-shards redis://shard-0:6379,redis://shard-1:6379,redis://shard-2:6379,redis://shard-3:6379 \
  --vnodes 150 \
  --output /tmp/migration-plan.json
```

**Output:**
```json
{
  "migration_id": "20250115-103000",
  "old_topology": {
    "shards": 3,
    "total_keys": 1500000
  },
  "new_topology": {
    "shards": 4,
    "total_keys": 1500000
  },
  "migrations": [
    {
      "from": "shard-0",
      "to": "shard-3",
      "keys_to_move": 125000,
      "estimated_duration": "45m"
    },
    {
      "from": "shard-1",
      "to": "shard-3",
      "keys_to_move": 130000,
      "estimated_duration": "47m"
    },
    {
      "from": "shard-2",
      "to": "shard-3",
      "keys_to_move": 120000,
      "estimated_duration": "43m"
    }
  ],
  "total_keys_to_move": 375000,
  "percentage_to_move": 25.0
}
```

**`dedup-migrate copy`**: Copy keys (Phase 2)
```bash
dedup-migrate copy \
  --plan /tmp/migration-plan.json \
  --throttle 1000  # keys per second \
  --checkpoint /tmp/migration-checkpoint.json \
  --log /tmp/migration.log
```

**`dedup-migrate verify`**: Verify consistency
```bash
dedup-migrate verify \
  --plan /tmp/migration-plan.json \
  --sample 10000  # sample 10k random keys \
  --output /tmp/verification-report.json
```

**`dedup-migrate cutover`**: Update ring (Phase 4)
```bash
dedup-migrate cutover \
  --plan /tmp/migration-plan.json \
  --dry-run  # Test without applying
# If dry-run succeeds, run without --dry-run
dedup-migrate cutover --plan /tmp/migration-plan.json
```

**`dedup-migrate cleanup`**: Delete old keys (Phase 5)
```bash
dedup-migrate cleanup \
  --plan /tmp/migration-plan.json \
  --wait-ttl 14d  # Wait for dedup TTL to expire \
  --batch-size 1000
```

---

## 4) Step-by-Step Procedures

### 4.1 Phase 1: Plan (1 hour)

**1. Create migration plan:**
```bash
# Identify current shards
CURRENT_SHARDS=$(kubectl get pods -n fractal-lba -l app=redis-dedup -o jsonpath='{.items[*].metadata.name}')
echo "Current shards: $CURRENT_SHARDS"

# Add new shard (provision Redis pod)
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-dedup-shard-3
  namespace: fractal-lba
spec:
  serviceName: redis-dedup
  replicas: 1
  selector:
    matchLabels:
      app: redis-dedup
      shard: "3"
  template:
    metadata:
      labels:
        app: redis-dedup
        shard: "3"
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
EOF

# Wait for new shard to be ready
kubectl wait --for=condition=ready pod -l app=redis-dedup,shard=3 -n fractal-lba --timeout=300s
```

**2. Generate migration plan:**
```bash
dedup-migrate plan \
  --old-shards redis://redis-dedup-shard-0.redis-dedup:6379,redis://redis-dedup-shard-1.redis-dedup:6379,redis://redis-dedup-shard-2.redis-dedup:6379 \
  --new-shards redis://redis-dedup-shard-0.redis-dedup:6379,redis://redis-dedup-shard-1.redis-dedup:6379,redis://redis-dedup-shard-2.redis-dedup:6379,redis://redis-dedup-shard-3.redis-dedup:6379 \
  --vnodes 150 \
  --output /tmp/migration-plan-$(date +%Y%m%d-%H%M%S).json

# Review plan
jq . /tmp/migration-plan-*.json
```

**3. Capacity check:**
```bash
# Verify new shard has capacity for migrated keys
NEW_SHARD_MEM=$(redis-cli -h redis-dedup-shard-3.redis-dedup info memory | grep used_memory_human | awk -F: '{print $2}')
echo "New shard memory: $NEW_SHARD_MEM (should be <50% of limit)"
```

**4. Backup current shards:**
```bash
# Snapshot all shards
for shard in 0 1 2; do
  redis-cli -h redis-dedup-shard-$shard.redis-dedup SAVE
  kubectl cp fractal-lba/redis-dedup-shard-$shard-0:/data/dump.rdb /tmp/shard-$shard-backup-$(date +%s).rdb
done
```

### 4.2 Phase 2: Pre-Copy (2-4 hours)

**1. Start background copy:**
```bash
# Throttle to 1000 keys/sec to avoid impacting live traffic
dedup-migrate copy \
  --plan /tmp/migration-plan-*.json \
  --throttle 1000 \
  --checkpoint /tmp/migration-checkpoint.json \
  --log /tmp/migration.log \
  &

COPY_PID=$!
echo "Migration PID: $COPY_PID"
```

**2. Monitor progress:**
```bash
# Watch checkpoint file
watch -n 30 'jq ".progress" /tmp/migration-checkpoint.json'

# Tail logs
tail -f /tmp/migration.log | grep -E "(Copied|Error)"

# Monitor shard metrics
watch -n 10 'redis-cli -h redis-dedup-shard-3.redis-dedup DBSIZE'
```

**3. Handle resumable checkpoints:**
```bash
# If copy is interrupted (network glitch, OOM), resume from checkpoint
kill $COPY_PID  # Simulate interruption

# Resume
dedup-migrate copy \
  --plan /tmp/migration-plan-*.json \
  --throttle 1000 \
  --checkpoint /tmp/migration-checkpoint.json \  # Resumes from last checkpoint
  --log /tmp/migration.log
```

**4. Pre-copy completion:**
```bash
# Wait for copy to finish
wait $COPY_PID
echo "Pre-copy completed: $(jq '.keys_copied' /tmp/migration-checkpoint.json) keys"
```

### 4.3 Phase 3: Dual-Write (30 minutes)

**1. Enable dual-write mode:**
```bash
# Update backend deployment to write to BOTH old and new shards
kubectl set env deployment/backend -n fractal-lba DEDUP_DUAL_WRITE=true
kubectl set env deployment/backend -n fractal-lba DEDUP_NEW_RING="shard-0,shard-1,shard-2,shard-3"

# Wait for rollout
kubectl rollout status deployment/backend -n fractal-lba
```

**2. Catch-up copy:**
```bash
# Copy keys written after pre-copy started
dedup-migrate copy \
  --plan /tmp/migration-plan-*.json \
  --throttle 5000  # Higher throttle (less traffic impact since dual-write active) \
  --checkpoint /tmp/migration-checkpoint-catchup.json \
  --incremental  # Only copy keys not already copied
```

**3. Verify consistency (sample checks):**
```bash
dedup-migrate verify \
  --plan /tmp/migration-plan-*.json \
  --sample 10000 \
  --output /tmp/verification-report.json

# Check for mismatches
jq '.mismatches' /tmp/verification-report.json
# Should be 0 (or very low <0.01%)
```

### 4.4 Phase 4: Cutover (5 minutes)

**⚠️ HIGH RISK PHASE:** Briefly elevated error rates possible if ring update not atomic.

**1. Test cutover (dry-run):**
```bash
dedup-migrate cutover \
  --plan /tmp/migration-plan-*.json \
  --dry-run

# Expected output:
# ✅ Ring topology valid
# ✅ All shards healthy
# ✅ Dual-write active
# ✅ Ready for cutover
```

**2. Execute cutover:**
```bash
# Update backend ring configuration
kubectl patch configmap backend-config -n fractal-lba --type merge -p '{
  "data": {
    "DEDUP_SHARDS": "redis://redis-dedup-shard-0.redis-dedup:6379,redis://redis-dedup-shard-1.redis-dedup:6379,redis://redis-dedup-shard-2.redis-dedup:6379,redis://redis-dedup-shard-3.redis-dedup:6379",
    "DEDUP_DUAL_WRITE": "false"
  }
}'

# Rolling restart to pick up new config
kubectl rollout restart deployment/backend -n fractal-lba
kubectl rollout status deployment/backend -n fractal-lba --timeout=300s
```

**3. Monitor error rates:**
```bash
# Watch for 5xx errors (target: <0.1% spike)
watch -n 5 'curl -s "http://prometheus:9090/api/v1/query?query=rate(flk_ingest_total{status=~\"5..\"}[1m])" | jq ".data.result[0].value[1]"'

# Watch dedup hit ratio (should remain stable ~40%)
watch -n 10 'curl -s "http://prometheus:9090/api/v1/query?query=flk_dedup_hits / flk_ingest_total" | jq ".data.result[0].value[1]"'
```

**4. Rollback plan (if cutover fails):**
```bash
# Revert to old ring
kubectl patch configmap backend-config -n fractal-lba --type merge -p '{
  "data": {
    "DEDUP_SHARDS": "redis://redis-dedup-shard-0.redis-dedup:6379,redis://redis-dedup-shard-1.redis-dedup:6379,redis://redis-dedup-shard-2.redis-dedup:6379",
    "DEDUP_DUAL_WRITE": "false"
  }
}'
kubectl rollout restart deployment/backend -n fractal-lba
```

### 4.5 Phase 5: Cleanup (1 hour)

**1. Verify dedup stability (30 minutes post-cutover):**
```bash
# Dedup hit ratio should stabilize after cache warm-up
curl -s "http://prometheus:9090/api/v1/query?query=flk_dedup_hits / flk_ingest_total" | jq ".data.result[0].value[1]"
# Target: >35% (slightly lower than baseline due to cache churn)
```

**2. Delete migrated keys from old shards (after TTL window):**
```bash
# Wait for dedup TTL to expire (default: 14 days)
# This ensures no stale reads from old shards

# After TTL expires, cleanup
dedup-migrate cleanup \
  --plan /tmp/migration-plan-*.json \
  --wait-ttl 14d \
  --batch-size 1000 \
  --log /tmp/cleanup.log
```

**3. Update monitoring dashboards:**
```bash
# Update Grafana dashboard to show 4 shards
# Edit observability/grafana/dashboards/dedup-health.json
jq '.panels[0].targets[0].expr |= sub("shard-0,shard-1,shard-2"; "shard-0,shard-1,shard-2,shard-3")' \
  observability/grafana/dashboards/dedup-health.json > /tmp/updated-dashboard.json

# Upload updated dashboard
curl -X POST http://grafana:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @/tmp/updated-dashboard.json
```

---

## 5) Troubleshooting

### 5.1 Copy Phase Stalls

**Symptom:** `dedup-migrate copy` progress stops at N% for >30 minutes.

**Diagnosis:**
```bash
# Check shard health
for shard in 0 1 2 3; do
  redis-cli -h redis-dedup-shard-$shard.redis-dedup PING
done

# Check network latency
for shard in 0 1 2 3; do
  time redis-cli -h redis-dedup-shard-$shard.redis-dedup PING
done
# Should be <10ms
```

**Resolution:**
- Increase `--throttle` if shards have spare capacity
- Check for OOM or disk pressure on source/target shards
- Resume from checkpoint after fixing underlying issue

### 5.2 Dual-Write Errors

**Symptom:** 5xx errors spike during dual-write phase.

**Diagnosis:**
```bash
# Check backend logs for write failures
kubectl logs -n fractal-lba -l app=backend --tail=100 | grep "dual-write failed"

# Check if new shard is reachable
kubectl exec -n fractal-lba deployment/backend -- redis-cli -h redis-dedup-shard-3.redis-dedup PING
```

**Resolution:**
- Verify new shard is healthy and reachable
- Check NetworkPolicy allows backend → new shard traffic
- Temporarily disable dual-write, fix shard, re-enable

### 5.3 Cutover Error Rate Spike

**Symptom:** 5xx errors >1% immediately after cutover.

**Diagnosis:**
```bash
# Check ring distribution (should be uniform)
for shard in 0 1 2 3; do
  COUNT=$(redis-cli -h redis-dedup-shard-$shard.redis-dedup DBSIZE | awk '{print $2}')
  echo "shard-$shard: $COUNT keys"
done
# Each shard should have ~25% of total keys (±5%)
```

**Resolution:**
- If one shard has significantly fewer keys → incomplete copy, rollback
- If error rate >5% → immediate rollback, investigate
- If error rate 1-5% → monitor for 5 minutes, may stabilize as cache warms up

### 5.4 Dedup Hit Ratio Drop

**Symptom:** Dedup hit ratio drops from 40% to <20% post-cutover.

**Root Cause:** Cache churn during migration (keys moved to new shard not yet "warm").

**Mitigation:**
- **Expected**: Hit ratio recovers over 1-2 hours as cache warms up
- **Proactive warming**: Pre-copy includes READ operations to warm cache during copy phase
- **Temporary increase capacity**: Add more backend replicas during migration to handle cache miss load

---

## 6) Performance Impact

### 6.1 Expected Metrics

**Baseline (before migration):**
- Dedup hit ratio: 40%
- p95 latency: 50ms
- Throughput: 2000 req/s

**During migration:**
| Phase | Dedup Hit Ratio | p95 Latency | Throughput | Notes |
|-------|-----------------|-------------|------------|-------|
| Pre-Copy | 40% (stable) | 50ms | 2000 req/s | No impact (background copy) |
| Dual-Write | 38% (-5%) | 60ms (+20%) | 1900 req/s | Slight degradation (2x writes) |
| Cutover | 25% (-37%) | 80ms (+60%) | 1800 req/s | Cache churn peak |
| Post-Cutover +1h | 35% (-12%) | 55ms (+10%) | 1950 req/s | Recovering |
| Post-Cutover +24h | 40% (baseline) | 50ms | 2000 req/s | Fully recovered |

**Worst Case (failed cutover, rollback):**
- Downtime: 2-5 minutes during rollback
- User-facing errors: <1% (cached results from old shards)

### 6.2 SLO Impact

**Target:** No SLO breach during migration (p95 < 200ms, error rate < 1%)

**Actual:**
- p95 latency: 80ms (worst case during cutover) → **within SLO**
- Error rate: <0.1% → **within SLO**
- Dedup hit ratio: Temporary drop acceptable (not an SLO)

---

## 7) Communication Templates

### 7.1 Migration Start

**Subject:** [MAINTENANCE] Shard Migration in Progress - Fractal LBA
**To:** #engineering, #sre

**Body:**
```
ℹ️ MAINTENANCE: Dedup shard migration in progress

Time: 2025-01-15 02:00 UTC (low-traffic window)
Scope: Adding shard-3 (3→4 shards)
Expected Duration: 4 hours
Expected Impact: None (zero-downtime migration)

Phases:
- [02:00-04:00] Pre-copy keys to new shard (background)
- [04:00-04:30] Dual-write mode (slight latency increase +10ms)
- [04:30-04:35] Cutover (brief cache churn)
- [04:35-06:00] Monitoring and cleanup

Updates: Every 30 minutes in #sre
```

### 7.2 Migration Complete

**Subject:** [COMPLETE] Shard Migration - Fractal LBA
**To:** #engineering, #sre

**Body:**
```
✅ COMPLETE: Dedup shard migration successful

Duration: 3h 47m (ahead of schedule)
Outcome:
- 4 shards now active (was 3)
- 375,000 keys migrated (25% of total)
- Zero downtime, no user-facing errors
- Dedup hit ratio: 38% (recovering, target 40% by 2025-01-15 12:00 UTC)

Metrics:
- p95 latency: 52ms (baseline: 50ms, within SLO)
- Throughput: 1980 req/s (baseline: 2000 req/s)
- Shard-3 memory: 1.2 GB (40% of limit, healthy)

Next Steps:
- Monitor dedup hit ratio recovery
- Cleanup old keys after TTL expires (2025-01-29)
```

---

## 8) Playbook Checklist

**Pre-Migration:**
- [ ] Baseline metrics recorded (dedup hit ratio, p95 latency, throughput)
- [ ] New shard provisioned and healthy
- [ ] Migration plan generated and reviewed
- [ ] Capacity check (new shard >50% headroom)
- [ ] Backups taken (all shards)
- [ ] Maintenance window scheduled (low-traffic period)

**Phase 1: Plan:**
- [ ] Current shards identified
- [ ] New ring topology computed (vnodes = 150)
- [ ] Migration plan saved (/tmp/migration-plan-*.json)
- [ ] Keys to migrate calculated (~25% for N→N+1)

**Phase 2: Pre-Copy:**
- [ ] Background copy started (throttled to 1000 keys/s)
- [ ] Progress monitored (checkpoint file)
- [ ] Pre-copy completed (100% keys copied)

**Phase 3: Dual-Write:**
- [ ] Dual-write mode enabled (backend env var)
- [ ] Catch-up copy completed (incremental keys)
- [ ] Consistency verified (sample 10k keys, <0.01% mismatches)

**Phase 4: Cutover:**
- [ ] Dry-run succeeded
- [ ] Ring configuration updated
- [ ] Backend pods restarted (rolling restart)
- [ ] Error rates monitored (<0.1% spike)
- [ ] Rollback plan ready

**Phase 5: Cleanup:**
- [ ] Dedup hit ratio stabilized (>35% after 1h)
- [ ] Cleanup scheduled (after TTL window)
- [ ] Monitoring dashboards updated
- [ ] Post-migration report published

---

## 9) Related Runbooks

- **dedup-outage.md**: Dedup store recovery (Phase 2)
- **geo-failover.md**: Multi-region failover (Phase 4)
- **tier-cold-hot-miss.md**: Tiered storage migration (Phase 4)

---

## 10) References

- CLAUDE_PHASE4.md WP2: Sharded Dedup Store + Live Migration
- backend/internal/sharding/router.go: Consistent hashing implementation
- Consistent Hashing paper: Karger et al. (1997)
- Redis replication: https://redis.io/docs/manual/replication/

---

**End of Runbook**
