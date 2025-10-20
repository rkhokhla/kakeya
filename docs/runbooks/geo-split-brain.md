# Runbook: Geo Split-Brain Resolution (Phase 4 WP1)

**Audience:** SRE, Database Administrators, Incident Commanders
**Scope:** Detection and resolution of split-brain scenarios in multi-region active-active topology
**Related:** CLAUDE_PHASE4.md WP1, geo-failover.md, dedup-outage.md

---

## 0) TL;DR

**Scenario:** Network partition heals, but both regions processed writes independently → dedup state diverged.
**Goal:** Reconcile state without data loss, restore consistency, prevent cascading failures.
**Key Actions:** Detect divergence via metrics, elect authoritative WAL, rebuild dedup state, verify idempotency.

---

## 1) What is Split-Brain?

**Definition:** Both regions believe they are authoritative and process writes independently during a network partition, leading to divergent state.

**In Fractal LBA context:**
- **Region A** and **Region B** both accept PCS submissions
- **Dedup stores** (Redis/Postgres) diverge: same `pcs_id` may have different outcomes
- **WAL segments** contain overlapping but non-identical writes
- **Risk:** Conflicting verification outcomes, incorrect metrics, audit trail gaps

**Phase 1 Protection:** `pcs_id` idempotency prevents double-effects, but divergent caches can cause inconsistent responses.

---

## 2) Detection

### 2.1 Symptoms

**Primary Indicators:**
1. **Dedup hit ratio anomaly**: Drops >20% after partition heals
2. **Metric divergence**: `flk_accepted` counters differ significantly between regions
3. **Audit log gaps**: WORM segments missing in one region
4. **User reports**: Same PCS returns different outcomes in different regions

**Prometheus Alerts:**
```yaml
# observability/prometheus/alerts.yml
- alert: GeoDedupDivergence
  expr: abs(flk_dedup_hits{region="eu-west"} - flk_dedup_hits{region="us-east"}) / flk_ingest_total > 0.2
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Dedup state divergence between regions"
    description: "Dedup hit ratio differs by >20% (split-brain indicator)"
    runbook: "docs/runbooks/geo-split-brain.md"
```

### 2.2 Diagnostic Queries

**1. Compare dedup key counts:**
```bash
# Redis
REGION_A_KEYS=$(redis-cli -h eu-west-redis DBSIZE | awk '{print $2}')
REGION_B_KEYS=$(redis-cli -h us-east-redis DBSIZE | awk '{print $2}')
DIFF=$(echo "scale=2; ($REGION_A_KEYS - $REGION_B_KEYS) / $REGION_A_KEYS * 100" | bc)
echo "Divergence: $DIFF%"
# If >10%, split-brain likely
```

**2. Sample conflicting PCS IDs:**
```bash
# Find PCS IDs in eu-west but not us-east
redis-cli -h eu-west-redis --scan --pattern "pcs:*" | sort > /tmp/eu-west-keys.txt
redis-cli -h us-east-redis --scan --pattern "pcs:*" | sort > /tmp/us-east-keys.txt
comm -23 /tmp/eu-west-keys.txt /tmp/us-east-keys.txt | head -20
# These PCS were processed only in eu-west during partition
```

**3. Compare verification outcomes:**
```bash
# Pick a sample PCS ID present in both regions
PCS_ID="abc123def456..."
OUTCOME_A=$(redis-cli -h eu-west-redis GET "pcs:$PCS_ID" | jq -r '.accepted')
OUTCOME_B=$(redis-cli -h us-east-redis GET "pcs:$PCS_ID" | jq -r '.accepted')
if [ "$OUTCOME_A" != "$OUTCOME_B" ]; then
  echo "❌ CONFLICT: eu-west=$OUTCOME_A, us-east=$OUTCOME_B"
fi
```

**4. Check WAL segment overlap:**
```bash
# List WAL segments written during partition window
PARTITION_START="2025-01-15T10:30:00Z"
PARTITION_END="2025-01-15T10:45:00Z"

aws s3 ls s3://flk-wal-eu-west/inbox/ --recursive | \
  awk -v start="$PARTITION_START" -v end="$PARTITION_END" \
  '$1" "$2 >= start && $1" "$2 <= end {print $4}' > /tmp/eu-west-wal.txt

aws s3 ls s3://flk-wal-us-east/inbox/ --recursive | \
  awk -v start="$PARTITION_START" -v end="$PARTITION_END" \
  '$1" "$2 >= start && $1" "$2 <= end {print $4}' > /tmp/us-east-wal.txt

# Compare segment counts
echo "eu-west segments: $(wc -l /tmp/eu-west-wal.txt)"
echo "us-east segments: $(wc -l /tmp/us-east-wal.txt)"
```

---

## 3) Impact Assessment

### 3.1 Severity Matrix

| Divergence % | Conflicting Outcomes | Severity | Action |
|--------------|---------------------|----------|--------|
| <5% | None | P3 (Low) | Monitor, schedule reconciliation during maintenance |
| 5-10% | <1% | P2 (Medium) | Reconcile within 4 hours |
| 10-20% | 1-5% | P1 (High) | Immediate reconciliation, page on-call |
| >20% | >5% | P0 (Critical) | Drain traffic, emergency reconciliation |

**Conflicting Outcomes:** PCS IDs where `accepted` status differs between regions.

### 3.2 User Impact

**Low Impact Scenarios:**
- PCS submitted to one region only (agent-region affinity)
- Divergent dedup caches but no conflicting outcomes
- Metrics slightly off, but audit trail complete

**High Impact Scenarios:**
- Same PCS routed to different regions mid-partition
- Conflicting `accepted` status causes downstream workflow failures
- Audit trail gaps (WORM segments missing in one region)
- Escalated PCS in one region, accepted in another → compliance risk

---

## 4) Reconciliation Strategy

### 4.1 Principles

**P1: WAL is source of truth** (Phase 1 invariant: WAL before parse)
**P2: First-write wins** (Phase 1 idempotency: `pcs_id` dedup)
**P3: No data loss** (both WALs must be merged)
**P4: Audit trail complete** (WORM entries must include all PCS)

### 4.2 Decision Tree

```
┌─────────────────────────────────────────┐
│ Split-brain detected (dedup divergence) │
└───────────────┬─────────────────────────┘
                │
                ▼
        ┌───────────────┐
        │ Conflicting   │
        │ outcomes?     │
        └───┬───────┬───┘
            │       │
           YES     NO
            │       │
            ▼       ▼
    ┌───────────┐ ┌──────────────┐
    │ P0: Drain │ │ P2: Merge    │
    │ traffic   │ │ WALs offline │
    │ Emergency │ │ (4h window)  │
    │ reconcile │ └──────────────┘
    └───────────┘
            │
            ▼
    ┌───────────────────┐
    │ Elect authoritative│
    │ WAL (earliest ts) │
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │ Rebuild dedup from│
    │ merged WAL        │
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │ Replicate to both │
    │ regions           │
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │ Verify consistency│
    │ Resume traffic    │
    └───────────────────┘
```

---

## 5) Reconciliation Procedures

### 5.1 Option A: Offline Reconciliation (Low Risk, Recommended)

**When:** Conflicting outcomes <1%, can afford 1-2 hour downtime

**Steps:**

**1. Drain traffic (both regions → 503)**
```bash
# Scale down backend in both regions
kubectl scale deployment/backend -n fractal-lba --replicas=0 --all-contexts

# Verify no in-flight requests
watch -n 5 'kubectl logs -n fractal-lba -l app=backend --tail=10 | grep "POST /v1/pcs/submit"'
# Wait until no new logs appear
```

**2. Snapshot dedup state (backup before wipe)**
```bash
# eu-west
redis-cli -h eu-west-redis SAVE
aws s3 cp /var/lib/redis/dump.rdb s3://flk-dedup-backups/eu-west-pre-reconcile-$(date +%s).rdb

# us-east
redis-cli -h us-east-redis SAVE
aws s3 cp /var/lib/redis/dump.rdb s3://flk-dedup-backups/us-east-pre-reconcile-$(date +%s).rdb
```

**3. Merge WAL segments**
```bash
# Download WAL from both regions
aws s3 sync s3://flk-wal-eu-west/inbox/ /tmp/wal-merge/eu-west/
aws s3 sync s3://flk-wal-us-east/inbox/ /tmp/wal-merge/us-east/

# Merge with timestamp ordering (earliest first)
find /tmp/wal-merge -name "*.jsonl" -exec stat -f "%m %N" {} \; | \
  sort -n | awk '{print $2}' > /tmp/merged-wal-order.txt

# Concatenate in order
cat $(cat /tmp/merged-wal-order.txt) > /tmp/merged-wal.jsonl
```

**4. Rebuild dedup state from merged WAL**
```bash
# Clear dedup in both regions
redis-cli -h eu-west-redis FLUSHDB
redis-cli -h us-east-redis FLUSHDB

# Replay merged WAL with idempotent dedup
./scripts/wal-replay.sh \
  --source /tmp/merged-wal.jsonl \
  --target redis://eu-west-redis \
  --idempotent \
  --log /tmp/replay.log
```

**5. Replicate rebuilt state to us-east**
```bash
# Snapshot eu-west (authoritative after replay)
redis-cli -h eu-west-redis SAVE
aws s3 cp /var/lib/redis/dump.rdb s3://flk-dedup-backups/eu-west-post-reconcile-$(date +%s).rdb

# Restore to us-east
aws s3 cp s3://flk-dedup-backups/eu-west-post-reconcile-*.rdb /tmp/restore.rdb
redis-cli -h us-east-redis --rdb /tmp/restore.rdb RESTORE
```

**6. Verify consistency**
```bash
# Compare key counts
REGION_A_KEYS=$(redis-cli -h eu-west-redis DBSIZE | awk '{print $2}')
REGION_B_KEYS=$(redis-cli -h us-east-redis DBSIZE | awk '{print $2}')
if [ "$REGION_A_KEYS" -eq "$REGION_B_KEYS" ]; then
  echo "✅ Dedup state consistent: $REGION_A_KEYS keys"
else
  echo "❌ Still diverged: eu-west=$REGION_A_KEYS, us-east=$REGION_B_KEYS"
  exit 1
fi

# Sample verification outcomes
for pcs_id in $(redis-cli -h eu-west-redis --scan --pattern "pcs:*" | head -100); do
  OUTCOME_A=$(redis-cli -h eu-west-redis GET "$pcs_id" | jq -r '.accepted')
  OUTCOME_B=$(redis-cli -h us-east-redis GET "$pcs_id" | jq -r '.accepted')
  if [ "$OUTCOME_A" != "$OUTCOME_B" ]; then
    echo "❌ CONFLICT: $pcs_id eu-west=$OUTCOME_A, us-east=$OUTCOME_B"
  fi
done
# Should output no conflicts
```

**7. Resume traffic**
```bash
# Scale up backend in both regions
kubectl scale deployment/backend -n fractal-lba --replicas=3 --all-contexts

# Gradual traffic ramp: 10% → 50% → 100% over 30 minutes
# Monitor error rates and dedup hit ratio
```

### 5.2 Option B: Online Reconciliation (High Risk, Emergency Only)

**When:** Cannot afford downtime, conflicting outcomes <0.1%

**Strategy:** Merge WALs, replay diff into both regions while serving traffic.

**⚠️ RISK:** Race conditions, partial state updates, user-visible inconsistencies.

**Not recommended** unless business-critical uptime requirement.

**Steps (high-level):**
1. Compute WAL diff (PCS IDs in A but not B, and vice versa)
2. Replay diffs into opposite region's dedup store with CAS (compare-and-set)
3. Monitor for CAS failures (indicates concurrent writes during reconciliation)
4. Log all conflicts for manual review

---

## 6) Post-Reconciliation Verification

### 6.1 Audit Trail Completeness

**Check WORM logs:**
```bash
# WORM segments should cover all PCS IDs in merged WAL
PCS_IDS_IN_WAL=$(jq -r '.pcs_id' /tmp/merged-wal.jsonl | sort | uniq)
PCS_IDS_IN_WORM=$(find /var/lib/worm/audit -name "*.jsonl" -exec jq -r '.pcs_id' {} \; | sort | uniq)

comm -23 <(echo "$PCS_IDS_IN_WAL") <(echo "$PCS_IDS_IN_WORM") > /tmp/missing-worm-entries.txt
if [ -s /tmp/missing-worm-entries.txt ]; then
  echo "❌ Missing WORM entries: $(wc -l /tmp/missing-worm-entries.txt)"
  # Backfill WORM from WAL
  ./scripts/worm-backfill.sh --source /tmp/merged-wal.jsonl --pcs-ids /tmp/missing-worm-entries.txt
else
  echo "✅ WORM audit trail complete"
fi
```

### 6.2 Metrics Reconciliation

**Recompute aggregate metrics:**
```bash
# Count accepted vs escalated from merged WAL
ACCEPTED=$(jq -r 'select(.outcome=="accepted") | .pcs_id' /tmp/merged-wal.jsonl | wc -l)
ESCALATED=$(jq -r 'select(.outcome=="escalated") | .pcs_id' /tmp/merged-wal.jsonl | wc -l)

# Compare to Prometheus counters
PROM_ACCEPTED=$(curl -s 'http://prometheus:9090/api/v1/query?query=flk_accepted' | jq -r '.data.result[0].value[1]')
PROM_ESCALATED=$(curl -s 'http://prometheus:9090/api/v1/query?query=flk_escalated' | jq -r '.data.result[0].value[1]')

echo "WAL: accepted=$ACCEPTED, escalated=$ESCALATED"
echo "Prometheus: accepted=$PROM_ACCEPTED, escalated=$PROM_ESCALATED"
# Should match within ±1% (some in-flight during snapshot)
```

### 6.3 E2E Smoke Tests

**Run multi-region E2E tests:**
```bash
# Submit test PCS to both regions, verify consistent outcomes
pytest tests/e2e-geo/test_split_brain_recovery.py -v
# All tests should pass
```

---

## 7) Prevention Strategies

### 7.1 Architecture Improvements

**1. Single-Writer per PCS ID:**
- Route PCS by `shard_id` hash to deterministic region
- Agents use region affinity based on `shard_id`
- Reduces split-brain window to cross-shard writes only

**2. Distributed Consensus (Future):**
- Raft/Paxos for dedup writes (high latency cost)
- CRDTs for eventually consistent dedup state
- Requires major architecture change

**3. WAL-First Verification:**
- Move verification logic to WAL replay phase (post-partition)
- Dedup becomes read-only cache (no divergence risk)
- Trade-off: increased latency (deferred verification)

### 7.2 Monitoring Improvements

**1. Real-Time Divergence Detection:**
```yaml
# Alert on live traffic
- alert: GeoLiveDivergence
  expr: abs(rate(flk_ingest_total{region="eu-west"}[1m]) - rate(flk_ingest_total{region="us-east"}[1m])) / rate(flk_ingest_total[1m]) > 0.3
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "Live traffic divergence between regions"
    description: "Traffic split >30% (partition or routing issue)"
```

**2. Dedup Consistency Probes:**
```bash
# Periodic background job: sample 100 random PCS IDs, verify outcomes match
#!/bin/bash
SAMPLE=$(redis-cli -h eu-west-redis --scan --pattern "pcs:*" | shuf | head -100)
for pcs_id in $SAMPLE; do
  OUTCOME_A=$(redis-cli -h eu-west-redis GET "$pcs_id" | jq -r '.accepted')
  OUTCOME_B=$(redis-cli -h us-east-redis GET "$pcs_id" | jq -r '.accepted')
  if [ "$OUTCOME_A" != "$OUTCOME_B" ]; then
    echo "DIVERGENCE: $pcs_id"
  fi
done
```

### 7.3 Partition Testing (Chaos Engineering)

**Quarterly Game-Day:**
```bash
# Simulate network partition with Chaos Mesh
kubectl apply -f - <<EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: geo-partition
spec:
  action: partition
  mode: one
  selector:
    namespaces:
      - fractal-lba
    labelSelectors:
      region: eu-west
  direction: both
  duration: "5m"
  externalTargets:
    - us-east-alb.example.com
EOF

# Let partition run for 5 minutes, then heal
# Verify reconciliation procedures detect and resolve divergence
```

---

## 8) Communication Templates

### 8.1 Incident Start

**Subject:** [P0] Geo Split-Brain Detected - Fractal LBA
**To:** #incidents, engineering-all, sre-all

**Body:**
```
🚨 CRITICAL: Geo split-brain detected in Fractal LBA

Time: 2025-01-15 10:45 UTC
Affected Regions: eu-west, us-east
Divergence: 15% dedup state, 0.3% conflicting outcomes
Impact: Inconsistent responses for ~300 PCS IDs

Current Actions:
- Draining traffic from both regions (ETA: 2 minutes)
- Initiating offline reconciliation (ETA: 90 minutes)
- WAL merge in progress
- Backup snapshots taken

Users may see 503 errors during reconciliation.
Updates every 15 minutes in #incidents.
```

### 8.2 Incident Resolution

**Subject:** [RESOLVED] Geo Split-Brain - Fractal LBA
**To:** #incidents, engineering-all

**Body:**
```
✅ RESOLVED: Geo split-brain reconciled successfully

Duration: 87 minutes (downtime: 65 minutes)
Conflicting PCS: 312 (0.28% of total)
Resolution: WAL replay with first-write wins, dedup rebuilt

Outcome:
- Dedup state consistent across both regions (verified)
- Audit trail complete (no data loss)
- Metrics reconciled (Prometheus counters updated)
- E2E tests passing

Root Cause: Network partition between eu-west and us-east (2025-01-15 10:30-10:45 UTC)

Next Steps:
- Post-mortem scheduled for 2025-01-16 10:00 UTC
- Implement real-time divergence alerts (ETA: 1 week)
- Chaos engineering drill scheduled (2025-02-01)
```

---

## 9) Playbook Checklist

**Detection (< 5 minutes):**
- [ ] Confirm split-brain via metrics (dedup divergence >10%)
- [ ] Sample conflicting PCS IDs
- [ ] Assess severity (P0/P1/P2 based on Section 3.1)
- [ ] Identify partition window (start/end timestamps)
- [ ] Post incident start notification

**Reconciliation (< 2 hours):**
- [ ] Drain traffic from both regions (if P0)
- [ ] Snapshot dedup state (backup before wipe)
- [ ] Download WAL segments from both regions
- [ ] Merge WAL with timestamp ordering
- [ ] Rebuild dedup from merged WAL (first-write wins)
- [ ] Replicate rebuilt state to both regions
- [ ] Verify key counts match

**Post-Reconciliation (< 30 minutes):**
- [ ] Sample 100 PCS IDs, verify outcomes match
- [ ] Check WORM audit trail completeness
- [ ] Recompute aggregate metrics
- [ ] Resume traffic (gradual ramp: 10% → 50% → 100%)
- [ ] Run E2E smoke tests
- [ ] Post incident resolution notification
- [ ] Schedule post-mortem

---

## 10) Related Runbooks

- **geo-failover.md**: Multi-region failover procedures (includes split-brain detection)
- **dedup-outage.md**: Dedup store recovery (Phase 2)
- **wal-compact.sh**: WAL retention and compaction (Phase 2)

---

## 11) References

- CLAUDE_PHASE4.md WP1: Multi-Region Active-Active & DR
- CLAUDE.md Section 2.2: Fault tolerance patterns (idempotency, WAL-first)
- CAP theorem: Split-brain is unavoidable in AP systems during partitions
- observability/prometheus/alerts.yml: GeoDedupDivergence alert

---

**End of Runbook**
