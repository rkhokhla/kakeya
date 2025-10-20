# Runbook: Multi-Region Failover (Phase 4 WP1)

**Audience:** SRE, DevOps, Incident Commanders
**Scope:** Active-active multi-region failover procedures for Fractal LBA + Kakeya FT Stack
**Related:** CLAUDE_PHASE4.md WP1, geo-split-brain.md, dedup-outage.md

---

## 0) TL;DR

**Scenario:** One region becomes unavailable (network partition, data center outage, cascading failure).
**Goal:** Maintain service availability with **RTO ≤ 5 minutes**, **RPO ≤ 2 minutes**, no data corruption.
**Key Actions:** Reroute traffic via GSLB, verify WAL replication lag, enable degraded mode if needed, monitor for split-brain.

---

## 1) Pre-Incident Preparation

### 1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Global Load Balancer                     │
│                  (Route53, Cloudflare, GSLB)                │
└───────────────────┬────────────────┬────────────────────────┘
                    │                │
         ┌──────────▼──────────┐  ┌─▼──────────────────────┐
         │   Region: eu-west   │  │   Region: us-east      │
         │  (Primary Active)   │  │  (Secondary Active)    │
         ├─────────────────────┤  ├────────────────────────┤
         │ Backend (3 replicas)│  │ Backend (3 replicas)   │
         │ Redis Dedup         │  │ Redis Dedup            │
         │ Postgres Warm       │  │ Postgres Warm          │
         │ S3 WAL (CRR→)       │  │ S3 WAL (←CRR)          │
         └─────────────────────┘  └────────────────────────┘
```

**Key Invariants:**
- **Idempotent replay**: `pcs_id` first-write wins across regions
- **WAL CRR**: S3 cross-region replication with versioning
- **Dedup TTL**: 14 days (must exceed max replication lag)
- **Region ID**: Each region has unique `REGION_ID` in metrics/logs

### 1.2 Health Probes

GSLB health checks:
- **Endpoint**: `GET /health`
- **Interval**: 10 seconds
- **Timeout**: 2 seconds
- **Threshold**: 3 consecutive failures → mark unhealthy

Backend readiness:
- Dedup store (Redis/Postgres) reachable
- WAL writer functional
- CPU/memory within bounds

### 1.3 Monitoring & Alerts

**Prometheus Alerts** (observability/prometheus/alerts.yml):
- `RegionDown`: Backend pods unavailable in region for >2 minutes
- `WalReplicationLag`: CRR lag >5 minutes (RPO warning)
- `CrossRegionLatency`: p95 latency between regions >200ms
- `DedupDivergence`: Dedup hit ratio drops >20% (split-brain indicator)

**Grafana Dashboards**:
- **Multi-Region Overview**: Traffic by region, error rates, latency
- **WAL Replication**: Lag chart, bytes replicated, failed transfers
- **Dedup Health**: Hit ratio, cache misses, shard distribution

---

## 2) Incident Detection

### 2.1 Symptoms

**Hard Failure (Region Down):**
- GSLB marks region unhealthy
- Alert: `RegionDown` fires
- 5xx errors spike in affected region
- Prometheus scrape failures for region

**Soft Failure (Degraded Performance):**
- p95/p99 latency increases >500ms
- Partial pod failures (some replicas down)
- Dedup store latency spikes
- WAL replication lag grows

### 2.2 Initial Triage (< 2 minutes)

**1. Confirm scope:**
```bash
# Check GSLB health
dig +short api.example.com  # Should return healthy region IPs only

# Check Prometheus for region status
curl -s 'http://prometheus:9090/api/v1/query?query=up{job="fractal-lba",region="eu-west"}' | jq '.data.result'

# Check backend pods
kubectl get pods -n fractal-lba -l app=backend,region=eu-west
```

**2. Identify root cause (preliminary):**
- Network partition? (ping, traceroute, cloud status pages)
- Node failures? (kubectl get nodes, cloud console)
- Application crash? (kubectl logs, error rates)
- Dedup store outage? (Redis/Postgres health)

**3. Assess impact:**
- Traffic percentage in failed region: check GSLB metrics
- Current RTO/RPO: check WAL replication lag
- User-facing errors: check frontend metrics, support tickets

---

## 3) Failover Procedures

### 3.1 Automatic Failover (Preferred)

**If GSLB health checks are working:**

GSLB automatically reroutes traffic to healthy region. **No manual action required** unless:
- WAL replication lag exceeds RPO (>2 minutes)
- Split-brain risk detected (see Section 5)

**Monitoring during automatic failover:**
```bash
# Watch traffic shift
watch -n 5 'kubectl top pods -n fractal-lba -l region=us-east'

# Check error rates
curl -s 'http://prometheus:9090/api/v1/query?query=rate(flk_ingest_total{status!~"2.."}[1m])' | jq '.data.result'

# Verify dedup idempotency
# (duplicate submissions should return cached results)
```

### 3.2 Manual Failover (GSLB Failure)

**If GSLB does not automatically reroute:**

**Step 1: Update DNS manually**
```bash
# Route53 example: point api.example.com to us-east ALB
aws route53 change-resource-record-sets --hosted-zone-id Z1234 --change-batch '{
  "Changes": [{
    "Action": "UPSERT",
    "ResourceRecordSet": {
      "Name": "api.example.com",
      "Type": "CNAME",
      "TTL": 60,
      "ResourceRecords": [{"Value": "us-east-alb.example.com"}]
    }
  }]
}'
```

**Step 2: Verify TTL propagation**
```bash
# Check DNS resolution from multiple locations
dig @8.8.8.8 api.example.com
dig @1.1.1.1 api.example.com
```

**Step 3: Monitor traffic shift**
```bash
# Traffic should shift to us-east within DNS TTL (60s)
watch -n 10 'curl -s http://prometheus:9090/api/v1/query?query=sum(rate(flk_ingest_total[1m])) by (region) | jq'
```

### 3.3 Degraded Mode (WAL Lag Exceeds RPO)

**If WAL replication lag >2 minutes:**

**Decision:** Accept potential duplicate processing or wait for replication?

**Option A: Continue with degraded mode (HIGH RISK)**
- Risk: Recent PCS may not be replicated yet → duplicate processing in failover region
- Mitigation: Idempotency via `pcs_id` dedup prevents double-effects
- Action: Enable `DEGRADED_MODE=true` env var, log all PCS IDs for reconciliation

**Option B: Pause writes until replication catches up (LOW RISK)**
- Risk: Service unavailable during catch-up window
- Mitigation: Return 503 with `Retry-After` header
- Action: Scale backend to 0, wait for WAL replication, then scale up

**Recommended: Option A (idempotency protects against duplicates)**

```bash
# Enable degraded mode in us-east
kubectl set env deployment/backend -n fractal-lba DEGRADED_MODE=true --container=backend

# Monitor WAL replication lag
watch -n 30 's3api list-object-versions --bucket flk-wal-eu-west --prefix inbox/ | jq ".Versions | length"'
```

---

## 4) Post-Failover Verification

### 4.1 Smoke Tests (< 5 minutes post-failover)

**1. Submit synthetic PCS:**
```bash
# Use test PCS with known signature
curl -X POST https://api.example.com/v1/pcs/submit \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: test' \
  -d @tests/data/golden_pcs.json
# Expected: 200 OK, accepted=true
```

**2. Verify dedup idempotency:**
```bash
# Submit same PCS again
curl -X POST https://api.example.com/v1/pcs/submit \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: test' \
  -d @tests/data/golden_pcs.json
# Expected: 200 OK (cached result), dedup hit metric increments
```

**3. Check WAL integrity:**
```bash
# List recent WAL segments in failover region
aws s3 ls s3://flk-wal-us-east/inbox/ --recursive | tail -20
# Verify timestamps are continuous (no gaps)
```

**4. Verify metrics:**
```bash
# Check Prometheus metrics in failover region
curl -s 'http://prometheus:9090/api/v1/query?query=flk_ingest_total{region="us-east"}' | jq '.data.result[0].value[1]'
# Should show increasing count
```

### 4.2 WAL Replay Verification (if degraded mode was used)

**If WAL replication lag caused duplicate submissions:**

**1. Identify overlapping window:**
```bash
# Find first WAL segment written after failover
FAILOVER_TIME="2025-01-15T10:30:00Z"
aws s3 ls s3://flk-wal-us-east/inbox/ --recursive | awk -v t="$FAILOVER_TIME" '$1" "$2 > t {print $4}'
```

**2. Replay WAL segments idempotently:**
```bash
# Download segments
aws s3 sync s3://flk-wal-us-east/inbox/ ./wal-replay/

# Replay with dedup (first-write wins)
for segment in ./wal-replay/*.jsonl; do
  while IFS= read -r line; do
    pcs_id=$(echo "$line" | jq -r '.pcs_id')
    # Check if already processed
    if ! redis-cli -h dedup-redis GET "pcs:$pcs_id" > /dev/null; then
      echo "Processing new PCS: $pcs_id"
      curl -X POST https://api.example.com/v1/pcs/submit -d "$line"
    else
      echo "Skipping duplicate: $pcs_id"
    fi
  done < "$segment"
done
```

**3. Reconciliation report:**
```bash
# Count duplicates detected
grep "Skipping duplicate" replay.log | wc -l
# Count new PCS processed
grep "Processing new PCS" replay.log | wc -l
```

---

## 5) Split-Brain Detection & Resolution

**Scenario:** Both regions believe they are primary (network partition heals, but dedup state diverged).

### 5.1 Detection

**Symptom:** Dedup hit ratio drops significantly after failover.

**Check for divergence:**
```bash
# Compare dedup state in both regions
REGION_A_KEYS=$(redis-cli -h eu-west-redis --scan --pattern "pcs:*" | wc -l)
REGION_B_KEYS=$(redis-cli -h us-east-redis --scan --pattern "pcs:*" | wc -l)
echo "eu-west keys: $REGION_A_KEYS, us-east keys: $REGION_B_KEYS"
# If difference >10%, investigate
```

### 5.2 Resolution Strategy

**WAL is source of truth** (Phase 1 invariant: WAL before parse).

**Step 1: Identify authoritative WAL**
```bash
# WAL with earliest complete replay wins
aws s3 ls s3://flk-wal-eu-west/inbox/ --recursive | sort
aws s3 ls s3://flk-wal-us-east/inbox/ --recursive | sort
# Choose region with most complete WAL coverage
```

**Step 2: Rebuild dedup state from WAL**
```bash
# Drain traffic from both regions (503)
kubectl scale deployment/backend -n fractal-lba --replicas=0 --all-namespaces

# Clear dedup state in both regions
redis-cli -h eu-west-redis FLUSHDB
redis-cli -h us-east-redis FLUSHDB

# Replay WAL from authoritative region (eu-west assumed)
./scripts/wal-replay.sh --source s3://flk-wal-eu-west/inbox/ --target redis://eu-west-redis

# Replicate rebuilt state to us-east
redis-cli -h eu-west-redis --rdb /tmp/dump.rdb SAVE
aws s3 cp /tmp/dump.rdb s3://flk-dedup-backups/eu-west-$(date +%s).rdb
# Restore in us-east
aws s3 cp s3://flk-dedup-backups/eu-west-*.rdb /tmp/restore.rdb
redis-cli -h us-east-redis --rdb /tmp/restore.rdb RESTORE

# Resume traffic
kubectl scale deployment/backend -n fractal-lba --replicas=3 --all-namespaces
```

**Step 3: Verify consistency**
```bash
# Both regions should now have identical dedup state
REGION_A_KEYS=$(redis-cli -h eu-west-redis DBSIZE | awk '{print $2}')
REGION_B_KEYS=$(redis-cli -h us-east-redis DBSIZE | awk '{print $2}')
if [ "$REGION_A_KEYS" -eq "$REGION_B_KEYS" ]; then
  echo "✅ Dedup state consistent"
else
  echo "❌ Still diverged: eu-west=$REGION_A_KEYS, us-east=$REGION_B_KEYS"
fi
```

---

## 6) Recovery of Failed Region

### 6.1 Once Failed Region is Healthy

**Step 1: Verify infrastructure**
```bash
# Check nodes
kubectl get nodes -l region=eu-west
# All nodes should be Ready

# Check backend pods
kubectl get pods -n fractal-lba -l app=backend,region=eu-west
# All pods should be Running with 1/1 Ready
```

**Step 2: Sync dedup state from healthy region**
```bash
# Snapshot us-east Redis (authoritative during outage)
redis-cli -h us-east-redis SAVE
aws s3 cp /var/lib/redis/dump.rdb s3://flk-dedup-backups/us-east-$(date +%s).rdb

# Restore to eu-west Redis
aws s3 cp s3://flk-dedup-backups/us-east-*.rdb /tmp/restore.rdb
redis-cli -h eu-west-redis FLUSHDB
redis-cli -h eu-west-redis --rdb /tmp/restore.rdb RESTORE
```

**Step 3: Enable GSLB health checks**
```bash
# Mark eu-west as healthy in GSLB
aws route53 change-resource-record-sets --hosted-zone-id Z1234 --change-batch '{
  "Changes": [{
    "Action": "UPSERT",
    "ResourceRecordSet": {
      "Name": "api.example.com",
      "Type": "CNAME",
      "TTL": 60,
      "SetIdentifier": "eu-west",
      "Weight": 50,
      "ResourceRecords": [{"Value": "eu-west-alb.example.com"}]
    }
  }]
}'
```

**Step 4: Gradual traffic ramp**
```bash
# Start with 10% traffic to eu-west
# Monitor error rates, latency, dedup hit ratio for 10 minutes
# If stable, increase to 50%, then 100% over 30 minutes
```

### 6.2 Post-Recovery Verification

**1. Check WAL continuity:**
```bash
# Ensure no gaps in WAL segments
aws s3 ls s3://flk-wal-eu-west/inbox/ --recursive | awk '{print $4}' | sort | while read f; do
  echo "Checking $f"
  aws s3api head-object --bucket flk-wal-eu-west --key "$f" | jq -r '.LastModified'
done
# Timestamps should be continuous
```

**2. Verify dedup parity:**
```bash
# Dedup state should be consistent across regions
diff <(redis-cli -h eu-west-redis KEYS "pcs:*" | sort) \
     <(redis-cli -h us-east-redis KEYS "pcs:*" | sort)
# Should show no differences (or only recent writes)
```

**3. Run E2E tests:**
```bash
# Submit test PCS to both regions
pytest tests/e2e-geo/test_multi_region.py -v
# All tests should pass
```

---

## 7) Communication Templates

### 7.1 Incident Start

**Subject:** [P1] Region Failover in Progress - Fractal LBA
**To:** #incidents, engineering-all

**Body:**
```
⚠️ INCIDENT: Multi-region failover initiated

Time: 2025-01-15 10:30 UTC
Affected Region: eu-west
Status: Traffic rerouted to us-east
Impact: No user-facing errors expected (automatic failover)
RTO: 5 minutes (target)
RPO: 2 minutes (target)

Current Actions:
- GSLB health checks marked eu-west unhealthy
- Traffic shifted to us-east (100%)
- Monitoring WAL replication lag: 45 seconds (within RPO)
- Investigating eu-west root cause

Updates: Every 15 minutes in #incidents
```

### 7.2 Incident Resolution

**Subject:** [RESOLVED] Multi-Region Failover - Fractal LBA
**To:** #incidents, engineering-all

**Body:**
```
✅ RESOLVED: Multi-region failover completed successfully

Duration: 4 minutes (RTO target: 5 minutes)
Data Loss: None (RPO: 0 minutes, within 2-minute target)
Root Cause: [Brief summary, e.g., "eu-west AZ-1 network partition"]

Outcome:
- 100% traffic on us-east during outage
- Zero user-facing errors (automatic failover)
- Dedup idempotency verified (no duplicate processing)
- WAL replication verified consistent

Next Steps:
- Post-mortem scheduled for 2025-01-16 14:00 UTC
- Monitor us-east for 24 hours
- Plan eu-west recovery (ETA: 2025-01-16 08:00 UTC)
```

---

## 8) Playbook Checklist

**Failover (< 5 minutes):**
- [ ] Confirm region is down (GSLB health checks)
- [ ] Verify automatic traffic reroute (or trigger manual)
- [ ] Check WAL replication lag (<2 minutes = within RPO)
- [ ] Monitor error rates in healthy region (target <1%)
- [ ] Enable degraded mode if WAL lag exceeds RPO
- [ ] Submit smoke test PCS, verify 200 OK
- [ ] Post incident start notification

**Post-Failover (< 30 minutes):**
- [ ] Verify dedup idempotency (submit duplicate PCS)
- [ ] Check WAL integrity (no gaps in segments)
- [ ] Run E2E tests against failover region
- [ ] Identify root cause of failure
- [ ] Plan recovery timeline

**Recovery (< 2 hours):**
- [ ] Verify failed region infrastructure healthy
- [ ] Sync dedup state from authoritative region
- [ ] Enable GSLB health checks for recovered region
- [ ] Gradual traffic ramp (10% → 50% → 100%)
- [ ] Verify WAL continuity and dedup parity
- [ ] Post incident resolution notification
- [ ] Schedule post-mortem

---

## 9) Related Runbooks

- **geo-split-brain.md**: Detailed split-brain resolution procedures
- **dedup-outage.md**: Dedup store recovery (Phase 2)
- **shard-migration.md**: Shard rebalancing during recovery (Phase 4)

---

## 10) References

- CLAUDE_PHASE4.md WP1: Multi-Region Active-Active & DR
- CLAUDE.md Section 2.2: Fault tolerance patterns
- docs/architecture/overview.md: Multi-region architecture diagram
- observability/prometheus/alerts.yml: RegionDown, WalReplicationLag alerts

---

**End of Runbook**
