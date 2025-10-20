# Runbook: Tiered Storage Cold Hit Latency (Phase 4 WP3)

**Audience:** SRE, Performance Engineers, Platform Engineers
**Scope:** Managing and optimizing tiered storage (hot/warm/cold) performance and cost
**Related:** CLAUDE_PHASE4.md WP3, backend/internal/tiering/manager.go

---

## 0) TL;DR

**Scenario:** Cold tier reads spike, causing p95 latency >200ms or storage costs explode.
**Goal:** Balance latency and cost via TTL policies, promote/demote strategies, capacity planning.
**Key Actions:** Monitor tier hit ratios, adjust TTL policies, preemptive warming, cost alerts.

---

## 1) Tiered Storage Architecture

### 1.1 Tier Definitions

```
┌──────────────────────────────────────────────────────────────┐
│                    Tiered Dedup Store                         │
└──────────────────────────────────────────────────────────────┘
         │
         ├─► Hot Tier (Redis)
         │   - TTL: 1 hour
         │   - Latency: p95 <5ms
         │   - Cost: $$$$ (high, in-memory)
         │   - Capacity: 10 GB (limited)
         │
         ├─► Warm Tier (Postgres)
         │   - TTL: 7 days
         │   - Latency: p95 <50ms
         │   - Cost: $$ (medium, SSD)
         │   - Capacity: 500 GB
         │
         └─► Cold Tier (Object Storage: S3/GCS)
             - TTL: Forever (or 90 days with lifecycle)
             - Latency: p95 <500ms
             - Cost: $ (low, object storage)
             - Capacity: Unlimited
```

**Key Trade-offs:**
- **Hot**: Fast, expensive, limited → cache recent/frequent PCS
- **Warm**: Medium speed/cost → medium-term storage
- **Cold**: Slow, cheap, infinite → archival and compliance

### 1.2 Read Path (Waterfall with Lazy Promotion)

**Implementation** (backend/internal/tiering/manager.go:74-111):
```go
func (ts *TieredStore) Get(ctx context.Context, key string) (*api.VerifyResult, error) {
    // Try hot tier first
    if ts.hot != nil {
        value, err := ts.hot.Get(ctx, key)
        if err == nil && value != nil {
            ts.metrics.HotHits++
            return value, nil  // ✅ Fast path
        }
    }

    // Try warm tier
    if ts.warm != nil {
        value, err := ts.warm.Get(ctx, key)
        if err == nil && value != nil {
            ts.metrics.WarmHits++
            go ts.promote(ctx, key, value, TierWarm, TierHot)  // 🔥 Lazy promote
            return value, nil
        }
    }

    // Try cold tier
    if ts.cold != nil {
        value, err := ts.cold.Get(ctx, key)
        if err == nil && value != nil {
            ts.metrics.ColdHits++
            go ts.promote(ctx, key, value, TierCold, TierWarm)  // ⚡ Lazy promote
            return value, nil  // ⚠️ Slow path
        }
    }

    return nil, fmt.Errorf("key not found in any tier: %s", key)
}
```

**Lazy Promotion:**
- Warm hit → async copy to Hot (no blocking latency)
- Cold hit → async copy to Warm (blocks user with cold latency first time)

### 1.3 Write Path

**New writes always go to Hot tier:**
```go
func (ts *TieredStore) Set(ctx context.Context, key string, value *api.VerifyResult, tenantID string) error {
    policy := ts.getPolicyForTenant(tenantID)
    if ts.hot != nil {
        return ts.hot.Set(ctx, key, value, policy.HotTTL)  // TTL = 1 hour
    }
    return nil
}
```

**Demotion (background job, TTL-driven):**
- Hot expires (TTL=1h) → background worker demotes to Warm (TTL=7d)
- Warm expires (TTL=7d) → background worker demotes to Cold (TTL=∞ or 90d)

---

## 2) Performance Symptoms & Detection

### 2.1 Symptom: Cold Hit Spike (High Latency)

**Indicators:**
- p95 latency >200ms (SLO breach)
- `tier_cold_hits` metric spikes
- Cold tier read rate >10% of total reads

**Prometheus Alert:**
```yaml
# observability/prometheus/alerts.yml
- alert: ColdTierLatencySpike
  expr: rate(tier_cold_hits[5m]) / rate(flk_ingest_total[5m]) > 0.1
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Cold tier read rate >10% (high latency risk)"
    description: "p95 latency may breach SLO (>200ms)"
    runbook: "docs/runbooks/tier-cold-hot-miss.md"
```

**Diagnosis:**
```bash
# Check tier hit distribution
curl -s 'http://prometheus:9090/api/v1/query?query=tier_hot_hits' | jq '.data.result[0].value[1]'
curl -s 'http://prometheus:9090/api/v1/query?query=tier_warm_hits' | jq '.data.result[0].value[1]'
curl -s 'http://prometheus:9090/api/v1/query?query=tier_cold_hits' | jq '.data.result[0].value[1]'

# Ideal distribution:
# Hot: 60-70%
# Warm: 20-30%
# Cold: <10%
```

**Root Causes:**
1. **Hot TTL too short**: Keys evicted before re-access
2. **Traffic pattern change**: Different PCS access pattern (e.g., new agent shard)
3. **Cache churn**: Shard migration or failover cleared hot cache
4. **Cold storage abuse**: Direct cold reads (API misconfiguration)

### 2.2 Symptom: Hot Tier Capacity Pressure

**Indicators:**
- Redis memory >85% of limit
- `tier_evictions` metric spikes
- Hot tier evicts keys before TTL expires (LRU eviction)

**Diagnosis:**
```bash
# Check Redis memory usage
redis-cli -h hot-redis info memory | grep used_memory_human
redis-cli -h hot-redis info memory | grep maxmemory_human
# If used > 85% of max → capacity pressure

# Check eviction count
redis-cli -h hot-redis info stats | grep evicted_keys
# If increasing rapidly → LRU evictions active
```

**Root Causes:**
1. **Hot TTL too long**: Keys stay in hot tier too long
2. **Traffic spike**: More unique PCS IDs than hot tier capacity
3. **Inefficient key sizes**: Large VerifyResult objects

### 2.3 Symptom: Cold Storage Cost Explosion

**Indicators:**
- Cloud billing alert: S3/GCS costs >$1000/month
- Cold tier size >10 TB
- Cold tier read/write requests >10M/month

**Diagnosis:**
```bash
# Check S3 bucket size
aws s3 ls s3://flk-cold-dedup --recursive --human-readable --summarize | grep "Total Size"

# Check request count (last 30 days)
aws cloudwatch get-metric-statistics \
  --namespace AWS/S3 \
  --metric-name NumberOfObjects \
  --dimensions Name=BucketName,Value=flk-cold-dedup \
  --start-time $(date -u -d '30 days ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 86400 \
  --statistics Sum
```

**Root Causes:**
1. **No cold TTL lifecycle**: Keys never deleted from cold tier
2. **Excessive promotion**: Cold hits trigger warm/hot promotions → duplicate storage
3. **Compliance over-retention**: Keeping PCS beyond required retention period

---

## 3) Remediation Procedures

### 3.1 Fix Cold Hit Spike (Latency)

**Option A: Increase Hot TTL (Quick Fix)**

**Effect:** Keep keys in hot tier longer → reduce warm/cold fallback

**Steps:**
```bash
# Update TierPolicy (backend/internal/tiering/manager.go:249-258)
kubectl patch configmap backend-config -n fractal-lba --type merge -p '{
  "data": {
    "TIER_HOT_TTL": "2h"  // Was 1h, now 2h
  }
}'

# Restart backend to apply
kubectl rollout restart deployment/backend -n fractal-lba
```

**Trade-off:** Higher hot tier memory pressure (may trigger LRU evictions)

**Monitor:**
```bash
# Watch cold hit rate (should drop within 10 minutes)
watch -n 60 'curl -s "http://prometheus:9090/api/v1/query?query=rate(tier_cold_hits[5m])" | jq ".data.result[0].value[1]"'
```

**Option B: Preemptive Warming (Medium Fix)**

**Effect:** Pre-load frequently accessed PCS into hot tier before user requests

**Steps:**
```bash
# Identify hot PCS IDs (top 10k most accessed in last 7 days)
# Fetch from WORM audit logs or Postgres warm tier access logs
psql -h warm-postgres -U fractal -d dedup -c \
  "SELECT pcs_id, COUNT(*) as access_count FROM access_log WHERE ts > NOW() - INTERVAL '7 days' GROUP BY pcs_id ORDER BY access_count DESC LIMIT 10000" \
  -t -A -F, > /tmp/hot-pcs-ids.csv

# Pre-load into hot tier
cat /tmp/hot-pcs-ids.csv | while IFS=, read pcs_id access_count; do
  # Fetch from warm tier
  VALUE=$(psql -h warm-postgres -U fractal -d dedup -c "SELECT value FROM dedup WHERE pcs_id='$pcs_id'" -t -A)
  # Write to hot tier
  redis-cli -h hot-redis SETEX "pcs:$pcs_id" 3600 "$VALUE"
done
```

**Trade-off:** One-time CPU/network spike during pre-load

**Option C: Per-Tenant Hot TTL (Long-Term Fix)**

**Effect:** Premium tenants get longer hot TTL → lower latency

**Steps:**
```yaml
# infra/helm/values.yaml
tiering:
  default:
    hot_ttl: 3600  # 1 hour
  tenants:
    premium-tenant:
      hot_ttl: 7200  # 2 hours
    standard-tenant:
      hot_ttl: 3600  # 1 hour
```

### 3.2 Fix Hot Tier Capacity Pressure

**Option A: Scale Up Hot Tier (Quick Fix)**

**Steps:**
```bash
# Increase Redis memory limit
kubectl patch statefulset redis-hot -n fractal-lba --type merge -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "redis",
          "resources": {
            "limits": {"memory": "8Gi"}  // Was 4Gi
          }
        }]
      }
    }
  }
}'

# Wait for rolling update
kubectl rollout status statefulset/redis-hot -n fractal-lba
```

**Trade-off:** Higher cost ($$$ → $$$$)

**Option B: Reduce Hot TTL (Medium Fix)**

**Steps:**
```bash
# Reduce TTL from 1h to 30m
kubectl patch configmap backend-config -n fractal-lba --type merge -p '{
  "data": {
    "TIER_HOT_TTL": "1800"  // 30 minutes
  }
}'
kubectl rollout restart deployment/backend -n fractal-lba
```

**Trade-off:** Increased warm/cold fallback → slightly higher p95 latency

**Option C: Compress VerifyResult (Long-Term Fix)**

**Steps:**
```go
// backend/internal/tiering/manager.go
// Compress VerifyResult before storing in hot tier
func (ts *TieredStore) Set(ctx context.Context, key string, value *api.VerifyResult, tenantID string) error {
    compressed := compress(value)  // gzip or snappy
    return ts.hot.Set(ctx, key, compressed, policy.HotTTL)
}

func (ts *TieredStore) Get(ctx context.Context, key string) (*api.VerifyResult, error) {
    compressed, err := ts.hot.Get(ctx, key)
    if err == nil {
        return decompress(compressed), nil
    }
    // fallback to warm/cold...
}
```

**Trade-off:** CPU overhead for compression/decompression (~5-10ms)

### 3.3 Fix Cold Storage Cost Explosion

**Option A: Enable Cold TTL Lifecycle (Quick Fix)**

**Steps:**
```bash
# S3 lifecycle policy: delete objects >90 days old
aws s3api put-bucket-lifecycle-configuration \
  --bucket flk-cold-dedup \
  --lifecycle-configuration '{
    "Rules": [{
      "Id": "DeleteOldPCS",
      "Status": "Enabled",
      "Filter": {"Prefix": "pcs/"},
      "Expiration": {"Days": 90}
    }]
  }'

# Verify lifecycle policy
aws s3api get-bucket-lifecycle-configuration --bucket flk-cold-dedup
```

**Trade-off:** Data deleted after 90 days (ensure compliance allows this)

**Option B: Transition to Glacier (Medium Fix)**

**Steps:**
```bash
# S3 lifecycle: transition to Glacier after 30 days, delete after 365 days
aws s3api put-bucket-lifecycle-configuration \
  --bucket flk-cold-dedup \
  --lifecycle-configuration '{
    "Rules": [{
      "Id": "TierToGlacier",
      "Status": "Enabled",
      "Filter": {"Prefix": "pcs/"},
      "Transitions": [{"Days": 30, "StorageClass": "GLACIER"}],
      "Expiration": {"Days": 365}
    }]
  }'
```

**Trade-off:** Glacier retrieval takes 3-5 hours (not suitable for live reads)

**Option C: Deduplicate Cold Storage (Long-Term Fix)**

**Steps:**
```bash
# Identify duplicate PCS IDs in cold storage (same PCS in hot+warm+cold)
aws s3 ls s3://flk-cold-dedup/pcs/ --recursive | awk '{print $4}' | sort > /tmp/cold-keys.txt
redis-cli -h hot-redis --scan --pattern "pcs:*" | sort > /tmp/hot-keys.txt
comm -12 /tmp/cold-keys.txt /tmp/hot-keys.txt > /tmp/duplicate-keys.txt

# Delete duplicates from cold (keep in hot/warm as source of truth)
cat /tmp/duplicate-keys.txt | while read key; do
  aws s3 rm s3://flk-cold-dedup/$key
done
```

**Trade-off:** Risk of accidental deletion (backup before running)

---

## 4) Monitoring & Alerting

### 4.1 Key Metrics

**Tier Hit Ratios:**
```promql
# Hot tier hit ratio (target: 60-70%)
tier_hot_hits / (tier_hot_hits + tier_warm_hits + tier_cold_hits)

# Warm tier hit ratio (target: 20-30%)
tier_warm_hits / (tier_hot_hits + tier_warm_hits + tier_cold_hits)

# Cold tier hit ratio (target: <10%)
tier_cold_hits / (tier_hot_hits + tier_warm_hits + tier_cold_hits)
```

**Latency by Tier:**
```promql
# p95 latency by tier
histogram_quantile(0.95, sum(rate(tier_read_duration_seconds_bucket[5m])) by (tier, le))
```

**Cost Metrics:**
```bash
# Hot tier cost (Redis): $memory_gb * $price_per_gb_hour
# Warm tier cost (Postgres): $storage_gb * $price_per_gb_month
# Cold tier cost (S3): $storage_gb * $0.023/month + $read_requests * $0.0004/1000
```

### 4.2 Prometheus Alerts

```yaml
# observability/prometheus/alerts.yml

# Latency SLO breach
- alert: TierLatencyBreachSLO
  expr: histogram_quantile(0.95, sum(rate(tier_read_duration_seconds_bucket[5m])) by (le)) > 0.2
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Tier read p95 latency >200ms (SLO breach)"
    runbook: "docs/runbooks/tier-cold-hot-miss.md"

# Cold tier read spike
- alert: ColdTierReadSpike
  expr: rate(tier_cold_hits[5m]) / rate(flk_ingest_total[5m]) > 0.15
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Cold tier read rate >15% (latency risk)"

# Hot tier capacity pressure
- alert: HotTierMemoryPressure
  expr: redis_memory_used_bytes{tier="hot"} / redis_memory_max_bytes{tier="hot"} > 0.85
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Hot tier memory >85% (eviction risk)"

# Cold tier cost alert
- alert: ColdTierCostHigh
  expr: s3_bucket_size_bytes{bucket="flk-cold-dedup"} > 10 * 1024^4  # 10 TB
  for: 1d
  labels:
    severity: info
  annotations:
    summary: "Cold tier storage >10TB (cost review recommended)"
```

### 4.3 Grafana Dashboard

**Panel 1: Tier Hit Distribution (Pie Chart)**
```promql
tier_hot_hits
tier_warm_hits
tier_cold_hits
```

**Panel 2: Tier Latency (Line Chart)**
```promql
histogram_quantile(0.95, sum(rate(tier_read_duration_seconds_bucket{tier="hot"}[5m])) by (le))
histogram_quantile(0.95, sum(rate(tier_read_duration_seconds_bucket{tier="warm"}[5m])) by (le))
histogram_quantile(0.95, sum(rate(tier_read_duration_seconds_bucket{tier="cold"}[5m])) by (le))
```

**Panel 3: Cost Trend (Area Chart)**
```promql
# Hot tier cost ($/hour)
redis_memory_used_bytes{tier="hot"} / 1024^3 * 0.05

# Warm tier cost ($/month)
pg_database_size_bytes{tier="warm"} / 1024^3 * 0.10

# Cold tier cost ($/month)
s3_bucket_size_bytes{bucket="flk-cold-dedup"} / 1024^3 * 0.023
```

---

## 5) Capacity Planning

### 5.1 Hot Tier Sizing

**Formula:**
```
Hot Memory = Avg PCS Size * Hot Hit Ratio * Ingest Rate * Hot TTL

Example:
- Avg PCS Size: 2 KB
- Hot Hit Ratio: 65%
- Ingest Rate: 1000 PCS/s
- Hot TTL: 1 hour (3600s)

Hot Memory = 2 KB * 0.65 * 1000 * 3600 = 4.68 GB

Recommended: 4.68 GB * 1.5 (safety margin) = 7 GB Redis
```

### 5.2 Warm Tier Sizing

**Formula:**
```
Warm Storage = Avg PCS Size * Warm Hit Ratio * Ingest Rate * Warm TTL

Example:
- Avg PCS Size: 2 KB
- Warm Hit Ratio: 25%
- Ingest Rate: 1000 PCS/s
- Warm TTL: 7 days (604800s)

Warm Storage = 2 KB * 0.25 * 1000 * 604800 = 302.4 GB

Recommended: 302 GB * 1.3 (safety margin) = 400 GB SSD
```

### 5.3 Cold Tier Sizing

**Formula:**
```
Cold Storage = Avg PCS Size * Ingest Rate * Retention Period

Example (1-year retention):
- Avg PCS Size: 2 KB
- Ingest Rate: 1000 PCS/s
- Retention: 365 days (31536000s)

Cold Storage = 2 KB * 1000 * 31536000 = 63 TB

Cost (S3 Standard): 63 TB * $0.023/GB/month = $1449/month
Cost (S3 Glacier): 63 TB * $0.004/GB/month = $252/month
```

---

## 6) Best Practices

### 6.1 TTL Policy Tuning

**Guideline:**
- Hot TTL: 1-2 hours (balance latency vs memory)
- Warm TTL: 7-14 days (compliance minimum)
- Cold TTL: 90-365 days (or ∞ for compliance)

**Tuning Process:**
1. Measure current tier hit ratios
2. If cold hits >10% → increase warm TTL
3. If hot memory >85% → decrease hot TTL or scale up
4. Re-measure after 24 hours

### 6.2 Promotion Strategy

**Current:** Lazy promotion (async after first read)
**Alternative:** Eager promotion (sync before returning to user)

**Trade-offs:**
| Strategy | Latency (first read) | Latency (second read) | Complexity |
|----------|----------------------|-----------------------|------------|
| Lazy | Slow (cold latency) | Fast (promoted) | Low |
| Eager | Slow (cold + promotion) | Fast (promoted) | Medium |

**Recommendation:** Stick with lazy promotion (Phase 4 implementation)

### 6.3 Cost Optimization

**1. Compress values in hot tier:**
- gzip or snappy compression
- Reduces memory by 50-70%
- Trade-off: +5-10ms CPU overhead

**2. Cold tier lifecycle policies:**
- Transition to Glacier after 30 days
- Delete after 365 days (if compliance allows)

**3. Per-tenant tier policies:**
- Premium tenants: longer hot TTL
- Standard tenants: shorter hot TTL, more cold reads

---

## 7) Playbook Checklist

**Cold Hit Spike (Latency):**
- [ ] Check tier hit distribution (hot/warm/cold %)
- [ ] Identify root cause (TTL too short, traffic pattern change, cache churn)
- [ ] Increase hot TTL (2h) or preemptive warming
- [ ] Monitor p95 latency (target: <200ms within 30 minutes)

**Hot Tier Capacity Pressure:**
- [ ] Check Redis memory usage (>85% = pressure)
- [ ] Scale up Redis (double memory) or reduce hot TTL
- [ ] Consider compression (long-term fix)
- [ ] Monitor eviction count (should drop to 0)

**Cold Storage Cost Explosion:**
- [ ] Check S3 bucket size (>10TB = high cost)
- [ ] Enable lifecycle policy (90-day deletion or Glacier transition)
- [ ] Deduplicate cold storage (remove hot+warm duplicates)
- [ ] Set billing alert ($1000/month threshold)

---

## 8) Related Runbooks

- **dedup-outage.md**: Dedup store recovery (Phase 2)
- **shard-migration.md**: Shard rebalancing (Phase 4)
- **geo-failover.md**: Multi-region failover (Phase 4)

---

## 9) References

- CLAUDE_PHASE4.md WP3: Tiered Storage (Hot→Warm→Cold)
- backend/internal/tiering/manager.go: TieredStore implementation
- AWS S3 Lifecycle Policies: https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html
- Redis Memory Optimization: https://redis.io/docs/manual/optimization/memory-optimization/

---

**End of Runbook**
