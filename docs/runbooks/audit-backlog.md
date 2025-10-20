# Runbook: Async Audit Backlog (Phase 4 WP4)

**Audience:** SRE, Compliance Officers, Platform Engineers
**Scope:** Managing async audit pipeline backlog and ensuring audit completeness
**Related:** CLAUDE_PHASE4.md WP4, backend/internal/audit/worm.go, PHASE3_REPORT.md

---

## 0) TL;DR

**Scenario:** Async audit workers fall behind, backlog grows, audit trail incomplete or delayed.
**Goal:** Drain backlog within SLO (<1 hour lag), ensure 100% audit coverage, prevent DLQ overflow.
**Key Actions:** Scale workers, investigate slow tasks, purge DLQ, verify audit completeness.

---

## 1) Async Audit Architecture

### 1.1 Why Async?

**Phase 3 Design** (WORM audit log):
- **Sync WORM append**: Every PCS ingestion writes minimal entry to WORM log (append-only, fsync)
- **Minimal fields**: `pcs_id`, `timestamp`, `tenant_id`, `outcome`, `verify_params_hash`
- **Latency impact**: ~2ms per PCS (acceptable)

**Phase 4 Enhancement** (async audit pipeline):
- **Heavy checks offloaded**: Extended anomaly analysis, batch anchoring, external attestations
- **Queue + Workers**: Decouple heavy tasks from hot path
- **Goal**: Keep ingest path p95 <200ms, drain audit tasks within 1 hour

### 1.2 Pipeline Components

```
┌──────────────────────────────────────────────────────────────┐
│                 Ingest Path (Hot Path)                       │
│  POST /v1/pcs/submit → Verify → Sync WORM Append (2ms)     │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Audit Queue (SQS/    │
                │  Redis Stream/Kafka)  │
                │  - Task: enrich_worm  │
                │  - Task: anchor_batch │
                │  - Task: attest_ext   │
                └───────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │      Audit Workers (HPA: 2-10 pods)   │
        │  - Read from queue                    │
        │  - Perform heavy task (5-60s)         │
        │  - Ack on success, DLQ on failure     │
        └───────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │  Enriched WORM Log + Anchoring Store  │
        │  - Extended anomaly scores            │
        │  - Segment Merkle roots anchored      │
        │  - External attestations              │
        └───────────────────────────────────────┘
```

### 1.3 Queue Schema

**Message Format (JSON):**
```json
{
  "task_id": "uuid",
  "task_type": "enrich_worm | anchor_batch | attest_external",
  "pcs_id": "...",
  "created_at": "2025-01-15T10:30:00Z",
  "retry_count": 0,
  "payload": {
    "worm_entry_id": "...",
    "merkle_root": "...",
    ...
  }
}
```

**Task Types:**
1. **enrich_worm**: Compute extended anomaly scores (VRF, sanity checks), append to WORM
2. **anchor_batch**: Compute Merkle root for WORM segment, push to external anchor (blockchain, Trillian)
3. **attest_external**: Request third-party attestation (notary service, compliance API)

---

## 2) Backlog Symptoms & Detection

### 2.1 Symptom: Audit Lag Exceeds SLO

**Indicators:**
- `audit_queue_age_seconds` metric >3600 (1 hour SLO breach)
- `audit_backlog_size` metric >10,000 messages
- Audit trail queries return "pending" status for recent PCS

**Prometheus Alert:**
```yaml
# observability/prometheus/alerts.yml
- alert: AuditBacklogHigh
  expr: audit_backlog_size > 10000
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Audit backlog >10k messages (lag risk)"
    description: "Async audit workers may be overwhelmed"
    runbook: "docs/runbooks/audit-backlog.md"

- alert: AuditLagExceedsSLO
  expr: audit_queue_age_seconds > 3600
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Audit lag >1 hour (SLO breach)"
    description: "Audit trail incomplete for recent PCS"
    runbook: "docs/runbooks/audit-backlog.md"
```

**Diagnosis:**
```bash
# Check queue depth
kubectl exec -n fractal-lba deployment/audit-worker -- redis-cli XLEN audit:queue
# If >10k → backlog

# Check oldest message age
kubectl exec -n fractal-lba deployment/audit-worker -- redis-cli XINFO STREAM audit:queue
# Look at "first-entry" timestamp, compare to now
```

### 2.2 Symptom: DLQ Overflow

**Indicators:**
- `audit_dlq_size` metric >1,000
- Repeated task failures (same `task_id` in DLQ multiple times)
- Worker logs show errors: "max retries exceeded"

**Diagnosis:**
```bash
# Check DLQ size
kubectl exec -n fractal-lba deployment/audit-worker -- redis-cli XLEN audit:dlq

# Inspect failed tasks
kubectl exec -n fractal-lba deployment/audit-worker -- redis-cli XRANGE audit:dlq - + COUNT 10
# Look for patterns: same task_type failing, error messages
```

**Common Failure Reasons:**
1. **External service timeout**: Anchoring service (blockchain node) down or slow
2. **Invalid payload**: Malformed WORM entry or missing fields
3. **Worker bug**: Code regression causing panic or error
4. **Resource exhaustion**: Worker OOM or CPU throttled

### 2.3 Symptom: Audit Completeness Gap

**Indicators:**
- Audit queries for recent PCS return "not found"
- WORM segment missing extended fields (anomaly scores, anchors)
- Compliance report shows <100% coverage

**Diagnosis:**
```bash
# Compare WORM entries to ingest count
WORM_COUNT=$(find /var/lib/worm/audit -name "*.jsonl" | xargs wc -l | tail -1 | awk '{print $1}')
INGEST_COUNT=$(curl -s 'http://prometheus:9090/api/v1/query?query=flk_ingest_total' | jq -r '.data.result[0].value[1]')
echo "WORM entries: $WORM_COUNT, Ingest total: $INGEST_COUNT"
# Should match (±1% for in-flight)

# Check for missing enrichment
grep '"enriched": false' /var/lib/worm/audit/*.jsonl | wc -l
# If >100 → backlog or worker failure
```

---

## 3) Remediation Procedures

### 3.1 Scale Up Audit Workers (Quick Fix)

**Effect:** Increase parallelism → drain backlog faster

**Steps:**
```bash
# Check current worker count
kubectl get pods -n fractal-lba -l app=audit-worker
# e.g., 2 pods running

# Scale up to 10 pods
kubectl scale deployment/audit-worker -n fractal-lba --replicas=10

# Monitor backlog drain rate
watch -n 30 'kubectl exec -n fractal-lba deployment/audit-worker -- redis-cli XLEN audit:queue'
# Should decrease steadily
```

**Auto-scaling (HPA):**
```yaml
# infra/helm/fractal-lba/templates/hpa-audit-worker.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: audit-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: audit-worker
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: External
    external:
      metric:
        name: audit_backlog_size
      target:
        type: AverageValue
        averageValue: "1000"  # Scale up if backlog >1000 per worker
```

**Trade-off:** Higher cost (more pods), potential resource contention

### 3.2 Investigate Slow Tasks (Medium Fix)

**Effect:** Identify and optimize bottleneck tasks

**Steps:**

**1. Check task latency distribution:**
```bash
# Query Prometheus for task duration histogram
curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95, sum(rate(audit_task_duration_seconds_bucket[5m])) by (task_type, le))' | jq

# Expected latencies:
# - enrich_worm: p95 <10s
# - anchor_batch: p95 <60s
# - attest_external: p95 <30s
```

**2. Identify slow task type:**
```bash
# If anchor_batch is slow (p95 >120s), investigate anchoring service
kubectl logs -n fractal-lba -l app=audit-worker --tail=100 | grep "anchor_batch" | grep "slow"

# Example output:
# [WARN] anchor_batch task_id=abc123 duration=145s (blockchain node timeout)
```

**3. Optimize slow task:**

**Option A: Increase timeout (if external service is slow but reliable)**
```bash
kubectl set env deployment/audit-worker -n fractal-lba ANCHOR_TIMEOUT=120s
# Was 60s, now 120s
```

**Option B: Batch multiple tasks (if task is I/O bound)**
```go
// backend/cmd/audit-worker/main.go
// Batch 100 WORM entries per anchor call (reduce round-trips)
func anchorBatch(entries []*WORMEntry) error {
    root := computeMerkleRoot(entries)
    return anchoringClient.Submit(root)  // 1 API call for 100 entries
}
```

**Option C: Circuit breaker (if external service is flaky)**
```go
// backend/cmd/audit-worker/main.go
// Skip anchoring if service is down (enqueue for retry later)
if !anchoringService.IsHealthy() {
    log.Warn("Anchoring service down, skipping task")
    return errRetryLater
}
```

### 3.3 Purge DLQ (High Risk, Emergency Only)

**Effect:** Clear permanently failed tasks to stop DLQ growth

**⚠️ RISK:** Audit completeness gap (some PCS will not have extended audit data)

**When to use:**
- DLQ >10,000 messages
- All messages are same failure (e.g., external service down for 24h)
- Compliance allows best-effort audit enrichment

**Steps:**

**1. Backup DLQ:**
```bash
kubectl exec -n fractal-lba deployment/audit-worker -- redis-cli XRANGE audit:dlq - + > /tmp/dlq-backup-$(date +%s).jsonl
# Save for post-mortem analysis
```

**2. Analyze DLQ patterns:**
```bash
# Count failures by task_type
jq -r '.task_type' /tmp/dlq-backup-*.jsonl | sort | uniq -c
# Example:
# 8523 anchor_batch
# 1234 attest_external
#  123 enrich_worm
```

**3. Purge DLQ (irreversible):**
```bash
kubectl exec -n fractal-lba deployment/audit-worker -- redis-cli DEL audit:dlq
# All DLQ messages deleted

# Re-enqueue fixable tasks (e.g., only anchor_batch failures, if service is back up)
jq -r 'select(.task_type == "anchor_batch")' /tmp/dlq-backup-*.jsonl | \
  kubectl exec -i -n fractal-lba deployment/audit-worker -- redis-cli XADD audit:queue "*" task "$(cat)"
```

**4. Document audit gap:**
```bash
# Create incident report
cat > /tmp/audit-gap-report.md <<EOF
# Audit Completeness Gap Report

**Incident:** DLQ purged on $(date)
**Affected PCS:** 8523 (anchor_batch failures)
**Root Cause:** Blockchain anchoring service outage (2025-01-14 10:00 - 2025-01-15 12:00 UTC)
**Mitigation:** DLQ purged, affected PCS re-enqueued for anchoring

**Audit Impact:**
- Sync WORM entries: ✅ 100% complete (minimal fields logged)
- Extended enrichment: ⚠️ 99.2% complete (0.8% missing during outage)
- Merkle anchoring: ⚠️ 98.5% complete (1.5% re-enqueued, pending)

**Compliance Status:** Best-effort audit maintained, gaps documented.
EOF
```

### 3.4 Verify Audit Completeness (Post-Remediation)

**Effect:** Ensure no PCS are missing audit entries

**Steps:**

**1. Compare ingest count to WORM entries:**
```bash
WORM_COUNT=$(find /var/lib/worm/audit -name "*.jsonl" | xargs wc -l | tail -1 | awk '{print $1}')
INGEST_COUNT=$(curl -s 'http://prometheus:9090/api/v1/query?query=flk_ingest_total' | jq -r '.data.result[0].value[1]')
DIFF=$(echo "$INGEST_COUNT - $WORM_COUNT" | bc)
echo "Missing WORM entries: $DIFF"
# Should be 0 (or <10 for in-flight)
```

**2. Check enrichment coverage:**
```bash
ENRICHED=$(grep '"enriched": true' /var/lib/worm/audit/*.jsonl | wc -l)
TOTAL=$(wc -l /var/lib/worm/audit/*.jsonl | tail -1 | awk '{print $1}')
COVERAGE=$(echo "scale=2; $ENRICHED / $TOTAL * 100" | bc)
echo "Enrichment coverage: $COVERAGE%"
# Target: >99%
```

**3. Sample audit queries:**
```bash
# Pick 100 random recent PCS IDs
SAMPLE_IDS=$(jq -r '.pcs_id' /var/lib/worm/audit/$(ls -t /var/lib/worm/audit | head -1) | shuf | head -100)

# Query audit API for each
for pcs_id in $SAMPLE_IDS; do
  AUDIT=$(curl -s "http://audit-api:8080/v1/audit/$pcs_id")
  if [ -z "$AUDIT" ]; then
    echo "Missing audit: $pcs_id"
  fi
done
# Should output 0 missing
```

---

## 4) Monitoring & Alerting

### 4.1 Key Metrics

**Queue Metrics:**
```promql
# Backlog size (messages in queue)
audit_backlog_size

# Oldest message age (seconds)
audit_queue_age_seconds

# DLQ size
audit_dlq_size
```

**Worker Metrics:**
```promql
# Tasks processed per second
rate(audit_tasks_processed[1m])

# Task duration (p95)
histogram_quantile(0.95, sum(rate(audit_task_duration_seconds_bucket[5m])) by (task_type, le))

# Task failure rate
rate(audit_tasks_failed[1m]) / rate(audit_tasks_processed[1m])
```

**Audit Coverage Metrics:**
```promql
# WORM entries with enrichment
audit_enriched_entries / audit_total_entries

# Anchored segments
audit_anchored_segments / audit_total_segments
```

### 4.2 Prometheus Alerts

```yaml
# observability/prometheus/alerts.yml

# Backlog SLO breach
- alert: AuditLagExceedsSLO
  expr: audit_queue_age_seconds > 3600
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Audit lag >1 hour (SLO breach)"
    runbook: "docs/runbooks/audit-backlog.md"

# Backlog growth
- alert: AuditBacklogGrowing
  expr: rate(audit_backlog_size[5m]) > 10
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Audit backlog growing >10/s (capacity issue)"

# DLQ overflow
- alert: AuditDLQOverflow
  expr: audit_dlq_size > 1000
  for: 10m
  labels:
    severity: critical
  annotations:
    summary: "Audit DLQ >1000 (repeated failures)"

# Worker failure rate
- alert: AuditWorkerFailureRateHigh
  expr: rate(audit_tasks_failed[5m]) / rate(audit_tasks_processed[5m]) > 0.05
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Audit worker failure rate >5%"

# Audit completeness gap
- alert: AuditCompletenessGap
  expr: audit_enriched_entries / audit_total_entries < 0.99
  for: 1h
  labels:
    severity: warning
  annotations:
    summary: "Audit enrichment coverage <99% (backlog or worker issue)"
```

### 4.3 Grafana Dashboard

**Panel 1: Backlog & Lag**
```promql
audit_backlog_size
audit_queue_age_seconds
```

**Panel 2: Worker Throughput**
```promql
rate(audit_tasks_processed[1m])
```

**Panel 3: Task Duration (Heatmap)**
```promql
sum(rate(audit_task_duration_seconds_bucket[5m])) by (task_type, le)
```

**Panel 4: DLQ Size**
```promql
audit_dlq_size
```

**Panel 5: Audit Coverage**
```promql
audit_enriched_entries / audit_total_entries * 100
```

---

## 5) Capacity Planning

### 5.1 Worker Sizing

**Formula:**
```
Workers Needed = (Ingest Rate * Avg Task Duration) / Worker Parallelism

Example:
- Ingest Rate: 1000 PCS/s
- Avg Task Duration: 10s (enrich_worm + anchor_batch + attest_external)
- Worker Parallelism: 10 (concurrent tasks per pod)

Workers Needed = (1000 * 10) / 10 = 1000 / 10 = 100 workers

With HPA: Min 10, Max 100, scale based on backlog
```

### 5.2 Queue Capacity

**Redis Stream:**
- **Capacity**: ~1M messages (2 GB memory)
- **Throughput**: ~10k messages/s

**SQS (if using AWS):**
- **Capacity**: Unlimited
- **Throughput**: ~3k messages/s (standard queue), unlimited (FIFO with batching)

**Kafka (if using):**
- **Capacity**: ~1 TB per partition
- **Throughput**: ~100k messages/s per partition

**Recommendation:** Start with Redis Stream (simple, sufficient for Phase 4 scale)

### 5.3 Backlog SLO

**Target:** 99% of audit tasks complete within 1 hour

**Monitoring:**
```promql
# Percentage of tasks completed within 1 hour
sum(rate(audit_tasks_processed{duration_bucket="<3600"}[1h])) /
sum(rate(audit_tasks_processed[1h])) > 0.99
```

---

## 6) Best Practices

### 6.1 Task Idempotency

**Problem:** Worker crashes mid-task, message redelivered, task executed twice.

**Solution:** Make all tasks idempotent:
```go
// backend/cmd/audit-worker/main.go
func enrichWORM(taskID, pcsID string) error {
    // Check if already enriched
    if wormLog.IsEnriched(pcsID) {
        log.Info("Task already completed, skipping", "task_id", taskID)
        return nil  // Ack without re-processing
    }

    // Perform enrichment
    enrichedData := computeAnomalyScores(pcsID)
    return wormLog.AppendEnrichment(pcsID, enrichedData)
}
```

### 6.2 Dead Letter Queue Management

**When to retry:**
- Transient errors (network timeout, service temporarily unavailable)
- Max retries: 3 with exponential backoff

**When to DLQ:**
- Permanent errors (malformed payload, missing dependencies)
- Max retries exceeded

**DLQ Review Process:**
- **Daily:** Check DLQ size, investigate patterns
- **Weekly:** Review DLQ messages, fix root causes
- **Monthly:** Purge DLQ after documenting audit gaps

### 6.3 Circuit Breaker Pattern

**Goal:** Stop overwhelming failing external services

**Implementation:**
```go
// backend/cmd/audit-worker/main.go
var anchoringCircuit = circuitbreaker.New(
    circuitbreaker.WithFailureThreshold(5),     // Open after 5 failures
    circuitbreaker.WithTimeout(60 * time.Second), // Try again after 60s
)

func anchorBatch(entries []*WORMEntry) error {
    return anchoringCircuit.Call(func() error {
        return anchoringClient.Submit(computeMerkleRoot(entries))
    })
}
```

**Effect:** Avoid filling DLQ with 10k anchor_batch failures during anchoring service outage

---

## 7) Playbook Checklist

**Backlog SLO Breach (Lag >1 hour):**
- [ ] Check backlog size (audit_backlog_size metric)
- [ ] Check worker count (kubectl get pods -l app=audit-worker)
- [ ] Scale up workers (2x current count)
- [ ] Monitor drain rate (backlog should decrease)
- [ ] Identify slow tasks (task duration histogram)
- [ ] Optimize slow tasks (increase timeout, batch, circuit breaker)

**DLQ Overflow (>1000 messages):**
- [ ] Backup DLQ (XRANGE to file)
- [ ] Analyze failure patterns (jq by task_type, error)
- [ ] Fix root cause (external service, worker bug, invalid payload)
- [ ] Re-enqueue fixable tasks
- [ ] Purge unfixable tasks (document audit gap)
- [ ] Update monitoring (prevent recurrence)

**Audit Completeness Gap (<99% enrichment):**
- [ ] Compare ingest count to WORM entries (should match ±1%)
- [ ] Check enrichment coverage (grep "enriched": true)
- [ ] Identify missing PCS IDs
- [ ] Re-enqueue enrichment tasks for missing PCS
- [ ] Verify completeness after re-processing
- [ ] Document any permanent gaps (compliance report)

---

## 8) Related Runbooks

- **dedup-outage.md**: Dedup store recovery (Phase 2)
- **geo-failover.md**: Multi-region failover (Phase 4)
- **tenant-slo-breach.md**: Tenant SLO incident response (Phase 3)

---

## 9) References

- CLAUDE_PHASE4.md WP4: Async Audit & Anchoring Pipeline
- backend/internal/audit/worm.go: WORM log implementation (Phase 3)
- PHASE3_REPORT.md: WORM audit log design
- Redis Streams: https://redis.io/docs/data-types/streams/
- Circuit Breaker pattern: https://martinfowler.com/bliki/CircuitBreaker.html

---

**End of Runbook**
