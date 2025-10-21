# Architecture: Multi-Region & Disaster Recovery

## Overview

This document describes the multi-region architecture and disaster recovery capabilities introduced in Phase 4-5.

## Active-Active Topology

### Design Principles

- **Active-Active:** All regions accept writes simultaneously
- **Idempotent Replay:** WAL segments replay safely with first-write wins
- **Eventual Consistency:** Regions converge via CRR within 60s (SLO)

### Regional Components

Each region deploys:
1. **Backend cluster** (3+ replicas with HPA)
2. **Dedup store** (Redis/Postgres, region-local)
3. **Inbox WAL** (persistent volume)
4. **CRR Shipper** (Phase 5, ships WAL segments to peers)
5. **CRR Reader** (Phase 5, reads segments from peers, replays to local dedup)

### Cross-Region Replication (CRR)

**Shipper Process:**
```
1. Monitor Inbox WAL for new segments
2. Compute SHA-256 checksum
3. Ship segment to target regions (HTTP POST with retry)
4. Persist watermark (last shipped offset)
```

**Reader Process:**
```
1. Fetch segment from source region
2. Verify checksum
3. Replay entries to local dedup (idempotent)
4. Update last-read watermark
```

**Key Properties:**
- At-least-once delivery (retries with exponential backoff)
- Idempotent replay (first-write wins, duplicates ignored)
- Resumable (watermark persistence)

## RTO/RPO Targets

### Recovery Time Objective (RTO)

**Target:** ≤ 5 minutes

**Procedures:**
1. Health probe detects region failure (<30s)
2. DNS/load balancer switches traffic (1-2 min)
3. Peer regions absorb load (HPA scales up, 2-3 min)

**See:** [Geo Failover Runbook](../runbooks/geo-failover.md)

### Recovery Point Objective (RPO)

**Target:** ≤ 2 minutes

**Rationale:** CRR lag SLO is 60s; in worst case, up to 2 min of writes may be in-flight

**Mitigation:**
- Inbox WAL fsync on every write (durability)
- CRR shipper runs continuously (low lag)
- Alerts fire when lag >60s

## Divergence Detection

### Scenarios

**Split-Brain:** Network partition isolates regions, they process different writes for same pcs_id

**Conflicting Outcomes:** Region A accepts PCS, Region B escalates (different tolerance params)

### Detection

**CRR Divergence Detector** (Phase 5):
```
1. Periodically scan dedup stores across regions
2. For each pcs_id, compare outcomes (accepted, escalated, rejected)
3. If mismatch: alert and log to WORM
```

**Prometheus Alert:** `CRRDivergenceDetected`

### Resolution

**See:** [Geo Split-Brain Runbook](../runbooks/geo-split-brain.md)

**Strategy:** WAL is source of truth, first-write wins

1. Identify overlap window (WAL entries for conflicting pcs_id)
2. Determine first write by timestamp
3. Reconcile dedup stores (overwrite with first-write outcome)
4. Re-validate with ensemble (Phase 7+)

## Deployment Patterns

### Two-Region Active-Active

```
[us-east-1]  ←→ CRR ←→  [us-west-2]
```

**Configuration:**
```yaml
# us-east-1
crr:
  enabled: true
  sourceRegion: us-east-1
  targetRegions: [us-west-2]

# us-west-2
crr:
  enabled: true
  sourceRegion: us-west-2
  targetRegions: [us-east-1]
```

### Three-Region Multi-Way

```
[us-east-1] ←→ [us-west-2]
     ↕              ↕
       [eu-west-1]
```

**Configuration:**
```yaml
# us-east-1
crr:
  enabled: true
  replicationMode: multi-way
  peers: [us-west-2, eu-west-1]
```

### Selective Replication (Phase 6)

Replicate only specific tenants to specific regions:

```yaml
crr:
  enabled: true
  replicationMode: selective
  policies:
    - tenantID: tenant-a
      targetRegions: [us-west-2]
    - tenantID: tenant-b
      targetRegions: [eu-west-1]
```

## Failover Procedures

### Automatic Failover

**Health Probes:**
- Kubernetes liveness/readiness checks
- External uptime monitoring (Pingdom, etc.)
- Inter-region health checks

**Actions:**
1. Failed health checks → pod restart (liveness)
2. Sustained failures → remove from service (readiness)
3. Entire region down → DNS/GLB redirects traffic

### Manual Failover

```bash
# Drain region (stop accepting new writes)
kubectl scale deployment fractal-lba --replicas=0 -n flk-system

# Wait for CRR lag to drain (<60s)
watch 'kubectl get metrics flk_crr_lag_seconds'

# Update DNS to remove failed region
# (Method depends on DNS provider: Route53, CloudDNS, etc.)
```

### Failback

```bash
# Rebuild dedup from WAL (if lost)
./scripts/rebuild-dedup-from-wal.sh --region us-east-1 --since-time <timestamp>

# Re-enable region in load balancer
# Gradually ramp traffic (10% → 25% → 50% → 100%)
```

## Testing & Validation

### Chaos Drills

**Monthly exercises:**
1. Kill random region (verify RTO <5 min)
2. Introduce network partition (verify split-brain detection)
3. Corrupt dedup store (verify rebuild from WAL)

**See:** [Chaos Test Suite](../testing/chaos.md)

### Metrics

Key metrics for geo-DR:
- `flk_crr_lag_seconds` (SLO: ≤60s)
- `flk_crr_divergence_count` (target: 0)
- `flk_region_health_status` (1=healthy, 0=degraded)

### Alerts

- `CRRLagExceeded`: Lag >60s for >5 min
- `CRRDivergenceDetected`: Conflicting outcomes found
- `RegionUnhealthy`: Region health check failing

## Related Documentation

- [Architecture Overview](./overview.md)
- [Invariants](./invariants.md)
- [Runbooks: Geo Failover](../runbooks/geo-failover.md)
- [Runbooks: Geo Split-Brain](../runbooks/geo-split-brain.md)
