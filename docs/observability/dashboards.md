# Observability: Dashboards

## Overview

This document describes the Grafana dashboards used to monitor the Fractal LBA verification layer, including operational metrics, buyer KPIs, and Phase 1-8 feature coverage.

## Dashboard Inventory

### 1. Buyer KPIs Dashboard v3 (Phase 8)

**Location:** `observability/grafana/buyer_dashboard_v3.json`

**Target Audience:** Investors, C-suite, product managers

**Key Panels (18 total):**

#### Row 1: Top-Level KPIs
1. **Hallucination Containment Rate** (SLO: ≥98%)
   - Query: `(sum(rate(flk_accepted[7d])) / sum(rate(flk_ingest_total[7d]))) * 100`
   - Color thresholds: red (<97%), yellow (97-98%), green (≥98%)

2. **Cost Per Trusted Task** (USD)
   - Query: Total cost (compute + storage + network + anchoring) / accepted tasks
   - Thresholds: green (<$0.0015), yellow ($0.0015-$0.0020), red (>$0.0020)

3. **HRS Model AUC** (SLO: ≥0.85 shadow, ≥0.82 online)
   - Query: `flk_hrs_model_auc`
   - Indicates prediction quality of Hallucination Risk Scorer

4. **Ensemble Agreement Rate** (SLO: ≥85%)
   - Query: `(sum(rate(flk_ensemble_agreed[7d])) / sum(rate(flk_ensemble_total[7d]))) * 100`
   - High agreement → good signal quality

#### Row 2: Operational Health
5. **Escalation Rate** (SLO: ≤2%)
   - Time series with SLO threshold line
   - Tracks low-confidence PCS sent for review

6. **Cost Breakdown** (pie chart)
   - Compute, Storage, Network, Anchoring
   - Helps identify optimization opportunities

#### Row 3: Performance
7. **HRS Prediction Latency P95** (SLO: ≤10ms)
8. **Ensemble Verification Latency P95** (SLO: ≤120ms)

#### Row 4: Anomaly & Risk
9. **Anomaly Detection Rate** (Phase 8 WP4)
10. **High-Risk PCS Rate** (HRS ≥0.7)

#### Row 5: Cost & Budget
11. **Budget Utilization by Tenant** (Top 10)
12. **Cost Savings from Optimization** (Phase 8 WP3)

#### Row 6: Ensemble & Events
13. **Ensemble Disagreement Events** (Last 20)
    - Table showing PCS IDs where strategies disagreed

#### Row 7: Model Evaluation
14. **HRS ROC/PR Curves**
    - True Positive Rate vs False Positive Rate
    - Helps evaluate model quality

15. **Confidence Distribution** (heatmap)
    - Distribution of HRS confidence scores

#### Row 8: Trends & Comparison
16. **Containment Delta vs Control** (Phase 7 → Phase 8)
    - Measures improvement vs Phase 6 baseline (98.2%)

17. **Cost Per Trusted Task Trend** (30-day window)
    - Per-tenant cost trends

#### Row 9: Deployment
18. **Model Version Distribution**
    - Active HRS model versions across deployments

### 2. Operational Metrics Dashboard

**Focus:** System health, latency, throughput, errors

**Key Metrics:**

- **Ingestion Rate:** `rate(flk_ingest_total[1m])`
- **Deduplication Hit Ratio:** `rate(flk_dedup_hits[5m]) / rate(flk_ingest_total[5m])`
- **Verification Latency:** `histogram_quantile(0.95, rate(flk_verify_latency_ms_bucket[5m]))`
- **Error Rate:** `rate(flk_errors_total[1m])`
- **WAL Lag:** Time between write and ack
- **Signature Failures:** `rate(flk_signature_errors[1m])`

### 3. Multi-Tenant Dashboard (Phase 3+)

**Per-tenant breakdowns:**

- Ingestion volume: `sum(rate(flk_ingest_total_by_tenant[5m])) by (tenant_id)`
- Dedup hit ratio by tenant
- Escalation rate by tenant
- Quota utilization: `flk_budget_used / flk_budget_cap` by tenant
- Signature failures by tenant

### 4. CRR & Geo-Replication Dashboard (Phase 4-5)

**Cross-region monitoring:**

- **CRR Lag:** Time between source write and target replay
  - SLO: ≤60s
  - Query: `flk_crr_lag_seconds`

- **Divergence Detection:** Count of divergent PCS IDs across regions
  - Query: `flk_crr_divergence_count`

- **WAL Shipping Status:** Success/failure rates
  - Query: `rate(flk_crr_ship_success[5m])`

- **Region Health:** Per-region ingestion and error rates

### 5. Tiering & Storage Dashboard (Phase 4+)

**Storage tier metrics:**

- **Hot Tier Hit Rate:** `rate(flk_tier_hot_hits[5m]) / rate(flk_tier_total_requests[5m])`
- **Warm Tier Hit Rate:** Similar for warm tier
- **Cold Tier Hit Rate:** Similar for cold tier
- **Promotion Events:** `rate(flk_tier_promotions[5m])`
- **Demotion Events:** `rate(flk_tier_demotions[5m])`
- **Storage Cost Breakdown:** By tier (hot: Redis, warm: Postgres, cold: S3/GCS)

### 6. Audit & Compliance Dashboard (Phase 5-6)

**Audit trail coverage:**

- **WORM Write Success Rate:** `rate(flk_worm_writes_success[5m])`
- **Audit Backlog:** `flk_audit_queue_depth`
  - SLO: p99 <1h
- **Anchoring Success Rate:** `rate(flk_anchoring_success[5m]) / rate(flk_anchoring_total[5m])`
  - SLO: ≥99%
- **Compliance Controls:** Pass/fail status for SOC2/ISO controls

## Dashboard Access

### Local Development

```bash
# Start Grafana with Docker Compose
docker-compose up grafana

# Access at http://localhost:3000
# Default credentials: admin/admin
```

### Kubernetes

```bash
# Port-forward to Grafana service
kubectl port-forward svc/grafana 3000:3000 -n observability

# Access at http://localhost:3000
```

### Production

- Access via Ingress: `https://grafana.example.com`
- SSO via OAuth/SAML (configured per deployment)
- Read-only access for stakeholders via viewer role

## Dashboard Maintenance

### Import New Dashboards

```bash
# From JSON file
curl -X POST \
  -H "Content-Type: application/json" \
  -d @observability/grafana/buyer_dashboard_v3.json \
  http://admin:admin@localhost:3000/api/dashboards/db
```

### Version Control

All dashboard JSON files are stored in `observability/grafana/`:
- `buyer_dashboard_v3.json` (Phase 8)
- `operational_metrics.json`
- `multi_tenant.json`
- `crr_geo.json`
- `tiering_storage.json`
- `audit_compliance.json`

Commit changes with version bumps in dashboard metadata.

### Alert Integration

Dashboards link to Prometheus alert rules (`observability/prometheus/alerts.yml`):
- Click panel title → "View in Explore" → See related alerts
- Alert annotations appear as vertical lines on graphs

## Customization

### Adding Custom Panels

1. Edit dashboard JSON or use Grafana UI
2. Export JSON via "Share Dashboard" → "Export" → "Save to file"
3. Commit to version control
4. Update CHANGELOG with panel description

### Variables

Dashboards support template variables:
- `$tenant_id`: Filter by tenant
- `$region`: Filter by region
- `$time_range`: Rolling window (7d default)

## Related Documentation

- [SLOs & Alerts](./slos.md)
- [Prometheus Metrics Reference](./metrics.md)
- [Troubleshooting Guide](../operations/troubleshooting.md)

## Runbooks

- [Dashboard Not Loading](../runbooks/grafana-unavailable.md)
- [Missing Metrics](../runbooks/prometheus-gaps.md)
- [High Escalation Rate](../runbooks/escalation-spike.md)
