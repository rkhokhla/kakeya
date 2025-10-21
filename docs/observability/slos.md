# Observability: SLOs & Alerts

## Overview

This document defines Service Level Objectives (SLOs) and associated Prometheus alerts for the Fractal LBA verification layer.

## Core SLOs (Phase 1-2)

### Verify Latency

**Objective:** p95 verification latency ≤ 200ms

**Query:**
```promql
histogram_quantile(0.95, rate(flk_verify_latency_ms_bucket[5m])) <= 200
```

**Alert:**
```yaml
- alert: VerifyLatencyHigh
  expr: histogram_quantile(0.95, rate(flk_verify_latency_ms_bucket[5m])) > 200
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Verify latency p95 >200ms"
    runbook: docs/runbooks/high-latency.md
```

### Escalation Rate

**Objective:** Escalation rate ≤ 2%

**Query:**
```promql
(rate(flk_escalated[1h]) / rate(flk_ingest_total[1h])) * 100 <= 2
```

**Alert:**
```yaml
- alert: EscalationRateHigh
  expr: (rate(flk_escalated[1h]) / rate(flk_ingest_total[1h])) * 100 > 2
  for: 30m
  labels:
    severity: warning
  annotations:
    summary: "Escalation rate >2%"
    runbook: docs/runbooks/escalation-spike.md
```

## Phase 3+ SLOs

### Hallucination Containment (Phase 6+)

**Objective:** ≥98% of PCS accepted or escalated (not rejected)

**Query:**
```promql
(rate(flk_accepted[7d]) / rate(flk_ingest_total[7d])) * 100 >= 98
```

## Phase 4+ SLOs

### CRR Lag

**Objective:** Cross-region replication lag ≤60s

**Query:**
```promql
flk_crr_lag_seconds <= 60
```

**Alert:**
```yaml
- alert: CRRLagExceeded
  expr: flk_crr_lag_seconds > 60
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "CRR lag >60s"
    runbook: docs/runbooks/crr-lag.md
```

### Hot Tier Latency

**Objective:** Hot tier p95 ≤5ms

**Query:**
```promql
histogram_quantile(0.95, rate(flk_tier_hot_latency_ms_bucket[5m])) <= 5
```

## Phase 7+ SLOs

### HRS Prediction Latency

**Objective:** p95 ≤10ms

**Query:**
```promql
histogram_quantile(0.95, rate(flk_hrs_latency_ms_bucket[5m])) <= 10
```

### Ensemble Agreement

**Objective:** ≥85% agreement across verification strategies

**Query:**
```promql
(rate(flk_ensemble_agreed[7d]) / rate(flk_ensemble_total[7d])) * 100 >= 85
```

## Phase 8 SLOs

### Cost Reconciliation

**Objective:** Billing reconciliation error ≤±3%

**Query:**
```promql
abs((flk_cost_internal_total - flk_cost_cloud_billing_total) / flk_cost_cloud_billing_total) <= 0.03
```

### Forecast Accuracy

**Objective:** Cost forecast MAPE ≤10%

**Query:**
```promql
flk_cost_forecast_mape <= 10
```

## Alert Severity Levels

### Critical

- Service degradation affecting users
- SLO breach with customer impact
- Security incidents

**Response Time:** <15 minutes

**Examples:**
- Backend down
- CRR lag >120s
- Signature spike >100/s

### Warning

- SLO trending toward breach
- Resource pressure
- Degraded performance

**Response Time:** <1 hour

**Examples:**
- Verify latency >200ms
- Escalation rate >2%
- Dedup store slow

### Info

- Informational events
- Maintenance windows
- Policy changes

**Response Time:** Next business day

## Alert Routing

```yaml
# alertmanager.yml
route:
  receiver: team-oncall
  group_by: [alertname, severity]
  routes:
    - match:
        severity: critical
      receiver: pagerduty
      continue: true
    - match:
        severity: warning
      receiver: slack-alerts
```

## SLO Dashboard

**Grafana Dashboard:** `observability/grafana/slo_dashboard.json`

**Panels:**
- SLO compliance summary (% of time in SLO)
- Error budget burn rate
- Historical SLO violations
- Time to remediation

## Related Documentation

- [Dashboards](./dashboards.md)
- [Prometheus Alerts](../../observability/prometheus/alerts.yml)
- [Runbooks](../runbooks/)
