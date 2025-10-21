# Operations: Policies

## Overview

This document describes the policy management system for the Fractal LBA verification layer, including verification parameters, regime thresholds, feature flags, and Kubernetes Operator-based policy management introduced in Phase 6-8.

## Policy Components

### 1. Verification Parameters

Server-side tolerances for PCS signal verification:

- **tolD** (default: 0.15): Maximum deviation for D̂ (fractal dimension) verification
- **tolCoh** (default: 0.05): Maximum deviation for coh★ (coherence) verification
- **tolR**: Compressibility tolerance (implementation-dependent)

### 2. Budget Weights

Formula: `budget = base + α(1 − r) + β·max(0, D̂ − D0) + γ·coh★`

Default parameters:
- **alpha** = 0.30 (compressibility weight)
- **beta** = 0.50 (fractal dimension weight)
- **gamma** = 0.20 (coherence weight)
- **base** = 0.10 (baseline budget)
- **D0** = 2.2 (dimension threshold)

### 3. Regime Thresholds

Three regimes determine trust tier:

- **sticky**: `coh★ ≥ 0.70` AND `D̂ ≤ 1.5` (high trust, low budget multiplier)
- **non_sticky**: `D̂ ≥ 2.6` (low trust, high budget multiplier)
- **mixed**: All other combinations (medium trust)

### 4. Feature Flags

- `disable_wal` (DANGEROUS): Skip WAL writes (never enable in production)
- `skip_signature` (DANGEROUS): Skip signature verification (never enable in production)
- `enable_ensemble` (Phase 7+): Enable N-of-M ensemble verification
- `enable_hrs_routing` (Phase 7+): Enable risk-aware routing via HRS
- `enable_anomaly_detection` (Phase 7+): Enable anomaly detection

## Kubernetes Operator CRDs (Phase 6+)

### TieringPolicy CRD

Manages tiered storage configuration (hot/warm/cold):

```yaml
apiVersion: fractal-lba.io/v1
kind: TieringPolicy
metadata:
  name: default-tiering
spec:
  hotTier:
    backend: redis
    ttl: 1h
    targetLatencyP95Ms: 5
  warmTier:
    backend: postgres
    ttl: 7d
    targetLatencyP95Ms: 50
  coldTier:
    backend: s3
    ttl: 90d
    targetLatencyP95Ms: 500
  predictivePromotion:
    enabled: true
    threshold: 0.7
```

### CRRPolicy CRD

Manages cross-region replication:

```yaml
apiVersion: fractal-lba.io/v1
kind: CRRPolicy
metadata:
  name: multi-region-crr
spec:
  replicationMode: full  # full | selective | multi-way
  sourceRegion: us-east-1
  targetRegions:
    - us-west-2
    - eu-west-1
  tenantSelector:
    include: ["tenant-a", "tenant-b"]
  autoReconcile:
    enabled: true
    mode: auto-safe
```

### RiskRoutingPolicy CRD (Phase 7+)

Risk-aware routing configuration:

```yaml
apiVersion: fractal-lba.io/v1
kind: RiskRoutingPolicy
metadata:
  name: default-risk-routing
spec:
  hrsConfig:
    modelVersion: "v1.2.0"
    minConfidence: 0.7
  riskBands:
    - threshold: 0.3
      action: accept
      budgetMultiplier: 1.0
    - threshold: 0.7
      action: rag_required
      budgetMultiplier: 0.7
      ensembleRequired: true
    - threshold: 0.9
      action: human_review
      budgetMultiplier: 0.3
```

### EnsemblePolicy CRD (Phase 7+)

Ensemble verification configuration:

```yaml
apiVersion: fractal-lba.io/v1
kind: EnsemblePolicy
metadata:
  name: default-ensemble
spec:
  nOfMConfig:
    n: 2
    m: 3
  checks:
    - name: pcs_recompute
      enabled: true
      weight: 1.0
      timeout: 100ms
    - name: micro_vote
      enabled: true
      weight: 0.8
      timeout: 30ms
    - name: rag_grounding
      enabled: true
      weight: 0.9
      timeout: 50ms
  siemIntegration:
    provider: splunk
    endpoint: https://splunk.example.com:8088
```

## Policy Versioning

Policies use SemVer (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes to verification invariants
- **MINOR**: New parameters with backward-compatible defaults
- **PATCH**: Documentation/clarification only

Policy hash (SHA-256) is logged in WORM audit trail for lineage tracking.

## Policy Lifecycle

### 1. Development

- Create policy YAML in `config/policies/`
- Validate with `policy validate` CLI
- Test in shadow mode with historical traces

### 2. Staging

- Apply to staging cluster with canary rollout
- Monitor for 24 hours with multi-objective gates
- Compare metrics vs control (Phase 8 policy simulator)

### 3. Production

- Apply with adaptive canary (5% → 10% → 25% → 50% → 100%)
- Auto-rollback on gate failures
- Monitor health gates: latency ≤200ms p95, escalation ≤2%, containment ≥98%

### 4. Rollback

Emergency rollback if:
- ≥2 health gates fail
- Critical alert fires (signature spike, dedup anomaly)
- Manual operator intervention

## Policy Best Practices

### Safety Invariants (DO NOT VIOLATE)

1. **Never disable WAL** (`disable_wal=false` always)
2. **Never skip signature verification in prod** (`skip_signature=false` always)
3. **Verify before dedup** (Phase 1 invariant)
4. **Monotonic N_j** (scale values must be non-decreasing)

### Tuning Guidelines

**If escalation rate too high (>2%):**
- Increase `tolD` by 0.05 increments (max: 0.25)
- Increase `tolCoh` by 0.02 increments (max: 0.10)
- Review signal computation in agents

**If false accepts detected:**
- Decrease `tolD` and `tolCoh`
- Enable ensemble verification with N=3, M=3
- Enable HRS routing with lower accept threshold

**If cost too high:**
- Enable predictive tiering (Phase 4+)
- Tune hot tier TTL (reduce from 1h to 30m)
- Enable cost advisor recommendations (Phase 8)

## Related Documentation

- [Architecture Overview](../architecture/overview.md)
- [Signal Computation](../architecture/signal-computation.md)
- [Security Overview](../security/overview.md)
- [SLOs & Alerts](../observability/slos.md)

## Runbooks

- [Policy Rollback Procedure](../runbooks/policy-rollback.md)
- [Signature Spike Response](../runbooks/signature-spike.md)
- [Escalation Rate Breach](../runbooks/escalation-breach.md)
