# Metrics Concurrency Architecture Decision

**Status:** Accepted
**Date:** 2025-10-22
**Context:** Phase 3-9 stub implementations

## Problem

All Phase 3-9 metrics structs have a concurrency bug where `GetMetrics()` methods return structs by value, which copies the embedded `sync.RWMutex`:

```go
type AnchoringMetrics struct {
    mu                    sync.RWMutex  // ← Problem!
    BatchesAnchored       int64
    SegmentsAnchored      int64
    // ...
}

func (b *BatchAnchoring) GetMetrics() AnchoringMetrics {
    b.metrics.mu.RLock()
    defer b.metrics.mu.RUnlock()
    return *b.metrics  // ← Copies the mutex!
}
```

**Impact:** `go vet` detects 24 "return copies lock value" errors across:
- internal/audit: 5 files (anchoring, optimizer, compliance, siem, worker)
- internal/crr: 4 files (divergence, reader, reconcile, shipper)
- internal/tiering: 3 files (colddriver, demoter, predictor)
- internal/sharding: 1 file (crossquery)
- internal/anomaly: 2 files (detector, detector_v2)
- internal/ensemble: 1 file (verifier)
- internal/cost: 4 files (billing_importer, forecaster)
- internal/hrs: 4 files (feature_store, model_registry, risk_scorer, training_scheduler)

## Solution Options Considered

### Option 1: Snapshot Pattern ✅ Correct (Future)
Create separate snapshot structs without mutexes:

```go
type AnchoringMetricsSnapshot struct {
    // No mutex field
    BatchesAnchored      int64
    SegmentsAnchored     int64
    // ...
}

func (b *BatchAnchoring) GetMetrics() AnchoringMetricsSnapshot {
    b.metrics.mu.RLock()
    defer b.metrics.mu.RUnlock()
    return AnchoringMetricsSnapshot{
        BatchesAnchored: b.metrics.BatchesAnchored,
        // ... copy all fields
    }
}
```

**Pros:**
- Thread-safe
- No data races
- Correct architecture

**Cons:**
- Requires matching all field names across 24 structs (420+ fields estimated)
- Phase 3-9 are stub implementations - field names may change
- High maintenance burden during rapid prototyping

### Option 2: Return Pointers ❌ Unsafe
Return `*Metrics` directly without copying:

```go
func (b *BatchAnchoring) GetMetrics() *AnchoringMetrics {
    return b.metrics  // No copy, no mutex copy
}
```

**Pros:**
- Simple, one-line change

**Cons:**
- **Data race:** Callers can read fields while updates happen
- Violates mutex synchronization guarantees
- Technically incorrect (even if pragmatic for metrics)

### Option 3: Advisory Linting ✅ **CHOSEN**
Make `go vet` advisory in CI, document the issue, fix during full implementation:

```yaml
- name: Run go vet
  working-directory: ./backend
  continue-on-error: true  # Advisory only - copy lock issues in Phase 3-9 stubs
  run: go vet ./...
```

**Pros:**
- Unblocks CI immediately
- Preserves visibility (warnings still show)
- No risk of introducing data races
- Defers proper fix to full implementation phase

**Cons:**
- Leaves technical debt
- Warnings remain in codebase

## Decision

**We chose Option 3 (Advisory Linting)** because:

1. **Phase 3-9 are stub implementations** - These packages are architectural placeholders with incomplete logic. Field names and structures will change during full implementation.

2. **CI must not be blocked** - Developers need green builds to merge PRs. Advisory linting preserves warnings without blocking progress.

3. **Proper fix requires full context** - Snapshot pattern (Option 1) requires knowing final field names. Implementing this now would create maintenance burden as stubs evolve.

4. **Safety over speed** - Option 2 (pointers) would be faster to implement but introduces data races. We prioritize correctness.

5. **Clear migration path** - When implementing Phase 3-9 fully, we'll use Option 1 (snapshot pattern) as the permanent fix.

## Implementation

### CI Configuration
`.github/workflows/ci.yml`:
```yaml
- name: Run go vet
  working-directory: ./backend
  continue-on-error: true  # Advisory only - copy lock issues in Phase 3-9 stubs
  run: go vet ./...
```

### Affected Files (24 total)
All files remain unchanged. The copy lock warnings are visible but don't fail CI:

```bash
$ go vet ./... 2>&1 | grep "return copies lock" | wc -l
24
```

### Test Status
All implemented packages continue passing tests:
- ✅ pkg/canonical
- ✅ internal/cache
- ✅ internal/verify
- ✅ internal/auth
- ✅ internal/routing

## Migration Plan (Phase 3-9 Full Implementation)

When implementing each phase fully:

1. **Define final struct fields** - Finalize metrics struct fields
2. **Create snapshot types** - Define `XxxMetricsSnapshot` structs without mutexes
3. **Implement Snapshot() methods** - Copy fields under mutex lock
4. **Update GetMetrics()** - Return snapshot instead of value
5. **Update callers** - Change type from `Metrics` to `MetricsSnapshot`
6. **Remove advisory flag** - Once all 24 files fixed, remove `continue-on-error: true`

### Example Implementation (Phase 5 - CRR)
```go
// Step 1: Final struct (crr/divergence.go)
type DivergenceMetrics struct {
    mu                   sync.RWMutex
    DivergedPCSCount     int64
    ReconciledCount      int64
    LastDivergenceAt     time.Time
    AvgDivergencePercent float64
}

// Step 2: Snapshot type (crr/metrics.go)
type DivergenceMetricsSnapshot struct {
    DivergedPCSCount     int64
    ReconciledCount      int64
    LastDivergenceAt     time.Time
    AvgDivergencePercent float64
}

// Step 3: Snapshot method
func (m *DivergenceMetrics) Snapshot() DivergenceMetricsSnapshot {
    m.mu.RLock()
    defer m.mu.RUnlock()
    return DivergenceMetricsSnapshot{
        DivergedPCSCount:     m.DivergedPCSCount,
        ReconciledCount:      m.ReconciledCount,
        LastDivergenceAt:     m.LastDivergenceAt,
        AvgDivergencePercent: m.AvgDivergencePercent,
    }
}

// Step 4: Update GetMetrics
func (d *DivergenceDetector) GetMetrics() DivergenceMetricsSnapshot {
    return d.metrics.Snapshot()
}
```

## Alternatives Rejected

### Make All Metrics Atomic
Use `atomic.Int64` instead of `int64` with mutexes:

**Rejected because:**
- Doesn't solve the mutex copy problem (still need sync for `time.Time`, `float64`)
- Complex migration for 24 structs
- Not suitable for all field types (e.g., `time.Time`, `string`)

### Use sync.Map
Replace struct fields with `sync.Map`:

**Rejected because:**
- Loses type safety
- More complex API
- Harder to export to Prometheus
- Overkill for simple metrics

## References

- Go vet documentation: https://pkg.go.dev/cmd/vet
- sync.RWMutex documentation: https://pkg.go.dev/sync#RWMutex
- Related GitHub issue: https://github.com/rkhokhla/kakeya/actions/runs/18717066142/job/53379135749

## Review & Approval

- **Proposed by:** Phase 3-9 Architecture Review
- **Reviewed by:** User (approved Option 3)
- **Status:** Accepted
- **Next review:** During Phase 3-9 full implementation (Q1 2025)
