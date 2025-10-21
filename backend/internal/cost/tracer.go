package cost

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// CostTracer provides per-request cost attribution (Phase 7 WP3)
// Tracks compute, storage, network, anchoring costs by tenant/model/task
type CostTracer struct {
	mu              sync.RWMutex
	spans           map[string]*CostSpan // span_id → span
	aggregates      map[string]*CostAggregate // tenant_id → aggregate
	costModel       *CostModel
	metrics         *CostMetrics
}

// CostSpan represents a single request span with cost tracking
type CostSpan struct {
	SpanID         string
	ParentSpanID   string
	TenantID       string
	ModelID        string
	TaskType       string // "pcs_verify", "ensemble_check", "rag_query", etc.

	// Cost components (USD)
	ComputeCost    float64
	StorageCost    float64
	NetworkCost    float64
	AnchoringCost  float64
	TotalCost      float64

	// Metrics
	ComputeTimeMs  float64
	StorageBytes   int64
	NetworkBytes   int64
	AnchoringOps   int

	StartTime      time.Time
	EndTime        time.Time
	Children       []string // Child span IDs
}

// CostAggregate tracks cumulative costs per tenant
type CostAggregate struct {
	TenantID       string
	ModelID        string
	WindowStart    time.Time
	WindowEnd      time.Time

	// Cumulative costs (USD)
	TotalCompute   float64
	TotalStorage   float64
	TotalNetwork   float64
	TotalAnchoring float64
	GrandTotal     float64

	// Request counts
	RequestCount   int64
	SuccessCount   int64
	ErrorCount     int64

	// Budget tracking
	SoftCap        float64
	HardCap        float64
	BudgetUsed     float64 // Percentage [0, 1]
}

// CostModel defines cost parameters (Phase 6 buyer KPI costs)
type CostModel struct {
	// Compute costs (USD per 100ms)
	ComputeUSD     float64 // $0.0001 per 100ms

	// Storage costs (USD per GB per month)
	StorageHotUSD  float64 // $0.023/GB/month (Redis)
	StorageWarmUSD float64 // $0.010/GB/month (Postgres)
	StorageColdUSD float64 // $0.004/GB/month (S3)

	// Network costs (USD per GB)
	NetworkUSD     float64 // $0.09/GB

	// Anchoring costs (USD per batch)
	AnchoringEthereumUSD   float64 // $2.00/batch
	AnchoringPolygonUSD    float64 // $0.01/batch
	AnchoringTimestampUSD  float64 // $0.01/batch
	AnchoringOpenTimestampUSD float64 // $0.00/batch (free)
}

// CostMetrics tracks cost attribution metrics
type CostMetrics struct {
	mu                   sync.RWMutex
	CostComputeUSD       *prometheus.CounterVec
	CostStorageUSD       *prometheus.CounterVec
	CostNetworkUSD       *prometheus.CounterVec
	CostAnchoringUSD     *prometheus.CounterVec
	CostPerTrustedTask   *prometheus.GaugeVec
	BudgetUsed           *prometheus.GaugeVec
}

// NewCostTracer creates a new cost tracer
func NewCostTracer() *CostTracer {
	return &CostTracer{
		spans:      make(map[string]*CostSpan),
		aggregates: make(map[string]*CostAggregate),
		costModel:  DefaultCostModel(),
		metrics: &CostMetrics{
			CostComputeUSD: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "flk_cost_compute_usd",
					Help: "Compute cost in USD",
				},
				[]string{"tenant_id", "model_id", "task_type"},
			),
			CostStorageUSD: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "flk_cost_storage_usd",
					Help: "Storage cost in USD",
				},
				[]string{"tenant_id", "tier"},
			),
			CostNetworkUSD: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "flk_cost_network_usd",
					Help: "Network cost in USD",
				},
				[]string{"tenant_id"},
			),
			CostAnchoringUSD: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "flk_cost_anchoring_usd",
					Help: "Anchoring cost in USD",
				},
				[]string{"tenant_id", "strategy"},
			),
			CostPerTrustedTask: promauto.NewGaugeVec(
				prometheus.GaugeOpts{
					Name: "flk_cost_per_trusted_task",
					Help: "Cost per trusted task in USD",
				},
				[]string{"tenant_id", "model_id"},
			),
			BudgetUsed: promauto.NewGaugeVec(
				prometheus.GaugeOpts{
					Name: "flk_budget_used",
					Help: "Budget utilization [0, 1]",
				},
				[]string{"tenant_id"},
			),
		},
	}
}

// DefaultCostModel returns default cost parameters
func DefaultCostModel() *CostModel {
	return &CostModel{
		ComputeUSD:             0.0001, // $0.0001 per 100ms
		StorageHotUSD:          0.023,  // $0.023/GB/month (Redis)
		StorageWarmUSD:         0.010,  // $0.010/GB/month (Postgres)
		StorageColdUSD:         0.004,  // $0.004/GB/month (S3)
		NetworkUSD:             0.09,   // $0.09/GB
		AnchoringEthereumUSD:   2.00,   // $2.00/batch
		AnchoringPolygonUSD:    0.01,   // $0.01/batch
		AnchoringTimestampUSD:  0.01,   // $0.01/batch
		AnchoringOpenTimestampUSD: 0.00, // Free
	}
}

// StartSpan creates a new cost span
func (ct *CostTracer) StartSpan(ctx context.Context, spanID, tenantID, modelID, taskType string) *CostSpan {
	span := &CostSpan{
		SpanID:    spanID,
		TenantID:  tenantID,
		ModelID:   modelID,
		TaskType:  taskType,
		StartTime: time.Now(),
		Children:  []string{},
	}

	ct.mu.Lock()
	ct.spans[spanID] = span
	ct.mu.Unlock()

	return span
}

// EndSpan finalizes a span and computes costs
func (ct *CostTracer) EndSpan(ctx context.Context, spanID string) error {
	ct.mu.Lock()
	span, ok := ct.spans[spanID]
	if !ok {
		ct.mu.Unlock()
		return fmt.Errorf("span not found: %s", spanID)
	}
	ct.mu.Unlock()

	span.EndTime = time.Now()

	// Compute costs
	span.ComputeCost = ct.computeComputeCost(span)
	span.StorageCost = ct.computeStorageCost(span)
	span.NetworkCost = ct.computeNetworkCost(span)
	span.AnchoringCost = ct.computeAnchoringCost(span)
	span.TotalCost = span.ComputeCost + span.StorageCost + span.NetworkCost + span.AnchoringCost

	// Update aggregates
	ct.updateAggregate(span)

	// Record metrics
	ct.recordCostMetrics(span)

	return nil
}

// RecordCompute records compute time for a span
func (ct *CostTracer) RecordCompute(spanID string, durationMs float64) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	if span, ok := ct.spans[spanID]; ok {
		span.ComputeTimeMs += durationMs
	}
}

// RecordStorage records storage usage for a span
func (ct *CostTracer) RecordStorage(spanID string, bytes int64, tier string) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	if span, ok := ct.spans[spanID]; ok {
		span.StorageBytes += bytes
	}
}

// RecordNetwork records network transfer for a span
func (ct *CostTracer) RecordNetwork(spanID string, bytes int64) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	if span, ok := ct.spans[spanID]; ok {
		span.NetworkBytes += bytes
	}
}

// RecordAnchoring records anchoring operations for a span
func (ct *CostTracer) RecordAnchoring(spanID string, ops int, strategy string) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	if span, ok := ct.spans[spanID]; ok {
		span.AnchoringOps += ops
	}
}

// computeComputeCost computes compute cost from duration
func (ct *CostTracer) computeComputeCost(span *CostSpan) float64 {
	// Cost = (duration_ms / 100) * cost_per_100ms
	return (span.ComputeTimeMs / 100.0) * ct.costModel.ComputeUSD
}

// computeStorageCost computes storage cost from bytes
func (ct *CostTracer) computeStorageCost(span *CostSpan) float64 {
	// Assume hot tier by default (Redis)
	// Cost = (bytes / 1GB) * (cost_per_GB_per_month / days_in_month / hours_in_day / ms_in_hour)
	gb := float64(span.StorageBytes) / (1024 * 1024 * 1024)
	durationHours := span.EndTime.Sub(span.StartTime).Hours()
	monthlyRate := ct.costModel.StorageHotUSD
	hourlyCost := monthlyRate / 720.0 // Approx 720 hours/month

	return gb * hourlyCost * durationHours
}

// computeNetworkCost computes network cost from bytes transferred
func (ct *CostTracer) computeNetworkCost(span *CostSpan) float64 {
	// Cost = (bytes / 1GB) * cost_per_GB
	gb := float64(span.NetworkBytes) / (1024 * 1024 * 1024)
	return gb * ct.costModel.NetworkUSD
}

// computeAnchoringCost computes anchoring cost from operations
func (ct *CostTracer) computeAnchoringCost(span *CostSpan) float64 {
	// Assume Polygon by default (cheapest non-free option)
	return float64(span.AnchoringOps) * ct.costModel.AnchoringPolygonUSD
}

// updateAggregate updates cumulative cost aggregate for tenant
func (ct *CostTracer) updateAggregate(span *CostSpan) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	key := span.TenantID
	agg, ok := ct.aggregates[key]
	if !ok {
		agg = &CostAggregate{
			TenantID:    span.TenantID,
			ModelID:     span.ModelID,
			WindowStart: time.Now().Truncate(24 * time.Hour), // Daily window
			SoftCap:     100.0, // Default $100/day soft cap
			HardCap:     200.0, // Default $200/day hard cap
		}
		ct.aggregates[key] = agg
	}

	agg.TotalCompute += span.ComputeCost
	agg.TotalStorage += span.StorageCost
	agg.TotalNetwork += span.NetworkCost
	agg.TotalAnchoring += span.AnchoringCost
	agg.GrandTotal += span.TotalCost
	agg.RequestCount++
	agg.WindowEnd = time.Now()

	// Update budget used
	agg.BudgetUsed = agg.GrandTotal / agg.HardCap
}

// recordCostMetrics records cost metrics to Prometheus
func (ct *CostTracer) recordCostMetrics(span *CostSpan) {
	ct.metrics.CostComputeUSD.WithLabelValues(span.TenantID, span.ModelID, span.TaskType).Add(span.ComputeCost)
	ct.metrics.CostStorageUSD.WithLabelValues(span.TenantID, "hot").Add(span.StorageCost)
	ct.metrics.CostNetworkUSD.WithLabelValues(span.TenantID).Add(span.NetworkCost)
	ct.metrics.CostAnchoringUSD.WithLabelValues(span.TenantID, "polygon").Add(span.AnchoringCost)

	// Update cost per trusted task (if task is trusted)
	ct.mu.RLock()
	if agg, ok := ct.aggregates[span.TenantID]; ok {
		if agg.SuccessCount > 0 {
			costPerTask := agg.GrandTotal / float64(agg.SuccessCount)
			ct.metrics.CostPerTrustedTask.WithLabelValues(span.TenantID, span.ModelID).Set(costPerTask)
		}
		ct.metrics.BudgetUsed.WithLabelValues(span.TenantID).Set(agg.BudgetUsed)
	}
	ct.mu.RUnlock()
}

// GetAggregate returns cost aggregate for a tenant
func (ct *CostTracer) GetAggregate(tenantID string) (*CostAggregate, bool) {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	agg, ok := ct.aggregates[tenantID]
	return agg, ok
}

// CheckBudget checks if tenant has exceeded budget caps
func (ct *CostTracer) CheckBudget(tenantID string) (exceeded bool, level string, remaining float64) {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	agg, ok := ct.aggregates[tenantID]
	if !ok {
		return false, "ok", 0
	}

	if agg.GrandTotal >= agg.HardCap {
		return true, "hard", 0
	}

	if agg.GrandTotal >= agg.SoftCap {
		remaining := agg.HardCap - agg.GrandTotal
		return true, "soft", remaining
	}

	remaining := agg.SoftCap - agg.GrandTotal
	return false, "ok", remaining
}

// SetBudgetCaps sets soft and hard budget caps for a tenant
func (ct *CostTracer) SetBudgetCaps(tenantID string, softCap, hardCap float64) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	agg, ok := ct.aggregates[tenantID]
	if !ok {
		agg = &CostAggregate{
			TenantID:    tenantID,
			WindowStart: time.Now().Truncate(24 * time.Hour),
		}
		ct.aggregates[tenantID] = agg
	}

	agg.SoftCap = softCap
	agg.HardCap = hardCap
	agg.BudgetUsed = agg.GrandTotal / hardCap
}

// ResetDailyWindow resets daily cost aggregates
func (ct *CostTracer) ResetDailyWindow() {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	now := time.Now()
	for tenantID, agg := range ct.aggregates {
		if now.Sub(agg.WindowStart) >= 24*time.Hour {
			// Archive old aggregate (could write to storage)
			fmt.Printf("Resetting daily window for tenant %s: total=$%.4f\n", tenantID, agg.GrandTotal)

			// Reset counters
			agg.TotalCompute = 0
			agg.TotalStorage = 0
			agg.TotalNetwork = 0
			agg.TotalAnchoring = 0
			agg.GrandTotal = 0
			agg.RequestCount = 0
			agg.SuccessCount = 0
			agg.ErrorCount = 0
			agg.BudgetUsed = 0
			agg.WindowStart = now.Truncate(24 * time.Hour)
			agg.WindowEnd = now
		}
	}
}

// GetCostBreakdown returns cost breakdown for a tenant
func (ct *CostTracer) GetCostBreakdown(tenantID string) map[string]float64 {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	agg, ok := ct.aggregates[tenantID]
	if !ok {
		return map[string]float64{}
	}

	return map[string]float64{
		"compute":   agg.TotalCompute,
		"storage":   agg.TotalStorage,
		"network":   agg.TotalNetwork,
		"anchoring": agg.TotalAnchoring,
		"total":     agg.GrandTotal,
	}
}

// ReconcileWithBill reconciles tracked costs with actual cloud bill
func (ct *CostTracer) ReconcileWithBill(actualBill float64) (float64, float64) {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	trackedTotal := 0.0
	for _, agg := range ct.aggregates {
		trackedTotal += agg.GrandTotal
	}

	diff := actualBill - trackedTotal
	diffPercent := 0.0
	if actualBill > 0 {
		diffPercent = (diff / actualBill) * 100
	}

	fmt.Printf("Cost reconciliation: tracked=$%.2f, actual=$%.2f, diff=$%.2f (%.1f%%)\n",
		trackedTotal, actualBill, diff, diffPercent)

	return diff, diffPercent
}

// GetMetrics returns cost metrics
func (ct *CostTracer) GetMetrics() *CostMetrics {
	return ct.metrics
}

// --- Cost Budget Policy (CRD Schema) ---

// CostBudgetAction defines actions when budget thresholds are exceeded
type CostBudgetAction struct {
	Threshold float64 // Budget usage threshold [0, 1]
	Action    string  // "degrade", "throttle", "queue", "deny"
	Message   string  // Customer-visible message
}

// CostBudgetPolicySpec defines cost budget policy
type CostBudgetPolicySpec struct {
	TenantID      string
	SoftCapUSD    float64 // Soft cap triggers degradation
	HardCapUSD    float64 // Hard cap triggers denial
	WindowDuration string // "daily", "weekly", "monthly"
	Actions       []CostBudgetAction
	Rollover      bool   // Allow unused budget to roll over
}

// DefaultCostBudgetPolicy returns a default cost budget policy
func DefaultCostBudgetPolicy(tenantID string) *CostBudgetPolicySpec {
	return &CostBudgetPolicySpec{
		TenantID:       tenantID,
		SoftCapUSD:     100.0,
		HardCapUSD:     200.0,
		WindowDuration: "daily",
		Actions: []CostBudgetAction{
			{
				Threshold: 0.7, // 70% of hard cap
				Action:    "degrade",
				Message:   "Budget usage at 70%, switching to cost-optimized mode",
			},
			{
				Threshold: 0.9, // 90% of hard cap
				Action:    "throttle",
				Message:   "Budget usage at 90%, request rate throttled",
			},
			{
				Threshold: 1.0, // 100% of hard cap
				Action:    "deny",
				Message:   "Daily budget exceeded, please contact support",
			},
		},
		Rollover: false,
	}
}

// ApplyAction applies cost budget action based on usage
func (cb *CostBudgetPolicySpec) ApplyAction(usagePercent float64) *CostBudgetAction {
	// Find highest threshold that's exceeded
	var applicableAction *CostBudgetAction
	for i := range cb.Actions {
		if usagePercent >= cb.Actions[i].Threshold {
			applicableAction = &cb.Actions[i]
		}
	}
	return applicableAction
}
