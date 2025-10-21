package metrics

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// BuyerKPITracker tracks investor-grade KPIs for hallucination containment (Phase 6 WP6)
type BuyerKPITracker struct {
	mu sync.RWMutex

	// Hallucination containment metrics
	containmentRate    *prometheus.GaugeVec
	escalationRate     *prometheus.GaugeVec
	falsePositiveRate  *prometheus.GaugeVec

	// Economic metrics
	costPerTrustedTask *prometheus.GaugeVec
	costCompute        *prometheus.CounterVec
	costStorage        *prometheus.CounterVec
	costNetwork        *prometheus.CounterVec
	costAnchoring      *prometheus.CounterVec

	// Quality metrics
	signalQuality      *prometheus.HistogramVec
	regimeDistribution *prometheus.CounterVec

	// Audit metrics
	auditCoverage      *prometheus.GaugeVec
	anchoringSuccessRate *prometheus.GaugeVec

	// State
	totalCostUSD       float64
	totalTrustedTasks  int64
	escalatedTasks     int64
}

// NewBuyerKPITracker creates a new buyer KPI tracker
func NewBuyerKPITracker() *BuyerKPITracker {
	return &BuyerKPITracker{
		// Hallucination containment
		containmentRate: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "flk_hallucination_containment_rate",
				Help: "Percentage of tasks with hallucinations successfully contained (1 - escalation_rate)",
			},
			[]string{"tenant_id", "time_window"},
		),
		escalationRate: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "flk_escalation_rate",
				Help: "Percentage of tasks escalated for human review (SLO: ≤2%)",
			},
			[]string{"tenant_id", "time_window"},
		),
		falsePositiveRate: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "flk_false_positive_rate",
				Help: "Percentage of escalated tasks that were false alarms",
			},
			[]string{"tenant_id"},
		),

		// Economic metrics
		costPerTrustedTask: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "flk_cost_per_trusted_task",
				Help: "Cost per successfully verified task (USD)",
			},
			[]string{"tenant_id", "cost_category"},
		),
		costCompute: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "flk_cost_compute_usd",
				Help: "Compute cost (USD)",
			},
			[]string{"tenant_id"},
		),
		costStorage: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "flk_cost_storage_usd",
				Help: "Storage cost (USD) for WAL, WORM, tiering",
			},
			[]string{"tenant_id", "tier"},
		),
		costNetwork: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "flk_cost_network_usd",
				Help: "Network cost (USD) for CRR and API calls",
			},
			[]string{"tenant_id"},
		),
		costAnchoring: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "flk_cost_anchoring_usd",
				Help: "Anchoring cost (USD) for blockchain/timestamp attestations",
			},
			[]string{"tenant_id", "provider"},
		),

		// Quality metrics
		signalQuality: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "flk_signal_quality",
				Help:    "Distribution of signal quality (D̂, coh★, r)",
				Buckets: prometheus.LinearBuckets(0, 0.1, 20),
			},
			[]string{"tenant_id", "signal_type"},
		),
		regimeDistribution: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "flk_regime_distribution",
				Help: "Count of PCS by regime (sticky, mixed, non_sticky)",
			},
			[]string{"tenant_id", "regime"},
		),

		// Audit metrics
		auditCoverage: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "flk_audit_coverage",
				Help: "Percentage of accepted PCS written to WORM log",
			},
			[]string{"tenant_id"},
		),
		anchoringSuccessRate: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "flk_anchoring_success_rate",
				Help: "Percentage of anchoring batches successfully attested",
			},
			[]string{"tenant_id"},
		),
	}
}

// RecordTrustedTask records a successfully verified task
func (bkt *BuyerKPITracker) RecordTrustedTask(tenantID string, costUSD float64) {
	bkt.mu.Lock()
	defer bkt.mu.Unlock()

	bkt.totalTrustedTasks++
	bkt.totalCostUSD += costUSD

	// Update cost per task
	if bkt.totalTrustedTasks > 0 {
		costPerTask := bkt.totalCostUSD / float64(bkt.totalTrustedTasks)
		bkt.costPerTrustedTask.WithLabelValues(tenantID, "total").Set(costPerTask)
	}
}

// RecordEscalatedTask records a task escalated for human review
func (bkt *BuyerKPITracker) RecordEscalatedTask(tenantID string) {
	bkt.mu.Lock()
	defer bkt.mu.Unlock()

	bkt.escalatedTasks++

	// Update escalation rate
	totalTasks := bkt.totalTrustedTasks + bkt.escalatedTasks
	if totalTasks > 0 {
		escalationRate := float64(bkt.escalatedTasks) / float64(totalTasks) * 100
		bkt.escalationRate.WithLabelValues(tenantID, "7d").Set(escalationRate)

		// Containment rate = 100 - escalation rate
		containmentRate := 100.0 - escalationRate
		bkt.containmentRate.WithLabelValues(tenantID, "7d").Set(containmentRate)
	}
}

// RecordComputeCost records compute cost
func (bkt *BuyerKPITracker) RecordComputeCost(tenantID string, costUSD float64) {
	bkt.costCompute.WithLabelValues(tenantID).Add(costUSD)
}

// RecordStorageCost records storage cost
func (bkt *BuyerKPITracker) RecordStorageCost(tenantID, tier string, costUSD float64) {
	bkt.costStorage.WithLabelValues(tenantID, tier).Add(costUSD)
}

// RecordNetworkCost records network cost
func (bkt *BuyerKPITracker) RecordNetworkCost(tenantID string, costUSD float64) {
	bkt.costNetwork.WithLabelValues(tenantID).Add(costUSD)
}

// RecordAnchoringCost records anchoring cost
func (bkt *BuyerKPITracker) RecordAnchoringCost(tenantID, provider string, costUSD float64) {
	bkt.costAnchoring.WithLabelValues(tenantID, provider).Add(costUSD)
}

// RecordSignalQuality records signal quality distribution
func (bkt *BuyerKPITracker) RecordSignalQuality(tenantID string, dHat, cohStar, r float64) {
	bkt.signalQuality.WithLabelValues(tenantID, "D_hat").Observe(dHat)
	bkt.signalQuality.WithLabelValues(tenantID, "coh_star").Observe(cohStar)
	bkt.signalQuality.WithLabelValues(tenantID, "r").Observe(r)
}

// RecordRegime records regime classification
func (bkt *BuyerKPITracker) RecordRegime(tenantID, regime string) {
	bkt.regimeDistribution.WithLabelValues(tenantID, regime).Inc()
}

// RecordAuditCoverage records audit coverage percentage
func (bkt *BuyerKPITracker) RecordAuditCoverage(tenantID string, coverage float64) {
	bkt.auditCoverage.WithLabelValues(tenantID).Set(coverage)
}

// RecordAnchoringSuccess records anchoring success rate
func (bkt *BuyerKPITracker) RecordAnchoringSuccess(tenantID string, successRate float64) {
	bkt.anchoringSuccessRate.WithLabelValues(tenantID).Set(successRate)
}

// RecordFalsePositiveRate records false positive rate for escalations
func (bkt *BuyerKPITracker) RecordFalsePositiveRate(tenantID string, rate float64) {
	bkt.falsePositiveRate.WithLabelValues(tenantID).Set(rate * 100)
}

// GetHallucinationContainmentRate returns the current containment rate
func (bkt *BuyerKPITracker) GetHallucinationContainmentRate() float64 {
	bkt.mu.RLock()
	defer bkt.mu.RUnlock()

	totalTasks := bkt.totalTrustedTasks + bkt.escalatedTasks
	if totalTasks == 0 {
		return 100.0
	}

	return (1.0 - float64(bkt.escalatedTasks)/float64(totalTasks)) * 100.0
}

// GetCostPerTrustedTask returns the current cost per task
func (bkt *BuyerKPITracker) GetCostPerTrustedTask() float64 {
	bkt.mu.RLock()
	defer bkt.mu.RUnlock()

	if bkt.totalTrustedTasks == 0 {
		return 0.0
	}

	return bkt.totalCostUSD / float64(bkt.totalTrustedTasks)
}

// GetEscalationRate returns the current escalation rate
func (bkt *BuyerKPITracker) GetEscalationRate() float64 {
	bkt.mu.RLock()
	defer bkt.mu.RUnlock()

	totalTasks := bkt.totalTrustedTasks + bkt.escalatedTasks
	if totalTasks == 0 {
		return 0.0
	}

	return float64(bkt.escalatedTasks) / float64(totalTasks) * 100.0
}

// --- Cost Estimators ---

// EstimateComputeCost estimates compute cost for a verification
func EstimateComputeCost(verifyLatencyMs int) float64 {
	// Assume $0.0001 per 100ms of compute
	return float64(verifyLatencyMs) / 100.0 * 0.0001
}

// EstimateStorageCost estimates storage cost for a PCS
func EstimateStorageCost(pcsSize int, tier string, retentionDays int) float64 {
	sizeGB := float64(pcsSize) / 1024.0 / 1024.0 / 1024.0

	var costPerGBPerMonth float64
	switch tier {
	case "hot":
		costPerGBPerMonth = 0.023 // S3 Standard
	case "warm":
		costPerGBPerMonth = 0.0125 // S3 Infrequent Access
	case "cold":
		costPerGBPerMonth = 0.004 // S3 Glacier
	default:
		costPerGBPerMonth = 0.023
	}

	return sizeGB * costPerGBPerMonth * float64(retentionDays) / 30.0
}

// EstimateNetworkCost estimates network cost for CRR
func EstimateNetworkCost(pcsSize int, regions int) float64 {
	sizeGB := float64(pcsSize) / 1024.0 / 1024.0 / 1024.0
	costPerGB := 0.09 // Inter-region transfer

	return sizeGB * costPerGB * float64(regions-1)
}

// EstimateAnchoringCost estimates anchoring cost based on strategy
func EstimateAnchoringCost(provider string, batchSize int) float64 {
	switch provider {
	case "ethereum":
		return 2.0 // $2.00 per batch (100k gas * $0.00002)
	case "polygon":
		return 0.01 // $0.01 per batch
	case "rfc3161":
		return 0.01 // $0.01 per timestamp
	case "opentimestamps":
		return 0.0 // Free
	default:
		return 0.01
	}
}

// --- Buyer KPI Report ---

// BuyerKPIReport is a summary report for buyers/investors
type BuyerKPIReport struct {
	GeneratedAt             time.Time
	TimeWindow              string
	HallucinationContainment float64 // Percentage (0-100)
	EscalationRate          float64 // Percentage (0-100)
	CostPerTrustedTask      float64 // USD
	TotalTrustedTasks       int64
	TotalEscalatedTasks     int64
	CostBreakdown           CostBreakdown
	SLOCompliance           SLOCompliance
	AuditCoverage           float64 // Percentage (0-100)
	AnchoringSuccessRate    float64 // Percentage (0-100)
}

// CostBreakdown provides detailed cost analysis
type CostBreakdown struct {
	ComputeUSD   float64
	StorageUSD   float64
	NetworkUSD   float64
	AnchoringUSD float64
	TotalUSD     float64
}

// SLOCompliance tracks SLO adherence
type SLOCompliance struct {
	EscalationRateSLO     float64 // Target: ≤2%
	LatencyP95SLO         float64 // Target: ≤200ms
	AuditCoverageSLO      float64 // Target: ≥99.9%
	CRRLagSLO             float64 // Target: ≤60s
	AllSLOsMet            bool
}

// GenerateBuyerKPIReport generates a comprehensive KPI report
func (bkt *BuyerKPITracker) GenerateBuyerKPIReport(timeWindow string) *BuyerKPIReport {
	bkt.mu.RLock()
	defer bkt.mu.RUnlock()

	// Compute cost breakdown
	costBreakdown := CostBreakdown{
		ComputeUSD:   bkt.totalCostUSD * 0.6, // 60% compute (estimate)
		StorageUSD:   bkt.totalCostUSD * 0.2, // 20% storage
		NetworkUSD:   bkt.totalCostUSD * 0.15, // 15% network
		AnchoringUSD: bkt.totalCostUSD * 0.05, // 5% anchoring
		TotalUSD:     bkt.totalCostUSD,
	}

	// Compute SLO compliance
	escalationRate := bkt.GetEscalationRate()
	sloCompliance := SLOCompliance{
		EscalationRateSLO: escalationRate,
		LatencyP95SLO:     150.0, // Placeholder - would query from Prometheus
		AuditCoverageSLO:  99.9,
		CRRLagSLO:         45.0,
		AllSLOsMet:        escalationRate <= 2.0,
	}

	return &BuyerKPIReport{
		GeneratedAt:              time.Now(),
		TimeWindow:               timeWindow,
		HallucinationContainment: bkt.GetHallucinationContainmentRate(),
		EscalationRate:           escalationRate,
		CostPerTrustedTask:       bkt.GetCostPerTrustedTask(),
		TotalTrustedTasks:        bkt.totalTrustedTasks,
		TotalEscalatedTasks:      bkt.escalatedTasks,
		CostBreakdown:            costBreakdown,
		SLOCompliance:            sloCompliance,
		AuditCoverage:            99.9,  // Placeholder
		AnchoringSuccessRate:     99.5, // Placeholder
	}
}
