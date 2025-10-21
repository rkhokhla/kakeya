package controllers

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// Phase 8 WP5: Adaptive canary with multi-objective gates and policy simulator

// AdaptiveCanaryController manages adaptive canary deployments
type AdaptiveCanaryController struct {
	mu               sync.RWMutex
	canaryDeployments map[string]*CanaryDeployment
	policySimulator  *PolicySimulator
	healthMonitor    *HealthMonitor
	rollbackManager  *RollbackManager
	metrics          *CanaryMetrics
}

// CanaryDeployment represents a canary deployment
type CanaryDeployment struct {
	ID               string
	Name             string
	ControlVersion   string
	CanaryVersion    string
	CurrentStep      int
	TotalSteps       int
	TrafficPercent   float64
	StepSizeStrategy string // "fixed", "adaptive"
	Status           string // "pending", "running", "completed", "rolled_back"
	StartedAt        time.Time
	CompletedAt      time.Time
	Steps            []*CanaryStep
	HealthGates      *MultiObjectiveGates
}

// CanaryStep represents a single canary step
type CanaryStep struct {
	StepNum        int
	TrafficPercent float64
	Duration       time.Duration
	StartTime      time.Time
	EndTime        time.Time
	Status         string // "pending", "running", "passed", "failed"
	HealthCheck    *HealthCheckResult
}

// MultiObjectiveGates defines health gates across multiple objectives
type MultiObjectiveGates struct {
	LatencyGate      *LatencyGate
	ErrorBudgetGate  *ErrorBudgetGate
	ContainmentGate  *ContainmentGate
	CostGate         *CostGate
}

// LatencyGate defines latency SLO gates
type LatencyGate struct {
	P95ThresholdMs   float64
	MaxDegradationPct float64 // Max % degradation vs control
}

// ErrorBudgetGate defines error budget gates
type ErrorBudgetGate struct {
	EscalationRateThreshold float64 // ≤2%
	BurnRateThreshold       float64 // Max error budget burn rate
}

// ContainmentGate defines hallucination containment gates
type ContainmentGate struct {
	MinContainmentRate float64 // ≥98%
	MaxEscapedPerHour  int
}

// CostGate defines cost gates
type CostGate struct {
	MaxCostIncreasePercent float64 // Max % cost increase vs control
	MaxCostPerTrustedTask  float64 // Max $ per trusted task
}

// HealthCheckResult represents health check outcome
type HealthCheckResult struct {
	Timestamp         time.Time
	LatencyP95Ms      float64
	EscalationRate    float64
	ContainmentRate   float64
	CostPerTaskUSD    float64
	ErrorBudgetBurn   float64
	Passed            bool
	FailedGates       []string
	Recommendation    string // "continue", "pause", "rollback"
}

// PolicySimulator simulates policy changes
type PolicySimulator struct {
	mu              sync.RWMutex
	historicalTraces []*TraceSnapshot
	simulations     []*SimulationResult
}

// TraceSnapshot represents historical system state
type TraceSnapshot struct {
	Timestamp       time.Time
	LatencyP95Ms    float64
	EscalationRate  float64
	ContainmentRate float64
	CostPerTaskUSD  float64
	RequestRate     float64
}

// SimulationResult represents simulation outcome
type SimulationResult struct {
	PolicyID          string
	PolicyDescription string
	SimulatedAt       time.Time
	Outcomes          []*SimulatedOutcome
	PredictedImpact   *ImpactAnalysis
	Recommendation    string // "approve", "review", "reject"
}

// SimulatedOutcome represents simulated system state
type SimulatedOutcome struct {
	Timestamp       time.Time
	LatencyP95Ms    float64
	EscalationRate  float64
	ContainmentRate float64
	CostPerTaskUSD  float64
}

// ImpactAnalysis represents predicted impact of policy change
type ImpactAnalysis struct {
	LatencyDeltaPercent      float64
	ContainmentDeltaPercent  float64
	CostDeltaPercent         float64
	RiskLevel                string // "low", "medium", "high"
	BreachedSLOs             []string
}

// HealthMonitor monitors canary health
type HealthMonitor struct {
	mu           sync.RWMutex
	metricsFetcher MetricsFetcher
	checkInterval time.Duration
}

// MetricsFetcher fetches metrics from Prometheus
type MetricsFetcher interface {
	FetchLatencyP95(ctx context.Context, version string) (float64, error)
	FetchEscalationRate(ctx context.Context, version string) (float64, error)
	FetchContainmentRate(ctx context.Context, version string) (float64, error)
	FetchCostPerTask(ctx context.Context, version string) (float64, error)
}

// RollbackManager manages rollbacks
type RollbackManager struct {
	mu               sync.RWMutex
	rollbackHistory  []*RollbackEvent
	autoRollbackEnabled bool
}

// RollbackEvent represents a rollback event
type RollbackEvent struct {
	DeploymentID  string
	Timestamp     time.Time
	Reason        string
	FailedGates   []string
	RollbackTime  time.Duration
}

// CanaryMetrics tracks canary metrics
type CanaryMetrics struct {
	mu                  sync.RWMutex
	TotalCanaries       int64
	CompletedCanaries   int64
	RolledBackCanaries  int64
	AvgCanaryDuration   time.Duration
	AvgStepsCompleted   float64
	AutoRollbacks       int64
}

// NewAdaptiveCanaryController creates a new adaptive canary controller
func NewAdaptiveCanaryController(metricsFetcher MetricsFetcher) *AdaptiveCanaryController {
	return &AdaptiveCanaryController{
		canaryDeployments: make(map[string]*CanaryDeployment),
		policySimulator:   NewPolicySimulator(),
		healthMonitor:     NewHealthMonitor(metricsFetcher),
		rollbackManager:   NewRollbackManager(true), // Auto-rollback enabled
		metrics:           &CanaryMetrics{},
	}
}

// NewPolicySimulator creates a new policy simulator
func NewPolicySimulator() *PolicySimulator {
	return &PolicySimulator{
		historicalTraces: []*TraceSnapshot{},
		simulations:      []*SimulationResult{},
	}
}

// NewHealthMonitor creates a new health monitor
func NewHealthMonitor(metricsFetcher MetricsFetcher) *HealthMonitor {
	return &HealthMonitor{
		metricsFetcher: metricsFetcher,
		checkInterval:  10 * time.Second,
	}
}

// NewRollbackManager creates a new rollback manager
func NewRollbackManager(autoRollbackEnabled bool) *RollbackManager {
	return &RollbackManager{
		rollbackHistory:     []*RollbackEvent{},
		autoRollbackEnabled: autoRollbackEnabled,
	}
}

// StartCanary starts an adaptive canary deployment
func (acc *AdaptiveCanaryController) StartCanary(ctx context.Context, name, controlVersion, canaryVersion string, gates *MultiObjectiveGates) (*CanaryDeployment, error) {
	deploymentID := fmt.Sprintf("canary-%s-%d", name, time.Now().Unix())

	deployment := &CanaryDeployment{
		ID:               deploymentID,
		Name:             name,
		ControlVersion:   controlVersion,
		CanaryVersion:    canaryVersion,
		CurrentStep:      0,
		TotalSteps:       5, // Start with 5 steps
		TrafficPercent:   0.0,
		StepSizeStrategy: "adaptive",
		Status:           "running",
		StartedAt:        time.Now(),
		Steps:            []*CanaryStep{},
		HealthGates:      gates,
	}

	// Initialize steps
	deployment.Steps = acc.generateAdaptiveSteps(deployment)

	acc.mu.Lock()
	acc.canaryDeployments[deploymentID] = deployment
	acc.mu.Unlock()

	acc.metrics.mu.Lock()
	acc.metrics.TotalCanaries++
	acc.metrics.mu.Unlock()

	fmt.Printf("Started canary deployment: id=%s, control=%s, canary=%s\n",
		deploymentID, controlVersion, canaryVersion)

	// Start canary progression
	go acc.progressCanary(ctx, deployment)

	return deployment, nil
}

// generateAdaptiveSteps generates adaptive canary steps
func (acc *AdaptiveCanaryController) generateAdaptiveSteps(deployment *CanaryDeployment) []*CanaryStep {
	// Adaptive step sizing: start small, accelerate if healthy
	// Step 0: 5% traffic, 10m
	// Step 1: 10% traffic, 10m
	// Step 2: 25% traffic, 15m
	// Step 3: 50% traffic, 20m
	// Step 4: 100% traffic, 30m

	steps := []*CanaryStep{
		{StepNum: 0, TrafficPercent: 5.0, Duration: 10 * time.Minute, Status: "pending"},
		{StepNum: 1, TrafficPercent: 10.0, Duration: 10 * time.Minute, Status: "pending"},
		{StepNum: 2, TrafficPercent: 25.0, Duration: 15 * time.Minute, Status: "pending"},
		{StepNum: 3, TrafficPercent: 50.0, Duration: 20 * time.Minute, Status: "pending"},
		{StepNum: 4, TrafficPercent: 100.0, Duration: 30 * time.Minute, Status: "pending"},
	}

	return steps
}

// progressCanary progresses canary through steps
func (acc *AdaptiveCanaryController) progressCanary(ctx context.Context, deployment *CanaryDeployment) {
	for stepNum, step := range deployment.Steps {
		deployment.CurrentStep = stepNum
		step.Status = "running"
		step.StartTime = time.Now()

		fmt.Printf("Canary step %d: traffic=%.0f%%, duration=%v\n",
			stepNum, step.TrafficPercent, step.Duration)

		// Update traffic routing
		if err := acc.updateTrafficRouting(deployment, step.TrafficPercent); err != nil {
			fmt.Printf("Failed to update traffic routing: %v\n", err)
			acc.rollback(ctx, deployment, "traffic_routing_failed", []string{})
			return
		}

		// Wait for step duration with periodic health checks
		ticker := time.NewTicker(acc.healthMonitor.checkInterval)
		defer ticker.Stop()

		stepDeadline := time.Now().Add(step.Duration)

		for time.Now().Before(stepDeadline) {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				// Check health
				healthCheck, err := acc.healthMonitor.CheckHealth(ctx, deployment)
				if err != nil {
					fmt.Printf("Health check failed: %v\n", err)
					continue
				}

				step.HealthCheck = healthCheck

				if !healthCheck.Passed {
					fmt.Printf("Health check failed: gates=%v\n", healthCheck.FailedGates)

					if healthCheck.Recommendation == "rollback" {
						acc.rollback(ctx, deployment, "health_gates_failed", healthCheck.FailedGates)
						return
					}
				}
			}
		}

		step.Status = "passed"
		step.EndTime = time.Now()

		fmt.Printf("Canary step %d passed\n", stepNum)

		// Adaptive step sizing: if health is excellent, accelerate next step
		if stepNum < len(deployment.Steps)-1 && step.HealthCheck != nil && acc.isHealthExcellent(step.HealthCheck) {
			nextStep := deployment.Steps[stepNum+1]
			nextStep.Duration = nextStep.Duration * 3 / 4 // Reduce duration by 25%
			fmt.Printf("Accelerating next step: new_duration=%v\n", nextStep.Duration)
		}
	}

	// All steps passed
	deployment.Status = "completed"
	deployment.CompletedAt = time.Now()

	acc.metrics.mu.Lock()
	acc.metrics.CompletedCanaries++
	duration := time.Since(deployment.StartedAt)
	acc.metrics.AvgCanaryDuration = (acc.metrics.AvgCanaryDuration*time.Duration(acc.metrics.CompletedCanaries-1) + duration) / time.Duration(acc.metrics.CompletedCanaries)
	acc.metrics.mu.Unlock()

	fmt.Printf("Canary deployment completed: id=%s, duration=%v\n", deployment.ID, duration)
}

// CheckHealth checks canary health against multi-objective gates
func (hm *HealthMonitor) CheckHealth(ctx context.Context, deployment *CanaryDeployment) (*HealthCheckResult, error) {
	// Fetch metrics for canary version
	latencyP95, err := hm.metricsFetcher.FetchLatencyP95(ctx, deployment.CanaryVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch latency: %w", err)
	}

	escalationRate, err := hm.metricsFetcher.FetchEscalationRate(ctx, deployment.CanaryVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch escalation rate: %w", err)
	}

	containmentRate, err := hm.metricsFetcher.FetchContainmentRate(ctx, deployment.CanaryVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch containment rate: %w", err)
	}

	costPerTask, err := hm.metricsFetcher.FetchCostPerTask(ctx, deployment.CanaryVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch cost per task: %w", err)
	}

	result := &HealthCheckResult{
		Timestamp:       time.Now(),
		LatencyP95Ms:    latencyP95,
		EscalationRate:  escalationRate,
		ContainmentRate: containmentRate,
		CostPerTaskUSD:  costPerTask,
		Passed:          true,
		FailedGates:     []string{},
	}

	gates := deployment.HealthGates

	// Check latency gate
	if latencyP95 > gates.LatencyGate.P95ThresholdMs {
		result.Passed = false
		result.FailedGates = append(result.FailedGates, fmt.Sprintf("latency_p95: %.1fms > %.1fms", latencyP95, gates.LatencyGate.P95ThresholdMs))
	}

	// Check error budget gate
	if escalationRate > gates.ErrorBudgetGate.EscalationRateThreshold {
		result.Passed = false
		result.FailedGates = append(result.FailedGates, fmt.Sprintf("escalation_rate: %.2f%% > %.2f%%", escalationRate*100, gates.ErrorBudgetGate.EscalationRateThreshold*100))
	}

	// Check containment gate
	if containmentRate < gates.ContainmentGate.MinContainmentRate {
		result.Passed = false
		result.FailedGates = append(result.FailedGates, fmt.Sprintf("containment_rate: %.2f%% < %.2f%%", containmentRate*100, gates.ContainmentGate.MinContainmentRate*100))
	}

	// Check cost gate
	if costPerTask > gates.CostGate.MaxCostPerTrustedTask {
		result.Passed = false
		result.FailedGates = append(result.FailedGates, fmt.Sprintf("cost_per_task: $%.4f > $%.4f", costPerTask, gates.CostGate.MaxCostPerTrustedTask))
	}

	// Determine recommendation
	if !result.Passed {
		if len(result.FailedGates) >= 2 {
			result.Recommendation = "rollback"
		} else {
			result.Recommendation = "pause"
		}
	} else {
		result.Recommendation = "continue"
	}

	return result, nil
}

// isHealthExcellent checks if health is excellent (all gates passed with margin)
func (acc *AdaptiveCanaryController) isHealthExcellent(healthCheck *HealthCheckResult) bool {
	// Health is excellent if all metrics beat SLOs by >20%
	return healthCheck.Passed &&
		healthCheck.LatencyP95Ms < 160 && // <160ms (20% better than 200ms SLO)
		healthCheck.EscalationRate < 0.016 && // <1.6% (20% better than 2% SLO)
		healthCheck.ContainmentRate > 0.984 // >98.4% (slightly better than 98% SLO)
}

// rollback rolls back canary deployment
func (acc *AdaptiveCanaryController) rollback(ctx context.Context, deployment *CanaryDeployment, reason string, failedGates []string) {
	rollbackStart := time.Now()

	fmt.Printf("Rolling back canary: id=%s, reason=%s, failed_gates=%v\n",
		deployment.ID, reason, failedGates)

	// Revert traffic to 0%
	if err := acc.updateTrafficRouting(deployment, 0.0); err != nil {
		fmt.Printf("Failed to revert traffic: %v\n", err)
	}

	deployment.Status = "rolled_back"
	deployment.CompletedAt = time.Now()

	rollbackDuration := time.Since(rollbackStart)

	// Record rollback event
	acc.rollbackManager.mu.Lock()
	acc.rollbackManager.rollbackHistory = append(acc.rollbackManager.rollbackHistory, &RollbackEvent{
		DeploymentID: deployment.ID,
		Timestamp:    rollbackStart,
		Reason:       reason,
		FailedGates:  failedGates,
		RollbackTime: rollbackDuration,
	})
	acc.rollbackManager.mu.Unlock()

	acc.metrics.mu.Lock()
	acc.metrics.RolledBackCanaries++
	acc.metrics.AutoRollbacks++
	acc.metrics.mu.Unlock()

	fmt.Printf("Rollback completed: duration=%v\n", rollbackDuration)
}

// updateTrafficRouting updates traffic routing percentage
func (acc *AdaptiveCanaryController) updateTrafficRouting(deployment *CanaryDeployment, trafficPercent float64) error {
	// Placeholder - in production, update Kubernetes Ingress/Service weights
	deployment.TrafficPercent = trafficPercent
	fmt.Printf("Updated traffic routing: canary=%s, traffic=%.1f%%\n", deployment.CanaryVersion, trafficPercent)
	return nil
}

// SimulatePolicy simulates policy change with historical traces
func (ps *PolicySimulator) SimulatePolicy(ctx context.Context, policyID, policyDescription string) (*SimulationResult, error) {
	fmt.Printf("Simulating policy: id=%s, description=%s\n", policyID, policyDescription)

	// Load historical traces (last 7 days)
	traces := ps.loadHistoricalTraces(7 * 24 * time.Hour)

	if len(traces) < 100 {
		return nil, fmt.Errorf("insufficient historical traces: %d (need ≥100)", len(traces))
	}

	// Simulate policy application on each trace
	outcomes := []*SimulatedOutcome{}
	for _, trace := range traces {
		outcome := ps.simulateTrace(trace, policyDescription)
		outcomes = append(outcomes, outcome)
	}

	// Analyze impact
	impact := ps.analyzeImpact(traces, outcomes)

	// Determine recommendation
	recommendation := "approve"
	if impact.RiskLevel == "high" || len(impact.BreachedSLOs) > 0 {
		recommendation = "reject"
	} else if impact.RiskLevel == "medium" {
		recommendation = "review"
	}

	result := &SimulationResult{
		PolicyID:          policyID,
		PolicyDescription: policyDescription,
		SimulatedAt:       time.Now(),
		Outcomes:          outcomes,
		PredictedImpact:   impact,
		Recommendation:    recommendation,
	}

	ps.mu.Lock()
	ps.simulations = append(ps.simulations, result)
	ps.mu.Unlock()

	fmt.Printf("Simulation completed: recommendation=%s, risk=%s\n",
		recommendation, impact.RiskLevel)

	return result, nil
}

// loadHistoricalTraces loads historical traces
func (ps *PolicySimulator) loadHistoricalTraces(duration time.Duration) []*TraceSnapshot {
	// Placeholder - in production, query Prometheus for historical metrics

	traces := []*TraceSnapshot{}
	endTime := time.Now()
	startTime := endTime.Add(-duration)

	for t := startTime; t.Before(endTime); t = t.Add(5 * time.Minute) {
		traces = append(traces, &TraceSnapshot{
			Timestamp:       t,
			LatencyP95Ms:    180.0 + 10*math.Sin(float64(len(traces))/12.0),
			EscalationRate:  0.018 + 0.002*math.Sin(float64(len(traces))/24.0),
			ContainmentRate: 0.982,
			CostPerTaskUSD:  0.0012,
			RequestRate:     100.0,
		})
	}

	return traces
}

// simulateTrace simulates policy on a single trace
func (ps *PolicySimulator) simulateTrace(trace *TraceSnapshot, policyDescription string) *SimulatedOutcome {
	// Placeholder - in production, apply policy transformations

	// Example: Ensemble optimization policy reduces latency by 5%, increases cost by 2%
	outcome := &SimulatedOutcome{
		Timestamp:       trace.Timestamp,
		LatencyP95Ms:    trace.LatencyP95Ms * 0.95,
		EscalationRate:  trace.EscalationRate,
		ContainmentRate: trace.ContainmentRate,
		CostPerTaskUSD:  trace.CostPerTaskUSD * 1.02,
	}

	return outcome
}

// analyzeImpact analyzes impact of policy change
func (ps *PolicySimulator) analyzeImpact(baseline, simulated []*TraceSnapshot) *ImpactAnalysis {
	// Compute average deltas
	avgBaselineLatency := 0.0
	avgSimulatedLatency := 0.0
	avgBaselineContainment := 0.0
	avgSimulatedContainment := 0.0
	avgBaselineCost := 0.0
	avgSimulatedCost := 0.0

	for i := range baseline {
		avgBaselineLatency += baseline[i].LatencyP95Ms
		avgBaselineContainment += baseline[i].ContainmentRate
		avgBaselineCost += baseline[i].CostPerTaskUSD
	}

	for _, outcome := range simulated {
		avgSimulatedLatency += outcome.LatencyP95Ms
		avgSimulatedContainment += outcome.ContainmentRate
		avgSimulatedCost += outcome.CostPerTaskUSD
	}

	n := float64(len(baseline))
	avgBaselineLatency /= n
	avgSimulatedLatency /= n
	avgBaselineContainment /= n
	avgSimulatedContainment /= n
	avgBaselineCost /= n
	avgSimulatedCost /= n

	latencyDelta := ((avgSimulatedLatency - avgBaselineLatency) / avgBaselineLatency) * 100
	containmentDelta := ((avgSimulatedContainment - avgBaselineContainment) / avgBaselineContainment) * 100
	costDelta := ((avgSimulatedCost - avgBaselineCost) / avgBaselineCost) * 100

	// Determine risk level
	riskLevel := "low"
	breachedSLOs := []string{}

	if avgSimulatedLatency > 200 {
		breachedSLOs = append(breachedSLOs, "latency_p95 > 200ms")
		riskLevel = "high"
	}

	if avgSimulatedContainment < 0.98 {
		breachedSLOs = append(breachedSLOs, "containment_rate < 98%")
		riskLevel = "high"
	}

	if costDelta > 10 {
		breachedSLOs = append(breachedSLOs, "cost_increase > 10%")
		if riskLevel != "high" {
			riskLevel = "medium"
		}
	}

	return &ImpactAnalysis{
		LatencyDeltaPercent:     latencyDelta,
		ContainmentDeltaPercent: containmentDelta,
		CostDeltaPercent:        costDelta,
		RiskLevel:               riskLevel,
		BreachedSLOs:            breachedSLOs,
	}
}

// GetMetrics returns canary metrics
func (acc *AdaptiveCanaryController) GetMetrics() CanaryMetrics {
	acc.metrics.mu.RLock()
	defer acc.metrics.mu.RUnlock()
	return *acc.metrics
}

// GetRollbackHistory returns rollback history
func (rm *RollbackManager) GetRollbackHistory() []*RollbackEvent {
	rm.mu.RLock()
	defer rm.mu.RUnlock()
	return rm.rollbackHistory
}
