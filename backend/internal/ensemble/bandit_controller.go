package ensemble

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// BanditController optimizes ensemble configuration per tenant using multi-armed bandits
// Phase 9 WP2: Thompson sampling/UCB for N-of-M and strategy weight tuning
type BanditController struct {
	mu sync.RWMutex

	// Per-tenant bandit state
	tenantBandits map[string]*TenantBandit

	// Reward function configuration
	containmentWeight float64 // Weight for containment improvement
	agreementWeight   float64 // Weight for agreement rate
	latencyPenalty    float64 // Penalty per ms over budget
	costPenalty       float64 // Penalty per $ over budget

	// Constraints
	maxLatencyMs float64 // e.g., 120ms p95
	maxCostDelta float64 // e.g., +7% over Phase 8 baseline

	// Exploration strategy
	explorationStrategy string  // "thompson_sampling", "ucb"
	ucbExplorationParam float64 // c parameter for UCB (e.g., 2.0)

	metrics *BanditMetrics
}

// TenantBandit tracks bandit state for a single tenant
type TenantBandit struct {
	TenantID string

	// Action space: combinations of (N, M, strategy_weights)
	// Example arms: (2,3), (3,4), (3,3) with different strategy weights
	arms []EnsembleArm

	// Per-arm statistics (Thompson sampling: Beta distribution)
	armStats []ArmStats

	// Recent history
	recentOutcomes []Outcome
	maxHistory     int // e.g., 1000 recent outcomes

	// Current active arm
	currentArm int

	// Exploration schedule
	explorationRate float64 // Probability of random arm selection
	totalPulls      int64
	lastUpdate      time.Time
}

// EnsembleArm represents an ensemble configuration option
type EnsembleArm struct {
	N int // Number of required agreements
	M int // Total strategies

	// Strategy weights (sum to 1.0)
	Weights map[string]float64 // strategy_name -> weight
}

// ArmStats tracks reward statistics for an arm (Thompson sampling)
type ArmStats struct {
	// Beta distribution parameters
	Alpha float64 // Successes + 1
	Beta  float64 // Failures + 1

	// UCB statistics
	TotalReward float64
	NumPulls    int64
	AvgReward   float64

	// Constraints tracking
	AvgLatencyMs float64
	AvgCostDelta float64
}

// Outcome represents the result of applying an ensemble configuration
type Outcome struct {
	Arm          int
	Contained    bool    // Did the PCS get contained (accepted or escalated appropriately)?
	Agreed       bool    // Did strategies agree?
	LatencyMs    float64
	CostDelta    float64 // % change vs baseline
	Reward       float64 // Computed reward
	Timestamp    time.Time
}

// BanditMetrics tracks bandit controller performance
type BanditMetrics struct {
	mu                   sync.RWMutex
	TotalDecisions       int64
	ExplorationDecisions int64
	ExploitationDecisions int64
	AvgReward            float64
	TenantCount          int
	LastUpdate           time.Time
}

// NewBanditController creates a bandit controller for ensemble optimization
func NewBanditController() *BanditController {
	return &BanditController{
		tenantBandits: make(map[string]*TenantBandit),

		// Reward function (Phase 9 multi-objective)
		containmentWeight: 0.5,
		agreementWeight:   0.3,
		latencyPenalty:    0.01,  // 1% penalty per ms over budget
		costPenalty:       0.1,   // 10% penalty per 1% cost increase

		// SLO constraints from CLAUDE_PHASE9.md
		maxLatencyMs: 120.0, // p95
		maxCostDelta: 0.07,  // +7%

		explorationStrategy: "thompson_sampling",
		ucbExplorationParam: 2.0,

		metrics: &BanditMetrics{},
	}
}

// SelectArm chooses an ensemble configuration for a tenant
func (bc *BanditController) SelectArm(ctx context.Context, tenantID string) (*EnsembleArm, error) {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	// Get or create tenant bandit
	bandit, ok := bc.tenantBandits[tenantID]
	if !ok {
		bandit = bc.initializeTenantBandit(tenantID)
		bc.tenantBandits[tenantID] = bandit
	}

	// Decide: explore or exploit
	var selectedArm int
	if bc.shouldExplore(bandit) {
		// Exploration: random arm
		selectedArm = rand.Intn(len(bandit.arms))
		bc.metrics.ExplorationDecisions++
	} else {
		// Exploitation: select best arm
		if bc.explorationStrategy == "thompson_sampling" {
			selectedArm = bc.thompsonSampling(bandit)
		} else {
			selectedArm = bc.ucbSelection(bandit)
		}
		bc.metrics.ExploitationDecisions++
	}

	bandit.currentArm = selectedArm
	bandit.totalPulls++

	bc.metrics.TotalDecisions++
	bc.metrics.LastUpdate = time.Now()

	return &bandit.arms[selectedArm], nil
}

// UpdateReward records the outcome of an arm pull
func (bc *BanditController) UpdateReward(ctx context.Context, tenantID string, outcome Outcome) error {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	bandit, ok := bc.tenantBandits[tenantID]
	if !ok {
		return fmt.Errorf("tenant bandit not found: %s", tenantID)
	}

	armIdx := outcome.Arm

	// Update arm statistics
	stats := &bandit.armStats[armIdx]
	stats.NumPulls++
	stats.TotalReward += outcome.Reward
	stats.AvgReward = stats.TotalReward / float64(stats.NumPulls)

	// Update Beta distribution (Thompson sampling)
	if outcome.Reward > 0.5 { // Threshold for "success"
		stats.Alpha++
	} else {
		stats.Beta++
	}

	// Update constraint tracking (EMA)
	alpha := 0.1
	stats.AvgLatencyMs = alpha*outcome.LatencyMs + (1-alpha)*stats.AvgLatencyMs
	stats.AvgCostDelta = alpha*outcome.CostDelta + (1-alpha)*stats.AvgCostDelta

	// Add to recent history
	bandit.recentOutcomes = append(bandit.recentOutcomes, outcome)
	if len(bandit.recentOutcomes) > bandit.maxHistory {
		bandit.recentOutcomes = bandit.recentOutcomes[1:]
	}

	bandit.lastUpdate = time.Now()

	// Update global metrics
	bc.updateGlobalMetrics(outcome.Reward)

	return nil
}

// thompsonSampling selects arm using Thompson sampling (Beta distribution)
func (bc *BanditController) thompsonSampling(bandit *TenantBandit) int {
	maxSample := -1.0
	selectedArm := 0

	for i, stats := range bandit.armStats {
		// Sample from Beta(alpha, beta)
		sample := bc.sampleBeta(stats.Alpha, stats.Beta)

		// Apply constraint penalties
		penalty := 0.0
		if stats.AvgLatencyMs > bc.maxLatencyMs {
			penalty += bc.latencyPenalty * (stats.AvgLatencyMs - bc.maxLatencyMs)
		}
		if stats.AvgCostDelta > bc.maxCostDelta {
			penalty += bc.costPenalty * (stats.AvgCostDelta - bc.maxCostDelta)
		}

		adjustedSample := sample - penalty

		if adjustedSample > maxSample {
			maxSample = adjustedSample
			selectedArm = i
		}
	}

	return selectedArm
}

// ucbSelection selects arm using Upper Confidence Bound
func (bc *BanditController) ucbSelection(bandit *TenantBandit) int {
	totalPulls := float64(bandit.totalPulls)
	maxUCB := -math.MaxFloat64
	selectedArm := 0

	for i, stats := range bandit.armStats {
		if stats.NumPulls == 0 {
			// Unpulled arm: infinite UCB, select immediately
			return i
		}

		// UCB formula: avgReward + c * sqrt(log(totalPulls) / numPulls)
		exploration := bc.ucbExplorationParam * math.Sqrt(math.Log(totalPulls)/float64(stats.NumPulls))
		ucb := stats.AvgReward + exploration

		// Apply constraint penalties
		penalty := 0.0
		if stats.AvgLatencyMs > bc.maxLatencyMs {
			penalty += bc.latencyPenalty * (stats.AvgLatencyMs - bc.maxLatencyMs)
		}
		if stats.AvgCostDelta > bc.maxCostDelta {
			penalty += bc.costPenalty * (stats.AvgCostDelta - bc.maxCostDelta)
		}

		adjustedUCB := ucb - penalty

		if adjustedUCB > maxUCB {
			maxUCB = adjustedUCB
			selectedArm = i
		}
	}

	return selectedArm
}

// sampleBeta samples from Beta(alpha, beta) using rejection sampling
func (bc *BanditController) sampleBeta(alpha, beta float64) float64 {
	// Simple rejection sampling for Beta distribution
	// In production, use proper library (e.g., gonum)

	if alpha <= 0 || beta <= 0 {
		return 0.5 // Fallback
	}

	// Approximate using Gamma distributions: X ~ Beta(α,β) = Gamma(α) / (Gamma(α) + Gamma(β))
	// For simplicity, use uniform approximation
	u := rand.Float64()

	// Crude approximation: mode of Beta(α,β) is (α-1)/(α+β-2)
	if alpha > 1 && beta > 1 {
		mode := (alpha - 1) / (alpha + beta - 2)
		// Sample around mode with noise
		sample := mode + 0.1*(u-0.5)
		if sample < 0 {
			sample = 0
		}
		if sample > 1 {
			sample = 1
		}
		return sample
	}

	return u // Fallback to uniform
}

// shouldExplore decides whether to explore (random arm) vs exploit
func (bc *BanditController) shouldExplore(bandit *TenantBandit) bool {
	// Epsilon-greedy with decay
	epsilon := bandit.explorationRate
	return rand.Float64() < epsilon
}

// initializeTenantBandit creates a new bandit for a tenant with default arms
func (bc *BanditController) initializeTenantBandit(tenantID string) *TenantBandit {
	// Define arm space: (N, M) configurations
	arms := []EnsembleArm{
		// Phase 8 baseline: 2-of-3
		{N: 2, M: 3, Weights: map[string]float64{
			"pcs_recompute": 1.0,
			"micro_vote":    0.8,
			"rag_grounding": 0.9,
		}},
		// More conservative: 3-of-3
		{N: 3, M: 3, Weights: map[string]float64{
			"pcs_recompute": 1.0,
			"micro_vote":    1.0,
			"rag_grounding": 1.0,
		}},
		// More permissive: 2-of-4
		{N: 2, M: 4, Weights: map[string]float64{
			"pcs_recompute": 1.0,
			"micro_vote":    0.7,
			"rag_grounding": 0.8,
			"lightweight":   0.5,
		}},
		// Balanced: 3-of-4
		{N: 3, M: 4, Weights: map[string]float64{
			"pcs_recompute": 1.0,
			"micro_vote":    0.9,
			"rag_grounding": 0.9,
			"lightweight":   0.6,
		}},
	}

	// Initialize statistics for each arm (uniform prior: Beta(1,1))
	armStats := make([]ArmStats, len(arms))
	for i := range armStats {
		armStats[i] = ArmStats{
			Alpha: 1.0, // Uniform prior
			Beta:  1.0,
		}
	}

	return &TenantBandit{
		TenantID:        tenantID,
		arms:            arms,
		armStats:        armStats,
		recentOutcomes:  []Outcome{},
		maxHistory:      1000,
		currentArm:      0,
		explorationRate: 0.1, // 10% exploration
		totalPulls:      0,
		lastUpdate:      time.Now(),
	}
}

// ComputeReward calculates reward for an outcome based on multi-objective function
func (bc *BanditController) ComputeReward(contained, agreed bool, latencyMs, costDelta float64) float64 {
	reward := 0.0

	// Containment reward
	if contained {
		reward += bc.containmentWeight
	}

	// Agreement reward
	if agreed {
		reward += bc.agreementWeight
	}

	// Latency penalty (if over budget)
	if latencyMs > bc.maxLatencyMs {
		reward -= bc.latencyPenalty * (latencyMs - bc.maxLatencyMs)
	}

	// Cost penalty (if over budget)
	if costDelta > bc.maxCostDelta {
		reward -= bc.costPenalty * (costDelta - bc.maxCostDelta)
	}

	// Normalize to [0, 1]
	maxReward := bc.containmentWeight + bc.agreementWeight
	reward = reward / maxReward
	if reward < 0 {
		reward = 0
	}
	if reward > 1 {
		reward = 1
	}

	return reward
}

// updateGlobalMetrics updates aggregate bandit metrics
func (bc *BanditController) updateGlobalMetrics(reward float64) {
	bc.metrics.mu.Lock()
	defer bc.metrics.mu.Unlock()

	// Update EMA of average reward
	alpha := 0.05
	if bc.metrics.TotalDecisions == 1 {
		bc.metrics.AvgReward = reward
	} else {
		bc.metrics.AvgReward = alpha*reward + (1-alpha)*bc.metrics.AvgReward
	}

	bc.metrics.TenantCount = len(bc.tenantBandits)
	bc.metrics.LastUpdate = time.Now()
}

// GetMetrics returns current bandit metrics
func (bc *BanditController) GetMetrics() *BanditMetrics {
	bc.metrics.mu.RLock()
	defer bc.metrics.mu.RUnlock()

	return &BanditMetrics{
		TotalDecisions:        bc.metrics.TotalDecisions,
		ExplorationDecisions:  bc.metrics.ExplorationDecisions,
		ExploitationDecisions: bc.metrics.ExploitationDecisions,
		AvgReward:             bc.metrics.AvgReward,
		TenantCount:           bc.metrics.TenantCount,
		LastUpdate:            bc.metrics.LastUpdate,
	}
}

// GetTenantStats returns statistics for a specific tenant
func (bc *BanditController) GetTenantStats(tenantID string) (*TenantBanditStats, error) {
	bc.mu.RLock()
	defer bc.mu.RUnlock()

	bandit, ok := bc.tenantBandits[tenantID]
	if !ok {
		return nil, fmt.Errorf("tenant not found: %s", tenantID)
	}

	// Find best arm
	bestArm := 0
	maxReward := -math.MaxFloat64
	for i, stats := range bandit.armStats {
		if stats.AvgReward > maxReward {
			maxReward = stats.AvgReward
			bestArm = i
		}
	}

	stats := &TenantBanditStats{
		TenantID:    tenantID,
		TotalPulls:  bandit.totalPulls,
		CurrentArm:  bandit.currentArm,
		BestArm:     bestArm,
		BestReward:  maxReward,
		ArmStats:    make([]ArmPerformance, len(bandit.arms)),
		LastUpdate:  bandit.lastUpdate,
	}

	for i, arm := range bandit.arms {
		stats.ArmStats[i] = ArmPerformance{
			ArmIndex:     i,
			N:            arm.N,
			M:            arm.M,
			NumPulls:     bandit.armStats[i].NumPulls,
			AvgReward:    bandit.armStats[i].AvgReward,
			AvgLatencyMs: bandit.armStats[i].AvgLatencyMs,
			AvgCostDelta: bandit.armStats[i].AvgCostDelta,
		}
	}

	return stats, nil
}

// TenantBanditStats summarizes a tenant's bandit state
type TenantBanditStats struct {
	TenantID   string
	TotalPulls int64
	CurrentArm int
	BestArm    int
	BestReward float64
	ArmStats   []ArmPerformance
	LastUpdate time.Time
}

// ArmPerformance captures per-arm metrics
type ArmPerformance struct {
	ArmIndex     int
	N            int
	M            int
	NumPulls     int64
	AvgReward    float64
	AvgLatencyMs float64
	AvgCostDelta float64
}
