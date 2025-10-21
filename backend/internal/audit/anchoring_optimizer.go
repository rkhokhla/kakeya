package audit

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// AnchoringPolicyOptimizer chooses optimal anchoring strategy (Phase 6 WP3)
type AnchoringPolicyOptimizer struct {
	mu            sync.RWMutex
	config        *OptimizerConfig
	costModel     *CostModel
	strategies    map[string]*AnchoringStrategy
	currentPolicy *AnchoringPolicy
	metrics       *OptimizerMetrics
}

// OptimizerConfig defines optimizer configuration
type OptimizerConfig struct {
	// MaxCostPerBatch is the maximum acceptable cost per batch (USD)
	MaxCostPerBatch float64

	// TargetLatency is the desired latency for attestation (seconds)
	TargetLatency int

	// RequiredDurability is the minimum durability level (1-5)
	RequiredDurability int

	// OptimizationGoal is the primary goal (cost, latency, durability)
	OptimizationGoal string

	// ReoptimizeInterval is how often to re-evaluate policy
	ReoptimizeInterval time.Duration
}

// CostModel defines cost parameters for different anchoring strategies
type CostModel struct {
	// Blockchain costs
	EthereumGasPrice   float64 // USD per gas unit
	EthereumGasPerTx   int     // Gas units per transaction
	PolygonGasPrice    float64
	PolygonGasPerTx    int

	// Timestamp service costs
	RFC3161CostPerStamp float64 // USD per timestamp

	// Storage costs
	IPFSCostPerGB      float64 // USD per GB per month
	AWSCostPerGB       float64

	// Operational overhead
	ComputeCostPerBatch float64
}

// AnchoringStrategy defines an anchoring strategy
type AnchoringStrategy struct {
	Name         string
	Provider     string  // ethereum, polygon, rfc3161, openTimestamps
	BatchSize    int     // Number of segment roots per batch
	Latency      int     // Expected latency in seconds
	Cost         float64 // Cost per batch in USD
	Durability   int     // Durability level (1-5)
	Reliability  float64 // Historical reliability (0.0-1.0)
}

// AnchoringPolicy defines the active anchoring policy
type AnchoringPolicy struct {
	Strategy      *AnchoringStrategy
	EffectiveFrom time.Time
	Reason        string
	Metadata      map[string]interface{}
}

// OptimizerMetrics tracks optimizer decisions
type OptimizerMetrics struct {
	mu                     sync.RWMutex
	PolicyChanges          int64
	TotalBatchesAnchored   int64
	TotalCostUSD           float64
	AverageLatencySeconds  float64
	CostSavingsPercent     float64
	LastOptimizationTime   time.Time
}

// NewAnchoringPolicyOptimizer creates a new anchoring policy optimizer
func NewAnchoringPolicyOptimizer(config *OptimizerConfig) *AnchoringPolicyOptimizer {
	optimizer := &AnchoringPolicyOptimizer{
		config:     config,
		costModel:  DefaultCostModel(),
		strategies: make(map[string]*AnchoringStrategy),
		metrics:    &OptimizerMetrics{},
	}

	// Register available strategies
	optimizer.registerStrategies()

	// Choose initial policy
	policy, _ := optimizer.OptimizePolicy(context.Background())
	optimizer.currentPolicy = policy

	return optimizer
}

// DefaultCostModel returns default cost parameters
func DefaultCostModel() *CostModel {
	return &CostModel{
		// Ethereum Mainnet (high cost, high durability)
		EthereumGasPrice: 0.00002, // $0.00002 per gas (~20 Gwei at $2000/ETH)
		EthereumGasPerTx: 100000,  // ~100k gas for data write

		// Polygon (low cost, good durability)
		PolygonGasPrice: 0.0000001, // $0.0000001 per gas
		PolygonGasPerTx: 100000,

		// RFC3161 Timestamp Authority
		RFC3161CostPerStamp: 0.01, // $0.01 per timestamp

		// Storage costs
		IPFSCostPerGB: 0.05,  // $0.05 per GB per month
		AWSCostPerGB:  0.023, // S3 Standard

		// Overhead
		ComputeCostPerBatch: 0.001, // $0.001 per batch
	}
}

// registerStrategies registers available anchoring strategies
func (apo *AnchoringPolicyOptimizer) registerStrategies() {
	// Ethereum Mainnet (highest durability, highest cost)
	apo.strategies["ethereum"] = &AnchoringStrategy{
		Name:        "Ethereum Mainnet",
		Provider:    "ethereum",
		BatchSize:   100,
		Latency:     300,  // 5 minutes (block confirmation)
		Cost:        2.0,  // $2.00 per batch (100k gas * $0.00002)
		Durability:  5,    // Maximum durability
		Reliability: 0.999,
	}

	// Polygon (good balance)
	apo.strategies["polygon"] = &AnchoringStrategy{
		Name:        "Polygon",
		Provider:    "polygon",
		BatchSize:   100,
		Latency:     60,   // 1 minute
		Cost:        0.01, // $0.01 per batch
		Durability:  4,
		Reliability: 0.995,
	}

	// RFC3161 Timestamp Authority (fast, moderate cost)
	apo.strategies["rfc3161"] = &AnchoringStrategy{
		Name:        "RFC3161 Timestamp",
		Provider:    "rfc3161",
		BatchSize:   100,
		Latency:     5,    // 5 seconds
		Cost:        0.01, // $0.01 per timestamp
		Durability:  3,
		Reliability: 0.99,
	}

	// OpenTimestamps (Bitcoin, delayed but low cost)
	apo.strategies["opentimestamps"] = &AnchoringStrategy{
		Name:        "OpenTimestamps (Bitcoin)",
		Provider:    "opentimestamps",
		BatchSize:   1000, // Can batch many entries
		Latency:     3600, // ~1 hour (Bitcoin block time)
		Cost:        0.0,  // Free (piggybacked on Bitcoin)
		Durability:  5,
		Reliability: 0.999,
	}

	// Hybrid strategy (timestamp now, blockchain later)
	apo.strategies["hybrid"] = &AnchoringStrategy{
		Name:        "Hybrid (RFC3161 + Polygon)",
		Provider:    "hybrid",
		BatchSize:   100,
		Latency:     5,     // Fast initial timestamp
		Cost:        0.011, // $0.01 (RFC3161) + $0.001 (Polygon later)
		Durability:  4,
		Reliability: 0.998,
	}
}

// OptimizePolicy chooses the optimal anchoring strategy
func (apo *AnchoringPolicyOptimizer) OptimizePolicy(ctx context.Context) (*AnchoringPolicy, error) {
	apo.mu.Lock()
	defer apo.mu.Unlock()

	// Score all strategies based on config
	scoredStrategies := make(map[string]float64)
	for name, strategy := range apo.strategies {
		score := apo.scoreStrategy(strategy)
		scoredStrategies[name] = score
	}

	// Find best strategy
	var bestStrategy *AnchoringStrategy
	var bestScore float64 = -math.MaxFloat64

	for name, score := range scoredStrategies {
		if score > bestScore {
			bestScore = score
			bestStrategy = apo.strategies[name]
		}
	}

	if bestStrategy == nil {
		return nil, fmt.Errorf("no suitable strategy found")
	}

	// Create policy
	policy := &AnchoringPolicy{
		Strategy:      bestStrategy,
		EffectiveFrom: time.Now(),
		Reason:        fmt.Sprintf("Optimized for %s (score: %.2f)", apo.config.OptimizationGoal, bestScore),
		Metadata: map[string]interface{}{
			"optimization_goal": apo.config.OptimizationGoal,
			"score":             bestScore,
			"alternatives":      scoredStrategies,
		},
	}

	// Record policy change
	if apo.currentPolicy == nil || apo.currentPolicy.Strategy.Name != policy.Strategy.Name {
		apo.metrics.PolicyChanges++
		fmt.Printf("Anchoring Optimizer: Policy changed to %s (reason: %s)\n",
			policy.Strategy.Name, policy.Reason)
	}

	apo.metrics.LastOptimizationTime = time.Now()

	return policy, nil
}

// scoreStrategy computes a score for a strategy based on config
func (apo *AnchoringPolicyOptimizer) scoreStrategy(strategy *AnchoringStrategy) float64 {
	score := 0.0

	// Check hard constraints
	if strategy.Cost > apo.config.MaxCostPerBatch {
		return -math.MaxFloat64 // Disqualify
	}

	if strategy.Durability < apo.config.RequiredDurability {
		return -math.MaxFloat64 // Disqualify
	}

	// Optimize based on goal
	switch apo.config.OptimizationGoal {
	case "cost":
		// Lower cost = higher score
		score = 100.0 / (strategy.Cost + 0.001) // Avoid div-by-zero

	case "latency":
		// Lower latency = higher score
		score = 1000.0 / (float64(strategy.Latency) + 1.0)

	case "durability":
		// Higher durability = higher score
		score = float64(strategy.Durability * 20)

	case "balanced":
		// Multi-objective optimization
		costScore := 100.0 / (strategy.Cost + 0.001)
		latencyScore := 1000.0 / (float64(strategy.Latency) + 1.0)
		durabilityScore := float64(strategy.Durability * 20)
		reliabilityScore := strategy.Reliability * 100

		// Weighted average
		score = 0.3*costScore + 0.2*latencyScore + 0.3*durabilityScore + 0.2*reliabilityScore

	default:
		score = 0.0
	}

	// Apply reliability penalty
	score *= strategy.Reliability

	return score
}

// GetCurrentPolicy returns the active anchoring policy
func (apo *AnchoringPolicyOptimizer) GetCurrentPolicy() *AnchoringPolicy {
	apo.mu.RLock()
	defer apo.mu.RUnlock()
	return apo.currentPolicy
}

// UpdatePolicy manually updates the anchoring policy
func (apo *AnchoringPolicyOptimizer) UpdatePolicy(strategyName string, reason string) error {
	apo.mu.Lock()
	defer apo.mu.Unlock()

	strategy, ok := apo.strategies[strategyName]
	if !ok {
		return fmt.Errorf("strategy not found: %s", strategyName)
	}

	apo.currentPolicy = &AnchoringPolicy{
		Strategy:      strategy,
		EffectiveFrom: time.Now(),
		Reason:        reason,
		Metadata:      map[string]interface{}{"manual_override": true},
	}

	apo.metrics.PolicyChanges++
	fmt.Printf("Anchoring Optimizer: Manual policy update to %s (reason: %s)\n", strategyName, reason)

	return nil
}

// RecordBatch records a batch anchoring operation
func (apo *AnchoringPolicyOptimizer) RecordBatch(cost float64, latency int) {
	apo.metrics.mu.Lock()
	defer apo.metrics.mu.Unlock()

	apo.metrics.TotalBatchesAnchored++
	apo.metrics.TotalCostUSD += cost

	// Update rolling average latency
	n := float64(apo.metrics.TotalBatchesAnchored)
	apo.metrics.AverageLatencySeconds = (apo.metrics.AverageLatencySeconds*(n-1) + float64(latency)) / n
}

// EstimateMonthlyCost estimates monthly anchoring cost
func (apo *AnchoringPolicyOptimizer) EstimateMonthlyCost(batchesPerDay int) float64 {
	policy := apo.GetCurrentPolicy()
	if policy == nil || policy.Strategy == nil {
		return 0.0
	}

	costPerBatch := policy.Strategy.Cost
	return costPerBatch * float64(batchesPerDay) * 30
}

// ComparativeAnalysis compares all strategies for a given workload
func (apo *AnchoringPolicyOptimizer) ComparativeAnalysis(batchesPerDay int) map[string]interface{} {
	apo.mu.RLock()
	defer apo.mu.RUnlock()

	analysis := make(map[string]interface{})

	for name, strategy := range apo.strategies {
		monthlyCost := strategy.Cost * float64(batchesPerDay) * 30
		analysis[name] = map[string]interface{}{
			"strategy":      strategy.Name,
			"cost_per_batch": strategy.Cost,
			"monthly_cost":  monthlyCost,
			"latency_sec":   strategy.Latency,
			"durability":    strategy.Durability,
			"reliability":   strategy.Reliability,
		}
	}

	return analysis
}

// GetMetrics returns optimizer metrics
func (apo *AnchoringPolicyOptimizer) GetMetrics() OptimizerMetrics {
	apo.metrics.mu.RLock()
	defer apo.metrics.mu.RUnlock()
	return *apo.metrics
}

// --- Cost Optimization Algorithms ---

// OptimizeForCostTarget chooses strategy that meets cost target
func (apo *AnchoringPolicyOptimizer) OptimizeForCostTarget(targetCostPerDay float64, batchesPerDay int) (*AnchoringStrategy, error) {
	targetCostPerBatch := targetCostPerDay / float64(batchesPerDay)

	apo.mu.RLock()
	defer apo.mu.RUnlock()

	// Find strategies within cost budget
	candidates := []*AnchoringStrategy{}
	for _, strategy := range apo.strategies {
		if strategy.Cost <= targetCostPerBatch {
			candidates = append(candidates, strategy)
		}
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("no strategy meets cost target of $%.4f per batch", targetCostPerBatch)
	}

	// Among candidates, choose highest durability
	var bestStrategy *AnchoringStrategy
	for _, candidate := range candidates {
		if bestStrategy == nil || candidate.Durability > bestStrategy.Durability {
			bestStrategy = candidate
		}
	}

	return bestStrategy, nil
}

// OptimizeForSLA chooses strategy that meets SLA requirements
func (apo *AnchoringPolicyOptimizer) OptimizeForSLA(maxLatencySec int, minDurability int) (*AnchoringStrategy, error) {
	apo.mu.RLock()
	defer apo.mu.RUnlock()

	// Find strategies meeting SLA
	candidates := []*AnchoringStrategy{}
	for _, strategy := range apo.strategies {
		if strategy.Latency <= maxLatencySec && strategy.Durability >= minDurability {
			candidates = append(candidates, strategy)
		}
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("no strategy meets SLA (latency ≤%ds, durability ≥%d)", maxLatencySec, minDurability)
	}

	// Among candidates, choose lowest cost
	var bestStrategy *AnchoringStrategy
	for _, candidate := range candidates {
		if bestStrategy == nil || candidate.Cost < bestStrategy.Cost {
			bestStrategy = candidate
		}
	}

	return bestStrategy, nil
}

// --- Dynamic Policy Adjustment ---

// StartAutoOptimization runs periodic policy optimization
func (apo *AnchoringPolicyOptimizer) StartAutoOptimization(ctx context.Context) {
	ticker := time.NewTicker(apo.config.ReoptimizeInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			policy, err := apo.OptimizePolicy(ctx)
			if err != nil {
				fmt.Printf("Anchoring Optimizer: Failed to optimize policy: %v\n", err)
				continue
			}

			apo.mu.Lock()
			apo.currentPolicy = policy
			apo.mu.Unlock()

		case <-ctx.Done():
			return
		}
	}
}

// UpdateCostModel updates cost parameters based on real-world data
func (apo *AnchoringPolicyOptimizer) UpdateCostModel(newModel *CostModel) {
	apo.mu.Lock()
	defer apo.mu.Unlock()

	apo.costModel = newModel

	// Recalculate strategy costs
	if eth, ok := apo.strategies["ethereum"]; ok {
		eth.Cost = float64(newModel.EthereumGasPerTx) * newModel.EthereumGasPrice
	}
	if polygon, ok := apo.strategies["polygon"]; ok {
		polygon.Cost = float64(newModel.PolygonGasPerTx) * newModel.PolygonGasPrice
	}
	if rfc3161, ok := apo.strategies["rfc3161"]; ok {
		rfc3161.Cost = newModel.RFC3161CostPerStamp
	}

	fmt.Println("Anchoring Optimizer: Cost model updated, strategy costs recalculated")
}
