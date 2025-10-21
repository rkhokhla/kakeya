package ensemble

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// EnsembleVerifier provides N-of-M ensemble verification (Phase 7 WP2)
// Combines multiple verification strategies with configurable acceptance rules
type EnsembleVerifier struct {
	mu           sync.RWMutex
	policy       *EnsemblePolicy
	strategies   []VerificationStrategy
	auditWriter  AuditWriter
	siemStream   SIEMStream
	metrics      *EnsembleMetrics
}

// EnsemblePolicy defines N-of-M acceptance rules
type EnsemblePolicy struct {
	Name           string
	N              int           // Minimum agreements required
	M              int           // Total strategies
	Timeout        time.Duration // Per-strategy timeout
	FailMode       string        // "open" or "closed"
	EnabledChecks  []string      // ["pcs_recompute", "retrieval_overlap", "micro_vote"]
	MinConfidence  float64       // Minimum confidence threshold [0, 1]
}

// VerificationStrategy interface for pluggable checks
type VerificationStrategy interface {
	Name() string
	Verify(ctx context.Context, evidence *VerificationEvidence) (*StrategyResult, error)
	Timeout() time.Duration
}

// VerificationEvidence contains inputs for verification strategies
type VerificationEvidence struct {
	// PCS fields
	PCSID       string
	MerkleRoot  string
	Epoch       int64
	ShardID     string
	DHat        float64
	CohStar     float64
	R           float64
	Budget      float64
	Scales      []int
	NjValues    map[string]int // Stringified scale → count
	Regime      string

	// Context
	TenantID    string
	Timestamp   time.Time

	// Optional retrieval context (for retrieval_overlap strategy)
	Citations   []string // Document IDs or snippets
	QueryText   string   // Original query (if applicable)

	// Optional cross-model context (for micro_vote strategy)
	ModelOutput string   // Model response to verify
	TaskType    string   // "rag", "summarization", etc.
}

// StrategyResult contains verification outcome from one strategy
type StrategyResult struct {
	StrategyName string
	Accepted     bool      // True if strategy accepts, false if rejects
	Confidence   float64   // Confidence in decision [0, 1]
	Details      string    // Human-readable explanation
	LatencyMs    float64
	ComputedAt   time.Time
	Metadata     map[string]interface{} // Strategy-specific metadata
}

// EnsembleResult aggregates N-of-M results
type EnsembleResult struct {
	Accepted      bool               // True if N-of-M threshold met
	TotalStrategies int              // M
	AgreedCount   int                // How many agreed (accepted=true)
	Results       []*StrategyResult  // Individual strategy results
	FinalDecision string             // "accept", "reject", "escalate"
	Confidence    float64            // Aggregate confidence
	LatencyMs     float64            // Total ensemble latency
	ComputedAt    time.Time
}

// AuditWriter writes WORM audit entries
type AuditWriter interface {
	WriteDisagreement(ctx context.Context, evidence *VerificationEvidence, result *EnsembleResult) error
}

// SIEMStream streams events to SIEM
type SIEMStream interface {
	Send(ctx context.Context, eventType string, payload map[string]interface{}) error
}

// EnsembleMetrics tracks ensemble performance
type EnsembleMetrics struct {
	mu                     sync.RWMutex
	TotalVerifications     int64
	Accepted               int64
	Rejected               int64
	Escalated              int64
	Disagreements          int64 // N-of-M threshold not met
	EnsembleLatencyMs      *prometheus.HistogramVec
	StrategyAgreementRate  *prometheus.GaugeVec
}

// --- Built-in Verification Strategies ---

// PCSRecomputeStrategy recomputes PCS signals for consistency
type PCSRecomputeStrategy struct {
	timeout    time.Duration
	tolerance  float64 // Tolerance for D̂ recomputation
}

// RetrievalOverlapStrategy checks citation overlap (shingle/Jaccard)
type RetrievalOverlapStrategy struct {
	timeout         time.Duration
	minJaccard      float64 // Minimum Jaccard similarity [0, 1]
	shingleSize     int     // N-gram size for shingling
}

// MicroVoteStrategy uses a lightweight auxiliary model for cross-check
type MicroVoteStrategy struct {
	timeout      time.Duration
	modelEndpoint string // Auxiliary model API endpoint
	threshold    float64 // Minimum agreement score [0, 1]
}

// NewEnsembleVerifier creates a new ensemble verifier
func NewEnsembleVerifier(policy *EnsemblePolicy, auditWriter AuditWriter, siemStream SIEMStream) *EnsembleVerifier {
	ev := &EnsembleVerifier{
		policy:      policy,
		strategies:  []VerificationStrategy{},
		auditWriter: auditWriter,
		siemStream:  siemStream,
		metrics: &EnsembleMetrics{
			EnsembleLatencyMs: promauto.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:    "flk_ensemble_latency_ms",
					Help:    "Ensemble verification latency in milliseconds",
					Buckets: []float64{10, 20, 50, 100, 200, 500},
				},
				[]string{"policy_name"},
			),
			StrategyAgreementRate: promauto.NewGaugeVec(
				prometheus.GaugeOpts{
					Name: "flk_ensemble_agreement_rate",
					Help: "Strategy agreement rate (0-1)",
				},
				[]string{"strategy_name"},
			),
		},
	}

	// Register strategies based on policy
	for _, checkName := range policy.EnabledChecks {
		switch checkName {
		case "pcs_recompute":
			ev.strategies = append(ev.strategies, NewPCSRecomputeStrategy(policy.Timeout, 0.15))
		case "retrieval_overlap":
			ev.strategies = append(ev.strategies, NewRetrievalOverlapStrategy(policy.Timeout, 0.6, 3))
		case "micro_vote":
			ev.strategies = append(ev.strategies, NewMicroVoteStrategy(policy.Timeout, "", 0.7))
		}
	}

	return ev
}

// Verify performs ensemble verification with N-of-M policy
func (ev *EnsembleVerifier) Verify(ctx context.Context, evidence *VerificationEvidence) (*EnsembleResult, error) {
	startTime := time.Now()

	ev.mu.RLock()
	policy := ev.policy
	strategies := ev.strategies
	ev.mu.RUnlock()

	results := make([]*StrategyResult, 0, len(strategies))
	resultChan := make(chan *StrategyResult, len(strategies))
	errorChan := make(chan error, len(strategies))

	// Run strategies in parallel
	var wg sync.WaitGroup
	for _, strategy := range strategies {
		wg.Add(1)
		go func(s VerificationStrategy) {
			defer wg.Done()

			// Create per-strategy timeout context
			strategyCtx, cancel := context.WithTimeout(ctx, s.Timeout())
			defer cancel()

			result, err := s.Verify(strategyCtx, evidence)
			if err != nil {
				errorChan <- fmt.Errorf("%s failed: %w", s.Name(), err)
				return
			}

			resultChan <- result
		}(strategy)
	}

	// Wait for all strategies to complete
	go func() {
		wg.Wait()
		close(resultChan)
		close(errorChan)
	}()

	// Collect results
	for result := range resultChan {
		results = append(results, result)
	}

	// Check for errors (non-fatal, strategies can fail)
	for err := range errorChan {
		fmt.Printf("Strategy error: %v\n", err)
	}

	// Compute N-of-M threshold
	agreedCount := 0
	totalConfidence := 0.0
	for _, result := range results {
		if result.Accepted {
			agreedCount++
			totalConfidence += result.Confidence
		}
	}

	accepted := agreedCount >= policy.N
	avgConfidence := 0.0
	if len(results) > 0 {
		avgConfidence = totalConfidence / float64(len(results))
	}

	// Determine final decision
	finalDecision := "escalate"
	if accepted && avgConfidence >= policy.MinConfidence {
		finalDecision = "accept"
	} else if !accepted && policy.FailMode == "closed" {
		finalDecision = "reject"
	}

	latencyMs := time.Since(startTime).Milliseconds()

	ensembleResult := &EnsembleResult{
		Accepted:        accepted,
		TotalStrategies: len(strategies),
		AgreedCount:     agreedCount,
		Results:         results,
		FinalDecision:   finalDecision,
		Confidence:      avgConfidence,
		LatencyMs:       float64(latencyMs),
		ComputedAt:      time.Now(),
	}

	// Record metrics
	ev.recordVerification(ensembleResult)

	// Handle disagreements
	if finalDecision == "escalate" || finalDecision == "reject" {
		ev.handleDisagreement(ctx, evidence, ensembleResult)
	}

	return ensembleResult, nil
}

// handleDisagreement logs disagreements to WORM and SIEM
func (ev *EnsembleVerifier) handleDisagreement(ctx context.Context, evidence *VerificationEvidence, result *EnsembleResult) {
	// Write to WORM audit log
	if ev.auditWriter != nil {
		if err := ev.auditWriter.WriteDisagreement(ctx, evidence, result); err != nil {
			fmt.Printf("Failed to write WORM audit entry: %v\n", err)
		}
	}

	// Stream to SIEM
	if ev.siemStream != nil {
		payload := map[string]interface{}{
			"event_type":       "ensemble_disagree",
			"pcs_id":           evidence.PCSID,
			"tenant_id":        evidence.TenantID,
			"agreed_count":     result.AgreedCount,
			"total_strategies": result.TotalStrategies,
			"final_decision":   result.FinalDecision,
			"confidence":       result.Confidence,
			"timestamp":        result.ComputedAt.Format(time.RFC3339),
		}

		if err := ev.siemStream.Send(ctx, "ensemble_disagree", payload); err != nil {
			fmt.Printf("Failed to send SIEM event: %v\n", err)
		}
	}

	// Increment disagreement counter
	ev.metrics.mu.Lock()
	ev.metrics.Disagreements++
	ev.metrics.mu.Unlock()
}

// recordVerification records ensemble metrics
func (ev *EnsembleVerifier) recordVerification(result *EnsembleResult) {
	ev.metrics.mu.Lock()
	defer ev.metrics.mu.Unlock()

	ev.metrics.TotalVerifications++
	switch result.FinalDecision {
	case "accept":
		ev.metrics.Accepted++
	case "reject":
		ev.metrics.Rejected++
	case "escalate":
		ev.metrics.Escalated++
	}

	// Record latency
	ev.metrics.EnsembleLatencyMs.WithLabelValues(ev.policy.Name).Observe(result.LatencyMs)

	// Record strategy agreement rates
	for _, strategyResult := range result.Results {
		agreementRate := 0.0
		if strategyResult.Accepted {
			agreementRate = 1.0
		}
		ev.metrics.StrategyAgreementRate.WithLabelValues(strategyResult.StrategyName).Set(agreementRate)
	}
}

// GetMetrics returns ensemble metrics
func (ev *EnsembleVerifier) GetMetrics() EnsembleMetrics {
	ev.metrics.mu.RLock()
	defer ev.metrics.mu.RUnlock()
	return *ev.metrics
}

// --- Strategy Implementations ---

// NewPCSRecomputeStrategy creates a new PCS recompute strategy
func NewPCSRecomputeStrategy(timeout time.Duration, tolerance float64) *PCSRecomputeStrategy {
	return &PCSRecomputeStrategy{
		timeout:   timeout,
		tolerance: tolerance,
	}
}

func (s *PCSRecomputeStrategy) Name() string {
	return "pcs_recompute"
}

func (s *PCSRecomputeStrategy) Timeout() time.Duration {
	return s.timeout
}

func (s *PCSRecomputeStrategy) Verify(ctx context.Context, evidence *VerificationEvidence) (*StrategyResult, error) {
	startTime := time.Now()

	// Recompute D̂ from scales and N_j
	recomputedDHat := s.recomputeFractalDimension(evidence.Scales, evidence.NjValues)

	// Check tolerance
	diff := math.Abs(recomputedDHat - evidence.DHat)
	accepted := diff <= s.tolerance

	// Compute confidence (inverse of normalized difference)
	confidence := 1.0 - math.Min(1.0, diff/s.tolerance)

	details := fmt.Sprintf("Recomputed D̂=%.3f vs reported D̂=%.3f (diff=%.3f, tol=%.3f)",
		recomputedDHat, evidence.DHat, diff, s.tolerance)

	latencyMs := time.Since(startTime).Milliseconds()

	return &StrategyResult{
		StrategyName: s.Name(),
		Accepted:     accepted,
		Confidence:   confidence,
		Details:      details,
		LatencyMs:    float64(latencyMs),
		ComputedAt:   time.Now(),
		Metadata: map[string]interface{}{
			"recomputed_d_hat": recomputedDHat,
			"reported_d_hat":   evidence.DHat,
			"difference":       diff,
		},
	}, nil
}

func (s *PCSRecomputeStrategy) recomputeFractalDimension(scales []int, njValues map[string]int) float64 {
	if len(scales) == 0 || len(njValues) == 0 {
		return 0
	}

	// Theil-Sen median slope: log2(scale) vs log2(N_j)
	var points [][2]float64
	for _, scale := range scales {
		nj, ok := njValues[fmt.Sprintf("%d", scale)]
		if !ok || nj == 0 {
			continue
		}
		x := math.Log2(float64(scale))
		y := math.Log2(float64(nj))
		points = append(points, [2]float64{x, y})
	}

	if len(points) < 2 {
		return 0
	}

	// Compute all pairwise slopes
	var slopes []float64
	for i := 0; i < len(points); i++ {
		for j := i + 1; j < len(points); j++ {
			dx := points[j][0] - points[i][0]
			if dx != 0 {
				dy := points[j][1] - points[i][1]
				slope := dy / dx
				slopes = append(slopes, slope)
			}
		}
	}

	if len(slopes) == 0 {
		return 0
	}

	// Median slope (simplified - use middle element after sort)
	// In production, use proper median implementation
	medianSlope := slopes[len(slopes)/2]

	return medianSlope
}

// NewRetrievalOverlapStrategy creates a new retrieval overlap strategy
func NewRetrievalOverlapStrategy(timeout time.Duration, minJaccard float64, shingleSize int) *RetrievalOverlapStrategy {
	return &RetrievalOverlapStrategy{
		timeout:     timeout,
		minJaccard:  minJaccard,
		shingleSize: shingleSize,
	}
}

func (s *RetrievalOverlapStrategy) Name() string {
	return "retrieval_overlap"
}

func (s *RetrievalOverlapStrategy) Timeout() time.Duration {
	return s.timeout
}

func (s *RetrievalOverlapStrategy) Verify(ctx context.Context, evidence *VerificationEvidence) (*StrategyResult, error) {
	startTime := time.Now()

	// Compute Jaccard similarity between citations
	jaccard := s.computeJaccardSimilarity(evidence.Citations)

	accepted := jaccard >= s.minJaccard
	confidence := jaccard // Jaccard is already [0, 1]

	details := fmt.Sprintf("Jaccard similarity=%.3f (min=%.3f, citations=%d)",
		jaccard, s.minJaccard, len(evidence.Citations))

	latencyMs := time.Since(startTime).Milliseconds()

	return &StrategyResult{
		StrategyName: s.Name(),
		Accepted:     accepted,
		Confidence:   confidence,
		Details:      details,
		LatencyMs:    float64(latencyMs),
		ComputedAt:   time.Now(),
		Metadata: map[string]interface{}{
			"jaccard":       jaccard,
			"citation_count": len(evidence.Citations),
		},
	}, nil
}

func (s *RetrievalOverlapStrategy) computeJaccardSimilarity(citations []string) float64 {
	if len(citations) < 2 {
		return 1.0 // Single citation → perfect overlap
	}

	// Compute shingles for each citation
	shingleSets := make([]map[string]bool, len(citations))
	for i, citation := range citations {
		shingleSets[i] = s.extractShingles(citation)
	}

	// Compute pairwise Jaccard and average
	totalJaccard := 0.0
	pairCount := 0
	for i := 0; i < len(shingleSets); i++ {
		for j := i + 1; j < len(shingleSets); j++ {
			jaccard := s.jaccardIndex(shingleSets[i], shingleSets[j])
			totalJaccard += jaccard
			pairCount++
		}
	}

	if pairCount == 0 {
		return 0
	}

	return totalJaccard / float64(pairCount)
}

func (s *RetrievalOverlapStrategy) extractShingles(text string) map[string]bool {
	shingles := make(map[string]bool)
	runes := []rune(text)
	for i := 0; i <= len(runes)-s.shingleSize; i++ {
		shingle := string(runes[i : i+s.shingleSize])
		shingles[shingle] = true
	}
	return shingles
}

func (s *RetrievalOverlapStrategy) jaccardIndex(setA, setB map[string]bool) float64 {
	intersection := 0
	union := make(map[string]bool)

	for k := range setA {
		union[k] = true
		if setB[k] {
			intersection++
		}
	}
	for k := range setB {
		union[k] = true
	}

	if len(union) == 0 {
		return 0
	}

	return float64(intersection) / float64(len(union))
}

// NewMicroVoteStrategy creates a new micro-vote strategy
func NewMicroVoteStrategy(timeout time.Duration, modelEndpoint string, threshold float64) *MicroVoteStrategy {
	return &MicroVoteStrategy{
		timeout:       timeout,
		modelEndpoint: modelEndpoint,
		threshold:     threshold,
	}
}

func (s *MicroVoteStrategy) Name() string {
	return "micro_vote"
}

func (s *MicroVoteStrategy) Timeout() time.Duration {
	return s.timeout
}

func (s *MicroVoteStrategy) Verify(ctx context.Context, evidence *VerificationEvidence) (*StrategyResult, error) {
	startTime := time.Now()

	// Placeholder: In production, call auxiliary model API
	// For now, use a heuristic based on signal quality
	agreementScore := s.computeAgreementScore(evidence)

	accepted := agreementScore >= s.threshold
	confidence := agreementScore

	details := fmt.Sprintf("Micro-vote agreement=%.3f (threshold=%.3f)",
		agreementScore, s.threshold)

	latencyMs := time.Since(startTime).Milliseconds()

	return &StrategyResult{
		StrategyName: s.Name(),
		Accepted:     accepted,
		Confidence:   confidence,
		Details:      details,
		LatencyMs:    float64(latencyMs),
		ComputedAt:   time.Now(),
		Metadata: map[string]interface{}{
			"agreement_score": agreementScore,
		},
	}, nil
}

func (s *MicroVoteStrategy) computeAgreementScore(evidence *VerificationEvidence) float64 {
	// Heuristic: High coherence + low D̂ → high agreement
	// In production, call auxiliary model
	score := (evidence.CohStar + (1.0 - evidence.DHat/3.5)) / 2.0
	return math.Max(0, math.Min(1, score))
}

// DefaultEnsemblePolicy returns a default N-of-M policy
func DefaultEnsemblePolicy() *EnsemblePolicy {
	return &EnsemblePolicy{
		Name:          "default-2of3",
		N:             2, // 2-of-3 acceptance
		M:             3,
		Timeout:       100 * time.Millisecond,
		FailMode:      "open", // Fail-open: escalate on disagreement
		EnabledChecks: []string{"pcs_recompute", "retrieval_overlap", "micro_vote"},
		MinConfidence: 0.7,
	}
}

// ComputePCSID computes pcs_id from evidence (for consistency checks)
func ComputePCSID(merkleRoot string, epoch int64, shardID string) string {
	payload := fmt.Sprintf("%s|%d|%s", merkleRoot, epoch, shardID)
	hash := sha256.Sum256([]byte(payload))
	return hex.EncodeToString(hash[:])
}
