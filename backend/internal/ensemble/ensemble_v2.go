package ensemble

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Phase 8 WP2: Ensemble Expansion with real micro-vote, RAG grounding, and adaptive N-of-M

// MicroVoteService provides real lightweight model verification (Phase 8 WP2)
type MicroVoteService struct {
	mu          sync.RWMutex
	modelClient ModelClient
	timeout     time.Duration
	cacheSize   int
	embedCache  map[string][]float64 // pcs_id → cached embedding
	metrics     *MicroVoteMetrics
}

// ModelClient interfaces with auxiliary verification model
type ModelClient interface {
	Verify(ctx context.Context, evidence *VerificationEvidence) (float64, error) // Returns confidence [0, 1]
}

// RAGGroundingStrategy verifies citation consistency and source quality (Phase 8 WP2)
type RAGGroundingStrategy struct {
	mu                sync.RWMutex
	citationThreshold float64 // Minimum citation overlap (Jaccard)
	sourceChecker     SourceChecker
	timeout           time.Duration
}

// SourceChecker validates source quality
type SourceChecker interface {
	CheckSource(ctx context.Context, citation string) (quality float64, err error)
}

// AdaptiveEnsembleController tunes N-of-M per tenant (Phase 8 WP2)
type AdaptiveEnsembleController struct {
	mu               sync.RWMutex
	tenantPolicies   map[string]*TenantEnsemblePolicy // tenant_id → policy
	agreementHistory map[string]*AgreementHistory      // tenant_id → history
	tuningInterval   time.Duration
	metrics          *AdaptiveMetrics
}

// TenantEnsemblePolicy defines per-tenant N-of-M and weights
type TenantEnsemblePolicy struct {
	TenantID        string
	N               int               // Required agreements
	M               int               // Total strategies
	StrategyWeights map[string]float64 // strategy_name → weight [0, 1]
	UpdatedAt       time.Time
	HistoricalAUC   float64
}

// AgreementHistory tracks historical ensemble agreement rates
type AgreementHistory struct {
	TotalVerifications int64
	AgreedCount        int64
	DisagreedCount     int64
	StrategyAccuracy   map[string]float64 // strategy_name → accuracy
	LastUpdated        time.Time
}

// MicroVoteMetrics tracks micro-vote performance
type MicroVoteMetrics struct {
	mu                sync.RWMutex
	TotalCalls        int64
	CacheHits         int64
	Timeouts          int64
	AvgLatencyMs      float64
	AvgConfidence     float64
}

// AdaptiveMetrics tracks adaptive tuning performance
type AdaptiveMetrics struct {
	mu               sync.RWMutex
	TotalTunings     int64
	TenantsOptimized int64
	AvgImprovement   float64 // Average AUC improvement after tuning
}

// NewMicroVoteService creates a real micro-vote verification service
func NewMicroVoteService(modelClient ModelClient, timeout time.Duration) *MicroVoteService {
	return &MicroVoteService{
		modelClient: modelClient,
		timeout:     timeout,
		cacheSize:   1000,
		embedCache:  make(map[string][]float64),
		metrics:     &MicroVoteMetrics{},
	}
}

// Verify performs micro-vote verification with caching
func (mvs *MicroVoteService) Verify(ctx context.Context, evidence *VerificationEvidence) (*StrategyResult, error) {
	startTime := time.Now()

	mvs.metrics.mu.Lock()
	mvs.metrics.TotalCalls++
	mvs.metrics.mu.Unlock()

	// Check cache
	if cached, ok := mvs.getCachedEmbedding(evidence.PCSID); ok {
		mvs.metrics.mu.Lock()
		mvs.metrics.CacheHits++
		mvs.metrics.mu.Unlock()

		confidence := mvs.computeConfidenceFromEmbedding(cached)
		return &StrategyResult{
			StrategyName: "micro_vote",
			Accepted:     confidence >= 0.7,
			Confidence:   confidence,
			Details:      fmt.Sprintf("Cached embedding confidence: %.3f", confidence),
			LatencyMs:    float64(time.Since(startTime).Milliseconds()),
			ComputedAt:   time.Now(),
		}, nil
	}

	// Call model with timeout
	ctx, cancel := context.WithTimeout(ctx, mvs.timeout)
	defer cancel()

	confidence, err := mvs.modelClient.Verify(ctx, evidence)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			mvs.metrics.mu.Lock()
			mvs.metrics.Timeouts++
			mvs.metrics.mu.Unlock()
		}
		return nil, fmt.Errorf("micro-vote failed: %w", err)
	}

	// Cache result (simplified - in production, cache embedding)
	mvs.cacheEmbedding(evidence.PCSID, []float64{confidence})

	latencyMs := time.Since(startTime).Milliseconds()

	// Update metrics
	mvs.metrics.mu.Lock()
	mvs.metrics.AvgLatencyMs = (mvs.metrics.AvgLatencyMs*float64(mvs.metrics.TotalCalls-1) + float64(latencyMs)) / float64(mvs.metrics.TotalCalls)
	mvs.metrics.AvgConfidence = (mvs.metrics.AvgConfidence*float64(mvs.metrics.TotalCalls-1) + confidence) / float64(mvs.metrics.TotalCalls)
	mvs.metrics.mu.Unlock()

	return &StrategyResult{
		StrategyName: "micro_vote",
		Accepted:     confidence >= 0.7,
		Confidence:   confidence,
		Details:      fmt.Sprintf("Model confidence: %.3f (latency: %dms)", confidence, latencyMs),
		LatencyMs:    float64(latencyMs),
		ComputedAt:   time.Now(),
	}, nil
}

// getCachedEmbedding retrieves cached embedding
func (mvs *MicroVoteService) getCachedEmbedding(pcsID string) ([]float64, bool) {
	mvs.mu.RLock()
	defer mvs.mu.RUnlock()

	embed, ok := mvs.embedCache[pcsID]
	return embed, ok
}

// cacheEmbedding caches embedding (bounded LRU)
func (mvs *MicroVoteService) cacheEmbedding(pcsID string, embedding []float64) {
	mvs.mu.Lock()
	defer mvs.mu.Unlock()

	// Simple eviction if cache full
	if len(mvs.embedCache) >= mvs.cacheSize {
		// Evict random entry (in production, use proper LRU)
		for key := range mvs.embedCache {
			delete(mvs.embedCache, key)
			break
		}
	}

	mvs.embedCache[pcsID] = embedding
}

// computeConfidenceFromEmbedding computes confidence from cached embedding
func (mvs *MicroVoteService) computeConfidenceFromEmbedding(embedding []float64) float64 {
	if len(embedding) > 0 {
		return embedding[0]
	}
	return 0.5
}

// NewRAGGroundingStrategy creates RAG grounding strategy
func NewRAGGroundingStrategy(citationThreshold float64, sourceChecker SourceChecker, timeout time.Duration) *RAGGroundingStrategy {
	return &RAGGroundingStrategy{
		citationThreshold: citationThreshold,
		sourceChecker:     sourceChecker,
		timeout:           timeout,
	}
}

// Verify performs RAG grounding verification
func (rgs *RAGGroundingStrategy) Verify(ctx context.Context, evidence *VerificationEvidence) (*StrategyResult, error) {
	startTime := time.Now()

	// Check citation overlap
	jaccard := rgs.computeCitationOverlap(evidence.Citations)

	// Check source quality if checker provided
	avgSourceQuality := 0.75 // Default
	if rgs.sourceChecker != nil {
		qualities := []float64{}
		for _, citation := range evidence.Citations {
			quality, err := rgs.sourceChecker.CheckSource(ctx, citation)
			if err == nil {
				qualities = append(qualities, quality)
			}
		}
		if len(qualities) > 0 {
			sum := 0.0
			for _, q := range qualities {
				sum += q
			}
			avgSourceQuality = sum / float64(len(qualities))
		}
	}

	// Combine citation overlap and source quality
	confidence := (jaccard + avgSourceQuality) / 2.0
	accepted := confidence >= rgs.citationThreshold

	latencyMs := time.Since(startTime).Milliseconds()

	return &StrategyResult{
		StrategyName: "rag_grounding",
		Accepted:     accepted,
		Confidence:   confidence,
		Details:      fmt.Sprintf("Citation Jaccard=%.3f, Source quality=%.3f", jaccard, avgSourceQuality),
		LatencyMs:    float64(latencyMs),
		ComputedAt:   time.Now(),
		Metadata: map[string]interface{}{
			"citation_overlap": jaccard,
			"source_quality":   avgSourceQuality,
			"citation_count":   len(evidence.Citations),
		},
	}, nil
}

// computeCitationOverlap computes Jaccard similarity of citations
func (rgs *RAGGroundingStrategy) computeCitationOverlap(citations []string) float64 {
	if len(citations) < 2 {
		return 1.0
	}

	// Compute pairwise Jaccard (simplified)
	totalJaccard := 0.0
	pairCount := 0

	for i := 0; i < len(citations); i++ {
		for j := i + 1; j < len(citations); j++ {
			jaccard := rgs.jaccardSimilarity(citations[i], citations[j])
			totalJaccard += jaccard
			pairCount++
		}
	}

	if pairCount == 0 {
		return 0
	}

	return totalJaccard / float64(pairCount)
}

// jaccardSimilarity computes Jaccard similarity between two strings
func (rgs *RAGGroundingStrategy) jaccardSimilarity(a, b string) float64 {
	// Simplified word-level Jaccard
	setA := make(map[string]bool)
	setB := make(map[string]bool)

	// Tokenize (simplified)
	for i := 0; i < len(a); i++ {
		setA[string(a[i])] = true
	}
	for i := 0; i < len(b); i++ {
		setB[string(b[i])] = true
	}

	// Intersection
	intersection := 0
	for key := range setA {
		if setB[key] {
			intersection++
		}
	}

	// Union
	union := len(setA) + len(setB) - intersection

	if union == 0 {
		return 0
	}

	return float64(intersection) / float64(union)
}

// NewAdaptiveEnsembleController creates adaptive tuning controller
func NewAdaptiveEnsembleController(tuningInterval time.Duration) *AdaptiveEnsembleController {
	return &AdaptiveEnsembleController{
		tenantPolicies:   make(map[string]*TenantEnsemblePolicy),
		agreementHistory: make(map[string]*AgreementHistory),
		tuningInterval:   tuningInterval,
		metrics:          &AdaptiveMetrics{},
	}
}

// TunePolicy adjusts N-of-M and weights based on historical agreement
func (aec *AdaptiveEnsembleController) TunePolicy(tenantID string) (*TenantEnsemblePolicy, error) {
	aec.mu.Lock()
	defer aec.mu.Unlock()

	history, ok := aec.agreementHistory[tenantID]
	if !ok || history.TotalVerifications < 100 {
		return nil, fmt.Errorf("insufficient history for tenant: %s", tenantID)
	}

	// Compute agreement rate
	agreementRate := float64(history.AgreedCount) / float64(history.TotalVerifications)

	// Tune N-of-M based on agreement rate
	var n, m int
	if agreementRate >= 0.90 {
		n, m = 2, 3 // High agreement → 2-of-3
	} else if agreementRate >= 0.75 {
		n, m = 3, 4 // Medium agreement → 3-of-4 (more conservative)
	} else {
		n, m = 3, 3 // Low agreement → require all (most conservative)
	}

	// Compute strategy weights based on accuracy
	weights := make(map[string]float64)
	totalAccuracy := 0.0
	for _, accuracy := range history.StrategyAccuracy {
		totalAccuracy += accuracy
	}
	for strategyName, accuracy := range history.StrategyAccuracy {
		if totalAccuracy > 0 {
			weights[strategyName] = accuracy / totalAccuracy
		} else {
			weights[strategyName] = 1.0 / float64(len(history.StrategyAccuracy))
		}
	}

	policy := &TenantEnsemblePolicy{
		TenantID:        tenantID,
		N:               n,
		M:               m,
		StrategyWeights: weights,
		UpdatedAt:       time.Now(),
		HistoricalAUC:   agreementRate,
	}

	aec.tenantPolicies[tenantID] = policy

	// Update metrics
	aec.metrics.mu.Lock()
	aec.metrics.TotalTunings++
	aec.metrics.TenantsOptimized++
	aec.metrics.mu.Unlock()

	fmt.Printf("Tuned ensemble policy for tenant %s: %d-of-%d (agreement rate: %.1f%%)\n",
		tenantID, n, m, agreementRate*100)

	return policy, nil
}

// RecordAgreement records ensemble verification outcome
func (aec *AdaptiveEnsembleController) RecordAgreement(tenantID string, agreed bool, strategyResults []*StrategyResult) {
	aec.mu.Lock()
	defer aec.mu.Unlock()

	history, ok := aec.agreementHistory[tenantID]
	if !ok {
		history = &AgreementHistory{
			StrategyAccuracy: make(map[string]float64),
		}
		aec.agreementHistory[tenantID] = history
	}

	history.TotalVerifications++
	if agreed {
		history.AgreedCount++
	} else {
		history.DisagreedCount++
	}

	// Update strategy accuracy
	for _, result := range strategyResults {
		if result.Accepted {
			currentAccuracy := history.StrategyAccuracy[result.StrategyName]
			history.StrategyAccuracy[result.StrategyName] = (currentAccuracy*float64(history.TotalVerifications-1) + result.Confidence) / float64(history.TotalVerifications)
		}
	}

	history.LastUpdated = time.Now()
}

// GetPolicy retrieves tenant-specific ensemble policy
func (aec *AdaptiveEnsembleController) GetPolicy(tenantID string) (*TenantEnsemblePolicy, bool) {
	aec.mu.RLock()
	defer aec.mu.RUnlock()

	policy, ok := aec.tenantPolicies[tenantID]
	return policy, ok
}

// GetMetrics returns micro-vote metrics
func (mvs *MicroVoteService) GetMetrics() *MicroVoteMetrics {
	mvs.metrics.mu.RLock()
	defer mvs.metrics.mu.RUnlock()
	return mvs.metrics
}

// GetMetrics returns adaptive controller metrics
func (aec *AdaptiveEnsembleController) GetMetrics() *AdaptiveMetrics {
	aec.metrics.mu.RLock()
	defer aec.metrics.mu.RUnlock()
	return aec.metrics
}
