package hrs

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// Explainer provides SHAP/LIME-style attributions for HRS predictions
// Phase 9 WP1: Explainable risk scores with PI-safe feature attributions
type Explainer struct {
	mu sync.RWMutex

	model            RiskModel
	baselineFeatures []float64 // Global feature means for SHAP background
	numSamples       int        // Number of perturbations for approximation
	cacheTTL         time.Duration
	cache            map[string]*CachedAttribution
	metrics          *ExplainabilityMetrics
}

// Attribution represents feature importance for a prediction
type Attribution struct {
	FeatureNames     []string            `json:"feature_names"`
	FeatureValues    []float64           `json:"feature_values"`
	Attributions     []float64           `json:"attributions"` // SHAP/LIME values
	PredictionScore  float64             `json:"prediction_score"`
	BaselineScore    float64             `json:"baseline_score"`
	Method           string              `json:"method"` // "shap", "lime"
	Confidence       float64             `json:"confidence"`
	ComputeTimeMs    float64             `json:"compute_time_ms"`
	Timestamp        time.Time           `json:"timestamp"`
}

// CachedAttribution stores computed attributions with TTL
type CachedAttribution struct {
	Attribution *Attribution
	ExpiresAt   time.Time
}

// ExplainabilityMetrics tracks explainer performance
type ExplainabilityMetrics struct {
	mu                   sync.RWMutex
	TotalExplanations    int64
	CacheHits            int64
	CacheMisses          int64
	AvgComputeTimeMs     float64
	P95ComputeTimeMs     float64
	SLOBreaches          int64 // >2ms
	LastUpdate           time.Time
}

// NewExplainer creates an explainer for the given model
func NewExplainer(model RiskModel, baselineFeatures []float64, numSamples int) *Explainer {
	if numSamples == 0 {
		numSamples = 100 // Default: 100 samples for SHAP approximation
	}

	return &Explainer{
		model:            model,
		baselineFeatures: baselineFeatures,
		numSamples:       numSamples,
		cacheTTL:         15 * time.Minute, // Cache explanations for 15 min
		cache:            make(map[string]*CachedAttribution),
		metrics:          &ExplainabilityMetrics{},
	}
}

// ExplainPrediction returns SHAP-style attributions for a prediction
// Async computation recommended for low latency; inline returns compact summary
func (e *Explainer) ExplainPrediction(ctx context.Context, features []float64, featureNames []string) (*Attribution, error) {
	start := time.Now()

	// Check cache
	cacheKey := e.computeCacheKey(features)
	if cached := e.getFromCache(cacheKey); cached != nil {
		e.metrics.mu.Lock()
		e.metrics.CacheHits++
		e.metrics.mu.Unlock()
		return cached, nil
	}

	e.metrics.mu.Lock()
	e.metrics.CacheMisses++
	e.metrics.mu.Unlock()

	// Compute baseline prediction (using global mean features)
	baselineScore, err := e.model.Predict(e.baselineFeatures)
	if err != nil {
		return nil, fmt.Errorf("baseline prediction failed: %w", err)
	}

	// Compute actual prediction
	predictionScore, err := e.model.Predict(features)
	if err != nil {
		return nil, fmt.Errorf("prediction failed: %w", err)
	}

	// SHAP approximation: Kernel SHAP with sampled coalitions
	attributions := e.computeSHAPApprox(features, predictionScore, baselineScore)

	// Compute confidence (inverse of normalized variance across attributions)
	confidence := e.computeConfidence(attributions)

	computeTime := time.Since(start).Seconds() * 1000 // ms

	attr := &Attribution{
		FeatureNames:    featureNames,
		FeatureValues:   features,
		Attributions:    attributions,
		PredictionScore: predictionScore,
		BaselineScore:   baselineScore,
		Method:          "shap_kernel_approx",
		Confidence:      confidence,
		ComputeTimeMs:   computeTime,
		Timestamp:       time.Now(),
	}

	// Cache result
	e.putInCache(cacheKey, attr)

	// Update metrics
	e.updateMetrics(computeTime)

	return attr, nil
}

// computeSHAPApprox implements Kernel SHAP approximation
// Uses sampled feature coalitions to estimate Shapley values
func (e *Explainer) computeSHAPApprox(features []float64, prediction, baseline float64) []float64 {
	numFeatures := len(features)
	attributions := make([]float64, numFeatures)

	// For each feature, compute marginal contribution
	for i := 0; i < numFeatures; i++ {
		var sumContrib float64
		var count int

		// Sample coalitions (feature subsets)
		for s := 0; s < e.numSamples; s++ {
			// Create coalition: include feature i with 50% probability
			coalition := make([]float64, numFeatures)
			includeTarget := (s % 2) == 0 // Alternate for balance

			for j := 0; j < numFeatures; j++ {
				if j == i {
					if includeTarget {
						coalition[j] = features[j]
					} else {
						coalition[j] = e.baselineFeatures[j]
					}
				} else {
					// Random coalition membership
					if (s+j)%3 == 0 { // Deterministic pseudo-random
						coalition[j] = features[j]
					} else {
						coalition[j] = e.baselineFeatures[j]
					}
				}
			}

			// Predict with this coalition
			coalScore, err := e.model.Predict(coalition)
			if err != nil {
				continue // Skip on error
			}

			// Marginal contribution of feature i
			if includeTarget {
				// Feature i included
				sumContrib += (coalScore - baseline)
			} else {
				// Feature i excluded
				sumContrib -= (coalScore - baseline)
			}
			count++
		}

		if count > 0 {
			attributions[i] = sumContrib / float64(count)
		}
	}

	// Normalize so sum equals (prediction - baseline)
	sumAttr := 0.0
	for _, a := range attributions {
		sumAttr += a
	}

	targetSum := prediction - baseline
	if math.Abs(sumAttr) > 1e-6 {
		scale := targetSum / sumAttr
		for i := range attributions {
			attributions[i] *= scale
		}
	}

	return attributions
}

// computeConfidence calculates confidence based on attribution variance
func (e *Explainer) computeConfidence(attributions []float64) float64 {
	if len(attributions) == 0 {
		return 0.0
	}

	// Compute variance
	mean := 0.0
	for _, a := range attributions {
		mean += math.Abs(a)
	}
	mean /= float64(len(attributions))

	variance := 0.0
	for _, a := range attributions {
		diff := math.Abs(a) - mean
		variance += diff * diff
	}
	variance /= float64(len(attributions))

	// Confidence: inverse of coefficient of variation
	if mean > 1e-6 {
		cv := math.Sqrt(variance) / mean
		confidence := 1.0 / (1.0 + cv) // Sigmoid-like mapping
		return math.Min(1.0, confidence)
	}

	return 0.5 // Neutral confidence if all attributions near zero
}

// computeCacheKey generates cache key from features (rounded to 3 decimals for fuzzy matching)
func (e *Explainer) computeCacheKey(features []float64) string {
	key := ""
	for _, f := range features {
		key += fmt.Sprintf("%.3f:", f)
	}
	return key
}

// getFromCache retrieves cached attribution if not expired
func (e *Explainer) getFromCache(key string) *Attribution {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if cached, ok := e.cache[key]; ok {
		if time.Now().Before(cached.ExpiresAt) {
			return cached.Attribution
		}
		// Expired, will be replaced
	}
	return nil
}

// putInCache stores attribution with TTL
func (e *Explainer) putInCache(key string, attr *Attribution) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.cache[key] = &CachedAttribution{
		Attribution: attr,
		ExpiresAt:   time.Now().Add(e.cacheTTL),
	}

	// Simple cleanup: if cache too large, clear old entries
	if len(e.cache) > 1000 {
		now := time.Now()
		for k, v := range e.cache {
			if now.After(v.ExpiresAt) {
				delete(e.cache, k)
			}
		}
	}
}

// updateMetrics updates explainability metrics
func (e *Explainer) updateMetrics(computeTimeMs float64) {
	e.metrics.mu.Lock()
	defer e.metrics.mu.Unlock()

	e.metrics.TotalExplanations++

	// Update moving average
	alpha := 0.1 // EMA smoothing factor
	if e.metrics.TotalExplanations == 1 {
		e.metrics.AvgComputeTimeMs = computeTimeMs
	} else {
		e.metrics.AvgComputeTimeMs = alpha*computeTimeMs + (1-alpha)*e.metrics.AvgComputeTimeMs
	}

	// Track SLO breaches (>2ms per CLAUDE_PHASE9.md)
	if computeTimeMs > 2.0 {
		e.metrics.SLOBreaches++
	}

	e.metrics.LastUpdate = time.Now()
}

// GetMetrics returns current explainability metrics
func (e *Explainer) GetMetrics() *ExplainabilityMetrics {
	e.metrics.mu.RLock()
	defer e.metrics.mu.RUnlock()

	return &ExplainabilityMetrics{
		TotalExplanations: e.metrics.TotalExplanations,
		CacheHits:         e.metrics.CacheHits,
		CacheMisses:       e.metrics.CacheMisses,
		AvgComputeTimeMs:  e.metrics.AvgComputeTimeMs,
		P95ComputeTimeMs:  e.metrics.P95ComputeTimeMs,
		SLOBreaches:       e.metrics.SLOBreaches,
		LastUpdate:        e.metrics.LastUpdate,
	}
}

// TopFeatures returns top N features by absolute attribution
func (attr *Attribution) TopFeatures(n int) []struct {
	Name        string
	Value       float64
	Attribution float64
} {
	type featureAttr struct {
		Name        string
		Value       float64
		Attribution float64
		AbsAttr     float64
	}

	features := make([]featureAttr, len(attr.FeatureNames))
	for i := range attr.FeatureNames {
		features[i] = featureAttr{
			Name:        attr.FeatureNames[i],
			Value:       attr.FeatureValues[i],
			Attribution: attr.Attributions[i],
			AbsAttr:     math.Abs(attr.Attributions[i]),
		}
	}

	// Sort by absolute attribution (descending)
	for i := 0; i < len(features)-1; i++ {
		for j := i + 1; j < len(features); j++ {
			if features[j].AbsAttr > features[i].AbsAttr {
				features[i], features[j] = features[j], features[i]
			}
		}
	}

	// Return top N
	limit := n
	if limit > len(features) {
		limit = len(features)
	}

	result := make([]struct {
		Name        string
		Value       float64
		Attribution float64
	}, limit)

	for i := 0; i < limit; i++ {
		result[i].Name = features[i].Name
		result[i].Value = features[i].Value
		result[i].Attribution = features[i].Attribution
	}

	return result
}
