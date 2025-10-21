package hrs

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// HallucinationRiskScorer provides real-time risk prediction (Phase 7 WP1)
// Strict latency budget: ≤10ms p95
type HallucinationRiskScorer struct {
	mu            sync.RWMutex
	featureStore  *FeatureStore
	model         RiskModel
	calibrator    *ProbabilityCalibrator
	metrics       *HRSMetrics
}

// RiskScore contains risk prediction with confidence intervals
type RiskScore struct {
	Risk     float64   // Predicted hallucination risk [0, 1]
	CILow    float64   // Lower bound of 95% confidence interval
	CIHigh   float64   // Upper bound of 95% confidence interval
	Features *PCSFeatures // Input features (for debugging)
	ModelVersion string
	ComputedAt   time.Time
	LatencyMs    float64
}

// RiskModel interface for pluggable models
type RiskModel interface {
	Predict(features *PCSFeatures) (float64, error)
	PredictWithUncertainty(features *PCSFeatures) (float64, float64, error) // mean, std
	GetVersion() string
}

// LogisticRegressionModel implements RiskModel with logistic regression
type LogisticRegressionModel struct {
	mu       sync.RWMutex
	weights  map[string]float64
	intercept float64
	version  string
}

// GradientBoostingModel implements RiskModel with gradient boosting (placeholder)
type GradientBoostingModel struct {
	mu       sync.RWMutex
	trees    []DecisionTree
	learningRate float64
	version  string
}

// DecisionTree represents a decision tree in ensemble
type DecisionTree struct {
	Feature   string
	Threshold float64
	LeftValue float64
	RightValue float64
}

// ProbabilityCalibrator calibrates raw model outputs to true probabilities
type ProbabilityCalibrator struct {
	mu        sync.RWMutex
	method    string // "platt" for Platt scaling
	a         float64 // Platt scaling parameter
	b         float64
}

// HRSMetrics tracks HRS performance
type HRSMetrics struct {
	mu                    sync.RWMutex
	PredictionsTotal      int64
	PredictionsHighRisk   int64
	PredictionLatencyMs   *prometheus.HistogramVec
	CalibrationError      float64
}

// NewHallucinationRiskScorer creates a new HRS
func NewHallucinationRiskScorer(featureStore *FeatureStore) *HallucinationRiskScorer {
	// Initialize with logistic regression model (default)
	model := NewLogisticRegressionModel()

	return &HallucinationRiskScorer{
		featureStore: featureStore,
		model:        model,
		calibrator:   NewProbabilityCalibrator("platt"),
		metrics: &HRSMetrics{
			PredictionLatencyMs: promauto.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:    "flk_hrs_latency_ms",
					Help:    "HRS prediction latency in milliseconds",
					Buckets: []float64{1, 2, 5, 10, 20, 50},
				},
				[]string{"model_version"},
			),
		},
	}
}

// NewLogisticRegressionModel creates a new logistic regression model
func NewLogisticRegressionModel() *LogisticRegressionModel {
	// Trained weights (placeholder - would be learned from data)
	weights := map[string]float64{
		"D_hat":              0.3,  // Higher D̂ → higher risk
		"coh_star":           -0.5, // Higher coherence → lower risk
		"r":                  -0.4, // Higher compressibility → lower risk
		"budget":             -0.6, // Higher budget → lower risk
		"signal_entropy":     0.2,  // Higher entropy → higher risk
		"coherence_delta":    0.15, // Deviation from normal → higher risk
		"compressibility_z":  0.1,  // Z-score deviation → higher risk
		"verify_latency_ms":  0.05, // Longer latency → higher risk
	}

	return &LogisticRegressionModel{
		weights:   weights,
		intercept: -0.5, // Bias term
		version:   "lr-v1.0",
	}
}

// NewProbabilityCalibrator creates a new probability calibrator
func NewProbabilityCalibrator(method string) *ProbabilityCalibrator {
	return &ProbabilityCalibrator{
		method: method,
		a:      1.0, // Platt scaling parameters (would be learned)
		b:      0.0,
	}
}

// PredictRisk predicts hallucination risk with confidence intervals
// Latency budget: ≤10ms p95
func (hrs *HallucinationRiskScorer) PredictRisk(ctx context.Context, features *PCSFeatures) (*RiskScore, error) {
	startTime := time.Now()

	// Check context timeout
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	// Get prediction with uncertainty
	rawScore, uncertainty, err := hrs.model.PredictWithUncertainty(features)
	if err != nil {
		return nil, fmt.Errorf("model prediction failed: %w", err)
	}

	// Calibrate probability
	calibratedScore := hrs.calibrator.Calibrate(rawScore)

	// Compute confidence intervals (95% CI)
	ciLow := math.Max(0, calibratedScore-1.96*uncertainty)
	ciHigh := math.Min(1, calibratedScore+1.96*uncertainty)

	latencyMs := time.Since(startTime).Milliseconds()

	riskScore := &RiskScore{
		Risk:         calibratedScore,
		CILow:        ciLow,
		CIHigh:       ciHigh,
		Features:     features,
		ModelVersion: hrs.model.GetVersion(),
		ComputedAt:   time.Now(),
		LatencyMs:    float64(latencyMs),
	}

	// Record metrics
	hrs.recordPrediction(riskScore)

	// Check latency budget
	if latencyMs > 10 {
		fmt.Printf("HRS latency warning: %.2fms (target: ≤10ms)\n", float64(latencyMs))
	}

	return riskScore, nil
}

// Predict implements RiskModel for LogisticRegressionModel
func (lrm *LogisticRegressionModel) Predict(features *PCSFeatures) (float64, error) {
	lrm.mu.RLock()
	defer lrm.mu.RUnlock()

	// Linear combination: z = w·x + b
	z := lrm.intercept
	z += lrm.weights["D_hat"] * features.DHat
	z += lrm.weights["coh_star"] * features.CohStar
	z += lrm.weights["r"] * features.R
	z += lrm.weights["budget"] * features.Budget
	z += lrm.weights["signal_entropy"] * features.SignalEntropy
	z += lrm.weights["coherence_delta"] * features.CoherenceDelta
	z += lrm.weights["compressibility_z"] * features.CompressibilityZ
	z += lrm.weights["verify_latency_ms"] * features.VerifyLatencyMs / 1000.0 // Normalize

	// Sigmoid: σ(z) = 1 / (1 + e^(-z))
	score := 1.0 / (1.0 + math.Exp(-z))

	return score, nil
}

// PredictWithUncertainty predicts with uncertainty estimate
func (lrm *LogisticRegressionModel) PredictWithUncertainty(features *PCSFeatures) (float64, float64, error) {
	score, err := lrm.Predict(features)
	if err != nil {
		return 0, 0, err
	}

	// Estimate uncertainty from model confidence
	// For logistic regression: uncertainty ∝ σ(z) * (1 - σ(z))
	uncertainty := score * (1 - score)

	// Scale uncertainty (calibrated empirically)
	uncertainty = math.Sqrt(uncertainty) * 0.1

	return score, uncertainty, nil
}

// GetVersion returns model version
func (lrm *LogisticRegressionModel) GetVersion() string {
	lrm.mu.RLock()
	defer lrm.mu.RUnlock()
	return lrm.version
}

// Calibrate applies Platt scaling to raw score
func (pc *ProbabilityCalibrator) Calibrate(rawScore float64) float64 {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	if pc.method == "platt" {
		// Platt scaling: p = 1 / (1 + exp(a*score + b))
		return 1.0 / (1.0 + math.Exp(pc.a*rawScore+pc.b))
	}

	// No calibration
	return rawScore
}

// UpdateCalibration updates calibrator parameters
func (pc *ProbabilityCalibrator) UpdateCalibration(a, b float64) {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	pc.a = a
	pc.b = b
	fmt.Printf("Calibrator updated: a=%.3f, b=%.3f\n", a, b)
}

// recordPrediction records prediction metrics
func (hrs *HallucinationRiskScorer) recordPrediction(score *RiskScore) {
	hrs.metrics.mu.Lock()
	defer hrs.metrics.mu.Unlock()

	hrs.metrics.PredictionsTotal++
	if score.Risk >= 0.7 {
		hrs.metrics.PredictionsHighRisk++
	}

	// Record latency histogram
	hrs.metrics.PredictionLatencyMs.WithLabelValues(score.ModelVersion).Observe(score.LatencyMs)
}

// GetMetrics returns HRS metrics
func (hrs *HallucinationRiskScorer) GetMetrics() HRSMetrics {
	hrs.metrics.mu.RLock()
	defer hrs.metrics.mu.RUnlock()
	return *hrs.metrics
}

// GetHighRiskRate returns percentage of high-risk predictions
func (hrs *HallucinationRiskScorer) GetHighRiskRate() float64 {
	hrs.metrics.mu.RLock()
	defer hrs.metrics.mu.RUnlock()

	if hrs.metrics.PredictionsTotal == 0 {
		return 0
	}

	return float64(hrs.metrics.PredictionsHighRisk) / float64(hrs.metrics.PredictionsTotal) * 100
}

// --- Gradient Boosting Model (Placeholder) ---

// Predict implements RiskModel for GradientBoostingModel
func (gbm *GradientBoostingModel) Predict(features *PCSFeatures) (float64, error) {
	gbm.mu.RLock()
	defer gbm.mu.RUnlock()

	// Sum predictions from all trees
	prediction := 0.0
	for _, tree := range gbm.trees {
		prediction += gbm.learningRate * tree.Predict(features)
	}

	// Apply sigmoid
	score := 1.0 / (1.0 + math.Exp(-prediction))
	return score, nil
}

// PredictWithUncertainty predicts with uncertainty estimate
func (gbm *GradientBoostingModel) PredictWithUncertainty(features *PCSFeatures) (float64, float64, error) {
	score, err := gbm.Predict(features)
	if err != nil {
		return 0, 0, err
	}

	// Uncertainty from ensemble variance
	uncertainty := 0.05 // Placeholder

	return score, uncertainty, nil
}

// GetVersion returns model version
func (gbm *GradientBoostingModel) GetVersion() string {
	gbm.mu.RLock()
	defer gbm.mu.RUnlock()
	return gbm.version
}

// Predict for DecisionTree
func (dt *DecisionTree) Predict(features *PCSFeatures) float64 {
	// Simple decision tree (placeholder)
	var value float64
	switch dt.Feature {
	case "D_hat":
		value = features.DHat
	case "coh_star":
		value = features.CohStar
	case "r":
		value = features.R
	default:
		value = 0
	}

	if value < dt.Threshold {
		return dt.LeftValue
	}
	return dt.RightValue
}

// --- Shadow Mode Evaluation ---

// EvaluationMetrics tracks shadow mode performance
type EvaluationMetrics struct {
	TruePositives  int64
	FalsePositives int64
	TrueNegatives  int64
	FalseNegatives int64
}

// ComputeAUC computes Area Under ROC Curve (placeholder - would use full ROC analysis)
func (em *EvaluationMetrics) ComputeAUC() float64 {
	// Simplified AUC estimate
	tpr := float64(em.TruePositives) / float64(em.TruePositives+em.FalseNegatives)
	fpr := float64(em.FalsePositives) / float64(em.FalsePositives+em.TrueNegatives)

	// AUC ≈ (1 + TPR - FPR) / 2
	return (1 + tpr - fpr) / 2
}

// ComputePrecision computes precision
func (em *EvaluationMetrics) ComputePrecision() float64 {
	total := em.TruePositives + em.FalsePositives
	if total == 0 {
		return 0
	}
	return float64(em.TruePositives) / float64(total)
}

// ComputeRecall computes recall
func (em *EvaluationMetrics) ComputeRecall() float64 {
	total := em.TruePositives + em.FalseNegatives
	if total == 0 {
		return 0
	}
	return float64(em.TruePositives) / float64(total)
}
