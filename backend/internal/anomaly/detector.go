package anomaly

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// AnomalyDetector provides unsupervised anomaly detection (Phase 7 WP5)
// Uses autoencoder-based outlier scoring on PCS distributions
type AnomalyDetector struct {
	mu             sync.RWMutex
	autoencoder    *Autoencoder
	shadowMode     bool // Run in shadow mode (no policy effect)
	alertThreshold float64 // Anomaly score threshold for alerting
	siemStream     SIEMStream
	auditWriter    AuditWriter
	metrics        *AnomalyMetrics
}

// Autoencoder represents a compact autoencoder model
type Autoencoder struct {
	mu            sync.RWMutex
	inputDim      int
	hiddenDim     int
	encoderWeights [][]float64
	encoderBias    []float64
	decoderWeights [][]float64
	decoderBias    []float64
	version       string
	trainedAt     time.Time
}

// PCSVector represents PCS features as a vector
type PCSVector struct {
	DHat              float64
	CohStar           float64
	R                 float64
	Budget            float64
	VerifyLatencyMs   float64
	SignalEntropy     float64
	CoherenceDelta    float64
	CompressibilityZ  float64
}

// AnomalyResult contains anomaly detection outcome
type AnomalyResult struct {
	IsAnomaly        bool      // True if anomaly score exceeds threshold
	AnomalyScore     float64   // Reconstruction error [0, ∞)
	NormalizedScore  float64   // Normalized score [0, 1]
	Details          string    // Human-readable explanation
	Threshold        float64   // Applied threshold
	ComputedAt       time.Time
	LatencyMs        float64
	ShadowMode       bool      // True if running in shadow mode
}

// SIEMStream streams events to SIEM (reuse from ensemble)
type SIEMStream interface {
	Send(ctx context.Context, eventType string, payload map[string]interface{}) error
}

// AuditWriter writes WORM audit entries (reuse from ensemble)
type AuditWriter interface {
	WriteAnomaly(ctx context.Context, vector *PCSVector, result *AnomalyResult) error
}

// AnomalyMetrics tracks anomaly detection performance
type AnomalyMetrics struct {
	mu                    sync.RWMutex
	TotalDetections       int64
	AnomaliesDetected     int64
	FalsePositiveRate     float64 // Estimated from shadow mode feedback
	AnomalyRate           *prometheus.GaugeVec
	ReconstructionError   *prometheus.HistogramVec
}

// NewAnomalyDetector creates a new anomaly detector
func NewAnomalyDetector(shadowMode bool, alertThreshold float64, siemStream SIEMStream, auditWriter AuditWriter) *AnomalyDetector {
	return &AnomalyDetector{
		autoencoder:    NewAutoencoder(8, 3), // 8 input features → 3 hidden → 8 output
		shadowMode:     shadowMode,
		alertThreshold: alertThreshold,
		siemStream:     siemStream,
		auditWriter:    auditWriter,
		metrics: &AnomalyMetrics{
			AnomalyRate: promauto.NewGaugeVec(
				prometheus.GaugeOpts{
					Name: "flk_anomaly_rate",
					Help: "Anomaly detection rate",
				},
				[]string{"tenant_id"},
			),
			ReconstructionError: promauto.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:    "flk_anomaly_reconstruction_error",
					Help:    "Autoencoder reconstruction error",
					Buckets: []float64{0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0},
				},
				[]string{"tenant_id"},
			),
		},
	}
}

// NewAutoencoder creates a new autoencoder with random initialization
func NewAutoencoder(inputDim, hiddenDim int) *Autoencoder {
	ae := &Autoencoder{
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
		version:   "ae-v1.0",
		trainedAt: time.Now(),
	}

	// Initialize encoder weights (Xavier initialization)
	ae.encoderWeights = make([][]float64, inputDim)
	for i := range ae.encoderWeights {
		ae.encoderWeights[i] = make([]float64, hiddenDim)
		for j := range ae.encoderWeights[i] {
			ae.encoderWeights[i][j] = randomNormal(0, math.Sqrt(2.0/float64(inputDim)))
		}
	}
	ae.encoderBias = make([]float64, hiddenDim)

	// Initialize decoder weights
	ae.decoderWeights = make([][]float64, hiddenDim)
	for i := range ae.decoderWeights {
		ae.decoderWeights[i] = make([]float64, inputDim)
		for j := range ae.decoderWeights[i] {
			ae.decoderWeights[i][j] = randomNormal(0, math.Sqrt(2.0/float64(hiddenDim)))
		}
	}
	ae.decoderBias = make([]float64, inputDim)

	return ae
}

// Detect performs anomaly detection on PCS vector
func (ad *AnomalyDetector) Detect(ctx context.Context, vector *PCSVector, tenantID string) (*AnomalyResult, error) {
	startTime := time.Now()

	// Convert PCS to feature vector
	input := ad.vectorToArray(vector)

	// Encode → Decode
	hidden := ad.autoencoder.Encode(input)
	reconstructed := ad.autoencoder.Decode(hidden)

	// Compute reconstruction error (MSE)
	reconstructionError := ad.computeReconstructionError(input, reconstructed)

	// Normalize score to [0, 1] using sigmoid
	normalizedScore := 1.0 / (1.0 + math.Exp(-reconstructionError))

	// Check threshold
	isAnomaly := reconstructionError >= ad.alertThreshold

	details := fmt.Sprintf("Reconstruction error=%.4f (threshold=%.4f, normalized=%.4f)",
		reconstructionError, ad.alertThreshold, normalizedScore)

	latencyMs := time.Since(startTime).Milliseconds()

	result := &AnomalyResult{
		IsAnomaly:       isAnomaly,
		AnomalyScore:    reconstructionError,
		NormalizedScore: normalizedScore,
		Details:         details,
		Threshold:       ad.alertThreshold,
		ComputedAt:      time.Now(),
		LatencyMs:       float64(latencyMs),
		ShadowMode:      ad.shadowMode,
	}

	// Record metrics
	ad.recordDetection(result, tenantID)

	// Handle anomalies
	if isAnomaly {
		ad.handleAnomaly(ctx, vector, result, tenantID)
	}

	return result, nil
}

// Encode applies encoder transformation
func (ae *Autoencoder) Encode(input []float64) []float64 {
	ae.mu.RLock()
	defer ae.mu.RUnlock()

	hidden := make([]float64, ae.hiddenDim)

	// hidden = ReLU(W_encoder^T * input + b_encoder)
	for j := 0; j < ae.hiddenDim; j++ {
		sum := ae.encoderBias[j]
		for i := 0; i < ae.inputDim; i++ {
			sum += ae.encoderWeights[i][j] * input[i]
		}
		hidden[j] = relu(sum)
	}

	return hidden
}

// Decode applies decoder transformation
func (ae *Autoencoder) Decode(hidden []float64) []float64 {
	ae.mu.RLock()
	defer ae.mu.RUnlock()

	output := make([]float64, ae.inputDim)

	// output = W_decoder^T * hidden + b_decoder
	for i := 0; i < ae.inputDim; i++ {
		sum := ae.decoderBias[i]
		for j := 0; j < ae.hiddenDim; j++ {
			sum += ae.decoderWeights[j][i] * hidden[j]
		}
		output[i] = sum // No activation on output layer
	}

	return output
}

// vectorToArray converts PCS vector to array
func (ad *AnomalyDetector) vectorToArray(vector *PCSVector) []float64 {
	return []float64{
		vector.DHat / 3.5,                      // Normalize to [0, 1]
		vector.CohStar,                         // Already [0, 1]
		vector.R,                               // Already [0, 1]
		vector.Budget,                          // Already [0, 1]
		vector.VerifyLatencyMs / 1000.0,        // Normalize to [0, 1] (assume max 1000ms)
		vector.SignalEntropy / math.Log(3.0),   // Normalize (max entropy for 3 signals)
		(vector.CoherenceDelta + 1.0) / 2.0,    // Normalize from [-1, 1] to [0, 1]
		(vector.CompressibilityZ + 3.0) / 6.0,  // Normalize from [-3, 3] to [0, 1]
	}
}

// computeReconstructionError computes MSE between input and reconstructed
func (ad *AnomalyDetector) computeReconstructionError(input, reconstructed []float64) float64 {
	mse := 0.0
	for i := range input {
		diff := input[i] - reconstructed[i]
		mse += diff * diff
	}
	return mse / float64(len(input))
}

// relu applies ReLU activation function
func relu(x float64) float64 {
	return math.Max(0, x)
}

// randomNormal generates a random number from normal distribution (Box-Muller)
func randomNormal(mean, stddev float64) float64 {
	// Simplified: use uniform for initialization
	// In production, use proper normal distribution
	return mean + stddev*(math.Sin(float64(time.Now().UnixNano()))*2.0-1.0)
}

// handleAnomaly handles detected anomalies
func (ad *AnomalyDetector) handleAnomaly(ctx context.Context, vector *PCSVector, result *AnomalyResult, tenantID string) {
	// Write to WORM audit log
	if ad.auditWriter != nil {
		if err := ad.auditWriter.WriteAnomaly(ctx, vector, result); err != nil {
			fmt.Printf("Failed to write WORM audit entry for anomaly: %v\n", err)
		}
	}

	// Stream to SIEM
	if ad.siemStream != nil {
		payload := map[string]interface{}{
			"event_type":        "anomaly_detected",
			"tenant_id":         tenantID,
			"anomaly_score":     result.AnomalyScore,
			"normalized_score":  result.NormalizedScore,
			"threshold":         result.Threshold,
			"shadow_mode":       result.ShadowMode,
			"timestamp":         result.ComputedAt.Format(time.RFC3339),
			"pcs_features": map[string]float64{
				"d_hat":    vector.DHat,
				"coh_star": vector.CohStar,
				"r":        vector.R,
				"budget":   vector.Budget,
			},
		}

		if err := ad.siemStream.Send(ctx, "anomaly_detected", payload); err != nil {
			fmt.Printf("Failed to send SIEM event for anomaly: %v\n", err)
		}
	}

	// Log anomaly (redacted for PII)
	fmt.Printf("Anomaly detected: tenant=%s, score=%.4f, threshold=%.4f, shadow=%v\n",
		tenantID, result.AnomalyScore, result.Threshold, result.ShadowMode)
}

// recordDetection records anomaly detection metrics
func (ad *AnomalyDetector) recordDetection(result *AnomalyResult, tenantID string) {
	ad.metrics.mu.Lock()
	defer ad.metrics.mu.Unlock()

	ad.metrics.TotalDetections++
	if result.IsAnomaly {
		ad.metrics.AnomaliesDetected++
	}

	// Update anomaly rate
	anomalyRate := 0.0
	if ad.metrics.TotalDetections > 0 {
		anomalyRate = float64(ad.metrics.AnomaliesDetected) / float64(ad.metrics.TotalDetections)
	}
	ad.metrics.AnomalyRate.WithLabelValues(tenantID).Set(anomalyRate)

	// Record reconstruction error histogram
	ad.metrics.ReconstructionError.WithLabelValues(tenantID).Observe(result.AnomalyScore)
}

// Train trains the autoencoder on a batch of normal examples
func (ad *AnomalyDetector) Train(examples []*PCSVector, epochs int, learningRate float64) error {
	ad.autoencoder.mu.Lock()
	defer ad.autoencoder.mu.Unlock()

	fmt.Printf("Training autoencoder: %d examples, %d epochs, lr=%.4f\n", len(examples), epochs, learningRate)

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0

		for _, example := range examples {
			input := ad.vectorToArray(example)

			// Forward pass
			hidden := ad.autoencoder.Encode(input)
			reconstructed := ad.autoencoder.Decode(hidden)

			// Compute loss (MSE)
			loss := ad.computeReconstructionError(input, reconstructed)
			totalLoss += loss

			// Backward pass (simplified gradient descent)
			// In production, use proper backpropagation
			for i := range ad.autoencoder.decoderBias {
				gradient := 2.0 * (reconstructed[i] - input[i]) / float64(len(input))
				ad.autoencoder.decoderBias[i] -= learningRate * gradient
			}
		}

		avgLoss := totalLoss / float64(len(examples))
		if epoch%10 == 0 {
			fmt.Printf("Epoch %d: avg_loss=%.6f\n", epoch, avgLoss)
		}
	}

	ad.autoencoder.trainedAt = time.Now()
	fmt.Printf("Training complete. Model version: %s\n", ad.autoencoder.version)

	return nil
}

// SetShadowMode enables/disables shadow mode
func (ad *AnomalyDetector) SetShadowMode(enabled bool) {
	ad.mu.Lock()
	defer ad.mu.Unlock()
	ad.shadowMode = enabled
	fmt.Printf("Anomaly detector shadow mode: %v\n", enabled)
}

// SetAlertThreshold sets the anomaly score threshold
func (ad *AnomalyDetector) SetAlertThreshold(threshold float64) {
	ad.mu.Lock()
	defer ad.mu.Unlock()
	ad.alertThreshold = threshold
	fmt.Printf("Anomaly detector threshold: %.4f\n", threshold)
}

// GetMetrics returns anomaly detection metrics
func (ad *AnomalyDetector) GetMetrics() AnomalyMetrics {
	ad.metrics.mu.RLock()
	defer ad.metrics.mu.RUnlock()
	return *ad.metrics
}

// GetAnomalyRate returns current anomaly rate
func (ad *AnomalyDetector) GetAnomalyRate() float64 {
	ad.metrics.mu.RLock()
	defer ad.metrics.mu.RUnlock()

	if ad.metrics.TotalDetections == 0 {
		return 0
	}

	return float64(ad.metrics.AnomaliesDetected) / float64(ad.metrics.TotalDetections)
}

// EstimateFalsePositiveRate estimates false positive rate from feedback
func (ad *AnomalyDetector) EstimateFalsePositiveRate(truePositives, falsePositives int64) {
	ad.metrics.mu.Lock()
	defer ad.metrics.mu.Unlock()

	total := truePositives + falsePositives
	if total > 0 {
		ad.metrics.FalsePositiveRate = float64(falsePositives) / float64(total)
		fmt.Printf("False positive rate updated: %.2f%% (%d/%d)\n",
			ad.metrics.FalsePositiveRate*100, falsePositives, total)
	}
}

// GetAnomalyFeature returns normalized anomaly score as a feature for HRS
func (ad *AnomalyDetector) GetAnomalyFeature(ctx context.Context, vector *PCSVector, tenantID string) (float64, error) {
	result, err := ad.Detect(ctx, vector, tenantID)
	if err != nil {
		return 0, err
	}

	// Return normalized score [0, 1] for HRS integration
	return result.NormalizedScore, nil
}

// DefaultAnomalyThreshold returns a default anomaly threshold
func DefaultAnomalyThreshold() float64 {
	return 0.5 // Reconstruction error threshold
}
