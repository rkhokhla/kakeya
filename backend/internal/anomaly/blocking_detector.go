package anomaly

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// BlockingDetector promotes Phase 8 VAE detector to blocking mode
// Phase 9 WP3: Dual-threshold blocking with active learning
type BlockingDetector struct {
	mu sync.RWMutex

	// Phase 8 VAE detector
	vaeDetector *DetectorV2

	// Dual thresholds
	blockThreshold   float64 // e.g., 0.9 (high confidence anomalies)
	guardThreshold   float64 // e.g., 0.5 (medium confidence → guardrail)
	uncertaintyMax   float64 // e.g., 0.2 (max uncertainty for blocking)

	// Mode: "shadow", "guardrail", "blocking"
	mode string

	// Active learning queue
	activeLearnQueue *ActiveLearningQueue
	wormLogger       WORMLogger
	siemStreamer     SIEMStreamer

	metrics *BlockingMetrics
}

// BlockingMetrics tracks blocking detector performance
type BlockingMetrics struct {
	mu                   sync.RWMutex
	TotalDetections      int64
	BlockedCount         int64
	GuardrailCount       int64
	AcceptedCount        int64
	UncertaintyOverrides int64
	ActiveLearnQueued    int64
	FalsePositiveRate    float64
	TruePositiveRate     float64
	LastUpdate           time.Time
}

// ActiveLearningQueue prioritizes samples for human labeling
type ActiveLearningQueue struct {
	mu     sync.RWMutex
	queue  []ActiveLearningSample
	maxSize int
}

// ActiveLearningSample represents a sample for labeling
type ActiveLearningSample struct {
	PCSID          string
	Features       []float64
	AnomalyScore   float64
	Uncertainty    float64
	Timestamp      time.Time
	PredictedLabel string // "anomaly", "normal"
	TrueLabel      string // Set by human after review
	Priority       float64 // Higher = more important to label
}

// WORMLogger interface (from Phase 3)
type WORMLogger interface {
	LogEvent(eventType string, details map[string]interface{}) error
}

// SIEMStreamer interface (from Phase 6)
type SIEMStreamer interface {
	StreamEvent(eventType string, severity string, details map[string]interface{}) error
}

// NewBlockingDetector creates a blocking-mode anomaly detector
func NewBlockingDetector(vaeDetector *DetectorV2, wormLogger WORMLogger, siemStreamer SIEMStreamer) *BlockingDetector {
	return &BlockingDetector{
		vaeDetector:      vaeDetector,
		blockThreshold:   0.9,  // High confidence anomalies
		guardThreshold:   0.5,  // Medium confidence
		uncertaintyMax:   0.2,  // Max uncertainty for blocking
		mode:             "blocking",
		activeLearnQueue: &ActiveLearningQueue{
			queue:   []ActiveLearningSample{},
			maxSize: 1000,
		},
		wormLogger:   wormLogger,
		siemStreamer: siemStreamer,
		metrics:      &BlockingMetrics{},
	}
}

// Detect runs anomaly detection with dual-threshold blocking
func (bd *BlockingDetector) Detect(ctx context.Context, pcsID string, features []float64) (string, float64, error) {
	bd.metrics.mu.Lock()
	bd.metrics.TotalDetections++
	bd.metrics.mu.Unlock()

	// Run VAE detection
	score, uncertainty, err := bd.vaeDetector.DetectWithUncertainty(ctx, features)
	if err != nil {
		return "error", 0, fmt.Errorf("VAE detection failed: %w", err)
	}

	// Decision logic based on dual thresholds
	var decision string
	var action string

	if score >= bd.blockThreshold && uncertainty <= bd.uncertaintyMax {
		// High confidence anomaly with low uncertainty → BLOCK
		decision = "block"
		action = "reject_pcs"

		bd.metrics.mu.Lock()
		bd.metrics.BlockedCount++
		bd.metrics.mu.Unlock()

		// Log to WORM
		bd.wormLogger.LogEvent("anomaly_blocked", map[string]interface{}{
			"pcs_id":      pcsID,
			"score":       score,
			"uncertainty": uncertainty,
			"timestamp":   time.Now(),
		})

		// Stream to SIEM
		bd.siemStreamer.StreamEvent("anomaly_detection", "critical", map[string]interface{}{
			"pcs_id":   pcsID,
			"score":    score,
			"decision": "blocked",
		})

	} else if score >= bd.guardThreshold {
		// Medium confidence or high uncertainty → GUARDRAIL (escalate to HRS/ensemble)
		decision = "guardrail"
		action = "escalate"

		bd.metrics.mu.Lock()
		bd.metrics.GuardrailCount++
		bd.metrics.mu.Unlock()

		// Add to active learning queue (edge case)
		bd.queueForActiveLearning(pcsID, features, score, uncertainty, "anomaly")

	} else {
		// Low score → ACCEPT
		decision = "accept"
		action = "continue"

		bd.metrics.mu.Lock()
		bd.metrics.AcceptedCount++
		bd.metrics.mu.Unlock()
	}

	bd.metrics.mu.Lock()
	bd.metrics.LastUpdate = time.Now()
	bd.metrics.mu.Unlock()

	return decision, score, nil
}

// queueForActiveLearning adds a sample to the active learning queue
func (bd *BlockingDetector) queueForActiveLearning(pcsID string, features []float64, score, uncertainty float64, predictedLabel string) {
	// Calculate priority: favor high uncertainty or near-threshold scores
	priority := uncertainty
	if math.Abs(score-bd.blockThreshold) < 0.1 {
		priority += 0.5 // Boost priority for borderline cases
	}

	sample := ActiveLearningSample{
		PCSID:          pcsID,
		Features:       features,
		AnomalyScore:   score,
		Uncertainty:    uncertainty,
		Timestamp:      time.Now(),
		PredictedLabel: predictedLabel,
		Priority:       priority,
	}

	bd.activeLearnQueue.Add(sample)

	bd.metrics.mu.Lock()
	bd.metrics.ActiveLearnQueued++
	bd.metrics.mu.Unlock()
}

// Add adds a sample to the active learning queue (prioritized)
func (alq *ActiveLearningQueue) Add(sample ActiveLearningSample) {
	alq.mu.Lock()
	defer alq.mu.Unlock()

	alq.queue = append(alq.queue, sample)

	// Sort by priority (descending)
	for i := len(alq.queue) - 1; i > 0; i-- {
		if alq.queue[i].Priority > alq.queue[i-1].Priority {
			alq.queue[i], alq.queue[i-1] = alq.queue[i-1], alq.queue[i]
		}
	}

	// Trim to max size
	if len(alq.queue) > alq.maxSize {
		alq.queue = alq.queue[:alq.maxSize]
	}
}

// GetTopSamples returns top N samples for labeling
func (alq *ActiveLearningQueue) GetTopSamples(n int) []ActiveLearningSample {
	alq.mu.RLock()
	defer alq.mu.RUnlock()

	limit := n
	if limit > len(alq.queue) {
		limit = len(alq.queue)
	}

	samples := make([]ActiveLearningSample, limit)
	copy(samples, alq.queue[:limit])
	return samples
}

// SubmitLabel updates a sample with human label
func (alq *ActiveLearningQueue) SubmitLabel(pcsID string, trueLabel string) error {
	alq.mu.Lock()
	defer alq.mu.Unlock()

	for i, sample := range alq.queue {
		if sample.PCSID == pcsID {
			alq.queue[i].TrueLabel = trueLabel
			return nil
		}
	}

	return fmt.Errorf("sample not found: %s", pcsID)
}

// GetMetrics returns blocking detector metrics
func (bd *BlockingDetector) GetMetrics() *BlockingMetrics {
	bd.metrics.mu.RLock()
	defer bd.metrics.mu.RUnlock()

	return &BlockingMetrics{
		TotalDetections:      bd.metrics.TotalDetections,
		BlockedCount:         bd.metrics.BlockedCount,
		GuardrailCount:       bd.metrics.GuardrailCount,
		AcceptedCount:        bd.metrics.AcceptedCount,
		UncertaintyOverrides: bd.metrics.UncertaintyOverrides,
		ActiveLearnQueued:    bd.metrics.ActiveLearnQueued,
		FalsePositiveRate:    bd.metrics.FalsePositiveRate,
		TruePositiveRate:     bd.metrics.TruePositiveRate,
		LastUpdate:           bd.metrics.LastUpdate,
	}
}

// UpdatePerformance recalculates FPR/TPR from labeled data
func (bd *BlockingDetector) UpdatePerformance() error {
	// Get labeled samples from active learning queue
	samples := bd.activeLearnQueue.GetTopSamples(1000)

	truePositives := 0
	falsePositives := 0
	trueNegatives := 0
	falseNegatives := 0

	for _, sample := range samples {
		if sample.TrueLabel == "" {
			continue // Skip unlabeled
		}

		predicted := sample.PredictedLabel
		actual := sample.TrueLabel

		if predicted == "anomaly" && actual == "anomaly" {
			truePositives++
		} else if predicted == "anomaly" && actual == "normal" {
			falsePositives++
		} else if predicted == "normal" && actual == "normal" {
			trueNegatives++
		} else if predicted == "normal" && actual == "anomaly" {
			falseNegatives++
		}
	}

	// Calculate FPR and TPR
	if falsePositives+trueNegatives > 0 {
		bd.metrics.FalsePositiveRate = float64(falsePositives) / float64(falsePositives+trueNegatives)
	}

	if truePositives+falseNegatives > 0 {
		bd.metrics.TruePositiveRate = float64(truePositives) / float64(truePositives+falseNegatives)
	}

	return nil
}
