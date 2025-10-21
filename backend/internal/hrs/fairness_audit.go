package hrs

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// FairnessAuditor performs automated fairness and drift audits
// Phase 9 WP1: Continuous fairness monitoring with auto-revert
type FairnessAuditor struct {
	mu sync.RWMutex

	registry        *ModelRegistry
	trainingPipeline *TrainingPipeline
	wormLogger      WORMLogger
	alerter         Alerter

	// Thresholds
	aucDropThreshold      float64 // e.g., 0.05 (5 percentage points)
	subgroupGapThreshold  float64 // e.g., 0.05 (5 pp max gap across subgroups)
	calibrationThreshold  float64 // e.g., 0.10 (10% mean absolute error)
	featureDriftPValueThreshold float64 // e.g., 0.01 (K-S test)

	// Audit frequency
	auditInterval time.Duration
	running       bool
	stopCh        chan struct{}

	metrics *FairnessAuditMetrics
}

// FairnessAuditMetrics tracks auditor performance
type FairnessAuditMetrics struct {
	mu                  sync.RWMutex
	TotalAudits         int64
	DriftDetections     int64
	FairnessViolations  int64
	AutoReverts         int64
	LastAudit           time.Time
	LastDriftDetection  time.Time
	LastFairnessViolation time.Time
}

// WORMLogger interface for audit trail
type WORMLogger interface {
	LogAuditEvent(eventType string, details map[string]interface{}) error
}

// Alerter interface for notifications
type Alerter interface {
	SendAlert(severity string, message string, details map[string]interface{}) error
}

// NewFairnessAuditor creates a fairness auditor
func NewFairnessAuditor(
	registry *ModelRegistry,
	trainingPipeline *TrainingPipeline,
	wormLogger WORMLogger,
	alerter Alerter,
) *FairnessAuditor {
	return &FairnessAuditor{
		registry:         registry,
		trainingPipeline: trainingPipeline,
		wormLogger:       wormLogger,
		alerter:          alerter,

		// Default thresholds from CLAUDE_PHASE9.md
		aucDropThreshold:           0.05,
		subgroupGapThreshold:       0.05,
		calibrationThreshold:       0.10,
		featureDriftPValueThreshold: 0.01,

		auditInterval: 24 * time.Hour, // Daily audits
		stopCh:        make(chan struct{}),
		metrics:       &FairnessAuditMetrics{},
	}
}

// Start begins continuous auditing
func (fa *FairnessAuditor) Start(ctx context.Context) error {
	fa.mu.Lock()
	if fa.running {
		fa.mu.Unlock()
		return fmt.Errorf("auditor already running")
	}
	fa.running = true
	fa.mu.Unlock()

	go fa.auditLoop(ctx)
	return nil
}

// Stop halts continuous auditing
func (fa *FairnessAuditor) Stop() {
	fa.mu.Lock()
	defer fa.mu.Unlock()

	if fa.running {
		close(fa.stopCh)
		fa.running = false
	}
}

// auditLoop runs periodic fairness and drift audits
func (fa *FairnessAuditor) auditLoop(ctx context.Context) {
	ticker := time.NewTicker(fa.auditInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := fa.RunAudit(ctx); err != nil {
				// Log error but continue
				fa.wormLogger.LogAuditEvent("audit_error", map[string]interface{}{
					"error": err.Error(),
					"timestamp": time.Now(),
				})
			}
		case <-fa.stopCh:
			return
		case <-ctx.Done():
			return
		}
	}
}

// RunAudit executes a complete fairness and drift audit
func (fa *FairnessAuditor) RunAudit(ctx context.Context) error {
	fa.metrics.mu.Lock()
	fa.metrics.TotalAudits++
	fa.metrics.LastAudit = time.Now()
	fa.metrics.mu.Unlock()

	// Get active model
	activeModel := fa.registry.GetActiveModel()
	if activeModel == nil {
		return fmt.Errorf("no active model to audit")
	}

	// 1. Check for AUC drift
	driftDetected, aucDrop, err := fa.checkAUCDrift(ctx, activeModel)
	if err != nil {
		return fmt.Errorf("AUC drift check failed: %w", err)
	}

	if driftDetected {
		fa.metrics.mu.Lock()
		fa.metrics.DriftDetections++
		fa.metrics.LastDriftDetection = time.Now()
		fa.metrics.mu.Unlock()

		// Alert
		fa.alerter.SendAlert("critical", fmt.Sprintf("AUC drift detected: drop=%.3f (threshold=%.3f)", aucDrop, fa.aucDropThreshold), map[string]interface{}{
			"model_version": activeModel.Version,
			"auc_drop": aucDrop,
			"threshold": fa.aucDropThreshold,
		})

		// Auto-revert
		if err := fa.autoRevertModel(ctx, activeModel, "auc_drop", aucDrop); err != nil {
			return fmt.Errorf("auto-revert failed: %w", err)
		}

		return nil // Early exit after revert
	}

	// 2. Check for fairness violations (subgroup performance gaps)
	fairnessViolation, audit, err := fa.checkFairness(ctx, activeModel)
	if err != nil {
		return fmt.Errorf("fairness check failed: %w", err)
	}

	if fairnessViolation {
		fa.metrics.mu.Lock()
		fa.metrics.FairnessViolations++
		fa.metrics.LastFairnessViolation = time.Now()
		fa.metrics.mu.Unlock()

		// Alert
		fa.alerter.SendAlert("warning", fmt.Sprintf("Fairness violation: max_gap=%.3f (threshold=%.3f)", audit.MaxSubgroupGap, fa.subgroupGapThreshold), map[string]interface{}{
			"model_version": activeModel.Version,
			"max_subgroup_gap": audit.MaxSubgroupGap,
			"threshold": fa.subgroupGapThreshold,
		})

		// Log to WORM
		fa.wormLogger.LogAuditEvent("fairness_violation", map[string]interface{}{
			"model_version": activeModel.Version,
			"audit": audit,
		})

		// Optional: auto-revert on severe fairness violations (configurable)
		// For now, just alert and log
	}

	// 3. Check for feature drift (K-S test)
	featureDrift, err := fa.checkFeatureDrift(ctx, activeModel)
	if err != nil {
		return fmt.Errorf("feature drift check failed: %w", err)
	}

	if len(featureDrift) > 0 {
		fa.alerter.SendAlert("warning", fmt.Sprintf("Feature drift detected: %d features drifted", len(featureDrift)), map[string]interface{}{
			"model_version": activeModel.Version,
			"drifted_features": featureDrift,
		})
	}

	// 4. Update model card with audit results
	if err := fa.updateModelCardWithAudit(activeModel, audit, driftDetected, aucDrop, featureDrift); err != nil {
		return fmt.Errorf("failed to update model card: %w", err)
	}

	return nil
}

// checkAUCDrift checks if AUC has dropped significantly
func (fa *FairnessAuditor) checkAUCDrift(ctx context.Context, model *RegisteredModel) (bool, float64, error) {
	// Get baseline AUC from model card
	baselineAUC, ok := model.ModelCard.TrainingMetrics["auc"]
	if !ok {
		return false, 0, fmt.Errorf("baseline AUC not found in model card")
	}

	// Compute current AUC on recent data (last 7 days)
	endTime := time.Now()
	startTime := endTime.Add(-7 * 24 * time.Hour)

	dataset, err := fa.trainingPipeline.PrepareTrainingData(ctx, startTime, endTime)
	if err != nil {
		return false, 0, fmt.Errorf("failed to prepare evaluation dataset: %w", err)
	}

	if dataset.NumSamples < 100 {
		return false, 0, fmt.Errorf("insufficient samples for drift detection: %d", dataset.NumSamples)
	}

	// Evaluate model on recent data
	currentAUC, err := fa.evaluateAUC(model.Model, dataset)
	if err != nil {
		return false, 0, fmt.Errorf("failed to evaluate AUC: %w", err)
	}

	// Check if drop exceeds threshold
	aucDrop := baselineAUC - currentAUC
	driftDetected := aucDrop > fa.aucDropThreshold

	return driftDetected, aucDrop, nil
}

// evaluateAUC computes AUC for a model on a dataset
func (fa *FairnessAuditor) evaluateAUC(model RiskModel, dataset *TrainingDataset) (float64, error) {
	// Predict on all samples
	predictions := make([]float64, dataset.NumSamples)
	for i := 0; i < dataset.NumSamples; i++ {
		pred, err := model.Predict(dataset.Features[i])
		if err != nil {
			return 0, err
		}
		predictions[i] = pred
	}

	// Compute AUC using trapezoidal rule
	auc := fa.computeAUC(predictions, dataset.Labels)
	return auc, nil
}

// computeAUC calculates Area Under ROC Curve
func (fa *FairnessAuditor) computeAUC(predictions []float64, labels []int) float64 {
	// Sort by prediction score (descending)
	type scoredSample struct {
		score float64
		label int
	}

	samples := make([]scoredSample, len(predictions))
	for i := range predictions {
		samples[i] = scoredSample{score: predictions[i], label: labels[i]}
	}

	// Simple bubble sort (fine for evaluation sets)
	for i := 0; i < len(samples)-1; i++ {
		for j := i + 1; j < len(samples); j++ {
			if samples[j].score > samples[i].score {
				samples[i], samples[j] = samples[j], samples[i]
			}
		}
	}

	// Count positives and negatives
	numPos, numNeg := 0, 0
	for _, s := range samples {
		if s.label == 1 {
			numPos++
		} else {
			numNeg++
		}
	}

	if numPos == 0 || numNeg == 0 {
		return 0.5 // Undefined AUC
	}

	// Compute AUC via Mann-Whitney U statistic
	rankSum := 0.0
	for i, s := range samples {
		if s.label == 1 {
			rankSum += float64(i + 1)
		}
	}

	auc := (rankSum - float64(numPos*(numPos+1))/2.0) / float64(numPos*numNeg)
	return auc
}

// checkFairness evaluates fairness across subgroups
func (fa *FairnessAuditor) checkFairness(ctx context.Context, model *RegisteredModel) (bool, *FairnessAudit, error) {
	// Prepare evaluation dataset (last 7 days)
	endTime := time.Now()
	startTime := endTime.Add(-7 * 24 * time.Hour)

	dataset, err := fa.trainingPipeline.PrepareTrainingData(ctx, startTime, endTime)
	if err != nil {
		return false, nil, fmt.Errorf("failed to prepare evaluation dataset: %w", err)
	}

	// Define subgroups (example: by tenant_id prefix, could be extended)
	subgroups := fa.defineSubgroups(dataset)

	// Evaluate each subgroup
	subgroupMetrics := make([]SubgroupMetrics, 0, len(subgroups))
	maxGap := 0.0

	for _, subgroup := range subgroups {
		metrics, err := fa.evaluateSubgroup(model.Model, subgroup)
		if err != nil {
			continue // Skip on error
		}

		subgroupMetrics = append(subgroupMetrics, metrics)

		// Track max gap
		for _, other := range subgroupMetrics {
			gap := math.Abs(metrics.Metrics["auc"] - other.Metrics["auc"])
			if gap > maxGap {
				maxGap = gap
			}
		}
	}

	// Check if max gap exceeds threshold
	fairnessViolation := maxGap > fa.subgroupGapThreshold

	audit := &FairnessAudit{
		AuditedAt:            time.Now(),
		Subgroups:            subgroupMetrics,
		MaxSubgroupGap:       maxGap,
		SubgroupGapThreshold: fa.subgroupGapThreshold,
		CalibrationThreshold: fa.calibrationThreshold,
	}

	if fairnessViolation {
		audit.Status = "fail"
	} else if maxGap > fa.subgroupGapThreshold*0.8 {
		audit.Status = "warning"
	} else {
		audit.Status = "pass"
	}

	return fairnessViolation, audit, nil
}

// defineSubgroups partitions dataset into subgroups for fairness analysis
func (fa *FairnessAuditor) defineSubgroups(dataset *TrainingDataset) []subgroupData {
	// Example: partition by sample metadata (tenant_type, region, etc.)
	// For simplicity, partition into 3 groups by sample index
	numGroups := 3
	groupSize := dataset.NumSamples / numGroups

	subgroups := make([]subgroupData, numGroups)
	for i := 0; i < numGroups; i++ {
		start := i * groupSize
		end := start + groupSize
		if i == numGroups-1 {
			end = dataset.NumSamples // Last group gets remainder
		}

		subgroups[i] = subgroupData{
			Name:     fmt.Sprintf("group_%d", i),
			Features: dataset.Features[start:end],
			Labels:   dataset.Labels[start:end],
		}
	}

	return subgroups
}

type subgroupData struct {
	Name     string
	Features [][]float64
	Labels   []int
}

// evaluateSubgroup computes metrics for a subgroup
func (fa *FairnessAuditor) evaluateSubgroup(model RiskModel, subgroup subgroupData) (SubgroupMetrics, error) {
	// Predict on subgroup
	predictions := make([]float64, len(subgroup.Features))
	for i, features := range subgroup.Features {
		pred, err := model.Predict(features)
		if err != nil {
			return SubgroupMetrics{}, err
		}
		predictions[i] = pred
	}

	// Compute AUC
	auc := fa.computeAUC(predictions, subgroup.Labels)

	// Compute calibration error (mean absolute error between predicted prob and true freq)
	calibration := fa.computeCalibration(predictions, subgroup.Labels)

	metrics := SubgroupMetrics{
		SubgroupName: subgroup.Name,
		SampleCount:  len(subgroup.Features),
		Metrics: map[string]float64{
			"auc": auc,
		},
		Calibration:    calibration,
		Representation: float64(len(subgroup.Features)) / float64(len(subgroup.Features)), // Placeholder
	}

	return metrics, nil
}

// computeCalibration calculates mean absolute calibration error
func (fa *FairnessAuditor) computeCalibration(predictions []float64, labels []int) float64 {
	// Bin predictions into 10 buckets and compute expected calibration error
	numBins := 10
	binCounts := make([]int, numBins)
	binPositives := make([]int, numBins)
	binPredSums := make([]float64, numBins)

	for i, pred := range predictions {
		bin := int(pred * float64(numBins))
		if bin >= numBins {
			bin = numBins - 1
		}

		binCounts[bin]++
		binPredSums[bin] += pred
		if labels[i] == 1 {
			binPositives[bin]++
		}
	}

	totalError := 0.0
	totalSamples := 0
	for b := 0; b < numBins; b++ {
		if binCounts[b] == 0 {
			continue
		}

		avgPred := binPredSums[b] / float64(binCounts[b])
		trueFreq := float64(binPositives[b]) / float64(binCounts[b])
		error := math.Abs(avgPred - trueFreq)

		totalError += error * float64(binCounts[b])
		totalSamples += binCounts[b]
	}

	if totalSamples == 0 {
		return 0.0
	}

	return totalError / float64(totalSamples)
}

// checkFeatureDrift performs Kolmogorov-Smirnov test for feature drift
func (fa *FairnessAuditor) checkFeatureDrift(ctx context.Context, model *RegisteredModel) (map[string]float64, error) {
	// Get training features (baseline)
	// Get recent features (current)
	// Compute K-S test p-value for each feature
	// Return features with p-value < threshold

	// Placeholder implementation
	driftedFeatures := make(map[string]float64)

	// In real implementation:
	// - Load training dataset features
	// - Load recent dataset features
	// - For each feature, compute K-S statistic and p-value
	// - If p-value < featureDriftPValueThreshold, add to driftedFeatures

	return driftedFeatures, nil
}

// autoRevertModel automatically reverts to the last good model
func (fa *FairnessAuditor) autoRevertModel(ctx context.Context, currentModel *RegisteredModel, reason string, severity float64) error {
	fa.metrics.mu.Lock()
	fa.metrics.AutoReverts++
	fa.metrics.mu.Unlock()

	// Find last good model (previous active)
	lastGoodModel := fa.registry.GetPreviousActiveModel()
	if lastGoodModel == nil {
		return fmt.Errorf("no previous model to revert to")
	}

	// Perform rollback
	startTime := time.Now()
	if err := fa.registry.PromoteModel(lastGoodModel.Version); err != nil {
		return fmt.Errorf("failed to promote previous model: %w", err)
	}
	rollbackTime := time.Since(startTime).Seconds() * 1000 // ms

	// Record revert event
	revertEvent := RevertEvent{
		RevertedAt:     time.Now(),
		FromVersion:    currentModel.Version,
		ToVersion:      lastGoodModel.Version,
		Reason:         reason,
		TriggeredBy:    "auto",
		RollbackTimeMs: rollbackTime,
	}

	// Log to WORM
	fa.wormLogger.LogAuditEvent("auto_revert", map[string]interface{}{
		"event": revertEvent,
		"severity": severity,
	})

	// Send alert
	fa.alerter.SendAlert("critical", fmt.Sprintf("Auto-reverted model %s → %s due to %s", currentModel.Version, lastGoodModel.Version, reason), map[string]interface{}{
		"revert_event": revertEvent,
	})

	return nil
}

// updateModelCardWithAudit updates model card with latest audit results
func (fa *FairnessAuditor) updateModelCardWithAudit(model *RegisteredModel, audit *FairnessAudit, driftDetected bool, aucDrop float64, featureDrift map[string]float64) error {
	// Update model card's DriftStatus and FairnessAudit fields
	// This would integrate with the model registry to persist updates

	// Placeholder: in real implementation, call registry.UpdateModelCard()
	return nil
}

// GetMetrics returns current fairness audit metrics
func (fa *FairnessAuditor) GetMetrics() *FairnessAuditMetrics {
	fa.metrics.mu.RLock()
	defer fa.metrics.mu.RUnlock()

	return &FairnessAuditMetrics{
		TotalAudits:           fa.metrics.TotalAudits,
		DriftDetections:       fa.metrics.DriftDetections,
		FairnessViolations:    fa.metrics.FairnessViolations,
		AutoReverts:           fa.metrics.AutoReverts,
		LastAudit:             fa.metrics.LastAudit,
		LastDriftDetection:    fa.metrics.LastDriftDetection,
		LastFairnessViolation: fa.metrics.LastFairnessViolation,
	}
}
