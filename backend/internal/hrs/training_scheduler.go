package hrs

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// TrainingScheduler manages periodic model retraining (Phase 8 WP1)
type TrainingScheduler struct {
	mu           sync.RWMutex
	schedule     *Schedule
	pipeline     *TrainingPipeline
	registry     *ModelRegistry
	driftMonitor *DriftMonitor
	running      bool
	stopCh       chan struct{}
	metrics      *SchedulerMetrics
}

// Schedule defines when training runs
type Schedule struct {
	Frequency    time.Duration // Daily, weekly, etc.
	TimeOfDay    time.Time     // Preferred time to run
	MaxDuration  time.Duration // Max allowed training time
	AutoDeploy   bool          // Auto-activate if metrics pass thresholds
}

// DriftMonitor detects feature and performance drift
type DriftMonitor struct {
	mu                   sync.RWMutex
	baselineFeatures     map[string]float64 // feature_name → baseline_mean
	currentFeatures      map[string]float64
	baselineAUC          float64
	currentAUC           float64
	driftThreshold       float64 // K-S test p-value threshold
	performanceThreshold float64 // AUC drop threshold
	alerts               []DriftAlert
}

// DriftAlert represents detected drift
type DriftAlert struct {
	Timestamp   time.Time
	Type        string  // "feature_drift", "performance_drift"
	Severity    string  // "warning", "critical"
	Description string
	Metric      float64
	Threshold   float64
}

// SchedulerMetrics tracks scheduler performance
type SchedulerMetrics struct {
	mu                  sync.RWMutex
	TotalRuns           int64
	SuccessfulRuns      int64
	FailedRuns          int64
	SkippedRuns         int64 // Skipped due to drift/time constraints
	LastRunTime         time.Time
	LastRunDuration     time.Duration
	DriftAlertsTriggered int64
}

// NewTrainingScheduler creates a new training scheduler
func NewTrainingScheduler() *TrainingScheduler {
	return &TrainingScheduler{
		schedule: &Schedule{
			Frequency:   24 * time.Hour, // Daily
			MaxDuration: 4 * time.Hour,
			AutoDeploy:  false, // Require manual activation by default
		},
		driftMonitor: NewDriftMonitor(),
		stopCh:       make(chan struct{}),
		metrics:      &SchedulerMetrics{},
	}
}

// NewDriftMonitor creates a new drift monitor
func NewDriftMonitor() *DriftMonitor {
	return &DriftMonitor{
		baselineFeatures:     make(map[string]float64),
		currentFeatures:      make(map[string]float64),
		driftThreshold:       0.05, // p-value < 0.05 indicates drift
		performanceThreshold: 0.05, // AUC drop > 0.05 triggers alert
		alerts:               []DriftAlert{},
	}
}

// Start starts the training scheduler
func (ts *TrainingScheduler) Start(ctx context.Context, pipeline *TrainingPipeline, registry *ModelRegistry) error {
	ts.mu.Lock()
	if ts.running {
		ts.mu.Unlock()
		return fmt.Errorf("scheduler already running")
	}
	ts.running = true
	ts.pipeline = pipeline
	ts.registry = registry
	ts.mu.Unlock()

	fmt.Printf("Training scheduler started: frequency=%v\n", ts.schedule.Frequency)

	ticker := time.NewTicker(ts.schedule.Frequency)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			ts.Stop()
			return ctx.Err()
		case <-ts.stopCh:
			return nil
		case <-ticker.C:
			if err := ts.runScheduledTraining(ctx); err != nil {
				fmt.Printf("Scheduled training failed: %v\n", err)
			}
		}
	}
}

// Stop stops the scheduler
func (ts *TrainingScheduler) Stop() {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	if !ts.running {
		return
	}

	close(ts.stopCh)
	ts.running = false
	fmt.Println("Training scheduler stopped")
}

// runScheduledTraining executes a scheduled training run
func (ts *TrainingScheduler) runScheduledTraining(ctx context.Context) error {
	startTime := time.Now()

	ts.metrics.mu.Lock()
	ts.metrics.TotalRuns++
	ts.metrics.mu.Unlock()

	// Check drift before training
	driftDetected, alerts := ts.driftMonitor.CheckDrift()
	if driftDetected {
		fmt.Printf("Drift detected: %d alerts\n", len(alerts))
		for _, alert := range alerts {
			fmt.Printf("  - %s: %s (metric=%.3f, threshold=%.3f)\n",
				alert.Type, alert.Description, alert.Metric, alert.Threshold)
		}

		ts.metrics.mu.Lock()
		ts.metrics.DriftAlertsTriggered += int64(len(alerts))
		ts.metrics.mu.Unlock()
	}

	// Prepare training data
	endTime := time.Now()
	startTrainData := endTime.Add(-30 * 24 * time.Hour) // Last 30 days
	dataset, err := ts.pipeline.PrepareTrainingData(ctx, startTrainData, endTime)
	if err != nil {
		ts.metrics.mu.Lock()
		ts.metrics.FailedRuns++
		ts.metrics.mu.Unlock()
		return fmt.Errorf("failed to prepare training data: %w", err)
	}

	// Train model
	trainedModel, err := ts.pipeline.TrainModel(ctx, dataset)
	if err != nil {
		ts.metrics.mu.Lock()
		ts.metrics.FailedRuns++
		ts.metrics.mu.Unlock()
		return fmt.Errorf("failed to train model: %w", err)
	}

	// Register model
	registeredModel, err := ts.registry.RegisterModel(trainedModel)
	if err != nil {
		ts.metrics.mu.Lock()
		ts.metrics.FailedRuns++
		ts.metrics.mu.Unlock()
		return fmt.Errorf("failed to register model: %w", err)
	}

	// Auto-deploy if configured and metrics pass
	if ts.schedule.AutoDeploy && ts.shouldAutoDeploy(trainedModel) {
		if err := ts.registry.ActivateModel(registeredModel.Version); err != nil {
			fmt.Printf("Auto-deployment failed: %v\n", err)
		} else {
			fmt.Printf("Auto-deployed model: %s\n", registeredModel.Version)
		}
	}

	duration := time.Since(startTime)

	ts.metrics.mu.Lock()
	ts.metrics.SuccessfulRuns++
	ts.metrics.LastRunTime = startTime
	ts.metrics.LastRunDuration = duration
	ts.metrics.mu.Unlock()

	fmt.Printf("Scheduled training completed: duration=%v, model=%s\n", duration, trainedModel.Version)

	return nil
}

// shouldAutoDeploy determines if model should be auto-deployed
func (ts *TrainingScheduler) shouldAutoDeploy(model *TrainedModel) bool {
	// Check AUC threshold
	if model.Metrics.AUC < 0.82 {
		fmt.Printf("Auto-deploy blocked: AUC %.3f < 0.82\n", model.Metrics.AUC)
		return false
	}

	// Check for significant performance degradation
	activeModel, err := ts.registry.GetActiveModel()
	if err == nil && activeModel.ModelCard != nil {
		aucDrop := activeModel.ModelCard.Metrics.AUC - model.Metrics.AUC
		if aucDrop > 0.05 {
			fmt.Printf("Auto-deploy blocked: AUC dropped by %.3f\n", aucDrop)
			return false
		}
	}

	return true
}

// CheckDrift checks for feature and performance drift
func (dm *DriftMonitor) CheckDrift() (bool, []DriftAlert) {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	alerts := []DriftAlert{}

	// Feature drift check (placeholder - in production, use K-S test)
	for featureName, baselineValue := range dm.baselineFeatures {
		if currentValue, ok := dm.currentFeatures[featureName]; ok {
			drift := (currentValue - baselineValue) / baselineValue
			if drift > 0.20 || drift < -0.20 {
				alerts = append(alerts, DriftAlert{
					Timestamp:   time.Now(),
					Type:        "feature_drift",
					Severity:    "warning",
					Description: fmt.Sprintf("Feature %s drifted by %.1f%%", featureName, drift*100),
					Metric:      drift,
					Threshold:   0.20,
				})
			}
		}
	}

	// Performance drift check
	if dm.baselineAUC > 0 && dm.currentAUC > 0 {
		aucDrop := dm.baselineAUC - dm.currentAUC
		if aucDrop > dm.performanceThreshold {
			alerts = append(alerts, DriftAlert{
				Timestamp:   time.Now(),
				Type:        "performance_drift",
				Severity:    "critical",
				Description: fmt.Sprintf("AUC dropped from %.3f to %.3f", dm.baselineAUC, dm.currentAUC),
				Metric:      aucDrop,
				Threshold:   dm.performanceThreshold,
			})
		}
	}

	dm.alerts = append(dm.alerts, alerts...)

	return len(alerts) > 0, alerts
}

// SetBaseline sets baseline metrics for drift detection
func (dm *DriftMonitor) SetBaseline(features map[string]float64, auc float64) {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	dm.baselineFeatures = features
	dm.baselineAUC = auc

	fmt.Printf("Drift monitor baseline set: %d features, AUC=%.3f\n", len(features), auc)
}

// UpdateCurrent updates current metrics for drift detection
func (dm *DriftMonitor) UpdateCurrent(features map[string]float64, auc float64) {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	dm.currentFeatures = features
	dm.currentAUC = auc
}

// GetAlerts returns recent drift alerts
func (dm *DriftMonitor) GetAlerts(since time.Time) []DriftAlert {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	recent := []DriftAlert{}
	for _, alert := range dm.alerts {
		if alert.Timestamp.After(since) {
			recent = append(recent, alert)
		}
	}

	return recent
}

// GetMetrics returns scheduler metrics
func (ts *TrainingScheduler) GetMetrics() SchedulerMetrics {
	ts.metrics.mu.RLock()
	defer ts.metrics.mu.RUnlock()
	return *ts.metrics
}
