package anomaly

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// Phase 8 WP4: Anomaly Detection v2 with VAE, clustering, and auto-thresholding

// DetectorV2 implements advanced anomaly detection with VAE and clustering
type DetectorV2 struct {
	mu                sync.RWMutex
	model             AnomalyModel
	clusterLabeler    *ClusterLabeler
	thresholdOptimizer *ThresholdOptimizer
	feedbackLoop      *FeedbackLoop
	mode              string // "shadow", "guardrail", "blocking"
	metrics           *DetectorV2Metrics
}

// AnomalyModel defines anomaly detection algorithms
type AnomalyModel interface {
	// Fit trains the model on normal samples
	Fit(samples []*AnomalySample) error
	// Score computes anomaly score [0, 1] with uncertainty
	Score(sample *AnomalySample) (*AnomalyScore, error)
	// GetModelType returns model type ("vae", "stacked_ae", "isolation_forest")
	GetModelType() string
}

// AnomalySample represents input for anomaly detection
type AnomalySample struct {
	PCSID        string
	TenantID     string
	Features     []float64 // 11 features from Phase 7 HRS
	Timestamp    time.Time
	GroundTruth  *bool // Optional ground truth for feedback
}

// AnomalyScore represents anomaly detection result
type AnomalyScore struct {
	Score             float64   // [0, 1], higher = more anomalous
	Uncertainty       float64   // Model uncertainty [0, 1]
	ClusterID         int       // Assigned cluster
	ClusterLabel      string    // Human-readable label
	ReconstructionErr float64   // Reconstruction error (for AE models)
	IsAnomaly         bool      // Binary classification
	Confidence        float64   // Classification confidence [0, 1]
	Timestamp         time.Time
}

// ClusterLabeler assigns semantic labels to anomaly clusters
type ClusterLabeler struct {
	mu             sync.RWMutex
	clusters       map[int]*AnomalyCluster
	labeledSamples map[int][]*LabeledSample
	nextClusterID  int
}

// AnomalyCluster represents a cluster of similar anomalies
type AnomalyCluster struct {
	ID           int
	Centroid     []float64
	Label        string // "extreme_d_hat", "coherence_spike", "zero_compressibility", etc.
	Description  string
	Severity     string // "low", "medium", "high", "critical"
	SampleCount  int
	FirstSeen    time.Time
	LastSeen     time.Time
}

// LabeledSample represents a sample with ground truth label
type LabeledSample struct {
	Sample      *AnomalySample
	IsAnomaly   bool
	Label       string
	LabeledBy   string // "human", "automated_rule", "feedback"
	LabeledAt   time.Time
}

// ThresholdOptimizer tunes anomaly threshold based on feedback
type ThresholdOptimizer struct {
	mu                sync.RWMutex
	currentThreshold  float64
	targetFPR         float64 // Target False Positive Rate
	targetTPR         float64 // Target True Positive Rate
	historicalScores  []*ScoredSample
	optimizationRuns  int
	lastOptimization  time.Time
}

// ScoredSample represents a scored sample with ground truth
type ScoredSample struct {
	Score      float64
	IsAnomaly  bool
	Timestamp  time.Time
}

// FeedbackLoop collects and processes human feedback
type FeedbackLoop struct {
	mu               sync.RWMutex
	feedbackQueue    []*FeedbackItem
	processedCount   int64
	acceptedCount    int64
	rejectedCount    int64
}

// FeedbackItem represents user feedback on anomaly detection
type FeedbackItem struct {
	PCSID         string
	PredictedScore float64
	UserLabel      bool   // true = anomaly, false = normal
	Comment        string
	SubmittedBy    string
	SubmittedAt    time.Time
	Processed      bool
}

// DetectorV2Metrics tracks detector performance
type DetectorV2Metrics struct {
	mu                  sync.RWMutex
	TotalSamples        int64
	AnomaliesDetected   int64
	FalsePositives      int64
	TruePositives       int64
	FalseNegatives      int64
	TrueNegatives       int64
	CurrentFPR          float64
	CurrentTPR          float64
	AvgScore            float64
	AvgUncertainty      float64
	ClustersFound       int
	ThresholdOptimizations int64
}

// NewDetectorV2 creates a new anomaly detector v2
func NewDetectorV2(model AnomalyModel, mode string) *DetectorV2 {
	return &DetectorV2{
		model:              model,
		clusterLabeler:     NewClusterLabeler(),
		thresholdOptimizer: NewThresholdOptimizer(0.5, 0.02, 0.95), // threshold=0.5, FPR≤2%, TPR≥95%
		feedbackLoop:       NewFeedbackLoop(),
		mode:               mode, // "shadow", "guardrail", "blocking"
		metrics:            &DetectorV2Metrics{},
	}
}

// NewClusterLabeler creates a new cluster labeler
func NewClusterLabeler() *ClusterLabeler {
	return &ClusterLabeler{
		clusters:       make(map[int]*AnomalyCluster),
		labeledSamples: make(map[int][]*LabeledSample),
		nextClusterID:  0,
	}
}

// NewThresholdOptimizer creates a new threshold optimizer
func NewThresholdOptimizer(initialThreshold, targetFPR, targetTPR float64) *ThresholdOptimizer {
	return &ThresholdOptimizer{
		currentThreshold: initialThreshold,
		targetFPR:        targetFPR,
		targetTPR:        targetTPR,
		historicalScores: []*ScoredSample{},
		optimizationRuns: 0,
	}
}

// NewFeedbackLoop creates a new feedback loop
func NewFeedbackLoop() *FeedbackLoop {
	return &FeedbackLoop{
		feedbackQueue: []*FeedbackItem{},
	}
}

// Detect performs anomaly detection on a sample
func (d *DetectorV2) Detect(ctx context.Context, sample *AnomalySample) (*AnomalyScore, error) {
	startTime := time.Now()

	d.metrics.mu.Lock()
	d.metrics.TotalSamples++
	d.metrics.mu.Unlock()

	// Score sample
	score, err := d.model.Score(sample)
	if err != nil {
		return nil, fmt.Errorf("failed to score sample: %w", err)
	}

	// Assign cluster
	clusterID := d.clusterLabeler.AssignCluster(sample.Features)
	cluster, _ := d.clusterLabeler.GetCluster(clusterID)
	if cluster != nil {
		score.ClusterID = clusterID
		score.ClusterLabel = cluster.Label
	}

	// Apply threshold
	threshold := d.thresholdOptimizer.GetThreshold()
	score.IsAnomaly = score.Score >= threshold
	score.Confidence = math.Abs(score.Score - threshold) / threshold
	score.Timestamp = startTime

	// Update metrics
	d.metrics.mu.Lock()
	if score.IsAnomaly {
		d.metrics.AnomaliesDetected++
	}
	d.metrics.AvgScore = (d.metrics.AvgScore*float64(d.metrics.TotalSamples-1) + score.Score) / float64(d.metrics.TotalSamples)
	d.metrics.AvgUncertainty = (d.metrics.AvgUncertainty*float64(d.metrics.TotalSamples-1) + score.Uncertainty) / float64(d.metrics.TotalSamples)
	d.metrics.mu.Unlock()

	// Mode-specific actions
	switch d.mode {
	case "shadow":
		// Log only, no blocking
		if score.IsAnomaly {
			fmt.Printf("Anomaly detected (shadow): pcs_id=%s, score=%.3f, cluster=%s\n",
				sample.PCSID, score.Score, score.ClusterLabel)
		}

	case "guardrail":
		// Feed to HRS as auxiliary feature
		if score.IsAnomaly {
			fmt.Printf("Anomaly detected (guardrail): pcs_id=%s, score=%.3f, cluster=%s\n",
				sample.PCSID, score.Score, score.ClusterLabel)
			// Integrate with HRS (Phase 8 WP1)
		}

	case "blocking":
		// Reject high-confidence anomalies
		if score.IsAnomaly && score.Confidence > 0.8 {
			return score, fmt.Errorf("anomaly rejected: score=%.3f, cluster=%s", score.Score, score.ClusterLabel)
		}
	}

	return score, nil
}

// Train trains the anomaly model on normal samples
func (d *DetectorV2) Train(ctx context.Context, samples []*AnomalySample) error {
	fmt.Printf("Training anomaly model: samples=%d, model=%s\n", len(samples), d.model.GetModelType())

	if err := d.model.Fit(samples); err != nil {
		return fmt.Errorf("failed to train model: %w", err)
	}

	// Rebuild clusters
	d.clusterLabeler.RebuildClusters(samples)

	fmt.Printf("Anomaly model trained: clusters=%d\n", d.clusterLabeler.GetClusterCount())

	return nil
}

// AssignCluster assigns sample to nearest cluster
func (cl *ClusterLabeler) AssignCluster(features []float64) int {
	cl.mu.Lock()
	defer cl.mu.Unlock()

	if len(cl.clusters) == 0 {
		// Create first cluster
		clusterID := cl.nextClusterID
		cl.nextClusterID++
		cl.clusters[clusterID] = &AnomalyCluster{
			ID:        clusterID,
			Centroid:  features,
			Label:     "unknown",
			FirstSeen: time.Now(),
			LastSeen:  time.Now(),
		}
		return clusterID
	}

	// Find nearest cluster
	minDist := math.Inf(1)
	nearestCluster := 0

	for id, cluster := range cl.clusters {
		dist := euclideanDistance(features, cluster.Centroid)
		if dist < minDist {
			minDist = dist
			nearestCluster = id
		}
	}

	// If too far, create new cluster
	if minDist > 2.0 { // Distance threshold
		clusterID := cl.nextClusterID
		cl.nextClusterID++
		cl.clusters[clusterID] = &AnomalyCluster{
			ID:        clusterID,
			Centroid:  features,
			Label:     "unknown",
			FirstSeen: time.Now(),
			LastSeen:  time.Now(),
		}
		return clusterID
	}

	// Update cluster
	cluster := cl.clusters[nearestCluster]
	cluster.SampleCount++
	cluster.LastSeen = time.Now()

	return nearestCluster
}

// RebuildClusters rebuilds clusters from samples
func (cl *ClusterLabeler) RebuildClusters(samples []*AnomalySample) {
	cl.mu.Lock()
	defer cl.mu.Unlock()

	// Simple k-means clustering (k=5)
	k := 5
	cl.clusters = make(map[int]*AnomalyCluster)

	if len(samples) < k {
		return
	}

	// Initialize centroids with first k samples
	for i := 0; i < k; i++ {
		cl.clusters[i] = &AnomalyCluster{
			ID:        i,
			Centroid:  samples[i].Features,
			Label:     cl.inferLabel(samples[i].Features),
			FirstSeen: time.Now(),
			LastSeen:  time.Now(),
		}
	}

	cl.nextClusterID = k

	fmt.Printf("Rebuilt clusters: count=%d\n", k)
}

// inferLabel infers semantic label from feature values
func (cl *ClusterLabeler) inferLabel(features []float64) string {
	// Features: [D̂, coh★, r, budget, VerifyLatencyMs, SignalEntropy, CoherenceDelta, CompressibilityZ, ...]

	if len(features) < 4 {
		return "unknown"
	}

	dHat := features[0]
	cohStar := features[1]
	r := features[2]

	// Infer label based on feature patterns
	if dHat > 3.0 {
		return "extreme_d_hat"
	}
	if cohStar > 0.95 {
		return "coherence_spike"
	}
	if r < 0.1 {
		return "zero_compressibility"
	}
	if dHat < 0.5 {
		return "low_fractal_dimension"
	}

	return "mixed_anomaly"
}

// GetCluster retrieves cluster by ID
func (cl *ClusterLabeler) GetCluster(clusterID int) (*AnomalyCluster, bool) {
	cl.mu.RLock()
	defer cl.mu.RUnlock()

	cluster, ok := cl.clusters[clusterID]
	return cluster, ok
}

// GetClusterCount returns number of clusters
func (cl *ClusterLabeler) GetClusterCount() int {
	cl.mu.RLock()
	defer cl.mu.RUnlock()

	return len(cl.clusters)
}

// OptimizeThreshold optimizes anomaly threshold based on feedback
func (to *ThresholdOptimizer) OptimizeThreshold() (float64, error) {
	to.mu.Lock()
	defer to.mu.Unlock()

	if len(to.historicalScores) < 100 {
		return to.currentThreshold, fmt.Errorf("insufficient feedback: %d samples (need ≥100)", len(to.historicalScores))
	}

	// Compute ROC curve and find optimal threshold
	bestThreshold := to.currentThreshold
	bestScore := 0.0

	for threshold := 0.1; threshold <= 0.9; threshold += 0.05 {
		tp, fp, fn, tn := to.computeConfusionMatrix(threshold)

		fpr := 0.0
		if fp+tn > 0 {
			fpr = float64(fp) / float64(fp+tn)
		}

		tpr := 0.0
		if tp+fn > 0 {
			tpr = float64(tp) / float64(tp+fn)
		}

		// Multi-objective score: maximize TPR, minimize FPR
		score := tpr - fpr

		// Constraint: FPR ≤ targetFPR, TPR ≥ targetTPR
		if fpr <= to.targetFPR && tpr >= to.targetTPR && score > bestScore {
			bestScore = score
			bestThreshold = threshold
		}
	}

	to.currentThreshold = bestThreshold
	to.optimizationRuns++
	to.lastOptimization = time.Now()

	fmt.Printf("Threshold optimized: new_threshold=%.3f, runs=%d\n", bestThreshold, to.optimizationRuns)

	return bestThreshold, nil
}

// computeConfusionMatrix computes confusion matrix for threshold
func (to *ThresholdOptimizer) computeConfusionMatrix(threshold float64) (tp, fp, fn, tn int) {
	for _, scored := range to.historicalScores {
		predicted := scored.Score >= threshold
		actual := scored.IsAnomaly

		if predicted && actual {
			tp++
		} else if predicted && !actual {
			fp++
		} else if !predicted && actual {
			fn++
		} else {
			tn++
		}
	}

	return tp, fp, fn, tn
}

// AddFeedback adds feedback to threshold optimizer
func (to *ThresholdOptimizer) AddFeedback(score float64, isAnomaly bool) {
	to.mu.Lock()
	defer to.mu.Unlock()

	to.historicalScores = append(to.historicalScores, &ScoredSample{
		Score:     score,
		IsAnomaly: isAnomaly,
		Timestamp: time.Now(),
	})
}

// GetThreshold returns current threshold
func (to *ThresholdOptimizer) GetThreshold() float64 {
	to.mu.RLock()
	defer to.mu.RUnlock()

	return to.currentThreshold
}

// SubmitFeedback submits user feedback
func (fl *FeedbackLoop) SubmitFeedback(pcsID string, predictedScore float64, userLabel bool, comment, submittedBy string) {
	fl.mu.Lock()
	defer fl.mu.Unlock()

	fl.feedbackQueue = append(fl.feedbackQueue, &FeedbackItem{
		PCSID:          pcsID,
		PredictedScore: predictedScore,
		UserLabel:      userLabel,
		Comment:        comment,
		SubmittedBy:    submittedBy,
		SubmittedAt:    time.Now(),
		Processed:      false,
	})

	fmt.Printf("Feedback submitted: pcs_id=%s, user_label=%t, predicted_score=%.3f\n",
		pcsID, userLabel, predictedScore)
}

// ProcessFeedback processes pending feedback
func (fl *FeedbackLoop) ProcessFeedback(detector *DetectorV2) error {
	fl.mu.Lock()
	defer fl.mu.Unlock()

	processed := 0
	for _, item := range fl.feedbackQueue {
		if item.Processed {
			continue
		}

		// Add to threshold optimizer
		detector.thresholdOptimizer.AddFeedback(item.PredictedScore, item.UserLabel)

		// Update metrics
		actual := item.UserLabel
		predicted := item.PredictedScore >= detector.thresholdOptimizer.GetThreshold()

		detector.metrics.mu.Lock()
		if predicted && actual {
			detector.metrics.TruePositives++
		} else if predicted && !actual {
			detector.metrics.FalsePositives++
		} else if !predicted && actual {
			detector.metrics.FalseNegatives++
		} else {
			detector.metrics.TrueNegatives++
		}

		// Update FPR and TPR
		total := detector.metrics.TruePositives + detector.metrics.FalsePositives + detector.metrics.FalseNegatives + detector.metrics.TrueNegatives
		if total > 0 {
			detector.metrics.CurrentFPR = float64(detector.metrics.FalsePositives) / float64(detector.metrics.FalsePositives+detector.metrics.TrueNegatives+1)
			detector.metrics.CurrentTPR = float64(detector.metrics.TruePositives) / float64(detector.metrics.TruePositives+detector.metrics.FalseNegatives+1)
		}
		detector.metrics.mu.Unlock()

		item.Processed = true
		processed++

		fl.processedCount++
		if item.UserLabel == (item.PredictedScore >= detector.thresholdOptimizer.GetThreshold()) {
			fl.acceptedCount++
		} else {
			fl.rejectedCount++
		}
	}

	fmt.Printf("Processed feedback: count=%d, accuracy=%.1f%%\n",
		processed, float64(fl.acceptedCount)/float64(fl.processedCount)*100)

	return nil
}

// GetMetrics returns detector metrics
func (d *DetectorV2) GetMetrics() DetectorV2Metrics {
	d.metrics.mu.RLock()
	defer d.metrics.mu.RUnlock()
	return *d.metrics
}

// VariationalAutoencoder implements VAE-based anomaly detection
type VariationalAutoencoder struct {
	mu            sync.RWMutex
	encoderDims   []int // [11, 8, 5, 3]
	decoderDims   []int // [3, 5, 8, 11]
	latentDim     int   // 3
	trained       bool
	encoderWeights [][][]float64 // Placeholder for weights
	decoderWeights [][][]float64
}

// NewVariationalAutoencoder creates a new VAE
func NewVariationalAutoencoder(inputDim, latentDim int) *VariationalAutoencoder {
	return &VariationalAutoencoder{
		encoderDims: []int{inputDim, 8, 5, latentDim},
		decoderDims: []int{latentDim, 5, 8, inputDim},
		latentDim:   latentDim,
		trained:     false,
	}
}

// Fit trains the VAE
func (vae *VariationalAutoencoder) Fit(samples []*AnomalySample) error {
	vae.mu.Lock()
	defer vae.mu.Unlock()

	fmt.Printf("Training VAE: samples=%d, latent_dim=%d\n", len(samples), vae.latentDim)

	// Placeholder - in production, use proper VAE training (PyTorch/TensorFlow)
	vae.trained = true

	return nil
}

// Score computes anomaly score using reconstruction error
func (vae *VariationalAutoencoder) Score(sample *AnomalySample) (*AnomalyScore, error) {
	vae.mu.RLock()
	defer vae.mu.RUnlock()

	if !vae.trained {
		return nil, fmt.Errorf("VAE not trained")
	}

	// Placeholder - in production, forward pass through VAE
	// Reconstruction error = MSE(input, reconstructed)
	reconstructionErr := 0.15 // Mock reconstruction error

	// Normalize to [0, 1] using sigmoid
	score := 1.0 / (1.0 + math.Exp(-5.0*(reconstructionErr-0.2)))

	// Uncertainty from latent space variance
	uncertainty := 0.1

	return &AnomalyScore{
		Score:             score,
		Uncertainty:       uncertainty,
		ReconstructionErr: reconstructionErr,
		Timestamp:         time.Now(),
	}, nil
}

// GetModelType returns model type
func (vae *VariationalAutoencoder) GetModelType() string {
	return "vae"
}

// euclideanDistance computes Euclidean distance between two vectors
func euclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}

	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return math.Sqrt(sum)
}
