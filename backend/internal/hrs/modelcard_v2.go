package hrs

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// ModelCardV2 extends Phase 8 ModelCard with explainability and fairness
// Phase 9 WP1: Comprehensive model governance with transparency
type ModelCardV2 struct {
	// Basic metadata (from Phase 8)
	ModelVersion    string                 `json:"model_version"`
	TrainedAt       time.Time              `json:"trained_at"`
	Algorithm       string                 `json:"algorithm"` // "logistic_regression", "gbdt", "mlp"
	DatasetHash     string                 `json:"dataset_hash"`
	TrainingMetrics map[string]float64     `json:"training_metrics"` // AUC, precision, recall, etc.

	// Explainability (Phase 9 WP1)
	ExplainabilityMethods []string           `json:"explainability_methods"` // ["shap_kernel", "lime"]
	FeatureImportance     []FeatureImportance `json:"feature_importance"`
	BaselineFeatures      []float64          `json:"baseline_features"`

	// Fairness & Bias (Phase 9 WP1)
	FairnessAudit *FairnessAudit `json:"fairness_audit"`

	// Limitations & Risks
	Limitations []string `json:"limitations"`
	BiasRisks   []string `json:"bias_risks"`
	UseCases    []string `json:"use_cases"`
	NonUseCases []string `json:"non_use_cases"`

	// Drift & Reversion
	DriftMonitoring *DriftStatus `json:"drift_monitoring"`
	RevertHistory   []RevertEvent `json:"revert_history"`

	// Compliance
	EthicalReview bool      `json:"ethical_review"`
	ReviewedBy    string    `json:"reviewed_by"`
	ApprovedAt    time.Time `json:"approved_at"`
}

// FeatureImportance captures global feature importance
type FeatureImportance struct {
	FeatureName       string  `json:"feature_name"`
	GlobalImportance  float64 `json:"global_importance"`  // e.g., mean |SHAP| across all samples
	PositiveInfluence float64 `json:"positive_influence"` // % samples where feature increases risk
	NegativeInfluence float64 `json:"negative_influence"` // % samples where feature decreases risk
}

// FairnessAudit tracks subgroup performance and bias metrics
type FairnessAudit struct {
	AuditedAt time.Time `json:"audited_at"`

	// Subgroup metrics (e.g., by tenant_type, model_version, region)
	Subgroups []SubgroupMetrics `json:"subgroups"`

	// Overall fairness scores
	MaxSubgroupGap  float64 `json:"max_subgroup_gap"`  // Max AUC difference across subgroups
	CalibrationBias float64 `json:"calibration_bias"` // Mean calibration error

	// Thresholds
	SubgroupGapThreshold float64 `json:"subgroup_gap_threshold"` // e.g., 0.05 (5 percentage points)
	CalibrationThreshold float64 `json:"calibration_threshold"`  // e.g., 0.10
	Status               string  `json:"status"`                 // "pass", "warning", "fail"
}

// SubgroupMetrics captures performance for a specific subgroup
type SubgroupMetrics struct {
	SubgroupName  string             `json:"subgroup_name"`  // e.g., "tenant_type=enterprise"
	SampleCount   int                `json:"sample_count"`
	Metrics       map[string]float64 `json:"metrics"` // AUC, precision, recall, FPR, TPR
	Calibration   float64            `json:"calibration"` // Mean absolute calibration error
	Representation float64           `json:"representation"` // % of total samples
}

// DriftStatus tracks model drift and alerts
type DriftStatus struct {
	LastChecked time.Time `json:"last_checked"`

	// AUC drift
	BaselineAUC float64 `json:"baseline_auc"` // From training
	CurrentAUC  float64 `json:"current_auc"`  // Recent performance
	AUCDrop     float64 `json:"auc_drop"`     // Baseline - Current

	// Feature drift (K-S test p-values)
	FeatureDrift map[string]float64 `json:"feature_drift"` // feature_name -> p-value

	// Alert thresholds
	AUCDropThreshold     float64 `json:"auc_drop_threshold"`     // e.g., 0.05
	FeatureDriftThreshold float64 `json:"feature_drift_threshold"` // e.g., 0.01 (p-value)

	// Status
	DriftDetected bool     `json:"drift_detected"`
	Alerts        []string `json:"alerts"`
}

// RevertEvent records model reversion history
type RevertEvent struct {
	RevertedAt     time.Time `json:"reverted_at"`
	FromVersion    string    `json:"from_version"`
	ToVersion      string    `json:"to_version"`
	Reason         string    `json:"reason"` // "auc_drop", "fairness_violation", "manual"
	TriggeredBy    string    `json:"triggered_by"` // "auto", "operator", "manual"
	RollbackTimeMs float64   `json:"rollback_time_ms"`
}

// ModelCardV2Builder helps construct model cards with validation
type ModelCardV2Builder struct {
	card *ModelCardV2
	mu   sync.Mutex
}

// NewModelCardV2Builder creates a builder for model card v2
func NewModelCardV2Builder(version string) *ModelCardV2Builder {
	return &ModelCardV2Builder{
		card: &ModelCardV2{
			ModelVersion:    version,
			TrainedAt:       time.Now(),
			TrainingMetrics: make(map[string]float64),
			Limitations:     []string{},
			BiasRisks:       []string{},
			UseCases:        []string{},
			NonUseCases:     []string{},
			RevertHistory:   []RevertEvent{},
		},
	}
}

// WithAlgorithm sets the model algorithm
func (b *ModelCardV2Builder) WithAlgorithm(algo string) *ModelCardV2Builder {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.card.Algorithm = algo
	return b
}

// WithDatasetHash sets the dataset hash
func (b *ModelCardV2Builder) WithDatasetHash(hash string) *ModelCardV2Builder {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.card.DatasetHash = hash
	return b
}

// WithTrainingMetrics sets training metrics
func (b *ModelCardV2Builder) WithTrainingMetrics(metrics map[string]float64) *ModelCardV2Builder {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.card.TrainingMetrics = metrics
	return b
}

// WithFeatureImportance sets global feature importance
func (b *ModelCardV2Builder) WithFeatureImportance(importance []FeatureImportance) *ModelCardV2Builder {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.card.FeatureImportance = importance
	return b
}

// WithFairnessAudit sets fairness audit results
func (b *ModelCardV2Builder) WithFairnessAudit(audit *FairnessAudit) *ModelCardV2Builder {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.card.FairnessAudit = audit
	return b
}

// WithLimitations adds model limitations
func (b *ModelCardV2Builder) WithLimitations(limitations []string) *ModelCardV2Builder {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.card.Limitations = limitations
	return b
}

// WithBiasRisks adds bias risk descriptions
func (b *ModelCardV2Builder) WithBiasRisks(risks []string) *ModelCardV2Builder {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.card.BiasRisks = risks
	return b
}

// WithUseCases defines appropriate use cases
func (b *ModelCardV2Builder) WithUseCases(useCases []string) *ModelCardV2Builder {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.card.UseCases = useCases
	return b
}

// WithEthicalReview marks ethical review status
func (b *ModelCardV2Builder) WithEthicalReview(reviewed bool, reviewer string) *ModelCardV2Builder {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.card.EthicalReview = reviewed
	b.card.ReviewedBy = reviewer
	if reviewed {
		b.card.ApprovedAt = time.Now()
	}
	return b
}

// Build constructs and validates the model card
func (b *ModelCardV2Builder) Build() (*ModelCardV2, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Validation
	if b.card.ModelVersion == "" {
		return nil, fmt.Errorf("model_version is required")
	}
	if b.card.Algorithm == "" {
		return nil, fmt.Errorf("algorithm is required")
	}
	if len(b.card.TrainingMetrics) == 0 {
		return nil, fmt.Errorf("training_metrics are required")
	}

	// Ensure AUC present
	if _, ok := b.card.TrainingMetrics["auc"]; !ok {
		return nil, fmt.Errorf("training_metrics must include 'auc'")
	}

	return b.card, nil
}

// ToJSON serializes model card to JSON
func (card *ModelCardV2) ToJSON() ([]byte, error) {
	return json.MarshalIndent(card, "", "  ")
}

// FromJSON deserializes model card from JSON
func FromJSON(data []byte) (*ModelCardV2, error) {
	var card ModelCardV2
	if err := json.Unmarshal(data, &card); err != nil {
		return nil, err
	}
	return &card, nil
}

// DefaultLimitations returns standard HRS limitations
func DefaultLimitations() []string {
	return []string{
		"Model trained on historical data; may not generalize to novel attack patterns",
		"Features are derived from PCS signals (D̂, coh★, r); does not analyze raw content",
		"Prediction latency budget: ≤10ms p95; explainability adds ≤2ms",
		"Calibration assumes stable tenant behavior; may drift over time",
		"No causality guarantees; correlations may be spurious",
	}
}

// DefaultBiasRisks returns standard bias risks
func DefaultBiasRisks() []string {
	return []string{
		"Tenant heterogeneity: performance may vary across tenant types (enterprise vs SMB)",
		"Temporal bias: recent data weighted more heavily; seasonal patterns may affect predictions",
		"Feature correlation: D̂ and coh★ may be correlated, leading to over-fitting",
		"Selection bias: only PCS with complete signals are used for training",
	}
}

// DefaultUseCases returns appropriate HRS use cases
func DefaultUseCases() []string {
	return []string{
		"Risk-based routing: escalate high-risk PCS to ensemble verification or human review",
		"Cost optimization: reduce ensemble overhead for low-risk PCS",
		"Anomaly triage: prioritize anomaly investigations based on risk score",
		"SLO monitoring: track risk score distributions for early warning of drift",
	}
}

// DefaultNonUseCases returns inappropriate use cases
func DefaultNonUseCases() []string {
	return []string{
		"Content moderation: HRS does not analyze text or detect toxic content",
		"Privacy detection: HRS does not identify PII or sensitive information",
		"Sole decision-making: HRS should complement, not replace, verification",
		"Long-term forecasting: predictions valid for current session only",
	}
}
