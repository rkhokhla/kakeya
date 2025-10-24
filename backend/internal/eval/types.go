package eval

import (
	"time"

	"github.com/fractal-lba/kakeya/internal/api"
)

// BenchmarkSample represents a single evaluation example.
type BenchmarkSample struct {
	ID           string                 `json:"id"`
	Prompt       string                 `json:"prompt"`
	Response     string                 `json:"response"`
	GroundTruth  bool                   `json:"ground_truth"`  // True if factually correct
	Metadata     map[string]interface{} `json:"metadata"`      // Benchmark-specific fields
	Source       string                 `json:"source"`        // "truthfulqa", "fever", "halueval", "hallulens"
	Category     string                 `json:"category"`      // Benchmark-specific category
	EmbeddingDim int                    `json:"embedding_dim"` // Dimension for embeddings
}

// VerificationResult contains the verification decision and signals.
type VerificationResult struct {
	SampleID      string                 `json:"sample_id"`
	Decision      Decision               `json:"decision"` // ACCEPT, ESCALATE, REJECT
	PCS           *api.PCS               `json:"pcs"`
	ConformalProb float64                `json:"conformal_prob"` // P(correct | signals)
	Signals       map[string]float64     `json:"signals"`        // D_hat, coh_star, r
	Timestamp     time.Time              `json:"timestamp"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// Decision represents verification decision.
type Decision string

const (
	DecisionAccept   Decision = "ACCEPT"
	DecisionEscalate Decision = "ESCALATE"
	DecisionReject   Decision = "REJECT"
)

// BaselineResult contains a baseline method's prediction.
type BaselineResult struct {
	SampleID  string                 `json:"sample_id"`
	Method    string                 `json:"method"` // "perplexity", "nli", "selfcheck", "rag", "gpt4"
	Score     float64                `json:"score"`  // Method-specific score
	Decision  Decision               `json:"decision"`
	Threshold float64                `json:"threshold"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// EvaluationMetrics contains all computed metrics for a run.
type EvaluationMetrics struct {
	// Confusion Matrix
	TruePositives  int `json:"true_positives"`  // Correct accepted
	TrueNegatives  int `json:"true_negatives"`  // Correct rejected
	FalsePositives int `json:"false_positives"` // Hallucination accepted
	FalseNegatives int `json:"false_negatives"` // Correct rejected

	// Derived Metrics
	Precision       float64 `json:"precision"`        // TP / (TP + FP)
	Recall          float64 `json:"recall"`           // TP / (TP + FN)
	F1Score         float64 `json:"f1_score"`         // 2 * (P * R) / (P + R)
	Accuracy        float64 `json:"accuracy"`         // (TP + TN) / (TP + TN + FP + FN)
	FalseAlarmRate  float64 `json:"false_alarm_rate"` // FP / (FP + TN)
	MissRate        float64 `json:"miss_rate"`        // FN / (FN + TP)
	EscalationRate  float64 `json:"escalation_rate"`  // Escalated / Total
	Miscoverage     float64 `json:"miscoverage"`      // Empirical error rate
	TargetDelta     float64 `json:"target_delta"`     // Theoretical miscoverage target
	MiscoverageGap  float64 `json:"miscoverage_gap"`  // |empirical - target|
	CalibratedError float64 `json:"calibrated_error"` // Well-calibrated if gap < tolerance

	// Calibration
	ECE         float64 `json:"ece"`          // Expected Calibration Error
	MaxCE       float64 `json:"max_ce"`       // Maximum Calibration Error
	Brier       float64 `json:"brier"`        // Brier score
	LogLoss     float64 `json:"log_loss"`     // Negative log-likelihood
	NumBins     int     `json:"num_bins"`     // Bins for ECE (typically 10)
	NumSamples  int     `json:"num_samples"`  // Total samples evaluated
	NumEscalate int     `json:"num_escalate"` // Number escalated

	// ROC/AUPRC
	AUC           float64     `json:"auc"`            // Area under ROC curve
	AUPRC         float64     `json:"auprc"`          // Area under precision-recall curve
	ROCCurve      []ROCPoint  `json:"roc_curve"`      // Full ROC curve
	PRCurve       []PRPoint   `json:"pr_curve"`       // Full PR curve
	OperatingPt   ROCPoint    `json:"operating_pt"`   // Current threshold operating point
	OptimalPt     ROCPoint    `json:"optimal_pt"`     // Optimal threshold (max Youden's J)
	ThresholdScan []Threshold `json:"threshold_scan"` // Performance at various thresholds

	// Bootstrap Confidence Intervals (1000 resamples)
	BootstrapCIs BootstrapCIs `json:"bootstrap_cis"`

	// Cost Analysis
	CostPerVerification float64 `json:"cost_per_verification"` // $/verification
	CostPerTrustedTask  float64 `json:"cost_per_trusted_task"` // $/accepted sample
	EscalationCost      float64 `json:"escalation_cost"`       // $/human review
	TotalCost           float64 `json:"total_cost"`            // Total $ for run

	// Timing
	AvgLatencyMs float64 `json:"avg_latency_ms"`
	P50LatencyMs float64 `json:"p50_latency_ms"`
	P95LatencyMs float64 `json:"p95_latency_ms"`
	P99LatencyMs float64 `json:"p99_latency_ms"`
}

// ROCPoint represents a point on the ROC curve.
type ROCPoint struct {
	Threshold   float64 `json:"threshold"`
	FPR         float64 `json:"fpr"` // False Positive Rate
	TPR         float64 `json:"tpr"` // True Positive Rate (Recall)
	YoudenJ     float64 `json:"youden_j"`
	Precision   float64 `json:"precision"`
	F1          float64 `json:"f1"`
	Specificity float64 `json:"specificity"` // 1 - FPR
}

// PRPoint represents a point on the precision-recall curve.
type PRPoint struct {
	Threshold float64 `json:"threshold"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
}

// Threshold represents performance at a specific threshold.
type Threshold struct {
	Value      float64 `json:"value"`
	TP         int     `json:"tp"`
	TN         int     `json:"tn"`
	FP         int     `json:"fp"`
	FN         int     `json:"fn"`
	Precision  float64 `json:"precision"`
	Recall     float64 `json:"recall"`
	F1         float64 `json:"f1"`
	Accuracy   float64 `json:"accuracy"`
	FAR        float64 `json:"far"` // False Alarm Rate
	MissRate   float64 `json:"miss_rate"`
	Escalated  int     `json:"escalated"`  // Samples in ambiguous region
	EscRate    float64 `json:"esc_rate"`   // Escalation rate
	Miscovered int     `json:"miscovered"` // Incorrectly classified
	MiscovRate float64 `json:"miscov_rate"`
}

// BootstrapCIs contains 95% confidence intervals from bootstrap resampling.
type BootstrapCIs struct {
	NumResamples int `json:"num_resamples"` // Typically 1000

	// Point estimates (original sample)
	Precision   float64 `json:"precision"`
	Recall      float64 `json:"recall"`
	F1          float64 `json:"f1"`
	Accuracy    float64 `json:"accuracy"`
	AUC         float64 `json:"auc"`
	AUPRC       float64 `json:"auprc"`
	ECE         float64 `json:"ece"`
	Miscoverage float64 `json:"miscoverage"`

	// 95% CIs [2.5th percentile, 97.5th percentile]
	PrecisionCI   [2]float64 `json:"precision_ci"`
	RecallCI      [2]float64 `json:"recall_ci"`
	F1CI          [2]float64 `json:"f1_ci"`
	AccuracyCI    [2]float64 `json:"accuracy_ci"`
	AUCCI         [2]float64 `json:"auc_ci"`
	AUPRCCI       [2]float64 `json:"auprc_ci"`
	ECECI         [2]float64 `json:"ece_ci"`
	MiscoverageCI [2]float64 `json:"miscoverage_ci"`

	// Standard errors (bootstrap)
	PrecisionSE   float64 `json:"precision_se"`
	RecallSE      float64 `json:"recall_se"`
	F1SE          float64 `json:"f1_se"`
	AccuracySE    float64 `json:"accuracy_se"`
	AUCSE         float64 `json:"auc_se"`
	AUPRCSE       float64 `json:"auprc_se"`
	ECESE         float64 `json:"ece_se"`
	MiscoverageSE float64 `json:"miscoverage_se"`
}

// ComparisonReport compares multiple methods (ASV vs baselines).
type ComparisonReport struct {
	BenchmarkName string                        `json:"benchmark_name"`
	NumSamples    int                           `json:"num_samples"`
	Methods       []string                      `json:"methods"` // ["asv", "perplexity", "nli", ...]
	Results       map[string]EvaluationMetrics  `json:"results"` // method -> metrics
	Rankings      map[string]int                `json:"rankings"`
	BestMethod    string                        `json:"best_method"`    // By F1
	BestROC       string                        `json:"best_roc"`       // By AUC
	BestCal       string                        `json:"best_cal"`       // By ECE
	Pairwise      map[string]map[string]float64 `json:"pairwise"`       // McNemar's test p-values
	StatTests     map[string]StatisticalTest    `json:"stat_tests"`     // Statistical significance
	CostAnalysis  CostComparison                `json:"cost_analysis"`  // Cost-benefit analysis
	Timestamp     time.Time                     `json:"timestamp"`
}

// StatisticalTest contains statistical significance test results.
type StatisticalTest struct {
	TestName   string  `json:"test_name"`   // "mcnemar", "bootstrap", "permutation"
	PValue     float64 `json:"p_value"`     // p-value
	Statistic  float64 `json:"statistic"`   // Test statistic
	Significant bool    `json:"significant"` // p < 0.05
	EffectSize float64 `json:"effect_size"` // Cohen's h or similar
	MethodA    string  `json:"method_a"`
	MethodB    string  `json:"method_b"`
}

// CostComparison compares cost-effectiveness across methods.
type CostComparison struct {
	Methods          []string           `json:"methods"`
	CostPerTrusted   map[string]float64 `json:"cost_per_trusted"`   // $/trusted task
	CostPerSample    map[string]float64 `json:"cost_per_sample"`    // $/sample
	TotalCost        map[string]float64 `json:"total_cost"`         // Total $
	ThroughputQPS    map[string]float64 `json:"throughput_qps"`     // Queries per second
	LatencyP95       map[string]float64 `json:"latency_p95"`        // P95 latency (ms)
	QualityAUC       map[string]float64 `json:"quality_auc"`        // AUC
	ROI              map[string]float64 `json:"roi"`                // ROI = savings / cost
	BestValueMethod  string             `json:"best_value_method"`  // Best cost-quality trade-off
	Pareto           []string           `json:"pareto"`             // Pareto-optimal methods
}
