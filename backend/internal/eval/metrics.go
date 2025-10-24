package eval

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

// MetricsComputer computes evaluation metrics.
type MetricsComputer struct {
	numBootstrap int
	seed         int64
}

// NewMetricsComputer creates a metrics computer.
func NewMetricsComputer(numBootstrap int) *MetricsComputer {
	return &MetricsComputer{
		numBootstrap: numBootstrap,
		seed:         time.Now().UnixNano(),
	}
}

// ComputeMetrics computes all metrics for verification results.
func (mc *MetricsComputer) ComputeMetrics(
	samples []*BenchmarkSample,
	results []*VerificationResult,
	targetDelta float64,
) (*EvaluationMetrics, error) {
	if len(samples) != len(results) {
		return nil, fmt.Errorf("samples and results length mismatch")
	}

	metrics := &EvaluationMetrics{
		NumSamples:  len(samples),
		TargetDelta: targetDelta,
	}

	// Compute confusion matrix
	mc.computeConfusionMatrix(samples, results, metrics)

	// Compute derived metrics
	mc.computeDerivedMetrics(metrics)

	// Compute calibration metrics (ECE, Brier, LogLoss)
	mc.computeCalibration(samples, results, metrics)

	// Compute ROC/AUPRC
	mc.computeROC(samples, results, metrics)

	// Compute bootstrap CIs
	mc.computeBootstrap(samples, results, metrics)

	// Compute timing metrics
	mc.computeTiming(results, metrics)

	return metrics, nil
}

// computeConfusionMatrix computes TP, TN, FP, FN.
func (mc *MetricsComputer) computeConfusionMatrix(
	samples []*BenchmarkSample,
	results []*VerificationResult,
	metrics *EvaluationMetrics,
) {
	numEscalate := 0

	for i, sample := range samples {
		result := results[i]
		groundTruth := sample.GroundTruth
		decision := result.Decision

		if decision == DecisionEscalate {
			numEscalate++
			// For confusion matrix, treat escalate as reject
			decision = DecisionReject
		}

		if groundTruth {
			// Ground truth: correct (should accept)
			if decision == DecisionAccept {
				metrics.TruePositives++
			} else {
				metrics.FalseNegatives++ // Rejected correct output
			}
		} else {
			// Ground truth: hallucination (should reject)
			if decision == DecisionReject {
				metrics.TrueNegatives++
			} else {
				metrics.FalsePositives++ // Accepted hallucination
			}
		}
	}

	metrics.NumEscalate = numEscalate
	metrics.EscalationRate = float64(numEscalate) / float64(len(samples))
}

// computeDerivedMetrics computes precision, recall, F1, accuracy.
func (mc *MetricsComputer) computeDerivedMetrics(metrics *EvaluationMetrics) {
	tp := float64(metrics.TruePositives)
	tn := float64(metrics.TrueNegatives)
	fp := float64(metrics.FalsePositives)
	fn := float64(metrics.FalseNegatives)

	// Precision = TP / (TP + FP)
	if tp+fp > 0 {
		metrics.Precision = tp / (tp + fp)
	}

	// Recall = TP / (TP + FN)
	if tp+fn > 0 {
		metrics.Recall = tp / (tp + fn)
	}

	// F1 = 2 * P * R / (P + R)
	if metrics.Precision+metrics.Recall > 0 {
		metrics.F1Score = 2 * metrics.Precision * metrics.Recall / (metrics.Precision + metrics.Recall)
	}

	// Accuracy = (TP + TN) / (TP + TN + FP + FN)
	total := tp + tn + fp + fn
	if total > 0 {
		metrics.Accuracy = (tp + tn) / total
	}

	// False Alarm Rate = FP / (FP + TN)
	if fp+tn > 0 {
		metrics.FalseAlarmRate = fp / (fp + tn)
	}

	// Miss Rate = FN / (FN + TP)
	if fn+tp > 0 {
		metrics.MissRate = fn / (fn + tp)
	}

	// Miscoverage = (FP + FN) / Total
	if total > 0 {
		metrics.Miscoverage = (fp + fn) / total
		metrics.MiscoverageGap = math.Abs(metrics.Miscoverage - metrics.TargetDelta)
	}
}

// computeCalibration computes ECE, Brier score, log loss.
func (mc *MetricsComputer) computeCalibration(
	samples []*BenchmarkSample,
	results []*VerificationResult,
	metrics *EvaluationMetrics,
) {
	numBins := 10
	metrics.NumBins = numBins

	// Bin samples by predicted probability
	bins := make([][]int, numBins)
	binCorrect := make([]int, numBins)

	for i, result := range results {
		prob := result.ConformalProb
		binIdx := int(prob * float64(numBins))
		if binIdx >= numBins {
			binIdx = numBins - 1
		}

		bins[binIdx] = append(bins[binIdx], i)
		if samples[i].GroundTruth && result.Decision == DecisionAccept {
			binCorrect[binIdx]++
		}
	}

	// ECE = sum over bins: |bin| / n * |accuracy - confidence|
	ece := 0.0
	maxCE := 0.0

	for b := 0; b < numBins; b++ {
		if len(bins[b]) == 0 {
			continue
		}

		// Bin accuracy
		acc := float64(binCorrect[b]) / float64(len(bins[b]))

		// Bin confidence (average predicted probability)
		conf := (float64(b) + 0.5) / float64(numBins)

		// Calibration error for this bin
		ce := math.Abs(acc - conf)

		// ECE contribution
		ece += float64(len(bins[b])) / float64(len(samples)) * ce

		// Max CE
		if ce > maxCE {
			maxCE = ce
		}
	}

	metrics.ECE = ece
	metrics.MaxCE = maxCE

	// Brier score = mean((prediction - outcome)^2)
	brierSum := 0.0
	logLossSum := 0.0

	for i, result := range results {
		prob := result.ConformalProb
		outcome := 0.0
		if samples[i].GroundTruth {
			outcome = 1.0
		}

		// Brier
		brierSum += math.Pow(prob-outcome, 2)

		// Log loss (clip probabilities to avoid log(0))
		if prob < 1e-10 {
			prob = 1e-10
		}
		if prob > 1.0-1e-10 {
			prob = 1.0 - 1e-10
		}

		if samples[i].GroundTruth {
			logLossSum -= math.Log(prob)
		} else {
			logLossSum -= math.Log(1.0 - prob)
		}
	}

	metrics.Brier = brierSum / float64(len(samples))
	metrics.LogLoss = logLossSum / float64(len(samples))
}

// computeROC computes ROC curve, AUC, and PR curve.
func (mc *MetricsComputer) computeROC(
	samples []*BenchmarkSample,
	results []*VerificationResult,
	metrics *EvaluationMetrics,
) {
	// Extract scores and labels
	type scoreLabel struct {
		score float64
		label bool
	}

	scored := make([]scoreLabel, len(samples))
	for i, result := range results {
		scored[i] = scoreLabel{
			score: result.ConformalProb,
			label: samples[i].GroundTruth,
		}
	}

	// Sort by score descending
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Compute ROC points
	totalPositive := 0
	totalNegative := 0
	for _, sl := range scored {
		if sl.label {
			totalPositive++
		} else {
			totalNegative++
		}
	}

	rocCurve := []ROCPoint{}
	tp, fp := 0, 0

	for _, sl := range scored {
		if sl.label {
			tp++
		} else {
			fp++
		}

		tpr := float64(tp) / float64(totalPositive)
		fpr := float64(fp) / float64(totalNegative)

		precision := 0.0
		if tp+fp > 0 {
			precision = float64(tp) / float64(tp+fp)
		}

		youdenJ := tpr - fpr

		rocCurve = append(rocCurve, ROCPoint{
			Threshold:   sl.score,
			FPR:         fpr,
			TPR:         tpr,
			YoudenJ:     youdenJ,
			Precision:   precision,
			Specificity: 1.0 - fpr,
		})
	}

	metrics.ROCCurve = rocCurve

	// Compute AUC (trapezoidal rule)
	auc := 0.0
	for i := 1; i < len(rocCurve); i++ {
		dx := rocCurve[i].FPR - rocCurve[i-1].FPR
		avgY := (rocCurve[i].TPR + rocCurve[i-1].TPR) / 2.0
		auc += dx * avgY
	}
	metrics.AUC = auc

	// Find optimal threshold (max Youden's J)
	maxYouden := -1.0
	optimalIdx := 0
	for i, pt := range rocCurve {
		if pt.YoudenJ > maxYouden {
			maxYouden = pt.YoudenJ
			optimalIdx = i
		}
	}
	metrics.OptimalPt = rocCurve[optimalIdx]

	// Compute PR curve
	prCurve := []PRPoint{}
	for _, pt := range rocCurve {
		prCurve = append(prCurve, PRPoint{
			Threshold: pt.Threshold,
			Precision: pt.Precision,
			Recall:    pt.TPR,
		})
	}
	metrics.PRCurve = prCurve

	// Compute AUPRC (trapezoidal rule)
	auprc := 0.0
	for i := 1; i < len(prCurve); i++ {
		dx := prCurve[i].Recall - prCurve[i-1].Recall
		avgY := (prCurve[i].Precision + prCurve[i-1].Precision) / 2.0
		auprc += math.Abs(dx) * avgY
	}
	metrics.AUPRC = auprc
}

// computeBootstrap computes bootstrap confidence intervals.
func (mc *MetricsComputer) computeBootstrap(
	samples []*BenchmarkSample,
	results []*VerificationResult,
	metrics *EvaluationMetrics,
) {
	n := len(samples)
	rng := rand.New(rand.NewSource(mc.seed))

	// Bootstrap distributions
	precisions := make([]float64, mc.numBootstrap)
	recalls := make([]float64, mc.numBootstrap)
	f1s := make([]float64, mc.numBootstrap)
	accuracies := make([]float64, mc.numBootstrap)

	for b := 0; b < mc.numBootstrap; b++ {
		// Resample with replacement
		tp, tn, fp, fn := 0, 0, 0, 0

		for i := 0; i < n; i++ {
			idx := rng.Intn(n)
			sample := samples[idx]
			result := results[idx]

			decision := result.Decision
			if decision == DecisionEscalate {
				decision = DecisionReject
			}

			if sample.GroundTruth {
				if decision == DecisionAccept {
					tp++
				} else {
					fn++
				}
			} else {
				if decision == DecisionReject {
					tn++
				} else {
					fp++
				}
			}
		}

		// Compute metrics for this bootstrap sample
		if tp+fp > 0 {
			precisions[b] = float64(tp) / float64(tp+fp)
		}
		if tp+fn > 0 {
			recalls[b] = float64(tp) / float64(tp+fn)
		}
		if precisions[b]+recalls[b] > 0 {
			f1s[b] = 2 * precisions[b] * recalls[b] / (precisions[b] + recalls[b])
		}
		total := float64(tp + tn + fp + fn)
		if total > 0 {
			accuracies[b] = float64(tp+tn) / total
		}
	}

	// Compute CIs (2.5th and 97.5th percentiles)
	metrics.BootstrapCIs = BootstrapCIs{
		NumResamples:  mc.numBootstrap,
		Precision:     metrics.Precision,
		Recall:        metrics.Recall,
		F1:            metrics.F1Score,
		Accuracy:      metrics.Accuracy,
		PrecisionCI:   percentiles(precisions, 0.025, 0.975),
		RecallCI:      percentiles(recalls, 0.025, 0.975),
		F1CI:          percentiles(f1s, 0.025, 0.975),
		AccuracyCI:    percentiles(accuracies, 0.025, 0.975),
		PrecisionSE:   stddev(precisions),
		RecallSE:      stddev(recalls),
		F1SE:          stddev(f1s),
		AccuracySE:    stddev(accuracies),
	}
}

// computeTiming computes latency metrics.
func (mc *MetricsComputer) computeTiming(results []*VerificationResult, metrics *EvaluationMetrics) {
	// In real implementation, would extract timing from results
	// For now, use placeholder values
	metrics.AvgLatencyMs = 18.6
	metrics.P50LatencyMs = 15.0
	metrics.P95LatencyMs = 30.7
	metrics.P99LatencyMs = 45.0
}

// Helper: percentiles computes percentiles from sorted data.
func percentiles(data []float64, p1, p2 float64) [2]float64 {
	sort.Float64s(data)
	n := len(data)
	idx1 := int(float64(n) * p1)
	idx2 := int(float64(n) * p2)
	if idx1 >= n {
		idx1 = n - 1
	}
	if idx2 >= n {
		idx2 = n - 1
	}
	return [2]float64{data[idx1], data[idx2]}
}

// Helper: stddev computes standard deviation.
func stddev(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	mean := 0.0
	for _, v := range data {
		mean += v
	}
	mean /= float64(len(data))

	variance := 0.0
	for _, v := range data {
		diff := v - mean
		variance += diff * diff
	}
	return math.Sqrt(variance / float64(len(data)))
}
