package eval

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// Comparator performs statistical comparisons between methods.
type Comparator struct {
	numPermutations int
	seed            int64
}

// NewComparator creates a new comparator.
func NewComparator(numPermutations int, seed int64) *Comparator {
	return &Comparator{
		numPermutations: numPermutations,
		seed:            seed,
	}
}

// CompareAll performs comprehensive comparison between all methods.
func (c *Comparator) CompareAll(
	samples []*BenchmarkSample,
	results map[string][]*VerificationResult,
) *ComparisonReport {
	report := &ComparisonReport{
		MethodMetrics:    make(map[string]*EvaluationMetrics),
		StatisticalTests: make(map[string]StatisticalTest),
	}

	// Extract method names
	for method := range results {
		report.MethodNames = append(report.MethodNames, method)
	}
	sort.Strings(report.MethodNames)

	// Pairwise comparisons
	for i := 0; i < len(report.MethodNames); i++ {
		method1 := report.MethodNames[i]
		for j := i + 1; j < len(report.MethodNames); j++ {
			method2 := report.MethodNames[j]

			testKey := fmt.Sprintf("%s_vs_%s", method1, method2)

			// McNemar's test
			mcnemar := c.McNemarTest(samples, results[method1], results[method2])
			mcnemar.TestName = fmt.Sprintf("McNemar_%s", testKey)
			report.StatisticalTests[testKey+"_mcnemar"] = mcnemar

			// Permutation test for accuracy difference
			permTest := c.PermutationTest(samples, results[method1], results[method2])
			permTest.TestName = fmt.Sprintf("Permutation_%s", testKey)
			report.StatisticalTests[testKey+"_perm"] = permTest
		}
	}

	return report
}

// McNemarTest performs McNemar's test for paired binary outcomes.
// Tests whether two methods have significantly different error rates.
//
// Null hypothesis: P(method1 correct, method2 wrong) = P(method1 wrong, method2 correct)
//
// Returns: StatisticalTest with chi-squared statistic and p-value
func (c *Comparator) McNemarTest(
	samples []*BenchmarkSample,
	results1, results2 []*VerificationResult,
) StatisticalTest {
	if len(samples) != len(results1) || len(samples) != len(results2) {
		return StatisticalTest{
			TestName:    "McNemar",
			Significant: false,
			PValue:      1.0,
		}
	}

	// Build 2x2 contingency table
	// Method1: Correct | Wrong
	// Method2:
	//   Correct    a   |   b
	//   Wrong      c   |   d

	a, b, c, d := 0, 0, 0, 0

	for i, sample := range samples {
		r1Correct := c.isCorrect(results1[i], sample.GroundTruth)
		r2Correct := c.isCorrect(results2[i], sample.GroundTruth)

		if r1Correct && r2Correct {
			a++
		} else if r1Correct && !r2Correct {
			b++
		} else if !r1Correct && r2Correct {
			c++
		} else {
			d++
		}
	}

	// McNemar's chi-squared = (b - c)^2 / (b + c)
	// With continuity correction: (|b - c| - 1)^2 / (b + c)

	chiSquared := 0.0
	if b+c > 0 {
		// Apply continuity correction
		numerator := math.Abs(float64(b-c)) - 1.0
		if numerator < 0 {
			numerator = 0
		}
		chiSquared = (numerator * numerator) / float64(b+c)
	}

	// Compute p-value using chi-squared distribution with df=1
	pValue := c.chiSquaredPValue(chiSquared, 1)

	// Significant if p < 0.05
	significant := pValue < 0.05

	// Effect size: proportion of discordant pairs favoring method1
	effectSize := 0.0
	if b+c > 0 {
		effectSize = float64(b-c) / float64(b+c)
	}

	return StatisticalTest{
		TestName:      "McNemar",
		TestStatistic: chiSquared,
		PValue:        pValue,
		Significant:   significant,
		EffectSize:    effectSize,
		Metadata: map[string]interface{}{
			"contingency": map[string]int{
				"both_correct":   a,
				"m1_correct_m2_wrong": b,
				"m1_wrong_m2_correct": c,
				"both_wrong":     d,
			},
			"discordant_pairs": b + c,
		},
	}
}

// PermutationTest performs permutation test for difference in accuracy.
// Tests whether two methods have significantly different accuracies.
//
// Null hypothesis: Methods have equal accuracy (labels are exchangeable)
//
// Returns: StatisticalTest with observed difference and p-value
func (c *Comparator) PermutationTest(
	samples []*BenchmarkSample,
	results1, results2 []*VerificationResult,
) StatisticalTest {
	if len(samples) != len(results1) || len(samples) != len(results2) {
		return StatisticalTest{
			TestName:    "Permutation",
			Significant: false,
			PValue:      1.0,
		}
	}

	// Observed difference in accuracy
	acc1 := c.computeAccuracy(samples, results1)
	acc2 := c.computeAccuracy(samples, results2)
	observedDiff := acc1 - acc2

	// Permutation test: shuffle labels between methods
	rng := rand.New(rand.NewSource(c.seed))
	count := 0

	for perm := 0; perm < c.numPermutations; perm++ {
		// Randomly swap results between methods
		perm1 := make([]*VerificationResult, len(results1))
		perm2 := make([]*VerificationResult, len(results2))

		for i := 0; i < len(results1); i++ {
			if rng.Float64() < 0.5 {
				perm1[i] = results1[i]
				perm2[i] = results2[i]
			} else {
				perm1[i] = results2[i]
				perm2[i] = results1[i]
			}
		}

		// Compute difference for this permutation
		permAcc1 := c.computeAccuracy(samples, perm1)
		permAcc2 := c.computeAccuracy(samples, perm2)
		permDiff := permAcc1 - permAcc2

		// Count how many permutations have difference >= observed
		if math.Abs(permDiff) >= math.Abs(observedDiff) {
			count++
		}
	}

	// Two-sided p-value
	pValue := float64(count) / float64(c.numPermutations)

	significant := pValue < 0.05

	return StatisticalTest{
		TestName:      "Permutation",
		TestStatistic: observedDiff,
		PValue:        pValue,
		Significant:   significant,
		EffectSize:    observedDiff, // Difference in accuracy
		Metadata: map[string]interface{}{
			"accuracy_method1":  acc1,
			"accuracy_method2":  acc2,
			"num_permutations":  c.numPermutations,
		},
	}
}

// BootstrapCompare performs bootstrap comparison of two methods.
// Computes confidence interval for difference in a metric.
func (c *Comparator) BootstrapCompare(
	samples []*BenchmarkSample,
	results1, results2 []*VerificationResult,
	metric string,
	numResamples int,
) (float64, [2]float64, error) {
	if len(samples) != len(results1) || len(samples) != len(results2) {
		return 0, [2]float64{}, fmt.Errorf("sample size mismatch")
	}

	n := len(samples)
	rng := rand.New(rand.NewSource(c.seed))

	// Observed difference
	obs1 := c.computeMetric(samples, results1, metric)
	obs2 := c.computeMetric(samples, results2, metric)
	observedDiff := obs1 - obs2

	// Bootstrap distribution
	diffs := make([]float64, numResamples)

	for b := 0; b < numResamples; b++ {
		// Resample with replacement
		resampleSamples := make([]*BenchmarkSample, n)
		resampleResults1 := make([]*VerificationResult, n)
		resampleResults2 := make([]*VerificationResult, n)

		for i := 0; i < n; i++ {
			idx := rng.Intn(n)
			resampleSamples[i] = samples[idx]
			resampleResults1[i] = results1[idx]
			resampleResults2[i] = results2[idx]
		}

		// Compute metric for resample
		m1 := c.computeMetric(resampleSamples, resampleResults1, metric)
		m2 := c.computeMetric(resampleSamples, resampleResults2, metric)
		diffs[b] = m1 - m2
	}

	// Compute 95% CI
	sort.Float64s(diffs)
	idx025 := int(float64(numResamples) * 0.025)
	idx975 := int(float64(numResamples) * 0.975)
	if idx025 >= numResamples {
		idx025 = numResamples - 1
	}
	if idx975 >= numResamples {
		idx975 = numResamples - 1
	}

	ci := [2]float64{diffs[idx025], diffs[idx975]}

	return observedDiff, ci, nil
}

// isCorrect checks if a verification result matches ground truth.
func (c *Comparator) isCorrect(result *VerificationResult, groundTruth bool) bool {
	// Treat ESCALATE as REJECT for binary classification
	predicted := (result.Decision == DecisionAccept)
	return predicted == groundTruth
}

// computeAccuracy computes accuracy for a set of results.
func (c *Comparator) computeAccuracy(
	samples []*BenchmarkSample,
	results []*VerificationResult,
) float64 {
	correct := 0
	for i, sample := range samples {
		if c.isCorrect(results[i], sample.GroundTruth) {
			correct++
		}
	}
	return float64(correct) / float64(len(samples))
}

// computeMetric computes a specific metric for a method.
func (c *Comparator) computeMetric(
	samples []*BenchmarkSample,
	results []*VerificationResult,
	metric string,
) float64 {
	switch metric {
	case "accuracy":
		return c.computeAccuracy(samples, results)
	case "precision":
		return c.computePrecision(samples, results)
	case "recall":
		return c.computeRecall(samples, results)
	case "f1":
		p := c.computePrecision(samples, results)
		r := c.computeRecall(samples, results)
		if p+r == 0 {
			return 0
		}
		return 2 * p * r / (p + r)
	default:
		return 0
	}
}

// computePrecision computes precision.
func (c *Comparator) computePrecision(
	samples []*BenchmarkSample,
	results []*VerificationResult,
) float64 {
	tp, fp := 0, 0
	for i, sample := range samples {
		predicted := (results[i].Decision == DecisionAccept)
		if predicted {
			if sample.GroundTruth {
				tp++
			} else {
				fp++
			}
		}
	}

	if tp+fp == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fp)
}

// computeRecall computes recall.
func (c *Comparator) computeRecall(
	samples []*BenchmarkSample,
	results []*VerificationResult,
) float64 {
	tp, fn := 0, 0
	for i, sample := range samples {
		if sample.GroundTruth {
			predicted := (results[i].Decision == DecisionAccept)
			if predicted {
				tp++
			} else {
				fn++
			}
		}
	}

	if tp+fn == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fn)
}

// chiSquaredPValue computes p-value for chi-squared statistic.
// Uses numerical approximation for chi-squared CDF.
func (c *Comparator) chiSquaredPValue(chiSq float64, df int) float64 {
	if chiSq <= 0 {
		return 1.0
	}

	// For df=1 (McNemar), use lookup table
	if df == 1 {
		thresholds := []float64{3.841, 6.635, 10.828}
		pvalues := []float64{0.05, 0.01, 0.001}

		for i, threshold := range thresholds {
			if chiSq < threshold {
				if i == 0 {
					return 0.10 // Between 0.05 and 1.0
				}
				return pvalues[i-1]
			}
		}
		return 0.0001 // Very significant
	}

	// For other df, use approximation
	return 0.05 // Placeholder
}

// cohensD computes Cohen's d effect size.
func cohensD(group1, group2 []float64) float64 {
	if len(group1) == 0 || len(group2) == 0 {
		return 0
	}

	mean1 := mean(group1)
	mean2 := mean(group2)

	// Pooled standard deviation
	var1 := variance(group1)
	var2 := variance(group2)
	pooledSD := math.Sqrt((var1 + var2) / 2.0)

	if pooledSD == 0 {
		return 0
	}

	return (mean1 - mean2) / pooledSD
}

// mean computes arithmetic mean.
func mean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

// variance computes sample variance.
func variance(data []float64) float64 {
	if len(data) <= 1 {
		return 0
	}

	m := mean(data)
	sumSq := 0.0
	for _, v := range data {
		diff := v - m
		sumSq += diff * diff
	}
	return sumSq / float64(len(data)-1)
}

// stddev computes standard deviation.
func stddev(data []float64) float64 {
	return math.Sqrt(variance(data))
}
