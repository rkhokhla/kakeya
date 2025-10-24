package conformal

import (
	"fmt"
	"math"
	"sort"
)

// DriftDetector monitors for distribution drift in nonconformity scores.
// Uses Kolmogorov-Smirnov (KS) test to compare recent vs. calibration distributions.
type DriftDetector struct {
	recentScores []float64 // Recent scores for comparison
	maxRecent    int       // Maximum recent scores to track
	ksThreshold  float64   // KS statistic threshold (typically 0.05-0.10)
}

// NewDriftDetector creates a new drift detector.
func NewDriftDetector(maxRecent int, ksThreshold float64) *DriftDetector {
	if maxRecent <= 0 {
		maxRecent = 100 // Default: track last 100 scores
	}
	if ksThreshold <= 0 || ksThreshold >= 1 {
		ksThreshold = 0.10 // Default: 10% drift threshold
	}
	return &DriftDetector{
		recentScores: make([]float64, 0, maxRecent),
		maxRecent:    maxRecent,
		ksThreshold:  ksThreshold,
	}
}

// AddScore adds a recent score for drift monitoring.
func (dd *DriftDetector) AddScore(score float64) {
	dd.recentScores = append(dd.recentScores, score)
	if len(dd.recentScores) > dd.maxRecent {
		// FIFO eviction
		dd.recentScores = dd.recentScores[1:]
	}
}

// DetectDrift performs KS test between recent scores and calibration set.
// Returns (drifted, ksStatistic, pValue, error).
func (dd *DriftDetector) DetectDrift(calibrationSet *CalibrationSet) (bool, float64, float64, error) {
	// Need sufficient recent scores
	if len(dd.recentScores) < 30 {
		return false, 0, 1.0, fmt.Errorf("insufficient recent scores: %d < 30", len(dd.recentScores))
	}

	// Extract calibration scores
	calibrationSet.mu.RLock()
	calibScores := make([]float64, len(calibrationSet.scores))
	for i, s := range calibrationSet.scores {
		calibScores[i] = s.Score
	}
	calibrationSet.mu.RUnlock()

	if len(calibScores) < 30 {
		return false, 0, 1.0, fmt.Errorf("insufficient calibration scores: %d < 30", len(calibScores))
	}

	// Perform two-sample KS test
	ksStatistic := ksTest2Sample(dd.recentScores, calibScores)

	// Compute approximate p-value
	n1, n2 := float64(len(dd.recentScores)), float64(len(calibScores))
	ne := (n1 * n2) / (n1 + n2) // Effective sample size
	lambda := math.Sqrt(ne) * ksStatistic

	// Kolmogorov distribution approximation
	pValue := ksPValue(lambda)

	// Drift detected if p-value < significance level (typically 0.05)
	drifted := pValue < 0.05

	return drifted, ksStatistic, pValue, nil
}

// ksTest2Sample computes the two-sample Kolmogorov-Smirnov test statistic.
// Returns D = max |F1(x) - F2(x)| where F1, F2 are empirical CDFs.
func ksTest2Sample(sample1, sample2 []float64) float64 {
	// Sort both samples
	s1 := make([]float64, len(sample1))
	s2 := make([]float64, len(sample2))
	copy(s1, sample1)
	copy(s2, sample2)
	sort.Float64s(s1)
	sort.Float64s(s2)

	n1, n2 := float64(len(s1)), float64(len(s2))

	// Merge and compute ECDFs
	i, j := 0, 0
	maxD := 0.0

	for i < len(s1) && j < len(s2) {
		d1, d2 := s1[i], s2[j]

		// Empirical CDF values
		cdf1 := float64(i) / n1
		cdf2 := float64(j) / n2

		// Compute |F1(x) - F2(x)|
		diff := math.Abs(cdf1 - cdf2)
		if diff > maxD {
			maxD = diff
		}

		// Advance pointer for smaller value
		if d1 < d2 {
			i++
		} else if d2 < d1 {
			j++
		} else {
			// Equal values - advance both
			i++
			j++
		}
	}

	// Check remaining values
	for i < len(s1) {
		diff := math.Abs(float64(i)/n1 - 1.0)
		if diff > maxD {
			maxD = diff
		}
		i++
	}
	for j < len(s2) {
		diff := math.Abs(1.0 - float64(j)/n2)
		if diff > maxD {
			maxD = diff
		}
		j++
	}

	return maxD
}

// ksPValue computes the approximate p-value for KS test.
// Uses Kolmogorov distribution approximation.
func ksPValue(lambda float64) float64 {
	if lambda <= 0 {
		return 1.0
	}

	// Kolmogorov distribution: P(D > x) ≈ 2 * sum_{k=1}^∞ (-1)^{k-1} * exp(-2k²x²)
	// Approximation using first few terms
	sum := 0.0
	for k := 1; k <= 10; k++ {
		sign := 1.0
		if k%2 == 0 {
			sign = -1.0
		}
		term := sign * math.Exp(-2*float64(k*k)*lambda*lambda)
		sum += term
	}

	pValue := 2 * sum
	if pValue < 0 {
		pValue = 0
	}
	if pValue > 1 {
		pValue = 1
	}

	return pValue
}

// DriftReport contains drift detection results.
type DriftReport struct {
	Drifted           bool    `json:"drifted"`
	KSStatistic       float64 `json:"ks_statistic"`
	PValue            float64 `json:"p_value"`
	RecentN           int     `json:"recent_n"`
	CalibrationN      int     `json:"calibration_n"`
	RecommendRecalib  bool    `json:"recommend_recalibration"`
	Message           string  `json:"message"`
}

// CheckDrift performs drift detection and returns a comprehensive report.
func (dd *DriftDetector) CheckDrift(calibrationSet *CalibrationSet) DriftReport {
	report := DriftReport{
		RecentN:      len(dd.recentScores),
		CalibrationN: calibrationSet.Size(),
	}

	drifted, ks, pVal, err := dd.DetectDrift(calibrationSet)
	if err != nil {
		report.Message = fmt.Sprintf("Drift detection failed: %v", err)
		return report
	}

	report.Drifted = drifted
	report.KSStatistic = ks
	report.PValue = pVal

	// Recommend recalibration if drifted or KS stat > threshold
	report.RecommendRecalib = drifted || ks > dd.ksThreshold

	if report.Drifted {
		report.Message = fmt.Sprintf("DRIFT DETECTED: p-value %.4f < 0.05. Recalibration required.", pVal)
	} else if report.RecommendRecalib {
		report.Message = fmt.Sprintf("Drift warning: KS=%.4f > %.2f threshold. Consider recalibration.", ks, dd.ksThreshold)
	} else {
		report.Message = "No significant drift detected."
	}

	return report
}

// Reset clears recent scores (e.g., after recalibration).
func (dd *DriftDetector) Reset() {
	dd.recentScores = dd.recentScores[:0]
}

// MiscoverageMonitor tracks empirical miscoverage rate vs. target delta.
type MiscoverageMonitor struct {
	recentDecisions []bool // True = correctly accepted/rejected, False = miscoverage
	maxDecisions    int
}

// NewMiscoverageMonitor creates a new miscoverage monitor.
func NewMiscoverageMonitor(maxDecisions int) *MiscoverageMonitor {
	if maxDecisions <= 0 {
		maxDecisions = 1000
	}
	return &MiscoverageMonitor{
		recentDecisions: make([]bool, 0, maxDecisions),
		maxDecisions:    maxDecisions,
	}
}

// AddDecision records whether a decision was correct (true) or miscoverage (false).
func (mm *MiscoverageMonitor) AddDecision(correct bool) {
	mm.recentDecisions = append(mm.recentDecisions, correct)
	if len(mm.recentDecisions) > mm.maxDecisions {
		mm.recentDecisions = mm.recentDecisions[1:]
	}
}

// ComputeMiscoverage returns the empirical miscoverage rate.
// Miscoverage = (# incorrect decisions) / (# total decisions)
func (mm *MiscoverageMonitor) ComputeMiscoverage() (float64, int) {
	if len(mm.recentDecisions) == 0 {
		return 0, 0
	}

	errors := 0
	for _, correct := range mm.recentDecisions {
		if !correct {
			errors++
		}
	}

	rate := float64(errors) / float64(len(mm.recentDecisions))
	return rate, len(mm.recentDecisions)
}

// CheckCalibration compares empirical miscoverage to target delta.
// Returns (wellCalibrated, empiricalRate, targetDelta, nDecisions).
func (mm *MiscoverageMonitor) CheckCalibration(targetDelta float64) (bool, float64, float64, int) {
	empiricalRate, n := mm.ComputeMiscoverage()

	// Allow 50% relative error (e.g., delta=0.05 → accept 0.025-0.075)
	tolerance := targetDelta * 0.5
	wellCalibrated := math.Abs(empiricalRate-targetDelta) <= tolerance

	return wellCalibrated, empiricalRate, targetDelta, n
}
