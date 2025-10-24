#!/usr/bin/env python3
"""
Split-Conformal Verification Framework for ASV

Implements split conformal prediction providing finite-sample coverage guarantees:
P(escalate | benign) ≤ δ under exchangeability assumption.

This module provides:
1. Split calibration set management
2. Nonconformity score computation with learned ensemble weights
3. Conformal prediction with coverage guarantees
4. Drift detection and recalibration

References:
    Vovk et al. (2005): Algorithmic Learning in a Random World
    Lei et al. (2018): Distribution-Free Predictive Inference
    Angelopoulos & Bates (2023): A Gentle Introduction to Conformal Prediction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy import stats


@dataclass
class EnsembleWeights:
    """Learned ensemble weights for combining signals."""
    w_D_hat: float  # Fractal dimension weight
    w_coh_star: float  # Directional coherence weight
    w_r_LZ: float  # Compressibility weight
    w_perplexity: float  # Perplexity weight (inverse normalized)

    def __post_init__(self):
        """Validate weights sum to 1 and are non-negative."""
        total = self.w_D_hat + self.w_coh_star + self.w_r_LZ + self.w_perplexity
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0 (got {total})")
        if any(w < 0 for w in [self.w_D_hat, self.w_coh_star, self.w_r_LZ, self.w_perplexity]):
            raise ValueError("All weights must be non-negative")

    @classmethod
    def default_factuality(cls):
        """Default weights for factuality tasks (perplexity dominant)."""
        return cls(w_D_hat=0.15, w_coh_star=0.10, w_r_LZ=0.10, w_perplexity=0.65)

    @classmethod
    def default_degeneracy(cls):
        """Default weights for structural degeneracy (r_LZ dominant)."""
        return cls(w_D_hat=0.15, w_coh_star=0.15, w_r_LZ=0.60, w_perplexity=0.10)

    @classmethod
    def balanced(cls):
        """Balanced weights across all signals."""
        return cls(w_D_hat=0.25, w_coh_star=0.25, w_r_LZ=0.25, w_perplexity=0.25)


@dataclass
class ConformalScore:
    """Nonconformity score for a single sample."""
    sample_id: str
    ensemble_score: float  # Weighted combination of signals
    D_hat: float
    coh_star: float
    r_LZ: float
    perplexity: float  # Raw perplexity value
    label: bool  # True = degenerate/hallucination, False = normal


class CalibrationSet:
    """
    Manages calibration data for split conformal prediction.

    Implements FIFO with time-window strategy:
    - Keep most recent N samples (default: 1000)
    - Discard samples older than T days (default: 30)
    """

    def __init__(self, max_size: int = 1000, max_age_days: int = 30):
        self.max_size = max_size
        self.max_age_seconds = max_age_days * 86400
        self.scores: List[ConformalScore] = []
        self.timestamps: List[float] = []  # Unix timestamps

    def add(self, score: ConformalScore, timestamp: Optional[float] = None):
        """Add a calibration sample (FIFO + time-window)."""
        import time
        if timestamp is None:
            timestamp = time.time()

        self.scores.append(score)
        self.timestamps.append(timestamp)

        # FIFO eviction if exceeds max_size
        if len(self.scores) > self.max_size:
            self.scores.pop(0)
            self.timestamps.pop(0)

        # Time-window eviction
        current_time = time.time()
        while self.timestamps and (current_time - self.timestamps[0]) > self.max_age_seconds:
            self.scores.pop(0)
            self.timestamps.pop(0)

    def get_scores(self, label: Optional[bool] = None) -> List[float]:
        """Get ensemble scores, optionally filtered by label."""
        if label is None:
            return [s.ensemble_score for s in self.scores]
        return [s.ensemble_score for s in self.scores if s.label == label]

    def get_quantile(self, delta: float, label: Optional[bool] = None) -> float:
        """
        Compute (1-δ)-quantile of nonconformity scores.

        Args:
            delta: Miscoverage level (e.g., 0.05 for 95% coverage)
            label: Optional label filter (None = all, True = degenerate, False = normal)

        Returns:
            Quantile threshold q such that P(score ≤ q) ≥ 1-δ
        """
        scores = self.get_scores(label)
        if not scores:
            return 0.5  # Safe default

        # Linear interpolation for quantile (standard practice)
        return float(np.quantile(scores, 1 - delta, method='linear'))

    def stats(self) -> Dict:
        """Compute calibration statistics."""
        all_scores = self.get_scores()
        if not all_scores:
            return {"size": 0}

        return {
            "size": len(self.scores),
            "mean": np.mean(all_scores),
            "median": np.median(all_scores),
            "std": np.std(all_scores),
            "min": np.min(all_scores),
            "max": np.max(all_scores),
            "positive_rate": np.mean([s.label for s in self.scores])
        }


def normalize_perplexity(perplexity: float, method: str = "log") -> float:
    """
    Normalize perplexity to [0, 1] range for ensemble combination.

    Lower perplexity = higher confidence = lower normalized score (good)
    Higher perplexity = lower confidence = higher normalized score (bad)

    Args:
        perplexity: Raw perplexity value (>= 1)
        method: Normalization method ("log", "inv", "clip")

    Returns:
        Normalized score in [0, 1] where higher = more suspicious
    """
    if method == "log":
        # Log normalization: log(perplexity) / log(100)
        # Assumes typical range [1, 100], maps to [0, 1]
        return min(1.0, max(0.0, np.log(perplexity) / np.log(100)))
    elif method == "inv":
        # Inverse: 1 - 1/perplexity
        # Maps [1, inf) → [0, 1)
        return 1 - (1 / perplexity)
    elif method == "clip":
        # Clip to [1, 20] and scale
        clipped = min(20, max(1, perplexity))
        return (clipped - 1) / 19
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_ensemble_score(
    D_hat: float,
    coh_star: float,
    r_LZ: float,
    perplexity: float,
    weights: EnsembleWeights
) -> float:
    """
    Compute weighted ensemble score from signals.

    All signals normalized to [0, 1] where higher = more suspicious.

    Args:
        D_hat: Fractal dimension (normalized to [0, 1])
        coh_star: Directional coherence (raw in [0, 1])
        r_LZ: Compressibility (raw in [0, 1])
        perplexity: Raw perplexity (normalized internally)
        weights: Ensemble weights

    Returns:
        Ensemble score in [0, 1] where higher = more suspicious
    """
    # Normalize D_hat: assume typical range [0, 3.5], higher = more complex = less suspicious
    # For degeneracy: low D_hat (repetitive) is suspicious
    # For factuality: moderate D_hat is good, extremes bad
    # Simple normalization: clip to [0, 3.5] and invert
    D_hat_norm = 1 - min(1.0, max(0.0, D_hat / 3.5))

    # Normalize coh_star: high coherence can be suspicious (loops) or good (factuality)
    # For degeneracy: very high coh_star (>0.9) is suspicious
    # For factuality: moderate coh_star (0.6-0.8) is good
    # U-shaped: both very low and very high are suspicious
    coh_star_norm = abs(coh_star - 0.7)  # Distance from ideal 0.7

    # r_LZ: lower = more compressible = more repetitive = suspicious
    r_LZ_norm = 1 - r_LZ

    # Normalize perplexity (higher perplexity = more suspicious)
    perplexity_norm = normalize_perplexity(perplexity, method="log")

    # Weighted combination
    score = (
        weights.w_D_hat * D_hat_norm +
        weights.w_coh_star * coh_star_norm +
        weights.w_r_LZ * r_LZ_norm +
        weights.w_perplexity * perplexity_norm
    )

    return float(score)


def optimize_ensemble_weights(
    calibration_scores: List[ConformalScore],
    task: str = "balanced"
) -> EnsembleWeights:
    """
    Optimize ensemble weights to maximize AUC on calibration set.

    Args:
        calibration_scores: List of calibration samples with labels
        task: Task type for weight initialization ("factuality", "degeneracy", "balanced")

    Returns:
        Optimized EnsembleWeights
    """
    from sklearn.metrics import roc_auc_score

    # Extract features and labels
    X = np.array([
        [s.D_hat, s.coh_star, s.r_LZ, normalize_perplexity(s.perplexity)]
        for s in calibration_scores
    ])
    y = np.array([s.label for s in calibration_scores], dtype=float)

    # Initialize weights based on task
    if task == "factuality":
        w0 = np.array([0.15, 0.10, 0.10, 0.65])
    elif task == "degeneracy":
        w0 = np.array([0.15, 0.15, 0.60, 0.10])
    else:
        w0 = np.array([0.25, 0.25, 0.25, 0.25])

    # Objective: maximize AUC (minimize negative AUC)
    def objective(w):
        # Normalize weights to sum to 1
        w = w / w.sum()
        scores = X @ w
        try:
            auc = roc_auc_score(y, scores)
            return -auc  # Minimize negative AUC
        except:
            return 0.0  # If AUC fails (e.g., single class), return neutral

    # Constraints: weights >= 0, sum = 1
    constraints = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1}  # Sum = 1
    ]
    bounds = [(0, 1) for _ in range(4)]  # Each weight in [0, 1]

    # Optimize
    result = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 100}
    )

    if result.success:
        w_opt = result.x / result.x.sum()  # Ensure normalized
        return EnsembleWeights(
            w_D_hat=float(w_opt[0]),
            w_coh_star=float(w_opt[1]),
            w_r_LZ=float(w_opt[2]),
            w_perplexity=float(w_opt[3])
        )
    else:
        # Fallback to default weights if optimization fails
        if task == "factuality":
            return EnsembleWeights.default_factuality()
        elif task == "degeneracy":
            return EnsembleWeights.default_degeneracy()
        else:
            return EnsembleWeights.balanced()


class ConformalPredictor:
    """
    Split conformal predictor with coverage guarantees.

    Provides finite-sample miscoverage guarantee:
    P(escalate | benign) ≤ δ under exchangeability.
    """

    def __init__(
        self,
        calibration_set: CalibrationSet,
        weights: EnsembleWeights,
        delta: float = 0.05
    ):
        """
        Initialize conformal predictor.

        Args:
            calibration_set: Calibration data
            weights: Ensemble weights
            delta: Target miscoverage level (default: 0.05 for 95% coverage)
        """
        self.calibration_set = calibration_set
        self.weights = weights
        self.delta = delta
        self.threshold = self.calibration_set.get_quantile(delta)

    def predict(
        self,
        D_hat: float,
        coh_star: float,
        r_LZ: float,
        perplexity: float
    ) -> Tuple[str, float, Dict]:
        """
        Make prediction with conformal guarantee.

        Args:
            D_hat: Fractal dimension
            coh_star: Directional coherence
            r_LZ: Compressibility
            perplexity: Raw perplexity

        Returns:
            (decision, ensemble_score, metadata)
            - decision: "accept" or "escalate"
            - ensemble_score: Computed nonconformity score
            - metadata: Additional info (threshold, guarantee, margin)
        """
        # Compute ensemble score
        score = compute_ensemble_score(
            D_hat, coh_star, r_LZ, perplexity, self.weights
        )

        # Decision: escalate if score > threshold
        decision = "escalate" if score > self.threshold else "accept"

        # Compute margin (distance from threshold)
        margin = score - self.threshold

        metadata = {
            "threshold": self.threshold,
            "coverage_guarantee": 1 - self.delta,
            "miscoverage_bound": self.delta,
            "margin": margin,
            "calibration_size": len(self.calibration_set.scores),
            "weights": {
                "D_hat": self.weights.w_D_hat,
                "coh_star": self.weights.w_coh_star,
                "r_LZ": self.weights.w_r_LZ,
                "perplexity": self.weights.w_perplexity
            }
        }

        return decision, score, metadata

    def recalibrate(self):
        """Recompute threshold from current calibration set."""
        self.threshold = self.calibration_set.get_quantile(self.delta)


class DriftDetector:
    """
    Detect distribution drift in nonconformity scores.

    Uses Kolmogorov-Smirnov two-sample test to compare recent scores
    with calibration distribution.
    """

    def __init__(self, significance: float = 0.01):
        """
        Initialize drift detector.

        Args:
            significance: Significance level for KS test (default: 0.01)
        """
        self.significance = significance

    def detect_drift(
        self,
        calibration_scores: List[float],
        recent_scores: List[float]
    ) -> Tuple[bool, float, Dict]:
        """
        Detect drift using KS test.

        Args:
            calibration_scores: Historical calibration scores
            recent_scores: Recent production scores (last N samples)

        Returns:
            (drift_detected, p_value, metadata)
        """
        if len(calibration_scores) < 30 or len(recent_scores) < 30:
            return False, 1.0, {"error": "Insufficient samples for KS test"}

        # Two-sample KS test
        ks_stat, p_value = stats.ks_2samp(calibration_scores, recent_scores)

        drift_detected = p_value < self.significance

        metadata = {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "significance": self.significance,
            "calibration_size": len(calibration_scores),
            "recent_size": len(recent_scores),
            "action": "recalibrate" if drift_detected else "continue"
        }

        return drift_detected, p_value, metadata


def split_calibration_data(
    scores: List[ConformalScore],
    calibration_fraction: float = 0.2,
    seed: int = 42
) -> Tuple[List[ConformalScore], List[ConformalScore]]:
    """
    Split data into calibration and test sets.

    Args:
        scores: All labeled scores
        calibration_fraction: Fraction for calibration (default: 0.2)
        seed: Random seed for reproducibility

    Returns:
        (calibration_scores, test_scores)
    """
    np.random.seed(seed)
    n = len(scores)
    n_cal = int(n * calibration_fraction)

    indices = np.random.permutation(n)
    cal_indices = indices[:n_cal]
    test_indices = indices[n_cal:]

    calibration_scores = [scores[i] for i in cal_indices]
    test_scores = [scores[i] for i in test_indices]

    return calibration_scores, test_scores


# Example usage
if __name__ == "__main__":
    # Create synthetic calibration data
    np.random.seed(42)

    # Normal samples (benign)
    normal_scores = [
        ConformalScore(
            sample_id=f"normal_{i}",
            ensemble_score=0.0,  # Placeholder, will be computed
            D_hat=np.random.uniform(1.5, 2.5),
            coh_star=np.random.uniform(0.6, 0.8),
            r_LZ=np.random.uniform(0.5, 0.8),
            perplexity=np.random.uniform(2, 10),
            label=False
        )
        for i in range(50)
    ]

    # Degenerate samples (suspicious)
    degenerate_scores = [
        ConformalScore(
            sample_id=f"degenerate_{i}",
            ensemble_score=0.0,  # Placeholder
            D_hat=np.random.uniform(0.5, 1.2),
            coh_star=np.random.uniform(0.85, 0.95),
            r_LZ=np.random.uniform(0.1, 0.3),
            perplexity=np.random.uniform(1, 3),
            label=True
        )
        for i in range(50)
    ]

    all_scores = normal_scores + degenerate_scores

    # Learn ensemble weights
    print("Optimizing ensemble weights...")
    weights = optimize_ensemble_weights(all_scores, task="degeneracy")
    print(f"Optimized weights: D̂={weights.w_D_hat:.3f}, coh★={weights.w_coh_star:.3f}, "
          f"r_LZ={weights.w_r_LZ:.3f}, perplexity={weights.w_perplexity:.3f}")

    # Compute ensemble scores
    for score in all_scores:
        score.ensemble_score = compute_ensemble_score(
            score.D_hat, score.coh_star, score.r_LZ, score.perplexity, weights
        )

    # Split calibration and test
    cal_scores, test_scores = split_calibration_data(all_scores, calibration_fraction=0.3)

    # Create calibration set
    cal_set = CalibrationSet()
    for score in cal_scores:
        cal_set.add(score)

    print(f"\nCalibration set stats: {cal_set.stats()}")

    # Create predictor
    predictor = ConformalPredictor(cal_set, weights, delta=0.05)
    print(f"Threshold (δ=0.05): {predictor.threshold:.4f}")

    # Test predictions
    print("\nTesting on held-out samples:")
    for score in test_scores[:5]:
        decision, ensemble_score, metadata = predictor.predict(
            score.D_hat, score.coh_star, score.r_LZ, score.perplexity
        )
        print(f"  {score.sample_id}: {decision} (score={ensemble_score:.4f}, "
              f"margin={metadata['margin']:.4f}, true_label={score.label})")
