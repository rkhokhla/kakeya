#!/usr/bin/env python3
"""
Conformal Evaluation Runner for ASV with Learned Ensemble Weights

This version uses split-conformal prediction with optimized ensemble weights,
integrating perplexity as a 4th signal alongside D̂, coh★, and r_LZ.

Key Features:
- Split-conformal prediction with finite-sample coverage guarantees
- Learned ensemble weights via AUC optimization
- Task-specific weight initialization (factuality vs degeneracy)
- Perplexity integrated as a core signal (not just baseline)
- Calibration set management with drift detection
- Miscoverage tracking and recalibration triggers

Usage:
    python evaluate_methods_conformal.py --benchmark truthfulqa --output results/
    python evaluate_methods_conformal.py --benchmark all --output results/
    python evaluate_methods_conformal.py --benchmark degeneracy --task degeneracy --output results/
"""

import argparse
import json
import logging
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Add agent src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "src"))

# Import conformal module
from conformal import (
    ConformalScore,
    CalibrationSet,
    ConformalPredictor,
    EnsembleWeights,
    optimize_ensemble_weights,
    split_calibration_data,
    DriftDetector,
)

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_recall_curve,
        roc_curve,
        f1_score,
        accuracy_score,
        precision_score,
        recall_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.error("scikit-learn not installed. Install with: pip install scikit-learn")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """Container for a single evaluation sample with all data."""
    sample_id: str
    benchmark: str
    is_hallucination: bool  # Ground truth label

    # ASV signals
    D_hat: Optional[float] = None
    coh_star: Optional[float] = None
    r_LZ: Optional[float] = None

    # Baseline metrics (perplexity now treated as signal)
    perplexity: Optional[float] = None
    log_perplexity: Optional[float] = None
    mean_token_prob: Optional[float] = None
    min_token_prob: Optional[float] = None
    entropy: Optional[float] = None

    def has_all_signals(self) -> bool:
        """Check if all signals including perplexity are available."""
        return all([
            self.D_hat is not None,
            self.coh_star is not None,
            self.r_LZ is not None,
            self.perplexity is not None,
        ])

    def has_baselines(self) -> bool:
        """Check if baseline metrics are available."""
        return all([
            self.perplexity is not None,
            self.mean_token_prob is not None,
            self.entropy is not None,
        ])


@dataclass
class MethodResults:
    """Evaluation results for a single method."""
    method_name: str
    auroc: float
    auprc: float
    f1_optimal: float
    accuracy_optimal: float
    precision_optimal: float
    recall_optimal: float
    optimal_threshold: float
    n_samples: int

    # Conformal-specific metrics
    conformal_threshold: Optional[float] = None
    coverage_guarantee: Optional[float] = None
    calibration_size: Optional[int] = None
    ensemble_weights: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            'method_name': self.method_name,
            'auroc': float(self.auroc),
            'auprc': float(self.auprc),
            'f1_optimal': float(self.f1_optimal),
            'accuracy_optimal': float(self.accuracy_optimal),
            'precision_optimal': float(self.precision_optimal),
            'recall_optimal': float(self.recall_optimal),
            'optimal_threshold': float(self.optimal_threshold),
            'n_samples': int(self.n_samples),
        }

        # Add conformal metrics if available
        if self.conformal_threshold is not None:
            result['conformal_threshold'] = float(self.conformal_threshold)
        if self.coverage_guarantee is not None:
            result['coverage_guarantee'] = float(self.coverage_guarantee)
        if self.calibration_size is not None:
            result['calibration_size'] = int(self.calibration_size)
        if self.ensemble_weights is not None:
            result['ensemble_weights'] = self.ensemble_weights

        return result


class ConformalEvaluationRunner:
    """Run evaluation using split-conformal prediction with learned ensemble weights."""

    def __init__(self, benchmark: str, task: str = "balanced", delta: float = 0.05):
        """
        Initialize conformal evaluation runner.

        Args:
            benchmark: Benchmark name ('truthfulqa', 'fever', 'halueval', 'degeneracy')
            task: Task type for weight initialization ('factuality', 'degeneracy', 'balanced')
            delta: Miscoverage level for conformal prediction (default: 0.05)
        """
        self.benchmark = benchmark
        self.task = task
        self.delta = delta
        self.logger = logging.getLogger(__name__)

        # Paths
        self.signals_dir = Path(f"data/signals/{benchmark}")
        self.baselines_dir = Path(f"data/baselines/{benchmark}")
        self.ground_truth_file = self._get_ground_truth_path()

    def _get_ground_truth_path(self) -> Path:
        """Get path to ground truth labels file."""
        if self.benchmark == 'truthfulqa':
            return Path("data/benchmarks/truthfulqa/TruthfulQA.csv")
        elif self.benchmark == 'fever':
            return Path("data/benchmarks/fever/shared_task_dev.jsonl")
        elif self.benchmark == 'halueval':
            return Path("data/benchmarks/halueval/qa_samples.json")
        elif self.benchmark == 'degeneracy':
            return Path("data/benchmarks/degeneracy/degeneracy_synthetic.jsonl")
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")

    def load_ground_truth(self) -> Dict[str, bool]:
        """
        Load ground truth hallucination labels.

        Returns:
            Dictionary mapping sample_id to is_hallucination (bool)
        """
        self.logger.info(f"Loading ground truth from LLM outputs...")

        labels = {}

        # Load from LLM outputs file which contains ground truth
        llm_outputs_file = Path(f"data/llm_outputs/{self.benchmark}_outputs.jsonl")

        if self.benchmark == 'truthfulqa':
            # TruthfulQA: Check if LLM response contains incorrect answer phrases
            with open(llm_outputs_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    sample_id = data['id']

                    llm_response = data['llm_response'].lower()

                    # Extract incorrect answers from metadata
                    incorrect_answers = data['metadata']['incorrect_answers'].lower().split(';')
                    incorrect_answers = [ans.strip() for ans in incorrect_answers]

                    # Check if LLM response contains any incorrect answer phrases
                    is_hallucination = False
                    for incorrect in incorrect_answers:
                        if len(incorrect) > 10 and incorrect in llm_response:
                            is_hallucination = True
                            break

                    labels[sample_id] = is_hallucination

        elif self.benchmark == 'fever':
            # FEVER has explicit "SUPPORTS", "REFUTES", "NOT ENOUGH INFO" labels
            # REFUTES → hallucination (claim contradicts evidence)
            with open(llm_outputs_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    sample_id = data['id']
                    # Use ground_truth boolean field or check label
                    if 'ground_truth' in data:
                        labels[sample_id] = data['ground_truth']
                    else:
                        labels[sample_id] = (data['metadata']['label'] == 'REFUTES')

        elif self.benchmark == 'halueval':
            # HaluEval has explicit 'hallucination' field in metadata
            with open(llm_outputs_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    sample_id = data['id']
                    # Use ground truth from metadata
                    labels[sample_id] = data['metadata'].get('hallucination', False)

        elif self.benchmark == 'degeneracy':
            # Degeneracy dataset: load directly from synthetic JSONL (no LLM outputs)
            degeneracy_file = Path("data/benchmarks/degeneracy/degeneracy_synthetic.jsonl")
            with open(degeneracy_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    sample_id = data['id']
                    # Use is_degenerate field as ground truth
                    labels[sample_id] = data['is_degenerate']

        self.logger.info(f"Loaded {len(labels)} ground truth labels")
        pos_count = sum(labels.values())
        self.logger.info(f"  Positive samples (hallucinations): {pos_count} ({pos_count/len(labels)*100:.1f}%)")

        return labels

    def load_samples(self) -> List[EvaluationSample]:
        """
        Load all evaluation samples with signals, baselines, and ground truth.

        Returns:
            List of EvaluationSample objects
        """
        self.logger.info("Loading evaluation samples...")

        # Load ground truth
        ground_truth = self.load_ground_truth()

        samples = []

        # Iterate through signal files (these exist for all samples we processed)
        for signal_file in sorted(self.signals_dir.glob("*.json")):
            sample_id = signal_file.stem

            # Skip if no ground truth (shouldn't happen)
            if sample_id not in ground_truth:
                self.logger.warning(f"No ground truth for {sample_id}, skipping")
                continue

            # Load signals
            with open(signal_file) as f:
                signals = json.load(f)

            # Load baselines
            baseline_file = self.baselines_dir / f"{sample_id}.json"
            if not baseline_file.exists():
                self.logger.warning(f"No baselines for {sample_id}, skipping")
                continue

            with open(baseline_file) as f:
                baselines = json.load(f)

            # Create evaluation sample
            sample = EvaluationSample(
                sample_id=sample_id,
                benchmark=self.benchmark,
                is_hallucination=ground_truth[sample_id],
                D_hat=signals.get('D_hat'),
                coh_star=signals.get('coh_star'),
                r_LZ=signals.get('r_LZ'),
                perplexity=baselines.get('perplexity'),
                log_perplexity=baselines.get('log_perplexity'),
                mean_token_prob=baselines.get('mean_token_prob'),
                min_token_prob=baselines.get('min_token_prob'),
                entropy=baselines.get('entropy'),
            )

            # Validate
            if not sample.has_all_signals():
                self.logger.warning(f"{sample_id}: Missing signals (D_hat={sample.D_hat}, coh_star={sample.coh_star}, r_LZ={sample.r_LZ}, perplexity={sample.perplexity})")
                continue

            if not sample.has_baselines():
                self.logger.warning(f"{sample_id}: Missing baselines")
                continue

            samples.append(sample)

        self.logger.info(f"Loaded {len(samples)} complete samples")
        return samples

    def evaluate_method(
        self,
        samples: List[EvaluationSample],
        scores: np.ndarray,
        method_name: str,
        conformal_metadata: Optional[Dict] = None,
    ) -> MethodResults:
        """
        Evaluate a single method using various metrics.

        Args:
            samples: List of evaluation samples
            scores: Array of detection scores (higher = more likely hallucination)
            method_name: Name of the method
            conformal_metadata: Optional conformal prediction metadata

        Returns:
            MethodResults containing all evaluation metrics
        """
        # Extract ground truth labels
        y_true = np.array([s.is_hallucination for s in samples])
        y_scores = np.array(scores)

        # AUROC (threshold-independent)
        auroc = roc_auc_score(y_true, y_scores)

        # AUPRC (better for imbalanced data)
        auprc = average_precision_score(y_true, y_scores)

        # Find optimal threshold using F1 score
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        # Compute metrics at optimal threshold
        y_pred = (y_scores >= optimal_threshold).astype(int)
        f1_optimal = f1_score(y_true, y_pred)
        accuracy_optimal = accuracy_score(y_true, y_pred)
        precision_optimal = precision_score(y_true, y_pred, zero_division=0)
        recall_optimal = recall_score(y_true, y_pred, zero_division=0)

        self.logger.info(f"\n{method_name}:")
        self.logger.info(f"  AUROC: {auroc:.4f}")
        self.logger.info(f"  AUPRC: {auprc:.4f}")
        self.logger.info(f"  F1 (optimal): {f1_optimal:.4f}")
        self.logger.info(f"  Accuracy (optimal): {accuracy_optimal:.4f}")
        self.logger.info(f"  Precision (optimal): {precision_optimal:.4f}")
        self.logger.info(f"  Recall (optimal): {recall_optimal:.4f}")
        self.logger.info(f"  Optimal threshold: {optimal_threshold:.4f}")

        # Add conformal metadata if available
        conformal_threshold = None
        coverage_guarantee = None
        calibration_size = None
        ensemble_weights = None

        if conformal_metadata:
            conformal_threshold = conformal_metadata.get('threshold')
            coverage_guarantee = conformal_metadata.get('coverage_guarantee')
            calibration_size = conformal_metadata.get('calibration_size')
            ensemble_weights = conformal_metadata.get('weights')

            if conformal_threshold is not None:
                self.logger.info(f"  Conformal threshold (δ={self.delta}): {conformal_threshold:.4f}")
            if coverage_guarantee is not None:
                self.logger.info(f"  Coverage guarantee: {coverage_guarantee:.1%}")
            if calibration_size is not None:
                self.logger.info(f"  Calibration size: {calibration_size}")
            if ensemble_weights:
                self.logger.info(f"  Ensemble weights: D̂={ensemble_weights['D_hat']:.3f}, "
                               f"coh★={ensemble_weights['coh_star']:.3f}, "
                               f"r_LZ={ensemble_weights['r_LZ']:.3f}, "
                               f"perplexity={ensemble_weights['perplexity']:.3f}")

        return MethodResults(
            method_name=method_name,
            auroc=auroc,
            auprc=auprc,
            f1_optimal=f1_optimal,
            accuracy_optimal=accuracy_optimal,
            precision_optimal=precision_optimal,
            recall_optimal=recall_optimal,
            optimal_threshold=optimal_threshold,
            n_samples=len(samples),
            conformal_threshold=conformal_threshold,
            coverage_guarantee=coverage_guarantee,
            calibration_size=calibration_size,
            ensemble_weights=ensemble_weights,
        )

    def run_evaluation(self) -> Dict[str, MethodResults]:
        """
        Run full evaluation using split-conformal prediction.

        Returns:
            Dictionary mapping method name to results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for evaluation")

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Conformal Evaluation: {self.benchmark.upper()} (task={self.task}, δ={self.delta})")
        self.logger.info(f"{'='*70}")

        # Load samples
        samples = self.load_samples()

        if len(samples) == 0:
            raise ValueError("No samples loaded. Check data directories.")

        results = {}

        # ========================================
        # CONFORMAL ASV ENSEMBLE WITH PERPLEXITY
        # ========================================

        self.logger.info("\n--- Learning Ensemble Weights ---")

        # Convert samples to ConformalScore objects
        conformal_scores = [
            ConformalScore(
                sample_id=s.sample_id,
                ensemble_score=0.0,  # Will be computed after weight optimization
                D_hat=s.D_hat,
                coh_star=s.coh_star,
                r_LZ=s.r_LZ,
                perplexity=s.perplexity,
                label=s.is_hallucination
            )
            for s in samples
        ]

        # Split into calibration and test (20% calibration, 80% test)
        cal_scores, test_scores = split_calibration_data(conformal_scores, calibration_fraction=0.2, seed=42)

        # Optimize ensemble weights on calibration set
        self.logger.info(f"Optimizing ensemble weights (task={self.task})...")
        weights = optimize_ensemble_weights(cal_scores, task=self.task)
        self.logger.info(f"Optimized weights: D̂={weights.w_D_hat:.3f}, coh★={weights.w_coh_star:.3f}, "
                        f"r_LZ={weights.w_r_LZ:.3f}, perplexity={weights.w_perplexity:.3f}")

        # Compute ensemble scores with optimized weights
        from conformal import compute_ensemble_score
        for score in conformal_scores:
            score.ensemble_score = compute_ensemble_score(
                score.D_hat, score.coh_star, score.r_LZ, score.perplexity, weights
            )

        # Create calibration set
        cal_set = CalibrationSet(max_size=1000, max_age_days=30)
        for score in cal_scores:
            cal_set.add(score)

        self.logger.info(f"Calibration set: {cal_set.stats()}")

        # Create conformal predictor
        predictor = ConformalPredictor(cal_set, weights, delta=self.delta)
        self.logger.info(f"Conformal threshold (δ={self.delta}): {predictor.threshold:.4f}")

        # Evaluate on full dataset using ensemble scores
        ensemble_scores = np.array([s.ensemble_score for s in conformal_scores])

        conformal_metadata = {
            'threshold': predictor.threshold,
            'coverage_guarantee': 1 - self.delta,
            'calibration_size': len(cal_scores),
            'weights': {
                'D_hat': weights.w_D_hat,
                'coh_star': weights.w_coh_star,
                'r_LZ': weights.w_r_LZ,
                'perplexity': weights.w_perplexity,
            }
        }

        results['ASV_Conformal_Ensemble'] = self.evaluate_method(
            samples,
            ensemble_scores,
            "ASV: Conformal Ensemble (D̂ + coh★ + r_LZ + perplexity)",
            conformal_metadata=conformal_metadata
        )

        # ========================================
        # INDIVIDUAL SIGNALS (for comparison)
        # ========================================

        self.logger.info("\n--- Individual Signals (Baseline Comparison) ---")

        # D̂ (lower = hallucination, so invert for scoring)
        D_scores = np.array([1 - (s.D_hat / 3.0) for s in samples])
        D_scores = np.clip(D_scores, 0, 1)
        results['ASV_D_hat'] = self.evaluate_method(samples, D_scores, "ASV: D̂ (Fractal Dimension)")

        # coh★ (lower = hallucination, so invert)
        coh_scores = np.array([1 - s.coh_star for s in samples])
        results['ASV_coh_star'] = self.evaluate_method(samples, coh_scores, "ASV: coh★ (Coherence)")

        # r_LZ (lower = hallucination, so invert)
        r_scores = np.array([1 - s.r_LZ for s in samples])
        results['ASV_r_LZ'] = self.evaluate_method(samples, r_scores, "ASV: r_LZ (Compressibility)")

        # Perplexity (higher = hallucination, normalize to [0, 1])
        perplexities = np.array([s.perplexity for s in samples])
        ppl_range = perplexities.max() - perplexities.min()
        if ppl_range > 1e-6:
            ppl_normalized = (perplexities - perplexities.min()) / ppl_range
        else:
            ppl_normalized = perplexities / (perplexities.mean() + 1e-10)

        if np.any(np.isnan(ppl_normalized)):
            self.logger.warning("NaN values in perplexity normalization, replacing with zeros")
            ppl_normalized = np.nan_to_num(ppl_normalized, nan=0.0)

        results['Baseline_Perplexity'] = self.evaluate_method(samples, ppl_normalized, "Baseline: Perplexity")

        # Other baselines
        mean_probs = np.array([s.mean_token_prob for s in samples])
        mean_prob_scores = 1 - mean_probs
        results['Baseline_MeanProb'] = self.evaluate_method(samples, mean_prob_scores, "Baseline: Mean Token Prob")

        min_probs = np.array([s.min_token_prob for s in samples])
        min_prob_scores = 1 - np.log10(min_probs + 1e-10) / np.log10(min_probs.min() + 1e-10)
        min_prob_scores = np.clip(min_prob_scores, 0, 1)
        results['Baseline_MinProb'] = self.evaluate_method(samples, min_prob_scores, "Baseline: Min Token Prob")

        entropies = np.array([s.entropy for s in samples])
        entropy_normalized = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-10)
        entropy_scores = 1 - entropy_normalized
        results['Baseline_Entropy'] = self.evaluate_method(samples, entropy_scores, "Baseline: Entropy")

        self.logger.info(f"\n{'='*70}")
        self.logger.info("Evaluation Complete!")
        self.logger.info(f"{'='*70}\n")

        return results


def save_results(
    results: Dict[str, MethodResults],
    benchmark: str,
    task: str,
    delta: float,
    output_dir: Path,
):
    """
    Save evaluation results to JSON file.

    Args:
        results: Dictionary of method results
        benchmark: Benchmark name
        task: Task type
        delta: Miscoverage level
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{benchmark}_conformal_results.json"

    results_dict = {
        'benchmark': benchmark,
        'task': task,
        'delta': delta,
        'methods': {name: res.to_dict() for name, res in results.items()},
        'summary': {
            'best_auroc': max(r.auroc for r in results.values()),
            'best_auprc': max(r.auprc for r in results.values()),
            'best_f1': max(r.f1_optimal for r in results.values()),
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"Results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate ASV with split-conformal prediction and learned ensemble weights'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        choices=['truthfulqa', 'fever', 'halueval', 'degeneracy', 'all'],
        help='Which benchmark to evaluate'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='balanced',
        choices=['factuality', 'degeneracy', 'balanced'],
        help='Task type for weight initialization (default: balanced)'
    )
    parser.add_argument(
        '--delta',
        type=float,
        default=0.05,
        help='Miscoverage level for conformal prediction (default: 0.05)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Process benchmarks
    benchmarks = ['truthfulqa', 'fever', 'halueval', 'degeneracy'] if args.benchmark == 'all' else [args.benchmark]

    output_dir = Path(args.output)

    for benchmark in benchmarks:
        try:
            # Auto-select task based on benchmark
            task = args.task
            if task == 'balanced':
                task = 'degeneracy' if benchmark == 'degeneracy' else 'factuality'

            logger.info(f"\n{'#'*70}")
            logger.info(f"# Processing: {benchmark.upper()} (task={task})")
            logger.info(f"{'#'*70}")

            runner = ConformalEvaluationRunner(benchmark, task=task, delta=args.delta)
            results = runner.run_evaluation()
            save_results(results, benchmark, task, args.delta, output_dir)
        except Exception as e:
            logger.error(f"Failed to evaluate {benchmark}: {e}", exc_info=True)
            continue

    logger.info("\n✅ All evaluations complete!")


if __name__ == '__main__':
    main()
