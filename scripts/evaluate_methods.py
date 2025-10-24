#!/usr/bin/env python3
"""
Evaluation Runner for ASV vs Baseline Methods

Loads computed signals, baselines, and ground truth labels to evaluate
hallucination detection performance across multiple methods:

ASV Signals:
- D̂ (fractal dimension) - lower indicates repetitive/hallucinated text
- coh★ (directional coherence) - lower indicates semantic drift
- r_LZ (compressibility) - lower indicates repetitive patterns
- Combined ASV score (weighted ensemble)

Baseline Methods:
- Perplexity (GPT-2) - higher indicates unusual text
- Mean token probability - lower indicates low confidence
- Min token probability - very low values indicate weak links
- Entropy - lower indicates less diverse predictions

Evaluation Metrics:
- AUROC (Area Under ROC Curve) - threshold-independent performance
- AUPRC (Area Under Precision-Recall Curve) - better for imbalanced data
- F1 Score at optimal threshold
- Accuracy at optimal threshold
- Precision/Recall at optimal threshold

Usage:
    python evaluate_methods.py --benchmark truthfulqa --output results/
    python evaluate_methods.py --benchmark all --output results/
"""

import argparse
import json
import logging
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Add agent src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "src"))

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

    # Baseline metrics
    perplexity: Optional[float] = None
    log_perplexity: Optional[float] = None
    mean_token_prob: Optional[float] = None
    min_token_prob: Optional[float] = None
    entropy: Optional[float] = None

    def has_signals(self) -> bool:
        """Check if ASV signals are available."""
        return all([
            self.D_hat is not None,
            self.coh_star is not None,
            self.r_LZ is not None,
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

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
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


class EvaluationRunner:
    """Run evaluation comparing ASV signals vs baseline methods."""

    def __init__(self, benchmark: str):
        """
        Initialize evaluation runner.

        Args:
            benchmark: Benchmark name ('truthfulqa', 'fever', 'halueval')
        """
        self.benchmark = benchmark
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
            if not sample.has_signals():
                self.logger.warning(f"{sample_id}: Missing signals")
                continue

            if not sample.has_baselines():
                self.logger.warning(f"{sample_id}: Missing baselines")
                continue

            samples.append(sample)

        self.logger.info(f"Loaded {len(samples)} complete samples")
        return samples

    def compute_asv_combined_score(self, sample: EvaluationSample) -> float:
        """
        Compute combined ASV score (higher = more likely hallucination).

        Formula: score = w1*(1 - D̂_norm) + w2*(1 - coh★) + w3*(1 - r_LZ)
        where lower D̂, coh★, r indicate hallucination

        Args:
            sample: Evaluation sample

        Returns:
            Combined score in [0, 1]
        """
        # Normalize D̂ to [0, 1] range (typical: 0-3)
        D_norm = np.clip(sample.D_hat / 3.0, 0, 1)

        # coh★ already in [0, 1]
        coh_norm = sample.coh_star

        # r_LZ already in [0, 1]
        r_norm = sample.r_LZ

        # Lower values indicate hallucination, so invert
        # Weights: D̂ is most important, then coh★, then r
        w1, w2, w3 = 0.5, 0.3, 0.2

        score = w1 * (1 - D_norm) + w2 * (1 - coh_norm) + w3 * (1 - r_norm)

        return float(score)

    def evaluate_method(
        self,
        samples: List[EvaluationSample],
        scores: np.ndarray,
        method_name: str,
    ) -> MethodResults:
        """
        Evaluate a single method using various metrics.

        Args:
            samples: List of evaluation samples
            scores: Array of detection scores (higher = more likely hallucination)
            method_name: Name of the method

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
        )

    def run_evaluation(self) -> Dict[str, MethodResults]:
        """
        Run full evaluation comparing all methods.

        Returns:
            Dictionary mapping method name to results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for evaluation")

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running Evaluation: {self.benchmark.upper()}")
        self.logger.info(f"{'='*60}")

        # Load samples
        samples = self.load_samples()

        if len(samples) == 0:
            raise ValueError("No samples loaded. Check data directories.")

        results = {}

        # 1. ASV Signals (Individual)

        # D̂ (lower = hallucination, so invert for scoring)
        D_scores = np.array([1 - (s.D_hat / 3.0) for s in samples])
        D_scores = np.clip(D_scores, 0, 1)
        results['ASV_D_hat'] = self.evaluate_method(samples, D_scores, "ASV: D̂ (Fractal Dimension)")

        # coh★ (lower = hallucination, so invert)
        coh_scores = np.array([1 - s.coh_star for s in samples])
        results['ASV_coh_star'] = self.evaluate_method(samples, coh_scores, "ASV: coh★ (Coherence)")

        # r_LZ (lower = hallucination, so invert)
        r_scores = np.array([1 - s.r_LZ for s in samples])
        results['ASV_r_LZ'] = self.evaluate_method(samples, r_scores, "ASV: r (Compressibility)")

        # ASV Combined (weighted ensemble)
        asv_combined = np.array([self.compute_asv_combined_score(s) for s in samples])
        results['ASV_Combined'] = self.evaluate_method(samples, asv_combined, "ASV: Combined Score")

        # 2. Baseline Methods

        # Perplexity (higher = hallucination, normalize to [0, 1])
        perplexities = np.array([s.perplexity for s in samples])
        ppl_range = perplexities.max() - perplexities.min()
        if ppl_range > 1e-6:  # Use epsilon to avoid floating point issues
            ppl_normalized = (perplexities - perplexities.min()) / ppl_range
        else:
            # All perplexities are the same, use raw values normalized by mean
            ppl_normalized = perplexities / (perplexities.mean() + 1e-10)

        # Final safety check for NaN values
        if np.any(np.isnan(ppl_normalized)):
            self.logger.warning("NaN values detected in perplexity normalization, replacing with zeros")
            ppl_normalized = np.nan_to_num(ppl_normalized, nan=0.0)

        results['Baseline_Perplexity'] = self.evaluate_method(samples, ppl_normalized, "Baseline: Perplexity")

        # Mean token probability (lower = hallucination, so invert)
        mean_probs = np.array([s.mean_token_prob for s in samples])
        mean_prob_scores = 1 - mean_probs
        results['Baseline_MeanProb'] = self.evaluate_method(samples, mean_prob_scores, "Baseline: Mean Token Prob")

        # Min token probability (lower = hallucination, so invert and normalize)
        min_probs = np.array([s.min_token_prob for s in samples])
        # Use log scale for min prob since it can be very small
        min_prob_scores = 1 - np.log10(min_probs + 1e-10) / np.log10(min_probs.min() + 1e-10)
        min_prob_scores = np.clip(min_prob_scores, 0, 1)
        results['Baseline_MinProb'] = self.evaluate_method(samples, min_prob_scores, "Baseline: Min Token Prob")

        # Entropy (lower = hallucination, so invert and normalize)
        entropies = np.array([s.entropy for s in samples])
        entropy_normalized = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-10)
        entropy_scores = 1 - entropy_normalized
        results['Baseline_Entropy'] = self.evaluate_method(samples, entropy_scores, "Baseline: Entropy")

        self.logger.info(f"\n{'='*60}")
        self.logger.info("Evaluation Complete!")
        self.logger.info(f"{'='*60}\n")

        return results


def save_results(
    results: Dict[str, MethodResults],
    benchmark: str,
    output_dir: Path,
):
    """
    Save evaluation results to JSON file.

    Args:
        results: Dictionary of method results
        benchmark: Benchmark name
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{benchmark}_results.json"

    results_dict = {
        'benchmark': benchmark,
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
        description='Evaluate ASV signals vs baseline methods'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        choices=['truthfulqa', 'fever', 'halueval', 'all'],
        help='Which benchmark to evaluate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Process benchmarks
    benchmarks = ['truthfulqa', 'fever', 'halueval'] if args.benchmark == 'all' else [args.benchmark]

    output_dir = Path(args.output)

    for benchmark in benchmarks:
        try:
            runner = EvaluationRunner(benchmark)
            results = runner.run_evaluation()
            save_results(results, benchmark, output_dir)
        except Exception as e:
            logger.error(f"Failed to evaluate {benchmark}: {e}", exc_info=True)
            continue

    logger.info("\n✅ All evaluations complete!")


if __name__ == '__main__':
    main()
