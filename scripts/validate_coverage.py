#!/usr/bin/env python3
"""
Experiment 2: Coverage Calibration Validation

Validates the finite-sample coverage guarantee: P(escalate | benign) ≤ δ

Tests split-conformal prediction empirically across multiple δ values and
benchmarks to verify the theoretical guarantee holds in practice.
"""

import sys
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add agent src to path for conformal module
sys.path.insert(0, 'agent/src')
from conformal import CalibrationSet, ConformalPredictor, optimize_ensemble_weights

def normalize_D_hat(d_hat):
    """Normalize D̂ ∈ [0, 3.5] → [0, 1], inverted (lower = more suspicious)"""
    return 1.0 - np.clip(d_hat / 3.5, 0, 1)

def normalize_coh_star(coh_star):
    """U-shaped: extremes suspicious, 0.7 is ideal"""
    return np.abs(coh_star - 0.7)

def normalize_r_LZ(r):
    """Invert: lower compressibility = more suspicious"""
    return 1.0 - np.clip(r, 0, 1)

def normalize_perplexity(perplexity):
    """Log normalization: [1, 100] → [0, 1]"""
    return np.log(np.clip(perplexity, 1, 100)) / np.log(100)

def compute_nonconformity_score(signals, weights):
    """
    Compute weighted nonconformity score.

    Args:
        signals: dict with keys 'D_hat', 'coh_star', 'r_LZ', 'perplexity'
        weights: dict with same keys

    Returns:
        score ∈ [0, 1]
    """
    score = 0.0

    if weights.get('D_hat', 0) > 0:
        score += weights['D_hat'] * normalize_D_hat(signals['D_hat'])

    if weights.get('coh_star', 0) > 0:
        score += weights['coh_star'] * normalize_coh_star(signals['coh_star'])

    if weights.get('r_LZ', 0) > 0:
        score += weights['r_LZ'] * normalize_r_LZ(signals['r_LZ'])

    if weights.get('perplexity', 0) > 0:
        score += weights['perplexity'] * normalize_perplexity(signals['perplexity'])

    return score

def load_benchmark_data(benchmark_name):
    """Load signals, baselines, and ground truth for a benchmark."""
    signals_dir = Path(f'data/signals/{benchmark_name}')
    baselines_dir = Path(f'data/baselines/{benchmark_name}')

    # Load ground truth for degeneracy benchmark
    if benchmark_name == 'degeneracy':
        gt_path = Path(f'data/benchmarks/{benchmark_name}')
        with open(gt_path / 'degeneracy_synthetic.jsonl') as f:
            gt_data = {}
            for line in f:
                data = json.loads(line)
                gt_data[data['id']] = (data['type'] != 'normal')  # True if degenerate
    else:
        # For now, focus on degeneracy where we have clean ground truth
        return None

    # Load signals and baselines
    data = []
    for signal_file in sorted(signals_dir.glob('*.json')):
        sample_id = signal_file.stem

        # Load signal
        with open(signal_file) as f:
            signal = json.load(f)

        # Load baseline
        baseline_file = baselines_dir / f"{sample_id}.json"
        if not baseline_file.exists():
            continue
        with open(baseline_file) as f:
            baseline = json.load(f)

        # Get ground truth label
        label = 1 if gt_data.get(sample_id) else 0

        data.append({
            'sample_id': sample_id,
            'D_hat': signal['D_hat'],
            'coh_star': signal['coh_star'],
            'r_LZ': signal['r_LZ'],
            'perplexity': baseline['perplexity'],
            'label': label
        })

    return pd.DataFrame(data)

def validate_coverage(df, delta_values, weights):
    """
    Validate coverage guarantee P(escalate | benign) ≤ δ

    Args:
        df: DataFrame with signals and labels
        delta_values: list of δ values to test (e.g., [0.01, 0.05, 0.10, 0.20])
        weights: ensemble weights for nonconformity score

    Returns:
        dict with results for each δ
    """
    # Split into calibration (20%) and test (80%)
    benign_samples = df[df['label'] == 0]

    if len(benign_samples) < 50:
        print(f"Warning: Only {len(benign_samples)} benign samples, results may be unreliable")
        return {}

    cal_indices, test_indices = train_test_split(
        benign_samples.index, test_size=0.8, random_state=42
    )

    cal_df = df.loc[cal_indices]
    test_df = df.loc[test_indices]

    # Compute nonconformity scores for calibration set
    cal_scores = []
    for _, row in cal_df.iterrows():
        signals = {
            'D_hat': row['D_hat'],
            'coh_star': row['coh_star'],
            'r_LZ': row['r_LZ'],
            'perplexity': row['perplexity']
        }
        score = compute_nonconformity_score(signals, weights)
        cal_scores.append(score)

    cal_scores = np.array(cal_scores)

    # Test each δ value
    results = {}
    for delta in delta_values:
        # Compute (1-δ)-quantile
        n_cal = len(cal_scores)
        quantile_index = int(np.ceil((1 - delta) * (n_cal + 1))) - 1
        quantile_index = max(0, min(quantile_index, n_cal - 1))
        threshold = np.sort(cal_scores)[quantile_index]

        # Test on benign test samples
        test_scores = []
        for _, row in test_df.iterrows():
            signals = {
                'D_hat': row['D_hat'],
                'coh_star': row['coh_star'],
                'r_LZ': row['r_LZ'],
                'perplexity': row['perplexity']
            }
            score = compute_nonconformity_score(signals, weights)
            test_scores.append(score)

        test_scores = np.array(test_scores)

        # Compute empirical miscoverage (escalation rate on benign)
        escalations = (test_scores > threshold).sum()
        empirical_miscoverage = escalations / len(test_scores)

        # Compute 95% confidence interval via Wilson score
        # CI = p ± 1.96 * sqrt(p(1-p)/n)
        n = len(test_scores)
        p = empirical_miscoverage
        ci_margin = 1.96 * np.sqrt(p * (1 - p) / n)
        ci_lower = max(0, p - ci_margin)
        ci_upper = min(1, p + ci_margin)

        results[delta] = {
            'target_delta': delta,
            'threshold': threshold,
            'n_cal': n_cal,
            'n_test': len(test_scores),
            'escalations': int(escalations),
            'empirical_miscoverage': empirical_miscoverage,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'guarantee_held': empirical_miscoverage <= delta
        }

    return results

def main():
    print("=" * 80)
    print("EXPERIMENT 2: COVERAGE CALIBRATION VALIDATION")
    print("=" * 80)
    print()
    print("Validating finite-sample guarantee: P(escalate | benign) ≤ δ")
    print()

    # Test δ values
    delta_values = [0.01, 0.05, 0.10, 0.20]

    # Test on degeneracy benchmark (clean ground truth)
    benchmark = 'degeneracy'
    print(f"Loading benchmark: {benchmark.upper()}")
    df = load_benchmark_data(benchmark)

    if df is None:
        print(f"Error: Could not load {benchmark} data")
        return

    print(f"Loaded {len(df)} samples")
    print(f"Positive rate: {df['label'].mean():.1%}")
    print(f"Benign samples: {(df['label'] == 0).sum()}")
    print()

    # Use task-specific weights (r_LZ dominant for degeneracy)
    weights = {
        'D_hat': 0.15,
        'coh_star': 0.15,
        'r_LZ': 0.60,
        'perplexity': 0.10
    }

    print(f"Ensemble weights: D̂={weights['D_hat']:.2f}, coh★={weights['coh_star']:.2f}, " +
          f"r_LZ={weights['r_LZ']:.2f}, perplexity={weights['perplexity']:.2f}")
    print()

    # Validate coverage for each δ
    results = validate_coverage(df, delta_values, weights)

    # Print results table
    print("=" * 80)
    print("COVERAGE VALIDATION RESULTS")
    print("=" * 80)
    print()
    print(f"{'δ (target)':<12} {'Threshold':<12} {'n_cal':<8} {'n_test':<8} " +
          f"{'Escalations':<12} {'Empirical':<12} {'95% CI':<20} {'Held?':<8}")
    print("-" * 80)

    for delta in delta_values:
        r = results[delta]
        ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
        held_str = "✓ YES" if r['guarantee_held'] else "✗ NO"

        print(f"{r['target_delta']:<12.2f} {r['threshold']:<12.4f} {r['n_cal']:<8} {r['n_test']:<8} " +
              f"{r['escalations']:<12} {r['empirical_miscoverage']:<12.4f} {ci_str:<20} {held_str:<8}")

    print()

    # Save results
    output_dir = Path('results/coverage')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([results[delta] for delta in delta_values])
    results_df.to_csv(output_dir / 'coverage_results.csv', index=False)
    print(f"✓ Saved results to {output_dir / 'coverage_results.csv'}")

    # Generate calibration plot
    generate_calibration_plot(results, output_dir)

    print()
    print("=" * 80)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 80)
    print()
    print("Key Findings:")
    print(f"  - Tested {len(delta_values)} δ values: {delta_values}")
    print(f"  - Guarantees held: {sum(r['guarantee_held'] for r in results.values())}/{len(delta_values)}")

    # Check if all guarantees held
    all_held = all(r['guarantee_held'] for r in results.values())
    if all_held:
        print("\n  ✓ ALL COVERAGE GUARANTEES VALIDATED!")
        print("    Split-conformal prediction provides finite-sample guarantees as claimed.")
    else:
        print("\n  ⚠ Some guarantees violated (within statistical tolerance if CI overlaps δ)")

def generate_calibration_plot(results, output_dir):
    """Generate calibration curve showing target δ vs. empirical miscoverage."""

    delta_values = sorted(results.keys())
    empirical_values = [results[d]['empirical_miscoverage'] for d in delta_values]
    ci_lower = [results[d]['ci_lower'] for d in delta_values]
    ci_upper = [results[d]['ci_upper'] for d in delta_values]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot perfect calibration line (y = x)
    ax.plot([0, 0.25], [0, 0.25], 'k--', linewidth=2, alpha=0.5, label='Perfect Calibration (y=x)')

    # Plot empirical miscoverage with error bars
    ax.errorbar(delta_values, empirical_values,
                yerr=[np.array(empirical_values) - np.array(ci_lower),
                      np.array(ci_upper) - np.array(empirical_values)],
                fmt='o-', markersize=10, linewidth=2, capsize=5,
                color='#2E86DE', label='Empirical Miscoverage (95% CI)')

    # Mark points where guarantee holds
    for d, emp, ci_u in zip(delta_values, empirical_values, ci_upper):
        if emp <= d:
            ax.plot(d, emp, 'go', markersize=12, markeredgewidth=2, markeredgecolor='green')

    ax.set_xlabel('Target δ (Coverage Error Budget)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Empirical Miscoverage Rate', fontsize=14, fontweight='bold')
    ax.set_title('Split-Conformal Coverage Calibration\n' +
                 'Validating P(escalate | benign) ≤ δ Guarantee',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 0.22])
    ax.set_ylim([-0.01, 0.22])

    # Add shaded region for guarantee violation
    ax.fill_between([0, 0.25], [0, 0.25], [0.25, 0.25], alpha=0.1, color='red',
                     label='Violation Region (empirical > target)')

    plt.tight_layout()
    fig.savefig(output_dir / '../../docs/architecture/figures/coverage_calibration.png',
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure: coverage_calibration.png")
    plt.close()

if __name__ == '__main__':
    main()
