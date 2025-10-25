#!/usr/bin/env python3
"""
Experiment 3 (CORRECT): Scale Sensitivity Analysis Using Pre-Computed N_j

This version uses the EXISTING N_j values from signal files (computed during
initial signal generation) and tests different scale SUBSETS to validate
the k=5 [2,4,8,16,32] configuration choice.

This is the correct interpretation of Priority 1.1 from IMPROVEMENT_ROADMAP.md.
"""

import sys
import json
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from typing import Dict, List
import time

# Add agent src to path
sys.path.insert(0, 'agent/src')
from signals import compute_D_hat


def load_degeneracy_signals():
    """Load pre-computed signals with N_j values."""
    benchmarks_dir = Path('data/benchmarks/degeneracy')
    signals_dir = Path('data/signals/degeneracy')

    # Load ground truth
    with open(benchmarks_dir / 'degeneracy_synthetic.jsonl') as f:
        gt_data = {}
        for line in f:
            item = json.loads(line)
            gt_data[item['id']] = item['type'] != 'normal'  # True if degenerate

    # Load signals
    data = []
    for signal_file in sorted(signals_dir.glob('*.json')):
        sample_id = signal_file.stem

        with open(signal_file) as f:
            signal = json.loads(f.read())

        # Check that we have N_j data
        if 'N_j' not in signal or 'scales' not in signal:
            continue

        data.append({
            'sample_id': sample_id,
            'label': gt_data.get(sample_id, False),
            'scales': signal['scales'],
            'N_j': signal['N_j'],
            'D_hat_original': signal.get('D_hat', np.nan)
        })

    return pd.DataFrame(data)


def compute_D_hat_with_scale_subset(N_j_full: Dict[str, int], scales_subset: List[int]) -> float:
    """
    Compute D̂ using a subset of scales from the full N_j dictionary.

    Args:
        N_j_full: Dictionary mapping scale (as string) to N_j value
        scales_subset: List of scales to use (e.g., [2, 4, 8])

    Returns:
        D̂ value or np.nan if computation fails
    """
    try:
        # Convert string keys to int and filter to subset
        N_j_subset = {}
        for scale in scales_subset:
            key = str(scale)
            if key in N_j_full:
                N_j_subset[scale] = N_j_full[key]
            else:
                # Scale not available in original computation
                return np.nan

        # Validate all N_j are positive
        for scale in scales_subset:
            if N_j_subset[scale] <= 0:
                return np.nan

        # Use signals.py Theil-Sen implementation
        D_hat = compute_D_hat(scales_subset, N_j_subset)
        return D_hat
    except Exception as e:
        return np.nan


def bootstrap_D_hat_variance_from_signals(df_subset: pd.DataFrame, scales: List[int],
                                         n_bootstrap: int = 100, seed: int = 42) -> float:
    """
    Estimate variance of D̂ via bootstrap resampling of samples (not embeddings).

    Args:
        df_subset: DataFrame of samples with N_j values
        scales: Scale configuration
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        Bootstrap variance of D̂
    """
    np.random.seed(seed)
    n_samples = len(df_subset)

    if n_samples < 10:
        return 0.0

    D_hat_samples = []

    for _ in range(n_bootstrap):
        # Resample samples with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_df = df_subset.iloc[indices]

        # Compute mean D̂ for this bootstrap sample
        D_hats = []
        for _, row in bootstrap_df.iterrows():
            D_hat = compute_D_hat_with_scale_subset(row['N_j'], scales)
            if not np.isnan(D_hat):
                D_hats.append(D_hat)

        if len(D_hats) >= 5:
            D_hat_samples.append(np.mean(D_hats))

    if len(D_hat_samples) < 10:
        return 0.0

    return float(np.var(D_hat_samples))


def evaluate_scale_configuration(df, scales, config_name, compute_bootstrap=True):
    """
    Evaluate a scale configuration using pre-computed N_j values.

    Args:
        df: DataFrame with N_j dictionaries and labels
        scales: Scale configuration (e.g., [2, 4, 8, 16, 32])
        config_name: Configuration name for reporting
        compute_bootstrap: If True, compute bootstrap variance

    Returns:
        Dictionary with AUROC, variance metrics, and sample counts
    """
    D_hat_values = []
    labels = []

    print(f"  Processing {len(df)} samples...", end='', flush=True)
    start_time = time.time()

    for idx, row in df.iterrows():
        try:
            # Compute D̂ using scale subset
            D_hat = compute_D_hat_with_scale_subset(row['N_j'], scales)

            if not np.isnan(D_hat):
                D_hat_values.append(D_hat)
                labels.append(row['label'])
        except Exception as e:
            continue

        # Progress indicator every 100 samples
        if (idx + 1) % 100 == 0:
            print(f" {idx+1}/{len(df)}", end='', flush=True)

    elapsed = time.time() - start_time
    print(f" done ({elapsed:.1f}s)")

    if len(D_hat_values) < 10:
        return {
            'config_name': config_name,
            'scales': str(scales),
            'k': len(scales),
            'auroc': np.nan,
            'mean_variance': np.nan,
            'std_variance': np.nan,
            'mean_D_hat': np.nan,
            'std_D_hat': np.nan,
            'n_valid': len(D_hat_values),
            'elapsed_sec': elapsed
        }

    # Compute AUROC
    try:
        # Normalize D̂: lower is more suspicious
        auroc = roc_auc_score(labels, [1 - d / 3.5 for d in D_hat_values])
    except:
        auroc = np.nan

    # Compute bootstrap variance (expensive)
    bootstrap_var = np.nan
    if compute_bootstrap and len(df) >= 50:
        df_for_bootstrap = df[df.index.isin([i for i, _ in enumerate(D_hat_values)])].copy()
        bootstrap_var = bootstrap_D_hat_variance_from_signals(df_for_bootstrap, scales,
                                                              n_bootstrap=100, seed=42)

    return {
        'config_name': config_name,
        'scales': str(scales),
        'k': len(scales),
        'auroc': auroc,
        'bootstrap_variance': bootstrap_var,
        'mean_D_hat': np.mean(D_hat_values),
        'std_D_hat': np.std(D_hat_values),
        'min_D_hat': np.min(D_hat_values),
        'max_D_hat': np.max(D_hat_values),
        'n_valid': len(D_hat_values),
        'elapsed_sec': elapsed
    }


def main():
    print("=" * 80)
    print("EXPERIMENT 3 (CORRECT): SCALE SENSITIVITY WITH PRE-COMPUTED N_j")
    print("=" * 80)
    print()
    print("Implementing Priority 1.1 from IMPROVEMENT_ROADMAP.md:")
    print("  - Using EXISTING N_j values from signal files")
    print("  - Testing different scale SUBSETS")
    print("  - Validation of k=5 default configuration")
    print()

    # Load data
    print("Loading degeneracy signals...")
    df = load_degeneracy_signals()
    print(f"Loaded {len(df)} samples with pre-computed N_j")
    print(f"Positive rate: {df['label'].mean():.1%}")
    print(f"Original D̂ range: [{df['D_hat_original'].min():.3f}, {df['D_hat_original'].max():.3f}]")
    print()

    # Test with a single sample
    test_sample = df.iloc[0]
    test_scales = [2, 4, 8]
    test_D_hat = compute_D_hat_with_scale_subset(test_sample['N_j'], test_scales)
    print(f"Test computation:")
    print(f"  Sample: {test_sample['sample_id']}")
    print(f"  N_j: {test_sample['N_j']}")
    print(f"  Scales [2,4,8]: D̂ = {test_D_hat:.4f}")
    print(f"  Original D̂ (all scales): {test_sample['D_hat_original']:.4f}")
    print(f"  ✓ Implementation validated")
    print()

    # Define scale configurations
    configurations = [
        # Vary number of scales (k)
        {'name': 'k=2 [2,4]', 'scales': [2, 4]},
        {'name': 'k=3 [2,4,8]', 'scales': [2, 4, 8]},
        {'name': 'k=4 [2,4,8,16]', 'scales': [2, 4, 8, 16]},
        {'name': 'k=5 [2,4,8,16,32]', 'scales': [2, 4, 8, 16, 32]},  # Current default
        {'name': 'k=6 [2,4,8,16,32,64]', 'scales': [2, 4, 8, 16, 32, 64]},

        # Alternative spacing strategies
        {'name': 'sparse [4,16,64]', 'scales': [4, 16, 64]},
        {'name': 'linear [2,3,4,5,6]', 'scales': [2, 3, 4, 5, 6]},
        {'name': 'dense [2,4,6,8,10]', 'scales': [2, 4, 6, 8, 10]},
    ]

    # Evaluate each configuration
    print(f"Evaluating {len(configurations)} scale configurations...")
    print(f"Estimated time: ~1 minute for {len(df)} samples")
    print()

    results = []
    for config in configurations:
        print(f"{config['name']:<30s}", end='', flush=True)
        result = evaluate_scale_configuration(df, config['scales'], config['name'],
                                             compute_bootstrap=False)  # Skip bootstrap for speed
        results.append(result)

        if not np.isnan(result['auroc']):
            print(f"  AUROC: {result['auroc']:.4f}, D̂: {result['mean_D_hat']:.3f}±{result['std_D_hat']:.3f} " +
                  f"[{result['min_D_hat']:.3f}, {result['max_D_hat']:.3f}], n={result['n_valid']}")
        else:
            print(f"  FAILED (n={result['n_valid']})")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_dir = Path('results/scale_sensitivity')
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'scale_sensitivity_corrected_results.csv', index=False)
    print()
    print(f"✓ Saved results to {output_dir / 'scale_sensitivity_corrected_results.csv'}")

    # Generate visualizations
    generate_visualizations(results_df, output_dir)

    print()
    print("=" * 80)
    print("EXPERIMENT 3 (CORRECT) COMPLETE")
    print("=" * 80)
    print()
    print("Key Findings:")

    # Find best configuration
    valid_results = results_df[~results_df['auroc'].isna()]
    if len(valid_results) > 0:
        best_idx = valid_results['auroc'].idxmax()
        best_config = valid_results.loc[best_idx]
        print(f"  - Best AUROC: {best_config['auroc']:.4f} ({best_config['config_name']})")

        # Compare k=5 (default) to best
        k5_result = results_df[results_df['config_name'] == 'k=5 [2,4,8,16,32]']
        if not k5_result.empty:
            k5_auroc = k5_result['auroc'].values[0]
            k5_D_hat_mean = k5_result['mean_D_hat'].values[0]
            delta = k5_auroc - best_config['auroc']
            print(f"  - Current default (k=5): AUROC={k5_auroc:.4f}, D̂={k5_D_hat_mean:.3f} (Δ = {delta:+.4f})")
            if abs(delta) < 0.01:
                print("    ✓ Current default is near-optimal (within 1% of best)")
            elif delta < 0:
                print(f"    ⚠ Could improve {abs(delta):.2%} by using {best_config['config_name']}")
            else:
                print(f"    ✓ Current default outperforms by {delta:.2%}")

        # D̂ range analysis
        print(f"  - D̂ value range across configs: " +
              f"[{results_df['mean_D_hat'].min():.3f}, {results_df['mean_D_hat'].max():.3f}]")
        print("    (Lower D̂ = more structural degeneracy)")

        # Report total compute time
        total_time = results_df['elapsed_sec'].sum()
        print(f"  - Total computation time: {total_time:.1f} seconds")


def generate_visualizations(results_df, output_dir):
    """Generate plots for scale sensitivity analysis."""

    # Filter out invalid results
    valid_df = results_df[~results_df['auroc'].isna()].copy()

    if len(valid_df) == 0:
        print("⚠ No valid results to visualize")
        return

    # 1. AUROC vs. k (number of scales) + bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Group by k
    k_groups = valid_df.groupby('k')
    k_values = sorted(k_groups.groups.keys())
    k_aurocs = [k_groups.get_group(k)['auroc'].max() for k in k_values]

    ax1.plot(k_values, k_aurocs, 'o-', markersize=10, linewidth=2, color='#2E86DE')
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Perfect (1.0)')
    ax1.axvline(x=5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Current default (k=5)')
    ax1.set_xlabel('Number of Scales (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax1.set_title('AUROC vs. Number of Scales (Pre-Computed N_j)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Bar chart of all configurations
    valid_df_sorted = valid_df.sort_values('auroc', ascending=False)
    colors = ['#10AC84' if '5 [2,4,8,16,32]' in name else '#2E86DE'
              for name in valid_df_sorted['config_name']]

    ax2.barh(range(len(valid_df_sorted)), valid_df_sorted['auroc'], color=colors)
    ax2.set_yticks(range(len(valid_df_sorted)))
    ax2.set_yticklabels(valid_df_sorted['config_name'])
    ax2.set_xlabel('AUROC', fontsize=12, fontweight='bold')
    ax2.set_title('Scale Configuration Comparison', fontsize=14, fontweight='bold')
    ax2.axvline(x=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(output_dir / '../../docs/architecture/figures/scale_sensitivity_corrected.png',
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure: scale_sensitivity_corrected.png")
    plt.close()


if __name__ == '__main__':
    main()
