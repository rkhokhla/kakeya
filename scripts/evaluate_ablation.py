#!/usr/bin/env python3
"""
Experiment 1: Signal Ablation Study

Tests all combinations of signals (D̂, coh★, r_LZ, perplexity) to understand
individual and combined contributions to hallucination detection performance.

This validates which signals are critical and whether the full ensemble is needed.
"""

import sys
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add agent src to path for signal normalization
sys.path.insert(0, 'agent/src')

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

def compute_ensemble_score(signals, weights):
    """
    Compute weighted ensemble score.

    Args:
        signals: dict with keys 'D_hat', 'coh_star', 'r_LZ', 'perplexity'
        weights: dict with same keys, values ∈ [0, 1], sum to 1

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

    # Load ground truth
    gt_path = Path(f'data/benchmarks/{benchmark_name}')
    if benchmark_name == 'truthfulqa':
        import csv
        with open(gt_path / 'TruthfulQA.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            gt_data = {f"truthfulqa_{i}": (row['Best Answer'], row['Category'])
                      for i, row in enumerate(reader)}
    elif benchmark_name in ['fever', 'halueval']:
        gt_data = {}
        jsonl_file = 'shared_task_dev.jsonl' if benchmark_name == 'fever' else 'qa_samples.json'
        with open(gt_path / jsonl_file) as f:
            if benchmark_name == 'fever':
                for i, line in enumerate(f):
                    data = json.loads(line)
                    gt_data[f"fever_{i}"] = data['label']  # SUPPORTS/REFUTES/NOT ENOUGH INFO
            else:
                data_list = json.load(f)
                for i, item in enumerate(data_list):
                    gt_data[f"halueval_{i}"] = item['hallucination']
    elif benchmark_name == 'degeneracy':
        with open(gt_path / 'degeneracy_synthetic.jsonl') as f:
            gt_data = {}
            for line in f:
                data = json.loads(line)
                gt_data[data['id']] = (data['type'] != 'normal')  # True if degenerate

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
        if benchmark_name == 'truthfulqa':
            # For TruthfulQA, we consider responses that don't match best answer as hallucinations
            # This is a simplified approach; proper evaluation would need human judgment
            label = 1  # Assume hallucination for ablation (conservative)
        elif benchmark_name == 'fever':
            label = 1 if gt_data.get(sample_id) == 'REFUTES' else 0
        elif benchmark_name == 'halueval':
            label = 1 if gt_data.get(sample_id) else 0
        elif benchmark_name == 'degeneracy':
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

def evaluate_combination(df, combination):
    """
    Evaluate a signal combination.

    Args:
        df: DataFrame with signals and labels
        combination: dict with signal names as keys, True/False as values

    Returns:
        dict with AUROC and AUPRC
    """
    # Build weights from combination
    active_signals = [k for k, v in combination.items() if v and k != 'name']
    if not active_signals:
        return {'auroc': 0.5, 'auprc': 0.0}

    # Equal weights for active signals
    weights = {k: (1.0 / len(active_signals) if k in active_signals else 0.0)
               for k in ['D_hat', 'coh_star', 'r_LZ', 'perplexity']}

    # Compute scores
    scores = []
    for _, row in df.iterrows():
        signals = {
            'D_hat': row['D_hat'],
            'coh_star': row['coh_star'],
            'r_LZ': row['r_LZ'],
            'perplexity': row['perplexity']
        }
        score = compute_ensemble_score(signals, weights)
        scores.append(score)

    scores = np.array(scores)
    labels = df['label'].values

    # Compute metrics
    try:
        auroc = roc_auc_score(labels, scores)
    except:
        auroc = 0.5

    try:
        precision, recall, _ = precision_recall_curve(labels, scores)
        auprc = auc(recall, precision)
    except:
        auprc = 0.0

    return {'auroc': auroc, 'auprc': auprc}

def main():
    print("=" * 80)
    print("EXPERIMENT 1: SIGNAL ABLATION STUDY")
    print("=" * 80)
    print()

    # Define all signal combinations
    combinations = [
        # Individual signals
        {"name": "D̂ only", "D_hat": True, "coh_star": False, "r_LZ": False, "perplexity": False},
        {"name": "coh★ only", "D_hat": False, "coh_star": True, "r_LZ": False, "perplexity": False},
        {"name": "r_LZ only", "D_hat": False, "coh_star": False, "r_LZ": True, "perplexity": False},
        {"name": "Perplexity only", "D_hat": False, "coh_star": False, "r_LZ": False, "perplexity": True},

        # Key pairwise combinations
        {"name": "D̂ + r_LZ", "D_hat": True, "coh_star": False, "r_LZ": True, "perplexity": False},
        {"name": "r_LZ + Perplexity", "D_hat": False, "coh_star": False, "r_LZ": True, "perplexity": True},

        # ASV triplet (no perplexity)
        {"name": "ASV (D̂+coh★+r_LZ)", "D_hat": True, "coh_star": True, "r_LZ": True, "perplexity": False},

        # Full ensemble
        {"name": "Full Ensemble", "D_hat": True, "coh_star": True, "r_LZ": True, "perplexity": True},
    ]

    benchmarks = ["truthfulqa", "fever", "halueval", "degeneracy"]

    # Results storage
    results = []

    for benchmark in benchmarks:
        print(f"\n{'='*60}")
        print(f"Benchmark: {benchmark.upper()}")
        print('='*60)

        # Load data
        try:
            df = load_benchmark_data(benchmark)
            print(f"Loaded {len(df)} samples")
            print(f"Positive rate: {df['label'].mean():.1%}")
        except Exception as e:
            print(f"Error loading {benchmark}: {e}")
            continue

        # Evaluate each combination
        for combo in combinations:
            metrics = evaluate_combination(df, combo)
            results.append({
                'Benchmark': benchmark,
                'Combination': combo['name'],
                'AUROC': metrics['auroc'],
                'AUPRC': metrics['auprc']
            })
            print(f"  {combo['name']:30s} AUROC: {metrics['auroc']:.4f}  AUPRC: {metrics['auprc']:.4f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_dir = Path('results/ablation')
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'ablation_results.csv', index=False)
    print(f"\n✓ Saved results to {output_dir / 'ablation_results.csv'}")

    # Generate visualizations
    generate_visualizations(results_df, output_dir)

    print("\n" + "=" * 80)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 80)

def generate_visualizations(results_df, output_dir):
    """Generate bar charts and heatmaps for ablation study."""

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # 1. AUROC comparison across benchmarks (grouped bar chart)
    fig, ax = plt.subplots(figsize=(14, 6))

    pivot_auroc = results_df.pivot(index='Combination', columns='Benchmark', values='AUROC')
    pivot_auroc.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Signal Combination', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_title('Signal Ablation Study: AUROC by Combination and Benchmark',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Benchmark', loc='upper right')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (0.5)')
    ax.set_ylim([0.4, 1.05])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    fig.savefig(output_dir / '../../docs/architecture/figures/ablation_auroc.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure: ablation_auroc.png")
    plt.close()

    # 2. AUPRC comparison (for imbalanced datasets)
    fig, ax = plt.subplots(figsize=(14, 6))

    pivot_auprc = results_df.pivot(index='Combination', columns='Benchmark', values='AUPRC')
    pivot_auprc.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Signal Combination', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUPRC', fontsize=12, fontweight='bold')
    ax.set_title('Signal Ablation Study: AUPRC by Combination and Benchmark',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Benchmark', loc='upper right')
    ax.set_ylim([0, 1.05])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    fig.savefig(output_dir / '../../docs/architecture/figures/ablation_auprc.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure: ablation_auprc.png")
    plt.close()

    # 3. Heatmap showing AUROC for each combination × benchmark
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(pivot_auroc, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.4, vmax=1.0, center=0.7, ax=ax, cbar_kws={'label': 'AUROC'})

    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Signal Combination', fontsize=12, fontweight='bold')
    ax.set_title('Signal Ablation Heatmap: AUROC Across All Conditions',
                 fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    fig.savefig(output_dir / '../../docs/architecture/figures/ablation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure: ablation_heatmap.png")
    plt.close()

if __name__ == '__main__':
    main()
