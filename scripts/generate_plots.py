#!/usr/bin/env python3
"""
Generate Plots and Tables from Evaluation Results

Creates publication-quality visualizations:
- ROC curves for each benchmark
- Precision-Recall curves for each benchmark
- Comparison bar charts (AUROC, AUPRC, F1)
- Performance heatmap across benchmarks
- Summary tables

Usage:
    python generate_plots.py
    python generate_plots.py --output-dir figures/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


# Add agent src to path for loading data
sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "src"))


def load_evaluation_data(benchmark: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load ground truth labels and method scores for a benchmark.

    Returns:
        y_true: Ground truth labels (0/1)
        method_scores: Dict mapping method name to scores
    """
    # Load evaluation samples
    from pathlib import Path
    import json

    signals_dir = Path(f"data/signals/{benchmark}")
    baselines_dir = Path(f"data/baselines/{benchmark}")
    llm_outputs_file = Path(f"data/llm_outputs/{benchmark}_outputs.jsonl")

    # Load ground truth labels (same logic as evaluate_methods.py)
    labels = {}
    samples = {}

    with open(llm_outputs_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            sample_id = data['id']
            samples[sample_id] = data

            if benchmark == 'truthfulqa':
                llm_response = data['llm_response'].lower()
                incorrect_answers = data['metadata']['incorrect_answers'].lower().split(';')
                is_hallucination = any(
                    len(incorrect) > 10 and incorrect in llm_response
                    for incorrect in incorrect_answers
                )
                labels[sample_id] = is_hallucination

            elif benchmark == 'fever':
                if 'ground_truth' in data:
                    labels[sample_id] = data['ground_truth']
                else:
                    labels[sample_id] = (data['metadata']['label'] == 'REFUTES')

            elif benchmark == 'halueval':
                labels[sample_id] = data['metadata'].get('hallucination', False)

    # Load signals and baselines
    signal_scores = {'D_hat': {}, 'coh_star': {}, 'r_LZ': {}}
    baseline_scores = {
        'perplexity': {},
        'log_perplexity': {},
        'mean_token_prob': {},
        'min_token_prob': {},
        'entropy': {}
    }

    # Load signals
    for signal_file in signals_dir.glob("*.json"):
        with open(signal_file) as f:
            signal_data = json.load(f)
            sample_id = signal_data['sample_id']
            if sample_id in labels:
                signal_scores['D_hat'][sample_id] = signal_data['D_hat']
                signal_scores['coh_star'][sample_id] = signal_data['coh_star']
                signal_scores['r_LZ'][sample_id] = signal_data['r_LZ']

    # Load baselines
    for baseline_file in baselines_dir.glob("*.json"):
        with open(baseline_file) as f:
            baseline_data = json.load(f)
            sample_id = baseline_data['sample_id']
            if sample_id in labels:
                baseline_scores['perplexity'][sample_id] = baseline_data['perplexity']
                baseline_scores['log_perplexity'][sample_id] = baseline_data['log_perplexity']
                baseline_scores['mean_token_prob'][sample_id] = baseline_data['mean_token_prob']
                baseline_scores['min_token_prob'][sample_id] = baseline_data['min_token_prob']
                baseline_scores['entropy'][sample_id] = baseline_data['entropy']

    # Align samples
    common_ids = set(labels.keys()) & set(signal_scores['D_hat'].keys()) & set(baseline_scores['perplexity'].keys())
    common_ids = sorted(common_ids)

    y_true = np.array([labels[sid] for sid in common_ids])

    # Compute method scores
    method_scores = {}

    # ASV signals (normalize and invert for hallucination detection)
    D_values = np.array([signal_scores['D_hat'][sid] for sid in common_ids])
    D_norm = np.clip(D_values / 3.0, 0, 1)
    method_scores['ASV D̂'] = 1 - D_norm

    coh_values = np.array([signal_scores['coh_star'][sid] for sid in common_ids])
    method_scores['ASV coh★'] = 1 - coh_values

    r_values = np.array([signal_scores['r_LZ'][sid] for sid in common_ids])
    method_scores['ASV r_LZ'] = 1 - r_values

    # ASV combined
    method_scores['ASV Combined'] = (
        0.5 * method_scores['ASV D̂'] +
        0.3 * method_scores['ASV coh★'] +
        0.2 * method_scores['ASV r_LZ']
    )

    # Baseline methods (normalize)
    ppl_values = np.array([baseline_scores['perplexity'][sid] for sid in common_ids])
    ppl_range = ppl_values.max() - ppl_values.min()
    if ppl_range > 1e-6:
        method_scores['Baseline Perplexity'] = (ppl_values - ppl_values.min()) / ppl_range
    else:
        method_scores['Baseline Perplexity'] = ppl_values / (ppl_values.mean() + 1e-10)
    method_scores['Baseline Perplexity'] = np.nan_to_num(method_scores['Baseline Perplexity'], nan=0.0)

    # Token probabilities (lower prob = higher hallucination risk)
    mean_prob = np.array([baseline_scores['mean_token_prob'][sid] for sid in common_ids])
    method_scores['Baseline Mean Token Prob'] = 1 - mean_prob

    min_prob = np.array([baseline_scores['min_token_prob'][sid] for sid in common_ids])
    method_scores['Baseline Min Token Prob'] = 1 - min_prob

    # Entropy (higher entropy = more uncertainty)
    entropy_values = np.array([baseline_scores['entropy'][sid] for sid in common_ids])
    entropy_range = entropy_values.max() - entropy_values.min()
    if entropy_range > 1e-6:
        method_scores['Baseline Entropy'] = (entropy_values - entropy_values.min()) / entropy_range
    else:
        method_scores['Baseline Entropy'] = entropy_values / (entropy_values.mean() + 1e-10)

    return y_true, method_scores


def plot_roc_curves(benchmark: str, y_true: np.ndarray, method_scores: Dict[str, np.ndarray],
                   output_dir: Path):
    """Plot ROC curves for all methods on a benchmark."""

    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors for different method types
    asv_color = sns.color_palette("Set1")[0]  # Red
    baseline_color = sns.color_palette("Set1")[1]  # Blue

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC=0.50)')

    # Plot each method
    for method_name, scores in method_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)

        # Compute AUC (manually to match sklearn)
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)

        # Choose color based on method type
        if method_name.startswith('ASV'):
            color = asv_color
            linestyle = '-'
        else:
            color = baseline_color
            linestyle = '--'

        ax.plot(fpr, tpr, linestyle=linestyle, lw=2, alpha=0.7,
                label=f'{method_name} (AUC={roc_auc:.3f})')

    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title(f'ROC Curves - {benchmark.upper()}', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f'{benchmark}_roc_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved ROC curves: {output_file}")


def plot_pr_curves(benchmark: str, y_true: np.ndarray, method_scores: Dict[str, np.ndarray],
                  output_dir: Path):
    """Plot Precision-Recall curves for all methods on a benchmark."""

    fig, ax = plt.subplots(figsize=(10, 8))

    # Baseline (random) = positive rate
    baseline_precision = y_true.mean()
    ax.axhline(y=baseline_precision, color='k', linestyle='--', lw=1, alpha=0.5,
               label=f'Random (AP={baseline_precision:.3f})')

    # Colors
    asv_color = sns.color_palette("Set1")[0]
    baseline_color = sns.color_palette("Set1")[1]

    # Plot each method
    for method_name, scores in method_scores.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)

        # Compute average precision
        from sklearn.metrics import average_precision_score
        avg_precision = average_precision_score(y_true, scores)

        if method_name.startswith('ASV'):
            color = asv_color
            linestyle = '-'
        else:
            color = baseline_color
            linestyle = '--'

        ax.plot(recall, precision, linestyle=linestyle, lw=2, alpha=0.7,
                label=f'{method_name} (AP={avg_precision:.3f})')

    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title(f'Precision-Recall Curves - {benchmark.upper()}', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f'{benchmark}_pr_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved PR curves: {output_file}")


def plot_comparison_bars(results_data: Dict[str, Dict], output_dir: Path):
    """Plot bar charts comparing methods across benchmarks."""

    benchmarks = ['truthfulqa', 'fever', 'halueval']

    # Extract data
    methods = []
    for benchmark in benchmarks:
        if benchmark in results_data and results_data[benchmark]:
            methods = list(results_data[benchmark]['methods'].keys())
            break

    # Create DataFrame
    rows = []
    for method_key in methods:
        row = {'Method': results_data[benchmarks[0]]['methods'][method_key]['method_name']}
        for benchmark in benchmarks:
            if benchmark in results_data and results_data[benchmark]:
                metrics = results_data[benchmark]['methods'].get(method_key, {})
                row[f'{benchmark}_auroc'] = metrics.get('auroc', 0)
                row[f'{benchmark}_auprc'] = metrics.get('auprc', 0)
                row[f'{benchmark}_f1'] = metrics.get('f1_optimal', 0)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Plot AUROC comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_to_plot = [
        ('auroc', 'AUROC', 0),
        ('auprc', 'AUPRC', 1),
        ('f1', 'F1 Score', 2)
    ]

    for metric_suffix, metric_label, ax_idx in metrics_to_plot:
        ax = axes[ax_idx]

        # Prepare data for grouped bar chart
        metric_cols = [f'{b}_{metric_suffix}' for b in benchmarks]
        plot_df = df[['Method'] + metric_cols].copy()

        # Shorten method names for display
        plot_df['Method'] = plot_df['Method'].str.replace('Baseline ', 'B_')
        plot_df['Method'] = plot_df['Method'].str.replace('ASV ', '')

        plot_df_melted = plot_df.melt(id_vars=['Method'], var_name='Benchmark', value_name=metric_label)
        plot_df_melted['Benchmark'] = plot_df_melted['Benchmark'].str.replace(f'_{metric_suffix}', '').str.upper()

        # Plot grouped bars
        x = np.arange(len(plot_df))
        width = 0.25

        for i, benchmark in enumerate(['TRUTHFULQA', 'FEVER', 'HALUEVAL']):
            values = plot_df_melted[plot_df_melted['Benchmark'] == benchmark][metric_label].values
            offset = width * (i - 1)
            ax.bar(x + offset, values, width, label=benchmark, alpha=0.8)

        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_label} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['Method'], rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])

    plt.tight_layout()
    output_file = output_dir / 'comparison_bars.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved comparison bars: {output_file}")


def plot_heatmap(results_data: Dict[str, Dict], output_dir: Path):
    """Plot heatmap of method performance across benchmarks."""

    benchmarks = ['truthfulqa', 'fever', 'halueval']

    # Extract data
    methods = []
    for benchmark in benchmarks:
        if benchmark in results_data and results_data[benchmark]:
            methods = list(results_data[benchmark]['methods'].keys())
            break

    # Create matrix (methods x benchmarks)
    method_names = []
    auroc_matrix = []

    for method_key in methods:
        method_name = None
        auroc_row = []

        for benchmark in benchmarks:
            if benchmark in results_data and results_data[benchmark]:
                metrics = results_data[benchmark]['methods'].get(method_key, {})
                if method_name is None:
                    method_name = metrics.get('method_name', method_key)
                auroc_row.append(metrics.get('auroc', 0))
            else:
                auroc_row.append(0)

        # Shorten method names
        method_name = method_name.replace('Baseline ', 'B_').replace('ASV ', '')
        method_names.append(method_name)
        auroc_matrix.append(auroc_row)

    auroc_matrix = np.array(auroc_matrix)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        auroc_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.4,
        vmax=0.7,
        xticklabels=[b.upper() for b in benchmarks],
        yticklabels=method_names,
        cbar_kws={'label': 'AUROC'},
        ax=ax
    )

    ax.set_title('Method Performance Heatmap (AUROC)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Benchmark', fontsize=14, fontweight='bold')
    ax.set_ylabel('Method', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_file = output_dir / 'performance_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved performance heatmap: {output_file}")


def generate_summary_table(results_data: Dict[str, Dict], output_dir: Path):
    """Generate LaTeX and CSV tables summarizing results."""

    benchmarks = ['truthfulqa', 'fever', 'halueval']

    # Extract data
    rows = []
    for benchmark in benchmarks:
        if benchmark not in results_data or not results_data[benchmark]:
            continue

        for method_key, metrics in results_data[benchmark]['methods'].items():
            rows.append({
                'Benchmark': benchmark.upper(),
                'Method': metrics['method_name'],
                'AUROC': metrics['auroc'],
                'AUPRC': metrics['auprc'],
                'F1': metrics['f1_optimal'],
                'Accuracy': metrics['accuracy_optimal'],
                'Precision': metrics['precision_optimal'],
                'Recall': metrics['recall_optimal'],
                'N': metrics['n_samples']
            })

    df = pd.DataFrame(rows)

    # Save CSV
    csv_file = output_dir / 'summary_table.csv'
    df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"✓ Saved summary table (CSV): {csv_file}")

    # Generate LaTeX table
    latex_file = output_dir / 'summary_table.tex'
    with open(latex_file, 'w') as f:
        f.write("% Summary of evaluation results\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Hallucination Detection Performance Across Benchmarks}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\begin{tabular}{llrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Benchmark & Method & AUROC & AUPRC & F1 & Acc & Prec & Recall \\\\\n")
        f.write("\\midrule\n")

        for benchmark in benchmarks:
            benchmark_df = df[df['Benchmark'] == benchmark.upper()]
            for _, row in benchmark_df.iterrows():
                method_short = row['Method'].replace('Baseline ', 'B-').replace('ASV ', '')
                f.write(f"{row['Benchmark']} & {method_short} & "
                       f"{row['AUROC']:.3f} & {row['AUPRC']:.3f} & {row['F1']:.3f} & "
                       f"{row['Accuracy']:.3f} & {row['Precision']:.3f} & {row['Recall']:.3f} \\\\\n")
            f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"✓ Saved summary table (LaTeX): {latex_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate evaluation plots and tables')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures',
        help='Output directory for figures'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Generating Evaluation Plots and Tables")
    print("=" * 70)

    # Load results
    benchmarks = ['truthfulqa', 'fever', 'halueval']
    results_data = {}

    for benchmark in benchmarks:
        results_file = Path(f"results/{benchmark}_results.json")
        if results_file.exists():
            with open(results_file) as f:
                results_data[benchmark] = json.load(f)
        else:
            print(f"⚠️  No results found for {benchmark}")
            results_data[benchmark] = None

    # Generate plots for each benchmark
    for benchmark in benchmarks:
        if results_data[benchmark] is None:
            continue

        print(f"\n📊 Generating plots for {benchmark.upper()}...")

        # Load evaluation data
        y_true, method_scores = load_evaluation_data(benchmark)

        # Generate ROC curves
        plot_roc_curves(benchmark, y_true, method_scores, output_dir)

        # Generate PR curves
        plot_pr_curves(benchmark, y_true, method_scores, output_dir)

    # Generate comparison visualizations
    print(f"\n📊 Generating cross-benchmark comparisons...")
    plot_comparison_bars(results_data, output_dir)
    plot_heatmap(results_data, output_dir)

    # Generate summary tables
    print(f"\n📊 Generating summary tables...")
    generate_summary_table(results_data, output_dir)

    print("\n" + "=" * 70)
    print(f"✅ All visualizations generated in: {output_dir.absolute()}")
    print("=" * 70)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
