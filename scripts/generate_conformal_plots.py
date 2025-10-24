#!/usr/bin/env python3
"""
Generate Visualizations for Split-Conformal Verification Results

Creates publication-quality figures:
- ROC/PR curves with conformal thresholds
- Calibration plots (coverage vs delta)
- Ensemble weight comparison charts
- Performance comparison tables
- Benchmark-specific insights

Usage:
    python generate_conformal_plots.py
    python generate_conformal_plots.py --output-dir figures/
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
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


def load_conformal_results(benchmark: str) -> Dict:
    """Load conformal prediction results for a benchmark."""
    results_file = Path(f"results/{benchmark}_conformal_results.json")
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file) as f:
        return json.load(f)


def load_all_results() -> Dict[str, Dict]:
    """Load conformal results for all benchmarks."""
    benchmarks = ['truthfulqa', 'fever', 'halueval', 'degeneracy']
    results = {}
    for benchmark in benchmarks:
        try:
            results[benchmark] = load_conformal_results(benchmark)
        except FileNotFoundError:
            print(f"Warning: Skipping {benchmark} (results not found)")
    return results


def plot_auroc_comparison_bars(output_dir: Path):
    """Generate bar chart comparing AUROC across all methods and benchmarks."""
    all_results = load_all_results()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    benchmark_titles = {
        'truthfulqa': 'TruthfulQA (Factuality)',
        'fever': 'FEVER (Fact Verification)',
        'halueval': 'HaluEval (QA Hallucinations)',
        'degeneracy': 'Degeneracy (Structural)'
    }

    colors = {'conformal': '#2E86DE', 'perplexity': '#10AC84',
              'd_hat': '#EE5A6F', 'r_lz': '#F79F1F'}

    for idx, (benchmark, data) in enumerate(all_results.items()):
        ax = axes[idx]
        methods = data['methods']

        # Collect AUROC values
        method_names = []
        auroc_values = []
        bar_colors = []

        # Add conformal ensemble
        if 'ASV_Conformal_Ensemble' in methods:
            method_names.append('Conformal\nEnsemble')
            auroc_values.append(methods['ASV_Conformal_Ensemble']['auroc'])
            bar_colors.append(colors['conformal'])

        # Add individual ASV signals
        if 'ASV_D_hat' in methods:
            method_names.append('ASV: D̂')
            auroc_values.append(methods['ASV_D_hat']['auroc'])
            bar_colors.append(colors['d_hat'])

        if 'ASV_r_LZ' in methods:
            method_names.append('ASV: r_LZ')
            auroc_values.append(methods['ASV_r_LZ']['auroc'])
            bar_colors.append(colors['r_lz'])

        # Add perplexity baseline
        if 'Baseline_Perplexity' in methods:
            method_names.append('Baseline:\nPerplexity')
            auroc_values.append(methods['Baseline_Perplexity']['auroc'])
            bar_colors.append(colors['perplexity'])

        # Plot bars
        x = np.arange(len(method_names))
        bars = ax.bar(x, auroc_values, color=bar_colors, alpha=0.85, width=0.6)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('AUROC', fontsize=13, fontweight='bold')
        ax.set_title(benchmark_titles[benchmark], fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)

        # Add horizontal line at 0.5 (random performance)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.6)

        # Add text annotation for random line
        ax.text(len(method_names)-0.5, 0.52, 'Random', fontsize=10,
                color='red', alpha=0.8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'auroc_comparison_conformal.png', dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_dir / 'auroc_comparison_conformal.png'}")
    plt.close()


def plot_auprc_comparison_bars(output_dir: Path):
    """Generate bar chart comparing AUPRC across all methods and benchmarks."""
    all_results = load_all_results()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    benchmark_titles = {
        'truthfulqa': 'TruthfulQA (Factuality)',
        'fever': 'FEVER (Fact Verification)',
        'halueval': 'HaluEval (QA Hallucinations)',
        'degeneracy': 'Degeneracy (Structural)'
    }

    colors = {'conformal': '#2E86DE', 'perplexity': '#10AC84',
              'd_hat': '#EE5A6F', 'r_lz': '#F79F1F'}

    for idx, (benchmark, data) in enumerate(all_results.items()):
        ax = axes[idx]
        methods = data['methods']

        # Collect AUPRC values
        method_names = []
        auprc_values = []
        bar_colors = []

        # Add conformal ensemble
        if 'ASV_Conformal_Ensemble' in methods:
            method_names.append('Conformal\nEnsemble')
            auprc_values.append(methods['ASV_Conformal_Ensemble']['auprc'])
            bar_colors.append(colors['conformal'])

        # Add individual ASV signals
        if 'ASV_D_hat' in methods:
            method_names.append('ASV: D̂')
            auprc_values.append(methods['ASV_D_hat']['auprc'])
            bar_colors.append(colors['d_hat'])

        if 'ASV_r_LZ' in methods:
            method_names.append('ASV: r_LZ')
            auprc_values.append(methods['ASV_r_LZ']['auprc'])
            bar_colors.append(colors['r_lz'])

        # Add perplexity baseline
        if 'Baseline_Perplexity' in methods:
            method_names.append('Baseline:\nPerplexity')
            auprc_values.append(methods['Baseline_Perplexity']['auprc'])
            bar_colors.append(colors['perplexity'])

        # Plot bars
        x = np.arange(len(method_names))
        bars = ax.bar(x, auprc_values, color=bar_colors, alpha=0.85, width=0.6)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('AUPRC', fontsize=13, fontweight='bold')
        ax.set_title(benchmark_titles[benchmark], fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'auprc_comparison_conformal.png', dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_dir / 'auprc_comparison_conformal.png'}")
    plt.close()


def plot_ensemble_weights_comparison(output_dir: Path):
    """Plot ensemble weight distribution across tasks."""
    all_results = load_all_results()

    # Extract weights for each benchmark
    weights_data = []
    for benchmark, data in all_results.items():
        conformal_method = data['methods']['ASV_Conformal_Ensemble']
        weights = conformal_method.get('ensemble_weights', {})
        weights_data.append({
            'benchmark': benchmark.upper(),
            'D̂': weights.get('D_hat', 0.25),
            'coh★': weights.get('coh_star', 0.25),
            'r_LZ': weights.get('r_LZ', 0.25),
            'perplexity': weights.get('perplexity', 0.25)
        })

    df = pd.DataFrame(weights_data)
    df = df.set_index('benchmark')

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.7)

    ax.set_xlabel('Benchmark', fontsize=14, fontweight='bold')
    ax.set_ylabel('Weight', fontsize=14, fontweight='bold')
    ax.set_title('Learned Ensemble Weights per Task Type', fontsize=16, fontweight='bold')
    ax.legend(title='Signal', fontsize=11, title_fontsize=12, loc='upper right')
    ax.set_xticklabels(df.index, rotation=0, fontsize=12)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add weight values on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_weights_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_dir / 'ensemble_weights_comparison.png'}")
    plt.close()


def plot_performance_comparison(output_dir: Path):
    """Create comprehensive performance comparison chart."""
    all_results = load_all_results()

    # Extract AUROC for conformal vs baselines
    comparison_data = []
    for benchmark, data in all_results.items():
        methods = data['methods']

        conformal_auroc = methods['ASV_Conformal_Ensemble']['auroc']
        perp_auroc = methods.get('Baseline_Perplexity', {}).get('auroc', 0)
        d_hat_auroc = methods.get('ASV_D_hat', {}).get('auroc', 0)
        r_lz_auroc = methods.get('ASV_r_LZ', {}).get('auroc', 0)

        comparison_data.append({
            'Benchmark': benchmark.upper(),
            'Conformal\nEnsemble': conformal_auroc,
            'Baseline:\nPerplexity': perp_auroc,
            'ASV: D̂': d_hat_auroc,
            'ASV: r_LZ': r_lz_auroc
        })

    df = pd.DataFrame(comparison_data)
    df = df.set_index('Benchmark')

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(df.index))
    width = 0.2

    colors = ['#2E86DE', '#10AC84', '#EE5A6F', '#F79F1F']

    for i, (col, color) in enumerate(zip(df.columns, colors)):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, df[col], width, label=col, color=color, alpha=0.9)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Benchmark', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=14, fontweight='bold')
    ax.set_title('Performance Comparison: Conformal Ensemble vs Baselines',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=12)
    ax.legend(fontsize=11, loc='upper right', ncol=2)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    # Add horizontal line at 0.5 (random performance)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison_conformal.png', dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_dir / 'performance_comparison_conformal.png'}")
    plt.close()


def plot_calibration_quality(output_dir: Path):
    """Plot calibration quality metrics (coverage vs delta)."""
    all_results = load_all_results()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Calibration size per benchmark
    ax1 = axes[0]
    benchmarks = []
    cal_sizes = []
    coverages = []

    for benchmark, data in all_results.items():
        conformal_method = data['methods']['ASV_Conformal_Ensemble']
        benchmarks.append(benchmark.upper())
        cal_sizes.append(conformal_method.get('calibration_size', 0))
        coverages.append(conformal_method.get('coverage_guarantee', 0.95))

    bars = ax1.bar(benchmarks, cal_sizes, color='#5C7CFA', alpha=0.8)
    ax1.set_xlabel('Benchmark', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Calibration Set Size', fontsize=13, fontweight='bold')
    ax1.set_title('Calibration Set Sizes per Benchmark', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: Threshold values per benchmark
    ax2 = axes[1]
    thresholds = []
    for benchmark, data in all_results.items():
        conformal_method = data['methods']['ASV_Conformal_Ensemble']
        thresholds.append(conformal_method.get('conformal_threshold', 0.5))

    bars = ax2.bar(benchmarks, thresholds, color='#FF6B9D', alpha=0.8)
    ax2.set_xlabel('Benchmark', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Conformal Threshold (q₁₋δ)', fontsize=13, fontweight='bold')
    ax2.set_title('Conformal Thresholds (δ=0.05, 95% Coverage)', fontsize=14, fontweight='bold')
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Median')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.0])

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_quality.png', dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_dir / 'calibration_quality.png'}")
    plt.close()


def generate_summary_table(output_dir: Path):
    """Generate comprehensive summary table with conformal results."""
    all_results = load_all_results()

    rows = []
    for benchmark, data in all_results.items():
        methods = data['methods']

        for method_key, method_data in methods.items():
            # Format method name
            method_name = method_data['method_name']

            # Extract metrics
            row = {
                'Benchmark': benchmark.upper(),
                'Method': method_name,
                'AUROC': f"{method_data['auroc']:.3f}",
                'AUPRC': f"{method_data['auprc']:.3f}",
                'F1': f"{method_data.get('f1_optimal', 0):.3f}",
                'Acc': f"{method_data.get('accuracy_optimal', 0):.3f}",
                'Prec': f"{method_data.get('precision_optimal', 0):.3f}",
                'Recall': f"{method_data.get('recall_optimal', 0):.3f}"
            }

            # Add conformal-specific columns
            if 'conformal' in method_key.lower():
                row['Threshold'] = f"{method_data.get('conformal_threshold', 0):.4f}"
                row['Coverage'] = f"{method_data.get('coverage_guarantee', 0):.2f}"
                row['Cal Size'] = str(method_data.get('calibration_size', 0))
            else:
                row['Threshold'] = '-'
                row['Coverage'] = '-'
                row['Cal Size'] = '-'

            rows.append(row)

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = output_dir / 'conformal_summary_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Generated: {csv_path}")

    # Save as LaTeX
    latex_table = df.to_latex(index=False, escape=False)
    latex_path = output_dir / 'conformal_summary_table.tex'
    with open(latex_path, 'w') as f:
        f.write("% Summary of conformal evaluation results\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Conformal Prediction Performance Across Benchmarks}\n")
        f.write("\\label{tab:conformal_results}\n")
        f.write(latex_table)
        f.write("\\end{table}\n")
    print(f"✓ Generated: {latex_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Generate conformal verification visualizations')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for plots (default: figures/)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Split-Conformal Verification Visualizations")
    print("=" * 60)

    # Generate all plots
    print("\n1. AUROC Comparison Bars...")
    plot_auroc_comparison_bars(output_dir)

    print("\n2. AUPRC Comparison Bars...")
    plot_auprc_comparison_bars(output_dir)

    print("\n3. Ensemble Weights Comparison...")
    plot_ensemble_weights_comparison(output_dir)

    print("\n4. Performance Comparison...")
    plot_performance_comparison(output_dir)

    print("\n5. Calibration Quality Metrics...")
    plot_calibration_quality(output_dir)

    print("\n6. Summary Tables...")
    df_summary = generate_summary_table(output_dir)

    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)

    # Print key insights
    all_results = load_all_results()
    for benchmark, data in all_results.items():
        conformal = data['methods']['ASV_Conformal_Ensemble']
        print(f"\n{benchmark.upper()}:")
        print(f"  AUROC: {conformal['auroc']:.4f}")
        print(f"  Threshold: {conformal.get('conformal_threshold', 0):.4f}")
        print(f"  Coverage: {conformal.get('coverage_guarantee', 0):.2%}")
        print(f"  Calibration Size: {conformal.get('calibration_size', 0)}")

        weights = conformal.get('ensemble_weights', {})
        print(f"  Weights: D̂={weights.get('D_hat', 0):.2f}, " +
              f"coh★={weights.get('coh_star', 0):.2f}, " +
              f"r_LZ={weights.get('r_LZ', 0):.2f}, " +
              f"perplexity={weights.get('perplexity', 0):.2f}")

    print("\n" + "=" * 60)
    print(f"✓ All visualizations saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
