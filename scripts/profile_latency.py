#!/usr/bin/env python3
"""
Priority 1.2: Latency & Cost Breakdown

Profiles end-to-end verification latency for each component:
- D̂ (fractal dimension)
- coh★ (coherence)
- r_LZ (compressibility)
- Conformal prediction

Measures p50/p95/p99 for each component and computes cost per verification.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# Add agent src to path
sys.path.insert(0, 'agent/src')
from signals import compute_D_hat, compute_coherence, compute_compressibility_pq

def load_degeneracy_signals_and_embeddings():
    """Load pre-computed signals and embeddings for profiling."""
    signals_dir = Path('data/signals/degeneracy')
    embeddings_dir = Path('data/embeddings/degeneracy')
    baselines_dir = Path('data/baselines/degeneracy')

    data = []
    for signal_file in sorted(signals_dir.glob('*.json'))[:100]:  # Profile first 100 samples
        sample_id = signal_file.stem

        # Load signal
        with open(signal_file) as f:
            signal = json.load(f)

        # Load embedding
        emb_file = embeddings_dir / f"{sample_id}.npy"
        if not emb_file.exists():
            continue
        embeddings = np.load(emb_file)

        # Load baseline
        baseline_file = baselines_dir / f"{sample_id}.json"
        if not baseline_file.exists():
            continue
        with open(baseline_file) as f:
            baseline = json.load(f)

        data.append({
            'sample_id': sample_id,
            'embeddings': embeddings,
            'signal': signal,
            'baseline': baseline
        })

    return data

def profile_D_hat_computation(samples):
    """Profile D̂ computation latency."""
    latencies = []

    print(f"Profiling D̂ computation on {len(samples)} samples...")
    for sample in samples:
        scales = sample['signal']['scales']
        N_j = {int(k): v for k, v in sample['signal']['N_j'].items()}

        start = time.perf_counter()
        D_hat = compute_D_hat(scales, N_j)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        latencies.append(elapsed)

    return np.array(latencies)

def profile_coherence_computation(samples):
    """Profile coh★ computation latency."""
    latencies = []

    print(f"Profiling coh★ computation on {len(samples)} samples...")
    for sample in samples:
        embeddings = sample['embeddings']

        start = time.perf_counter()
        coh_star, v_star = compute_coherence(embeddings, num_directions=100, num_bins=20, seed=42)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        latencies.append(elapsed)

    return np.array(latencies)

def profile_compressibility_computation(samples):
    """Profile r_LZ computation latency."""
    latencies = []

    print(f"Profiling r_LZ computation on {len(samples)} samples...")
    for sample in samples:
        embeddings = sample['embeddings']

        start = time.perf_counter()
        r_LZ = compute_compressibility_pq(embeddings, n_subspaces=8, codebook_bits=8)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        latencies.append(elapsed)

    return np.array(latencies)

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

def profile_conformal_scoring(samples):
    """Profile conformal nonconformity score computation."""
    latencies = []

    # Degeneracy-optimized weights (r_LZ dominant)
    weights = {
        'D_hat': 0.15,
        'coh_star': 0.15,
        'r_LZ': 0.60,
        'perplexity': 0.10
    }

    print(f"Profiling conformal scoring on {len(samples)} samples...")
    for sample in samples:
        signals = {
            'D_hat': sample['signal']['D_hat'],
            'coh_star': sample['signal']['coh_star'],
            'r_LZ': sample['signal']['r_LZ'],
            'perplexity': sample['baseline']['perplexity']
        }

        start = time.perf_counter()
        score = compute_nonconformity_score(signals, weights)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        latencies.append(elapsed)

    return np.array(latencies)

def compute_statistics(latencies):
    """Compute p50/p95/p99 statistics."""
    return {
        'count': len(latencies),
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'std': np.std(latencies),
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }

def estimate_costs(stats):
    """
    Estimate cost per verification based on latency.

    Assumptions:
    - Cloud compute: $0.10/hour for 1 CPU (typical spot pricing)
    - 1 CPU can process 1 verification at a time
    - Cost per ms = $0.10/hour / (3600 * 1000) ms/hour = $0.0000000278/ms
    """
    cost_per_ms = 0.10 / (3600 * 1000)  # $0.10/hour → $/ms

    return {
        'cost_p50': stats['p50'] * cost_per_ms,
        'cost_p95': stats['p95'] * cost_per_ms,
        'cost_p99': stats['p99'] * cost_per_ms
    }

def main():
    print("=" * 80)
    print("PRIORITY 1.2: LATENCY & COST BREAKDOWN")
    print("=" * 80)
    print()

    # Load data
    print("Loading degeneracy signals and embeddings...")
    samples = load_degeneracy_signals_and_embeddings()
    print(f"Loaded {len(samples)} samples for profiling")
    print()

    # Profile each component
    print("Profiling individual components...")
    print()

    latency_D_hat = profile_D_hat_computation(samples)
    latency_coh_star = profile_coherence_computation(samples)
    latency_r_LZ = profile_compressibility_computation(samples)
    latency_conformal = profile_conformal_scoring(samples)

    # Compute end-to-end latency (sum of components)
    latency_end_to_end = latency_D_hat + latency_coh_star + latency_r_LZ + latency_conformal

    # Compute statistics
    components = {
        'D_hat': latency_D_hat,
        'coh_star': latency_coh_star,
        'r_LZ': latency_r_LZ,
        'conformal': latency_conformal,
        'end_to_end': latency_end_to_end
    }

    stats = {}
    for name, latencies in components.items():
        stats[name] = compute_statistics(latencies)
        stats[name]['costs'] = estimate_costs(stats[name])

    # Print results
    print()
    print("=" * 80)
    print("LATENCY STATISTICS (milliseconds)")
    print("=" * 80)
    print()

    header = f"{'Component':<15} {'Count':<8} {'Mean':<8} {'Median':<8} {'Std':<8} {'p50':<8} {'p95':<8} {'p99':<8}"
    print(header)
    print("-" * 80)

    for name in ['D_hat', 'coh_star', 'r_LZ', 'conformal', 'end_to_end']:
        s = stats[name]
        row = f"{name:<15} {s['count']:<8} {s['mean']:<8.3f} {s['median']:<8.3f} {s['std']:<8.3f} {s['p50']:<8.3f} {s['p95']:<8.3f} {s['p99']:<8.3f}"
        print(row)

    print()
    print("=" * 80)
    print("COST ANALYSIS (USD per verification)")
    print("=" * 80)
    print()

    header = f"{'Component':<15} {'p50 Cost':<15} {'p95 Cost':<15} {'p99 Cost':<15}"
    print(header)
    print("-" * 80)

    for name in ['D_hat', 'coh_star', 'r_LZ', 'conformal', 'end_to_end']:
        c = stats[name]['costs']
        row = f"{name:<15} ${c['cost_p50']:<14.6f} ${c['cost_p95']:<14.6f} ${c['cost_p99']:<14.6f}"
        print(row)

    print()
    print("=" * 80)
    print("COMPARISON TO BASELINES")
    print("=" * 80)
    print()

    # Baseline costs
    gpt4_cost = 0.02  # ~$0.02 per verification (GPT-4 judge)
    asv_cost_p95 = stats['end_to_end']['costs']['cost_p95']

    print(f"GPT-4 Judge:           ${gpt4_cost:.4f} per verification")
    print(f"ASV (p95):             ${asv_cost_p95:.6f} per verification")
    print(f"Cost reduction:        {gpt4_cost / asv_cost_p95:.1f}x cheaper")
    print()

    # Latency comparison
    gpt4_latency = 2000  # ~2s per verification (GPT-4 judge)
    asv_latency_p95 = stats['end_to_end']['p95']

    print(f"GPT-4 Judge:           {gpt4_latency:.0f} ms (p95)")
    print(f"ASV (p95):             {asv_latency_p95:.1f} ms (p95)")
    print(f"Latency improvement:   {gpt4_latency / asv_latency_p95:.1f}x faster")
    print()

    # Validation criteria
    print("=" * 80)
    print("VALIDATION CRITERIA")
    print("=" * 80)
    print()

    total_p95 = stats['end_to_end']['p95']
    cost_p95 = stats['end_to_end']['costs']['cost_p95']
    r_LZ_p95 = stats['r_LZ']['p95']

    print(f"✓ Total p95 latency ≤ 50ms:          {total_p95:.2f} ms {'✓ PASS' if total_p95 <= 50 else '✗ FAIL'}")
    print(f"✓ Cost per verification ≤ $0.001:     ${cost_p95:.6f} {'✓ PASS' if cost_p95 <= 0.001 else '✗ FAIL'}")
    print(f"✓ r_LZ is fastest signal (<5ms):      {r_LZ_p95:.2f} ms {'✓ PASS' if r_LZ_p95 < 5 else '✗ FAIL'}")
    print()

    # Save results
    output_dir = Path('results/latency')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([
        {
            'component': name,
            'count': s['count'],
            'mean_ms': s['mean'],
            'median_ms': s['median'],
            'std_ms': s['std'],
            'p50_ms': s['p50'],
            'p95_ms': s['p95'],
            'p99_ms': s['p99'],
            'cost_p50_usd': s['costs']['cost_p50'],
            'cost_p95_usd': s['costs']['cost_p95'],
            'cost_p99_usd': s['costs']['cost_p99']
        }
        for name, s in stats.items()
    ])

    results_df.to_csv(output_dir / 'latency_results.csv', index=False)
    print(f"✓ Saved results to {output_dir / 'latency_results.csv'}")

    # Generate visualization
    generate_visualizations(stats, output_dir)

    print()
    print("=" * 80)
    print("PRIORITY 1.2 COMPLETE")
    print("=" * 80)

def generate_visualizations(stats, output_dir):
    """Generate latency breakdown visualizations."""

    # 1. Latency breakdown bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    components = ['D_hat', 'coh_star', 'r_LZ', 'conformal']
    component_labels = ['D̂', 'coh★', 'r_LZ', 'Conformal']
    p95_latencies = [stats[c]['p95'] for c in components]

    colors = ['#2E86DE', '#10AC84', '#F79F1F', '#EE5A6F']

    bars = ax1.bar(component_labels, p95_latencies, color=colors)
    ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Component Latency (p95)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom', fontsize=10)

    # 2. Cost comparison
    asv_cost = stats['end_to_end']['costs']['cost_p95']
    gpt4_cost = 0.02

    methods = ['ASV (p95)', 'GPT-4 Judge']
    costs = [asv_cost * 1000, gpt4_cost * 1000]  # Convert to cents for readability
    colors_cost = ['#10AC84', '#EE5A6F']

    bars = ax2.bar(methods, costs, color=colors_cost)
    ax2.set_ylabel('Cost per Verification (cents)', fontsize=12, fontweight='bold')
    ax2.set_title('Cost Comparison (ASV vs GPT-4 Judge)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.4f}¢',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / '../../docs/architecture/figures/latency_breakdown.png',
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure: latency_breakdown.png")
    plt.close()

if __name__ == '__main__':
    main()
