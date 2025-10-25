#!/usr/bin/env python3
"""
Re-analyze full public dataset with length filtering to address short-text false positives.

Context: OUTLIER_INSPECTION_FINDINGS.md revealed 76% of outliers are short responses
(1-10 words), not structural degeneracy. This script recomputes statistics with:
1. Length filtering: exclude responses < 10 tokens
2. Length normalization: r_LZ_norm = r_LZ * (1 + alpha/sqrt(n))
3. Compare both approaches to raw r_LZ

Goal: Provide corrected outlier statistics for Sections 6.4 and 7 of whitepaper.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuration
RESULTS_PATH = Path("/Users/roman.khokhla/my_stuff/kakeya/results/full_public_dataset_analysis/full_public_dataset_results.csv")
SUMMARY_PATH = Path("/Users/roman.khokhla/my_stuff/kakeya/results/full_public_dataset_analysis/full_analysis_summary.json")
JSONL_FILES = {
    'truthfulqa': Path("/Users/roman.khokhla/my_stuff/kakeya/data/llm_outputs/truthfulqa_outputs.jsonl"),
    'fever': Path("/Users/roman.khokhla/my_stuff/kakeya/data/llm_outputs/fever_outputs.jsonl"),
    'halueval': Path("/Users/roman.khokhla/my_stuff/kakeya/data/llm_outputs/halueval_outputs.jsonl"),
}
OUTPUT_DIR = Path("/Users/roman.khokhla/my_stuff/kakeya/results/corrected_public_dataset_analysis/")
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_LENGTH_TOKENS = 10  # Threshold from OUTLIER_INSPECTION_FINDINGS.md
ALPHA_NORMALIZATION = 5  # Length normalization parameter

print("="*80)
print("RE-ANALYSIS: Full Public Dataset with Length Filtering")
print("="*80)
print(f"\nAddressing short-text false positives (OUTLIER_INSPECTION_FINDINGS.md)")
print(f"- 76% of outliers were < 10 words (e.g., 'Canada', 'Steve Jobs')")
print(f"- Root cause: r_LZ conflates brevity with compressibility")
print(f"\nApproaches:")
print(f"1. Length filtering: exclude responses < {MIN_LENGTH_TOKENS} tokens")
print(f"2. Length normalization: r_LZ_norm = r_LZ * (1 + {ALPHA_NORMALIZATION}/sqrt(n))")
print("="*80)

# Load results
print(f"\n[1/8] Loading results from {RESULTS_PATH.name}...")
df = pd.read_csv(RESULTS_PATH)
print(f"✓ Loaded {len(df)} samples")

# Load summary
with open(SUMMARY_PATH) as f:
    summary = json.load(f)
    original_threshold = summary['distribution_stats']['threshold_outlier']
    print(f"✓ Original outlier threshold: {original_threshold:.4f}")

# Load text data to get token counts
print(f"\n[2/8] Loading text data for token counting...")
text_data = {}
for source, jsonl_path in JSONL_FILES.items():
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                sample_id = record['id']
                text = record.get('llm_response', '')
                # Simple token count (split on whitespace)
                n_tokens = len(text.split())
                text_data[sample_id] = {
                    'text': text,
                    'n_tokens': n_tokens,
                    'source': source
                }
        print(f"✓ Loaded {len([k for k in text_data.keys() if k.startswith(source)])} from {source}")

# Merge token counts into dataframe
print(f"\n[3/8] Computing token counts and length-normalized scores...")
df['n_tokens'] = df.apply(lambda row: text_data.get(row.get('sample_id', ''), {}).get('n_tokens', 0), axis=1)

# Compute length-normalized r_LZ
def compute_rlz_normalized(r_lz, n_tokens, alpha=ALPHA_NORMALIZATION):
    """Length normalization: r_LZ_norm = r_LZ * (1 + alpha/sqrt(n))"""
    if n_tokens < 1:
        return r_lz  # No normalization if token count unknown
    penalty = 1 + alpha / np.sqrt(n_tokens)
    return min(r_lz * penalty, 1.0)  # Cap at 1.0

df['r_lz_normalized'] = df.apply(lambda row: compute_rlz_normalized(row['asv_score'], row['n_tokens']), axis=1)

print(f"✓ Token counts: min={df['n_tokens'].min()}, median={df['n_tokens'].median():.1f}, max={df['n_tokens'].max()}")
print(f"✓ Computed r_LZ_normalized for all {len(df)} samples")

# Approach 1: Length filtering (exclude short texts)
print(f"\n[4/8] Approach 1: Length filtering (n_tokens >= {MIN_LENGTH_TOKENS})...")
df_filtered = df[df['n_tokens'] >= MIN_LENGTH_TOKENS].copy()
print(f"✓ Kept {len(df_filtered)}/{len(df)} samples ({len(df_filtered)/len(df)*100:.1f}%)")
print(f"✓ Excluded {len(df) - len(df_filtered)} short responses")

# Compute new outlier threshold (5th percentile) on filtered data
filtered_threshold = df_filtered['asv_score'].quantile(0.05)
filtered_outliers = df_filtered[df_filtered['asv_score'] <= filtered_threshold]
print(f"✓ New outlier threshold (5th percentile): {filtered_threshold:.4f}")
print(f"✓ Outliers detected (filtered): {len(filtered_outliers)} ({len(filtered_outliers)/len(df_filtered)*100:.1f}%)")

# Approach 2: Length normalization (use normalized scores)
print(f"\n[5/8] Approach 2: Length normalization (r_LZ_norm)...")
norm_threshold = df['r_lz_normalized'].quantile(0.05)
norm_outliers = df[df['r_lz_normalized'] <= norm_threshold]
print(f"✓ Normalized outlier threshold (5th percentile): {norm_threshold:.4f}")
print(f"✓ Outliers detected (normalized): {len(norm_outliers)} ({len(norm_outliers)/len(df)*100:.1f}%)")

# Check how many normalized outliers are still short texts
norm_outliers_short = norm_outliers[norm_outliers['n_tokens'] < MIN_LENGTH_TOKENS]
print(f"✓ Of {len(norm_outliers)} normalized outliers, {len(norm_outliers_short)} are still short texts ({len(norm_outliers_short)/len(norm_outliers)*100:.1f}%)")

# Distribution statistics for all three approaches
print(f"\n[6/8] Distribution statistics comparison...")

stats_comparison = {
    'Original (raw r_LZ)': {
        'n_samples': len(df),
        'mean': df['asv_score'].mean(),
        'std': df['asv_score'].std(),
        'median': df['asv_score'].median(),
        'q25': df['asv_score'].quantile(0.25),
        'q75': df['asv_score'].quantile(0.75),
        'outlier_threshold': original_threshold,
        'n_outliers': len(df[df['asv_score'] <= original_threshold]),
        'outlier_pct': len(df[df['asv_score'] <= original_threshold]) / len(df) * 100
    },
    'Filtered (n >= 10)': {
        'n_samples': len(df_filtered),
        'mean': df_filtered['asv_score'].mean(),
        'std': df_filtered['asv_score'].std(),
        'median': df_filtered['asv_score'].median(),
        'q25': df_filtered['asv_score'].quantile(0.25),
        'q75': df_filtered['asv_score'].quantile(0.75),
        'outlier_threshold': filtered_threshold,
        'n_outliers': len(filtered_outliers),
        'outlier_pct': len(filtered_outliers) / len(df_filtered) * 100
    },
    'Normalized (r_LZ_norm)': {
        'n_samples': len(df),
        'mean': df['r_lz_normalized'].mean(),
        'std': df['r_lz_normalized'].std(),
        'median': df['r_lz_normalized'].median(),
        'q25': df['r_lz_normalized'].quantile(0.25),
        'q75': df['r_lz_normalized'].quantile(0.75),
        'outlier_threshold': norm_threshold,
        'n_outliers': len(norm_outliers),
        'outlier_pct': len(norm_outliers) / len(df) * 100
    }
}

for approach, stats_dict in stats_comparison.items():
    print(f"\n{approach}:")
    print(f"  n_samples: {stats_dict['n_samples']}")
    print(f"  mean: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
    print(f"  median: {stats_dict['median']:.4f} [Q25={stats_dict['q25']:.4f}, Q75={stats_dict['q75']:.4f}]")
    print(f"  outlier_threshold: {stats_dict['outlier_threshold']:.4f}")
    print(f"  n_outliers: {stats_dict['n_outliers']} ({stats_dict['outlier_pct']:.1f}%)")

# Save corrected results
print(f"\n[7/8] Saving corrected analysis results...")

# Save filtered dataset
filtered_output = OUTPUT_DIR / "filtered_public_dataset_results.csv"
df_filtered.to_csv(filtered_output, index=False)
print(f"✓ Saved filtered results: {filtered_output}")

# Save normalized dataset
normalized_output = OUTPUT_DIR / "normalized_public_dataset_results.csv"
df[['sample_id', 'source', 'asv_score', 'r_lz_normalized', 'n_tokens']].to_csv(normalized_output, index=False)
print(f"✓ Saved normalized results: {normalized_output}")

# Save corrected summary
corrected_summary = {
    'metadata': {
        'total_samples': len(df),
        'filtered_samples': len(df_filtered),
        'normalization_alpha': ALPHA_NORMALIZATION,
        'min_length_tokens': MIN_LENGTH_TOKENS,
        'original_outliers': len(df[df['asv_score'] <= original_threshold]),
        'original_outliers_pct': len(df[df['asv_score'] <= original_threshold]) / len(df) * 100,
        'finding': '76% of original outliers were short texts (< 10 tokens)'
    },
    'approaches': stats_comparison,
    'recommendation': 'Use filtered approach (n >= 10 tokens) for production deployment to avoid false positives'
}

summary_output = OUTPUT_DIR / "corrected_analysis_summary.json"
with open(summary_output, 'w') as f:
    json.dump(corrected_summary, f, indent=2)
print(f"✓ Saved corrected summary: {summary_output}")

# Generate comparison visualizations
print(f"\n[8/8] Generating comparison visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Corrected Outlier Analysis: Addressing Short-Text False Positives', fontsize=16, fontweight='bold')

# Plot 1: Distribution comparison (histograms)
ax = axes[0, 0]
ax.hist(df['asv_score'], bins=50, alpha=0.5, label='Original r_LZ', color='blue')
ax.hist(df_filtered['asv_score'], bins=50, alpha=0.5, label=f'Filtered (n≥{MIN_LENGTH_TOKENS})', color='green')
ax.axvline(original_threshold, color='blue', linestyle='--', label=f'Original threshold ({original_threshold:.3f})')
ax.axvline(filtered_threshold, color='green', linestyle='--', label=f'Filtered threshold ({filtered_threshold:.3f})')
ax.set_xlabel('ASV Score (r_LZ)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution Comparison')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: r_LZ vs n_tokens scatter
ax = axes[0, 1]
scatter = ax.scatter(df['n_tokens'], df['asv_score'], alpha=0.3, s=10, c=df['asv_score'], cmap='RdYlGn')
ax.axhline(original_threshold, color='red', linestyle='--', label=f'Original threshold ({original_threshold:.3f})')
ax.axvline(MIN_LENGTH_TOKENS, color='orange', linestyle='--', label=f'Min length ({MIN_LENGTH_TOKENS} tokens)')
ax.set_xlabel('Number of Tokens')
ax.set_ylabel('ASV Score (r_LZ)')
ax.set_title('r_LZ vs Token Count (False Positive Region)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='r_LZ score')

# Plot 3: Box plots by approach
ax = axes[0, 2]
box_data = [
    df['asv_score'],
    df_filtered['asv_score'],
    df['r_lz_normalized']
]
bp = ax.boxplot(box_data, labels=['Original', f'Filtered\n(n≥{MIN_LENGTH_TOKENS})', 'Normalized'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['blue', 'green', 'purple']):
    patch.set_facecolor(color)
    patch.set_alpha(0.3)
ax.set_ylabel('ASV Score')
ax.set_title('Distribution Comparison (Box Plots)')
ax.grid(alpha=0.3, axis='y')

# Plot 4: Outlier counts comparison (bar chart)
ax = axes[1, 0]
approaches = ['Original\n(raw r_LZ)', 'Filtered\n(n≥10)', 'Normalized\n(r_LZ_norm)']
outlier_counts = [
    len(df[df['asv_score'] <= original_threshold]),
    len(filtered_outliers),
    len(norm_outliers)
]
outlier_pcts = [
    len(df[df['asv_score'] <= original_threshold]) / len(df) * 100,
    len(filtered_outliers) / len(df_filtered) * 100,
    len(norm_outliers) / len(df) * 100
]
bars = ax.bar(approaches, outlier_counts, color=['blue', 'green', 'purple'], alpha=0.6)
for bar, count, pct in zip(bars, outlier_counts, outlier_pcts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Number of Outliers')
ax.set_title('Outlier Counts by Approach')
ax.grid(alpha=0.3, axis='y')

# Plot 5: Length distribution of outliers
ax = axes[1, 1]
original_outliers = df[df['asv_score'] <= original_threshold]
ax.hist(original_outliers['n_tokens'], bins=30, alpha=0.6, label='Original outliers', color='red', edgecolor='black')
ax.hist(filtered_outliers['n_tokens'], bins=30, alpha=0.6, label='Filtered outliers', color='green', edgecolor='black')
ax.axvline(MIN_LENGTH_TOKENS, color='orange', linestyle='--', linewidth=2, label=f'Min length ({MIN_LENGTH_TOKENS} tokens)')
ax.set_xlabel('Number of Tokens')
ax.set_ylabel('Frequency')
ax.set_title('Token Length Distribution of Outliers')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 6: Source breakdown of outliers
ax = axes[1, 2]
original_outliers_sources = original_outliers.get('source', pd.Series(['unknown']*len(original_outliers))).value_counts()
filtered_outliers_sources = filtered_outliers.get('source', pd.Series(['unknown']*len(filtered_outliers))).value_counts()

x = np.arange(len(original_outliers_sources))
width = 0.35
bars1 = ax.bar(x - width/2, original_outliers_sources.values, width, label='Original', color='blue', alpha=0.6)
bars2 = ax.bar(x + width/2, filtered_outliers_sources.values, width, label='Filtered', color='green', alpha=0.6)

ax.set_xlabel('Source')
ax.set_ylabel('Number of Outliers')
ax.set_title('Outlier Distribution by Source')
ax.set_xticks(x)
ax.set_xticklabels(original_outliers_sources.index, rotation=45, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
fig_output = OUTPUT_DIR / "corrected_outlier_analysis.png"
plt.savefig(fig_output, dpi=300, bbox_inches='tight')
print(f"✓ Saved comparison figure: {fig_output}")
plt.close()

print("\n" + "="*80)
print("SUMMARY OF CORRECTED ANALYSIS")
print("="*80)
print(f"\nOriginal analysis (raw r_LZ):")
print(f"  - {len(df[df['asv_score'] <= original_threshold])} outliers ({len(df[df['asv_score'] <= original_threshold])/len(df)*100:.1f}%)")
print(f"  - 76% were short texts < 10 tokens (FALSE POSITIVES)")
print(f"\nCorrected analysis (filtered, n≥{MIN_LENGTH_TOKENS}):")
print(f"  - {len(filtered_outliers)} outliers ({len(filtered_outliers)/len(df_filtered)*100:.1f}%)")
print(f"  - Excluded {len(df) - len(df_filtered)} short responses")
print(f"  - Remaining outliers are more likely genuine structural anomalies")
print(f"\nAlternative (normalized r_LZ):")
print(f"  - {len(norm_outliers)} outliers ({len(norm_outliers)/len(df)*100:.1f}%)")
print(f"  - {len(norm_outliers_short)} still short texts ({len(norm_outliers_short)/len(norm_outliers)*100:.1f}%)")
print(f"\nRecommendation: Use FILTERED approach for whitepaper Section 6.4")
print(f"  - More conservative (fewer false positives)")
print(f"  - Cleaner interpretation (n≥{MIN_LENGTH_TOKENS} tokens)")
print(f"  - Honest limitation disclosure (excluded {len(df) - len(df_filtered)} short responses)")
print("="*80)

print(f"\n✅ Re-analysis complete!")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Files generated:")
print(f"  - filtered_public_dataset_results.csv ({len(df_filtered)} samples)")
print(f"  - normalized_public_dataset_results.csv ({len(df)} samples)")
print(f"  - corrected_analysis_summary.json")
print(f"  - corrected_outlier_analysis.png")
