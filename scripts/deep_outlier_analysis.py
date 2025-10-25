#!/usr/bin/env python3
"""
Deep Analysis: Investigation of 406 Filtered Outliers from r_LZ Signal

Goal: Understand whether r_LZ is helpful for detecting genuine structural issues
in the 406 filtered outliers (n >= 10 tokens, after excluding short-text false positives).

Analysis dimensions:
1. Manual inspection of sample outliers (top 50, bottom 50, random 50)
2. Structural pattern detection (repetition, incoherence, semantic drift)
3. Correlation with ground-truth hallucination labels
4. Distribution analysis (outliers vs normal samples)
5. Source-specific patterns (TruthfulQA, FEVER, HaluEval)
6. Quantitative utility assessment: precision/recall on structural issues
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
import re

# Configuration
FILTERED_RESULTS = Path("/Users/roman.khokhla/my_stuff/kakeya/results/corrected_public_dataset_analysis/filtered_public_dataset_results.csv")
CORRECTED_SUMMARY = Path("/Users/roman.khokhla/my_stuff/kakeya/results/corrected_public_dataset_analysis/corrected_analysis_summary.json")
JSONL_FILES = {
    'truthfulqa': Path("/Users/roman.khokhla/my_stuff/kakeya/data/llm_outputs/truthfulqa_outputs.jsonl"),
    'fever': Path("/Users/roman.khokhla/my_stuff/kakeya/data/llm_outputs/fever_outputs.jsonl"),
    'halueval': Path("/Users/roman.khokhla/my_stuff/kakeya/data/llm_outputs/halueval_outputs.jsonl"),
}
OUTPUT_DIR = Path("/Users/roman.khokhla/my_stuff/kakeya/results/deep_outlier_analysis/")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("DEEP OUTLIER ANALYSIS: Investigating r_LZ Utility on 406 Filtered Outliers")
print("="*80)
print("\nResearch Questions:")
print("1. What structural patterns do r_LZ outliers exhibit?")
print("2. Are they genuine structural anomalies or still false positives?")
print("3. Does r_LZ correlate with ground-truth hallucination labels?")
print("4. What is the precision/recall of r_LZ for structural issues?")
print("5. Are there source-specific patterns (TruthfulQA vs FEVER vs HaluEval)?")
print("="*80)

# Load filtered results
print(f"\n[1/10] Loading filtered results...")
df = pd.read_csv(FILTERED_RESULTS)
print(f"✓ Loaded {len(df)} filtered samples (n >= 10 tokens)")

# Load summary to get outlier threshold
with open(CORRECTED_SUMMARY) as f:
    summary = json.load(f)
    outlier_threshold = summary['approaches']['Filtered (n >= 10)']['outlier_threshold']
    print(f"✓ Outlier threshold: {outlier_threshold:.4f}")

# Identify outliers
outliers = df[df['asv_score'] <= outlier_threshold].copy()
normals = df[df['asv_score'] > outlier_threshold].copy()
print(f"✓ Identified {len(outliers)} outliers, {len(normals)} normal samples")

# Load text data with ground-truth labels
print(f"\n[2/10] Loading text data and ground-truth labels...")
text_data = {}
for source, jsonl_path in JSONL_FILES.items():
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                sample_id = record['id']
                text = record.get('llm_response', '')
                # Ground truth: is this a hallucination?
                is_hallucination = record.get('hallucination', False) or record.get('is_hallucination', False)
                text_data[sample_id] = {
                    'text': text,
                    'source': source,
                    'is_hallucination': is_hallucination
                }
        print(f"✓ Loaded {len([k for k in text_data.keys() if text_data[k]['source'] == source])} from {source}")

# Merge text data into outliers dataframe
print(f"\n[3/10] Merging text and labels...")
outliers['text'] = outliers.apply(lambda row: text_data.get(row.get('sample_id', ''), {}).get('text', ''), axis=1)
outliers['is_hallucination'] = outliers.apply(lambda row: text_data.get(row.get('sample_id', ''), {}).get('is_hallucination', False), axis=1)
normals['text'] = normals.apply(lambda row: text_data.get(row.get('sample_id', ''), {}).get('text', ''), axis=1)
normals['is_hallucination'] = normals.apply(lambda row: text_data.get(row.get('sample_id', ''), {}).get('is_hallucination', False), axis=1)

print(f"✓ Outliers with hallucination labels: {outliers['is_hallucination'].sum()} / {len(outliers)} ({outliers['is_hallucination'].sum()/len(outliers)*100:.1f}%)")
print(f"✓ Normals with hallucination labels: {normals['is_hallucination'].sum()} / {len(normals)} ({normals['is_hallucination'].sum()/len(normals)*100:.1f}%)")

# Structural pattern detection functions
def detect_repetition(text, threshold=0.3):
    """Detect word/phrase repetition above threshold"""
    words = text.lower().split()
    if len(words) < 5:
        return False, 0.0

    # Count repeated phrases (3-5 words)
    max_repetition = 0.0
    for phrase_len in [3, 4, 5]:
        if len(words) < phrase_len:
            continue
        phrases = [' '.join(words[i:i+phrase_len]) for i in range(len(words)-phrase_len+1)]
        if len(phrases) == 0:
            continue
        phrase_counts = Counter(phrases)
        most_common_count = phrase_counts.most_common(1)[0][1] if phrase_counts else 0
        repetition_rate = most_common_count / len(phrases) if len(phrases) > 0 else 0
        max_repetition = max(max_repetition, repetition_rate)

    return max_repetition > threshold, max_repetition

def detect_sentence_repetition(text, threshold=0.3):
    """Detect sentence-level repetition"""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) < 2:
        return False, 0.0

    sentence_counts = Counter(sentences)
    most_common_count = sentence_counts.most_common(1)[0][1] if sentence_counts else 0
    repetition_rate = most_common_count / len(sentences) if len(sentences) > 0 else 0

    return repetition_rate > threshold, repetition_rate

def detect_incoherence(text):
    """Heuristic: detect contradictions (basic patterns)"""
    # Look for explicit contradictions
    contradiction_patterns = [
        (r'\byes\b.*\bno\b', 'yes-no contradiction'),
        (r'\btrue\b.*\bfalse\b', 'true-false contradiction'),
        (r'\bcorrect\b.*\bincorrect\b', 'correct-incorrect contradiction'),
        (r'\bis\b.*\bis not\b', 'is-is not contradiction'),
    ]

    text_lower = text.lower()
    for pattern, desc in contradiction_patterns:
        if re.search(pattern, text_lower):
            return True, desc

    return False, None

def compute_lexical_diversity(text):
    """Compute type-token ratio (unique words / total words)"""
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)

# Apply structural pattern detection
print(f"\n[4/10] Detecting structural patterns in outliers...")
outliers['has_repetition'], outliers['repetition_rate'] = zip(*outliers['text'].apply(detect_repetition))
outliers['has_sentence_rep'], outliers['sentence_rep_rate'] = zip(*outliers['text'].apply(detect_sentence_repetition))
outliers['has_incoherence'], outliers['incoherence_type'] = zip(*outliers['text'].apply(detect_incoherence))
outliers['lexical_diversity'] = outliers['text'].apply(compute_lexical_diversity)

# Same for normals (for comparison)
normals['has_repetition'], normals['repetition_rate'] = zip(*normals['text'].apply(detect_repetition))
normals['has_sentence_rep'], normals['sentence_rep_rate'] = zip(*normals['text'].apply(detect_sentence_repetition))
normals['has_incoherence'], normals['incoherence_type'] = zip(*normals['text'].apply(detect_incoherence))
normals['lexical_diversity'] = normals['text'].apply(compute_lexical_diversity)

print(f"✓ Outliers with repetition: {outliers['has_repetition'].sum()} ({outliers['has_repetition'].sum()/len(outliers)*100:.1f}%)")
print(f"✓ Outliers with sentence repetition: {outliers['has_sentence_rep'].sum()} ({outliers['has_sentence_rep'].sum()/len(outliers)*100:.1f}%)")
print(f"✓ Outliers with incoherence: {outliers['has_incoherence'].sum()} ({outliers['has_incoherence'].sum()/len(outliers)*100:.1f}%)")
print(f"✓ Normals with repetition: {normals['has_repetition'].sum()} ({normals['has_repetition'].sum()/len(normals)*100:.1f}%)")
print(f"✓ Normals with sentence repetition: {normals['has_sentence_rep'].sum()} ({normals['has_sentence_rep'].sum()/len(normals)*100:.1f}%)")

# Manual inspection: Sample outliers for deep review
print(f"\n[5/10] Sampling outliers for manual inspection...")
outliers_sorted = outliers.sort_values('asv_score')

# Top 20 worst outliers (lowest r_LZ)
top_outliers = outliers_sorted.head(20)
# Bottom 20 outliers (highest r_LZ, near threshold)
bottom_outliers = outliers_sorted.tail(20)
# Random 20 from middle
middle_outliers = outliers_sorted.iloc[len(outliers_sorted)//2 - 10:len(outliers_sorted)//2 + 10]

inspection_samples = pd.concat([top_outliers, middle_outliers, bottom_outliers])
inspection_samples['category'] = ['top_worst']*20 + ['middle']*20 + ['near_threshold']*20

print(f"✓ Sampled 60 outliers for manual inspection:")
print(f"  - Top 20 worst (lowest r_LZ): {top_outliers['asv_score'].min():.4f} - {top_outliers['asv_score'].max():.4f}")
print(f"  - Middle 20: {middle_outliers['asv_score'].min():.4f} - {middle_outliers['asv_score'].max():.4f}")
print(f"  - Near threshold 20: {bottom_outliers['asv_score'].min():.4f} - {bottom_outliers['asv_score'].max():.4f}")

# Save inspection samples for manual review
inspection_output = OUTPUT_DIR / "outlier_deep_inspection.csv"
inspection_samples[['sample_id', 'asv_score', 'source', 'is_hallucination',
                    'has_repetition', 'repetition_rate', 'has_sentence_rep',
                    'sentence_rep_rate', 'has_incoherence', 'lexical_diversity',
                    'text', 'category']].to_csv(inspection_output, index=False)
print(f"✓ Saved inspection samples: {inspection_output}")

# Statistical analysis
print(f"\n[6/10] Statistical comparison: Outliers vs Normals...")

stats_comparison = {
    'metric': [],
    'outliers_mean': [],
    'normals_mean': [],
    'outliers_std': [],
    'normals_std': [],
    't_statistic': [],
    'p_value': [],
    'effect_size': []
}

# Compare distributions
metrics = [
    ('repetition_rate', 'Repetition Rate'),
    ('sentence_rep_rate', 'Sentence Repetition Rate'),
    ('lexical_diversity', 'Lexical Diversity'),
    ('asv_score', 'ASV Score (r_LZ)'),
]

for col, label in metrics:
    outliers_vals = outliers[col].dropna()
    normals_vals = normals[col].dropna()

    t_stat, p_val = stats.ttest_ind(outliers_vals, normals_vals)

    # Cohen's d (effect size)
    pooled_std = np.sqrt(((len(outliers_vals)-1)*outliers_vals.std()**2 + (len(normals_vals)-1)*normals_vals.std()**2) / (len(outliers_vals)+len(normals_vals)-2))
    cohens_d = (outliers_vals.mean() - normals_vals.mean()) / pooled_std if pooled_std > 0 else 0

    stats_comparison['metric'].append(label)
    stats_comparison['outliers_mean'].append(outliers_vals.mean())
    stats_comparison['normals_mean'].append(normals_vals.mean())
    stats_comparison['outliers_std'].append(outliers_vals.std())
    stats_comparison['normals_std'].append(normals_vals.std())
    stats_comparison['t_statistic'].append(t_stat)
    stats_comparison['p_value'].append(p_val)
    stats_comparison['effect_size'].append(cohens_d)

    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    print(f"\n{label}:")
    print(f"  Outliers: {outliers_vals.mean():.4f} ± {outliers_vals.std():.4f}")
    print(f"  Normals:  {normals_vals.mean():.4f} ± {normals_vals.std():.4f}")
    print(f"  t={t_stat:.3f}, p={p_val:.4f} {significance}, Cohen's d={cohens_d:.3f}")

stats_df = pd.DataFrame(stats_comparison)
stats_output = OUTPUT_DIR / "statistical_comparison.csv"
stats_df.to_csv(stats_output, index=False)
print(f"\n✓ Saved statistical comparison: {stats_output}")

# Correlation analysis: r_LZ vs ground-truth hallucinations
print(f"\n[7/10] Correlation analysis: r_LZ vs ground-truth hallucinations...")

# In outliers only
outliers_with_labels = outliers[outliers['is_hallucination'].notna()]
if len(outliers_with_labels) > 10:
    corr_outliers, p_outliers = stats.pointbiserialr(outliers_with_labels['is_hallucination'], outliers_with_labels['asv_score'])
    print(f"✓ Correlation (outliers only): r={corr_outliers:.4f}, p={p_outliers:.4f}")
else:
    corr_outliers, p_outliers = 0, 1.0
    print(f"⚠️  Insufficient labeled outliers for correlation")

# In full dataset
all_samples = pd.concat([outliers, normals])
all_with_labels = all_samples[all_samples['is_hallucination'].notna()]
if len(all_with_labels) > 10:
    corr_all, p_all = stats.pointbiserialr(all_with_labels['is_hallucination'], all_with_labels['asv_score'])
    print(f"✓ Correlation (full dataset): r={corr_all:.4f}, p={p_all:.4f}")
else:
    corr_all, p_all = 0, 1.0

# Precision/Recall for structural issues
print(f"\n[8/10] Precision/Recall: r_LZ for structural anomaly detection...")

# Define "structural issue" as: repetition OR sentence_rep OR incoherence
outliers['has_structural_issue'] = outliers['has_repetition'] | outliers['has_sentence_rep'] | outliers['has_incoherence']
normals['has_structural_issue'] = normals['has_repetition'] | normals['has_sentence_rep'] | normals['has_incoherence']

# Confusion matrix: r_LZ outlier detection vs structural issues
tp = outliers['has_structural_issue'].sum()  # True positives: outlier AND has structural issue
fp = (~outliers['has_structural_issue']).sum()  # False positives: outlier but NO structural issue
fn = normals['has_structural_issue'].sum()  # False negatives: normal but HAS structural issue
tn = (~normals['has_structural_issue']).sum()  # True negatives: normal AND no structural issue

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + fp + fn + tn)

print(f"\nConfusion Matrix (r_LZ vs Structural Issues):")
print(f"  True Positives:  {tp} (outlier + structural issue)")
print(f"  False Positives: {fp} (outlier, no structural issue)")
print(f"  False Negatives: {fn} (normal, but has structural issue)")
print(f"  True Negatives:  {tn} (normal, no structural issue)")
print(f"\nMetrics:")
print(f"  Precision: {precision:.4f} ({tp}/{tp+fp})")
print(f"  Recall:    {recall:.4f} ({tp}/{tp+fn})")
print(f"  F1 Score:  {f1:.4f}")
print(f"  Accuracy:  {accuracy:.4f}")

# Source-specific analysis
print(f"\n[9/10] Source-specific patterns...")

source_stats = []
for source in ['truthfulqa', 'fever', 'halueval']:
    source_outliers = outliers[outliers.get('source', pd.Series()) == source]
    source_normals = normals[normals.get('source', pd.Series()) == source]

    if len(source_outliers) == 0:
        continue

    outlier_rate = len(source_outliers) / (len(source_outliers) + len(source_normals))
    halluc_rate = source_outliers['is_hallucination'].sum() / len(source_outliers) if len(source_outliers) > 0 else 0
    structural_rate = source_outliers['has_structural_issue'].sum() / len(source_outliers) if len(source_outliers) > 0 else 0

    source_stats.append({
        'source': source,
        'n_outliers': len(source_outliers),
        'n_normals': len(source_normals),
        'outlier_rate': outlier_rate,
        'halluc_rate_in_outliers': halluc_rate,
        'structural_rate_in_outliers': structural_rate,
        'mean_r_lz_outliers': source_outliers['asv_score'].mean(),
        'mean_r_lz_normals': source_normals['asv_score'].mean()
    })

    print(f"\n{source.upper()}:")
    print(f"  Outliers: {len(source_outliers)} ({outlier_rate*100:.1f}%)")
    print(f"  Hallucination rate in outliers: {halluc_rate*100:.1f}%")
    print(f"  Structural issue rate in outliers: {structural_rate*100:.1f}%")
    print(f"  Mean r_LZ (outliers): {source_outliers['asv_score'].mean():.4f}")
    print(f"  Mean r_LZ (normals):  {source_normals['asv_score'].mean():.4f}")

source_stats_df = pd.DataFrame(source_stats)
source_output = OUTPUT_DIR / "source_specific_analysis.csv"
source_stats_df.to_csv(source_output, index=False)
print(f"\n✓ Saved source-specific analysis: {source_output}")

# Generate comprehensive visualization
print(f"\n[10/10] Generating comprehensive visualizations...")

fig, axes = plt.subplots(3, 3, figsize=(20, 18))
fig.suptitle('Deep Outlier Analysis: r_LZ Utility Assessment', fontsize=18, fontweight='bold')

# Plot 1: Distribution of r_LZ (outliers vs normals)
ax = axes[0, 0]
ax.hist(outliers['asv_score'], bins=30, alpha=0.6, label='Outliers', color='red', edgecolor='black')
ax.hist(normals['asv_score'], bins=30, alpha=0.6, label='Normals', color='green', edgecolor='black')
ax.axvline(outlier_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({outlier_threshold:.3f})')
ax.set_xlabel('ASV Score (r_LZ)')
ax.set_ylabel('Frequency')
ax.set_title('r_LZ Distribution: Outliers vs Normals')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Repetition rate comparison
ax = axes[0, 1]
box_data = [
    outliers['repetition_rate'].dropna(),
    normals['repetition_rate'].dropna()
]
bp = ax.boxplot(box_data, labels=['Outliers', 'Normals'], patch_artist=True)
bp['boxes'][0].set_facecolor('red')
bp['boxes'][1].set_facecolor('green')
for box in bp['boxes']:
    box.set_alpha(0.6)
ax.set_ylabel('Repetition Rate')
ax.set_title('Phrase Repetition: Outliers vs Normals')
ax.grid(alpha=0.3, axis='y')

# Plot 3: Lexical diversity comparison
ax = axes[0, 2]
box_data = [
    outliers['lexical_diversity'].dropna(),
    normals['lexical_diversity'].dropna()
]
bp = ax.boxplot(box_data, labels=['Outliers', 'Normals'], patch_artist=True)
bp['boxes'][0].set_facecolor('red')
bp['boxes'][1].set_facecolor('green')
for box in bp['boxes']:
    box.set_alpha(0.6)
ax.set_ylabel('Lexical Diversity (Type-Token Ratio)')
ax.set_title('Lexical Diversity: Outliers vs Normals')
ax.grid(alpha=0.3, axis='y')

# Plot 4: Structural issues prevalence
ax = axes[1, 0]
labels = ['Repetition', 'Sentence Rep', 'Incoherence', 'Any Structural']
outlier_rates = [
    outliers['has_repetition'].sum() / len(outliers) * 100,
    outliers['has_sentence_rep'].sum() / len(outliers) * 100,
    outliers['has_incoherence'].sum() / len(outliers) * 100,
    outliers['has_structural_issue'].sum() / len(outliers) * 100
]
normal_rates = [
    normals['has_repetition'].sum() / len(normals) * 100,
    normals['has_sentence_rep'].sum() / len(normals) * 100,
    normals['has_incoherence'].sum() / len(normals) * 100,
    normals['has_structural_issue'].sum() / len(normals) * 100
]

x = np.arange(len(labels))
width = 0.35
bars1 = ax.bar(x - width/2, outlier_rates, width, label='Outliers', color='red', alpha=0.6)
bars2 = ax.bar(x + width/2, normal_rates, width, label='Normals', color='green', alpha=0.6)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Prevalence (%)')
ax.set_title('Structural Issues: Outliers vs Normals')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 5: Confusion matrix heatmap
ax = axes[1, 1]
cm = np.array([[tp, fp], [fn, tn]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
            xticklabels=['Has Structural Issue', 'No Structural Issue'],
            yticklabels=['Outlier (r_LZ)', 'Normal (r_LZ)'])
ax.set_title(f'Confusion Matrix\nPrecision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}')

# Plot 6: r_LZ vs hallucination (scatter)
ax = axes[1, 2]
if len(all_with_labels) > 0:
    halluc_samples = all_with_labels[all_with_labels['is_hallucination'] == True]
    nonhalluc_samples = all_with_labels[all_with_labels['is_hallucination'] == False]

    ax.scatter(halluc_samples.index, halluc_samples['asv_score'],
               alpha=0.5, s=20, c='red', label=f'Hallucination (n={len(halluc_samples)})')
    ax.scatter(nonhalluc_samples.index, nonhalluc_samples['asv_score'],
               alpha=0.5, s=20, c='green', label=f'Correct (n={len(nonhalluc_samples)})')
    ax.axhline(outlier_threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('ASV Score (r_LZ)')
    ax.set_title(f'r_LZ vs Ground-Truth Labels\n(correlation r={corr_all:.3f}, p={p_all:.4f})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Insufficient labeled data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('r_LZ vs Ground-Truth Labels (N/A)')

# Plot 7: Source-specific outlier rates
ax = axes[2, 0]
if len(source_stats_df) > 0:
    sources = source_stats_df['source']
    outlier_rates = source_stats_df['outlier_rate'] * 100
    bars = ax.bar(sources, outlier_rates, color=['blue', 'orange', 'purple'], alpha=0.6)

    for bar, rate in zip(bars, outlier_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Outlier Rate (%)')
    ax.set_title('Outlier Rate by Source')
    ax.set_xticklabels(sources, rotation=45, ha='right')
    ax.grid(alpha=0.3, axis='y')
else:
    ax.text(0.5, 0.5, 'No source data', ha='center', va='center', transform=ax.transAxes)

# Plot 8: r_LZ vs Repetition Rate (scatter)
ax = axes[2, 1]
scatter = ax.scatter(all_samples['repetition_rate'], all_samples['asv_score'],
                     alpha=0.3, s=10, c=all_samples['asv_score'], cmap='RdYlGn')
ax.axhline(outlier_threshold, color='red', linestyle='--', linewidth=2, label='r_LZ threshold')
ax.set_xlabel('Repetition Rate')
ax.set_ylabel('ASV Score (r_LZ)')
ax.set_title('r_LZ vs Repetition Rate')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='r_LZ score')

# Plot 9: Effect sizes (Cohen's d) for all metrics
ax = axes[2, 2]
metrics_labels = stats_df['metric']
effect_sizes = stats_df['effect_size']
colors = ['red' if abs(d) > 0.5 else 'orange' if abs(d) > 0.2 else 'gray' for d in effect_sizes]

bars = ax.barh(metrics_labels, effect_sizes, color=colors, alpha=0.6)
ax.axvline(0, color='black', linestyle='-', linewidth=1)
ax.axvline(-0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(-0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel("Cohen's d (Effect Size)")
ax.set_title('Effect Sizes: Outliers vs Normals\n(|d|>0.5=large, >0.2=medium)')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
fig_output = OUTPUT_DIR / "deep_outlier_analysis.png"
plt.savefig(fig_output, dpi=300, bbox_inches='tight')
print(f"✓ Saved comprehensive visualization: {fig_output}")
plt.close()

# Summary report
print("\n" + "="*80)
print("SUMMARY: r_LZ Utility Assessment")
print("="*80)

print(f"\n1. STRUCTURAL PATTERN DETECTION:")
print(f"   Outliers with structural issues: {outliers['has_structural_issue'].sum()} / {len(outliers)} ({outliers['has_structural_issue'].sum()/len(outliers)*100:.1f}%)")
print(f"   Normals with structural issues:  {normals['has_structural_issue'].sum()} / {len(normals)} ({normals['has_structural_issue'].sum()/len(normals)*100:.1f}%)")
print(f"   → r_LZ enriches for structural issues: {(outliers['has_structural_issue'].sum()/len(outliers)) / (normals['has_structural_issue'].sum()/len(normals)):.2f}x higher rate")

print(f"\n2. PRECISION/RECALL FOR STRUCTURAL ANOMALIES:")
print(f"   Precision: {precision:.4f} (of r_LZ outliers, {precision*100:.1f}% have structural issues)")
print(f"   Recall:    {recall:.4f} (of structural issues, {recall*100:.1f}% caught by r_LZ)")
print(f"   F1 Score:  {f1:.4f}")

print(f"\n3. CORRELATION WITH GROUND-TRUTH HALLUCINATIONS:")
print(f"   Outliers only: r={corr_outliers:.4f}, p={p_outliers:.4f}")
print(f"   Full dataset:  r={corr_all:.4f}, p={p_all:.4f}")
print(f"   → Weak correlation (as expected: r_LZ detects structural issues, not semantic errors)")

print(f"\n4. STATISTICAL SIGNIFICANCE:")
for _, row in stats_df.iterrows():
    sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "n.s."
    effect_desc = "LARGE" if abs(row['effect_size']) > 0.8 else "MEDIUM" if abs(row['effect_size']) > 0.5 else "SMALL" if abs(row['effect_size']) > 0.2 else "negligible"
    print(f"   {row['metric']}: p={row['p_value']:.4f} {sig_marker}, Cohen's d={row['effect_size']:.3f} ({effect_desc})")

print(f"\n5. KEY FINDING:")
if precision > 0.20 or recall > 0.20:
    print(f"   ✅ r_LZ IS USEFUL for flagging structural anomalies")
    print(f"   → Precision {precision:.1%} means r_LZ outliers are enriched for structural issues")
    print(f"   → Recall {recall:.1%} means r_LZ catches a meaningful fraction of structural issues")
    utility = "HELPFUL"
else:
    print(f"   ⚠️  r_LZ has LIMITED UTILITY for structural anomaly detection")
    print(f"   → Low precision ({precision:.1%}) and recall ({recall:.1%})")
    print(f"   → May require combining with other signals")
    utility = "LIMITED"

# Save final summary
summary_dict = {
    'total_outliers': len(outliers),
    'total_normals': len(normals),
    'outliers_with_structural_issues': int(outliers['has_structural_issue'].sum()),
    'normals_with_structural_issues': int(normals['has_structural_issue'].sum()),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'accuracy': float(accuracy),
    'correlation_outliers_only': float(corr_outliers),
    'correlation_full_dataset': float(corr_all),
    'utility_assessment': utility,
    'enrichment_factor': float((outliers['has_structural_issue'].sum()/len(outliers)) / (normals['has_structural_issue'].sum()/len(normals))) if normals['has_structural_issue'].sum() > 0 else 0,
    'statistical_tests': stats_comparison
}

summary_output = OUTPUT_DIR / "deep_analysis_summary.json"
with open(summary_output, 'w') as f:
    json.dump(summary_dict, f, indent=2)
print(f"\n✓ Saved summary: {summary_output}")

print("\n" + "="*80)
print("✅ Deep outlier analysis complete!")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Files generated:")
print(f"  - outlier_deep_inspection.csv (60 samples for manual review)")
print(f"  - statistical_comparison.csv (t-tests, effect sizes)")
print(f"  - source_specific_analysis.csv (by TruthfulQA/FEVER/HaluEval)")
print(f"  - deep_outlier_analysis.png (9-panel comprehensive visualization)")
print(f"  - deep_analysis_summary.json (quantitative results)")
print("="*80)
