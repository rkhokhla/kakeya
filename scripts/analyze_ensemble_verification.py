#!/usr/bin/env python3
"""
Ensemble Verification Analysis: Combining Geometric Signals with Perplexity

Research Question: Can we improve hallucination detection beyond perplexity alone
by adding geometric signals (r_LZ, lexical diversity, repetition) that capture
different failure modes?

Hypothesis: Hallucinations are multi-modal:
1. Factual errors → perplexity wins
2. Structural pathology → r_LZ/repetition (if present)
3. Quality markers → lexical diversity (inverse: high diversity = likely correct)

Data: 8,290 real GPT-4 outputs with ground-truth hallucination labels
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
FILTERED_RESULTS = Path("/Users/roman.khokhla/my_stuff/kakeya/results/corrected_public_dataset_analysis/filtered_public_dataset_results.csv")
DEEP_ANALYSIS = Path("/Users/roman.khokhla/my_stuff/kakeya/results/deep_outlier_analysis/outlier_deep_inspection.csv")
JSONL_FILES = {
    'truthfulqa': Path("/Users/roman.khokhla/my_stuff/kakeya/data/llm_outputs/truthfulqa_outputs.jsonl"),
    'fever': Path("/Users/roman.khokhla/my_stuff/kakeya/data/llm_outputs/fever_outputs.jsonl"),
    'halueval': Path("/Users/roman.khokhla/my_stuff/kakeya/data/llm_outputs/halueval_outputs.jsonl"),
}
OUTPUT_DIR = Path("/Users/roman.khokhla/my_stuff/kakeya/results/ensemble_verification/")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("ENSEMBLE VERIFICATION ANALYSIS: Combining Geometric Signals with Perplexity")
print("="*80)
print("\nGoal: Test whether ensemble approaches (perplexity + geometric signals)")
print("outperform perplexity alone for hallucination detection on 8,290 GPT-4 outputs.")
print("="*80)

# Load filtered results (n >= 10 tokens)
print(f"\n[1/10] Loading filtered results...")
df = pd.read_csv(FILTERED_RESULTS)
print(f"✓ Loaded {len(df)} samples (filtered, n >= 10 tokens)")

# Load text data with ground truth labels
print(f"\n[2/10] Loading ground truth labels...")
text_data = {}
for source, jsonl_path in JSONL_FILES.items():
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                sample_id = record['id']
                text = record.get('llm_response', '')
                is_hallucination = record.get('hallucination', False) or record.get('is_hallucination', False)
                text_data[sample_id] = {
                    'text': text,
                    'source': source,
                    'is_hallucination': is_hallucination
                }
        print(f"✓ Loaded {len([k for k in text_data.keys() if text_data[k]['source'] == source])} from {source}")

# Merge labels
df['text'] = df.apply(lambda row: text_data.get(row.get('sample_id', ''), {}).get('text', ''), axis=1)
df['is_hallucination'] = df.apply(lambda row: text_data.get(row.get('sample_id', ''), {}).get('is_hallucination', None), axis=1)
df['source'] = df.apply(lambda row: text_data.get(row.get('sample_id', ''), {}).get('source', 'unknown'), axis=1)

# Drop samples without labels
df_labeled = df[df['is_hallucination'].notna()].copy()
print(f"✓ {len(df_labeled)} samples have ground truth labels ({len(df_labeled)/len(df)*100:.1f}%)")
print(f"  - Hallucinations: {df_labeled['is_hallucination'].sum()} ({df_labeled['is_hallucination'].sum()/len(df_labeled)*100:.1f}%)")
print(f"  - Correct: {(~df_labeled['is_hallucination']).sum()} ({(~df_labeled['is_hallucination']).sum()/len(df_labeled)*100:.1f}%)")

# Compute additional features
print(f"\n[3/10] Computing additional features...")

def compute_lexical_diversity(text):
    """Type-token ratio"""
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)

def compute_sentence_repetition(text):
    """Sentence repetition rate"""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) < 2:
        return 0.0
    from collections import Counter
    sentence_counts = Counter(sentences)
    most_common_count = sentence_counts.most_common(1)[0][1] if sentence_counts else 0
    return most_common_count / len(sentences) if len(sentences) > 0 else 0

def compute_perplexity_proxy(text):
    """Proxy: character-level entropy (higher = more uncertain)"""
    from collections import Counter
    if len(text) == 0:
        return 0.0
    char_counts = Counter(text.lower())
    total = sum(char_counts.values())
    entropy = -sum((count/total) * np.log2(count/total) for count in char_counts.values())
    return entropy

df_labeled['lexical_diversity'] = df_labeled['text'].apply(compute_lexical_diversity)
df_labeled['sentence_repetition'] = df_labeled['text'].apply(compute_sentence_repetition)
df_labeled['perplexity_proxy'] = df_labeled['text'].apply(compute_perplexity_proxy)
df_labeled['length_tokens'] = df_labeled['text'].apply(lambda t: len(t.split()))

print(f"✓ Computed features:")
print(f"  - r_LZ (compressibility): mean={df_labeled['asv_score'].mean():.3f}, std={df_labeled['asv_score'].std():.3f}")
print(f"  - Lexical diversity: mean={df_labeled['lexical_diversity'].mean():.3f}, std={df_labeled['lexical_diversity'].std():.3f}")
print(f"  - Sentence repetition: mean={df_labeled['sentence_repetition'].mean():.3f}, std={df_labeled['sentence_repetition'].std():.3f}")
print(f"  - Perplexity proxy: mean={df_labeled['perplexity_proxy'].mean():.3f}, std={df_labeled['perplexity_proxy'].std():.3f}")

# Split train/test (70/30)
print(f"\n[4/10] Splitting train/test (70/30)...")
train_df, test_df = train_test_split(df_labeled, test_size=0.3, random_state=42, stratify=df_labeled['is_hallucination'])
print(f"✓ Train: {len(train_df)} samples ({train_df['is_hallucination'].sum()} hallucinations)")
print(f"✓ Test: {len(test_df)} samples ({test_df['is_hallucination'].sum()} hallucinations)")

# Define feature sets
feature_sets = {
    'Perplexity (baseline)': ['perplexity_proxy'],
    'r_LZ (compressibility)': ['asv_score'],
    'Lexical diversity': ['lexical_diversity'],
    'Perplexity + r_LZ': ['perplexity_proxy', 'asv_score'],
    'Perplexity + Lexical diversity': ['perplexity_proxy', 'lexical_diversity'],
    'Perplexity + Repetition': ['perplexity_proxy', 'sentence_repetition'],
    'Perplexity + Length': ['perplexity_proxy', 'length_tokens'],
    'Full ensemble': ['perplexity_proxy', 'asv_score', 'lexical_diversity', 'sentence_repetition', 'length_tokens'],
}

# Train models
print(f"\n[5/10] Training logistic regression models...")
results = {}

for name, features in feature_sets.items():
    X_train = train_df[features].values
    y_train = train_df['is_hallucination'].values.astype(int)
    X_test = test_df[features].values
    y_test = test_df['is_hallucination'].values.astype(int)

    # Train logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    auroc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    results[name] = {
        'features': features,
        'auroc': auroc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'model': model
    }

    print(f"✓ {name}: AUROC={auroc:.3f}, Acc={accuracy:.3f}, F1={f1:.3f}")

# McNemar's test: compare perplexity baseline to ensembles
print(f"\n[6/10] Statistical significance tests (McNemar's)...")
baseline_name = 'Perplexity (baseline)'
baseline_y_pred = results[baseline_name]['y_pred']
y_test = results[baseline_name]['y_test']

mcnemar_results = []
for name, res in results.items():
    if name == baseline_name:
        continue

    # Contingency table
    y_pred = res['y_pred']
    b = ((baseline_y_pred == y_test) & (y_pred != y_test)).sum()  # Baseline correct, ensemble wrong
    c = ((baseline_y_pred != y_test) & (y_pred == y_test)).sum()  # Baseline wrong, ensemble correct

    if b + c > 0:
        chi_squared = (abs(b - c) - 1) ** 2 / (b + c)  # Continuity correction
        p_value = 1 - stats.chi2.cdf(chi_squared, df=1)
    else:
        chi_squared, p_value = 0, 1.0

    mcnemar_results.append({
        'comparison': f'{baseline_name} vs {name}',
        'b': b,
        'c': c,
        'chi_squared': chi_squared,
        'p_value': p_value,
        'significant': p_value < 0.05
    })

    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
    print(f"  {name}: χ²={chi_squared:.3f}, p={p_value:.4f} {sig_marker} (b={b}, c={c})")

# Bootstrap confidence intervals
print(f"\n[7/10] Computing bootstrap confidence intervals (1000 resamples)...")
np.random.seed(42)
n_bootstrap = 1000

bootstrap_aurocs = {name: [] for name in results.keys()}
for name, res in results.items():
    y_test = res['y_test']
    y_proba = res['y_proba']

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
        y_test_boot = y_test[indices]
        y_proba_boot = y_proba[indices]

        if len(np.unique(y_test_boot)) > 1:  # Need both classes
            auroc_boot = roc_auc_score(y_test_boot, y_proba_boot)
            bootstrap_aurocs[name].append(auroc_boot)

for name, aurocs in bootstrap_aurocs.items():
    if len(aurocs) > 0:
        ci_lower = np.percentile(aurocs, 2.5)
        ci_upper = np.percentile(aurocs, 97.5)
        results[name]['auroc_ci'] = (ci_lower, ci_upper)
        print(f"✓ {name}: AUROC {results[name]['auroc']:.3f} [95% CI: {ci_lower:.3f}-{ci_upper:.3f}]")

# Source-specific analysis
print(f"\n[8/10] Source-specific performance...")
source_results = {}
for source in ['truthfulqa', 'fever', 'halueval']:
    test_source = test_df[test_df['source'] == source]
    if len(test_source) == 0:
        continue

    print(f"\n{source.upper()}:")
    for name, features in feature_sets.items():
        if name not in ['Perplexity (baseline)', 'Perplexity + Lexical diversity', 'Full ensemble']:
            continue  # Only report key methods

        X_test_source = test_source[features].values
        y_test_source = test_source['is_hallucination'].values.astype(int)

        model = results[name]['model']
        y_proba_source = model.predict_proba(X_test_source)[:, 1]

        if len(np.unique(y_test_source)) > 1:
            auroc_source = roc_auc_score(y_test_source, y_proba_source)
            print(f"  {name}: AUROC={auroc_source:.3f} (n={len(test_source)})")

# Save results summary
print(f"\n[9/10] Saving results...")
summary = {
    'n_train': len(train_df),
    'n_test': len(test_df),
    'hallucination_rate_train': float(train_df['is_hallucination'].sum() / len(train_df)),
    'hallucination_rate_test': float(test_df['is_hallucination'].sum() / len(test_df)),
    'models': {name: {
        'features': res['features'],
        'auroc': float(res['auroc']),
        'auroc_ci': [float(res['auroc_ci'][0]), float(res['auroc_ci'][1])] if 'auroc_ci' in res else None,
        'accuracy': float(res['accuracy']),
        'precision': float(res['precision']),
        'recall': float(res['recall']),
        'f1': float(res['f1'])
    } for name, res in results.items()},
    'mcnemar_tests': mcnemar_results
}

summary_output = OUTPUT_DIR / "ensemble_verification_summary.json"
with open(summary_output, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Saved summary: {summary_output}")

# Generate visualizations
print(f"\n[10/10] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Ensemble Verification: Combining Geometric Signals with Perplexity', fontsize=16, fontweight='bold')

# Plot 1: AUROC comparison with error bars
ax = axes[0, 0]
names = list(results.keys())
aurocs = [results[name]['auroc'] for name in names]
ci_lowers = [results[name]['auroc_ci'][0] if 'auroc_ci' in results[name] else results[name]['auroc'] for name in names]
ci_uppers = [results[name]['auroc_ci'][1] if 'auroc_ci' in results[name] else results[name]['auroc'] for name in names]
errors = [[aurocs[i] - ci_lowers[i] for i in range(len(names))],
          [ci_uppers[i] - aurocs[i] for i in range(len(names))]]

x_pos = np.arange(len(names))
bars = ax.barh(x_pos, aurocs, xerr=errors, capsize=5, alpha=0.7, color='steelblue')
ax.set_yticks(x_pos)
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('AUROC')
ax.set_title('AUROC Comparison (with 95% CI)')
ax.axvline(results[baseline_name]['auroc'], color='red', linestyle='--', label='Baseline')
ax.legend()
ax.grid(alpha=0.3, axis='x')

# Plot 2: Accuracy, Precision, Recall comparison
ax = axes[0, 1]
metrics_data = []
for name in names:
    metrics_data.append([
        results[name]['accuracy'],
        results[name]['precision'],
        results[name]['recall']
    ])

x = np.arange(len(['Accuracy', 'Precision', 'Recall']))
width = 0.15
for i, name in enumerate(names):
    offset = width * (i - len(names)/2)
    ax.bar(x + offset, metrics_data[i], width, label=name if i < 4 else None, alpha=0.7)

ax.set_ylabel('Score')
ax.set_title('Classification Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall'])
ax.legend(fontsize=7, loc='lower right')
ax.grid(alpha=0.3, axis='y')

# Plot 3: Feature importance (from best model)
ax = axes[1, 0]
best_model_name = max(results.keys(), key=lambda k: results[k]['auroc'])
best_model = results[best_model_name]['model']
best_features = results[best_model_name]['features']

if hasattr(best_model, 'coef_'):
    coefs = best_model.coef_[0]
    ax.barh(best_features, coefs, alpha=0.7, color='coral')
    ax.set_xlabel('Logistic Regression Coefficient')
    ax.set_title(f'Feature Importance: {best_model_name}')
    ax.grid(alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'No feature importance available', ha='center', va='center', transform=ax.transAxes)

# Plot 4: McNemar's test results
ax = axes[1, 1]
if len(mcnemar_results) > 0:
    comparisons = [r['comparison'].replace(baseline_name + ' vs ', '') for r in mcnemar_results]
    p_values = [r['p_value'] for r in mcnemar_results]
    colors = ['green' if p < 0.05 else 'gray' for p in p_values]

    ax.barh(comparisons, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
    ax.axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    ax.set_xlabel('-log10(p-value)')
    ax.set_title('Statistical Significance vs Baseline (McNemar\'s Test)')
    ax.legend()
    ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
fig_output = OUTPUT_DIR / "ensemble_verification_analysis.png"
plt.savefig(fig_output, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: {fig_output}")
plt.close()

# Summary report
print("\n" + "="*80)
print("SUMMARY: Ensemble Verification Results")
print("="*80)
print(f"\n1. BEST PERFORMING MODEL:")
print(f"   {best_model_name}: AUROC={results[best_model_name]['auroc']:.3f}, F1={results[best_model_name]['f1']:.3f}")

print(f"\n2. IMPROVEMENT OVER BASELINE:")
baseline_auroc = results[baseline_name]['auroc']
for name, res in results.items():
    if name == baseline_name:
        continue
    improvement = res['auroc'] - baseline_auroc
    print(f"   {name}: +{improvement:.3f} AUROC ({improvement/baseline_auroc*100:+.1f}%)")

print(f"\n3. STATISTICALLY SIGNIFICANT IMPROVEMENTS (p < 0.05):")
significant_improvements = [r for r in mcnemar_results if r['significant']]
if significant_improvements:
    for r in significant_improvements:
        print(f"   {r['comparison']}: χ²={r['chi_squared']:.3f}, p={r['p_value']:.4f}")
else:
    print("   None found")

print(f"\n4. KEY FINDINGS:")
print(f"   - Baseline (perplexity): AUROC {results[baseline_name]['auroc']:.3f}")
print(f"   - Best ensemble: AUROC {results[best_model_name]['auroc']:.3f}")
print(f"   - Improvement: {(results[best_model_name]['auroc'] - baseline_auroc):.3f} AUROC")

print("\n" + "="*80)
print(f"✅ Analysis complete! Output directory: {OUTPUT_DIR}")
print("="*80)
