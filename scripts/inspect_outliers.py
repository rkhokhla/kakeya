#!/usr/bin/env python3
"""
Inspect the 415 outliers from full-scale public dataset analysis.

Goal: Determine if ASV outliers (low r_LZ score) correspond to actual
problematic outputs (structural degeneracy) or false positives.

Manual inspection of top 50 outliers to classify into:
1. True structural degeneracy (loops, repetition, drift, incoherence)
2. False positives (benign outputs with low compressibility)
3. Edge cases (ambiguous, hard to classify)
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

# Load results
results_path = Path("/Users/roman.khokhla/my_stuff/kakeya/results/full_public_dataset_analysis/full_public_dataset_results.csv")
summary_path = Path("/Users/roman.khokhla/my_stuff/kakeya/results/full_public_dataset_analysis/full_analysis_summary.json")

print("Loading full public dataset results...")
df = pd.read_csv(results_path)
print(f"✓ Loaded {len(df)} samples")

# Load summary to get outlier threshold
with open(summary_path) as f:
    summary = json.load(f)
    outlier_threshold = summary['distribution_stats']['threshold_outlier']
    print(f"✓ Outlier threshold: {outlier_threshold:.4f}")

# Load actual text from JSONL files
print("\nLoading actual text from JSONL files...")
text_data = {}

jsonl_files = {
    'truthfulqa': Path("/Users/roman.khokhla/my_stuff/kakeya/data/llm_outputs/truthfulqa_outputs.jsonl"),
    'fever': Path("/Users/roman.khokhla/my_stuff/kakeya/data/llm_outputs/fever_outputs.jsonl"),
    'halueval': Path("/Users/roman.khokhla/my_stuff/kakeya/data/llm_outputs/halueval_outputs.jsonl"),
}

for source, jsonl_path in jsonl_files.items():
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                sample_id = record['id']
                text = record.get('llm_response', '')
                text_data[sample_id] = text
        print(f"✓ Loaded {len([k for k in text_data.keys() if k.startswith(source)])} samples from {source}")
    else:
        print(f"⚠️  File not found: {jsonl_path}")

print(f"✓ Total text samples loaded: {len(text_data)}")

# Identify outliers
outliers = df[df['asv_score'] <= outlier_threshold].copy()
outliers = outliers.sort_values('asv_score')  # Lowest scores first (most anomalous)
print(f"\n✓ Found {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

# Take top 50 for manual inspection
top_outliers = outliers.head(50)

print(f"\n{'='*80}")
print(f"TOP 50 OUTLIERS (sorted by ASV score, lowest = most anomalous)")
print(f"{'='*80}\n")

# Prepare inspection data
inspection_data = []

for idx, row in top_outliers.iterrows():
    sample_id = row.get('sample_id', idx)
    score = row['asv_score']
    source = row.get('source', 'unknown')

    # Load text from text_data dictionary
    text = text_data.get(sample_id, '')

    # Truncate text for display
    text_preview = text[:200] + "..." if len(text) > 200 else text

    print(f"Sample {sample_id} | Score: {score:.4f} | Source: {source}")
    print(f"Text: {text_preview}")

    # Automatic heuristic classification (for initial screening)
    # These are just heuristics - manual review will be final
    repetition_score = 0
    incoherence_score = 0
    drift_score = 0

    # Check for repetition
    words = text.lower().split()
    if len(words) > 5:
        # Count repeated phrases
        for phrase_len in [3, 4, 5]:
            phrases = [' '.join(words[i:i+phrase_len]) for i in range(len(words)-phrase_len+1)]
            unique_phrases = len(set(phrases))
            if unique_phrases < len(phrases) * 0.7:  # >30% repetition
                repetition_score += 1

    # Check for sentence repetition
    sentences = text.split('.')
    if len(sentences) > 2:
        unique_sentences = len(set(s.strip() for s in sentences if s.strip()))
        if unique_sentences < len(sentences) * 0.8:  # >20% repeated sentences
            repetition_score += 2

    # Heuristic classification
    if repetition_score >= 2:
        auto_classification = "LIKELY_REPETITION"
    elif len(text) < 50:
        auto_classification = "TOO_SHORT"
    elif len(set(words)) / len(words) < 0.3:  # Low lexical diversity
        auto_classification = "POSSIBLE_REPETITION"
    else:
        auto_classification = "NEEDS_MANUAL_REVIEW"

    print(f"Auto-classification: {auto_classification}")
    print(f"{'-'*80}\n")

    inspection_data.append({
        'sample_id': sample_id,
        'asv_score': score,
        'source': source,
        'text': text,
        'text_length': len(text),
        'word_count': len(words),
        'unique_word_ratio': len(set(words)) / len(words) if words else 0,
        'auto_classification': auto_classification,
        'manual_classification': '',  # To be filled by human reviewer
        'notes': ''
    })

# Save inspection sheet
inspection_df = pd.DataFrame(inspection_data)
output_path = Path("/Users/roman.khokhla/my_stuff/kakeya/results/full_public_dataset_analysis/outlier_inspection.csv")
inspection_df.to_csv(output_path, index=False)
print(f"✓ Saved inspection sheet: {output_path}")

# Statistics
print(f"\n{'='*80}")
print(f"OUTLIER STATISTICS")
print(f"{'='*80}")
print(f"Total outliers: {len(outliers)}")
print(f"Inspected: {len(top_outliers)}")
print(f"Auto-classifications:")
for classification in inspection_df['auto_classification'].value_counts().items():
    print(f"  - {classification[0]}: {classification[1]} ({classification[1]/len(inspection_df)*100:.1f}%)")

print(f"\nScore distribution of outliers:")
print(f"  - Min: {outliers['asv_score'].min():.4f}")
print(f"  - Q25: {outliers['asv_score'].quantile(0.25):.4f}")
print(f"  - Median: {outliers['asv_score'].median():.4f}")
print(f"  - Q75: {outliers['asv_score'].quantile(0.75):.4f}")
print(f"  - Max: {outliers['asv_score'].max():.4f}")

print(f"\nSource breakdown of outliers:")
for source in outliers.get('source', pd.Series(['unknown']*len(outliers))).value_counts().items():
    print(f"  - {source[0]}: {source[1]} ({source[1]/len(outliers)*100:.1f}%)")

print(f"\n{'='*80}")
print(f"NEXT STEPS")
print(f"{'='*80}")
print(f"1. Open: {output_path}")
print(f"2. Manually review 'text' column for each sample")
print(f"3. Fill 'manual_classification' with one of:")
print(f"   - TRUE_POSITIVE: Structural degeneracy (loops, repetition, drift, incoherence)")
print(f"   - FALSE_POSITIVE: Benign output, no clear structural issues")
print(f"   - EDGE_CASE: Ambiguous, hard to classify")
print(f"4. Add notes explaining the classification")
print(f"5. Re-run analysis after manual classification to compute precision")
