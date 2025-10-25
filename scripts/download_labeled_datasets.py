#!/usr/bin/env python3
"""
Download and process public datasets with binary hallucination labels.

Downloads:
1. HaluBench - HuggingFace datasets
2. BEGIN Benchmark - GitHub
3. RARR - GitHub

Converts all to standard JSONL format with is_hallucination field.
"""

import json
import os
from pathlib import Path
import urllib.request
import zipfile
import tarfile

# Paths
DATA_DIR = Path("/Users/roman.khokhla/my_stuff/kakeya/data")
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
OUTPUT_DIR = DATA_DIR / "llm_outputs"

BENCHMARKS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOADING LABELED HALLUCINATION DATASETS")
print("="*80)

# ============================================================================
# 1. HaluBench (HuggingFace)
# ============================================================================

print("\n[1/3] Downloading HaluBench from HuggingFace...")

try:
    from datasets import load_dataset

    # Try to load HaluBench
    # Note: This might fail if the dataset name is incorrect or requires auth
    print("  → Attempting to load from HuggingFace datasets library...")

    # Common HuggingFace dataset names for hallucination benchmarks
    possible_names = [
        "HaluBench/HaluBench",
        "halubench",
        "potsawee/wiki_bio_gpt3_hallucination",  # Alternative
    ]

    dataset = None
    for name in possible_names:
        try:
            print(f"  → Trying {name}...")
            dataset = load_dataset(name, trust_remote_code=True)
            print(f"  ✓ Loaded {name}")
            break
        except Exception as e:
            print(f"  ✗ {name} not found: {str(e)[:100]}")
            continue

    if dataset:
        # Convert to JSONL
        output_file = OUTPUT_DIR / "halubench_outputs.jsonl"
        samples = []

        # Handle different splits
        for split_name in dataset.keys():
            print(f"  → Processing split: {split_name}")
            split_data = dataset[split_name]

            for idx, sample in enumerate(split_data):
                # HaluBench structure: gpt3_text vs wiki_bio_text with sentence-level annotations
                # annotation field: list of "accurate", "minor_inaccurate", "major_inaccurate"
                # Consider major_inaccurate or minor_inaccurate as hallucinations

                annotations = sample.get('annotation', [])
                is_hallucination = any(
                    ann in ['major_inaccurate', 'minor_inaccurate']
                    for ann in annotations
                ) if annotations else False

                samples.append({
                    'id': f'halubench_{split_name}_{idx}',
                    'source': 'halubench',
                    'prompt': '',  # HaluBench is GPT-3 text vs Wikipedia (no explicit prompt)
                    'llm_response': sample.get('gpt3_text', ''),
                    'is_hallucination': is_hallucination,
                    'hallucination_type': 'factual' if is_hallucination else 'none',
                    'ground_truth': sample.get('wiki_bio_text', ''),
                    'metadata': {k: v for k, v in sample.items() if k not in ['gpt3_text', 'wiki_bio_text', 'annotation']}
                })

        with open(output_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

        print(f"  ✓ Saved {len(samples)} samples to {output_file}")
    else:
        print("  ⚠️  Could not load HaluBench from HuggingFace")
        print("  → Skipping HaluBench (may require manual download)")

except ImportError:
    print("  ✗ datasets library not installed, skipping HaluBench")
except Exception as e:
    print(f"  ✗ Error downloading HaluBench: {e}")

# ============================================================================
# 2. BEGIN Benchmark (GitHub)
# ============================================================================

print("\n[2/3] Downloading BEGIN Benchmark from GitHub...")

begin_url = "https://raw.githubusercontent.com/Libr-AI/BEGIN-benchmark/main/data/begin.jsonl"
begin_output = BENCHMARKS_DIR / "begin.jsonl"

try:
    print(f"  → Downloading from {begin_url}")
    urllib.request.urlretrieve(begin_url, begin_output)

    # Process BEGIN format
    output_file = OUTPUT_DIR / "begin_outputs.jsonl"
    samples = []

    with open(begin_output) as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)

                # BEGIN format: {question, answer, grounded (bool)}
                samples.append({
                    'id': f'begin_{line_num}',
                    'source': 'begin',
                    'prompt': sample.get('question', sample.get('prompt', '')),
                    'llm_response': sample.get('answer', sample.get('response', '')),
                    # grounded=True means NOT hallucinated, so invert
                    'is_hallucination': not sample.get('grounded', True),
                    'ground_truth': sample.get('ground_truth', ''),
                    'metadata': {k: v for k, v in sample.items() if k not in ['question', 'answer', 'grounded']}
                })
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Skipping line {line_num}: invalid JSON")
                continue

    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"  ✓ Downloaded and processed BEGIN")
    print(f"  ✓ Saved {len(samples)} samples to {output_file}")

    # Print hallucination rate
    halluc_rate = sum(s['is_hallucination'] for s in samples) / len(samples) * 100
    print(f"  → Hallucination rate: {halluc_rate:.1f}%")

except urllib.error.HTTPError as e:
    print(f"  ✗ HTTP Error {e.code}: {e.reason}")
    print(f"  → URL may have changed. Check: https://github.com/Libr-AI/BEGIN-benchmark")
except Exception as e:
    print(f"  ✗ Error downloading BEGIN: {e}")

# ============================================================================
# 3. RARR (GitHub)
# ============================================================================

print("\n[3/3] Downloading RARR from GitHub...")

rarr_url = "https://raw.githubusercontent.com/anthonywchen/RARR/main/data/rarr.jsonl"
rarr_output = BENCHMARKS_DIR / "rarr.jsonl"

try:
    print(f"  → Downloading from {rarr_url}")
    urllib.request.urlretrieve(rarr_url, rarr_output)

    # Process RARR format
    output_file = OUTPUT_DIR / "rarr_outputs.jsonl"
    samples = []

    with open(rarr_output) as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)

                # RARR format: {question, answer, has_hallucination (bool), revised_answer}
                samples.append({
                    'id': f'rarr_{line_num}',
                    'source': 'rarr',
                    'prompt': sample.get('question', sample.get('prompt', '')),
                    'llm_response': sample.get('answer', sample.get('response', '')),
                    'is_hallucination': sample.get('has_hallucination', False),
                    'revised_answer': sample.get('revised_answer', ''),
                    'metadata': {k: v for k, v in sample.items() if k not in ['question', 'answer', 'has_hallucination']}
                })
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Skipping line {line_num}: invalid JSON")
                continue

    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"  ✓ Downloaded and processed RARR")
    print(f"  ✓ Saved {len(samples)} samples to {output_file}")

    # Print hallucination rate
    halluc_rate = sum(s['is_hallucination'] for s in samples) / len(samples) * 100
    print(f"  → Hallucination rate: {halluc_rate:.1f}%")

except urllib.error.HTTPError as e:
    print(f"  ✗ HTTP Error {e.code}: {e.reason}")
    print(f"  → URL may have changed. Check: https://github.com/anthonywchen/RARR")
except Exception as e:
    print(f"  ✗ Error downloading RARR: {e}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Check what we got
datasets_found = []
for name in ['halubench', 'begin', 'rarr']:
    output_file = OUTPUT_DIR / f"{name}_outputs.jsonl"
    if output_file.exists():
        with open(output_file) as f:
            count = sum(1 for _ in f)
        datasets_found.append((name, count))

if datasets_found:
    print("\n✓ Successfully processed datasets:")
    for name, count in datasets_found:
        print(f"  - {name}: {count:,} samples")

    total_samples = sum(count for _, count in datasets_found)
    print(f"\n→ Total: {total_samples:,} samples with binary hallucination labels")
    print(f"→ Output directory: {OUTPUT_DIR}")

    print("\nNext steps:")
    print("1. Run: python3 scripts/analyze_ensemble_verification.py")
    print("2. Train ensemble models on labeled data")
    print("3. Update whitepaper with results")
else:
    print("\n⚠️  No datasets successfully downloaded")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Verify GitHub URLs are still valid")
    print("3. For HaluBench, try manual download from HuggingFace")
    print("4. Some datasets may require authentication")

print("="*80)
