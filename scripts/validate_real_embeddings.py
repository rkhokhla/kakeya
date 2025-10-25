#!/usr/bin/env python3
"""
Real Embedding Validation for ASV (Option B - No Human Involvement)

Generates 100 real LLM outputs with structural degeneracies and validates
ASV signals using actual embeddings (not synthetic).

Uses:
- GPT-3.5-turbo to generate degenerate outputs
- GPT-2 embeddings (actual token embeddings, not synthetic)
- ASV signals: D̂, coh★, r_LZ
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import time
import json
from tqdm import tqdm
import os

# OpenAI for generation
from openai import OpenAI

# Transformers for real embeddings
import torch
from transformers import GPT2Tokenizer, GPT2Model

# Set up paths
RESULTS_DIR = Path("results/real_embeddings")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Cost tracking
TOTAL_COST = 0.0


class RealLLMSample:
    """Real LLM output with actual embeddings."""
    def __init__(self, sample_id: str, prompt: str, text: str,
                 is_degenerate: bool, degeneracy_type: str):
        self.sample_id = sample_id
        self.prompt = prompt
        self.text = text
        self.is_degenerate = is_degenerate
        self.degeneracy_type = degeneracy_type
        self.embeddings = None  # Will be computed from real model


def generate_real_degenerate_samples(client: OpenAI, n_samples: int = 100) -> List[RealLLMSample]:
    """Generate real LLM outputs with structural degeneracies."""
    global TOTAL_COST

    samples = []

    # Prompts designed to elicit degeneracies
    degenerate_prompts = {
        "repetition_loop": [
            "Repeat the phrase 'the quick brown fox' exactly 20 times.",
            "Say 'hello world' over and over again at least 15 times.",
            "Write the word 'test' repeatedly for an entire paragraph.",
        ],
        "semantic_drift": [
            "Start by describing a car, then suddenly switch to talking about cooking, then switch to discussing space exploration.",
            "Begin explaining mathematics, then abruptly change to describing weather, then jump to discussing movies.",
            "Start with a story about a dog, then randomly switch to explaining computers, then shift to talking about music.",
        ],
        "incoherent": [
            "Write a paragraph where each sentence contradicts the previous one.",
            "Generate text where nothing makes logical sense and ideas are completely disconnected.",
            "Create text with random topic jumps where no sentence follows from the previous one.",
        ],
    }

    # Normal prompts
    normal_prompts = [
        "Explain the concept of photosynthesis in simple terms.",
        "Describe the water cycle step by step.",
        "What is the process of cellular respiration?",
    ]

    print("\n[1] Generating real LLM outputs with structural degeneracies...")
    print(f"Target: {n_samples} samples (25 each: loops, drift, incoherent, normal)")

    samples_per_type = n_samples // 4

    # Generate degenerate samples
    for deg_type, prompts in degenerate_prompts.items():
        for i in range(samples_per_type):
            prompt = prompts[i % len(prompts)]

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,  # Higher temperature for more varied degeneracies
                    max_tokens=200,
                )

                text = response.choices[0].message.content

                # Cost tracking
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost = (input_tokens / 1000 * 0.0015) + (output_tokens / 1000 * 0.002)
                TOTAL_COST += cost

                sample = RealLLMSample(
                    sample_id=f"{deg_type}_{i:03d}",
                    prompt=prompt,
                    text=text,
                    is_degenerate=True,
                    degeneracy_type=deg_type,
                )
                samples.append(sample)

            except Exception as e:
                print(f"Error generating {deg_type} sample {i}: {e}")
                continue

    # Generate normal samples
    for i in range(samples_per_type):
        prompt = normal_prompts[i % len(normal_prompts)]

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
            )

            text = response.choices[0].message.content

            # Cost tracking
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens / 1000 * 0.0015) + (output_tokens / 1000 * 0.002)
            TOTAL_COST += cost

            sample = RealLLMSample(
                sample_id=f"normal_{i:03d}",
                prompt=prompt,
                text=text,
                is_degenerate=False,
                degeneracy_type="normal",
            )
            samples.append(sample)

        except Exception as e:
            print(f"Error generating normal sample {i}: {e}")
            continue

    print(f"✓ Generated {len(samples)} real LLM outputs")
    print(f"✓ Total generation cost: ${TOTAL_COST:.3f}")

    return samples


def extract_real_embeddings(samples: List[RealLLMSample]) -> List[RealLLMSample]:
    """Extract real GPT-2 embeddings from LLM outputs."""
    print("\n[2] Extracting real GPT-2 embeddings from outputs...")

    # Load GPT-2 model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token

    for sample in tqdm(samples, desc="Extracting embeddings"):
        # Tokenize
        inputs = tokenizer(
            sample.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use last layer hidden states as embeddings
            embeddings = outputs.last_hidden_state[0].numpy()  # (seq_len, 768)

        sample.embeddings = embeddings

    print(f"✓ Extracted embeddings for {len(samples)} samples")
    print(f"  Embedding shape: {samples[0].embeddings.shape} (seq_len, 768)")

    return samples


def compute_asv_signals(sample: RealLLMSample) -> Dict:
    """Compute ASV signals (D̂, coh★, r_LZ) on real embeddings."""
    embeddings = sample.embeddings

    # D̂ (fractal dimension)
    D_hat = compute_fractal_dim(embeddings)

    # coh★ (directional coherence)
    coh_star = compute_coherence(embeddings)

    # r_LZ (compressibility)
    r_LZ = compute_compressibility(embeddings)

    # ASV score (weighted ensemble)
    score = compute_asv_score(D_hat, coh_star, r_LZ)

    return {
        "sample_id": sample.sample_id,
        "degeneracy_type": sample.degeneracy_type,
        "is_degenerate": sample.is_degenerate,
        "D_hat": D_hat,
        "coh_star": coh_star,
        "r_LZ": r_LZ,
        "asv_score": score,
        "text_length": len(sample.text),
        "num_tokens": len(embeddings),
    }


def compute_fractal_dim(embeddings: np.ndarray) -> float:
    """Compute fractal dimension D̂ via Theil-Sen."""
    scales = [2, 4, 8, 16, 32]
    counts = []

    for scale in scales:
        grid_cells = set()
        for emb in embeddings:
            coords = emb[:3]  # Project to 3D
            cell = tuple((coords * scale).astype(int))
            grid_cells.add(cell)
        counts.append(len(grid_cells))

    # Theil-Sen: median of pairwise slopes
    log_scales = np.log2(scales)
    log_counts = np.log2(counts)

    slopes = []
    for i in range(len(scales)):
        for j in range(i+1, len(scales)):
            slope = (log_counts[j] - log_counts[i]) / (log_scales[j] - log_scales[i])
            slopes.append(slope)

    D_hat = np.median(slopes) if slopes else 1.5
    return max(0.5, min(3.0, D_hat))


def compute_coherence(embeddings: np.ndarray) -> float:
    """Compute directional coherence coh★."""
    n_directions = 100
    n_bins = 20

    max_concentration = 0.0
    for _ in range(n_directions):
        # Random direction
        direction = np.random.randn(embeddings.shape[1])
        direction /= np.linalg.norm(direction)

        # Project embeddings
        projections = embeddings @ direction

        # Histogram concentration
        hist, _ = np.histogram(projections, bins=n_bins)
        concentration = hist.max() / len(embeddings)
        max_concentration = max(max_concentration, concentration)

    return max_concentration


def compute_compressibility(embeddings: np.ndarray) -> float:
    """Compute compressibility r_LZ via product quantization."""
    import zlib

    n_subspaces = 8
    codebook_bits = 8
    d_sub = embeddings.shape[1] // n_subspaces

    # Product quantization
    codes = []
    for i in range(n_subspaces):
        sub_emb = embeddings[:, i*d_sub:(i+1)*d_sub]
        n_clusters = 2 ** codebook_bits

        # Simple quantization: assign to nearest random centroid
        centroids = np.random.randn(n_clusters, d_sub)
        distances = np.linalg.norm(sub_emb[:, None, :] - centroids[None, :, :], axis=2)
        cluster_ids = np.argmin(distances, axis=1)
        codes.append(cluster_ids.astype(np.uint8))

    # LZ compression
    code_bytes = np.concatenate(codes).tobytes()
    compressed = zlib.compress(code_bytes, level=6)
    r_LZ = len(compressed) / len(code_bytes)

    return r_LZ


def compute_asv_score(D_hat: float, coh_star: float, r_LZ: float) -> float:
    """Compute ASV ensemble score (higher = more normal/acceptable)."""
    # Learned weights for degeneracy detection
    w_D = 0.15
    w_coh = 0.15
    w_r = 0.60  # r_LZ is dominant for degeneracy
    w_perp = 0.10  # Perplexity not available here

    # Normalize signals
    # D̂: sweet spot 1.5-2.5
    if 1.5 <= D_hat <= 2.5:
        D_norm = 1.0
    elif D_hat < 1.5:
        D_norm = D_hat / 1.5
    else:
        D_norm = max(0.0, 1.0 - (D_hat - 2.5) / 0.5)

    # coh★: moderate is good (0.3-0.6)
    if 0.3 <= coh_star <= 0.6:
        coh_norm = 1.0
    elif coh_star < 0.3:
        coh_norm = coh_star / 0.3
    else:
        coh_norm = max(0.0, 1.0 - (coh_star - 0.6) / 0.4)

    # r_LZ: moderate is good (0.4-0.7)
    if 0.4 <= r_LZ <= 0.7:
        r_norm = 1.0
    elif r_LZ < 0.4:
        r_norm = max(0.0, r_LZ / 0.4)
    else:
        r_norm = max(0.0, 1.0 - (r_LZ - 0.7) / 0.3)

    score = w_D * D_norm + w_coh * coh_norm + w_r * r_norm
    return max(0.0, min(1.0, score))


def evaluate_asv_real_embeddings(samples: List[RealLLMSample]) -> pd.DataFrame:
    """Evaluate ASV on real embeddings."""
    print("\n[3] Computing ASV signals on real embeddings...")

    results = []
    for sample in tqdm(samples, desc="Computing ASV signals"):
        result = compute_asv_signals(sample)
        results.append(result)

    df = pd.DataFrame(results)

    # Compute metrics
    print("\n[4] Computing metrics...")

    y_true = df['is_degenerate'].values
    y_scores = 1.0 - df['asv_score'].values  # Higher score for degenerate

    # AUROC
    from sklearn.metrics import roc_auc_score, roc_curve
    auroc = roc_auc_score(y_true, y_scores)

    # Optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]

    # Predictions at optimal threshold
    y_pred = y_scores >= optimal_threshold

    # Confusion matrix
    tp = ((y_pred == True) & (y_true == True)).sum()
    tn = ((y_pred == False) & (y_true == False)).sum()
    fp = ((y_pred == True) & (y_true == False)).sum()
    fn = ((y_pred == False) & (y_true == True)).sum()

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "n_samples": int(len(samples)),
        "n_degenerate": int(y_true.sum()),
        "n_normal": int((~y_true).sum()),
        "auroc": float(auroc),
        "optimal_threshold": float(optimal_threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    # Per-degeneracy-type breakdown
    type_metrics = []
    for deg_type in df['degeneracy_type'].unique():
        type_df = df[df['degeneracy_type'] == deg_type]
        type_y_true = type_df['is_degenerate'].values
        type_y_scores = 1.0 - type_df['asv_score'].values

        if len(np.unique(type_y_true)) > 1:  # Only if both classes present
            type_auroc = roc_auc_score(type_y_true, type_y_scores)
        else:
            type_auroc = np.nan

        type_metrics.append({
            "degeneracy_type": deg_type,
            "n_samples": len(type_df),
            "mean_D_hat": type_df['D_hat'].mean(),
            "mean_coh_star": type_df['coh_star'].mean(),
            "mean_r_LZ": type_df['r_LZ'].mean(),
            "mean_asv_score": type_df['asv_score'].mean(),
            "auroc": type_auroc,
        })

    type_metrics_df = pd.DataFrame(type_metrics)

    return df, metrics, type_metrics_df


def main():
    """Main execution."""
    print("="*80)
    print("Real Embedding Validation for ASV (Option B - No Human)")
    print("="*80)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not set")
        return

    client = OpenAI(api_key=api_key)

    # Generate real LLM outputs
    samples = generate_real_degenerate_samples(client, n_samples=100)

    # Extract real embeddings
    samples = extract_real_embeddings(samples)

    # Evaluate ASV
    results_df, metrics, type_metrics_df = evaluate_asv_real_embeddings(samples)

    # Save results
    print("\n[5] Saving results...")

    results_df.to_csv(RESULTS_DIR / "real_embeddings_results.csv", index=False)
    type_metrics_df.to_csv(RESULTS_DIR / "real_embeddings_type_metrics.csv", index=False)

    with open(RESULTS_DIR / "real_embeddings_summary.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save samples (without embeddings for size)
    samples_data = []
    for sample in samples:
        samples_data.append({
            "sample_id": sample.sample_id,
            "prompt": sample.prompt,
            "text": sample.text,
            "is_degenerate": sample.is_degenerate,
            "degeneracy_type": sample.degeneracy_type,
        })

    with open(RESULTS_DIR / "real_embeddings_samples.json", 'w') as f:
        json.dump(samples_data, f, indent=2)

    print(f"✓ Saved results to {RESULTS_DIR}/")

    # Print summary
    print("\n" + "="*80)
    print("REAL EMBEDDING VALIDATION RESULTS")
    print("="*80)
    print(f"\nSamples: {metrics['n_samples']} ({metrics['n_degenerate']} degenerate, {metrics['n_normal']} normal)")
    print(f"\nOverall Metrics:")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['tp']}  FP: {metrics['fp']}")
    print(f"  FN: {metrics['fn']}  TN: {metrics['tn']}")

    print(f"\nPer-Type Metrics:")
    print(type_metrics_df.to_string(index=False))

    print(f"\n💰 Total Cost: ${TOTAL_COST:.3f}")
    print(f"📁 Results: {RESULTS_DIR}/")

    print("\n" + "="*80)
    print("✓ Real Embedding Validation Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
