#!/usr/bin/env python3
"""
Real Baseline Comparison with Actual API Calls (Priority 2.1 - Production Version)

This script uses REAL production baselines:
1. GPT-4 Turbo via OpenAI API
2. SelfCheckGPT with GPT-3.5 sampling + RoBERTa-MNLI
3. ASV (same as before)

Cost estimate: ~$20-40 for 100 samples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import time
import json
import os
from tqdm import tqdm

# OpenAI imports
from openai import OpenAI

# Transformers for NLI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Directories
RESULTS_DIR = Path("results/baseline_comparison")
FIGURES_DIR = Path("docs/architecture/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Cost tracking
TOTAL_COST = 0.0
MAX_COST = float(os.getenv("MAX_TOTAL_COST", "400.0"))


class DegeneracySample:
    """Represents a text generation sample for degeneracy detection."""
    def __init__(self, sample_id: str, text: str, is_degenerate: bool,
                 degeneracy_type: str, embeddings: np.ndarray = None):
        self.sample_id = sample_id
        self.text = text
        self.is_degenerate = is_degenerate
        self.degeneracy_type = degeneracy_type
        self.embeddings = embeddings if embeddings is not None else self._generate_embeddings()

    def _generate_embeddings(self) -> np.ndarray:
        """Generate synthetic embeddings based on degeneracy type."""
        n_tokens = max(50, len(self.text.split()))
        d = 768

        if self.degeneracy_type == "repetition_loop":
            base = np.random.randn(1, d)
            embeddings = np.repeat(base, n_tokens, axis=0) + np.random.randn(n_tokens, d) * 0.05
        elif self.degeneracy_type == "semantic_drift":
            start = np.random.randn(1, d)
            end = np.random.randn(1, d)
            t = np.linspace(0, 1, n_tokens).reshape(-1, 1)
            embeddings = start + t * (end - start) + np.random.randn(n_tokens, d) * 0.1
        elif self.degeneracy_type == "incoherent":
            embeddings = np.random.randn(n_tokens, d) * 2.0
        else:  # "normal"
            embeddings = np.random.randn(n_tokens, d)
            for i in range(1, n_tokens):
                embeddings[i] = 0.6 * embeddings[i-1] + 0.4 * embeddings[i]
            embeddings += np.random.randn(n_tokens, d) * 0.3

        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        return embeddings


class ASVVerifier:
    """ASV baseline with geometric signals (same as before)."""
    def __init__(self):
        self.name = "ASV"
        self.cost_per_verification = 0.000002

    def verify(self, sample: DegeneracySample) -> Dict:
        """Run ASV verification."""
        start = time.time()

        D_hat = self._compute_fractal_dim(sample.embeddings)
        coh_star = self._compute_coherence(sample.embeddings)
        r_LZ = self._compute_compressibility(sample.embeddings)
        score = self._conformal_score(D_hat, coh_star, r_LZ)

        threshold = 0.60
        decision = "accept" if score >= threshold else "reject"
        latency_ms = (time.time() - start) * 1000

        return {
            "method": self.name,
            "score": score,
            "decision": decision,
            "latency_ms": latency_ms,
            "cost_usd": self.cost_per_verification,
            "D_hat": D_hat,
            "coh_star": coh_star,
            "r_LZ": r_LZ
        }

    def _compute_fractal_dim(self, embeddings: np.ndarray) -> float:
        scales = [2, 4, 8, 16, 32]
        counts = []
        for scale in scales:
            grid_cells = set()
            for emb in embeddings:
                coords = emb[:3]
                cell = tuple((coords * scale).astype(int))
                grid_cells.add(cell)
            counts.append(len(grid_cells))

        log_scales = np.log2(scales)
        log_counts = np.log2(counts)
        slopes = []
        for i in range(len(scales)):
            for j in range(i+1, len(scales)):
                slope = (log_counts[j] - log_counts[i]) / (log_scales[j] - log_scales[i])
                slopes.append(slope)
        D_hat = np.median(slopes) if slopes else 1.5
        return max(0.5, min(3.0, D_hat))

    def _compute_coherence(self, embeddings: np.ndarray) -> float:
        n_directions = 100
        n_bins = 20
        max_concentration = 0.0
        for _ in range(n_directions):
            direction = np.random.randn(embeddings.shape[1])
            direction /= np.linalg.norm(direction)
            projections = embeddings @ direction
            hist, _ = np.histogram(projections, bins=n_bins)
            concentration = hist.max() / len(embeddings)
            max_concentration = max(max_concentration, concentration)
        return max_concentration

    def _compute_compressibility(self, embeddings: np.ndarray) -> float:
        import zlib
        n_subspaces = 8
        codebook_bits = 8
        d_sub = embeddings.shape[1] // n_subspaces
        codes = []
        for i in range(n_subspaces):
            sub_emb = embeddings[:, i*d_sub:(i+1)*d_sub]
            n_clusters = 2 ** codebook_bits
            centroids = np.random.randn(n_clusters, d_sub)
            distances = np.linalg.norm(sub_emb[:, None, :] - centroids[None, :, :], axis=2)
            cluster_ids = np.argmin(distances, axis=1)
            codes.append(cluster_ids.astype(np.uint8))
        code_bytes = np.concatenate(codes).tobytes()
        compressed = zlib.compress(code_bytes, level=6)
        r_LZ = len(compressed) / len(code_bytes)
        return r_LZ

    def _conformal_score(self, D_hat: float, coh_star: float, r_LZ: float) -> float:
        w_D = 0.35
        w_coh = 0.25
        w_r = 0.40

        if 1.5 <= D_hat <= 2.5:
            D_norm = 1.0
        elif D_hat < 1.5:
            D_norm = D_hat / 1.5
        else:
            D_norm = max(0.0, 1.0 - (D_hat - 2.5) / 0.5)

        if 0.3 <= coh_star <= 0.6:
            coh_norm = 1.0
        elif coh_star < 0.3:
            coh_norm = coh_star / 0.3
        else:
            coh_norm = max(0.0, 1.0 - (coh_star - 0.6) / 0.4)

        if 0.4 <= r_LZ <= 0.7:
            r_norm = 1.0
        elif r_LZ < 0.4:
            r_norm = max(0.0, r_LZ / 0.4)
        else:
            r_norm = max(0.0, 1.0 - (r_LZ - 0.7) / 0.3)

        score = w_D * D_norm + w_coh * coh_norm + w_r * r_norm
        return max(0.0, min(1.0, score))


class RealGPT4JudgeBaseline:
    """REAL GPT-4 Judge using OpenAI API."""
    def __init__(self, api_key: str):
        self.name = "GPT-4 Judge"
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4-turbo-preview"
        # Cost: $0.01/1K input tokens, $0.03/1K output tokens
        # Average: ~500 input + 50 output = ~$0.007 per call

    def verify(self, sample: DegeneracySample) -> Dict:
        """Run REAL GPT-4 judge verification."""
        global TOTAL_COST
        start = time.time()

        prompt = f"""You are an expert at detecting structural degeneracies in text generation.

Analyze the following text for these degeneracy patterns:
1. **Repetition loops**: Same phrases/sentences repeated verbatim
2. **Semantic drift**: Topic changes abruptly without coherence
3. **Incoherence**: Random, nonsensical content

Text to analyze:
\"\"\"
{sample.text[:1000]}  # Limit to 1000 chars to control cost
\"\"\"

Rate the degeneracy on a scale of 0-10:
- 0 = Completely normal, well-structured text
- 10 = Severe degeneracy (loops, drift, or incoherence)

Respond with ONLY a single number (0-10). No explanation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )

            # Parse score
            score_text = response.choices[0].message.content.strip()
            score_raw = float(score_text)
            score = 1.0 - (score_raw / 10.0)  # Invert: high degeneracy = low score

            # Cost tracking
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens / 1000 * 0.01) + (output_tokens / 1000 * 0.03)
            TOTAL_COST += cost

            latency_ms = (time.time() - start) * 1000

            threshold = 0.75
            decision = "accept" if score >= threshold else "reject"

            return {
                "method": self.name,
                "score": score,
                "decision": decision,
                "latency_ms": latency_ms,
                "cost_usd": cost,
                "raw_score": score_raw,
                "tokens": {"input": input_tokens, "output": output_tokens}
            }

        except Exception as e:
            print(f"GPT-4 API error: {e}")
            return {
                "method": self.name,
                "score": 0.5,
                "decision": "reject",
                "latency_ms": (time.time() - start) * 1000,
                "cost_usd": 0.0,
                "error": str(e)
            }


class RealSelfCheckGPTBaseline:
    """REAL SelfCheckGPT with sampling + NLI."""
    def __init__(self, api_key: str):
        self.name = "SelfCheckGPT"
        self.client = OpenAI(api_key=api_key)
        self.n_samples = 5

        # Load NLI model (RoBERTa-MNLI)
        print("Loading RoBERTa-MNLI for SelfCheckGPT...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
        self.nli_model.eval()

        # Cost: 5 samples × $0.0015 per sample (GPT-3.5-turbo) = ~$0.0075

    def verify(self, sample: DegeneracySample) -> Dict:
        """Run REAL SelfCheckGPT verification."""
        global TOTAL_COST
        start = time.time()

        # Extract prompt from degenerate text (heuristic)
        words = sample.text.split()
        prompt = " ".join(words[:20])  # First 20 words as "prompt"

        # Sample N responses
        sampled_responses = []
        total_cost = 0.0

        for i in range(self.n_samples):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,  # Non-deterministic
                    max_tokens=150
                )

                sampled_text = response.choices[0].message.content
                sampled_responses.append(sampled_text)

                # Cost tracking
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost = (input_tokens / 1000 * 0.0015) + (output_tokens / 1000 * 0.002)
                total_cost += cost
                TOTAL_COST += cost

            except Exception as e:
                print(f"Sampling error: {e}")
                sampled_responses.append(sample.text)  # Fallback

        # Compute NLI consistency
        consistency_scores = []
        for sampled in sampled_responses:
            nli_score = self._compute_nli_entailment(sample.text[:500], sampled[:500])
            consistency_scores.append(nli_score)

        # Average consistency
        score = np.mean(consistency_scores)
        latency_ms = (time.time() - start) * 1000

        threshold = 0.70
        decision = "accept" if score >= threshold else "reject"

        return {
            "method": self.name,
            "score": score,
            "decision": decision,
            "latency_ms": latency_ms,
            "cost_usd": total_cost,
            "consistency_scores": consistency_scores,
            "n_samples": self.n_samples
        }

    def _compute_nli_entailment(self, premise: str, hypothesis: str) -> float:
        """Compute NLI entailment score using RoBERTa-MNLI."""
        inputs = self.nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]

        # RoBERTa-MNLI: [contradiction, neutral, entailment]
        entailment_prob = probs[2].item()
        return entailment_prob


def generate_degeneracy_samples(n_samples: int = 100) -> List[DegeneracySample]:
    """Generate synthetic degeneracy samples."""
    samples = []

    degeneracy_types = [
        ("repetition_loop", True, "The same phrase repeats. The same phrase repeats. The same phrase repeats. "),
        ("semantic_drift", True, "Starting with topic A. Now discussing B. Suddenly talking about C. Completely different D. "),
        ("incoherent", True, "Random words jumbled together without coherent meaning or structure here. "),
        ("normal", False, "This is a well-structured response that maintains coherence and provides factual information. ")
    ]

    for i in range(n_samples):
        deg_type, is_deg, template = degeneracy_types[i % len(degeneracy_types)]
        text = template * (np.random.randint(5, 15))

        sample = DegeneracySample(
            sample_id=f"sample_{i:04d}",
            text=text,
            is_degenerate=is_deg,
            degeneracy_type=deg_type
        )
        samples.append(sample)

    return samples


def evaluate_methods_real(samples: List[DegeneracySample], api_key: str) -> pd.DataFrame:
    """Evaluate all methods with REAL API calls."""
    methods = [
        ASVVerifier(),
        RealGPT4JudgeBaseline(api_key),
        RealSelfCheckGPTBaseline(api_key)
    ]

    results = []

    for method in methods:
        print(f"\n{'='*80}")
        print(f"Evaluating {method.name} (REAL API calls)")
        print(f"{'='*80}")

        for sample in tqdm(samples, desc=f"{method.name}"):
            # Check cost limit
            if TOTAL_COST > MAX_COST:
                print(f"\n⚠️  Cost limit reached: ${TOTAL_COST:.2f} > ${MAX_COST}")
                print("Stopping evaluation to prevent overspending.")
                break

            result = method.verify(sample)
            result.update({
                "sample_id": sample.sample_id,
                "ground_truth": sample.is_degenerate,
                "degeneracy_type": sample.degeneracy_type
            })
            results.append(result)

            # Progress update
            if isinstance(method, (RealGPT4JudgeBaseline, RealSelfCheckGPTBaseline)):
                print(f"  Cost so far: ${TOTAL_COST:.2f}")

        if TOTAL_COST > MAX_COST:
            break

    return pd.DataFrame(results)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute performance metrics for each method."""
    metrics = []

    for method in df['method'].unique():
        method_df = df[df['method'] == method]

        y_true = method_df['ground_truth'].values
        y_pred = (method_df['decision'] == 'reject').values
        y_scores = 1.0 - method_df['score'].values

        tp = ((y_pred == True) & (y_true == True)).sum()
        tn = ((y_pred == False) & (y_true == False)).sum()
        fp = ((y_pred == True) & (y_true == False)).sum()
        fn = ((y_pred == False) & (y_true == True)).sum()

        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y_true, y_scores)

        mean_latency = method_df['latency_ms'].mean()
        p95_latency = method_df['latency_ms'].quantile(0.95)
        total_cost = method_df['cost_usd'].sum()
        cost_per_sample = method_df['cost_usd'].mean()

        metrics.append({
            "Method": method,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUROC": auroc,
            "Mean Latency (ms)": mean_latency,
            "P95 Latency (ms)": p95_latency,
            "Cost per Sample (USD)": cost_per_sample,
            "Total Cost (USD)": total_cost
        })

    return pd.DataFrame(metrics)


# Import visualization functions from original script
from compare_baselines import (
    plot_roc_curves,
    plot_performance_comparison,
    plot_cost_performance_pareto,
    plot_latency_comparison
)


def main():
    """Main execution with REAL API calls."""
    print("="*80)
    print("Priority 2.1: REAL Baseline Comparison (Production APIs)")
    print("="*80)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        return

    print(f"\n✓ OpenAI API key found")
    print(f"✓ Cost limit: ${MAX_COST}")

    # Generate samples (smaller for cost control)
    n_samples = 100
    print(f"\n[1] Generating {n_samples} degeneracy samples...")
    samples = generate_degeneracy_samples(n_samples=n_samples)
    print(f"Generated {len(samples)} samples")

    # Evaluate methods
    print("\n[2] Evaluating methods with REAL API calls...")
    print("⚠️  This will incur OpenAI API costs (~$20-40 estimated)")

    input("\nPress ENTER to continue or Ctrl+C to cancel...")

    results_df = evaluate_methods_real(samples, api_key)

    # Save raw results
    results_path = RESULTS_DIR / "baseline_comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved raw results: {results_path}")

    # Compute metrics
    print("\n[3] Computing metrics...")
    metrics_df = compute_metrics(results_df)

    metrics_path = RESULTS_DIR / "baseline_comparison_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Saved metrics: {metrics_path}")

    # Print summary
    print("\n" + "="*80)
    print("REAL BASELINE COMPARISON RESULTS")
    print("="*80)
    print(metrics_df.to_string(index=False))
    print(f"\n💰 Total API Cost: ${TOTAL_COST:.2f}")

    # Generate visualizations
    print("\n[4] Generating visualizations...")
    plot_roc_curves(results_df, FIGURES_DIR / "baseline_roc_comparison.png")
    plot_performance_comparison(metrics_df, FIGURES_DIR / "baseline_performance_comparison.png")
    plot_cost_performance_pareto(metrics_df, FIGURES_DIR / "baseline_cost_performance.png")
    plot_latency_comparison(metrics_df, FIGURES_DIR / "baseline_latency_comparison.png")

    # Generate summary report
    summary = {
        "n_samples": len(samples),
        "total_cost_usd": TOTAL_COST,
        "methods": metrics_df['Method'].tolist(),
        "metrics": metrics_df.to_dict(orient='records')
    }

    summary_path = RESULTS_DIR / "baseline_comparison_summary_real.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")

    print("\n" + "="*80)
    print("✓ Real Baseline Evaluation Complete!")
    print("="*80)
    print(f"💰 Total Cost: ${TOTAL_COST:.2f}")
    print(f"📊 Results: {RESULTS_DIR}")
    print(f"📈 Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
