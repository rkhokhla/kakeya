#!/usr/bin/env python3
"""
Process degeneracy dataset: extract embeddings, compute signals, compute baselines, and evaluate.

This is an end-to-end pipeline for the structural degeneracy evaluation.
"""

import json
import logging
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

# Add agent src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "src"))

from embedding_extractor import create_extractor
from signals import compute_coherence, compute_compressibility_pq, compute_D_hat
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 10]
    return sentences


def extract_embeddings(input_path: Path, output_dir: Path):
    """Extract embeddings from degeneracy samples."""
    logger.info(f"Extracting embeddings from {input_path}")

    # Create extractor
    extractor = create_extractor(model_name='gpt2', device='cpu')

    # Load samples
    samples = []
    with open(input_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} samples")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract embeddings for each sample
    for sample in tqdm(samples, desc="Extracting embeddings"):
        sample_id = sample['id']
        text = sample['text']

        # Split into sentences
        sentences = split_into_sentences(text)

        if len(sentences) < 2:
            # For very short texts, use the whole text as one sentence
            sentences = [text]

        # Extract embeddings
        try:
            embeddings = extractor.extract_batch(sentences)

            # Save
            output_path = output_dir / f"{sample_id}.npy"
            np.save(output_path, embeddings)

        except Exception as e:
            logger.warning(f"Failed to extract embeddings for {sample_id}: {e}")
            # Save empty array
            np.save(output_dir / f"{sample_id}.npy", np.array([]))

    logger.info(f"✅ Embeddings saved to {output_dir}")


def compute_signals_for_degeneracy(embeddings_dir: Path, output_dir: Path, input_path: Path):
    """Compute ASV signals for degeneracy samples."""
    logger.info(f"Computing signals from {embeddings_dir}")

    # Load sample metadata
    samples = {}
    with open(input_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            samples[sample['id']] = sample

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each embedding file
    embedding_files = list(embeddings_dir.glob('*.npy'))

    for emb_file in tqdm(embedding_files, desc="Computing signals"):
        sample_id = emb_file.stem

        if sample_id not in samples:
            logger.warning(f"Sample {sample_id} not found in metadata")
            continue

        # Load embeddings
        embeddings = np.load(emb_file)

        if embeddings.shape[0] < 2:
            logger.warning(f"Skipping {sample_id}: too few sentences ({embeddings.shape[0]})")
            continue

        try:
            # Compute D̂ (fractal dimension)
            scales = [2, 4, 8, 16, 32]
            N_j = {}
            for scale in scales:
                try:
                    # Count unique non-empty cells at this scale
                    n_cells = int(embeddings.shape[0] / scale) + 1
                    cells = embeddings.reshape(n_cells, -1) if embeddings.shape[0] >= scale else embeddings
                    unique_cells = len(set(map(tuple, cells)))
                    N_j[scale] = unique_cells
                except Exception:
                    N_j[scale] = 1  # Fallback

            D_hat = compute_D_hat(scales, N_j)

            # Compute coh★ (directional coherence)
            coh_star, v_star = compute_coherence(embeddings, num_directions=100, num_bins=20, seed=42)

            # Compute r_LZ (compressibility with PQ)
            r_LZ = compute_compressibility_pq(embeddings)

            # Save signals
            signal_data = {
                'id': sample_id,
                'D_hat': float(D_hat),
                'coh_star': float(coh_star),
                'r_LZ': float(r_LZ),
                'scales': scales,
                'N_j': N_j,
                'metadata': samples[sample_id].get('metadata', {})
            }

            output_path = output_dir / f"{sample_id}.json"
            with open(output_path, 'w') as f:
                json.dump(signal_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to compute signals for {sample_id}: {e}")

    logger.info(f"✅ Signals saved to {output_dir}")


def compute_baselines_for_degeneracy(input_path: Path, output_dir: Path):
    """Compute baseline metrics for degeneracy samples."""
    logger.info(f"Computing baselines from {input_path}")

    # Load model
    logger.info("Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Load samples
    samples = []
    with open(input_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each sample
    for sample in tqdm(samples, desc="Computing baselines"):
        sample_id = sample['id']
        text = sample['text']

        try:
            # Tokenize
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            input_ids = inputs['input_ids']

            # Get logits
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits

            # Compute token probabilities
            log_probs = torch.log_softmax(logits, dim=-1)

            # Get probabilities for actual tokens
            token_log_probs = []
            for i in range(input_ids.shape[1] - 1):
                token_id = input_ids[0, i + 1].item()
                token_log_prob = log_probs[0, i, token_id].item()
                token_log_probs.append(token_log_prob)

            if len(token_log_probs) == 0:
                continue

            # Compute metrics
            perplexity = np.exp(-np.mean(token_log_probs))
            mean_token_prob = np.exp(np.mean(token_log_probs))
            min_token_prob = np.exp(min(token_log_probs))
            entropy = -np.mean(token_log_probs)

            # Save
            baseline_data = {
                'id': sample_id,
                'perplexity': float(perplexity),
                'mean_token_prob': float(mean_token_prob),
                'min_token_prob': float(min_token_prob),
                'entropy': float(entropy)
            }

            output_path = output_dir / f"{sample_id}.json"
            with open(output_path, 'w') as f:
                json.dump(baseline_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to compute baselines for {sample_id}: {e}")

    logger.info(f"✅ Baselines saved to {output_dir}")


def main():
    """Main pipeline."""
    print("=== Degeneracy Dataset Processing Pipeline ===\n")

    # Paths
    input_path = Path('data/benchmarks/degeneracy/degeneracy_synthetic.jsonl')
    embeddings_dir = Path('data/embeddings/degeneracy')
    signals_dir = Path('data/signals/degeneracy')
    baselines_dir = Path('data/baselines/degeneracy')

    # Step 1: Extract embeddings
    if not embeddings_dir.exists() or len(list(embeddings_dir.glob('*.npy'))) == 0:
        print("\nStep 1: Extracting embeddings...")
        extract_embeddings(input_path, embeddings_dir)
    else:
        print(f"\nStep 1: ✅ Embeddings already exist ({len(list(embeddings_dir.glob('*.npy')))} files)")

    # Step 2: Compute signals
    if not signals_dir.exists() or len(list(signals_dir.glob('*.json'))) == 0:
        print("\nStep 2: Computing ASV signals...")
        compute_signals_for_degeneracy(embeddings_dir, signals_dir, input_path)
    else:
        print(f"\nStep 2: ✅ Signals already exist ({len(list(signals_dir.glob('*.json')))} files)")

    # Step 3: Compute baselines
    if not baselines_dir.exists() or len(list(baselines_dir.glob('*.json'))) == 0:
        print("\nStep 3: Computing baseline metrics...")
        compute_baselines_for_degeneracy(input_path, baselines_dir)
    else:
        print(f"\nStep 3: ✅ Baselines already exist ({len(list(baselines_dir.glob('*.json')))} files)")

    print("\n✅ All processing complete!")
    print(f"   Embeddings: {embeddings_dir}")
    print(f"   Signals: {signals_dir}")
    print(f"   Baselines: {baselines_dir}")
    print("\nReady for evaluation with evaluate_methods.py --benchmark degeneracy")


if __name__ == '__main__':
    main()
