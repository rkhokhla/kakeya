#!/usr/bin/env python3
"""
Compute Signals from LLM Outputs

Processes LLM-generated responses and computes three signals:
- D̂ (intrinsic dimensionality)
- coh★ (coherence score)
- r_LZ (Lempel-Ziv complexity ratio)

For D̂ and coh★ to be valid, we need multiple embeddings per sample.
This script splits texts into sentences, extracts embeddings for each sentence,
then computes signals from the sentence embeddings.

Usage:
    python compute_signals.py --benchmark truthfulqa
    python compute_signals.py --benchmark all
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import numpy as np
from tqdm import tqdm
import re

# Add agent src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "src"))

from signal_computer import create_signal_computer, SignalComputationError
from embedding_extractor import create_extractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/signal_computation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple regex.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Simple sentence splitting (can be improved with NLTK if needed)
    # Split on . ! ? followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Filter out very short sentences (< 10 chars)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 10]

    return sentences


def load_llm_outputs(benchmark: str) -> List[Dict]:
    """Load LLM outputs for a benchmark."""
    file_path = f"data/llm_outputs/{benchmark}_outputs.jsonl"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading outputs from: {file_path}")
    outputs = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            outputs.append(data)

    logger.info(f"Loaded {len(outputs)} outputs")
    return outputs


def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load computation checkpoint if exists."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                logger.info(
                    f"Loaded checkpoint: {len(checkpoint['completed_ids'])} samples completed"
                )
                return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    return {
        'completed_ids': [],
        'stats': {
            'total_samples': 0,
            'successful': 0,
            'failed': 0,
            'start_time': datetime.now().isoformat(),
        }
    }


def save_checkpoint(checkpoint_path: str, checkpoint: Dict):
    """Save computation checkpoint."""
    try:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def save_signals(
    output_path: str,
    sample_id: str,
    signals: Dict,
):
    """Save computed signals to JSON file."""
    try:
        os.makedirs(output_path, exist_ok=True)

        json_path = os.path.join(output_path, f"{sample_id}.json")
        with open(json_path, 'w') as f:
            json.dump(signals, f, indent=2)

    except Exception as e:
        raise RuntimeError(f"Failed to save signals for {sample_id}: {e}")


def compute_signals_for_benchmark(
    benchmark: str,
    model_name: str = 'gpt2',
    batch_size: int = 32,
    device: str = None,
):
    """
    Compute signals for a single benchmark.

    Args:
        benchmark: Benchmark name
        model_name: Model for embedding extraction
        batch_size: Batch size for embedding extraction
        device: Device to use
    """
    logger.info(f"=== Computing signals for {benchmark.upper()} ===")

    # Paths
    input_file = f"data/llm_outputs/{benchmark}_outputs.jsonl"
    output_dir = f"data/signals/{benchmark}"
    checkpoint_file = f"data/checkpoints/{benchmark}_signals.json"

    # Load LLM outputs
    try:
        outputs = load_llm_outputs(benchmark)
    except Exception as e:
        logger.error(f"Failed to load outputs: {e}")
        return

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_file)
    completed_ids = set(checkpoint['completed_ids'])

    # Filter remaining samples
    remaining = [o for o in outputs if o['id'] not in completed_ids]
    logger.info(f"Remaining samples: {len(remaining)} / {len(outputs)}")

    if not remaining:
        logger.info(f"{benchmark.upper()} already complete!")
        return

    # Create embedding extractor
    try:
        logger.info("Loading embedding model...")
        extractor = create_extractor(
            model_name=model_name,
            pooling='mean',
            device=device,
            batch_size=batch_size,
        )
        logger.info(f"Model loaded: {extractor.get_model_info()}")
    except Exception as e:
        logger.error(f"Failed to create extractor: {e}")
        return

    # Create signal computer
    signal_computer = create_signal_computer()

    # Process each sample
    failed_samples = []
    successful = 0

    try:
        with tqdm(total=len(remaining), desc=f"Computing {benchmark}") as pbar:
            for sample in remaining:
                sample_id = sample['id']
                text = sample['llm_response']

                try:
                    # Split into sentences
                    sentences = split_into_sentences(text)

                    if len(sentences) == 0:
                        logger.warning(f"{sample_id}: No sentences found, using full text")
                        sentences = [text]

                    logger.debug(f"{sample_id}: {len(sentences)} sentences")

                    # Extract embeddings for each sentence
                    # Process in batches
                    all_embeddings = []
                    for i in range(0, len(sentences), batch_size):
                        batch_sentences = sentences[i:i+batch_size]
                        batch_embeddings = extractor.extract_batch(batch_sentences)
                        all_embeddings.append(batch_embeddings)

                    # Concatenate all embeddings
                    embeddings = np.vstack(all_embeddings)

                    logger.debug(f"{sample_id}: embeddings shape = {embeddings.shape}")

                    # Compute signals
                    result = signal_computer.compute_all_signals(embeddings, text)

                    # Save signals
                    signals_dict = result.to_dict()
                    signals_dict['sample_id'] = sample_id
                    signals_dict['benchmark'] = benchmark
                    signals_dict['n_sentences'] = len(sentences)

                    save_signals(output_dir, sample_id, signals_dict)

                    # Update checkpoint
                    checkpoint['completed_ids'].append(sample_id)
                    successful += 1

                    # Log if invalid
                    if not result.is_valid():
                        logger.warning(
                            f"{sample_id}: Invalid signals - {result.warnings}"
                        )

                except Exception as e:
                    logger.error(f"Failed to process {sample_id}: {e}")
                    failed_samples.append(sample_id)

                # Save checkpoint periodically
                if successful % 100 == 0:
                    checkpoint['stats']['total_samples'] = len(checkpoint['completed_ids'])
                    checkpoint['stats']['successful'] = successful
                    checkpoint['stats']['failed'] = len(failed_samples)
                    save_checkpoint(checkpoint_file, checkpoint)

                pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("Computation interrupted by user")
        save_checkpoint(checkpoint_file, checkpoint)
        raise

    except Exception as e:
        logger.error(f"Computation failed: {e}")
        save_checkpoint(checkpoint_file, checkpoint)
        raise

    # Final summary
    logger.info(f"\n=== {benchmark.upper()} Computation Complete ===")
    logger.info(f"Total samples: {len(outputs)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(failed_samples)}")
    if failed_samples:
        logger.warning(f"Failed sample IDs: {failed_samples[:10]}...")

    # Final validation: check output directory
    saved_files = len(list(Path(output_dir).glob("*.json")))
    logger.info(f"Saved signals: {saved_files}")

    if saved_files != successful:
        logger.error(
            f"Mismatch: {successful} successful computations "
            f"but {saved_files} files saved"
        )

    # Update final checkpoint
    checkpoint['stats']['end_time'] = datetime.now().isoformat()
    checkpoint['stats']['total_samples'] = len(checkpoint['completed_ids'])
    checkpoint['stats']['successful'] = successful
    checkpoint['stats']['failed'] = len(failed_samples)
    save_checkpoint(checkpoint_file, checkpoint)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compute signals from LLM outputs'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        choices=['truthfulqa', 'fever', 'halueval', 'all'],
        help='Which benchmark to process'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        choices=['gpt2', 'roberta-base', 'bert-base-uncased'],
        help='Embedding model to use'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding extraction'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'mps', 'cpu'],
        help='Device to use (auto-detect if not specified)'
    )

    args = parser.parse_args()

    # Create directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/signals', exist_ok=True)
    os.makedirs('data/checkpoints', exist_ok=True)

    # Process benchmarks
    benchmarks = ['truthfulqa', 'fever', 'halueval'] if args.benchmark == 'all' else [args.benchmark]

    for benchmark in benchmarks:
        try:
            compute_signals_for_benchmark(
                benchmark=benchmark,
                model_name=args.model,
                batch_size=args.batch_size,
                device=args.device,
            )
        except Exception as e:
            logger.error(f"Failed to compute signals for {benchmark}: {e}")
            continue

    logger.info("\n=== All signal computations complete ===")


if __name__ == '__main__':
    main()
