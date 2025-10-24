#!/usr/bin/env python3
"""
Extract Embeddings from LLM Outputs

Processes LLM-generated responses and extracts embeddings using transformer models.
Supports checkpoint/resume, batch processing, and comprehensive validation.

Usage:
    python extract_embeddings.py --benchmark truthfulqa --model gpt2
    python extract_embeddings.py --benchmark all --model roberta-base
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Add agent src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "src"))

from embedding_extractor import create_extractor, EmbeddingExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/embedding_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EmbeddingExtractionError(Exception):
    """Raised when embedding extraction fails."""
    pass


def load_llm_outputs(file_path: str) -> List[Dict]:
    """
    Load LLM outputs from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of output dictionaries

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is invalid or empty
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading outputs from: {file_path}")
    outputs = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Validate required fields
                    if 'id' not in data:
                        raise ValueError(f"Line {line_num}: missing 'id' field")
                    if 'llm_response' not in data:
                        raise ValueError(f"Line {line_num}: missing 'llm_response' field")
                    # Validate llm_response is string
                    if not isinstance(data['llm_response'], str):
                        raise ValueError(f"Line {line_num}: 'llm_response' must be string")
                    outputs.append(data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Line {line_num}: invalid JSON - {e}")

    except Exception as e:
        raise ValueError(f"Failed to load {file_path}: {e}")

    if not outputs:
        raise ValueError(f"File is empty or contains no valid data: {file_path}")

    logger.info(f"Loaded {len(outputs)} outputs")
    return outputs


def load_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load extraction checkpoint if exists.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Checkpoint dictionary with 'completed_ids' and 'stats'
    """
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
    """Save extraction checkpoint."""
    try:
        # Create directory if doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def save_embeddings(
    output_path: str,
    sample_id: str,
    embedding: np.ndarray,
    metadata: Dict,
):
    """
    Save embedding to NPZ file.

    Args:
        output_path: Base output directory
        sample_id: Unique sample identifier
        embedding: Embedding array
        metadata: Additional metadata to save

    Raises:
        RuntimeError: If save fails
    """
    try:
        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Save as .npz file
        npz_path = os.path.join(output_path, f"{sample_id}.npz")
        np.savez_compressed(
            npz_path,
            embedding=embedding,
            **metadata  # Save all metadata as separate arrays
        )

    except Exception as e:
        raise RuntimeError(f"Failed to save embedding for {sample_id}: {e}")


def validate_embedding(
    embedding: np.ndarray,
    expected_dim: int,
    sample_id: str,
) -> bool:
    """
    Validate embedding array.

    Args:
        embedding: Embedding to validate
        expected_dim: Expected embedding dimension
        sample_id: Sample ID for error messages

    Returns:
        True if valid, False otherwise
    """
    # Check shape
    if embedding.ndim != 1:
        logger.error(f"{sample_id}: embedding must be 1D, got shape {embedding.shape}")
        return False

    if embedding.shape[0] != expected_dim:
        logger.error(
            f"{sample_id}: expected dimension {expected_dim}, "
            f"got {embedding.shape[0]}"
        )
        return False

    # Check for NaN/Inf
    if np.isnan(embedding).any():
        logger.error(f"{sample_id}: embedding contains NaN values")
        return False

    if np.isinf(embedding).any():
        logger.error(f"{sample_id}: embedding contains Inf values")
        return False

    # Check that embedding is not all zeros
    if np.allclose(embedding, 0.0):
        logger.warning(f"{sample_id}: embedding is all zeros")
        # This is a warning, not an error - might be valid for empty text

    return True


def extract_for_benchmark(
    benchmark: str,
    model_name: str,
    batch_size: int,
    device: Optional[str] = None,
):
    """
    Extract embeddings for a single benchmark.

    Args:
        benchmark: Benchmark name ('truthfulqa', 'fever', 'halueval')
        model_name: Model to use ('gpt2', 'roberta-base', 'bert-base-uncased')
        batch_size: Batch size for processing
        device: Device to use (None for auto-detect)

    Raises:
        EmbeddingExtractionError: If extraction fails
    """
    logger.info(f"=== Extracting embeddings for {benchmark.upper()} ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Batch size: {batch_size}")

    # Paths
    input_file = f"data/llm_outputs/{benchmark}_outputs.jsonl"
    output_dir = f"data/embeddings/{model_name}/{benchmark}"
    checkpoint_file = f"data/checkpoints/{benchmark}_{model_name}_embeddings.json"

    # Load LLM outputs
    try:
        outputs = load_llm_outputs(input_file)
    except Exception as e:
        raise EmbeddingExtractionError(f"Failed to load outputs: {e}")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_file)
    completed_ids = set(checkpoint['completed_ids'])

    # Filter remaining samples
    remaining = [o for o in outputs if o['id'] not in completed_ids]
    logger.info(f"Remaining samples: {len(remaining)} / {len(outputs)}")

    if not remaining:
        logger.info(f"{benchmark.upper()} already complete!")
        return

    # Create extractor
    try:
        logger.info("Loading embedding model...")
        extractor = create_extractor(
            model_name=model_name,
            pooling='mean',  # Mean pooling is most robust
            device=device,
            batch_size=batch_size,
        )
        logger.info(f"Model loaded: {extractor.get_model_info()}")
        expected_dim = extractor.embedding_dim
    except Exception as e:
        raise EmbeddingExtractionError(f"Failed to create extractor: {e}")

    # Process in batches
    failed_samples = []
    successful = 0

    try:
        # Create batches
        num_batches = (len(remaining) + batch_size - 1) // batch_size

        with tqdm(total=len(remaining), desc=f"Extracting {benchmark}") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(remaining))
                batch = remaining[start_idx:end_idx]

                # Extract texts and IDs
                texts = [sample['llm_response'] for sample in batch]
                sample_ids = [sample['id'] for sample in batch]

                try:
                    # Extract embeddings for batch
                    embeddings = extractor.extract_batch(texts)

                    # Validate and save each embedding
                    for i, (sample_id, embedding, sample) in enumerate(
                        zip(sample_ids, embeddings, batch)
                    ):
                        # Validate embedding
                        if not validate_embedding(embedding, expected_dim, sample_id):
                            logger.error(f"Validation failed for {sample_id}")
                            failed_samples.append(sample_id)
                            continue

                        # Save embedding
                        try:
                            metadata = {
                                'sample_id': sample_id,
                                'model_name': model_name,
                                'text_length': len(sample['llm_response']),
                                'benchmark': benchmark,
                            }
                            save_embeddings(output_dir, sample_id, embedding, metadata)

                            # Update checkpoint
                            checkpoint['completed_ids'].append(sample_id)
                            successful += 1

                        except Exception as e:
                            logger.error(f"Failed to save {sample_id}: {e}")
                            failed_samples.append(sample_id)

                    # Save checkpoint after each batch
                    checkpoint['stats']['total_samples'] = len(checkpoint['completed_ids'])
                    checkpoint['stats']['successful'] = successful
                    checkpoint['stats']['failed'] = len(failed_samples)
                    save_checkpoint(checkpoint_file, checkpoint)

                    pbar.update(len(batch))

                except Exception as e:
                    logger.error(f"Batch {batch_idx} extraction failed: {e}")
                    # Mark all samples in batch as failed
                    failed_samples.extend(sample_ids)
                    pbar.update(len(batch))

    except KeyboardInterrupt:
        logger.warning("Extraction interrupted by user")
        save_checkpoint(checkpoint_file, checkpoint)
        raise

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        save_checkpoint(checkpoint_file, checkpoint)
        raise EmbeddingExtractionError(f"Extraction failed: {e}")

    # Final summary
    logger.info(f"\n=== {benchmark.upper()} Extraction Complete ===")
    logger.info(f"Total samples: {len(outputs)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(failed_samples)}")
    if failed_samples:
        logger.warning(f"Failed sample IDs: {failed_samples[:10]}...")  # Show first 10

    # Final validation: check output directory
    saved_files = len(list(Path(output_dir).glob("*.npz")))
    logger.info(f"Saved embeddings: {saved_files}")

    if saved_files != successful:
        logger.error(
            f"Mismatch: {successful} successful extractions "
            f"but {saved_files} files saved"
        )

    # Update final checkpoint
    checkpoint['stats']['end_time'] = datetime.now().isoformat()
    save_checkpoint(checkpoint_file, checkpoint)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract embeddings from LLM outputs'
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
        help='Batch size for processing'
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
    os.makedirs('data/embeddings', exist_ok=True)
    os.makedirs('data/checkpoints', exist_ok=True)

    # Process benchmarks
    benchmarks = ['truthfulqa', 'fever', 'halueval'] if args.benchmark == 'all' else [args.benchmark]

    for benchmark in benchmarks:
        try:
            extract_for_benchmark(
                benchmark=benchmark,
                model_name=args.model,
                batch_size=args.batch_size,
                device=args.device,
            )
        except EmbeddingExtractionError as e:
            logger.error(f"Failed to extract {benchmark}: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error for {benchmark}: {e}")
            continue

    logger.info("\n=== All extractions complete ===")


if __name__ == '__main__':
    main()
