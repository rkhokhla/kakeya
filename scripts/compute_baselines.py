#!/usr/bin/env python3
"""
Compute Baseline Metrics from LLM Outputs

Processes LLM-generated responses and computes standard baseline metrics:
- Perplexity (GPT-2)
- Token-level probabilities (mean, min)
- Entropy
- BERT-Score (if reference available)

Usage:
    python compute_baselines.py --benchmark truthfulqa
    python compute_baselines.py --benchmark all --batch-size 16
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm

# Add agent src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "src"))

from baseline_methods import create_baseline_detector, BaselineMethodsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/baseline_computation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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


def save_baselines(
    output_path: str,
    sample_id: str,
    baselines: Dict,
):
    """Save computed baselines to JSON file."""
    try:
        os.makedirs(output_path, exist_ok=True)

        json_path = os.path.join(output_path, f"{sample_id}.json")
        with open(json_path, 'w') as f:
            json.dump(baselines, f, indent=2)

    except Exception as e:
        raise RuntimeError(f"Failed to save baselines for {sample_id}: {e}")


def compute_baselines_for_benchmark(
    benchmark: str,
    model_name: str = 'gpt2',
    device: str = None,
):
    """
    Compute baseline metrics for a single benchmark.

    Args:
        benchmark: Benchmark name
        model_name: Model for perplexity computation
        device: Device to use
    """
    logger.info(f"=== Computing baselines for {benchmark.upper()} ===")

    # Paths
    input_file = f"data/llm_outputs/{benchmark}_outputs.jsonl"
    output_dir = f"data/baselines/{benchmark}"
    checkpoint_file = f"data/checkpoints/{benchmark}_baselines.json"

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

    # Create baseline detector
    try:
        logger.info(f"Loading {model_name} for baseline computation...")
        detector = create_baseline_detector(
            model_name=model_name,
            device=device,
            max_length=1024,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to create detector: {e}")
        return

    # Process each sample
    failed_samples = []
    successful = 0

    try:
        with tqdm(total=len(remaining), desc=f"Computing {benchmark}") as pbar:
            for sample in remaining:
                sample_id = sample['id']
                text = sample['llm_response']

                try:
                    logger.debug(f"{sample_id}: Computing baselines for {len(text)} chars")

                    # Compute all baseline metrics
                    result = detector.compute_all_metrics(text)

                    # Save baselines
                    baselines_dict = result.to_dict()
                    baselines_dict['sample_id'] = sample_id
                    baselines_dict['benchmark'] = benchmark
                    baselines_dict['model'] = model_name

                    save_baselines(output_dir, sample_id, baselines_dict)

                    # Update checkpoint
                    checkpoint['completed_ids'].append(sample_id)
                    successful += 1

                    # Log if invalid
                    if not result.is_valid():
                        logger.warning(
                            f"{sample_id}: Invalid baselines - {result.warnings}"
                        )

                    # Log high perplexity (potential issue)
                    if result.perplexity > 500:
                        logger.warning(
                            f"{sample_id}: Very high perplexity={result.perplexity:.2f}"
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
    logger.info(f"Saved baselines: {saved_files}")

    if saved_files != len(checkpoint['completed_ids']):
        logger.error(
            f"Mismatch: {len(checkpoint['completed_ids'])} completed "
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
        description='Compute baseline metrics from LLM outputs'
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
        choices=['gpt2', 'gpt2-medium', 'gpt2-large'],
        help='Model to use for perplexity'
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
    os.makedirs('data/baselines', exist_ok=True)
    os.makedirs('data/checkpoints', exist_ok=True)

    # Process benchmarks
    benchmarks = ['truthfulqa', 'fever', 'halueval'] if args.benchmark == 'all' else [args.benchmark]

    for benchmark in benchmarks:
        try:
            compute_baselines_for_benchmark(
                benchmark=benchmark,
                model_name=args.model,
                device=args.device,
            )
        except Exception as e:
            logger.error(f"Failed to compute baselines for {benchmark}: {e}")
            continue

    logger.info("\n=== All baseline computations complete ===")


if __name__ == '__main__':
    main()
