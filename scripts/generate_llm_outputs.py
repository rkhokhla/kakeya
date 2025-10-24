#!/usr/bin/env python3
"""
Generate LLM outputs for benchmark samples (Phase 3).

Usage:
    python scripts/generate_llm_outputs.py --config config/eval_config.yaml
    python scripts/generate_llm_outputs.py --benchmark truthfulqa --limit 100
    python scripts/generate_llm_outputs.py --resume-from data/checkpoints/truthfulqa_checkpoint.json

This script:
1. Loads benchmark samples
2. Generates LLM responses using OpenAI API
3. Saves outputs to data/llm_outputs/
4. Tracks cost and progress
5. Supports resume from checkpoints
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import time

# Add agent/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "src"))

from llm_client import LLMClient

# Import benchmark loaders (simple Python versions)
import csv
import random


def load_truthfulqa(path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    """Load TruthfulQA benchmark."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples > 0 and i >= max_samples:
                break
            samples.append({
                "id": f"truthfulqa_{i+1}",
                "source": "truthfulqa",
                "prompt": row["Question"],
                "category": row["Category"],
                "ground_truth": row["Best Answer"],
                "metadata": {
                    "type": row["Type"],
                    "correct_answers": row["Correct Answers"],
                    "incorrect_answers": row["Incorrect Answers"],
                }
            })
    return samples


def load_fever(path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    """Load FEVER benchmark."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break
            record = json.loads(line)
            samples.append({
                "id": f"fever_{record['id']}",
                "source": "fever",
                "prompt": f"Is the following claim factually correct? {record['claim']}",
                "category": record.get("label", ""),
                "ground_truth": record.get("label") == "SUPPORTS",
                "metadata": {
                    "claim": record["claim"],
                    "label": record.get("label"),
                    "evidence": record.get("evidence"),
                }
            })

    # Sample if needed
    if max_samples > 0 and len(samples) > max_samples:
        random.seed(42)
        samples = random.sample(samples, max_samples)

    return samples


def load_halueval(path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    """Load HaluEval benchmark."""
    with open(path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    samples = []
    for i, record in enumerate(records):
        if max_samples > 0 and i >= max_samples:
            break
        samples.append({
            "id": f"halueval_qa_{i+1}",
            "source": "halueval",
            "prompt": record["question"],
            "category": "qa",
            "ground_truth": not record["hallucination"],  # True if not hallucinated
            "metadata": {
                "knowledge": record.get("knowledge", ""),
                "hallucination": record["hallucination"],
            }
        })

    # Sample if needed
    if max_samples > 0 and len(samples) > max_samples:
        random.seed(42)
        samples = random.sample(samples, max_samples)

    return samples


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint file."""
    if not os.path.exists(checkpoint_path):
        return {"completed": [], "stats": {}}

    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def save_checkpoint(checkpoint_path: str, data: Dict[str, Any]):
    """Save checkpoint file."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)


def save_output(output_path: str, sample: Dict[str, Any]):
    """Append sample to output JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'a') as f:
        f.write(json.dumps(sample) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Generate LLM outputs for benchmarks")
    parser.add_argument("--config", default="config/eval_config.yaml", help="Config file")
    parser.add_argument("--benchmark", choices=["truthfulqa", "fever", "halueval", "all"],
                        default="all", help="Which benchmark to process")
    parser.add_argument("--limit", type=int, help="Limit samples per benchmark")
    parser.add_argument("--resume-from", help="Resume from checkpoint file")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no API calls)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    # Load config (for now, use hardcoded values)
    # TODO: Parse YAML config file

    benchmarks_config = {
        "truthfulqa": {
            "path": "data/benchmarks/truthfulqa/TruthfulQA.csv",
            "max_samples": args.limit or 0,
            "loader": load_truthfulqa,
        },
        "fever": {
            "path": "data/benchmarks/fever/shared_task_dev.jsonl",
            "max_samples": args.limit or 2500,
            "loader": load_fever,
        },
        "halueval": {
            "path": "data/benchmarks/halueval/qa_samples.json",
            "max_samples": args.limit or 5000,
            "loader": load_halueval,
        },
    }

    # Determine which benchmarks to process
    if args.benchmark == "all":
        benchmarks_to_process = list(benchmarks_config.keys())
    else:
        benchmarks_to_process = [args.benchmark]

    # Initialize LLM client
    if not args.dry_run:
        if not os.getenv("OPENAI_API_KEY"):
            logging.error("OPENAI_API_KEY not set. Set it with:")
            logging.error("  export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)

        client = LLMClient(
            generation_model="gpt-4-turbo-preview",
            max_retries=3,
        )
    else:
        client = None
        logging.info("DRY RUN MODE: No API calls will be made")

    # Process each benchmark
    total_samples = 0
    total_cost = 0.0

    for benchmark_name in benchmarks_to_process:
        config = benchmarks_config[benchmark_name]

        logging.info(f"\n{'='*60}")
        logging.info(f"Processing: {benchmark_name}")
        logging.info(f"{'='*60}")

        # Load samples
        samples = config["loader"](config["path"], config["max_samples"])
        logging.info(f"Loaded {len(samples)} samples from {benchmark_name}")

        # Setup paths
        output_path = f"data/llm_outputs/{benchmark_name}_outputs.jsonl"
        checkpoint_path = f"data/checkpoints/{benchmark_name}_checkpoint.json"

        # Load checkpoint if resuming
        checkpoint = load_checkpoint(checkpoint_path)
        completed_ids = set(checkpoint.get("completed", []))

        # Filter out completed samples
        remaining_samples = [s for s in samples if s["id"] not in completed_ids]
        logging.info(f"Remaining samples: {len(remaining_samples)} (already completed: {len(completed_ids)})")

        if len(remaining_samples) == 0:
            logging.info(f"All samples already completed for {benchmark_name}")
            continue

        # Generate outputs
        for i, sample in enumerate(remaining_samples):
            sample_id = sample["id"]
            prompt = sample["prompt"]

            logging.info(f"\n[{i+1}/{len(remaining_samples)}] Processing {sample_id}")
            logging.info(f"Prompt: {prompt[:100]}...")

            if args.dry_run:
                # Dry run: just log
                result = {
                    "response": "[DRY RUN] Generated response would appear here",
                    "model": "gpt-4-turbo-preview",
                    "tokens": {"prompt": 50, "completion": 100, "total": 150},
                    "cost": 0.0015,
                }
            else:
                # Real API call
                try:
                    result = client.generate(
                        prompt=prompt,
                        system_prompt="You are a helpful assistant. Answer the question concisely and accurately.",
                        max_tokens=256,
                        temperature=0.7,
                    )
                except Exception as e:
                    logging.error(f"Error generating response for {sample_id}: {e}")
                    continue

            # Save output
            output_sample = {
                **sample,
                "llm_response": result["response"],
                "llm_model": result["model"],
                "llm_tokens": result["tokens"],
                "llm_cost": result["cost"],
            }
            save_output(output_path, output_sample)

            # Update checkpoint
            completed_ids.add(sample_id)
            checkpoint["completed"] = list(completed_ids)
            if "stats" not in checkpoint:
                checkpoint["stats"] = {}
            checkpoint["stats"]["total_samples"] = len(completed_ids)
            checkpoint["stats"]["total_cost"] = checkpoint.get("stats", {}).get("total_cost", 0.0) + result["cost"]
            save_checkpoint(checkpoint_path, checkpoint)

            total_samples += 1
            total_cost += result["cost"]

            logging.info(f"Response: {result['response'][:100]}...")
            logging.info(f"Cost: ${result['cost']:.6f} | Total: ${total_cost:.2f}")

            # Rate limiting (avoid hitting OpenAI rate limits)
            if not args.dry_run and i < len(remaining_samples) - 1:
                time.sleep(0.5)  # 2 requests per second

        logging.info(f"\n✓ Completed {benchmark_name}: {len(completed_ids)} samples")

    # Final summary
    logging.info(f"\n{'='*60}")
    logging.info(f"SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"Total samples processed: {total_samples}")
    logging.info(f"Total cost: ${total_cost:.2f}")

    if not args.dry_run and client:
        logging.info("\nClient stats:")
        logging.info(json.dumps(client.get_stats(), indent=2))


if __name__ == "__main__":
    main()
