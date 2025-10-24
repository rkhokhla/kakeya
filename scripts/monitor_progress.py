#!/usr/bin/env python3
"""Monitor LLM generation progress across all benchmarks."""

import json
import os
from pathlib import Path

def load_checkpoint(name):
    """Load checkpoint stats."""
    path = f"data/checkpoints/{name}_checkpoint.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            return data.get("stats", {})
    return {}

def main():
    print("=== LLM Generation Progress Monitor ===\n")

    benchmarks = [
        ("truthfulqa", 789),
        ("fever", 2500),
        ("halueval", 5000),
    ]

    total_samples = 0
    total_cost = 0.0
    total_target = sum(target for _, target in benchmarks)

    for name, target in benchmarks:
        stats = load_checkpoint(name)
        completed = stats.get("total_samples", 0)
        cost = stats.get("total_cost", 0.0)

        if completed > 0:
            progress = (completed / target) * 100
            print(f"📊 {name.upper()}:")
            print(f"   Completed: {completed:,} / {target:,} samples")
            print(f"   Cost: ${cost:.4f}")
            print(f"   Progress: {progress:.1f}%")
            print()

            total_samples += completed
            total_cost += cost

    if total_samples > 0:
        print("=== Total Summary ===")
        print(f"Total Completed: {total_samples:,} / {total_target:,} samples")
        print(f"Total Cost: ${total_cost:.4f}")
        print(f"Overall Progress: {(total_samples/total_target)*100:.1f}%")
        print()

    # Estimate remaining
    if total_samples > 0 and total_samples < total_target:
        avg_cost = total_cost / total_samples
        remaining = total_target - total_samples
        est_remaining_cost = remaining * avg_cost
        print(f"📈 Estimates:")
        print(f"   Avg cost/sample: ${avg_cost:.6f}")
        print(f"   Remaining: {remaining:,} samples")
        print(f"   Est. remaining cost: ${est_remaining_cost:.2f}")
        print(f"   Est. total cost: ${total_cost + est_remaining_cost:.2f}")
        print()

    print("💡 Monitor logs:")
    print("   tail -f logs/truthfulqa_generation.log")
    print("   tail -f logs/fever_generation.log")
    print("   tail -f logs/halueval_generation.log")

if __name__ == "__main__":
    main()
