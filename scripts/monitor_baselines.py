#!/usr/bin/env python3
"""Monitor baseline computation progress."""

import json
import os
from pathlib import Path

def main():
    print("=== Baseline Computation Progress Monitor ===\n")

    benchmarks = [
        ("truthfulqa", 790),
        ("fever", 2500),
        ("halueval", 5000),
    ]

    total_target = sum(target for _, target in benchmarks)
    total_completed = 0

    for name, target in benchmarks:
        # Count completed baselines
        baseline_dir = f"data/baselines/{name}"
        if os.path.exists(baseline_dir):
            completed = len(list(Path(baseline_dir).glob("*.json")))
        else:
            completed = 0

        if completed > 0:
            progress = (completed / target) * 100
            print(f"📊 {name.upper()}:")
            print(f"   Completed: {completed:,} / {target:,} samples")
            print(f"   Progress: {progress:.1f}%")

            # Load checkpoint for stats
            checkpoint_file = f"data/checkpoints/{name}_baselines.json"
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file) as f:
                    checkpoint = json.load(f)
                    stats = checkpoint.get('stats', {})
                    failed = stats.get('failed', 0)
                    if failed > 0:
                        print(f"   Failed: {failed}")
            print()

            total_completed += completed

    if total_completed > 0:
        print("=== Total Summary ===")
        print(f"Total Completed: {total_completed:,} / {total_target:,} samples")
        print(f"Overall Progress: {(total_completed/total_target)*100:.1f}%")
        print()

    print("💡 Monitor logs:")
    print("   tail -f logs/fever_baselines.log")
    print("   tail -f logs/halueval_baselines.log")

if __name__ == "__main__":
    main()
