#!/usr/bin/env python3
"""Monitor signal computation progress."""

import json
import os
from pathlib import Path

def main():
    print("=== Signal Computation Progress Monitor ===\n")

    benchmarks = [
        ("truthfulqa", 790),
        ("fever", 2500),
        ("halueval", 5000),
    ]

    total_target = sum(target for _, target in benchmarks)
    total_completed = 0

    for name, target in benchmarks:
        # Count completed signals
        signal_dir = f"data/signals/{name}"
        if os.path.exists(signal_dir):
            completed = len(list(Path(signal_dir).glob("*.json")))
        else:
            completed = 0

        if completed > 0:
            progress = (completed / target) * 100
            print(f"📊 {name.upper()}:")
            print(f"   Completed: {completed:,} / {target:,} samples")
            print(f"   Progress: {progress:.1f}%")

            # Load checkpoint for stats
            checkpoint_file = f"data/checkpoints/{name}_signals.json"
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
    print("   tail -f logs/truthfulqa_signals.log")
    print("   tail -f logs/fever_signals.log")
    print("   tail -f logs/halueval_signals.log")

if __name__ == "__main__":
    main()
