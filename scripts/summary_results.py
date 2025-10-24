#!/usr/bin/env python3
"""
Display Summary of Evaluation Results

Shows best performing methods across all benchmarks with key metrics.
"""

import json
import sys
from pathlib import Path
from typing import Dict


def load_results(benchmark: str) -> Dict:
    """Load results for a benchmark."""
    results_file = Path(f"results/{benchmark}_results.json")
    if not results_file.exists():
        return None

    with open(results_file) as f:
        return json.load(f)


def print_benchmark_summary(benchmark: str, results: Dict):
    """Print summary for a single benchmark."""
    if not results:
        print(f"\n❌ {benchmark.upper()}: No results found")
        return

    print(f"\n{'='*70}")
    print(f"📊 {benchmark.upper()} RESULTS")
    print(f"{'='*70}")

    methods = results['methods']
    summary = results.get('summary', {})

    # Sort methods by AUROC
    sorted_methods = sorted(
        methods.items(),
        key=lambda x: x[1]['auroc'],
        reverse=True
    )

    print(f"\nBest AUROC: {summary.get('best_auroc', 'N/A'):.4f}")
    print(f"Best AUPRC: {summary.get('best_auprc', 'N/A'):.4f}")
    print(f"Best F1: {summary.get('best_f1', 'N/A'):.4f}")
    print(f"\nRanked by AUROC:")
    print(f"{'─'*70}")
    print(f"{'Method':<35} {'AUROC':<10} {'AUPRC':<10} {'F1':<10}")
    print(f"{'─'*70}")

    for name, metrics in sorted_methods:
        method_name = metrics['method_name']
        auroc = metrics['auroc']
        auprc = metrics['auprc']
        f1 = metrics['f1_optimal']

        # Highlight best method
        prefix = "⭐" if auroc == summary.get('best_auroc') else "  "

        print(f"{prefix} {method_name:<33} {auroc:<10.4f} {auprc:<10.4f} {f1:<10.4f}")


def print_overall_summary(all_results: Dict[str, Dict]):
    """Print overall summary across all benchmarks."""
    print(f"\n{'='*70}")
    print(f"📈 OVERALL SUMMARY")
    print(f"{'='*70}\n")

    for benchmark, results in all_results.items():
        if not results:
            continue

        methods = results['methods']
        summary = results.get('summary', {})

        print(f"{benchmark.upper():<15} Best AUROC: {summary.get('best_auroc', 0):.4f}  "
              f"Best AUPRC: {summary.get('best_auprc', 0):.4f}  "
              f"Best F1: {summary.get('best_f1', 0):.4f}")

    # Cross-benchmark comparison
    print(f"\n{'─'*70}")
    print("Method Performance Across Benchmarks (AUROC):")
    print(f"{'─'*70}")

    # Collect all method names
    all_method_names = set()
    for results in all_results.values():
        if results:
            all_method_names.update(results['methods'].keys())

    # Print each method's performance across benchmarks
    for method_key in sorted(all_method_names):
        method_name = None
        scores = []

        for benchmark in ['truthfulqa', 'fever', 'halueval']:
            results = all_results.get(benchmark)
            if results and method_key in results['methods']:
                if method_name is None:
                    method_name = results['methods'][method_key]['method_name']
                auroc = results['methods'][method_key]['auroc']
                scores.append(f"{auroc:.4f}")
            else:
                scores.append("  N/A  ")

        if method_name:
            print(f"{method_name:<35} TQ: {scores[0]}  FV: {scores[1]}  HE: {scores[2]}")


def main():
    """Main entry point."""
    benchmarks = ['truthfulqa', 'fever', 'halueval']

    # Load all results
    all_results = {}
    for benchmark in benchmarks:
        all_results[benchmark] = load_results(benchmark)

    # Print individual benchmark summaries
    for benchmark in benchmarks:
        print_benchmark_summary(benchmark, all_results[benchmark])

    # Print overall summary
    print_overall_summary(all_results)

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
