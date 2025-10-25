#!/usr/bin/env python3
"""
Generate comparison bar chart showing AUROC for r_LZ and perplexity
on factuality benchmarks vs structural degeneracy.

This figure demonstrates that r_LZ and perplexity are complementary:
- Perplexity excels on factuality tasks (TruthfulQA, FEVER, HaluEval)
- r_LZ achieves perfect detection on structural degeneracy
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from whitepaper Section 6.1 (Factuality) and 6.2 (Structural Degeneracy)
benchmarks = ['TruthfulQA\n(Factuality)', 'FEVER\n(Factuality)', 'HaluEval\n(Factuality)', 'Structural\nDegeneracy']

# AUROC values from whitepaper results
rlz_auroc = [0.535, 0.578, 0.498, 1.000]  # r_LZ performance
perplexity_auroc = [0.615, 0.598, 0.500, 0.018]  # Perplexity performance

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(benchmarks))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, rlz_auroc, width, label='ASV: $r_{LZ}$ (Compressibility)',
               color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, perplexity_auroc, width, label='Baseline: Perplexity',
               color='#A23B72', alpha=0.8)

# Add value labels on bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

autolabel(bars1)
autolabel(bars2)

# Customize plot
ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
ax.set_xlabel('Benchmark Task', fontsize=12, fontweight='bold')
ax.set_title('AUROC Comparison: Factuality vs. Structural Degeneracy\nASV and Perplexity are Complementary Tools',
             fontsize=13, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=10)
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random (0.50)')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add annotations
ax.text(0, 0.65, 'Perplexity\nwins', ha='center', fontsize=9, style='italic', color='#A23B72')
ax.text(1, 0.65, 'Perplexity\nwins', ha='center', fontsize=9, style='italic', color='#A23B72')
ax.text(2, 0.55, 'Tied\n(both ~0.50)', ha='center', fontsize=9, style='italic', color='gray')
ax.text(3, 0.5, 'r_LZ\nPERFECT', ha='center', fontsize=10, style='italic',
        color='#2E86AB', fontweight='bold')
ax.text(3, 0.1, 'Perplexity\nFAILS', ha='center', fontsize=9, style='italic',
        color='#A23B72', fontweight='bold')

# Add note
fig.text(0.5, 0.02,
         'Note: r_LZ (compressibility) detects structural degeneracy (loops, repetition), not factual errors.\n'
         'Perplexity detects unlikely/incorrect facts. Use both in production for comprehensive verification.',
         ha='center', fontsize=9, style='italic', wrap=True)

plt.tight_layout(rect=[0, 0.05, 1, 1])

# Save figure
output_path = '/Users/roman.khokhla/my_stuff/kakeya/docs/architecture/figures/comparison_bars.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Generated: {output_path}")
print(f"  - TruthfulQA (Factuality): r_LZ={rlz_auroc[0]:.3f}, Perplexity={perplexity_auroc[0]:.3f}")
print(f"  - FEVER (Factuality): r_LZ={rlz_auroc[1]:.3f}, Perplexity={perplexity_auroc[1]:.3f}")
print(f"  - HaluEval (Factuality): r_LZ={rlz_auroc[2]:.3f}, Perplexity={perplexity_auroc[2]:.3f}")
print(f"  - Structural Degeneracy: r_LZ={rlz_auroc[3]:.3f}, Perplexity={perplexity_auroc[3]:.3f}")
print("\n✓ Key finding: r_LZ achieves PERFECT detection (AUROC 1.000) on structural degeneracy")
print("✓ Perplexity completely fails (AUROC 0.018, worse than random) on structural degeneracy")
print("✓ Perplexity wins on factuality tasks (as expected)")
