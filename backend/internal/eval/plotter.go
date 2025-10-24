package eval

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Plotter generates plots and visualizations for evaluation results.
type Plotter struct {
	outputDir string
}

// NewPlotter creates a new plotter.
func NewPlotter(outputDir string) *Plotter {
	return &Plotter{
		outputDir: outputDir,
	}
}

// PlotAll generates all plots for a comparison report.
func (p *Plotter) PlotAll(report *ComparisonReport) error {
	// Create output directory
	if err := os.MkdirAll(p.outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output dir: %w", err)
	}

	// 1. ROC curves
	if err := p.PlotROCCurves(report); err != nil {
		return fmt.Errorf("ROC plot failed: %w", err)
	}

	// 2. PR curves
	if err := p.PlotPRCurves(report); err != nil {
		return fmt.Errorf("PR plot failed: %w", err)
	}

	// 3. Calibration plots
	if err := p.PlotCalibration(report); err != nil {
		return fmt.Errorf("calibration plot failed: %w", err)
	}

	// 4. Confusion matrices
	if err := p.PlotConfusionMatrices(report); err != nil {
		return fmt.Errorf("confusion matrix plot failed: %w", err)
	}

	// 5. Performance comparison table
	if err := p.GeneratePerformanceTable(report); err != nil {
		return fmt.Errorf("performance table failed: %w", err)
	}

	// 6. Cost comparison plot
	if err := p.PlotCostComparison(report); err != nil {
		return fmt.Errorf("cost plot failed: %w", err)
	}

	// 7. Statistical tests table
	if err := p.GenerateStatisticalTestsTable(report); err != nil {
		return fmt.Errorf("statistical tests table failed: %w", err)
	}

	return nil
}

// PlotROCCurves generates ROC curve plot for all methods.
func (p *Plotter) PlotROCCurves(report *ComparisonReport) error {
	// Generate Python script for plotting
	script := `#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json

# Load data
with open('roc_data.json', 'r') as f:
    data = json.load(f)

plt.figure(figsize=(10, 8))

for method, curve in data.items():
    fpr = [pt['fpr'] for pt in curve]
    tpr = [pt['tpr'] for pt in curve]
    auc = data['aucs'][method]
    plt.plot(fpr, tpr, label=f'{method} (AUC={auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - ASV vs Baselines', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300)
plt.savefig('roc_curves.pdf')
print('Saved roc_curves.png')
`

	scriptPath := filepath.Join(p.outputDir, "plot_roc.py")
	if err := os.WriteFile(scriptPath, []byte(script), 0755); err != nil {
		return err
	}

	// Generate JSON data
	dataMap := make(map[string]interface{})
	aucs := make(map[string]float64)

	for method, metrics := range report.MethodMetrics {
		dataMap[method] = metrics.ROCCurve
		aucs[method] = metrics.AUC
	}
	dataMap["aucs"] = aucs

	// Write data file (placeholder - would use actual JSON marshaling)
	dataPath := filepath.Join(p.outputDir, "roc_data.json")
	fmt.Printf("ROC data would be written to: %s\n", dataPath)

	return nil
}

// PlotPRCurves generates precision-recall curve plot.
func (p *Plotter) PlotPRCurves(report *ComparisonReport) error {
	script := `#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json

with open('pr_data.json', 'r') as f:
    data = json.load(f)

plt.figure(figsize=(10, 8))

for method, curve in data.items():
    recall = [pt['recall'] for pt in curve]
    precision = [pt['precision'] for pt in curve]
    auprc = data['auprcs'][method]
    plt.plot(recall, precision, label=f'{method} (AUPRC={auprc:.3f})', linewidth=2)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pr_curves.png', dpi=300)
plt.savefig('pr_curves.pdf')
print('Saved pr_curves.png')
`

	scriptPath := filepath.Join(p.outputDir, "plot_pr.py")
	return os.WriteFile(scriptPath, []byte(script), 0755)
}

// PlotCalibration generates calibration plots (reliability diagrams).
func (p *Plotter) PlotCalibration(report *ComparisonReport) error {
	script := `#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json

with open('calibration_data.json', 'r') as f:
    data = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (method, metrics) in enumerate(data.items()):
    if idx >= 6:
        break

    ax = axes[idx]

    # Reliability diagram
    bin_centers = np.linspace(0.05, 0.95, 10)
    accuracies = metrics['bin_accuracies']  # Placeholder

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax.plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=8, label=f'ECE={metrics["ece"]:.3f}')
    ax.set_xlabel('Predicted Probability', fontsize=10)
    ax.set_ylabel('Observed Frequency', fontsize=10)
    ax.set_title(f'{method}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calibration_plots.png', dpi=300)
plt.savefig('calibration_plots.pdf')
print('Saved calibration_plots.png')
`

	scriptPath := filepath.Join(p.outputDir, "plot_calibration.py")
	return os.WriteFile(scriptPath, []byte(script), 0755)
}

// PlotConfusionMatrices generates confusion matrix heatmaps.
func (p *Plotter) PlotConfusionMatrices(report *ComparisonReport) error {
	script := `#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

with open('confusion_data.json', 'r') as f:
    data = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (method, metrics) in enumerate(data.items()):
    if idx >= 6:
        break

    ax = axes[idx]

    # Confusion matrix
    cm = np.array([
        [metrics['tp'], metrics['fn']],
        [metrics['fp'], metrics['tn']]
    ])

    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Predicted Pos', 'Predicted Neg'],
                yticklabels=['Actual Pos', 'Actual Neg'],
                ax=ax, cbar_kws={'label': 'Proportion'})
    ax.set_title(f'{method}', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300)
plt.savefig('confusion_matrices.pdf')
print('Saved confusion_matrices.png')
`

	scriptPath := filepath.Join(p.outputDir, "plot_confusion.py")
	return os.WriteFile(scriptPath, []byte(script), 0755)
}

// GeneratePerformanceTable generates a LaTeX/Markdown table of all metrics.
func (p *Plotter) GeneratePerformanceTable(report *ComparisonReport) error {
	var md strings.Builder

	// Header
	md.WriteString("# Performance Comparison Table\n\n")
	md.WriteString("| Method | Accuracy | Precision | Recall | F1 | AUC | AUPRC | ECE | Miscoverage | Escalation Rate |\n")
	md.WriteString("|--------|----------|-----------|--------|-------|------|-------|-----|-------------|----------------|\n")

	// Rows
	for _, method := range report.MethodNames {
		metrics := report.MethodMetrics[method]
		md.WriteString(fmt.Sprintf("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |\n",
			method,
			metrics.Accuracy,
			metrics.Precision,
			metrics.Recall,
			metrics.F1Score,
			metrics.AUC,
			metrics.AUPRC,
			metrics.ECE,
			metrics.Miscoverage,
			metrics.EscalationRate,
		))
	}

	md.WriteString("\n## Bootstrap Confidence Intervals (95%)\n\n")
	md.WriteString("| Method | Precision CI | Recall CI | F1 CI | Accuracy CI |\n")
	md.WriteString("|--------|--------------|-----------|-------|-------------|\n")

	for _, method := range report.MethodNames {
		metrics := report.MethodMetrics[method]
		ci := metrics.BootstrapCIs
		md.WriteString(fmt.Sprintf("| %s | [%.3f, %.3f] | [%.3f, %.3f] | [%.3f, %.3f] | [%.3f, %.3f] |\n",
			method,
			ci.PrecisionCI[0], ci.PrecisionCI[1],
			ci.RecallCI[0], ci.RecallCI[1],
			ci.F1CI[0], ci.F1CI[1],
			ci.AccuracyCI[0], ci.AccuracyCI[1],
		))
	}

	// LaTeX version
	md.WriteString("\n## LaTeX Table\n\n")
	md.WriteString("```latex\n")
	md.WriteString("\\begin{table}[h]\n")
	md.WriteString("\\centering\n")
	md.WriteString("\\caption{Performance comparison of ASV and baseline methods}\n")
	md.WriteString("\\begin{tabular}{lcccccccc}\n")
	md.WriteString("\\hline\n")
	md.WriteString("Method & Acc & Prec & Rec & F1 & AUC & ECE & Misc & Esc \\\\\n")
	md.WriteString("\\hline\n")

	for _, method := range report.MethodNames {
		metrics := report.MethodMetrics[method]
		md.WriteString(fmt.Sprintf("%s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\\n",
			method,
			metrics.Accuracy,
			metrics.Precision,
			metrics.Recall,
			metrics.F1Score,
			metrics.AUC,
			metrics.ECE,
			metrics.Miscoverage,
			metrics.EscalationRate,
		))
	}

	md.WriteString("\\hline\n")
	md.WriteString("\\end{tabular}\n")
	md.WriteString("\\end{table}\n")
	md.WriteString("```\n")

	tablePath := filepath.Join(p.outputDir, "performance_table.md")
	return os.WriteFile(tablePath, []byte(md.String()), 0644)
}

// PlotCostComparison generates cost comparison bar plot.
func (p *Plotter) PlotCostComparison(report *ComparisonReport) error {
	script := `#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json

with open('cost_data.json', 'r') as f:
    data = json.load(f)

methods = list(data['method_costs'].keys())
costs = [data['method_costs'][m] for m in methods]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Cost per verification
costs_per_verif = [data['total_cost'][m] for m in methods]
ax1.bar(methods, costs_per_verif, color='steelblue', alpha=0.7)
ax1.set_ylabel('Cost per Verification ($)', fontsize=12)
ax1.set_title('Cost per Verification', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, axis='y')
ax1.tick_params(axis='x', rotation=45)

# Cost per trusted task
ax2.bar(methods, costs, color='coral', alpha=0.7)
ax2.set_ylabel('Cost per Trusted Task ($)', fontsize=12)
ax2.set_title('Cost per Trusted Task', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='x', rotation=45)

# Highlight most cost-effective
most_cost_effective = data['most_cost_effective']
for i, method in enumerate(methods):
    if method == most_cost_effective:
        ax2.get_children()[i].set_color('green')
        ax2.get_children()[i].set_alpha(0.9)

plt.tight_layout()
plt.savefig('cost_comparison.png', dpi=300)
plt.savefig('cost_comparison.pdf')
print('Saved cost_comparison.png')
`

	scriptPath := filepath.Join(p.outputDir, "plot_cost.py")
	return os.WriteFile(scriptPath, []byte(script), 0755)
}

// GenerateStatisticalTestsTable generates table of statistical test results.
func (p *Plotter) GenerateStatisticalTestsTable(report *ComparisonReport) error {
	var md strings.Builder

	md.WriteString("# Statistical Significance Tests\n\n")
	md.WriteString("## McNemar's Test (Paired Binary Outcomes)\n\n")
	md.WriteString("| Comparison | Chi-Squared | p-value | Significant | Effect Size |\n")
	md.WriteString("|------------|-------------|---------|-------------|-------------|\n")

	for testKey, test := range report.StatisticalTests {
		if !strings.Contains(test.TestName, "McNemar") {
			continue
		}

		sigStr := "✗"
		if test.Significant {
			sigStr = "✓"
		}

		md.WriteString(fmt.Sprintf("| %s | %.3f | %.4f | %s | %.3f |\n",
			testKey,
			test.TestStatistic,
			test.PValue,
			sigStr,
			test.EffectSize,
		))
	}

	md.WriteString("\n## Permutation Test (Accuracy Difference)\n\n")
	md.WriteString("| Comparison | Observed Diff | p-value | Significant | Effect Size |\n")
	md.WriteString("|------------|---------------|---------|-------------|-------------|\n")

	for testKey, test := range report.StatisticalTests {
		if !strings.Contains(test.TestName, "Permutation") {
			continue
		}

		sigStr := "✗"
		if test.Significant {
			sigStr = "✓"
		}

		md.WriteString(fmt.Sprintf("| %s | %.4f | %.4f | %s | %.4f |\n",
			testKey,
			test.TestStatistic,
			test.PValue,
			sigStr,
			test.EffectSize,
		))
	}

	md.WriteString("\n**Interpretation:**\n")
	md.WriteString("- ✓ Significant: p < 0.05 (reject null hypothesis)\n")
	md.WriteString("- ✗ Not significant: p ≥ 0.05 (fail to reject null)\n")
	md.WriteString("- Effect size: positive favors first method, negative favors second\n")

	tablePath := filepath.Join(p.outputDir, "statistical_tests.md")
	return os.WriteFile(tablePath, []byte(md.String()), 0644)
}

// GenerateSummaryReport generates comprehensive summary report.
func (p *Plotter) GenerateSummaryReport(report *ComparisonReport) error {
	var md strings.Builder

	md.WriteString("# ASV Evaluation Report\n\n")
	md.WriteString(fmt.Sprintf("**Generated:** %s\n\n", report.Timestamp.Format("2006-01-02 15:04:05")))
	md.WriteString(fmt.Sprintf("**Target Miscoverage (δ):** %.1f%%\n\n", report.TargetDelta*100))

	md.WriteString("## Executive Summary\n\n")
	md.WriteString(report.Summary)
	md.WriteString("\n\n")

	md.WriteString("## Key Findings\n\n")

	// Find best method for each metric
	bestAcc := ""
	maxAcc := 0.0
	for method, metrics := range report.MethodMetrics {
		if metrics.Accuracy > maxAcc {
			maxAcc = metrics.Accuracy
			bestAcc = method
		}
	}

	bestAUC := ""
	maxAUC := 0.0
	for method, metrics := range report.MethodMetrics {
		if metrics.AUC > maxAUC {
			maxAUC = metrics.AUC
			bestAUC = method
		}
	}

	bestCal := ""
	minECE := 1.0
	for method, metrics := range report.MethodMetrics {
		if metrics.ECE < minECE {
			minECE = metrics.ECE
			bestCal = method
		}
	}

	md.WriteString(fmt.Sprintf("- **Best Accuracy:** %s (%.3f)\n", bestAcc, maxAcc))
	md.WriteString(fmt.Sprintf("- **Best AUC:** %s (%.3f)\n", bestAUC, maxAUC))
	md.WriteString(fmt.Sprintf("- **Best Calibration:** %s (ECE=%.3f)\n", bestCal, minECE))
	md.WriteString(fmt.Sprintf("- **Most Cost-Effective:** %s\n\n", report.CostComparison.MostCostEffective))

	md.WriteString("## Detailed Results\n\n")
	md.WriteString("See:\n")
	md.WriteString("- `performance_table.md` for full metrics\n")
	md.WriteString("- `statistical_tests.md` for significance tests\n")
	md.WriteString("- `roc_curves.png` for ROC analysis\n")
	md.WriteString("- `pr_curves.png` for precision-recall analysis\n")
	md.WriteString("- `calibration_plots.png` for calibration analysis\n")
	md.WriteString("- `confusion_matrices.png` for error analysis\n")
	md.WriteString("- `cost_comparison.png` for cost analysis\n\n")

	reportPath := filepath.Join(p.outputDir, "SUMMARY.md")
	return os.WriteFile(reportPath, []byte(md.String()), 0644)
}
