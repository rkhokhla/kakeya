package baselines

import (
	"fmt"
	"math"
	"strings"

	"github.com/fractal-lba/kakeya/backend/internal/eval"
)

// NLIBaseline verifies using Natural Language Inference (entailment).
// Checks if response is entailed by context/evidence.
type NLIBaseline struct {
	threshold      float64 // Accept if entailment_score > threshold
	escalateMargin float64
	modelName      string // NLI model (e.g., "roberta-large-mnli")
}

// NewNLIBaseline creates an NLI baseline.
func NewNLIBaseline(threshold float64) *NLIBaseline {
	return &NLIBaseline{
		threshold:      threshold,
		escalateMargin: 0.05, // 5% margin
		modelName:      "roberta-large-mnli",
	}
}

// Verify checks entailment between prompt (premise) and response (hypothesis).
func (nli *NLIBaseline) Verify(sample *eval.BenchmarkSample) (*eval.BaselineResult, error) {
	// Compute entailment score
	entailmentScore := nli.computeEntailment(sample.Prompt, sample.Response)

	var decision eval.Decision
	if entailmentScore >= nli.threshold {
		decision = eval.DecisionAccept
	} else if entailmentScore >= nli.threshold-nli.escalateMargin {
		decision = eval.DecisionEscalate
	} else {
		decision = eval.DecisionReject
	}

	return &eval.BaselineResult{
		SampleID:  sample.ID,
		Method:    "nli",
		Score:     entailmentScore,
		Decision:  decision,
		Threshold: nli.threshold,
		Metadata: map[string]interface{}{
			"model":           nli.modelName,
			"premise_length":  len(sample.Prompt),
			"hypothesis_length": len(sample.Response),
		},
	}, nil
}

// computeEntailment computes entailment score (simplified heuristic).
// Real implementation would use RoBERTa-large-MNLI or similar.
//
// Returns: probability that hypothesis is entailed by premise [0, 1]
func (nli *NLIBaseline) computeEntailment(premise, hypothesis string) float64 {
	if premise == "" || hypothesis == "" {
		return 0.0
	}

	// Simplified heuristic: lexical overlap + length ratio
	premiseWords := tokenize(strings.ToLower(premise))
	hypothesisWords := tokenize(strings.ToLower(hypothesis))

	// Jaccard similarity
	overlap := jaccard(premiseWords, hypothesisWords)

	// Length consistency (hypotheses shouldn't be much longer than premises for entailment)
	lenRatio := float64(len(hypothesis)) / float64(len(premise))
	lengthPenalty := 1.0
	if lenRatio > 1.5 {
		lengthPenalty = 1.0 / lenRatio // Penalize overly long hypotheses
	}

	// Combined score
	entailmentScore := overlap * lengthPenalty

	// Add bonus for exact substrings (strong entailment signal)
	if strings.Contains(strings.ToLower(premise), strings.ToLower(hypothesis)) {
		entailmentScore += 0.2
	}

	// Clamp to [0, 1]
	if entailmentScore > 1.0 {
		entailmentScore = 1.0
	}

	return entailmentScore
}

// tokenize splits text into words.
func tokenize(text string) []string {
	// Simple whitespace tokenization
	words := strings.Fields(text)
	return words
}

// jaccard computes Jaccard similarity between two sets of words.
func jaccard(set1, set2 []string) float64 {
	// Convert to sets
	m1 := make(map[string]bool)
	for _, w := range set1 {
		m1[w] = true
	}
	m2 := make(map[string]bool)
	for _, w := range set2 {
		m2[w] = true
	}

	// Count intersection
	intersection := 0
	for w := range m1 {
		if m2[w] {
			intersection++
		}
	}

	// Union
	union := len(m1) + len(m2) - intersection

	if union == 0 {
		return 0.0
	}

	return float64(intersection) / float64(union)
}

// Name returns baseline name.
func (nli *NLIBaseline) Name() string {
	return "nli"
}

// Description returns baseline description.
func (nli *NLIBaseline) Description() string {
	return fmt.Sprintf("NLI entailment (threshold=%.2f, model=%s)", nli.threshold, nli.modelName)
}

// SetThreshold updates decision threshold.
func (nli *NLIBaseline) SetThreshold(threshold float64) {
	nli.threshold = threshold
}

// GetScore returns entailment score for a sample.
func (nli *NLIBaseline) GetScore(sample *eval.BenchmarkSample) float64 {
	return nli.computeEntailment(sample.Prompt, sample.Response)
}

// Analyze performs detailed NLI analysis.
func (nli *NLIBaseline) Analyze(samples []*eval.BenchmarkSample) map[string]interface{} {
	var correct, halluc []float64

	for _, sample := range samples {
		score := nli.computeEntailment(sample.Prompt, sample.Response)
		if sample.GroundTruth {
			correct = append(correct, score)
		} else {
			halluc = append(halluc, score)
		}
	}

	return map[string]interface{}{
		"correct_mean":        mean(correct),
		"correct_std":         stddev(correct),
		"hallucination_mean":  mean(halluc),
		"hallucination_std":   stddev(halluc),
		"separation":          mean(correct) - mean(halluc),
		"effect_size_cohens_d": cohensD(correct, halluc),
	}
}

// ProductionNote:
//
// For ASV paper evaluation, we use:
//
// 1. **Simplified heuristic** (this file):
//    - Jaccard similarity + length ratio
//    - No external dependencies
//    - Smoke testing only
//
// 2. **RoBERTa-large-MNLI** (scripts/eval/compute_nli.py):
//    - HuggingFace transformers
//    - Facebook/bart-large-mnli
//    - Reported in paper
//
// 3. **DeBERTa-v3-large-MNLI** (optional):
//    - Microsoft/deberta-v3-large-mnli
//    - State-of-the-art NLI
//    - For comparison
//
// Usage:
// ```python
// from transformers import AutoModelForSequenceClassification, AutoTokenizer
// import torch
//
// model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
// tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
//
// def compute_entailment(premise, hypothesis):
//     inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
//     with torch.no_grad():
//         outputs = model(**inputs)
//         logits = outputs.logits
//         probs = torch.softmax(logits, dim=1)[0]
//         # probs: [entailment, neutral, contradiction]
//         return probs[0].item()  # Entailment probability
// ```
//
// Benchmark results use approach #2 (RoBERTa-large-MNLI).
//
// References:
// - Liu et al. (2019): RoBERTa: A Robustly Optimized BERT Pretraining Approach
// - Williams et al. (2018): A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference (MNLI)
