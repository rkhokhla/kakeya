package baselines

import (
	"fmt"
	"math"
	"strings"

	"github.com/fractal-lba/kakeya/backend/internal/eval"
)

// PerplexityBaseline verifies using perplexity thresholding.
// High perplexity → high uncertainty → likely hallucination.
type PerplexityBaseline struct {
	threshold      float64 // Accept if perplexity < threshold
	escalateMargin float64 // Escalate if within margin of threshold
	modelName      string  // Model used for perplexity (e.g., "gpt2")
}

// NewPerplexityBaseline creates a perplexity baseline.
func NewPerplexityBaseline(threshold float64) *PerplexityBaseline {
	return &PerplexityBaseline{
		threshold:      threshold,
		escalateMargin: threshold * 0.1, // 10% margin for escalation
		modelName:      "gpt2",           // Default model
	}
}

// Verify computes perplexity and makes decision.
func (pb *PerplexityBaseline) Verify(sample *eval.BenchmarkSample) (*eval.BaselineResult, error) {
	// Compute perplexity (simplified: use response length and character entropy)
	ppl := pb.computePerplexity(sample.Response)

	var decision eval.Decision
	if ppl <= pb.threshold {
		decision = eval.DecisionAccept
	} else if ppl <= pb.threshold+pb.escalateMargin {
		decision = eval.DecisionEscalate
	} else {
		decision = eval.DecisionReject
	}

	return &eval.BaselineResult{
		SampleID:  sample.ID,
		Method:    "perplexity",
		Score:     ppl,
		Decision:  decision,
		Threshold: pb.threshold,
		Metadata: map[string]interface{}{
			"model":           pb.modelName,
			"response_length": len(sample.Response),
			"escalate_margin": pb.escalateMargin,
		},
	}, nil
}

// computePerplexity computes simplified perplexity estimate.
// Real implementation would use a language model (GPT-2, etc.).
// This is a proxy based on character-level entropy.
func (pb *PerplexityBaseline) computePerplexity(text string) float64 {
	if len(text) == 0 {
		return 100.0 // High perplexity for empty text
	}

	// Character frequency distribution
	freq := make(map[rune]int)
	total := 0
	for _, r := range text {
		freq[r]++
		total++
	}

	// Compute entropy
	entropy := 0.0
	for _, count := range freq {
		p := float64(count) / float64(total)
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}

	// Perplexity = 2^entropy
	ppl := math.Pow(2, entropy)

	// Normalize by text length (longer text → higher perplexity typically)
	lengthFactor := 1.0 + math.Log(float64(len(text))/100.0)
	if lengthFactor < 1.0 {
		lengthFactor = 1.0
	}

	return ppl * lengthFactor
}

// Name returns baseline name.
func (pb *PerplexityBaseline) Name() string {
	return "perplexity"
}

// Description returns baseline description.
func (pb *PerplexityBaseline) Description() string {
	return fmt.Sprintf("Perplexity thresholding (threshold=%.2f, model=%s)", pb.threshold, pb.modelName)
}

// SetThreshold updates the decision threshold.
func (pb *PerplexityBaseline) SetThreshold(threshold float64) {
	pb.threshold = threshold
	pb.escalateMargin = threshold * 0.1
}

// GetScore returns the raw score for a sample (for ROC curve analysis).
func (pb *PerplexityBaseline) GetScore(sample *eval.BenchmarkSample) float64 {
	return pb.computePerplexity(sample.Response)
}

// Note: In production, this would use a real language model API:
// - OpenAI's tokenizer + log probabilities
// - HuggingFace transformers (GPT-2, LLaMA, etc.)
// - Local model deployment with TorchServe/ONNX
//
// Example with GPT-2:
// ```python
// from transformers import GPT2LMHeadModel, GPT2Tokenizer
// model = GPT2LMHeadModel.from_pretrained('gpt2')
// tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
// tokens = tokenizer.encode(text, return_tensors='pt')
// with torch.no_grad():
//     outputs = model(tokens, labels=tokens)
//     loss = outputs.loss
//     perplexity = torch.exp(loss).item()
// ```
//
// For ASV evaluation, we provide:
// 1. This simplified proxy (character entropy)
// 2. Integration points for real LM APIs
// 3. Scripts for batch perplexity computation
//
// See: scripts/eval/compute_perplexity.py for full implementation.

// ComputePerplexityBatch computes perplexity for multiple samples.
// In production, this would batch API calls for efficiency.
func ComputePerplexityBatch(samples []*eval.BenchmarkSample, pb *PerplexityBaseline) ([]float64, error) {
	scores := make([]float64, len(samples))
	for i, sample := range samples {
		scores[i] = pb.computePerplexity(sample.Response)
	}
	return scores, nil
}

// EstimateThreshold estimates optimal threshold from calibration data.
func EstimateThreshold(samples []*eval.BenchmarkSample, pb *PerplexityBaseline, targetFPR float64) float64 {
	// Compute perplexity for all samples
	scores := make([]float64, len(samples))
	for i, sample := range samples {
		scores[i] = pb.computePerplexity(sample.Response)
	}

	// Sort scores
	sortScores(scores)

	// Find threshold that achieves target FPR
	// For perplexity: high score = hallucination, so threshold is upper bound
	idx := int(float64(len(scores)) * (1.0 - targetFPR))
	if idx >= len(scores) {
		idx = len(scores) - 1
	}

	return scores[idx]
}

// sortScores sorts float64 slice in place.
func sortScores(scores []float64) {
	// Simple bubble sort for small arrays
	n := len(scores)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if scores[j] > scores[j+1] {
				scores[j], scores[j+1] = scores[j+1], scores[j]
			}
		}
	}
}

// Analyze performs detailed analysis of perplexity distribution.
func (pb *PerplexityBaseline) Analyze(samples []*eval.BenchmarkSample) map[string]interface{} {
	var correct, halluc []float64

	for _, sample := range samples {
		ppl := pb.computePerplexity(sample.Response)
		if sample.GroundTruth {
			correct = append(correct, ppl)
		} else {
			halluc = append(halluc, ppl)
		}
	}

	return map[string]interface{}{
		"correct_mean":        mean(correct),
		"correct_std":         stddev(correct),
		"hallucination_mean":  mean(halluc),
		"hallucination_std":   stddev(halluc),
		"separation":          mean(halluc) - mean(correct),
		"effect_size_cohens_d": cohensD(correct, halluc),
		"num_correct":         len(correct),
		"num_hallucination":   len(halluc),
	}
}

// Helper functions for statistics.
func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func stddev(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	m := mean(values)
	variance := 0.0
	for _, v := range values {
		diff := v - m
		variance += diff * diff
	}
	return math.Sqrt(variance / float64(len(values)))
}

func cohensD(group1, group2 []float64) float64 {
	m1 := mean(group1)
	m2 := mean(group2)
	s1 := stddev(group1)
	s2 := stddev(group2)
	n1 := float64(len(group1))
	n2 := float64(len(group2))

	// Pooled standard deviation
	pooledSD := math.Sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1 + n2 - 2))

	if pooledSD == 0 {
		return 0
	}

	return (m1 - m2) / pooledSD
}

// ProductionNote: For evaluation in the ASV paper, we use:
//
// 1. **Simplified proxy** (this implementation):
//    - Fast, no external dependencies
//    - Character-level entropy
//    - Good for smoke testing
//
// 2. **GPT-2 perplexity** (scripts/eval/compute_perplexity.py):
//    - HuggingFace transformers
//    - Actual perplexity scores
//    - Reported in paper
//
// 3. **Model-specific perplexity** (if evaluating specific model):
//    - Use the same model that generated the response
//    - E.g., GPT-4 logprobs API
//    - Most accurate for that model
//
// Benchmark results will use approach #2 (GPT-2) unless otherwise noted.
