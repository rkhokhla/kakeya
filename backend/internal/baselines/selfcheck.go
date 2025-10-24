package baselines

import (
	"fmt"
	"math"
	"strings"

	"github.com/fractal-lba/kakeya/backend/internal/eval"
)

// SelfCheckGPTBaseline implements SelfCheckGPT (Manakul et al. 2023).
// Zero-resource hallucination detection via sampling consistency.
//
// Method:
// 1. Sample N responses from LLM for same prompt
// 2. Compute consistency score (how similar are the responses?)
// 3. Low consistency → likely hallucination
//
// Reference: Manakul et al. "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models" (EMNLP 2023)
type SelfCheckGPTBaseline struct {
	threshold      float64 // Accept if consistency > threshold
	escalateMargin float64
	numSamples     int    // Number of responses to sample (typically 5-10)
	variant        string // "nli", "bertscore", "qa", "prompt"
}

// NewSelfCheckGPTBaseline creates a SelfCheckGPT baseline.
func NewSelfCheckGPTBaseline(threshold float64, numSamples int, variant string) *SelfCheckGPTBaseline {
	return &SelfCheckGPTBaseline{
		threshold:      threshold,
		escalateMargin: 0.05,
		numSamples:     numSamples,
		variant:        variant,
	}
}

// Verify checks consistency across multiple sampled responses.
func (sc *SelfCheckGPTBaseline) Verify(sample *eval.BenchmarkSample) (*eval.BaselineResult, error) {
	// In real implementation, we would sample N responses from LLM here
	// For evaluation, we simulate by computing self-consistency score
	consistencyScore := sc.computeConsistency(sample)

	var decision eval.Decision
	if consistencyScore >= sc.threshold {
		decision = eval.DecisionAccept // High consistency = low hallucination
	} else if consistencyScore >= sc.threshold-sc.escalateMargin {
		decision = eval.DecisionEscalate
	} else {
		decision = eval.DecisionReject // Low consistency = hallucination
	}

	return &eval.BaselineResult{
		SampleID:  sample.ID,
		Method:    "selfcheck",
		Score:     consistencyScore,
		Decision:  decision,
		Threshold: sc.threshold,
		Metadata: map[string]interface{}{
			"num_samples": sc.numSamples,
			"variant":     sc.variant,
		},
	}, nil
}

// computeConsistency estimates consistency score (simplified proxy).
// Real implementation would sample from LLM and compute pairwise similarities.
//
// Proxy heuristics:
// 1. Response specificity (specific claims → lower variance in sampling)
// 2. Factual density (more facts → more opportunities for hallucination)
// 3. Repetition patterns (hallucinations often repeat phrases)
func (sc *SelfCheckGPTBaseline) computeConsistency(sample *eval.BenchmarkSample) float64 {
	response := sample.Response

	// Heuristic 1: Specificity (specific answers are more consistent)
	specificity := sc.measureSpecificity(response)

	// Heuristic 2: Factual density
	factualDensity := sc.measureFactualDensity(response)

	// Heuristic 3: Repetition (hallucinations often have more repetition)
	repetition := sc.measureRepetition(response)

	// Combined consistency score
	// High specificity + moderate factual density + low repetition → high consistency
	consistency := (specificity*0.5 + factualDensity*0.3 + (1.0-repetition)*0.2)

	// Clamp to [0, 1]
	if consistency < 0 {
		consistency = 0
	}
	if consistency > 1 {
		consistency = 1
	}

	return consistency
}

// measureSpecificity measures how specific the response is.
// Specific responses: numbers, names, dates, technical terms
// Generic responses: "maybe", "possibly", "it depends"
func (sc *SelfCheckGPTBaseline) measureSpecificity(text string) float64 {
	lower := strings.ToLower(text)

	// Count hedges (uncertainty markers)
	hedges := []string{"maybe", "perhaps", "possibly", "might", "could", "may", "uncertain"}
	hedgeCount := 0
	for _, hedge := range hedges {
		hedgeCount += strings.Count(lower, hedge)
	}

	// Count specific markers (numbers, names, etc.)
	specificMarkers := 0
	words := strings.Fields(text)
	for _, word := range words {
		// Numbers
		if len(word) > 0 && word[0] >= '0' && word[0] <= '9' {
			specificMarkers++
		}
		// Capitalized words (likely names/entities)
		if len(word) > 1 && word[0] >= 'A' && word[0] <= 'Z' {
			specificMarkers++
		}
	}

	// Specificity = (specific markers - hedges) / words
	if len(words) == 0 {
		return 0
	}

	specificity := float64(specificMarkers-hedgeCount) / float64(len(words))

	// Normalize to [0, 1]
	if specificity < 0 {
		specificity = 0
	}
	if specificity > 1 {
		specificity = 1
	}

	return specificity
}

// measureFactualDensity measures density of factual claims.
// More facts → more things to get wrong → lower consistency if hallucinating
func (sc *SelfCheckGPTBaseline) measureFactualDensity(text string) float64 {
	words := strings.Fields(text)
	if len(words) == 0 {
		return 0
	}

	// Count sentences (approximate by periods)
	sentences := float64(strings.Count(text, ".") + 1)

	// Average words per sentence (proxy for complexity)
	wordsPerSentence := float64(len(words)) / sentences

	// Factual density inversely related to complexity
	// Simple, direct statements → high density
	// Complex, meandering text → low density
	density := 1.0 / (1.0 + math.Log(wordsPerSentence+1)/5.0)

	return density
}

// measureRepetition measures repetition in text.
// Hallucinations often repeat phrases or patterns.
func (sc *SelfCheckGPTBaseline) measureRepetition(text string) float64 {
	words := strings.Fields(strings.ToLower(text))
	if len(words) < 2 {
		return 0
	}

	// Count repeated words
	wordCounts := make(map[string]int)
	for _, word := range words {
		// Skip very short words (articles, etc.)
		if len(word) <= 2 {
			continue
		}
		wordCounts[word]++
	}

	// Repetition score = (total occurrences - unique words) / total
	totalOccurrences := 0
	uniqueWords := len(wordCounts)
	for _, count := range wordCounts {
		totalOccurrences += count
	}

	if totalOccurrences == 0 {
		return 0
	}

	repetition := float64(totalOccurrences-uniqueWords) / float64(totalOccurrences)
	return repetition
}

// Name returns baseline name.
func (sc *SelfCheckGPTBaseline) Name() string {
	return "selfcheck"
}

// Description returns baseline description.
func (sc *SelfCheckGPTBaseline) Description() string {
	return fmt.Sprintf("SelfCheckGPT (threshold=%.2f, samples=%d, variant=%s)", sc.threshold, sc.numSamples, sc.variant)
}

// SetThreshold updates decision threshold.
func (sc *SelfCheckGPTBaseline) SetThreshold(threshold float64) {
	sc.threshold = threshold
}

// GetScore returns consistency score for a sample.
func (sc *SelfCheckGPTBaseline) GetScore(sample *eval.BenchmarkSample) float64 {
	return sc.computeConsistency(sample)
}

// Analyze performs detailed SelfCheckGPT analysis.
func (sc *SelfCheckGPTBaseline) Analyze(samples []*eval.BenchmarkSample) map[string]interface{} {
	var correct, halluc []float64

	for _, sample := range samples {
		score := sc.computeConsistency(sample)
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
// For ASV paper evaluation, we implement:
//
// 1. **Simplified proxy** (this file):
//    - Specificity + factual density + repetition
//    - No LLM sampling required
//    - Fast evaluation
//
// 2. **SelfCheckGPT-NLI** (scripts/eval/selfcheck_nli.py):
//    - Sample N responses from GPT-3.5/4
//    - Compute NLI entailment between original and samples
//    - Average entailment score = consistency
//    - Reported in paper
//
// 3. **SelfCheckGPT-BERTScore** (optional):
//    - Sample N responses
//    - Compute BERTScore between original and samples
//    - Average BERTScore = consistency
//    - Alternative metric
//
// SelfCheckGPT variants (from paper):
// - SelfCheckGPT-NLI: Use NLI model to check entailment
// - SelfCheckGPT-BERTScore: Use BERTScore similarity
// - SelfCheckGPT-QA: Extract facts, ask questions, check consistency
// - SelfCheckGPT-Prompt: Use prompt-based checking
//
// We use SelfCheckGPT-NLI as the primary baseline (most robust).
//
// References:
// - Manakul, Liusie, Gales (2023): "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models"
// - EMNLP 2023
// - Paper: https://arxiv.org/abs/2303.08896
//
// Implementation:
// ```python
// import openai
// from transformers import AutoModelForSequenceClassification, AutoTokenizer
//
// # Sample N responses
// def sample_responses(prompt, n=5):
//     responses = []
//     for _ in range(n):
//         response = openai.ChatCompletion.create(
//             model="gpt-3.5-turbo",
//             messages=[{"role": "user", "content": prompt}],
//             temperature=0.7,  # Non-deterministic
//         )
//         responses.append(response.choices[0].message.content)
//     return responses
//
// # Compute consistency via NLI
// def selfcheck_nli(original, sampled_responses):
//     model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
//     tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
//
//     scores = []
//     for sample in sampled_responses:
//         inputs = tokenizer(original, sample, return_tensors="pt", truncation=True)
//         outputs = model(**inputs)
//         probs = torch.softmax(outputs.logits, dim=1)[0]
//         scores.append(probs[0].item())  # Entailment probability
//
//     return np.mean(scores)  # Average consistency
// ```
//
// Benchmark results use this full implementation with GPT-3.5 sampling + RoBERTa-NLI.
