package eval

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// BenchmarkLoader loads benchmark datasets.
type BenchmarkLoader struct {
	dataDir string
}

// NewBenchmarkLoader creates a new benchmark loader.
func NewBenchmarkLoader(dataDir string) *BenchmarkLoader {
	return &BenchmarkLoader{dataDir: dataDir}
}

// LoadTruthfulQA loads TruthfulQA benchmark (817 questions).
// Format: CSV with columns: Type, Category, Question, Best Answer, Correct Answers, Incorrect Answers
func (bl *BenchmarkLoader) LoadTruthfulQA() ([]BenchmarkSample, error) {
	path := filepath.Join(bl.dataDir, "truthfulqa", "TruthfulQA.csv")
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open TruthfulQA: %w", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	header, err := reader.Read() // Skip header
	if err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	// Map header to indices
	colIdx := make(map[string]int)
	for i, col := range header {
		colIdx[col] = i
	}

	samples := []BenchmarkSample{}
	id := 1

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read row: %w", err)
		}

		question := record[colIdx["Question"]]
		bestAnswer := record[colIdx["Best Answer"]]
		category := record[colIdx["Category"]]
		qtype := record[colIdx["Type"]]

		// TruthfulQA provides best answer as ground truth
		samples = append(samples, BenchmarkSample{
			ID:          fmt.Sprintf("truthfulqa_%d", id),
			Prompt:      question,
			Response:    bestAnswer, // Ground truth answer
			GroundTruth: true,       // Best answer is correct
			Source:      "truthfulqa",
			Category:    category,
			Metadata: map[string]interface{}{
				"type":             qtype,
				"correct_answers":  record[colIdx["Correct Answers"]],
				"incorrect_answers": record[colIdx["Incorrect Answers"]],
			},
		})
		id++
	}

	return samples, nil
}

// LoadFEVER loads FEVER benchmark (185k claims, use dev set ~20k for evaluation).
// Format: JSONL with fields: id, claim, label (SUPPORTS, REFUTES, NOT ENOUGH INFO)
func (bl *BenchmarkLoader) LoadFEVER(maxSamples int) ([]BenchmarkSample, error) {
	// Use dev set for evaluation (paper.json or shared_task_dev.jsonl)
	path := filepath.Join(bl.dataDir, "fever", "shared_task_dev.jsonl")
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open FEVER: %w", err)
	}
	defer f.Close()

	samples := []BenchmarkSample{}
	scanner := bufio.NewScanner(f)
	id := 1

	for scanner.Scan() && (maxSamples == 0 || id <= maxSamples) {
		var record struct {
			ID        int    `json:"id"`
			Claim     string `json:"claim"`
			Label     string `json:"label"`
			Evidence  interface{} `json:"evidence"`
		}

		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			continue // Skip malformed lines
		}

		// Ground truth: SUPPORTS = true, REFUTES/NEI = false (hallucination)
		groundTruth := record.Label == "SUPPORTS"

		samples = append(samples, BenchmarkSample{
			ID:          fmt.Sprintf("fever_%d", record.ID),
			Prompt:      "",           // No prompt in FEVER
			Response:    record.Claim, // Claim to verify
			GroundTruth: groundTruth,
			Source:      "fever",
			Category:    record.Label,
			Metadata: map[string]interface{}{
				"label":    record.Label,
				"evidence": record.Evidence,
			},
		})
		id++
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scanner error: %w", err)
	}

	return samples, nil
}

// LoadHaluEval loads HaluEval benchmark (5k samples).
// Format: JSON with multiple tasks (QA, Dialogue, Summarization)
func (bl *BenchmarkLoader) LoadHaluEval() ([]BenchmarkSample, error) {
	tasks := []string{"qa", "dialogue", "summarization"}
	samples := []BenchmarkSample{}
	id := 1

	for _, task := range tasks {
		path := filepath.Join(bl.dataDir, "halueval", fmt.Sprintf("%s_samples.json", task))
		f, err := os.Open(path)
		if err != nil {
			// Skip if file doesn't exist
			continue
		}

		var records []struct {
			Question      string `json:"question"`
			Answer        string `json:"answer"`
			GroundTruth   string `json:"right_answer"`
			Hallucination bool   `json:"hallucination"`
			KnowledgeType string `json:"knowledge_type"`
		}

		if err := json.NewDecoder(f).Decode(&records); err != nil {
			f.Close()
			return nil, fmt.Errorf("failed to decode %s: %w", task, err)
		}
		f.Close()

		for _, r := range records {
			samples = append(samples, BenchmarkSample{
				ID:          fmt.Sprintf("halueval_%s_%d", task, id),
				Prompt:      r.Question,
				Response:    r.Answer,
				GroundTruth: !r.Hallucination, // True if no hallucination
				Source:      "halueval",
				Category:    task,
				Metadata: map[string]interface{}{
					"knowledge_type": r.KnowledgeType,
					"ground_truth":   r.GroundTruth,
				},
			})
			id++
		}
	}

	if len(samples) == 0 {
		return nil, fmt.Errorf("no HaluEval samples loaded")
	}

	return samples, nil
}

// LoadHalluLens loads HalluLens benchmark (ACL 2025).
// Format: JSONL with unified taxonomy
func (bl *BenchmarkLoader) LoadHalluLens(maxSamples int) ([]BenchmarkSample, error) {
	path := filepath.Join(bl.dataDir, "hallulens", "hallulens_data.jsonl")
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open HalluLens: %w", err)
	}
	defer f.Close()

	samples := []BenchmarkSample{}
	scanner := bufio.NewScanner(f)
	id := 1

	for scanner.Scan() && (maxSamples == 0 || id <= maxSamples) {
		var record struct {
			ID              string   `json:"id"`
			Task            string   `json:"task"`
			Prompt          string   `json:"prompt"`
			Response        string   `json:"response"`
			HallucinationType string `json:"hallucination_type"`
			Hallucinated    bool     `json:"hallucinated"`
			Category        string   `json:"category"`
		}

		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			continue
		}

		samples = append(samples, BenchmarkSample{
			ID:          fmt.Sprintf("hallulens_%s", record.ID),
			Prompt:      record.Prompt,
			Response:    record.Response,
			GroundTruth: !record.Hallucinated,
			Source:      "hallulens",
			Category:    record.Category,
			Metadata: map[string]interface{}{
				"task":                record.Task,
				"hallucination_type":  record.HallucinationType,
				"hallucinated":        record.Hallucinated,
			},
		})
		id++
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scanner error: %w", err)
	}

	return samples, nil
}

// LoadAllBenchmarks loads all benchmarks with optional limits.
func (bl *BenchmarkLoader) LoadAllBenchmarks(limits map[string]int) (map[string][]BenchmarkSample, error) {
	benchmarks := make(map[string][]BenchmarkSample)

	// TruthfulQA (full dataset ~817)
	if samples, err := bl.LoadTruthfulQA(); err == nil {
		if limit, ok := limits["truthfulqa"]; ok && limit > 0 && limit < len(samples) {
			samples = samples[:limit]
		}
		benchmarks["truthfulqa"] = samples
	} else {
		fmt.Printf("Warning: Failed to load TruthfulQA: %v\n", err)
	}

	// FEVER (use dev set, ~20k samples)
	if samples, err := bl.LoadFEVER(limits["fever"]); err == nil {
		benchmarks["fever"] = samples
	} else {
		fmt.Printf("Warning: Failed to load FEVER: %v\n", err)
	}

	// HaluEval (~5k samples)
	if samples, err := bl.LoadHaluEval(); err == nil {
		if limit, ok := limits["halueval"]; ok && limit > 0 && limit < len(samples) {
			samples = samples[:limit]
		}
		benchmarks["halueval"] = samples
	} else {
		fmt.Printf("Warning: Failed to load HaluEval: %v\n", err)
	}

	// HalluLens
	if samples, err := bl.LoadHalluLens(limits["hallulens"]); err == nil {
		benchmarks["hallulens"] = samples
	} else {
		fmt.Printf("Warning: Failed to load HalluLens: %v\n", err)
	}

	if len(benchmarks) == 0 {
		return nil, fmt.Errorf("no benchmarks loaded successfully")
	}

	return benchmarks, nil
}

// ParseGroundTruth parses ground truth labels from various formats.
func ParseGroundTruth(label string) bool {
	lower := strings.ToLower(strings.TrimSpace(label))
	switch lower {
	case "true", "1", "yes", "correct", "supports", "factual":
		return true
	case "false", "0", "no", "incorrect", "refutes", "hallucination":
		return false
	default:
		// Try parsing as number
		if val, err := strconv.ParseFloat(lower, 64); err == nil {
			return val > 0.5
		}
		return false
	}
}

// SplitTrainTest splits samples into train (calibration) and test sets.
func SplitTrainTest(samples []BenchmarkSample, trainRatio float64, seed int64) (train, test []BenchmarkSample) {
	// Shuffle with fixed seed for reproducibility
	shuffled := make([]BenchmarkSample, len(samples))
	copy(shuffled, samples)

	// Simple deterministic shuffle based on seed
	for i := len(shuffled) - 1; i > 0; i-- {
		j := int((int64(i+1) * seed) % int64(i+1))
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		seed = seed*1103515245 + 12345 // LCG
	}

	splitIdx := int(float64(len(shuffled)) * trainRatio)
	train = shuffled[:splitIdx]
	test = shuffled[splitIdx:]
	return
}
