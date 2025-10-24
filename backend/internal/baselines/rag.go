package baselines

import (
	"fmt"

	"github.com/fractal-lba/kakeya/backend/internal/eval"
)

// RAGBaseline verifies RAG faithfulness (response grounded in retrieved context).
type RAGBaseline struct {
	threshold      float64
	escalateMargin float64
}

// NewRAGBaseline creates a RAG faithfulness baseline.
func NewRAGBaseline(threshold float64) *RAGBaseline {
	return &RAGBaseline{
		threshold:      threshold,
		escalateMargin: 0.05,
	}
}

// Verify checks if response is faithful to retrieved context.
func (rb *RAGBaseline) Verify(sample *eval.BenchmarkSample) (*eval.BaselineResult, error) {
	// RAG faithfulness: overlap between response and prompt (retrieved context)
	faithfulness := jaccard(tokenize(sample.Prompt), tokenize(sample.Response))

	var decision eval.Decision
	if faithfulness >= rb.threshold {
		decision = eval.DecisionAccept
	} else if faithfulness >= rb.threshold-rb.escalateMargin {
		decision = eval.DecisionEscalate
	} else {
		decision = eval.DecisionReject
	}

	return &eval.BaselineResult{
		SampleID:  sample.ID,
		Method:    "rag",
		Score:     faithfulness,
		Decision:  decision,
		Threshold: rb.threshold,
		Metadata:  map[string]interface{}{"metric": "jaccard"},
	}, nil
}

func (rb *RAGBaseline) Name() string        { return "rag" }
func (rb *RAGBaseline) SetThreshold(t float64) { rb.threshold = t }
func (rb *RAGBaseline) GetScore(s *eval.BenchmarkSample) float64 {
	return jaccard(tokenize(s.Prompt), tokenize(s.Response))
}

// ProductionNote: Real RAG faithfulness would use:
// - Citation checking (response claims cited in retrieved docs)
// - Entailment verification (response entailed by context)
// - Hallucination-specific metrics (HHEM, RAGTruth, etc.)
//
// See scripts/eval/rag_faithfulness.py for full implementation.
