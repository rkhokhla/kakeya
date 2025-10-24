# Real Evaluation Implementation Plan (Option C)

> **Decision Date:** 2025-10-24
> **Approach:** Full real evaluation with actual LLM embeddings
> **Timeline:** 1-2 weeks
> **Estimated Cost:** $200-500 in API calls
> **Goal:** Publication-grade results with complete academic integrity

---

## Executive Summary

We are implementing a complete, production-grade evaluation of ASV using:
- **Real benchmark data** (TruthfulQA, FEVER, HaluEval, HalluLens)
- **Real LLM outputs** (via OpenAI/Anthropic APIs)
- **Real embeddings** extracted from LLM API responses
- **Real signal computation** (D̂, coh★, r_LZ from actual embeddings)
- **Real baseline methods** (GPT-2 perplexity, RoBERTa-MNLI, GPT-4-as-judge)

This will give us verifiable, reproducible results for academic publication.

---

## Phase 1: Data Infrastructure Setup

### Tasks
- [x] Create data directory structure
- [ ] Download TruthfulQA (817 samples)
- [ ] Download FEVER dev set (~20k samples, will sample 2,500)
- [ ] Download HaluEval (~5k samples, will sample 4,000)
- [ ] Investigate HalluLens availability (ACL 2025, may not be public yet)
- [ ] Create data loaders that match `backend/internal/eval/benchmarks.go`

### Directory Structure
```
data/
  benchmarks/
    truthfulqa.csv
    fever_dev.jsonl
    halueval.json
    hallulens.jsonl (if available)
  llm_outputs/
    truthfulqa_gpt4.jsonl
    fever_gpt4.jsonl
    halueval_gpt4.jsonl
  embeddings/
    truthfulqa_embeddings.npz
    fever_embeddings.npz
    halueval_embeddings.npz
  signals/
    truthfulqa_signals.jsonl
    fever_signals.jsonl
    halueval_signals.jsonl
```

### Time Estimate
- 2-4 hours

---

## Phase 2: LLM Integration

### Implementation

**File: `backend/internal/llm/client.go`**
```go
package llm

type LLMClient interface {
    Generate(prompt string, options GenerateOptions) (*Response, error)
    GetEmbeddings(text string) ([]float64, error)
}

type OpenAIClient struct {
    apiKey string
    model  string
}

type GenerateOptions struct {
    MaxTokens   int
    Temperature float64
    TopP        float64
}

type Response struct {
    Text       string
    Embeddings []float64  // If available
    TokenCount int
}
```

**API Choices:**
1. **OpenAI GPT-4** (generation + embeddings)
   - Model: `gpt-4-turbo-preview`
   - Embeddings: `text-embedding-3-large` (3072 dimensions)
   - Cost: ~$0.01/1k input tokens, ~$0.03/1k output tokens
   - Embeddings: ~$0.13/1M tokens

2. **Anthropic Claude** (generation only, no embeddings API)
   - Model: `claude-3-opus-20240229`
   - Cost: ~$0.015/1k input, ~$0.075/1k output
   - Would need separate embeddings (HuggingFace)

**Recommendation:** Start with OpenAI for simplicity (generation + embeddings in one API)

### Tasks
- [ ] Implement `OpenAIClient`
- [ ] Add rate limiting and retry logic
- [ ] Add cost tracking
- [ ] Test with sample requests
- [ ] Store API key securely (env var)

### Time Estimate
- 4-6 hours

### Cost Estimate (8,200 samples)
- Generation: 8,200 samples × ~300 tokens avg × $0.02/1k = **~$50**
- Embeddings: 8,200 samples × ~200 tokens × $0.13/1M = **~$0.20**
- **Total Phase 2: ~$50**

---

## Phase 3: Generate LLM Outputs

### Implementation

**File: `backend/cmd/generate_outputs/main.go`**
- Load benchmark samples
- For each sample, send prompt to LLM
- Store response with metadata
- Track progress (resume capability)
- Estimate remaining time and cost

### Output Format (JSONL)
```json
{
  "benchmark": "truthfulqa",
  "sample_id": "001",
  "prompt": "What is the capital of France?",
  "ground_truth": true,
  "llm_output": "The capital of France is Paris.",
  "model": "gpt-4-turbo-preview",
  "timestamp": "2025-10-24T10:30:00Z",
  "token_count": 12,
  "cost_usd": 0.00036
}
```

### Considerations
- **Rate limits:** OpenAI allows ~500 requests/min for GPT-4
- **Batching:** Process in batches of 100 with progress saves
- **Resume capability:** Skip already-processed samples
- **Error handling:** Retry transient failures, log permanent failures

### Tasks
- [ ] Implement generation CLI
- [ ] Add progress tracking
- [ ] Test with 10 samples
- [ ] Run on TruthfulQA (817 samples, ~2 hours)
- [ ] Run on FEVER sample (2,500 samples, ~5 hours)
- [ ] Run on HaluEval sample (4,000 samples, ~8 hours)

### Time Estimate
- Implementation: 6-8 hours
- Execution: ~15 hours (can run overnight)

---

## Phase 4: Extract Embeddings

### Implementation

**Option A: Use OpenAI Embeddings API**
- Call `text-embedding-3-large` for each LLM output
- Get token-level embeddings if available (may only have sentence-level)
- Dimension: 3072

**Option B: Use Open-Source Model**
- Run locally with sentence-transformers
- Model: `all-mpnet-base-v2` (768 dimensions)
- Requires GPU for reasonable speed

**Recommendation:** Try OpenAI first; if no token-level embeddings, use local model

### Challenge: Token-Level Embeddings

OpenAI embeddings API typically returns **sentence-level** embeddings, not token-level trajectories needed for D̂, coh★, r_LZ.

**Solutions:**
1. **Use logprobs from generation API** (gives token probabilities)
2. **Use hidden states from local model** (requires HuggingFace)
3. **Approximate from sentence embeddings** (split text, embed windows)

**Likely approach:** Use HuggingFace transformers locally to get actual hidden states

### Implementation Path

**File: `backend/internal/embeddings/extractor.go`**
```go
package embeddings

type EmbeddingExtractor interface {
    Extract(text string) (*Trajectory, error)
}

type Trajectory struct {
    Tokens     []string      // Token strings
    Embeddings [][]float64   // [num_tokens, embedding_dim]
    Dimension  int
}
```

**Python Script: `scripts/extract_embeddings.py`**
```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def extract_hidden_states(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Get last layer hidden states
    hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, 768]
    return hidden_states.squeeze(0).numpy()  # [seq_len, 768]
```

### Tasks
- [ ] Research OpenAI token-level embedding access
- [ ] If not available: Set up HuggingFace pipeline
- [ ] Implement embedding extraction
- [ ] Test with sample outputs
- [ ] Process all LLM outputs (8,200 samples)
- [ ] Store as numpy arrays or HDF5

### Time Estimate
- Research/setup: 4-6 hours
- Execution: 2-4 hours (GPU) or 8-12 hours (CPU)

---

## Phase 5: Implement Signal Computation

### Implementation

**File: `backend/internal/signals/compute.go`**

Need to implement the actual Python signal computation code (currently stubs).

### 5.1 Fractal Dimension (D̂)

**Algorithm:**
1. Compute bounding box of embeddings in d-dimensional space
2. For each scale s ∈ {2, 4, 8, 16, 32}:
   - Divide bounding box into s^d hypercubes
   - Count non-empty hypercubes: N(s)
3. Compute slopes for all pairs: m_ij = (log N_j - log N_i) / (log s_j - log s_i)
4. Return median slope (Theil-Sen estimator)

**File: `agent/src/signals.py` (needs real implementation)**
```python
def compute_fractal_dimension(embeddings, scales=[2,4,8,16,32]):
    """
    embeddings: numpy array [num_tokens, embedding_dim]
    returns: D_hat (fractal slope)
    """
    import numpy as np

    # Find bounding box
    mins = embeddings.min(axis=0)
    maxs = embeddings.max(axis=0)

    N_j = {}
    for scale in scales:
        # Quantize each dimension to scale bins
        quantized = np.floor((embeddings - mins) / (maxs - mins + 1e-10) * scale).astype(int)
        quantized = np.clip(quantized, 0, scale - 1)

        # Count unique hypercubes
        unique_cells = set(tuple(row) for row in quantized)
        N_j[str(scale)] = len(unique_cells)

    # Theil-Sen: median of pairwise slopes
    slopes = []
    for i, s_i in enumerate(scales):
        for j in range(i+1, len(scales)):
            s_j = scales[j]
            slope = (np.log2(N_j[str(s_j)]) - np.log2(N_j[str(s_i)])) / \
                    (np.log2(s_j) - np.log2(s_i))
            slopes.append(slope)

    return np.median(slopes)
```

### 5.2 Directional Coherence (coh★)

**Algorithm:**
1. Sample M=100 random directions on unit sphere
2. For each direction v:
   - Project all embeddings: p_i = <e_i, v>
   - Bin projections into B=20 equal-width bins
   - coh(v) = max_bin (count_bin / num_tokens)
3. Return max coh(v) over all sampled directions

**Implementation:**
```python
def compute_coherence(embeddings, num_directions=100, num_bins=20, seed=42):
    """
    embeddings: numpy array [num_tokens, embedding_dim]
    returns: coh_star (max directional coherence)
    """
    import numpy as np

    np.random.seed(seed)
    d = embeddings.shape[1]

    max_coh = 0.0
    best_direction = None

    for _ in range(num_directions):
        # Sample random direction on unit sphere
        direction = np.random.randn(d)
        direction /= np.linalg.norm(direction)

        # Project embeddings
        projections = embeddings @ direction

        # Bin projections
        hist, _ = np.histogram(projections, bins=num_bins)

        # Coherence = max bin proportion
        coh = hist.max() / len(projections)

        if coh > max_coh:
            max_coh = coh
            best_direction = direction

    return max_coh
```

### 5.3 Compressibility (r_LZ)

**Algorithm:**
1. Product quantize embeddings (8 subspaces, 8 bits each)
2. Convert to byte sequence
3. Compress with zlib
4. Return compressed_size / original_size

**Implementation:**
```python
def compute_compressibility(embeddings, n_subspaces=8, codebook_bits=8):
    """
    embeddings: numpy array [num_tokens, embedding_dim]
    returns: r (compressibility ratio)
    """
    import numpy as np
    from sklearn.cluster import KMeans
    import zlib

    # Product quantization
    d = embeddings.shape[1]
    subspace_dim = d // n_subspaces

    # Pad if necessary
    if d % n_subspaces != 0:
        pad_width = n_subspaces - (d % n_subspaces)
        embeddings = np.pad(embeddings, ((0, 0), (0, pad_width)), mode='constant')
        d = embeddings.shape[1]
        subspace_dim = d // n_subspaces

    codes = []
    for i in range(n_subspaces):
        subspace = embeddings[:, i*subspace_dim:(i+1)*subspace_dim]

        # K-means clustering
        n_clusters = 2 ** codebook_bits
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(subspace)

        codes.append(labels.astype(np.uint8))

    # Concatenate codes
    code_sequence = np.concatenate(codes)

    # Compress
    compressed = zlib.compress(code_sequence.tobytes(), level=6)

    # Compression ratio
    r = len(compressed) / len(code_sequence)

    return r
```

### Tasks
- [ ] Implement real signal computation in `agent/src/signals.py`
- [ ] Add unit tests with known cases
- [ ] Validate against illustrative table from whitepaper
- [ ] Create CLI to compute signals from embedding files
- [ ] Process all embeddings (8,200 samples)

### Time Estimate
- Implementation: 8-12 hours
- Testing: 4-6 hours
- Execution: 2-4 hours

---

## Phase 6: Implement Real Baseline Methods

### 6.1 Perplexity (GPT-2)

**File: `backend/internal/baselines/perplexity_real.go`**

Call HuggingFace API or use local GPT-2:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def compute_perplexity(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    perplexity = torch.exp(loss).item()
    return perplexity
```

### 6.2 NLI Entailment (RoBERTa-MNLI)

**File: `backend/internal/baselines/nli_real.go`**

```python
from transformers import pipeline

nli_classifier = pipeline('text-classification', model='roberta-large-mnli')

def check_entailment(premise, hypothesis):
    result = nli_classifier(f"{premise} [SEP] {hypothesis}")
    # result: {'label': 'ENTAILMENT', 'score': 0.95}
    return result['score'] if result['label'] == 'ENTAILMENT' else 1 - result['score']
```

### 6.3 SelfCheckGPT

Generate 5 alternative responses, compute NLI consistency.

### 6.4 GPT-4-as-judge

Call OpenAI API with factuality evaluation prompt.

### Tasks
- [ ] Implement GPT-2 perplexity
- [ ] Implement RoBERTa-MNLI
- [ ] Implement SelfCheckGPT (5 samples per input)
- [ ] Implement GPT-4-as-judge
- [ ] Run on all samples
- [ ] Store baseline scores

### Time Estimate
- Implementation: 12-16 hours
- Execution: 8-12 hours (GPT-4 calls expensive)

### Cost Estimate
- SelfCheckGPT: 8,200 samples × 5 generations × $0.02/1k tokens × 300 tokens = **~$250**
- GPT-4-as-judge: 8,200 samples × $0.02/1k × 500 tokens = **~$82**
- **Total Phase 6: ~$330**

---

## Phase 7: Evaluation Runner CLI

**File: `backend/cmd/eval/main.go`**

Integrate everything:

```go
func main() {
    // Load signals
    asvSignals := loadSignals("data/signals/")

    // Load baseline scores
    baselineScores := loadBaselineScores("data/baselines/")

    // Load ground truth
    samples := loadSamples("data/benchmarks/")

    // Split train/test
    train, test := splitTrainTest(samples, 0.7, seed)

    // Calibrate ASV
    calibSet := calibrateASV(train, asvSignals)

    // Run evaluation
    runner := eval.NewEvaluationRunner(...)
    report := runner.RunEvaluation(...)

    // Generate outputs
    plotter := eval.NewPlotter("results/")
    plotter.PlotAll(report)
}
```

### Tasks
- [ ] Implement end-to-end CLI
- [ ] Test with subset
- [ ] Run on full data
- [ ] Generate all plots and tables

### Time Estimate
- 6-8 hours

---

## Phase 8: Run Full Evaluation

### Execution
- Run evaluation CLI
- Generate all metrics
- Compute statistical tests
- Bootstrap confidence intervals
- Generate plots (via Python scripts)

### Outputs
- `results/performance_table.md`
- `results/statistical_tests.md`
- `results/roc_curves.png`
- `results/calibration_plots.png`
- `results/confusion_matrices.png`
- `results/cost_comparison.png`
- `results/SUMMARY.md`

### Time Estimate
- 2-4 hours execution

---

## Phase 9: Generate Plots

Use the Python plotting scripts from `plotter.go`.

### Tasks
- [ ] Install matplotlib, seaborn
- [ ] Run plotting scripts
- [ ] Generate camera-ready figures (300 DPI)
- [ ] Create LaTeX-formatted tables

### Time Estimate
- 2-4 hours

---

## Phase 10: Update Whitepaper

### Changes

**Section 7: Replace with REAL results**
- Table 1: Real performance metrics
- Table 2: Real cost comparison
- Table 3: Real latency measurements
- Statistical tests: Real p-values from actual McNemar's test
- Bootstrap CIs: Real confidence intervals

**Appendix B: Add actual plots**
- Include generated PNG files or describe with real data

**Section 7.6: Update limitations**
- Remove "synthetic PCS" caveat
- Keep "simplified baselines" if we use heuristics
- Add any new limitations discovered

### Tasks
- [ ] Update all tables with real numbers
- [ ] Update all text with real results
- [ ] Add plot files to appendix
- [ ] Proofread for consistency
- [ ] Update abstract/intro/conclusion with real numbers

### Time Estimate
- 4-6 hours

---

## Total Estimates

### Time
- **Phase 1:** 2-4 hours
- **Phase 2:** 4-6 hours
- **Phase 3:** 6-8 hours impl + 15 hours exec = 21-23 hours
- **Phase 4:** 4-6 hours setup + 2-12 hours exec = 6-18 hours
- **Phase 5:** 12-18 hours impl + 2-4 hours exec = 14-22 hours
- **Phase 6:** 12-16 hours impl + 8-12 hours exec = 20-28 hours
- **Phase 7:** 6-8 hours
- **Phase 8:** 2-4 hours
- **Phase 9:** 2-4 hours
- **Phase 10:** 4-6 hours

**Total: 92-125 hours** (~2-3 weeks of focused work)

### Cost
- **Phase 3:** ~$50 (LLM generation)
- **Phase 6:** ~$330 (SelfCheckGPT + GPT-4-as-judge)
- **Total API costs: ~$380-400**

### Risks
1. **Embeddings challenge:** OpenAI may not provide token-level embeddings
   - Mitigation: Use HuggingFace local models
2. **HalluLens unavailable:** ACL 2025 dataset may not be public
   - Mitigation: Use 3 benchmarks (still 7k+ samples)
3. **API rate limits:** May slow execution
   - Mitigation: Implement batching and resume capability
4. **Cost overruns:** Could exceed $400 if more samples needed
   - Mitigation: Start with smaller sample (e.g., 1k samples), validate, then scale

---

## Success Criteria

✅ **Real benchmarks downloaded and loaded**
✅ **Real LLM outputs generated for all samples**
✅ **Real embeddings extracted (token-level trajectories)**
✅ **Real signals computed (D̂, coh★, r_LZ from embeddings)**
✅ **Real baseline methods executed**
✅ **Real evaluation pipeline run end-to-end**
✅ **Real statistical tests with actual p-values**
✅ **Real plots generated from actual data**
✅ **Whitepaper updated with verifiable numbers**
✅ **All code, data, results reproducible and shareable**

---

## Next Steps

**Immediate (today):**
1. Set up OpenAI API key
2. Create data directories
3. Start Phase 1: Download TruthfulQA

**This week:**
- Complete Phases 1-4 (data + embeddings)
- Implement signal computation (Phase 5)

**Next week:**
- Complete baseline implementations (Phase 6)
- Run full evaluation (Phases 7-8)

**Week after:**
- Generate plots (Phase 9)
- Update whitepaper (Phase 10)
- Submit to arXiv

---

**Ready to start?** Let me know if you want to proceed with this plan, or if we should adjust scope/budget.
