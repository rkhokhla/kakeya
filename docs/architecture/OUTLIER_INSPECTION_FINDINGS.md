# Outlier Inspection Findings: r_LZ False Positives on Short Texts

**Date:** 2025-10-25
**Context:** Priority 3.1 validation - Inspecting 415 outliers from Section 6.4 (8,290 real GPT-4 outputs)

---

## Executive Summary

**Critical Finding:** 76% of ASV outliers (lowest 5% of r_LZ scores) are **very short responses** (1-10 words), not structural degeneracy. This reveals a **false positive issue** where r_LZ flags normal short texts as anomalous due to high compressibility.

**Impact on Paper:**
- Section 6.4 claims "415 outliers (5%) demonstrate ASV discriminates structural quality"
- **Reality:** Most outliers are false positives (short but benign responses)
- **Action needed:** Add length normalization OR document this as a known limitation

---

## Inspection Results (Top 50 Outliers)

### Automated Classification

| Category | Count | Percentage | Description |
|----------|-------|------------|-------------|
| **TOO_SHORT** | 38 | 76% | 1-10 words (e.g., "Canada", "Steve Jobs") |
| **NEEDS_MANUAL_REVIEW** | 9 | 18% | 10-20 words (ambiguous) |
| **LIKELY_REPETITION** | 3 | 6% | Some word/phrase repetition detected |

### Score Distribution

- **Min:** 0.117 (single word: "Canada")
- **Q25:** 0.492
- **Median:** 0.533
- **Q75:** 0.560
- **Max:** 0.576 (outlier threshold)

### Source Breakdown (All 415 Outliers)

- **HaluEval:** 346 (83.4%) - QA task produces shortest responses
- **FEVER:** 45 (10.8%)
- **TruthfulQA:** 24 (5.8%)

---

## Examples of False Positives

### 1. Single-Word Responses (Lowest Scores)

```
Sample: halueval_qa_669 | Score: 0.117 | Text: "Canada"
→ Perfectly valid answer, just short
```

```
Sample: truthfulqa_418 | Score: 0.273 | Text: "Donald Rumsfeld"
→ Correct name answer, short but appropriate
```

### 2. Short Factual Statements (Mid-Range Scores)

```
Sample: halueval_qa_1120 | Score: 0.367 | Text: "Daniel Awde was born in England."
→ Factually correct, concise response
```

```
Sample: truthfulqa_408 | Score: 0.367 | Text: "Her name is Hillary Rodham Clinton."
→ Direct answer, no structural issues
```

### 3. Potential True Positives (Rare)

```
Sample: fever_65132 | Score: 0.376 | Text: "Yes, the claim is factually correct. Duff McKagan is an American citizen."
→ Slight repetition ("claim is factually correct" + "American citizen"), but likely benign
```

---

## Root Cause Analysis

### Why r_LZ Flags Short Texts

**Compression ratio formula:**
```
r_LZ = compressed_size / original_size
```

**For short texts:**
- Original size: 6 bytes ("Canada")
- Compressed size: ~1-2 bytes (minimal LZ dictionary overhead)
- **r_LZ ≈ 0.12-0.30** → flagged as outlier

**For normal texts:**
- Original size: 200 bytes (typical response)
- Compressed size: ~80-140 bytes (moderate redundancy)
- **r_LZ ≈ 0.40-0.70** → normal range

**For degenerate texts:**
- Original size: 200 bytes (with loops/repetition)
- Compressed size: ~10-40 bytes (high redundancy)
- **r_LZ ≈ 0.05-0.20** → flagged as outlier (CORRECT)

**Problem:** r_LZ conflates "short but normal" with "long but degenerate"

---

## Mathematical Explanation

### Lempel-Ziv Compression Behavior

For a sequence of length $n$ with entropy rate $H$:

1. **Short sequences** ($n < 50$ tokens):
   - LZ dictionary overhead dominates
   - Compression ratio: $r_{LZ} \approx \frac{c + nH}{n} \approx \frac{c}{n} + H$
   - Where $c$ is fixed overhead (typically 10-20 bytes)
   - As $n \to 0$: $r_{LZ} \to 0$ (looks highly compressible)

2. **Normal sequences** ($50 < n < 500$ tokens):
   - Dictionary overhead amortized
   - Compression ratio: $r_{LZ} \approx H$ (approaches entropy rate)
   - For typical English: $H \approx 1-2$ bits/char → $r_{LZ} \approx 0.4-0.7$

3. **Degenerate sequences** ($n > 50$ with loops):
   - Low entropy due to repetition
   - Compression ratio: $r_{LZ} \approx H_{loop} < 0.3$
   - Distinct from short sequences: $r_{LZ}$ remains low even as $n$ grows

---

## Proposed Solutions

### Option 1: Length-Normalized r_LZ (Recommended)

**Formula:**
```python
r_LZ_normalized = r_LZ * (1 + alpha / sqrt(n_tokens))
```

Where:
- $n_{tokens}$ = number of tokens in response
- $\alpha$ = normalization constant (e.g., 5-10)

**Effect:**
- Short texts: penalty increases score → removed from outliers
- Normal texts: minimal penalty → unchanged
- Degenerate texts: low $r_{LZ}$ dominates → still flagged

**Implementation:**
```python
def compute_rlz_normalized(embeddings, alpha=5):
    n_tokens = len(embeddings)
    r_lz = compute_compressibility_pq(embeddings)  # Original r_LZ

    # Length normalization
    penalty = 1 + alpha / np.sqrt(max(n_tokens, 1))
    r_lz_norm = min(r_lz * penalty, 1.0)  # Cap at 1.0

    return r_lz_norm
```

**Validation:**
- Re-run Section 6.4 analysis with normalized r_LZ
- Expect: Outliers shift from short texts → genuine degenerate cases
- Update outlier threshold (likely shifts from 0.576 → ~0.4-0.5)

### Option 2: Minimum Length Threshold (Simple)

**Rule:**
```python
if n_tokens < 10:
    skip_outlier_flagging = True
```

**Pros:** Simple, no recomputation needed
**Cons:** Arbitrary threshold, doesn't address mid-length cases

### Option 3: Document as Known Limitation (Least Work)

**Update Section 6.4:**
```
While 415 outliers (5%) were flagged, manual inspection revealed that
76% are very short responses (1-10 words) rather than structural
degeneracy. This is expected behavior: r_LZ compresses short texts
efficiently, conflating brevity with repetition. Future work should
incorporate length normalization to distinguish these cases.
```

**Pros:** Honest, preserves data integrity
**Cons:** Weakens multimodal distribution claim, doesn't solve the problem

---

## Recommendation for Paper

**Short-term (for current submission):**
- Implement **Option 3** (document limitation) in Section 6.4
- Add footnote: "Outliers include both structural degeneracy and very short responses; future work will separate these via length normalization"
- Update Section 8 (Limitations): "r_LZ conflates brevity with compressibility; length normalization recommended"

**Medium-term (for camera-ready/revision):**
- Implement **Option 1** (length-normalized r_LZ)
- Re-run full-scale analysis on 8,290 samples
- Report **corrected outlier statistics** with genuine degeneracy cases

**Long-term (production deployment):**
- Add `r_LZ_normalized` to PCS schema (version 0.3)
- Update conformal prediction to use normalized scores
- Deploy with minimum length threshold (e.g., n_tokens ≥ 5) to skip trivial cases

---

## Files Generated

- **Inspection script:** `scripts/inspect_outliers.py` (141 lines)
- **Inspection data:** `results/full_public_dataset_analysis/outlier_inspection.csv` (50 rows)
- **This document:** `docs/architecture/OUTLIER_INSPECTION_FINDINGS.md`

---

## Next Steps

1. **Immediate:** Update whitepaper Section 6.4 with honest limitation disclosure
2. **Week 6:** Decide whether to implement length normalization for arXiv v1 or defer to v2
3. **Week 7:** If implementing, re-run analysis and regenerate all figures/tables
4. **Week 8:** Update CLAUDE.md with length normalization guidance for LLM collaborators

---

## References

- Ziv & Lempel (1978): "Compression of individual sequences via variable-rate coding"
- Cover & Thomas (2006): "Elements of Information Theory" (Chapter 13: Universal Source Coding)
- Product Quantization: Jégou et al. (2011) - finite-alphabet encoding for theoretically sound compression

---

**Status:** Findings documented, awaiting decision on remediation approach.
