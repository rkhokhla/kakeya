# Signal Computation - Mathematical Foundations

Deep dive into the mathematical principles behind D̂, coh★, and r.

## Overview

The Fractal LBA + Kakeya FT Stack computes three core signals that characterize distributed event streams:

1. **D̂** (D-hat): Fractal dimension via Theil-Sen regression
2. **coh★** (coherence star): Directional coherence via histogram projection
3. **r**: Compressibility ratio via zlib

These signals enable regime classification (sticky/mixed/non_sticky) and budget allocation.

---

## 1. Fractal Dimension (D̂)

### Intuition

Fractal dimension measures how "space-filling" a point cloud is:
- **D̂ ≈ 1**: Points lie on a line (1-dimensional)
- **D̂ ≈ 2**: Points fill a plane (2-dimensional)
- **D̂ ≈ 3**: Points fill a volume (3-dimensional)

For event streams, low D̂ suggests **clustered** behavior (e.g., hotspots), while high D̂ suggests **diffuse** behavior.

### Box-Counting Method

We use a **multi-scale box-counting** approach:

1. **Partition space** into a grid of scale `s` (e.g., s=2, 4, 8, 16, 32)
2. **Count non-empty boxes** at each scale → `N_j(s)`
3. **Measure scaling relationship**: `N_j(s) ~ s^D̂`

Taking logarithms:
```
log₂(N_j) = D̂ · log₂(s) + c
```

This is a linear relationship where **slope = D̂**.

### Theil-Sen Regression

Instead of least-squares (sensitive to outliers), we use **Theil-Sen**:

1. Compute **all pairwise slopes**:
   ```
   m_ij = (log₂(N_j[i]) - log₂(N_j[j])) / (log₂(s[i]) - log₂(s[j]))
   ```

2. Take the **median** slope:
   ```
   D̂ = median(m_ij)
   ```

### Example

**Data**:
```
scales: [2, 4, 8, 16, 32]
N_j:    [3, 5, 9, 17, 31]
```

**Log-log points**:
```
(log₂(2), log₂(3))   = (1.0, 1.585)
(log₂(4), log₂(5))   = (2.0, 2.322)
(log₂(8), log₂(9))   = (3.0, 3.170)
(log₂(16), log₂(17)) = (4.0, 4.087)
(log₂(32), log₂(31)) = (5.0, 4.954)
```

**Pairwise slopes** (sample):
```
m_01 = (2.322 - 1.585) / (2.0 - 1.0) = 0.737
m_02 = (3.170 - 1.585) / (3.0 - 1.0) = 0.793
...
```

**Median slope** ≈ **0.96** → D̂ = 0.96

**Interpretation**: Points cluster along a line (D̂ ≈ 1).

### Implementation

```python
import math

def compute_D_hat(scales, N_j):
    # Build (log2(s), log2(N_j)) pairs
    points = [(math.log2(s), math.log2(N_j[s])) for s in scales]

    # Compute all pairwise slopes
    slopes = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[j][0] - points[i][0]
            if abs(dx) < 1e-9:
                continue
            dy = points[j][1] - points[i][1]
            slope = dy / dx
            slopes.append(slope)

    # Median slope
    slopes.sort()
    D_hat = slopes[len(slopes) // 2]

    return round(D_hat, 9)
```

### Robustness

**Why Theil-Sen?**
- Insensitive to outliers (up to 29.3%)
- Breakdown point > OLS (0%)
- Computationally efficient: O(n²)

**Alternative**: Least-squares is O(n) but fails if one scale is corrupted.

---

## 2. Directional Coherence (coh★)

### Intuition

Coherence measures if points **align** along a specific direction. Inspired by the **Kakeya needle problem**: can you rotate a needle through 360° in minimal area?

High coherence (coh★ ≈ 1) means events are **directionally concentrated** (e.g., traffic flowing in one direction). Low coherence (coh★ ≈ 0) means events are **isotropic** (uniformly distributed).

### Algorithm

1. **Sample random unit directions** `v` on the unit sphere:
   ```python
   v = np.random.randn(3)
   v = v / np.linalg.norm(v)
   ```

2. For each direction `v`:
   a. **Project** all points onto `v`:
      ```python
      projections = points @ v  # Dot product
      ```

   b. **Histogram** the projections into bins:
      ```python
      hist, _ = np.histogram(projections, bins=20)
      ```

   c. **Coherence** = max fraction in any bin:
      ```python
      coh_v = hist.max() / len(points)
      ```

3. **coh★** = maximum coherence across all sampled directions:
   ```
   coh★ = max(coh_v for v in sampled_directions)
   ```

### Example

**Points**: 100 3D points clustered along x-axis

**Direction**: v = [1, 0, 0]

**Projections**: 90 points project to x ∈ [0.8, 1.2]

**Histogram**:
```
Bin [0.8, 1.0): 45 points
Bin [1.0, 1.2): 45 points
Other bins:     10 points
```

**Coherence** = 45/100 = 0.45

**Best direction**: v_star ≈ [1, 0.1, 0] gives coh_v = 0.90 → **coh★ = 0.90**

### Implementation

```python
import numpy as np

def compute_coherence(points, num_directions=100, num_bins=20):
    max_coherence = 0.0
    best_direction = np.array([1.0, 0.0, 0.0])

    for _ in range(num_directions):
        # Random unit direction
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)

        # Project points
        projections = points @ v

        # Histogram
        hist, _ = np.histogram(projections, bins=num_bins)

        # Coherence
        coh_v = hist.max() / len(points) if len(points) > 0 else 0.0

        if coh_v > max_coherence:
            max_coherence = coh_v
            best_direction = v

    return round(max_coherence, 9), best_direction
```

### Tuning

**num_directions**:
- Higher → more accurate, slower
- Recommended: 100-1000

**num_bins**:
- Higher → finer resolution, lower coherence
- Lower → coarser, higher coherence
- Recommended: 20-50

### Connection to Kakeya

The **Kakeya needle problem** asks: what's the minimum area in which a unit line segment can be continuously rotated 360°?

**Coherence** measures the opposite: given a cloud, what's the **maximum concentration** along any direction? High concentration means the cloud is "needle-like" in some orientation.

---

## 3. Compressibility (r)

### Intuition

Compressibility quantifies **structure** or **predictability**:
- **r ≈ 0**: Highly compressible (e.g., repeated patterns)
- **r ≈ 1**: Incompressible (e.g., random noise)

For event streams, low r suggests **regularity** (e.g., periodic events), while high r suggests **randomness**.

### Algorithm

1. **Serialize** event data to bytes
2. **Compress** with zlib (DEFLATE, level=9)
3. **Ratio**:
   ```
   r = len(compressed) / len(raw)
   ```

### Example

**Highly Compressible**:
```python
data = b"a" * 1000  # 1000 bytes
compressed = zlib.compress(data, level=9)  # ~20 bytes
r = 20 / 1000 = 0.02  # Highly compressible
```

**Random Data**:
```python
data = os.urandom(1000)  # Random bytes
compressed = zlib.compress(data, level=9)  # ~1005 bytes (no gain)
r = 1005 / 1000 = 1.005 ≈ 1.0  # Incompressible
```

### Implementation

```python
import zlib

def compute_compressibility(data):
    if len(data) == 0:
        return 1.0

    compressed = zlib.compress(data, level=9)
    ratio = len(compressed) / len(data)

    # Clamp to [0, 1]
    ratio = max(0.0, min(1.0, ratio))

    return round(ratio, 9)
```

### Limitations

- **Compression ratio** is a **proxy** for Kolmogorov complexity
- True Kolmogorov complexity is uncomputable
- zlib is heuristic, not optimal

---

## 4. Regime Classification

### Definitions

**sticky**: `coh★ ≥ 0.70 and D̂ ≤ 1.5`
- Events cluster along a low-dimensional manifold
- Example: Congested traffic, hotspot in network

**non_sticky**: `D̂ ≥ 2.6`
- Events diffuse across high-dimensional space
- Example: Uniform random traffic

**mixed**: Otherwise
- Intermediate behavior
- Example: Bursty but spatially distributed

### Phase Diagram

```
      coh★
       1.0  │
            │  sticky
       0.70 ├─────────────
            │    mixed     │ non_sticky
       0.0  └──────┬───────┴─────────
                 1.5       2.6       D̂
```

### Rationale

**Thresholds**:
- `coh★ = 0.70`: Empirically, 70% concentration indicates strong alignment
- `D̂ = 1.5`: Between 1D and 2D (line vs. plane)
- `D̂ = 2.6`: Between 2D and 3D (plane vs. volume)

These are **heuristic** and domain-specific. Adjust based on your data.

---

## 5. Budget Computation

### Formula

```
budget = base + α(1 - r) + β·max(0, D̂ - D₀) + γ·coh★
```

Where:
- **base** = 0.10 (minimum allocation)
- **α** = 0.30 (weight for structure)
- **β** = 0.50 (weight for dimensionality)
- **γ** = 0.20 (weight for coherence)
- **D₀** = 2.2 (dimension threshold)

**Clamped** to [0, 1].

### Interpretation

**Components**:

1. **α(1 - r)**: Reward compressible (structured) data
   - r=0 → +0.30
   - r=1 → +0.00

2. **β·max(0, D̂ - D₀)**: Penalize low-dimensional data
   - D̂=1.5 → +0.00 (below threshold)
   - D̂=3.0 → +0.40

3. **γ·coh★**: Reward coherent data
   - coh★=0 → +0.00
   - coh★=1 → +0.20

### Example

**Sticky Regime**:
```
D̂ = 1.3, coh★ = 0.75, r = 0.40

budget = 0.10 + 0.30(1 - 0.40) + 0.50·max(0, 1.3 - 2.2) + 0.20·0.75
       = 0.10 + 0.18 + 0.00 + 0.15
       = 0.43
```

**Non-sticky Regime**:
```
D̂ = 2.8, coh★ = 0.30, r = 0.90

budget = 0.10 + 0.30(1 - 0.90) + 0.50·max(0, 2.8 - 2.2) + 0.20·0.30
       = 0.10 + 0.03 + 0.30 + 0.06
       = 0.49
```

### Use Cases

**Budget** can represent:
- **Computational resources** to allocate
- **Confidence score** for downstream processing
- **Priority** in a queue

Domain-specific! Adjust weights (α, β, γ) based on your application.

---

## 6. Verification Tolerances

### Tolerances

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `tolD` | 0.15 | D̂ recomputation tolerance |
| `tolCoh` | 0.05 | coh★ bounds tolerance |

### D̂ Tolerance

**Check**:
```
|D̂_claimed - D̂_recomputed| ≤ tolD
```

**Rationale**:
- Theil-Sen is robust but not exact
- Floating-point errors accumulate
- 15% tolerance allows for small variations

**Example**:
```
D̂_claimed = 1.45
D̂_recomputed = 1.55
Diff = 0.10 ≤ 0.15  → PASS
```

### coh★ Bounds

**Check**:
```
0 ≤ coh★ ≤ 1 + tolCoh
```

**Rationale**:
- Theoretical bound is [0, 1]
- Floating-point rounding can exceed 1.0 slightly
- 5% tolerance (1.05) catches egregious errors

**Example**:
```
coh★ = 1.03  → PASS (within tolerance)
coh★ = 1.10  → FAIL (exceeds tolerance)
```

---

## 7. Numeric Stability

### Rounding to 9 Decimals

**Why 9?**
- IEEE 754 double precision: ~15-17 significant decimal digits
- 9 decimals provides stability while retaining precision
- Matches Bitcoin's satoshi (8 decimals) + 1 for margin

**Implementation**:
```python
def round_9(x):
    return round(x, 9)

# Example
x = 1.4123456789012345
y = round_9(x)  # 1.412345679
```

**Signature Stability**:
```python
# Before signing
payload = {
    "D_hat": round_9(1.4123456789),
    "coh_star": round_9(0.7345678901),
    "r": round_9(0.8712345678),
    "budget": round_9(0.4212345678)
}
```

Without rounding, floating-point drift causes signature mismatches across machines.

---

## 8. Future Improvements

### VRF-Based Direction Sampling

**Problem**: Adversary can steer coherence by manipulating direction sampling seed.

**Solution**: Use **Verifiable Random Function (VRF)** to prove directions were sampled fairly:

```python
from vrf import VRF

vrf_key = VRF.keygen()
seed = hash(merkle_root + epoch)

directions = []
for i in range(100):
    proof, v = vrf_key.prove(seed + i)
    directions.append(v)
    # Backend can verify: vrf_key.verify(seed + i, v, proof)
```

### Formal D̂ Bounds

**Problem**: Theil-Sen tolerance is heuristic.

**Solution**: Compute **confidence intervals** from bootstrap:

```python
from scipy.stats import bootstrap

slopes = compute_all_slopes(scales, N_j)
ci = bootstrap(slopes, np.median, confidence_level=0.95)

# Claim: D̂ ± margin
D_hat_claimed = ci.mean()
margin = ci.std() * 2
```

Backend checks: `D̂_recomputed ∈ [D̂_claimed - margin, D̂_claimed + margin]`

---

## Further Reading

- **Fractal Dimension**: Mandelbrot, "The Fractal Geometry of Nature" (1982)
- **Theil-Sen Regression**: Sen, "Estimates of the Regression Coefficient Based on Kendall's Tau" (1968)
- **Kakeya Problem**: Wolff, "Recent Work on the Kakeya Problem" (1999)
- **Kolmogorov Complexity**: Li & Vitányi, "An Introduction to Kolmogorov Complexity" (2008)

---

## Next Steps

- [Architecture Overview](overview.md)
- [Verification Engine Design](../development/testing.md)
- [API Reference](../api/rest-api.md)
