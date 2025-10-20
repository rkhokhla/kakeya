"""
Signal computation module for Fractal LBA + Kakeya FT Stack.
Computes D̂ (fractal dimension), coh★ (coherence), and r (compressibility).
"""

import math
import zlib
from typing import Dict, List, Tuple
import numpy as np


def compute_D_hat(scales: List[int], N_j: Dict[int, int]) -> float:
    """
    Compute fractal dimension D̂ using Theil-Sen robust regression.

    Args:
        scales: List of spatial scales (e.g., [2, 4, 8, 16, 32])
        N_j: Dictionary mapping scale to number of unique nonempty cells

    Returns:
        Median slope (D̂) rounded to 9 decimals
    """
    if len(scales) < 2:
        raise ValueError("Need at least 2 scales for regression")

    # Build (log2(scale), log2(N_j)) pairs
    points = []
    for scale in scales:
        n = N_j.get(scale)
        if n is None:
            raise ValueError(f"Missing N_j entry for scale {scale}")
        if n <= 0:
            raise ValueError(f"N_j must be positive for scale {scale}")

        points.append((math.log2(scale), math.log2(n)))

    # Theil-Sen: compute all pairwise slopes, take median
    slopes = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[j][0] - points[i][0]
            if abs(dx) < 1e-9:
                continue
            dy = points[j][1] - points[i][1]
            slope = dy / dx
            slopes.append(slope)

    if not slopes:
        raise ValueError("No valid slopes computed")

    slopes.sort()
    median_slope = slopes[len(slopes) // 2]

    return round(median_slope, 9)


def compute_coherence(
    points: np.ndarray,
    num_directions: int = 100,
    num_bins: int = 20,
    seed: int = None
) -> Tuple[float, np.ndarray]:
    """
    Compute directional coherence coh★ using histogram projection.

    Per CLAUDE_PHASE1.md:
    - Sample directions uniformly on sphere
    - Bin projections linearly between min/max
    - Handle zero-width case (pmax == pmin) with single-bin behavior
    - Support reproducibility via seed

    Args:
        points: Nx3 array of 3D points
        num_directions: Number of random directions to sample
        num_bins: Number of histogram bins for projection (default 20, recommended 64)
        seed: Random seed for reproducibility (optional)

    Returns:
        (coh_star, v_star): Maximum coherence and corresponding direction
    """
    if len(points) == 0:
        return 0.0, np.array([0.0, 0.0, 0.0])

    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    max_coherence = 0.0
    best_direction = np.array([1.0, 0.0, 0.0])

    # Sample random unit directions
    for _ in range(num_directions):
        # Random direction on unit sphere (uniform via normal distribution)
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)

        # Project points onto direction
        projections = points @ v

        # Handle zero-width case (all points project to same value)
        pmin, pmax = projections.min(), projections.max()
        if abs(pmax - pmin) < 1e-9:
            # All points in single bin → coh = 1.0
            coherence = 1.0
        else:
            # Create histogram with linear bins between [pmin, pmax]
            hist, _ = np.histogram(projections, bins=num_bins, range=(pmin, pmax))

            # Coherence = max fraction in any bin
            coherence = hist.max() / len(points) if len(points) > 0 else 0.0

        if coherence > max_coherence:
            max_coherence = coherence
            best_direction = v

    return round(max_coherence, 9), best_direction


def compute_compressibility(data: bytes) -> float:
    """
    Compute compressibility ratio r = compressed_size / raw_size.

    Per CLAUDE_PHASE1.md: Use zlib level=6 for balance between speed and compression.

    Args:
        data: Raw byte data (canonical row format, UTF-8 encoded)

    Returns:
        Compression ratio r ∈ [0, 1], rounded to 9 decimals
    """
    if len(data) == 0:
        return 1.0  # Empty stream guard

    compressed = zlib.compress(data, level=6)  # Level 6 per CLAUDE_PHASE1
    ratio = len(compressed) / len(data)

    # Clamp to [0, 1]
    ratio = max(0.0, min(1.0, ratio))

    return round(ratio, 9)


def classify_regime(D_hat: float, coh_star: float) -> str:
    """
    Classify regime based on D̂ and coh★.

    Args:
        D_hat: Fractal dimension
        coh_star: Directional coherence

    Returns:
        "sticky", "mixed", or "non_sticky"
    """
    if coh_star >= 0.70 and D_hat <= 1.5:
        return "sticky"
    if D_hat >= 2.6:
        return "non_sticky"
    return "mixed"


def compute_budget(
    D_hat: float,
    coh_star: float,
    r: float,
    alpha: float = 0.30,
    beta: float = 0.50,
    gamma: float = 0.20,
    base: float = 0.10,
    D0: float = 2.2
) -> float:
    """
    Compute budget allocation according to CLAUDE.md formula.

    Args:
        D_hat: Fractal dimension
        coh_star: Directional coherence
        r: Compressibility
        alpha, beta, gamma: Weight parameters
        base: Base budget
        D0: Dimension threshold

    Returns:
        Budget ∈ [0, 1], rounded to 9 decimals
    """
    budget = base + alpha * (1 - r) + beta * max(0, D_hat - D0) + gamma * coh_star

    # Clamp to [0, 1]
    budget = max(0.0, min(1.0, budget))

    return round(budget, 9)


def round_9(x: float) -> float:
    """Round float to 9 decimal places for signature stability."""
    return round(x, 9)
