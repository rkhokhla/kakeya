"""
Unit tests for signal computation module.
"""

import pytest
import numpy as np
from agent.src import signals


def test_compute_D_hat():
    """Test fractal dimension computation."""
    scales = [2, 4, 8, 16, 32]
    N_j = {2: 3, 4: 5, 8: 9, 16: 17, 32: 31}

    D_hat = signals.compute_D_hat(scales, N_j)

    # Slope should be close to 1.0 for this data
    assert 0.8 <= D_hat <= 1.2, f"Expected D_hat near 1.0, got {D_hat}"


def test_compute_D_hat_missing_scale():
    """Test error handling for missing scale."""
    scales = [2, 4, 8]
    N_j = {2: 3, 4: 5}  # Missing scale 8

    with pytest.raises(ValueError, match="Missing N_j entry"):
        signals.compute_D_hat(scales, N_j)


def test_classify_regime():
    """Test regime classification."""
    assert signals.classify_regime(1.3, 0.75) == "sticky"
    assert signals.classify_regime(2.7, 0.50) == "non_sticky"
    assert signals.classify_regime(2.0, 0.60) == "mixed"


def test_compute_coherence():
    """Test directional coherence computation."""
    # Create clustered points
    points = np.random.randn(100, 3)
    points[:50] += [1, 0, 0]  # Cluster in x direction

    coh_star, v_star = signals.compute_coherence(points, num_directions=50)

    # Should have some coherence
    assert 0 <= coh_star <= 1, f"Coherence out of bounds: {coh_star}"
    assert len(v_star) == 3, "Direction vector should be 3D"


def test_compute_compressibility():
    """Test compressibility computation."""
    # Highly compressible data
    data = b"a" * 1000
    r1 = signals.compute_compressibility(data)
    assert r1 < 0.1, f"Expected high compression, got r={r1}"

    # Random data (low compressibility)
    data = np.random.bytes(1000)
    r2 = signals.compute_compressibility(data)
    assert r2 > 0.5, f"Expected low compression, got r={r2}"


def test_compute_budget():
    """Test budget computation."""
    budget = signals.compute_budget(
        D_hat=2.5,
        coh_star=0.70,
        r=0.50,
        alpha=0.30,
        beta=0.50,
        gamma=0.20,
        base=0.10,
        D0=2.2
    )

    # Budget should be in [0, 1]
    assert 0 <= budget <= 1, f"Budget out of bounds: {budget}"


def test_round_9():
    """Test 9-decimal rounding."""
    x = 1.123456789012345
    rounded = signals.round_9(x)
    assert rounded == 1.123456789
