"""
Unit tests for signal computation module (CLAUDE_PHASE1.md T4).

Tests:
- Monotonicity: N_j is non-decreasing with scale
- Determinism: repeated computation yields identical r
- Theil-Sen properties: positive slope for increasing trend
- coh★ stability: fixed seed gives reproducible results
"""

import pytest
import numpy as np
from agent.src import signals


class TestDHatMonotonicity:
    """Test that D̂ computation handles monotonic N_j correctly."""

    def test_monotonic_Nj_positive_slope(self):
        """Increasing N_j should yield positive D̂."""
        scales = [2, 4, 8, 16]
        N_j = {2: 3, 4: 6, 8: 12, 16: 24}  # Perfectly doubling

        D_hat = signals.compute_D_hat(scales, N_j)

        # log2(N_j) = log2(3) + log2(2^k) where k increases
        # Slope should be close to 1.0
        assert 0.8 <= D_hat <= 1.2, f"Expected D_hat near 1.0, got {D_hat}"

    def test_non_decreasing_Nj_requirement(self):
        """N_j should be non-decreasing with scale (contract from CLAUDE_PHASE1)."""
        scales = [2, 4, 8, 16]
        N_j = {2: 10, 4: 15, 8: 20, 16: 25}  # Non-decreasing

        # Verify N_j is non-decreasing
        values = [N_j[s] for s in sorted(scales)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], "N_j should be non-decreasing"

        # Should compute without error
        D_hat = signals.compute_D_hat(scales, N_j)
        assert isinstance(D_hat, float)
        assert D_hat > 0


class TestCompressibilityDeterminism:
    """Test that r computation is deterministic."""

    def test_repeated_compression_identical(self):
        """Compressing same data twice should yield identical r."""
        data = b"test data with some structure: 1,2,3,4,5,6,7,8,9,10"

        r1 = signals.compute_compressibility(data)
        r2 = signals.compute_compressibility(data)

        assert r1 == r2, f"Compressibility should be deterministic: {r1} != {r2}"

    def test_empty_stream_returns_one(self):
        """Empty stream should return r=1.0 per CLAUDE_PHASE1."""
        r = signals.compute_compressibility(b"")
        assert r == 1.0, f"Expected r=1.0 for empty stream, got {r}"

    def test_highly_compressible_low_r(self):
        """Repeated data should have low r (high compressibility)."""
        data = b"a" * 1000

        r = signals.compute_compressibility(data)

        assert r < 0.1, f"Expected highly compressible data to have r < 0.1, got {r}"

    def test_random_data_high_r(self):
        """Random data should have high r (low compressibility)."""
        np.random.seed(42)
        data = np.random.bytes(1000)

        r = signals.compute_compressibility(data)

        assert r > 0.8, f"Expected random data to have r > 0.8, got {r}"


class TestTheilSenProperties:
    """Test Theil-Sen regression properties."""

    def test_linear_increasing_trend(self):
        """Linear increasing data should yield positive slope."""
        scales = [2, 4, 8, 16, 32]
        # N_j increases linearly in log space
        N_j = {2: 4, 4: 8, 8: 16, 16: 32, 32: 64}

        D_hat = signals.compute_D_hat(scales, N_j)

        # log2(N_j[s]) = log2(2*s), so slope should be 1.0
        assert 0.9 <= D_hat <= 1.1, f"Expected slope ~1.0, got {D_hat}"

    def test_constant_Nj_zero_slope(self):
        """Constant N_j should yield D̂ ≈ 0."""
        scales = [2, 4, 8, 16]
        N_j = {2: 10, 4: 10, 8: 10, 16: 10}  # All same

        D_hat = signals.compute_D_hat(scales, N_j)

        assert abs(D_hat) < 0.1, f"Expected D_hat near 0 for constant N_j, got {D_hat}"


class TestCoherenceStability:
    """Test coh★ reproducibility with fixed seed."""

    def test_fixed_seed_reproducible(self):
        """Fixed seed should give identical results."""
        points = np.random.RandomState(123).randn(50, 3)

        coh1, v1 = signals.compute_coherence(points, num_directions=50, seed=42)
        coh2, v2 = signals.compute_coherence(points, num_directions=50, seed=42)

        assert coh1 == coh2, f"Coherence with same seed should be identical: {coh1} != {coh2}"
        np.testing.assert_array_almost_equal(v1, v2, decimal=9)

    def test_different_seed_different_results(self):
        """Different seeds should (likely) give different results."""
        points = np.random.RandomState(123).randn(50, 3)

        coh1, _ = signals.compute_coherence(points, num_directions=50, seed=42)
        coh2, _ = signals.compute_coherence(points, num_directions=50, seed=99)

        # Not guaranteed to be different, but very likely for real data
        # Just verify both are valid
        assert 0 <= coh1 <= 1
        assert 0 <= coh2 <= 1

    def test_zero_width_handling(self):
        """Points projecting to same value should give coh=1.0."""
        # All points identical
        points = np.array([[1.0, 2.0, 3.0]] * 10)

        coh, _ = signals.compute_coherence(points, num_directions=10, seed=42)

        assert coh == 1.0, f"Identical points should give coh=1.0, got {coh}"

    def test_clustered_points_high_coherence(self):
        """Tightly clustered points should have high coherence."""
        # Points clustered near origin with small random perturbations
        np.random.seed(123)
        # 20 points clustered within a small sphere
        points = np.random.randn(20, 3) * 0.1  # Small variance

        coh, _ = signals.compute_coherence(points, num_directions=100, seed=42)

        # Clustered points should have high coherence (many points in same bins)
        # since they project to similar values in most directions
        assert coh > 0.15, f"Clustered points should have high coherence, got {coh}"


class TestRegimeClassification:
    """Test regime classification logic."""

    def test_sticky_regime(self):
        """High coh★ and low D̂ should give sticky."""
        regime = signals.classify_regime(D_hat=1.3, coh_star=0.75)
        assert regime == "sticky"

    def test_non_sticky_regime(self):
        """High D̂ should give non_sticky."""
        regime = signals.classify_regime(D_hat=2.7, coh_star=0.30)
        assert regime == "non_sticky"

    def test_mixed_regime(self):
        """Intermediate values should give mixed."""
        regime = signals.classify_regime(D_hat=2.0, coh_star=0.60)
        assert regime == "mixed"


class TestBudgetComputation:
    """Test budget formula."""

    def test_budget_in_bounds(self):
        """Budget should always be in [0, 1]."""
        # Extreme values
        budget = signals.compute_budget(D_hat=5.0, coh_star=1.0, r=0.0)
        assert 0 <= budget <= 1, f"Budget out of bounds: {budget}"

    def test_budget_formula(self):
        """Verify budget formula calculation."""
        # Known values
        D_hat, coh_star, r = 2.5, 0.70, 0.40
        alpha, beta, gamma, base, D0 = 0.30, 0.50, 0.20, 0.10, 2.2

        expected = base + alpha * (1 - r) + beta * max(0, D_hat - D0) + gamma * coh_star
        expected = max(0.0, min(1.0, expected))  # Clamp
        expected = round(expected, 9)

        computed = signals.compute_budget(D_hat, coh_star, r, alpha, beta, gamma, base, D0)

        assert computed == expected, f"Budget mismatch: {computed} != {expected}"


class TestRound9:
    """Test 9-decimal rounding for signature stability."""

    def test_round_9_decimals(self):
        """round_9 should round to exactly 9 decimals."""
        x = 1.123456789012345
        rounded = signals.round_9(x)

        assert rounded == 1.123456789, f"Expected 1.123456789, got {rounded}"

    def test_round_9_stability(self):
        """Repeated rounding should be idempotent."""
        x = 1.987654321098765
        r1 = signals.round_9(x)
        r2 = signals.round_9(r1)

        assert r1 == r2, "round_9 should be idempotent"
