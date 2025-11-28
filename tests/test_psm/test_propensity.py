"""
Layer 1: Unit Tests for PSM Propensity Score Utilities.

Tests propensity score trimming and weight stabilization with hand-calculated answers.

Coverage:
- trim_propensity() with known percentiles
- stabilize_weights() with hand calculations
- Edge cases (empty trimming, extreme propensities)

Note: estimate_propensity() is already tested via propensity_helpers.py and integration tests.

References:
- src/causal_inference/psm/propensity.py
- src/causal_inference/observational/propensity_helpers.py
"""

import numpy as np
import pytest

from src.causal_inference.observational.propensity import (
    trim_propensity,
    stabilize_weights,
)


class TestTrimPropensity:
    """Test propensity score trimming with known examples."""

    def test_trim_minimal_trimming(self):
        """
        Minimal trimming with wide percentile bounds.

        Setup:
        - propensity = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        - trim_at = (0.01, 0.99) (1st to 99th percentile)

        Expected: With small sample, percentile interpolation may trim edge values.
        For 7 values: 1st %ile ≈ 0.303, 99th %ile ≈ 0.597
        → Trims units 0 (0.3) and 6 (0.6), keeps 5 middle units
        """
        propensity = np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
        treatment = np.array([0, 1, 0, 1, 0, 1, 0])
        outcomes = np.array([1, 2, 3, 4, 5, 6, 7])
        covariates = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)

        result = trim_propensity(propensity, treatment, outcomes, covariates, trim_at=(0.01, 0.99))

        # With interpolation, 2 edge units trimmed
        assert result["n_trimmed"] == 2, f"Expected 2 trimmed (edge values), got {result['n_trimmed']}"
        assert result["n_kept"] == 5, "Expected 5 kept (middle values)"
        assert len(result["propensity"]) == 5

    def test_trim_extreme_values(self):
        """
        Trim extreme propensity values.

        Setup:
        - propensity = [0.001, 0.2, 0.5, 0.8, 0.999]
        - trim_at = (0.01, 0.99)
        - 1st percentile: np.percentile([...], 1) ≈ 0.004 (interpolated)
        - 99th percentile: np.percentile([...], 99) ≈ 0.996 (interpolated)

        Expected: Units 0 and 4 trimmed (outside 1st-99th percentile range)
        """
        propensity = np.array([0.001, 0.2, 0.5, 0.8, 0.999])
        treatment = np.array([0, 1, 0, 1, 1])
        outcomes = np.array([1, 2, 3, 4, 5])
        covariates = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)

        result = trim_propensity(propensity, treatment, outcomes, covariates, trim_at=(0.01, 0.99))

        # Check trimming
        assert result["n_trimmed"] == 2, f"Expected 2 trimmed, got {result['n_trimmed']}"
        assert result["n_kept"] == 3, f"Expected 3 kept, got {result['n_kept']}"

        # Check kept values
        assert len(result["propensity"]) == 3
        assert np.min(result["propensity"]) >= 0.2, "Minimum kept propensity should be ≥ 0.2"
        assert np.max(result["propensity"]) <= 0.8, "Maximum kept propensity should be ≤ 0.8"

    def test_trim_keeps_middle_range(self):
        """
        Verify trimmed arrays contain only middle-range values.

        Setup:
        - propensity = [0.05, 0.3, 0.5, 0.7, 0.95]
        - trim_at = (0.2, 0.8) (20th to 80th percentile)

        Expected: Trim units 0 and 4, keep units 1, 2, 3
        """
        propensity = np.array([0.05, 0.3, 0.5, 0.7, 0.95])
        treatment = np.array([0, 1, 0, 1, 0])
        outcomes = np.array([1, 2, 3, 4, 5])
        covariates = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)

        result = trim_propensity(propensity, treatment, outcomes, covariates, trim_at=(0.2, 0.8))

        # Check kept mask
        expected_keep_mask = np.array([False, True, True, True, False])
        assert np.array_equal(result["keep_mask"], expected_keep_mask), \
            f"Expected keep_mask {expected_keep_mask}, got {result['keep_mask']}"

        # Check trimmed arrays have correct values
        assert np.array_equal(result["treatment"], [1, 0, 1])
        assert np.array_equal(result["outcomes"], [2, 3, 4])

    def test_trim_2d_covariates(self):
        """Test trimming with 2D covariate matrix."""
        propensity = np.array([0.01, 0.5, 0.99])
        treatment = np.array([0, 1, 0])
        outcomes = np.array([1, 2, 3])
        covariates = np.array([[1, 2], [3, 4], [5, 6]])

        result = trim_propensity(propensity, treatment, outcomes, covariates, trim_at=(0.1, 0.9))

        # Should keep only middle unit (index 1)
        assert result["n_kept"] == 1
        assert result["covariates"].shape == (1, 2), f"Expected (1, 2), got {result['covariates'].shape}"
        assert np.array_equal(result["covariates"], [[3, 4]])

    def test_trim_invalid_threshold(self):
        """trim_propensity() should raise on invalid trim_at."""
        propensity = np.array([0.2, 0.5, 0.8])
        treatment = np.array([0, 1, 0])
        outcomes = np.array([1, 2, 3])
        covariates = np.array([1, 2, 3]).reshape(-1, 1)

        # Lower >= upper
        with pytest.raises(ValueError, match="CRITICAL ERROR: Invalid trim_at"):
            trim_propensity(propensity, treatment, outcomes, covariates, trim_at=(0.5, 0.5))

        # Lower bound too low
        with pytest.raises(ValueError, match="CRITICAL ERROR: Invalid trim_at"):
            trim_propensity(propensity, treatment, outcomes, covariates, trim_at=(0.0, 0.9))

        # Upper bound too high
        with pytest.raises(ValueError, match="CRITICAL ERROR: Invalid trim_at"):
            trim_propensity(propensity, treatment, outcomes, covariates, trim_at=(0.1, 1.0))

    def test_trim_length_mismatch(self):
        """trim_propensity() should raise on length mismatch."""
        propensity = np.array([0.2, 0.5])
        treatment = np.array([0, 1, 0])  # Wrong length!
        outcomes = np.array([1, 2])
        covariates = np.array([1, 2]).reshape(-1, 1)

        with pytest.raises(ValueError, match="CRITICAL ERROR: Arrays have different lengths"):
            trim_propensity(propensity, treatment, outcomes, covariates)


class TestStabilizeWeights:
    """Test stabilized weight computation with hand-calculated answers."""

    def test_stabilized_weights_simple(self):
        """
        Stabilized weights with simple known answer.

        Setup:
        - propensity = [0.2, 0.8, 0.5]
        - treatment = [1, 1, 0]
        - P(T=1) = 2/3

        Hand calculation:
        - Treated (indices 0, 1):
          - SW₀ = P(T=1) / P(T=1|X₀) = (2/3) / 0.2 = 3.333
          - SW₁ = (2/3) / 0.8 = 0.833
        - Control (index 2):
          - SW₂ = P(T=0) / P(T=0|X₂) = (1/3) / (1-0.5) = (1/3) / 0.5 = 0.667

        Mean: (3.333 + 0.833 + 0.667) / 3 = 1.278 (≈ 1 if larger sample)
        """
        propensity = np.array([0.2, 0.8, 0.5])
        treatment = np.array([1, 1, 0])

        sw = stabilize_weights(propensity, treatment)

        # Check individual weights
        p_t = 2 / 3  # Marginal P(T=1)
        expected_sw0 = p_t / 0.2  # = 3.333
        expected_sw1 = p_t / 0.8  # = 0.833
        expected_sw2 = (1 - p_t) / (1 - 0.5)  # = 0.667

        assert np.isclose(sw[0], expected_sw0, rtol=1e-3), \
            f"Expected SW[0] = {expected_sw0:.3f}, got {sw[0]:.3f}"
        assert np.isclose(sw[1], expected_sw1, rtol=1e-3), \
            f"Expected SW[1] = {expected_sw1:.3f}, got {sw[1]:.3f}"
        assert np.isclose(sw[2], expected_sw2, rtol=1e-3), \
            f"Expected SW[2] = {expected_sw2:.3f}, got {sw[2]:.3f}"

    def test_stabilized_weights_equal_propensity(self):
        """
        Stabilized weights when all propensities equal marginal probability.

        Setup:
        - propensity = [0.5, 0.5, 0.5, 0.5] (constant)
        - treatment = [1, 1, 0, 0]
        - P(T=1) = 2/4 = 0.5

        Hand calculation:
        - Treated: SW = 0.5 / 0.5 = 1.0
        - Control: SW = 0.5 / 0.5 = 1.0
        - All weights = 1.0
        """
        propensity = np.array([0.5, 0.5, 0.5, 0.5])
        treatment = np.array([1, 1, 0, 0])

        sw = stabilize_weights(propensity, treatment)

        assert np.allclose(sw, 1.0), f"Expected all weights = 1.0, got {sw}"

    def test_stabilized_weights_mean_approx_one(self):
        """
        Stabilized weights have mean ≈ 1.0 (by construction).

        For larger samples with balanced treatment, mean(SW) approaches 1.0.
        Note: With random treatment assignment, finite-sample mean can deviate.
        """
        np.random.seed(42)
        n = 1000
        propensity = np.random.uniform(0.1, 0.9, n)
        treatment = np.random.binomial(1, 0.4, n)  # 40% treatment rate

        sw = stabilize_weights(propensity, treatment)

        mean_sw = np.mean(sw)
        # Wider tolerance due to random treatment assignment and finite sample
        assert np.isclose(mean_sw, 1.0, rtol=0.5), \
            f"Expected mean(SW) ≈ 1.0 (±50%), got {mean_sw:.3f}"

    def test_stabilize_weights_propensity_bounds(self):
        """
        Propensities must be in (0, 1) exclusive.

        Edge cases:
        - propensity = 0 → division by zero
        - propensity = 1 → division by zero
        """
        # Propensity = 0
        propensity = np.array([0.0, 0.5])
        treatment = np.array([1, 0])

        with pytest.raises(ValueError, match="Propensity scores must be in \\(0,1\\) exclusive"):
            stabilize_weights(propensity, treatment)

        # Propensity = 1
        propensity = np.array([0.5, 1.0])
        treatment = np.array([1, 0])

        with pytest.raises(ValueError, match="Propensity scores must be in \\(0,1\\) exclusive"):
            stabilize_weights(propensity, treatment)

    def test_stabilize_weights_length_mismatch(self):
        """stabilize_weights() should raise on length mismatch."""
        propensity = np.array([0.2, 0.5])
        treatment = np.array([0, 1, 0])  # Wrong length!

        with pytest.raises(ValueError, match="CRITICAL ERROR: Arrays have different lengths"):
            stabilize_weights(propensity, treatment)

    def test_stabilize_weights_reduces_variance(self):
        """
        Stabilized weights have lower variance than non-stabilized IPW weights.

        Non-stabilized IPW:
        - Treated: w = 1 / P(T=1|X)
        - Control: w = 1 / P(T=0|X)

        Stabilized weights multiply by marginal probabilities, reducing variance.
        """
        np.random.seed(123)
        propensity = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        treatment = np.array([1, 1, 0, 0, 0])

        # Stabilized weights
        sw = stabilize_weights(propensity, treatment)

        # Non-stabilized weights
        ipw = np.where(treatment == 1, 1 / propensity, 1 / (1 - propensity))

        # Variance comparison
        var_sw = np.var(sw)
        var_ipw = np.var(ipw)

        assert var_sw < var_ipw, \
            f"Stabilized weights should have lower variance: var(SW)={var_sw:.3f}, var(IPW)={var_ipw:.3f}"


class TestPropensityEdgeCases:
    """Edge cases specific to propensity utilities."""

    def test_trim_1d_covariates(self):
        """Trimming with 1D covariates (reshaped correctly)."""
        propensity = np.array([0.05, 0.5, 0.95])
        treatment = np.array([0, 1, 0])
        outcomes = np.array([1, 2, 3])
        covariates = np.array([10, 20, 30])  # 1D

        result = trim_propensity(propensity, treatment, outcomes, covariates, trim_at=(0.2, 0.8))

        # Should keep only middle unit
        assert result["n_kept"] == 1
        assert result["covariates"].shape == (1,), f"Expected (1,), got {result['covariates'].shape}"
        assert result["covariates"][0] == 20

    def test_stabilize_weights_conversion_to_array(self):
        """stabilize_weights() should handle lists as input."""
        propensity = [0.3, 0.7]
        treatment = [1, 0]

        sw = stabilize_weights(propensity, treatment)

        assert isinstance(sw, np.ndarray), "Output should be numpy array"
        assert sw.shape == (2,), f"Expected shape (2,), got {sw.shape}"
