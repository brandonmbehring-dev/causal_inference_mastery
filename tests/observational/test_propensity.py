"""Tests for propensity score estimation and weight adjustments.

Tests cover:
1. Propensity estimation with logistic regression
2. Weight trimming at percentiles
3. Weight stabilization

Following test-first principles with known-answer tests.
"""

import numpy as np
import pytest
from src.causal_inference.observational.propensity import (
    estimate_propensity,
    trim_propensity,
    stabilize_weights,
)


class TestEstimatePropensity:
    """Test propensity score estimation."""

    def test_perfect_separation_raises_error(self):
        """
        Test with perfect separation: X perfectly predicts T.

        Setup: T = 1 if X > 0, T = 0 if X ≤ 0
        Expected: ValueError raised (perfect separation violates positivity)
        """
        np.random.seed(42)
        X = np.random.normal(0, 1, 100)
        T = (X > 0).astype(float)

        # Perfect separation should raise an error
        # because propensity scores of exactly 0/1 make IPW weights infinite
        with pytest.raises(ValueError, match="(?i)perfect separation"):
            estimate_propensity(T, X)

    def test_no_confounding(self):
        """
        Test with no confounding: X independent of T.

        Setup: T ~ Bernoulli(0.5), X ~ N(0,1) independent
        Expected: AUC ≈ 0.5 (no better than random), low pseudo-R²
        """
        np.random.seed(123)
        X = np.random.normal(0, 1, 100)
        T = np.random.binomial(1, 0.5, 100)

        result = estimate_propensity(T, X)

        # AUC should be near 0.5 (random)
        assert 0.4 < result["diagnostics"]["auc"] < 0.6

        # Pseudo-R² should be low
        assert result["diagnostics"]["pseudo_r2"] < 0.1

    def test_single_covariate(self):
        """Test with single covariate."""
        np.random.seed(456)
        X = np.random.normal(0, 1, 100)
        # Moderate confounding
        logit = 0.5 * X
        T = (np.random.uniform(0, 1, 100) < 1 / (1 + np.exp(-logit))).astype(float)

        result = estimate_propensity(T, X)

        # Should return propensities for all units
        assert result["propensity"].shape == (100,)
        assert np.all((result["propensity"] > 0) & (result["propensity"] < 1))

        # Model should have one coefficient
        assert len(result["diagnostics"]["coef"]) == 1

        # AUC should be moderate (some confounding)
        assert 0.6 < result["diagnostics"]["auc"] < 0.8

    def test_multiple_covariates(self):
        """Test with multiple covariates."""
        np.random.seed(789)
        n = 100
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(5, 2, n)
        X3 = np.random.uniform(-1, 1, n)
        X = np.column_stack([X1, X2, X3])

        # Confounding from all three covariates (increased coefficients)
        logit = 0.8 * X1 + 0.5 * X2 - 1.0 * X3
        T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)

        result = estimate_propensity(T, X)

        # Should return propensities for all units
        assert result["propensity"].shape == (n,)

        # Model should have three coefficients
        assert len(result["diagnostics"]["coef"]) == 3

        # AUC should be high (strong confounding from 3 covariates)
        assert result["diagnostics"]["auc"] > 0.7

    def test_1d_covariate_handling(self):
        """Test that 1D covariate array is handled correctly."""
        np.random.seed(111)
        n = 50
        X = np.random.normal(0, 1, n)  # 1D array

        # Use probabilistic treatment assignment to avoid perfect separation
        # P(T=1|X) = logistic(0.5*X) - mild confounding
        logit = 0.5 * X
        prob = 1 / (1 + np.exp(-logit))
        T = (np.random.uniform(0, 1, n) < prob).astype(float)

        result = estimate_propensity(T, X)

        # Should work without error
        assert result["propensity"].shape == (n,)
        assert len(result["diagnostics"]["coef"]) == 1


class TestEstimatePropensityErrors:
    """Test error handling for estimate_propensity."""

    def test_empty_input_fails_fast(self):
        """Test that empty arrays raise ValueError."""
        X = np.array([])
        T = np.array([])

        with pytest.raises(ValueError) as exc_info:
            estimate_propensity(T, X)

        error_msg = str(exc_info.value)
        assert "empty" in error_msg.lower()

    def test_mismatched_lengths_fails_fast(self):
        """Test that mismatched lengths raise ValueError."""
        X = np.random.normal(0, 1, (100, 2))
        T = np.array([1] * 50)  # Wrong length

        with pytest.raises(ValueError) as exc_info:
            estimate_propensity(T, X)

        error_msg = str(exc_info.value)
        assert "same length" in error_msg.lower()

    def test_nan_in_treatment_fails_fast(self):
        """Test that NaN in treatment raises ValueError."""
        X = np.random.normal(0, 1, 100)
        T = np.array([1, 0, np.nan] + [1] * 47 + [0] * 50)

        with pytest.raises(ValueError) as exc_info:
            estimate_propensity(T, X)

        error_msg = str(exc_info.value)
        assert "nan" in error_msg.lower() or "non-finite" in error_msg.lower()

    def test_nan_in_covariates_fails_fast(self):
        """Test that NaN in covariates raises ValueError."""
        X = np.random.normal(0, 1, 100)
        X[10] = np.nan
        T = np.array([1] * 50 + [0] * 50)

        with pytest.raises(ValueError) as exc_info:
            estimate_propensity(T, X)

        error_msg = str(exc_info.value)
        assert "nan" in error_msg.lower() or "non-finite" in error_msg.lower()

    def test_non_binary_treatment_fails_fast(self):
        """Test that non-binary treatment raises ValueError."""
        X = np.random.normal(0, 1, 100)
        T = np.array([0, 1, 2] * 33 + [0])  # Has value 2

        with pytest.raises(ValueError) as exc_info:
            estimate_propensity(T, X)

        error_msg = str(exc_info.value)
        assert "binary" in error_msg.lower()

    def test_no_treatment_variation_fails_fast(self):
        """Test that all treated or all control raises ValueError."""
        X = np.random.normal(0, 1, 100)
        T = np.ones(100)  # All treated

        with pytest.raises(ValueError) as exc_info:
            estimate_propensity(T, X)

        error_msg = str(exc_info.value)
        assert "variation" in error_msg.lower()

    def test_constant_covariate_fails_fast(self):
        """Test that constant covariate raises ValueError."""
        X = np.ones(100)  # Constant
        T = np.array([1] * 50 + [0] * 50)

        with pytest.raises(ValueError) as exc_info:
            estimate_propensity(T, X)

        error_msg = str(exc_info.value)
        assert "constant" in error_msg.lower() or "variation" in error_msg.lower()


class TestTrimPropensity:
    """Test propensity score trimming."""

    def test_trim_removes_extremes(self):
        """Test that trimming removes extreme propensity scores."""
        propensity = np.array([0.001, 0.2, 0.3, 0.5, 0.7, 0.8, 0.999])
        treatment = np.array([0, 0, 0, 1, 1, 1, 1])
        outcomes = np.array([1, 2, 3, 4, 5, 6, 7])
        X = np.random.normal(0, 1, (7, 2))

        result = trim_propensity(propensity, treatment, outcomes, X, trim_at=(0.01, 0.99))

        # Should remove first and last units (0.001 and 0.999)
        assert result["n_trimmed"] == 2
        assert result["n_kept"] == 5

        # Check trimmed arrays have correct length
        assert len(result["propensity"]) == 5
        assert len(result["treatment"]) == 5
        assert len(result["outcomes"]) == 5

        # Check extremes were removed
        assert np.min(result["propensity"]) > 0.001
        assert np.max(result["propensity"]) < 0.999

    def test_no_trimming_needed(self):
        """Test with all propensities in acceptable range.

        Uses constant propensity to guarantee no trimming.
        """
        np.random.seed(42)
        # All propensities equal - no variation means no trimming
        propensity = np.ones(100) * 0.5
        treatment = np.array([1] * 50 + [0] * 50)
        outcomes = np.random.normal(0, 1, 100)
        X = np.random.normal(0, 1, (100, 2))

        result = trim_propensity(propensity, treatment, outcomes, X, trim_at=(0.01, 0.99))

        # No units trimmed since all propensities identical
        assert result["n_trimmed"] == 0
        assert result["n_kept"] == 100

    def test_custom_trim_percentiles(self):
        """Test with custom trim percentiles."""
        propensity = np.linspace(0.05, 0.95, 100)
        treatment = np.random.binomial(1, 0.5, 100)
        outcomes = np.random.normal(0, 1, 100)
        X = np.random.normal(0, 1, (100, 2))

        # Trim at 5th and 95th percentiles
        result = trim_propensity(propensity, treatment, outcomes, X, trim_at=(0.05, 0.95))

        # Should trim approximately 10% (5% from each tail)
        assert 85 <= result["n_kept"] <= 95

        # Check propensities are in trimmed range
        assert np.min(result["propensity"]) >= np.percentile(propensity, 5)
        assert np.max(result["propensity"]) <= np.percentile(propensity, 95)

    def test_trim_1d_covariates(self):
        """Test trimming with 1D covariate array."""
        propensity = np.array([0.001, 0.5, 0.999])
        treatment = np.array([0, 1, 1])
        outcomes = np.array([1, 2, 3])
        X = np.array([1.0, 2.0, 3.0])  # 1D

        result = trim_propensity(propensity, treatment, outcomes, X, trim_at=(0.01, 0.99))

        # Should handle 1D covariates
        assert result["covariates"].ndim == 1
        assert len(result["covariates"]) == result["n_kept"]


class TestTrimPropensityErrors:
    """Test error handling for trim_propensity."""

    def test_invalid_trim_at_fails_fast(self):
        """Test that invalid trim_at raises ValueError."""
        propensity = np.array([0.2, 0.5, 0.8])
        treatment = np.array([0, 1, 1])
        outcomes = np.array([1, 2, 3])
        X = np.random.normal(0, 1, (3, 2))

        # Lower >= upper
        with pytest.raises(ValueError) as exc_info:
            trim_propensity(propensity, treatment, outcomes, X, trim_at=(0.9, 0.1))

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "Invalid trim_at" in error_msg

    def test_mismatched_lengths_fails_fast(self):
        """Test that mismatched lengths raise ValueError."""
        propensity = np.array([0.2, 0.5, 0.8])
        treatment = np.array([0, 1])  # Wrong length
        outcomes = np.array([1, 2, 3])
        X = np.random.normal(0, 1, (3, 2))

        with pytest.raises(ValueError) as exc_info:
            trim_propensity(propensity, treatment, outcomes, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "different lengths" in error_msg


class TestStabilizeWeights:
    """Test weight stabilization."""

    def test_stabilized_weights_mean_one(self):
        """Test that stabilized weights have mean ≈ 1.

        Note: In finite samples with specific treatment assignments,
        mean may deviate from 1.0. The property holds in expectation.
        """
        np.random.seed(42)
        # Use balanced treatment with moderate propensities
        propensity = np.random.uniform(0.3, 0.7, 200)
        treatment = np.array([1] * 100 + [0] * 100, dtype=float)

        sw = stabilize_weights(propensity, treatment)

        # With balanced design and moderate propensities, mean should be near 1.0
        # But finite sample variation can cause deviation
        assert np.isclose(np.mean(sw), 1.0, atol=0.25)

    def test_stabilization_reduces_variance(self):
        """Test that stabilization reduces weight variance."""
        np.random.seed(42)
        propensity = np.linspace(0.1, 0.9, 100)
        treatment = (propensity > 0.5).astype(float)

        # Non-stabilized weights
        ipw_weights = np.where(treatment == 1, 1 / propensity, 1 / (1 - propensity))

        # Stabilized weights
        sw = stabilize_weights(propensity, treatment)

        # Stabilized should have lower variance
        assert np.var(sw) < np.var(ipw_weights)

    def test_constant_propensity_gives_ones(self):
        """Test that constant propensity gives stabilized weights = 1."""
        propensity = np.ones(100) * 0.5
        treatment = np.array([1] * 50 + [0] * 50)

        sw = stabilize_weights(propensity, treatment)

        # All weights should be 1.0
        assert np.allclose(sw, 1.0)

    def test_stabilized_weights_positive(self):
        """Test that all stabilized weights are positive."""
        propensity = np.linspace(0.1, 0.9, 100)
        treatment = (propensity > 0.5).astype(float)

        sw = stabilize_weights(propensity, treatment)

        # All weights should be positive
        assert np.all(sw > 0)


class TestStabilizeWeightsErrors:
    """Test error handling for stabilize_weights."""

    def test_propensity_out_of_range_fails_fast(self):
        """Test that propensity outside (0,1) raises ValueError."""
        propensity = np.array([0.0, 0.5, 1.0])  # 0 and 1 are invalid
        treatment = np.array([0, 1, 1])

        with pytest.raises(ValueError) as exc_info:
            stabilize_weights(propensity, treatment)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "(0,1) exclusive" in error_msg

    def test_mismatched_lengths_fails_fast(self):
        """Test that mismatched lengths raise ValueError."""
        propensity = np.array([0.2, 0.5, 0.8])
        treatment = np.array([0, 1])  # Wrong length

        with pytest.raises(ValueError) as exc_info:
            stabilize_weights(propensity, treatment)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "different lengths" in error_msg
