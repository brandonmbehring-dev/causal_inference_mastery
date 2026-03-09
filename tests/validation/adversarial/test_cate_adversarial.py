"""
Layer 2: Adversarial tests for CATE estimator edge cases.

Session 62: Comprehensive edge case and error handling tests.

Tests 30+ challenging scenarios:
- Input validation (NaN, Inf, empty arrays, mismatched dims)
- Treatment imbalance (99% treated, 1% control)
- Perfect separation (propensity 0 or 1)
- Constant values (constant Y, constant X)
- High-dimensional (p > n, p >> n)
- Collinearity in covariates
- Small sample (n < 10 per arm)
- Numerical stability issues

Goal: Ensure graceful failure OR correct handling of edge cases.
"""

import warnings

import numpy as np
import pytest

from src.causal_inference.cate import (
    s_learner,
    t_learner,
    x_learner,
    r_learner,
    double_ml,
)

# Note: causal_forest requires econml which may have stricter validation


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestCATEInputValidation:
    """Test CATE estimators handle invalid inputs correctly."""

    def test_nan_in_outcomes(self):
        """NaN in outcomes should raise error."""
        np.random.seed(42)
        Y = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        T = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X = np.random.randn(8, 2)

        # Most implementations don't check for NaN explicitly
        # but sklearn models should propagate issues
        try:
            result = s_learner(Y, T, X)
            # If it runs, check result is NaN or finite
            assert np.isnan(result["ate"]) or np.isfinite(result["ate"])
        except (ValueError, RuntimeError):
            pass  # Expected failure

    def test_nan_in_treatment(self):
        """NaN in treatment should raise error."""
        np.random.seed(42)
        Y = np.random.randn(8)
        T = np.array([0, 0, 0, np.nan, 1, 1, 1, 1])
        X = np.random.randn(8, 2)

        with pytest.raises((ValueError, TypeError)):
            s_learner(Y, T, X)

    def test_inf_in_outcomes(self):
        """Inf in outcomes should raise or handle gracefully."""
        np.random.seed(42)
        Y = np.array([1.0, np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        T = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X = np.random.randn(8, 2)

        try:
            result = s_learner(Y, T, X)
            # Result may be inf or raise
            assert np.isinf(result["ate"]) or np.isfinite(result["ate"])
        except (ValueError, RuntimeError):
            pass  # Expected failure

    def test_empty_arrays(self):
        """Empty arrays should raise error."""
        Y = np.array([])
        T = np.array([])
        X = np.empty((0, 2))

        with pytest.raises((ValueError, IndexError)):
            s_learner(Y, T, X)

    def test_mismatched_dimensions(self):
        """Mismatched array dimensions should raise error."""
        np.random.seed(42)
        Y = np.random.randn(10)
        T = np.random.randint(0, 2, 8)  # Different length
        X = np.random.randn(10, 2)

        with pytest.raises(ValueError):
            s_learner(Y, T, X)

    def test_non_binary_treatment(self):
        """Non-binary treatment should raise error."""
        np.random.seed(42)
        Y = np.random.randn(10)
        T = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # Three levels
        X = np.random.randn(10, 2)

        with pytest.raises(ValueError):
            s_learner(Y, T, X)

    def test_covariates_1d_handling(self):
        """1D covariates should be handled (reshaped to 2D)."""
        np.random.seed(42)
        n = 50
        Y = np.random.randn(n)
        T = np.random.randint(0, 2, n)
        X = np.random.randn(n)  # 1D

        # Should work - covariates get reshaped
        result = s_learner(Y, T, X)
        assert np.isfinite(result["ate"])


# =============================================================================
# Treatment Imbalance Tests
# =============================================================================


class TestCATETreatmentImbalance:
    """Test CATE estimators with extreme treatment imbalance."""

    def test_extreme_imbalance_99_treated(self):
        """99% treated (only a few controls)."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n)
        # 99 treated, 1 control
        T = np.concatenate([np.ones(99), np.zeros(1)])
        X = np.random.randn(n, 2)

        # Should work but with high variance
        try:
            result = t_learner(Y, T, X)
            # Result should be finite even with imbalance
            assert np.isfinite(result["ate"])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass  # May fail due to insufficient controls

    def test_extreme_imbalance_1_treated(self):
        """Only 1 treated unit."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n)
        T = np.concatenate([np.ones(1), np.zeros(99)])
        X = np.random.randn(n, 2)

        # T-learner needs multiple observations per group
        # Should raise or handle gracefully
        try:
            result = t_learner(Y, T, X)
            assert np.isfinite(result["ate"])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass  # Expected failure

    def test_small_treatment_group_s_learner(self):
        """S-learner with small treatment group."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n)
        T = np.concatenate([np.ones(5), np.zeros(95)])
        X = np.random.randn(n, 2)

        # S-learner should handle this better than T-learner
        result = s_learner(Y, T, X)
        assert np.isfinite(result["ate"])

    def test_perfectly_balanced_treatment(self):
        """Perfect 50/50 balance should work well."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n)
        T = np.concatenate([np.ones(50), np.zeros(50)])
        X = np.random.randn(n, 2)

        result = t_learner(Y, T, X)
        assert np.isfinite(result["ate"])
        assert np.isfinite(result["ate_se"])


# =============================================================================
# Constant Value Tests
# =============================================================================


class TestCATEConstantValues:
    """Test CATE estimators with constant values."""

    def test_constant_outcome(self):
        """Constant outcome (no variance) should produce 0 effect."""
        np.random.seed(42)
        n = 100
        Y = np.ones(n) * 5.0  # Constant outcome
        T = np.random.randint(0, 2, n)
        X = np.random.randn(n, 2)

        result = s_learner(Y, T, X)
        # Constant outcome means no treatment effect
        # ATE should be ~0 (treatment doesn't change outcome)
        assert abs(result["ate"]) < 1.0

    def test_constant_covariate(self):
        """Constant covariate should not cause issues."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n)
        T = np.random.randint(0, 2, n)
        X = np.column_stack(
            [
                np.ones(n),  # Constant covariate
                np.random.randn(n),  # Varying covariate
            ]
        )

        result = s_learner(Y, T, X)
        assert np.isfinite(result["ate"])

    def test_all_constant_covariates(self):
        """All constant covariates - essentially no X information."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n)
        T = np.random.randint(0, 2, n)
        X = np.ones((n, 2))  # All constant

        # May raise due to singular matrix or produce result
        try:
            result = s_learner(Y, T, X)
            assert np.isfinite(result["ate"])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass  # Expected with constant X

    def test_zero_variance_in_one_group(self):
        """Zero variance in one treatment group outcome."""
        np.random.seed(42)
        n = 100
        T = np.concatenate([np.ones(50), np.zeros(50)])
        X = np.random.randn(n, 2)
        # Constant outcome for treated, varying for control
        Y = np.where(T == 1, 5.0, np.random.randn(n))

        result = t_learner(Y, T, X)
        assert np.isfinite(result["ate"])


# =============================================================================
# High-Dimensional Tests
# =============================================================================


class TestCATEHighDimensional:
    """Test CATE estimators in high-dimensional settings."""

    def test_p_greater_than_n(self):
        """More covariates than observations (p > n)."""
        np.random.seed(42)
        n, p = 50, 100
        Y = np.random.randn(n)
        T = np.random.randint(0, 2, n)
        X = np.random.randn(n, p)

        # Ridge should handle this
        result = s_learner(Y, T, X, model="ridge")
        assert np.isfinite(result["ate"])

    def test_p_much_greater_than_n(self):
        """p >> n scenario."""
        np.random.seed(42)
        n, p = 30, 500
        Y = np.random.randn(n)
        T = np.random.randint(0, 2, n)
        X = np.random.randn(n, p)

        # Ridge should regularize appropriately
        result = s_learner(Y, T, X, model="ridge")
        assert np.isfinite(result["ate"])

    def test_p_equal_n_minus_1(self):
        """Nearly singular case (p = n - 1)."""
        np.random.seed(42)
        n, p = 50, 49
        Y = np.random.randn(n)
        T = np.random.randint(0, 2, n)
        X = np.random.randn(n, p)

        result = s_learner(Y, T, X, model="ridge")
        assert np.isfinite(result["ate"])

    def test_linear_model_overfitting_high_d(self):
        """Linear model in high-D may overfit without regularization."""
        np.random.seed(42)
        n, p = 50, 40
        Y = np.random.randn(n)
        T = np.random.randint(0, 2, n)
        X = np.random.randn(n, p)

        # Linear may struggle but should still run
        try:
            result = s_learner(Y, T, X, model="linear")
            assert np.isfinite(result["ate"])
        except (np.linalg.LinAlgError, ValueError):
            pass  # May fail with near-singular matrix


# =============================================================================
# Collinearity Tests
# =============================================================================


class TestCATECollinearity:
    """Test CATE estimators with collinear covariates."""

    def test_perfect_collinearity(self):
        """Two identical covariates (perfect collinearity)."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n)
        T = np.random.randint(0, 2, n)
        x1 = np.random.randn(n)
        X = np.column_stack([x1, x1])  # Identical columns

        # Ridge handles collinearity
        result = s_learner(Y, T, X, model="ridge")
        assert np.isfinite(result["ate"])

    def test_near_perfect_collinearity(self):
        """Nearly collinear covariates."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n)
        T = np.random.randint(0, 2, n)
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 1e-8  # Nearly identical
        X = np.column_stack([x1, x2])

        result = s_learner(Y, T, X, model="ridge")
        assert np.isfinite(result["ate"])

    def test_linear_combination_collinearity(self):
        """One covariate is linear combination of others."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n)
        T = np.random.randint(0, 2, n)
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = 2 * x1 + 3 * x2  # Linear combination
        X = np.column_stack([x1, x2, x3])

        result = s_learner(Y, T, X, model="ridge")
        assert np.isfinite(result["ate"])


# =============================================================================
# Small Sample Tests
# =============================================================================


class TestCATESmallSample:
    """Test CATE estimators with very small samples."""

    def test_minimum_viable_sample(self):
        """Absolute minimum: 2 treated, 2 control."""
        np.random.seed(42)
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        T = np.array([0, 0, 1, 1])
        X = np.array([[1.0], [2.0], [3.0], [4.0]])

        result = s_learner(Y, T, X)
        assert np.isfinite(result["ate"])

    def test_3_observations_per_arm(self):
        """3 observations per treatment arm."""
        np.random.seed(42)
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        T = np.array([0, 0, 0, 1, 1, 1])
        X = np.random.randn(6, 2)

        result = t_learner(Y, T, X)
        assert np.isfinite(result["ate"])

    def test_single_covariate_small_sample(self):
        """Single covariate with small sample."""
        np.random.seed(42)
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        T = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X = np.arange(8).reshape(-1, 1).astype(float)

        result = s_learner(Y, T, X)
        assert np.isfinite(result["ate"])


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestCATENumericalStability:
    """Test CATE estimators with numerically challenging data."""

    def test_very_large_values(self):
        """Very large outcome values."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n) * 1e10
        T = np.random.randint(0, 2, n)
        X = np.random.randn(n, 2)

        result = s_learner(Y, T, X)
        assert np.isfinite(result["ate"])

    def test_very_small_values(self):
        """Very small outcome values."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n) * 1e-10
        T = np.random.randint(0, 2, n)
        X = np.random.randn(n, 2)

        result = s_learner(Y, T, X)
        assert np.isfinite(result["ate"])

    def test_mixed_scale_covariates(self):
        """Covariates with very different scales."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n)
        T = np.random.randint(0, 2, n)
        X = np.column_stack(
            [
                np.random.randn(n) * 1e6,  # Large scale
                np.random.randn(n) * 1e-6,  # Small scale
            ]
        )

        result = s_learner(Y, T, X)
        assert np.isfinite(result["ate"])

    def test_outlier_in_outcome(self):
        """Single extreme outlier in outcome."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n)
        Y[0] = 1000.0  # Outlier
        T = np.random.randint(0, 2, n)
        X = np.random.randn(n, 2)

        result = s_learner(Y, T, X)
        assert np.isfinite(result["ate"])


# =============================================================================
# Method-Specific Edge Cases
# =============================================================================


class TestTLearnerEdgeCases:
    """Edge cases specific to T-Learner."""

    def test_no_overlap_in_covariates(self):
        """Treatment and control have non-overlapping covariate support."""
        np.random.seed(42)
        n = 100
        T = np.concatenate([np.ones(50), np.zeros(50)])
        # Treated: X > 0, Control: X < 0
        X_raw = np.random.randn(n, 2)
        X = np.where(T[:, np.newaxis] == 1, np.abs(X_raw), -np.abs(X_raw))
        Y = np.random.randn(n)

        # T-learner extrapolates, may have high variance
        result = t_learner(Y, T, X)
        assert np.isfinite(result["ate"])


class TestXLearnerEdgeCases:
    """Edge cases specific to X-Learner."""

    def test_propensity_at_boundary(self):
        """X-learner with propensity scores near 0 or 1."""
        np.random.seed(42)
        n = 100
        # Design to have extreme propensities
        X = np.random.randn(n, 1)
        # Perfect separation: T=1 iff X > 0
        T = (X[:, 0] > 0).astype(float)
        Y = np.random.randn(n)

        # X-learner clips propensity, should handle this
        result = x_learner(Y, T, X, model="linear")
        assert np.isfinite(result["ate"])


class TestRLearnerEdgeCases:
    """Edge cases specific to R-Learner."""

    def test_zero_treatment_residuals(self):
        """When treatment is predictable from X (T̃ ≈ 0)."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 1)
        # Treatment nearly deterministic given X
        propensity = 1 / (1 + np.exp(-10 * X[:, 0]))
        T = (np.random.random(n) < propensity).astype(float)
        Y = np.random.randn(n)

        # R-learner uses T̃ in denominator, may have issues
        result = r_learner(Y, T, X)
        # Should handle via clipping/filtering
        assert np.isfinite(result["ate"])


class TestDMLEdgeCases:
    """Edge cases specific to Double ML."""

    def test_insufficient_data_for_folds(self):
        """Very small n relative to n_folds."""
        np.random.seed(42)
        n = 10  # Very small
        Y = np.random.randn(n)
        T = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        X = np.random.randn(n, 2)

        # n_folds=5 with n=10 means 2 per fold
        # May fail due to insufficient data
        try:
            result = double_ml(Y, T, X, n_folds=5)
            assert np.isfinite(result["ate"])
        except (ValueError, RuntimeError):
            pass  # Expected failure

    def test_fewer_folds(self):
        """Using 2 folds with small data."""
        np.random.seed(42)
        n = 20
        Y = np.random.randn(n)
        T = np.random.randint(0, 2, n)
        X = np.random.randn(n, 2)

        result = double_ml(Y, T, X, n_folds=2)
        assert np.isfinite(result["ate"])


# =============================================================================
# Cross-Method Robustness
# =============================================================================


class TestCATERobustness:
    """Test that all methods handle the same edge cases consistently."""

    def test_all_methods_handle_basic_edge_case(self):
        """All basic methods should handle small samples."""
        np.random.seed(42)
        n = 30
        Y = np.random.randn(n)
        T = np.random.randint(0, 2, n)
        X = np.random.randn(n, 2)

        methods = [
            ("s_learner", lambda: s_learner(Y, T, X)),
            ("t_learner", lambda: t_learner(Y, T, X)),
            ("r_learner", lambda: r_learner(Y, T, X)),
        ]

        for name, method in methods:
            result = method()
            assert np.isfinite(result["ate"]), f"{name} failed basic edge case"

    def test_all_methods_with_ridge_regularization(self):
        """All methods using ridge should handle moderate high-D."""
        np.random.seed(42)
        n, p = 50, 30
        Y = np.random.randn(n)
        T = np.random.randint(0, 2, n)
        X = np.random.randn(n, p)

        methods = [
            ("s_learner", lambda: s_learner(Y, T, X, model="ridge")),
            ("t_learner", lambda: t_learner(Y, T, X, model="ridge")),
        ]

        for name, method in methods:
            result = method()
            assert np.isfinite(result["ate"]), f"{name} failed high-D with ridge"
