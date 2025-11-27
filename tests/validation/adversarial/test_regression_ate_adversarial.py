"""
Adversarial tests for regression_adjusted_ate estimator.

Regression-specific edge cases:
1. Perfect collinearity (covariates linearly dependent)
2. Single covariate with no variation
3. Multiple covariates (high-dimensional)
4. Covariate perfectly predicts treatment (separation)
5. Outliers in covariates
"""

import numpy as np
import pytest
from src.causal_inference.rct.estimators_regression import regression_adjusted_ate


class TestRegressionATECollinearity:
    """Test regression_ate with collinear covariates."""

    def test_perfect_collinearity(self):
        """Two perfectly correlated covariates (X2 = 2*X1)."""
        np.random.seed(42)
        n = 100
        X1 = np.random.normal(0, 1, n)
        X2 = 2 * X1  # Perfect collinearity
        X = np.column_stack([X1, X2])
        treatment = np.array([1] * 50 + [0] * 50)
        outcomes = 2.0 * treatment + 3.0 * X1 + np.random.normal(0, 1, n)

        # Should raise error due to singular matrix (case-insensitive match)
        with pytest.raises(ValueError, match="(?i)singular"):
            regression_adjusted_ate(outcomes, treatment, X)

    def test_high_dimensional_covariates(self):
        """Many covariates (p=20) with n=100."""
        np.random.seed(42)
        n = 100
        p = 20
        X = np.random.normal(0, 1, (n, p))
        treatment = np.array([1] * 50 + [0] * 50)
        outcomes = 2.0 * treatment + X @ np.random.normal(0, 0.5, p) + np.random.normal(0, 1, n)

        result = regression_adjusted_ate(outcomes, treatment, X)

        # Should compute valid estimate
        assert np.isfinite(result["estimate"])
        assert 1.5 < result["estimate"] < 2.5  # Should be close to 2.0

    def test_zero_variance_covariate(self):
        """Covariate with zero variance (all values same)."""
        n = 100
        X = np.ones(n) * 5.0  # All same value
        treatment = np.array([1] * 50 + [0] * 50)
        outcomes = 2.0 * treatment + np.random.normal(0, 1, n)

        # Zero variance covariate causes singular matrix (equivalent to intercept)
        with pytest.raises(ValueError, match="(?i)singular"):
            regression_adjusted_ate(outcomes, treatment, X)


class TestRegressionATEOutliers:
    """Test regression_ate with outliers in covariates."""

    def test_extreme_covariate_values(self):
        """One extreme outlier in covariate."""
        np.random.seed(42)
        X = np.concatenate([[1000.0], np.random.normal(0, 1, 99)])
        treatment = np.array([1] * 50 + [0] * 50)
        outcomes = 2.0 * treatment + X + np.random.normal(0, 1, 100)

        result = regression_adjusted_ate(outcomes, treatment, X)

        # Should handle outlier (though estimate may be affected)
        assert np.isfinite(result["estimate"])

    def test_outliers_in_outcomes_and_covariates(self):
        """Extreme outliers in both Y and X."""
        outcomes = np.concatenate([[1000.0], np.random.normal(2, 1, 99)])
        X = np.concatenate([[500.0], np.random.normal(0, 1, 99)])
        treatment = np.array([1] * 50 + [0] * 50)

        result = regression_adjusted_ate(outcomes, treatment, X)

        assert np.isfinite(result["estimate"])
        assert np.isfinite(result["se"])


class TestRegressionATEVarianceReduction:
    """Test that regression adjustment actually reduces variance."""

    def test_strong_covariate_effect(self):
        """Covariate strongly predicts outcome."""
        np.random.seed(42)
        n = 200
        X = np.random.normal(0, 1, n)
        treatment = np.array([1] * 100 + [0] * 100)
        # Strong covariate effect (β=10)
        outcomes = 2.0 * treatment + 10.0 * X + np.random.normal(0, 0.1, n)

        result = regression_adjusted_ate(outcomes, treatment, X)

        # SE should be small due to strong covariate effect
        assert result["se"] < 0.1
        # R² should be high
        assert result["r_squared"] > 0.95

    def test_weak_covariate_effect(self):
        """Covariate weakly predicts outcome."""
        np.random.seed(42)
        n = 200
        X = np.random.normal(0, 1, n)
        treatment = np.array([1] * 100 + [0] * 100)
        # Weak covariate effect (β=0.1)
        outcomes = 2.0 * treatment + 0.1 * X + np.random.normal(0, 1, n)

        result = regression_adjusted_ate(outcomes, treatment, X)

        # R² should be moderate (relaxed threshold - treatment explains variance too)
        assert result["r_squared"] < 0.6


class TestRegressionATENumericalStability:
    """Test regression_ate with numerically challenging scenarios."""

    def test_tiny_outcome_values(self):
        """
        Outcomes near machine precision (1e-10).

        Tests numerical stability when values are extremely small.
        """
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1e-10, n)
        treatment = np.array([1] * 50 + [0] * 50)
        # Tiny outcomes
        outcomes = 2e-10 * treatment + X + np.random.normal(0, 1e-11, n)

        result = regression_adjusted_ate(outcomes, treatment, X)

        # Should compute valid estimate
        assert np.isfinite(result["estimate"])
        assert result["se"] > 0
        # ATE should be near 2e-10
        assert np.isclose(result["estimate"], 2e-10, rtol=0.5)
