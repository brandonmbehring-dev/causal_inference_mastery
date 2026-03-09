"""Test regression-adjusted ATE estimator.

Regression adjustment (ANCOVA) improves precision by controlling for pre-treatment
covariates. Under randomization, it reduces variance without biasing the estimate.

Key principle: Y = alpha + tau*T + beta*X + epsilon, where tau is the ATE.

Following test-first development: write tests before implementation.
"""

import numpy as np
import pytest

from src.causal_inference.rct.estimators import simple_ate
from src.causal_inference.rct.estimators_regression import regression_adjusted_ate


class TestRegressionAdjustedKnownAnswers:
    """Test regression_adjusted_ate with hand-calculated values."""

    def test_regression_adjustment_reduces_variance(self):
        """
        Test that regression adjustment reduces SE compared to simple difference-in-means.

        Setup: Create data with strong covariate correlation to outcomes.
        - X is strongly correlated with Y (beta = 10)
        - Treatment effect = 5 (constant)
        - Without adjustment: High variance from X variation
        - With adjustment: Lower variance (control for X)
        """
        np.random.seed(123)  # Different seed for balanced treatment assignment
        n = 200  # Larger n for more stable estimates

        # Strong covariate
        X = np.random.normal(10, 5, n)

        # Deterministic balanced treatment for stable test
        treatment = np.array([1, 0] * (n // 2))
        np.random.shuffle(treatment)

        # Y = 5*T + 10*X + noise (strong X effect)
        outcomes = 5 * treatment + 10 * X + np.random.normal(0, 1, n)

        # Estimate both ways
        simple_result = simple_ate(outcomes, treatment)
        adjusted_result = regression_adjusted_ate(outcomes, treatment, X)

        # Both should estimate ~5.0 (generous tolerance for stochastic test)
        assert np.isclose(simple_result["estimate"], 5.0, atol=3.0)
        assert np.isclose(adjusted_result["estimate"], 5.0, atol=3.0)

        # KEY TEST: Adjusted should have smaller SE (variance reduction from controlling X)
        # This is the main point - regression adjustment removes X variation
        assert adjusted_result["se"] < simple_result["se"], (
            f"Adjusted SE ({adjusted_result['se']}) should be < simple SE ({simple_result['se']})"
        )

    def test_regression_adjusted_known_answer(self):
        """
        Test with hand-calculated regression-adjusted ATE.

        Data: Y = 2 + 5*T + 3*X + 0 (no noise)
        - Control (T=0): Y = 2 + 3*X
        - Treated (T=1): Y = 7 + 3*X
        - True ATE = 5 (constant treatment effect)

        Regression should recover tau = 5 exactly.
        """
        # Perfect data with no noise
        X = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
        treatment = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        outcomes = 2 + 5 * treatment + 3 * X  # Y = 2 + 5*T + 3*X

        result = regression_adjusted_ate(outcomes, treatment, X)

        # Should recover exact ATE = 5 (no noise)
        assert np.isclose(result["estimate"], 5.0, atol=1e-10), (
            f"Expected ATE=5.0, got {result['estimate']}"
        )

        # Should have coefficient for X
        assert "covariate_coef" in result
        assert np.isclose(result["covariate_coef"], 3.0, atol=1e-10)

    def test_regression_with_multiple_covariates(self):
        """
        Test regression adjustment with multiple covariates.

        Y = 1 + 4*T + 2*X1 + 3*X2 + 0
        Should recover tau = 4 and beta = [2, 3].
        """
        # Multiple covariates
        X1 = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        X2 = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0])
        X = np.column_stack([X1, X2])

        treatment = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        outcomes = 1 + 4 * treatment + 2 * X1 + 3 * X2

        result = regression_adjusted_ate(outcomes, treatment, X)

        # Should recover exact ATE = 4
        assert np.isclose(result["estimate"], 4.0, atol=1e-10)

        # Should have coefficients for both covariates
        assert "covariate_coef" in result
        assert len(result["covariate_coef"]) == 2
        assert np.isclose(result["covariate_coef"][0], 2.0, atol=1e-10)
        assert np.isclose(result["covariate_coef"][1], 3.0, atol=1e-10)

    def test_no_covariate_variation_detected(self):
        """
        Test that constant covariate (no variation) causes singular matrix error.

        When X is constant (all same value), it's collinear with intercept.
        This should be detected and raise an error.
        """
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        X = np.ones(4)  # Constant covariate (collinear with intercept)

        with pytest.raises(ValueError) as exc_info:
            regression_adjusted_ate(outcomes, treatment, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "singular" in error_msg.lower() or "collinearity" in error_msg.lower()

    def test_zero_treatment_effect_recovery(self):
        """
        Test that regression adjustment recovers zero ATE when tau = 0.

        Setup: Y = 2 + 0*T + 3*X (no treatment effect)
        Should recover tau = 0 exactly.
        """
        X = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
        treatment = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        outcomes = 2 + 0 * treatment + 3 * X  # tau = 0

        result = regression_adjusted_ate(outcomes, treatment, X)

        # Should recover exact ATE = 0
        assert np.isclose(result["estimate"], 0.0, atol=1e-10), (
            f"Expected ATE=0.0, got {result['estimate']}"
        )

        # Should have coefficient for X
        assert "covariate_coef" in result
        assert np.isclose(result["covariate_coef"], 3.0, atol=1e-10)


class TestRegressionAdjustedErrorHandling:
    """Test error handling for regression_adjusted_ate."""

    def test_empty_input_fails_fast(self):
        """Test that empty input arrays raise ValueError."""
        treatment = np.array([])
        outcomes = np.array([])
        X = np.array([])

        with pytest.raises(ValueError) as exc_info:
            regression_adjusted_ate(outcomes, treatment, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "empty" in error_msg.lower()

    def test_invalid_alpha_fails_fast(self):
        """Test that invalid alpha raises ValueError."""
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        X = np.array([1.0, 2.0, 3.0, 4.0])

        # alpha = 0 (invalid)
        with pytest.raises(ValueError) as exc_info:
            regression_adjusted_ate(outcomes, treatment, X, alpha=0.0)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "alpha" in error_msg.lower()

        # alpha = 1 (invalid)
        with pytest.raises(ValueError) as exc_info:
            regression_adjusted_ate(outcomes, treatment, X, alpha=1.0)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "alpha" in error_msg.lower()

    def test_all_treated_fails_fast(self):
        """Test that data with no control units raises ValueError."""
        treatment = np.array([1, 1, 1, 1])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        X = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError) as exc_info:
            regression_adjusted_ate(outcomes, treatment, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "no control" in error_msg.lower()

    def test_all_control_fails_fast(self):
        """Test that data with no treated units raises ValueError."""
        treatment = np.array([0, 0, 0, 0])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        X = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError) as exc_info:
            regression_adjusted_ate(outcomes, treatment, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "no treated" in error_msg.lower()

    def test_mismatched_covariate_length(self):
        """Test that mismatched covariate length raises ValueError."""
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        X = np.array([1.0, 2.0])  # Wrong length

        with pytest.raises(ValueError) as exc_info:
            regression_adjusted_ate(outcomes, treatment, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "covariate" in error_msg.lower() or "length" in error_msg.lower()

    def test_nan_in_covariates(self):
        """Test that NaN in covariates raises ValueError."""
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        X = np.array([1.0, 2.0, np.nan, 4.0])

        with pytest.raises(ValueError) as exc_info:
            regression_adjusted_ate(outcomes, treatment, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "nan" in error_msg.lower()

    def test_infinite_covariates(self):
        """Test that infinite values in covariates raise ValueError."""
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        X = np.array([1.0, np.inf, 3.0, 4.0])

        with pytest.raises(ValueError) as exc_info:
            regression_adjusted_ate(outcomes, treatment, X)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "infinite" in error_msg.lower()


class TestRegressionAdjustedProperties:
    """Test statistical properties of regression adjustment."""

    def test_efficiency_gain_with_predictive_covariate(self):
        """
        Test that efficiency gain is larger when covariate is more predictive.

        Strong predictor (R^2 ~ 0.9) should give larger SE reduction than
        weak predictor (R^2 ~ 0.1).
        """
        np.random.seed(42)
        n = 200
        treatment = np.random.binomial(1, 0.5, n)

        # Strong predictor: X explains most of Y variance
        X_strong = np.random.normal(0, 10, n)
        outcomes_strong = 5 * treatment + 10 * X_strong + np.random.normal(0, 1, n)

        # Weak predictor: X explains little of Y variance
        X_weak = np.random.normal(0, 1, n)
        outcomes_weak = 5 * treatment + 0.5 * X_weak + np.random.normal(0, 10, n)

        # Simple estimates (no adjustment)
        simple_strong = simple_ate(outcomes_strong, treatment)
        simple_weak = simple_ate(outcomes_weak, treatment)

        # Adjusted estimates
        adjusted_strong = regression_adjusted_ate(outcomes_strong, treatment, X_strong)
        adjusted_weak = regression_adjusted_ate(outcomes_weak, treatment, X_weak)

        # SE reduction from strong predictor
        se_reduction_strong = (simple_strong["se"] - adjusted_strong["se"]) / simple_strong["se"]

        # SE reduction from weak predictor
        se_reduction_weak = (simple_weak["se"] - adjusted_weak["se"]) / simple_weak["se"]

        # Strong predictor should give larger relative SE reduction
        assert se_reduction_strong > se_reduction_weak, (
            f"Strong predictor reduction ({se_reduction_strong:.2%}) should be > "
            f"weak predictor reduction ({se_reduction_weak:.2%})"
        )

    def test_unbiased_under_randomization(self):
        """
        Test that regression adjustment is unbiased under randomization.

        Even with imbalanced covariates, RCT ensures E[tau_hat] = tau.
        """
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 5, n)
        treatment = np.random.binomial(1, 0.5, n)
        true_ate = 3.0

        # Generate outcomes with true ATE = 3
        outcomes = true_ate * treatment + 2 * X + np.random.normal(0, 2, n)

        result = regression_adjusted_ate(outcomes, treatment, X)

        # Should be close to true ATE (within ~2 SEs for 95% confidence)
        assert np.isclose(result["estimate"], true_ate, atol=2 * result["se"])


# Marker for pytest
pytestmark = pytest.mark.unit
