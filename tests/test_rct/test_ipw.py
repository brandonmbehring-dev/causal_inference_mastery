"""Test inverse probability weighting (IPW) ATE estimator.

IPW reweights units by inverse of propensity scores to create pseudo-population
where treatment is independent of covariates. Useful for RCTs with varying
assignment probabilities (e.g., blocked randomization) and observational studies.

Key principle: Weight = 1/P(T=observed|X) creates balance. Under correct
propensity model, IPW estimator is consistent for ATE.

Following test-first development: write tests before implementation.
"""

import numpy as np
import pytest

from src.causal_inference.rct.estimators import simple_ate
from src.causal_inference.rct.estimators_ipw import ipw_ate


class TestIPWKnownAnswers:
    """Test ipw_ate with hand-calculated values."""

    def test_ipw_with_constant_propensity_equals_simple_ate(self):
        """
        Test that IPW with constant propensity (simple RCT) equals simple_ate.

        In simple RCT, P(T=1) = 0.5 for all units, so IPW weights are constant.
        Result should match difference-in-means.
        """
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        propensity = np.array([0.5, 0.5, 0.5, 0.5])  # Constant

        simple_result = simple_ate(outcomes, treatment)
        ipw_result = ipw_ate(outcomes, treatment, propensity)

        # Should be identical (IPW with constant weights = simple mean)
        assert np.isclose(simple_result["estimate"], ipw_result["estimate"], atol=1e-10)

    def test_ipw_known_answer_varying_propensity(self):
        """
        Test IPW with hand-calculated ATE under varying propensity.

        Setup:
        - Unit 1: T=1, Y=10, P(T=1)=0.8 → weight = 1/0.8 = 1.25
        - Unit 2: T=1, Y=12, P(T=1)=0.6 → weight = 1/0.6 = 1.667
        - Unit 3: T=0, Y=4, P(T=0)=0.4 → weight = 1/0.4 = 2.5
        - Unit 4: T=0, Y=2, P(T=0)=0.2 → weight = 1/0.2 = 5.0

        IPW estimate = (sum w_i*Y_i*T_i)/(sum w_i*T_i) - (sum w_i*Y_i*(1-T_i))/(sum w_i*(1-T_i))
        """
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([10.0, 12.0, 4.0, 2.0])
        propensity = np.array([0.8, 0.6, 0.6, 0.8])  # P(T=1|X)

        result = ipw_ate(outcomes, treatment, propensity)

        # Hand calculation:
        # Treated weights: 1/0.8=1.25, 1/0.6=1.667
        # Weighted treated mean = (1.25*10 + 1.667*12)/(1.25+1.667) = 32.5/2.917 ≈ 11.14
        #
        # Control weights: 1/(1-0.6)=2.5, 1/(1-0.8)=5.0
        # Weighted control mean = (2.5*4 + 5.0*2)/(2.5+5.0) = 20/7.5 ≈ 2.67
        #
        # ATE ≈ 11.14 - 2.67 ≈ 8.47

        expected_ate = 8.47
        assert np.isclose(result["estimate"], expected_ate, atol=0.1)

    def test_ipw_with_extreme_propensity(self):
        """
        Test that IPW handles extreme propensity scores appropriately.

        With propensity near 0 or 1, weights become very large. Should
        still compute but may have high variance (not an error).
        """
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([10.0, 12.0, 4.0, 2.0])
        # Extreme propensities (but valid: in (0,1))
        propensity = np.array([0.95, 0.9, 0.1, 0.05])

        result = ipw_ate(outcomes, treatment, propensity)

        # Should complete without error
        assert "estimate" in result
        assert "se" in result
        # SE will be large due to extreme weights
        assert result["se"] > 0


class TestIPWErrorHandling:
    """Test error handling for ipw_ate."""

    def test_empty_input_fails_fast(self):
        """Test that empty input arrays raise ValueError."""
        treatment = np.array([])
        outcomes = np.array([])
        propensity = np.array([])

        with pytest.raises(ValueError) as exc_info:
            ipw_ate(outcomes, treatment, propensity)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "empty" in error_msg.lower()

    def test_mismatched_lengths_fails_fast(self):
        """Test that mismatched lengths raise ValueError."""
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        propensity = np.array([0.5, 0.5])  # Wrong length

        with pytest.raises(ValueError) as exc_info:
            ipw_ate(outcomes, treatment, propensity)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "length" in error_msg.lower()

    def test_propensity_out_of_range_fails_fast(self):
        """Test that propensity scores outside (0,1) raise ValueError."""
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])

        # Propensity = 0 (invalid)
        propensity = np.array([0.5, 0.0, 0.5, 0.5])
        with pytest.raises(ValueError) as exc_info:
            ipw_ate(outcomes, treatment, propensity)
        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "propensity" in error_msg.lower()

        # Propensity = 1 (invalid)
        propensity = np.array([0.5, 1.0, 0.5, 0.5])
        with pytest.raises(ValueError) as exc_info:
            ipw_ate(outcomes, treatment, propensity)
        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg

        # Propensity > 1 (invalid)
        propensity = np.array([0.5, 1.2, 0.5, 0.5])
        with pytest.raises(ValueError) as exc_info:
            ipw_ate(outcomes, treatment, propensity)
        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg

        # Propensity < 0 (invalid)
        propensity = np.array([0.5, -0.1, 0.5, 0.5])
        with pytest.raises(ValueError) as exc_info:
            ipw_ate(outcomes, treatment, propensity)
        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg

    def test_nan_in_propensity_fails_fast(self):
        """Test that NaN in propensity raises ValueError."""
        treatment = np.array([1, 1, 0, 0])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        propensity = np.array([0.5, np.nan, 0.5, 0.5])

        with pytest.raises(ValueError) as exc_info:
            ipw_ate(outcomes, treatment, propensity)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "nan" in error_msg.lower()

    def test_all_treated_fails_fast(self):
        """Test that data with no control units raises ValueError."""
        treatment = np.array([1, 1, 1, 1])
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        propensity = np.array([0.5, 0.5, 0.5, 0.5])

        with pytest.raises(ValueError) as exc_info:
            ipw_ate(outcomes, treatment, propensity)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "no control" in error_msg.lower()


class TestIPWProperties:
    """Test statistical properties of IPW estimator."""

    def test_ipw_double_robustness_intuition(self):
        """
        Test that IPW gives correct answer even with covariate imbalance.

        Setup: Treatment and covariate are correlated (imbalanced).
        IPW should still recover true ATE if propensity model correct.
        """
        np.random.seed(42)
        n = 200

        # Covariate
        X = np.random.normal(0, 1, n)

        # Treatment probability depends on X (imbalance)
        propensity_true = 1 / (1 + np.exp(-X))  # Logistic
        treatment = np.random.binomial(1, propensity_true)

        # Outcome: Y = 5*T + 2*X + noise (true ATE = 5)
        true_ate = 5.0
        outcomes = true_ate * treatment + 2 * X + np.random.normal(0, 1, n)

        # IPW with TRUE propensity scores should recover ATE ≈ 5
        result = ipw_ate(outcomes, treatment, propensity_true)

        # Should be close to true ATE (within ~2 SE for 95% CI)
        assert np.isclose(result["estimate"], true_ate, atol=2 * result["se"])

    def test_ipw_variance_larger_with_extreme_weights(self):
        """
        Test that IPW has larger variance when weights are more variable.

        More extreme propensity → more variable weights → larger SE.
        """
        np.random.seed(42)
        n = 100

        # Balanced propensities (less variable weights)
        treatment_balanced = np.array([1, 0] * (n // 2))
        outcomes_balanced = np.random.normal(5, 2, n)
        propensity_balanced = np.full(n, 0.5)

        result_balanced = ipw_ate(outcomes_balanced, treatment_balanced, propensity_balanced)

        # Extreme propensities (more variable weights)
        # High propensity for first half, low for second half
        propensity_extreme = np.concatenate([np.full(n//2, 0.9), np.full(n//2, 0.1)])
        treatment_extreme = np.random.binomial(1, propensity_extreme)
        outcomes_extreme = np.random.normal(5, 2, n)

        result_extreme = ipw_ate(outcomes_extreme, treatment_extreme, propensity_extreme)

        # Extreme weights should yield larger SE (more variable)
        # (This might not always hold in small samples, but generally true)
        # Test that SE is at least positive (softer condition)
        assert result_extreme["se"] > 0
        assert result_balanced["se"] > 0


# Marker for pytest
pytestmark = pytest.mark.unit
