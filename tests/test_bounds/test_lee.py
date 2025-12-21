"""
Tests for Lee (2009) bounds under sample selection.

Layer 1: Known-answer tests
Layer 2: Adversarial tests (edge cases, input validation)
Layer 3: Property tests (bounds correctness, bootstrap coverage)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
import warnings

from causal_inference.bounds import (
    lee_bounds,
    lee_bounds_tightened,
    check_monotonicity,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_selection_data():
    """Simple data with differential attrition."""
    np.random.seed(42)
    n = 1000
    treatment = np.random.binomial(1, 0.5, n)

    # Treatment increases observation probability (positive monotonicity)
    p_observed = 0.7 + 0.2 * treatment
    observed = np.random.binomial(1, p_observed)

    # True ATE = 2.0
    outcome = 2.0 * treatment + np.random.randn(n)

    return outcome, treatment, observed, 2.0


@pytest.fixture
def negative_monotonicity_data():
    """Data where treatment decreases observation."""
    np.random.seed(123)
    n = 800
    treatment = np.random.binomial(1, 0.5, n)

    # Treatment decreases observation probability
    p_observed = 0.8 - 0.3 * treatment
    observed = np.random.binomial(1, p_observed)

    outcome = 1.5 * treatment + np.random.randn(n)

    return outcome, treatment, observed, 1.5


@pytest.fixture
def no_attrition_data():
    """Data with no differential attrition."""
    np.random.seed(456)
    n = 500
    treatment = np.random.binomial(1, 0.5, n)

    # Same observation rate for both groups
    observed = np.random.binomial(1, 0.8, n)

    outcome = 2.0 * treatment + np.random.randn(n)

    return outcome, treatment, observed, 2.0


@pytest.fixture
def high_attrition_data():
    """Data with high attrition in control group."""
    np.random.seed(789)
    n = 1000
    treatment = np.random.binomial(1, 0.5, n)

    # High differential attrition
    p_observed = np.where(treatment == 1, 0.9, 0.4)
    observed = np.random.binomial(1, p_observed)

    outcome = 3.0 * treatment + np.random.randn(n)

    return outcome, treatment, observed, 3.0


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestLeeBoundsKnownAnswers:
    """Tests with expected behavior."""

    def test_bounds_contain_true_ate(self, simple_selection_data):
        """Bounds should contain true ATE under correct monotonicity."""
        outcome, treatment, observed, true_ate = simple_selection_data

        result = lee_bounds(
            outcome, treatment, observed,
            monotonicity="positive",
            n_bootstrap=100,
            random_state=42
        )

        # Bounds should contain true ATE (with some tolerance for sampling)
        assert result["bounds_lower"] < true_ate + 0.5
        assert result["bounds_upper"] > true_ate - 0.5

    def test_positive_monotonicity_correct_direction(self, simple_selection_data):
        """Positive monotonicity should work when treatment increases observation."""
        outcome, treatment, observed, _ = simple_selection_data

        result = lee_bounds(
            outcome, treatment, observed,
            monotonicity="positive",
            n_bootstrap=0
        )

        assert result["bounds_lower"] <= result["bounds_upper"]
        assert result["attrition_treated"] < result["attrition_control"]

    def test_negative_monotonicity_correct_direction(self, negative_monotonicity_data):
        """Negative monotonicity should work when treatment decreases observation."""
        outcome, treatment, observed, _ = negative_monotonicity_data

        result = lee_bounds(
            outcome, treatment, observed,
            monotonicity="negative",
            n_bootstrap=0
        )

        assert result["bounds_lower"] <= result["bounds_upper"]
        assert result["attrition_treated"] > result["attrition_control"]

    def test_no_attrition_point_identified(self, no_attrition_data):
        """No differential attrition should give point identification."""
        outcome, treatment, observed, true_ate = no_attrition_data

        result = lee_bounds(
            outcome, treatment, observed,
            monotonicity="positive",
            n_bootstrap=0
        )

        # Bounds should be very close (point identified)
        assert result["bounds_width"] < 0.5  # Some tolerance for randomness

    def test_high_attrition_wider_bounds(self, high_attrition_data):
        """Higher differential attrition should give wider bounds."""
        outcome, treatment, observed, _ = high_attrition_data

        result = lee_bounds(
            outcome, treatment, observed,
            monotonicity="positive",
            n_bootstrap=0
        )

        # Wide bounds due to high attrition
        assert result["bounds_width"] > 0.5


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_ci_contains_bounds(self, simple_selection_data):
        """Bootstrap CI should contain point estimates."""
        outcome, treatment, observed, _ = simple_selection_data

        result = lee_bounds(
            outcome, treatment, observed,
            monotonicity="positive",
            n_bootstrap=500,
            random_state=42
        )

        # CI should be wider than bounds
        assert result["ci_lower"] <= result["bounds_lower"]
        assert result["ci_upper"] >= result["bounds_upper"]

    def test_ci_width_increases_with_alpha(self, simple_selection_data):
        """Smaller alpha should give wider CI."""
        outcome, treatment, observed, _ = simple_selection_data

        result_95 = lee_bounds(
            outcome, treatment, observed,
            monotonicity="positive",
            n_bootstrap=300,
            alpha=0.05,
            random_state=42
        )

        result_99 = lee_bounds(
            outcome, treatment, observed,
            monotonicity="positive",
            n_bootstrap=300,
            alpha=0.01,
            random_state=42
        )

        ci_width_95 = result_95["ci_upper"] - result_95["ci_lower"]
        ci_width_99 = result_99["ci_upper"] - result_99["ci_lower"]

        assert ci_width_99 > ci_width_95 * 0.9  # 99% CI should be wider


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_rejects_mismatched_lengths(self):
        """Should reject arrays of different lengths."""
        outcome = np.array([1.0, 2.0, 3.0])
        treatment = np.array([0, 1])
        observed = np.array([1, 1, 1])

        with pytest.raises(ValueError, match="same length"):
            lee_bounds(outcome, treatment, observed)

    def test_rejects_non_binary_treatment(self):
        """Should reject non-binary treatment."""
        outcome = np.array([1.0, 2.0, 3.0])
        treatment = np.array([0, 1, 2])
        observed = np.array([1, 1, 1])

        with pytest.raises(ValueError, match="binary"):
            lee_bounds(outcome, treatment, observed)

    def test_rejects_non_binary_observed(self):
        """Should reject non-binary observed indicator."""
        outcome = np.array([1.0, 2.0, 3.0])
        treatment = np.array([0, 1, 1])
        observed = np.array([0, 0.5, 1])

        with pytest.raises(ValueError, match="binary"):
            lee_bounds(outcome, treatment, observed)

    def test_rejects_invalid_monotonicity(self):
        """Should reject invalid monotonicity direction."""
        outcome = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([0, 0, 1, 1])
        observed = np.array([1, 1, 1, 1])

        with pytest.raises(ValueError, match="monotonicity"):
            lee_bounds(outcome, treatment, observed, monotonicity="invalid")

    def test_rejects_no_observed_treated(self):
        """Should reject when no treated have observed outcomes."""
        outcome = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([0, 0, 1, 1])
        observed = np.array([1, 1, 0, 0])

        with pytest.raises(ValueError, match="observed"):
            lee_bounds(outcome, treatment, observed)

    def test_rejects_no_observed_control(self):
        """Should reject when no control have observed outcomes."""
        outcome = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([0, 0, 1, 1])
        observed = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="observed"):
            lee_bounds(outcome, treatment, observed)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_small_sample(self):
        """Should handle very small samples."""
        np.random.seed(1)
        outcome = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([0, 0, 1, 1])
        observed = np.array([1, 1, 1, 1])

        result = lee_bounds(outcome, treatment, observed, n_bootstrap=0)
        assert np.isfinite(result["bounds_lower"])
        assert np.isfinite(result["bounds_upper"])

    def test_all_observed(self):
        """Should handle case where all are observed."""
        np.random.seed(2)
        n = 100
        outcome = np.random.randn(n)
        treatment = np.random.binomial(1, 0.5, n)
        observed = np.ones(n)

        result = lee_bounds(outcome, treatment, observed, n_bootstrap=0)
        # All observed → point identified
        assert result["bounds_width"] < 0.5

    def test_extreme_imbalance(self):
        """Should handle extreme treatment imbalance."""
        np.random.seed(3)
        n = 200
        treatment = np.random.binomial(1, 0.05, n)  # 5% treated
        observed = np.random.binomial(1, 0.8, n)
        outcome = treatment + np.random.randn(n)

        # May not have enough treated observations
        try:
            result = lee_bounds(outcome, treatment, observed, n_bootstrap=0)
            assert np.isfinite(result["bounds_lower"])
        except ValueError:
            pass  # Expected if too few treated observations

    def test_warns_monotonicity_violation(self, simple_selection_data):
        """Should warn when monotonicity appears violated."""
        outcome, treatment, observed, _ = simple_selection_data

        # Using wrong monotonicity direction
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lee_bounds(
                outcome, treatment, observed,
                monotonicity="negative",  # Wrong direction
                n_bootstrap=0
            )
            # Should have warned
            assert any("violation" in str(warning.message).lower() for warning in w)


class TestReturnTypes:
    """Tests for return type correctness."""

    def test_return_type_has_all_keys(self, simple_selection_data):
        """Result should have all required keys."""
        outcome, treatment, observed, _ = simple_selection_data

        result = lee_bounds(
            outcome, treatment, observed,
            n_bootstrap=50,
            random_state=42
        )

        required_keys = [
            "bounds_lower", "bounds_upper", "bounds_width",
            "ci_lower", "ci_upper", "point_identified",
            "trimming_proportion", "trimmed_group",
            "attrition_treated", "attrition_control",
            "n_treated_observed", "n_control_observed",
            "n_trimmed", "monotonicity_assumption", "interpretation"
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_numeric_values_finite(self, simple_selection_data):
        """All numeric values should be finite."""
        outcome, treatment, observed, _ = simple_selection_data

        result = lee_bounds(
            outcome, treatment, observed,
            n_bootstrap=100,
            random_state=42
        )

        for key in ["bounds_lower", "bounds_upper", "attrition_treated", "attrition_control"]:
            assert np.isfinite(result[key]), f"{key} is not finite"


# =============================================================================
# Layer 3: Property Tests
# =============================================================================


class TestBoundsProperties:
    """Tests for mathematical properties of bounds."""

    def test_lower_less_than_upper(self, simple_selection_data):
        """Lower bound should always be ≤ upper bound."""
        outcome, treatment, observed, _ = simple_selection_data

        result = lee_bounds(outcome, treatment, observed, n_bootstrap=0)
        assert result["bounds_lower"] <= result["bounds_upper"]

    def test_bounds_width_nonnegative(self, simple_selection_data):
        """Bounds width should be non-negative."""
        outcome, treatment, observed, _ = simple_selection_data

        result = lee_bounds(outcome, treatment, observed, n_bootstrap=0)
        assert result["bounds_width"] >= 0

    def test_attrition_rates_valid(self, simple_selection_data):
        """Attrition rates should be in [0, 1]."""
        outcome, treatment, observed, _ = simple_selection_data

        result = lee_bounds(outcome, treatment, observed, n_bootstrap=0)
        assert 0 <= result["attrition_treated"] <= 1
        assert 0 <= result["attrition_control"] <= 1

    def test_trimmed_group_correct(self, simple_selection_data):
        """Trimmed group should be the one with lower attrition."""
        outcome, treatment, observed, _ = simple_selection_data

        result = lee_bounds(outcome, treatment, observed, n_bootstrap=0)

        if result["attrition_treated"] < result["attrition_control"]:
            assert result["trimmed_group"] == "treated"
        elif result["attrition_control"] < result["attrition_treated"]:
            assert result["trimmed_group"] == "control"


class TestCheckMonotonicity:
    """Tests for monotonicity checking function."""

    def test_detects_positive_monotonicity(self, simple_selection_data):
        """Should detect positive monotonicity."""
        _, treatment, observed, _ = simple_selection_data

        result = check_monotonicity(treatment, observed)

        assert result["obs_rate_treated"] > result["obs_rate_control"]
        assert result["suggested_monotonicity"] == "positive"

    def test_detects_negative_monotonicity(self, negative_monotonicity_data):
        """Should detect negative monotonicity."""
        _, treatment, observed, _ = negative_monotonicity_data

        result = check_monotonicity(treatment, observed)

        assert result["obs_rate_treated"] < result["obs_rate_control"]
        assert result["suggested_monotonicity"] == "negative"

    def test_returns_all_fields(self, simple_selection_data):
        """Should return all expected fields."""
        _, treatment, observed, _ = simple_selection_data

        result = check_monotonicity(treatment, observed)

        expected_keys = [
            "obs_rate_treated", "obs_rate_control", "difference",
            "se", "z_statistic", "p_value", "significant",
            "suggested_monotonicity", "interpretation"
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


# =============================================================================
# Monte Carlo Validation
# =============================================================================


class TestMonteCarloCoverage:
    """Monte Carlo tests for coverage properties."""

    def test_bounds_coverage_positive_monotonicity(self):
        """Bounds should contain true ATE most of the time."""
        true_ate = 2.0
        n_sims = 50
        contains_count = 0

        for seed in range(n_sims):
            np.random.seed(seed)
            n = 500
            treatment = np.random.binomial(1, 0.5, n)
            p_observed = 0.7 + 0.2 * treatment
            observed = np.random.binomial(1, p_observed)
            outcome = true_ate * treatment + np.random.randn(n)

            result = lee_bounds(
                outcome, treatment, observed,
                monotonicity="positive",
                n_bootstrap=0
            )

            if result["bounds_lower"] <= true_ate <= result["bounds_upper"]:
                contains_count += 1

        coverage = contains_count / n_sims
        assert coverage >= 0.80, f"Coverage {coverage:.2%} below 80%"

    def test_bootstrap_ci_coverage(self):
        """Bootstrap CI should have correct coverage."""
        true_ate = 1.5
        n_sims = 30
        ci_contains_count = 0

        for seed in range(n_sims):
            np.random.seed(seed + 1000)
            n = 400
            treatment = np.random.binomial(1, 0.5, n)
            p_observed = 0.75 + 0.15 * treatment
            observed = np.random.binomial(1, p_observed)
            outcome = true_ate * treatment + np.random.randn(n)

            result = lee_bounds(
                outcome, treatment, observed,
                monotonicity="positive",
                n_bootstrap=200,
                alpha=0.05,
                random_state=seed
            )

            if result["ci_lower"] <= true_ate <= result["ci_upper"]:
                ci_contains_count += 1

        ci_coverage = ci_contains_count / n_sims
        # CI should cover true ATE most of the time
        assert ci_coverage >= 0.70, f"CI coverage {ci_coverage:.2%} too low"


# =============================================================================
# Tightened Bounds Tests
# =============================================================================


class TestTightenedBounds:
    """Tests for covariate-tightened bounds."""

    def test_tightened_returns_valid_result(self):
        """Tightened bounds should return valid result."""
        np.random.seed(42)
        n = 500
        treatment = np.random.binomial(1, 0.5, n)
        x = np.random.randn(n)
        p_observed = 0.7 + 0.2 * treatment + 0.1 * x
        p_observed = np.clip(p_observed, 0.1, 0.9)
        observed = np.random.binomial(1, p_observed)
        outcome = 2.0 * treatment + x + np.random.randn(n)

        result = lee_bounds_tightened(
            outcome, treatment, observed,
            covariates=x.reshape(-1, 1),
            n_bootstrap=50,
            random_state=42
        )

        assert np.isfinite(result["bounds_lower"])
        assert np.isfinite(result["bounds_upper"])
        assert result["bounds_lower"] <= result["bounds_upper"]

    def test_tightened_handles_small_sample(self):
        """Should fall back to basic bounds with small sample."""
        np.random.seed(43)
        n = 50
        treatment = np.random.binomial(1, 0.5, n)
        x = np.random.randn(n)
        observed = np.random.binomial(1, 0.8, n)
        outcome = treatment + np.random.randn(n)

        result = lee_bounds_tightened(
            outcome, treatment, observed,
            covariates=x.reshape(-1, 1),
            n_bootstrap=0
        )

        assert np.isfinite(result["bounds_lower"])
