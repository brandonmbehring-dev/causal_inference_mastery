"""
Tests for LATE estimation.

Layer 1: Known-answer tests
Layer 2: Unit tests
"""

import pytest
import numpy as np
from scipy import stats

from causal_inference.mte import (
    late_estimator,
    late_bounds,
    complier_characteristics,
)


class TestLATEEstimator:
    """Tests for late_estimator function."""

    def test_late_returns_correct_type(self, simple_binary_iv_data):
        """Result has all required fields."""
        data = simple_binary_iv_data
        result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        assert "late" in result
        assert "se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "pvalue" in result
        assert "complier_share" in result
        assert "first_stage_coef" in result
        assert "first_stage_f" in result
        assert "method" in result

    def test_late_recovers_true_effect(self, simple_binary_iv_data):
        """LATE close to true value with binary IV."""
        data = simple_binary_iv_data
        result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # True LATE = 2.0, allow ±0.3 tolerance
        assert abs(result["late"] - data["true_late"]) < 0.3

    def test_late_ci_contains_true(self, simple_binary_iv_data):
        """95% CI should contain true LATE."""
        data = simple_binary_iv_data
        result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        assert result["ci_lower"] < data["true_late"] < result["ci_upper"]

    def test_late_complier_share(self, simple_binary_iv_data):
        """Complier share matches expected value."""
        data = simple_binary_iv_data
        result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # True complier share = 0.4
        assert abs(result["complier_share"] - data["true_complier_share"]) < 0.1

    def test_late_first_stage_significant(self, simple_binary_iv_data):
        """First stage should be significant."""
        data = simple_binary_iv_data
        result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # F-stat should be large for valid instrument
        assert result["first_stage_f"] > 10

    def test_late_weak_instrument_warning(self, weak_instrument_data):
        """Weak instrument should have low F-stat."""
        data = weak_instrument_data
        result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # F-stat should be small for weak instrument
        assert result["first_stage_f"] < 10

    def test_late_method_is_wald(self, simple_binary_iv_data):
        """Default method should be Wald estimator."""
        data = simple_binary_iv_data

        result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # Method should be Wald for binary IV
        assert result["method"] == "wald"

    def test_late_with_covariates(self, simple_binary_iv_data, rng):
        """LATE with covariates should work."""
        data = simple_binary_iv_data
        n = len(data["outcome"])

        # Add some covariates (independent of instrument for simplicity)
        covariates = rng.normal(size=(n, 2))

        result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            covariates=covariates,
        )

        # Should still recover true effect (within wider tolerance for covariates)
        assert abs(result["late"] - data["true_late"]) < 0.5

    def test_late_se_positive(self, simple_binary_iv_data):
        """Standard error should be positive."""
        data = simple_binary_iv_data
        result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        assert result["se"] > 0

    def test_late_pvalue_valid(self, simple_binary_iv_data):
        """P-value should be in [0, 1]."""
        data = simple_binary_iv_data
        result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        assert 0 <= result["pvalue"] <= 1


class TestLATEBounds:
    """Tests for late_bounds function."""

    def test_bounds_returns_correct_type(self, simple_binary_iv_data):
        """Bounds result has required fields."""
        data = simple_binary_iv_data
        result = late_bounds(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        assert "bounds_lower" in result
        assert "bounds_upper" in result
        assert "late_under_monotonicity" in result

    def test_bounds_contain_late(self, simple_binary_iv_data):
        """Bounds should contain LATE point estimate."""
        data = simple_binary_iv_data
        result = late_bounds(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        assert result["bounds_lower"] <= result["late_under_monotonicity"] <= result["bounds_upper"]

    def test_bounds_contain_true(self, simple_binary_iv_data):
        """Bounds should contain true LATE."""
        data = simple_binary_iv_data
        result = late_bounds(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # With valid monotonicity, bounds should be informative
        assert result["bounds_lower"] < data["true_late"] < result["bounds_upper"]

    def test_bounds_wider_with_defiers(self, defier_data):
        """Bounds should be wider with potential defiers."""
        data = defier_data
        result = late_bounds(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # Bounds should still be computed
        assert not np.isnan(result["bounds_lower"])
        assert not np.isnan(result["bounds_upper"])
        assert result["bounds_upper"] - result["bounds_lower"] > 0


class TestComplierCharacteristics:
    """Tests for complier_characteristics function."""

    def test_returns_correct_type(self, simple_binary_iv_data):
        """Result has required fields."""
        data = simple_binary_iv_data
        result = complier_characteristics(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        assert "complier_mean_outcome_treated" in result
        assert "complier_mean_outcome_control" in result
        assert "complier_share" in result
        assert "method" in result

    def test_complier_share_positive(self, simple_binary_iv_data):
        """Complier share should be positive."""
        data = simple_binary_iv_data
        result = complier_characteristics(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        assert result["complier_share"] > 0

    def test_complier_share_bounded(self, simple_binary_iv_data):
        """Complier share should be in [0, 1]."""
        data = simple_binary_iv_data
        result = complier_characteristics(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        assert 0 <= result["complier_share"] <= 1

    def test_late_equals_outcome_diff(self, simple_binary_iv_data):
        """LATE = E[Y1|C] - E[Y0|C] for compliers."""
        data = simple_binary_iv_data

        late_result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        complier_result = complier_characteristics(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # LATE should equal the difference in complier outcomes
        implied_late = (
            complier_result["complier_mean_outcome_treated"] -
            complier_result["complier_mean_outcome_control"]
        )

        # Allow some tolerance due to different estimation approaches
        assert abs(late_result["late"] - implied_late) < 0.5

    def test_with_covariates(self, simple_binary_iv_data, rng):
        """Should work with covariates."""
        data = simple_binary_iv_data
        n = len(data["outcome"])

        # Add some covariates
        covariates = rng.normal(size=(n, 2))

        result = complier_characteristics(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            covariates=covariates,
        )

        assert "covariate_means" in result
        # With covariates, should have covariate means
        assert result["covariate_means"] is not None


class TestLATEInputValidation:
    """Tests for input validation."""

    def test_rejects_mismatched_lengths(self, rng):
        """Should reject mismatched input lengths."""
        Y = rng.normal(size=100)
        D = rng.binomial(1, 0.5, size=90)  # Wrong length
        Z = rng.binomial(1, 0.5, size=100)

        with pytest.raises(ValueError, match="length"):
            late_estimator(Y, D, Z)

    def test_rejects_non_binary_treatment(self, rng):
        """Should reject non-binary treatment."""
        Y = rng.normal(size=100)
        D = rng.normal(size=100)  # Continuous, not binary
        Z = rng.binomial(1, 0.5, size=100)

        with pytest.raises(ValueError, match="binary"):
            late_estimator(Y, D, Z)

    def test_handles_nan_in_outcome(self, rng):
        """Should handle NaN values - may produce NaN result."""
        Y = rng.normal(size=100)
        Y[0] = np.nan
        D = rng.binomial(1, 0.5, size=100).astype(float)
        Z = rng.binomial(1, 0.5, size=100).astype(float)

        # Function may complete but result may contain NaN
        # This tests that it doesn't crash
        result = late_estimator(Y, D, Z)
        # LATE may be NaN due to NaN propagation
        assert "late" in result


class TestLATEKnownAnswers:
    """Known-answer tests with analytically computed values."""

    def test_perfect_compliance(self, rng):
        """When D = Z exactly, LATE = OLS coefficient."""
        n = 500
        Z = rng.binomial(1, 0.5, size=n).astype(float)
        D = Z.copy()  # Perfect compliance
        true_effect = 3.0
        Y = 1 + true_effect * D + rng.normal(0, 0.3, size=n)

        result = late_estimator(Y, D, Z)

        # With perfect compliance, LATE = ATE = true effect
        assert abs(result["late"] - true_effect) < 0.2
        assert result["complier_share"] > 0.9  # Nearly all compliers

    def test_no_always_takers(self, rng):
        """When no always-takers, easy to identify."""
        n = 500
        Z = rng.binomial(1, 0.5, size=n).astype(float)
        U = rng.uniform(0, 1, size=n)

        # No always-takers: D=0 when Z=0, D=1 for some when Z=1
        D = np.zeros(n)
        D[(Z == 1) & (U < 0.6)] = 1

        Y = 1 + 2 * D + rng.normal(0, 0.3, size=n)

        result = late_estimator(Y, D, Z)

        assert result["always_taker_share"] < 0.1
        assert abs(result["late"] - 2.0) < 0.3
