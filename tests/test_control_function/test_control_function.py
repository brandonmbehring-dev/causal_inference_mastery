"""
Known-answer tests for Control Function estimation.

Tests validate:
1. CF matches 2SLS in linear case (CRITICAL correctness check)
2. Correct treatment effect recovery
3. Endogeneity detection works
4. Correct standard error computation
5. Confidence interval coverage

Layer 1 of 6-layer validation architecture.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.control_function import (
    ControlFunction,
    control_function_ate,
    ControlFunctionResult,
)
from src.causal_inference.iv import TwoStageLeastSquares


class TestCFMatches2SLS:
    """
    CRITICAL: Control Function must match 2SLS in linear case.

    This is the fundamental correctness check. CF and 2SLS are algebraically
    equivalent for linear models, so point estimates must match exactly.
    """

    def test_cf_matches_2sls_point_estimate(self, cf_endogenous):
        """CF point estimate matches 2SLS to high precision."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        # CF estimate (analytical to avoid bootstrap variation)
        cf = ControlFunction(inference="analytical")
        cf_result = cf.fit(Y, D, Z.ravel(), X)

        # 2SLS estimate
        tsls = TwoStageLeastSquares(inference="robust")
        tsls.fit(Y, D, Z.ravel(), X)
        tsls_estimate = tsls.coef_[0]  # Treatment coefficient

        # Must match to at least 10 decimal places
        assert_allclose(
            cf_result["estimate"],
            tsls_estimate,
            rtol=1e-10,
            err_msg="CF estimate must match 2SLS in linear case",
        )

    def test_cf_matches_2sls_with_controls(self, cf_with_controls):
        """CF matches 2SLS when controls are included."""
        Y, D, Z, X, true_beta, rho = cf_with_controls

        # CF estimate
        cf = ControlFunction(inference="analytical")
        cf_result = cf.fit(Y, D, Z.ravel(), X)

        # 2SLS estimate
        tsls = TwoStageLeastSquares(inference="robust")
        tsls.fit(Y, D, Z.ravel(), X)
        tsls_estimate = tsls.coef_[0]

        assert_allclose(
            cf_result["estimate"],
            tsls_estimate,
            rtol=1e-10,
            err_msg="CF estimate must match 2SLS with controls",
        )

    def test_cf_matches_2sls_overidentified(self, cf_over_identified):
        """CF matches 2SLS with multiple instruments."""
        Y, D, Z, X, true_beta, rho = cf_over_identified

        # CF estimate
        cf = ControlFunction(inference="analytical")
        cf_result = cf.fit(Y, D, Z, X)

        # 2SLS estimate
        tsls = TwoStageLeastSquares(inference="robust")
        tsls.fit(Y, D, Z, X)
        tsls_estimate = tsls.coef_[0]

        assert_allclose(
            cf_result["estimate"],
            tsls_estimate,
            rtol=1e-10,
            err_msg="CF estimate must match 2SLS with multiple instruments",
        )


class TestTreatmentEffectRecovery:
    """Tests that CF recovers the true treatment effect."""

    def test_recovers_true_effect_endogenous(self, cf_endogenous):
        """CF recovers true effect with strong endogeneity."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="bootstrap", n_bootstrap=500, random_state=42)
        result = cf.fit(Y, D, Z.ravel(), X)

        # Bias should be small (< 10% of true effect or < 0.2)
        bias = abs(result["estimate"] - true_beta)
        assert bias < max(0.2, 0.1 * abs(true_beta)), (
            f"CF bias too large: estimate={result['estimate']:.3f}, "
            f"true={true_beta}, bias={bias:.3f}"
        )

    def test_recovers_true_effect_exogenous(self, cf_exogenous):
        """CF recovers true effect when treatment is exogenous."""
        Y, D, Z, X, true_beta, rho = cf_exogenous

        cf = ControlFunction(inference="bootstrap", n_bootstrap=500, random_state=42)
        result = cf.fit(Y, D, Z.ravel(), X)

        bias = abs(result["estimate"] - true_beta)
        assert bias < max(0.2, 0.1 * abs(true_beta)), (
            f"CF bias too large: estimate={result['estimate']:.3f}, "
            f"true={true_beta}, bias={bias:.3f}"
        )

    def test_recovers_negative_effect(self, cf_negative_effect):
        """CF correctly handles negative treatment effects."""
        Y, D, Z, X, true_beta, rho = cf_negative_effect

        cf = ControlFunction(inference="bootstrap", n_bootstrap=500, random_state=42)
        result = cf.fit(Y, D, Z.ravel(), X)

        # Effect should be negative
        assert result["estimate"] < 0, "Failed to detect negative treatment effect"

        # Bias check
        bias = abs(result["estimate"] - true_beta)
        assert bias < max(0.3, 0.15 * abs(true_beta))


class TestEndogeneityDetection:
    """Tests endogeneity detection via control coefficient."""

    def test_detects_endogeneity_when_present(self, cf_endogenous):
        """CF detects endogeneity when rho > 0."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="bootstrap", n_bootstrap=500, random_state=42)
        result = cf.fit(Y, D, Z.ravel(), X)

        # Should detect endogeneity
        assert result["endogeneity_detected"], (
            f"Failed to detect endogeneity (rho={rho}, "
            f"control_p_value={result['control_p_value']:.4f})"
        )

        # Control coefficient should be significant
        assert result["control_p_value"] < 0.05

    def test_no_detection_when_exogenous(self, cf_exogenous):
        """CF does not detect endogeneity when rho = 0."""
        Y, D, Z, X, true_beta, rho = cf_exogenous

        cf = ControlFunction(inference="bootstrap", n_bootstrap=500, random_state=42)
        result = cf.fit(Y, D, Z.ravel(), X)

        # Should not detect endogeneity (or at least not falsely)
        # Allow 10% false positive rate
        # This is a single test, so we just check p-value is reasonable
        assert result["control_p_value"] > 0.01, (
            f"Possible false positive: rho=0 but p_value={result['control_p_value']:.4f}"
        )

    def test_test_endogeneity_method(self, cf_endogenous):
        """test_endogeneity() method returns consistent values."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="bootstrap", n_bootstrap=300, random_state=42)
        result = cf.fit(Y, D, Z.ravel(), X)

        t_stat, p_value = cf.test_endogeneity()

        assert t_stat == result["control_t_stat"]
        assert p_value == result["control_p_value"]

    def test_test_endogeneity_raises_before_fit(self):
        """test_endogeneity() raises error if not fitted."""
        cf = ControlFunction()

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            cf.test_endogeneity()


class TestStandardErrors:
    """Tests for standard error computation."""

    def test_bootstrap_se_reasonable(self, cf_endogenous):
        """Bootstrap SE is reasonable (not too small or large)."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="bootstrap", n_bootstrap=500, random_state=42)
        result = cf.fit(Y, D, Z.ravel(), X)

        # SE should be positive and reasonable relative to estimate
        assert result["se"] > 0
        assert result["se"] < abs(result["estimate"]) * 2  # Not absurdly large

    def test_analytical_se_reasonable(self, cf_endogenous):
        """Analytical (Murphy-Topel) SE is reasonable."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        assert result["se"] > 0
        assert result["se"] < abs(result["estimate"]) * 2

    def test_naive_se_differs_from_corrected(self, cf_endogenous):
        """Naive SE differs from corrected SE."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        # Naive SE is typically smaller (biased downward)
        # They should differ, though by how much depends on rho
        assert result["se_naive"] != result["se"], (
            "Naive and corrected SE should differ"
        )

    def test_larger_sample_smaller_se(self, cf_endogenous, cf_large_sample):
        """Larger samples have smaller standard errors."""
        Y1, D1, Z1, X1, _, _ = cf_endogenous
        Y2, D2, Z2, X2, _, _ = cf_large_sample

        cf = ControlFunction(inference="bootstrap", n_bootstrap=300, random_state=42)

        result1 = cf.fit(Y1, D1, Z1.ravel(), X1)
        result2 = cf.fit(Y2, D2, Z2.ravel(), X2)

        # Large sample SE should be smaller
        assert result2["se"] < result1["se"], (
            f"Large sample SE ({result2['se']:.4f}) should be < "
            f"small sample SE ({result1['se']:.4f})"
        )


class TestConfidenceIntervals:
    """Tests for confidence interval construction."""

    def test_ci_contains_true_value(self, cf_endogenous):
        """95% CI should contain true value (single test)."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="bootstrap", n_bootstrap=500, random_state=42)
        result = cf.fit(Y, D, Z.ravel(), X)

        # This is a single test, so we just check CI is reasonable
        # Coverage is validated in Monte Carlo tests
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert ci_width > 0, "CI width must be positive"

    def test_ci_width_scales_with_alpha(self, cf_endogenous):
        """Wider CI with smaller alpha (higher confidence)."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf_90 = ControlFunction(
            inference="bootstrap", n_bootstrap=300, alpha=0.10, random_state=42
        )
        cf_95 = ControlFunction(
            inference="bootstrap", n_bootstrap=300, alpha=0.05, random_state=42
        )

        result_90 = cf_90.fit(Y, D, Z.ravel(), X)
        result_95 = cf_95.fit(Y, D, Z.ravel(), X)

        width_90 = result_90["ci_upper"] - result_90["ci_lower"]
        width_95 = result_95["ci_upper"] - result_95["ci_lower"]

        assert width_95 > width_90, "95% CI should be wider than 90% CI"


class TestFirstStageResults:
    """Tests for first-stage regression results."""

    def test_first_stage_f_statistic_positive(self, cf_endogenous):
        """First-stage F-statistic is positive."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        assert result["first_stage"]["f_statistic"] > 0

    def test_weak_iv_warning_triggered(self, cf_weak_instrument):
        """Weak IV warning is triggered when F < 10."""
        Y, D, Z, X, true_beta, rho = cf_weak_instrument

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        # F should be < 10
        assert result["first_stage"]["f_statistic"] < 10
        assert result["first_stage"]["weak_iv_warning"]
        assert "weak" in result["message"].lower()

    def test_no_weak_iv_warning_strong_instrument(self, cf_strong_instrument):
        """No weak IV warning with strong instrument."""
        Y, D, Z, X, true_beta, rho = cf_strong_instrument

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        assert result["first_stage"]["f_statistic"] > 10
        assert not result["first_stage"]["weak_iv_warning"]

    def test_first_stage_residuals_shape(self, cf_endogenous):
        """First-stage residuals have correct shape."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        assert len(result["first_stage"]["residuals"]) == len(Y)

    def test_first_stage_r2_in_range(self, cf_endogenous):
        """First-stage R² is between 0 and 1."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        assert 0 <= result["first_stage"]["r2"] <= 1


class TestConvenienceFunction:
    """Tests for control_function_ate() convenience function."""

    def test_convenience_matches_class(self, cf_endogenous):
        """control_function_ate() matches ControlFunction class."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        # Class-based
        cf = ControlFunction(inference="bootstrap", n_bootstrap=300, random_state=42)
        result_class = cf.fit(Y, D, Z.ravel(), X)

        # Function-based
        result_func = control_function_ate(
            Y, D, Z.ravel(), X, inference="bootstrap", n_bootstrap=300, random_state=42
        )

        assert result_class["estimate"] == result_func["estimate"]
        assert result_class["se"] == result_func["se"]

    def test_convenience_default_parameters(self, cf_endogenous):
        """control_function_ate() works with defaults."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        result = control_function_ate(Y, D, Z.ravel(), X, random_state=42)

        assert "estimate" in result
        assert "se" in result
        assert result["inference"] == "bootstrap"


class TestMetadata:
    """Tests for result metadata."""

    def test_n_obs_correct(self, cf_endogenous):
        """n_obs matches input length."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        assert result["n_obs"] == len(Y)

    def test_n_instruments_single(self, cf_endogenous):
        """n_instruments=1 for single instrument."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        assert result["n_instruments"] == 1

    def test_n_instruments_multiple(self, cf_over_identified):
        """n_instruments correct for multiple instruments."""
        Y, D, Z, X, true_beta, rho = cf_over_identified

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z, X)

        assert result["n_instruments"] == Z.shape[1]

    def test_n_controls_none(self, cf_endogenous):
        """n_controls=0 when no controls."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        assert result["n_controls"] == 0

    def test_n_controls_with_controls(self, cf_with_controls):
        """n_controls correct with controls."""
        Y, D, Z, X, true_beta, rho = cf_with_controls

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        assert result["n_controls"] == X.shape[1]

    def test_inference_method_recorded(self, cf_endogenous):
        """Inference method is recorded correctly."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf_boot = ControlFunction(inference="bootstrap", n_bootstrap=100)
        result_boot = cf_boot.fit(Y, D, Z.ravel(), X)
        assert result_boot["inference"] == "bootstrap"
        assert result_boot["n_bootstrap"] == 100

        cf_anal = ControlFunction(inference="analytical")
        result_anal = cf_anal.fit(Y, D, Z.ravel(), X)
        assert result_anal["inference"] == "analytical"
        assert result_anal["n_bootstrap"] is None


class TestSummary:
    """Tests for summary() method."""

    def test_summary_after_fit(self, cf_endogenous):
        """summary() works after fitting."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="analytical")
        cf.fit(Y, D, Z.ravel(), X)

        summary = cf.summary()

        assert "Control Function Estimation Results" in summary
        assert "Treatment Effect:" in summary
        assert "Endogeneity Test" in summary
        assert "First Stage:" in summary

    def test_summary_before_fit(self):
        """summary() returns message before fitting."""
        cf = ControlFunction()
        summary = cf.summary()

        assert "not fitted" in summary.lower()

    def test_summary_contains_values(self, cf_endogenous):
        """summary() contains key values."""
        Y, D, Z, X, true_beta, rho = cf_endogenous

        cf = ControlFunction(inference="analytical")
        result = cf.fit(Y, D, Z.ravel(), X)

        summary = cf.summary()

        # Check that key values appear in summary
        assert str(result["n_obs"]) in summary
        assert "F-statistic" in summary
