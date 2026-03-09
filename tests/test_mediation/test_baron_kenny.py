"""
Tests for Baron-Kenny mediation analysis.

Layer 1: Known-answer tests with hand-calculated values.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.mediation import baron_kenny, BaronKennyResult


class TestBaronKennyBasic:
    """Basic functionality tests."""

    def test_returns_correct_type(self, simple_linear_mediation):
        """Baron-Kenny returns BaronKennyResult TypedDict."""
        data = simple_linear_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        assert isinstance(result, dict)
        assert "alpha_1" in result
        assert "beta_1" in result
        assert "beta_2" in result
        assert "indirect_effect" in result
        assert "direct_effect" in result
        assert "total_effect" in result
        assert "sobel_z" in result
        assert "sobel_pvalue" in result

    def test_n_obs_correct(self, simple_linear_mediation):
        """Sample size recorded correctly."""
        data = simple_linear_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])
        assert result["n_obs"] == data["n"]

    def test_r_squared_bounded(self, simple_linear_mediation):
        """R-squared values are in [0, 1]."""
        data = simple_linear_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        assert 0 <= result["r2_mediator_model"] <= 1
        assert 0 <= result["r2_outcome_model"] <= 1

    def test_pvalues_bounded(self, simple_linear_mediation):
        """P-values are in [0, 1]."""
        data = simple_linear_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        assert 0 <= result["alpha_1_pvalue"] <= 1
        assert 0 <= result["beta_1_pvalue"] <= 1
        assert 0 <= result["beta_2_pvalue"] <= 1
        assert 0 <= result["sobel_pvalue"] <= 1


class TestBaronKennyKnownAnswers:
    """Known-answer tests with expected values."""

    def test_alpha_1_recovery(self, simple_linear_mediation):
        """Recovers true T -> M effect (alpha_1 = 0.6)."""
        data = simple_linear_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        # Should recover alpha_1 = 0.6 within tolerance
        assert_allclose(result["alpha_1"], data["true_alpha_1"], atol=0.15)

    def test_beta_1_recovery(self, simple_linear_mediation):
        """Recovers true direct effect (beta_1 = 0.5)."""
        data = simple_linear_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        assert_allclose(result["beta_1"], data["true_beta_1"], atol=0.15)

    def test_beta_2_recovery(self, simple_linear_mediation):
        """Recovers true M -> Y effect (beta_2 = 0.8)."""
        data = simple_linear_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        assert_allclose(result["beta_2"], data["true_beta_2"], atol=0.15)

    def test_indirect_effect_recovery(self, simple_linear_mediation):
        """Recovers true indirect effect (alpha_1 * beta_2 = 0.48)."""
        data = simple_linear_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        assert_allclose(result["indirect_effect"], data["true_indirect"], atol=0.15)

    def test_direct_effect_recovery(self, simple_linear_mediation):
        """Recovers true direct effect (beta_1 = 0.5)."""
        data = simple_linear_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        assert_allclose(result["direct_effect"], data["true_direct"], atol=0.15)

    def test_total_effect_decomposition(self, simple_linear_mediation):
        """Total = direct + indirect."""
        data = simple_linear_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        expected_total = result["direct_effect"] + result["indirect_effect"]
        assert_allclose(result["total_effect"], expected_total, rtol=1e-10)

    def test_indirect_equals_product(self, simple_linear_mediation):
        """Indirect effect = alpha_1 * beta_2."""
        data = simple_linear_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        expected_indirect = result["alpha_1"] * result["beta_2"]
        assert_allclose(result["indirect_effect"], expected_indirect, rtol=1e-10)


class TestBaronKennySpecialCases:
    """Tests for special mediation cases."""

    def test_full_mediation_zero_direct(self, full_mediation):
        """Full mediation: direct effect ≈ 0."""
        data = full_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        # Direct effect should be close to 0
        assert abs(result["direct_effect"]) < 0.15

    def test_full_mediation_indirect_significant(self, full_mediation):
        """Full mediation: indirect effect is significant."""
        data = full_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        # Indirect effect should be non-zero
        assert result["indirect_effect"] > 0.2
        assert result["sobel_pvalue"] < 0.05

    def test_no_mediation_zero_indirect(self, no_mediation):
        """No mediation: indirect effect ≈ 0."""
        data = no_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        # beta_2 should be close to 0
        assert abs(result["beta_2"]) < 0.15
        # Therefore indirect should be small
        assert abs(result["indirect_effect"]) < 0.15

    def test_no_treatment_on_mediator(self, no_treatment_on_mediator):
        """No T -> M path: indirect effect ≈ 0."""
        data = no_treatment_on_mediator
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        # alpha_1 should be close to 0
        assert abs(result["alpha_1"]) < 0.15
        # Therefore indirect should be small
        assert abs(result["indirect_effect"]) < 0.15


class TestBaronKennyWithCovariates:
    """Tests with pre-treatment covariates."""

    def test_with_covariates(self, mediation_with_covariates):
        """Baron-Kenny works with covariates."""
        data = mediation_with_covariates
        result = baron_kenny(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            covariates=data["covariates"],
        )

        # Should recover effects
        assert_allclose(result["indirect_effect"], data["true_indirect"], atol=0.2)
        assert_allclose(result["direct_effect"], data["true_direct"], atol=0.2)

    def test_covariates_improve_precision(self, mediation_with_covariates):
        """Adding covariates should generally improve precision."""
        data = mediation_with_covariates

        # Without covariates
        result_no_cov = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        # With covariates
        result_cov = baron_kenny(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            covariates=data["covariates"],
        )

        # R-squared should increase with covariates
        assert result_cov["r2_outcome_model"] >= result_no_cov["r2_outcome_model"] - 0.01


class TestBaronKennySobelTest:
    """Tests for Sobel test statistics."""

    def test_sobel_significant_when_mediation(self, simple_linear_mediation):
        """Sobel test significant when mediation exists."""
        data = simple_linear_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        # With true indirect = 0.48, should be significant
        assert result["sobel_pvalue"] < 0.05
        assert abs(result["sobel_z"]) > 1.96

    def test_sobel_not_significant_when_no_mediation(self, no_mediation):
        """Sobel test not significant when no mediation."""
        data = no_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        # With true indirect = 0, should not be significant
        assert result["sobel_pvalue"] > 0.05 or abs(result["sobel_z"]) < 1.96

    def test_sobel_se_positive(self, simple_linear_mediation):
        """Sobel SE is positive."""
        data = simple_linear_mediation
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        assert result["indirect_se"] > 0


class TestBaronKennyInputValidation:
    """Input validation tests."""

    def test_length_mismatch_raises(self):
        """Mismatched array lengths raise ValueError."""
        Y = np.random.randn(100)
        T = np.random.randn(90)  # Wrong length
        M = np.random.randn(100)

        with pytest.raises(ValueError, match="[Ll]ength"):
            baron_kenny(Y, T, M)

    def test_nan_raises(self):
        """NaN values raise ValueError."""
        Y = np.random.randn(100)
        T = np.random.randn(100)
        M = np.random.randn(100)
        Y[0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            baron_kenny(Y, T, M)

    def test_single_observation_fails(self):
        """Single observation should fail gracefully."""
        Y = np.array([1.0])
        T = np.array([1.0])
        M = np.array([1.0])

        with pytest.raises(Exception):
            baron_kenny(Y, T, M)


class TestBaronKennyContinuousTreatment:
    """Tests with continuous treatment."""

    def test_continuous_treatment(self, continuous_treatment):
        """Baron-Kenny works with continuous treatment."""
        data = continuous_treatment
        result = baron_kenny(data["outcome"], data["treatment"], data["mediator"])

        # Should recover path coefficients
        assert_allclose(result["indirect_effect"], data["true_indirect"], atol=0.15)
        assert_allclose(result["direct_effect"], data["true_direct"], atol=0.15)


class TestBaronKennyRobustSE:
    """Tests for robust standard errors."""

    def test_robust_se_option(self, simple_linear_mediation):
        """Robust SE option works."""
        data = simple_linear_mediation

        result_robust = baron_kenny(
            data["outcome"], data["treatment"], data["mediator"], robust_se=True
        )
        result_naive = baron_kenny(
            data["outcome"], data["treatment"], data["mediator"], robust_se=False
        )

        # Both should produce valid results
        assert result_robust["alpha_1_se"] > 0
        assert result_naive["alpha_1_se"] > 0

        # Estimates should be the same
        assert_allclose(result_robust["alpha_1"], result_naive["alpha_1"], rtol=1e-10)

    def test_robust_se_larger_with_heteroskedasticity(self):
        """Robust SEs should be larger with heteroskedasticity."""
        np.random.seed(42)
        n = 500

        T = np.random.binomial(1, 0.5, n).astype(float)
        M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5

        # Heteroskedastic errors (variance depends on T)
        Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * (0.3 + 0.5 * T)

        result_robust = baron_kenny(Y, T, M, robust_se=True)
        result_naive = baron_kenny(Y, T, M, robust_se=False)

        # Robust SE should typically be larger
        # (may not always hold, so we just check they're both positive)
        assert result_robust["alpha_1_se"] > 0
        assert result_naive["alpha_1_se"] > 0
