"""
Tests for Controlled Direct Effect estimation.

CDE(m) = E[Y(1,m) - Y(0,m)] at fixed mediator value m.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.mediation import controlled_direct_effect, CDEResult


class TestCDEBasic:
    """Basic functionality tests."""

    def test_returns_correct_type(self, simple_linear_mediation):
        """Returns CDEResult TypedDict."""
        data = simple_linear_mediation
        result = controlled_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            mediator_value=0.5,
        )

        assert isinstance(result, dict)
        assert "cde" in result
        assert "se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "pvalue" in result
        assert "mediator_value" in result

    def test_mediator_value_recorded(self, simple_linear_mediation):
        """Mediator value is recorded correctly."""
        data = simple_linear_mediation
        result = controlled_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            mediator_value=1.5,
        )

        assert result["mediator_value"] == 1.5

    def test_ci_ordered(self, simple_linear_mediation):
        """CI is properly ordered."""
        data = simple_linear_mediation
        result = controlled_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            mediator_value=0.5,
        )

        assert result["ci_lower"] < result["ci_upper"]

    def test_se_positive(self, simple_linear_mediation):
        """SE is positive."""
        data = simple_linear_mediation
        result = controlled_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            mediator_value=0.5,
        )

        assert result["se"] > 0

    def test_pvalue_bounded(self, simple_linear_mediation):
        """P-value is in [0, 1]."""
        data = simple_linear_mediation
        result = controlled_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            mediator_value=0.5,
        )

        assert 0 <= result["pvalue"] <= 1


class TestCDEKnownAnswers:
    """Known-answer tests."""

    def test_cde_equals_direct_in_linear(self, simple_linear_mediation):
        """
        In linear model without interaction, CDE = beta_1 at all m.

        Since Y = beta_0 + beta_1*T + beta_2*M + e,
        CDE(m) = beta_1 regardless of m.
        """
        data = simple_linear_mediation

        # CDE at different mediator values should be similar
        cde_low = controlled_direct_effect(
            data["outcome"], data["treatment"], data["mediator"], mediator_value=0.0
        )
        cde_med = controlled_direct_effect(
            data["outcome"], data["treatment"], data["mediator"], mediator_value=0.5
        )
        cde_high = controlled_direct_effect(
            data["outcome"], data["treatment"], data["mediator"], mediator_value=1.0
        )

        # All should be close to beta_1 = 0.5
        assert_allclose(cde_low["cde"], data["true_direct"], atol=0.15)
        assert_allclose(cde_med["cde"], data["true_direct"], atol=0.15)
        assert_allclose(cde_high["cde"], data["true_direct"], atol=0.15)

        # All should be similar to each other
        assert_allclose(cde_low["cde"], cde_med["cde"], atol=0.1)
        assert_allclose(cde_med["cde"], cde_high["cde"], atol=0.1)

    def test_cde_recovers_direct_effect(self, simple_linear_mediation):
        """CDE recovers true direct effect (beta_1 = 0.5)."""
        data = simple_linear_mediation
        result = controlled_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            mediator_value=0.5,
        )

        assert_allclose(result["cde"], data["true_direct"], atol=0.15)

    def test_full_mediation_cde_zero(self, full_mediation):
        """Full mediation: CDE ≈ 0."""
        data = full_mediation
        result = controlled_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            mediator_value=0.5,
        )

        # Direct effect is 0 in full mediation
        assert abs(result["cde"]) < 0.2

    def test_cde_significant(self, simple_linear_mediation):
        """CDE is significant when true direct effect exists."""
        data = simple_linear_mediation
        result = controlled_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            mediator_value=0.5,
        )

        # With true direct = 0.5, should be significant
        assert result["pvalue"] < 0.05


class TestCDEWithCovariates:
    """Tests with covariates."""

    def test_cde_with_covariates(self, mediation_with_covariates):
        """CDE estimation works with covariates."""
        data = mediation_with_covariates
        result = controlled_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            mediator_value=0.5,
            covariates=data["covariates"],
        )

        # Should recover direct effect
        assert_allclose(result["cde"], data["true_direct"], atol=0.2)


class TestCDEDifferentMediatorValues:
    """Test CDE at different mediator values."""

    def test_cde_at_mean(self, simple_linear_mediation):
        """CDE at mean mediator value."""
        data = simple_linear_mediation
        m_mean = data["mediator"].mean()

        result = controlled_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            mediator_value=m_mean,
        )

        assert_allclose(result["cde"], data["true_direct"], atol=0.15)

    def test_cde_at_quantiles(self, simple_linear_mediation):
        """CDE at different quantiles of M."""
        data = simple_linear_mediation

        q25 = np.percentile(data["mediator"], 25)
        q50 = np.percentile(data["mediator"], 50)
        q75 = np.percentile(data["mediator"], 75)

        cde_q25 = controlled_direct_effect(
            data["outcome"], data["treatment"], data["mediator"], mediator_value=q25
        )
        cde_q50 = controlled_direct_effect(
            data["outcome"], data["treatment"], data["mediator"], mediator_value=q50
        )
        cde_q75 = controlled_direct_effect(
            data["outcome"], data["treatment"], data["mediator"], mediator_value=q75
        )

        # In linear model, all should be similar
        assert_allclose(cde_q25["cde"], cde_q50["cde"], atol=0.1)
        assert_allclose(cde_q50["cde"], cde_q75["cde"], atol=0.1)


class TestCDEAlphaLevel:
    """Test different significance levels."""

    def test_alpha_affects_ci_width(self, simple_linear_mediation):
        """Higher alpha gives narrower CI."""
        data = simple_linear_mediation

        result_95 = controlled_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            mediator_value=0.5,
            alpha=0.05,
        )

        result_90 = controlled_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            mediator_value=0.5,
            alpha=0.10,
        )

        width_95 = result_95["ci_upper"] - result_95["ci_lower"]
        width_90 = result_90["ci_upper"] - result_90["ci_lower"]

        # 95% CI should be wider than 90% CI
        assert width_95 > width_90


class TestCDEInputValidation:
    """Input validation tests."""

    def test_length_mismatch(self):
        """Mismatched lengths raise error."""
        Y = np.random.randn(100)
        T = np.random.randn(90)
        M = np.random.randn(100)

        with pytest.raises(Exception):
            controlled_direct_effect(Y, T, M, mediator_value=0.5)
