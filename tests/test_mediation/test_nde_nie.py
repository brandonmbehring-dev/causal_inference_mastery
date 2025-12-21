"""
Tests for Natural Direct/Indirect Effects via simulation.

Layer 1: Known-answer tests for simulation-based mediation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.mediation import (
    mediation_analysis,
    natural_direct_effect,
    natural_indirect_effect,
    MediationResult,
)


class TestMediationAnalysisBasic:
    """Basic functionality tests for mediation_analysis."""

    def test_returns_correct_type(self, simple_linear_mediation):
        """Returns MediationResult TypedDict."""
        data = simple_linear_mediation
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_bootstrap=50,
            n_simulations=100,
        )

        assert isinstance(result, dict)
        assert "total_effect" in result
        assert "direct_effect" in result
        assert "indirect_effect" in result
        assert "proportion_mediated" in result
        assert "method" in result

    def test_method_recorded(self, simple_linear_mediation):
        """Method is recorded correctly."""
        data = simple_linear_mediation

        result_bk = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            method="baron_kenny",
            n_bootstrap=50,
        )
        assert result_bk["method"] == "baron_kenny"

        result_sim = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            method="simulation",
            n_bootstrap=50,
            n_simulations=100,
        )
        assert result_sim["method"] == "simulation"

    def test_ci_ordered(self, simple_linear_mediation):
        """Confidence intervals are ordered (lower < upper)."""
        data = simple_linear_mediation
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_bootstrap=100,
            n_simulations=100,
        )

        assert result["te_ci"][0] < result["te_ci"][1]
        assert result["de_ci"][0] < result["de_ci"][1]
        assert result["ie_ci"][0] < result["ie_ci"][1]

    def test_se_positive(self, simple_linear_mediation):
        """Standard errors are positive."""
        data = simple_linear_mediation
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_bootstrap=100,
            n_simulations=100,
        )

        assert result["te_se"] > 0
        assert result["de_se"] > 0
        assert result["ie_se"] > 0


class TestNaturalEffectsKnownAnswers:
    """Known-answer tests for natural effects."""

    def test_nde_recovery(self, simple_linear_mediation):
        """Recovers true NDE (direct effect = 0.5)."""
        data = simple_linear_mediation
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            method="simulation",
            n_bootstrap=200,
            n_simulations=200,
            random_state=42,
        )

        # NDE should be close to true direct effect
        assert_allclose(result["direct_effect"], data["true_direct"], atol=0.2)

    def test_nie_recovery(self, simple_linear_mediation):
        """Recovers true NIE (indirect effect = 0.48)."""
        data = simple_linear_mediation
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            method="simulation",
            n_bootstrap=200,
            n_simulations=200,
            random_state=42,
        )

        # NIE should be close to true indirect effect
        assert_allclose(result["indirect_effect"], data["true_indirect"], atol=0.2)

    def test_total_effect_recovery(self, simple_linear_mediation):
        """Recovers true total effect."""
        data = simple_linear_mediation
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            method="simulation",
            n_bootstrap=200,
            n_simulations=200,
            random_state=42,
        )

        assert_allclose(result["total_effect"], data["true_total"], atol=0.25)

    def test_decomposition_holds(self, simple_linear_mediation):
        """TE = NDE + NIE."""
        data = simple_linear_mediation
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_bootstrap=100,
            n_simulations=100,
        )

        # Total should equal sum of direct and indirect
        expected_total = result["direct_effect"] + result["indirect_effect"]
        assert_allclose(result["total_effect"], expected_total, rtol=0.1)


class TestNaturalEffectsSpecialCases:
    """Tests for special cases."""

    def test_full_mediation(self, full_mediation):
        """Full mediation: NDE ≈ 0, NIE > 0."""
        data = full_mediation
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            method="simulation",
            n_bootstrap=200,
            n_simulations=200,
            random_state=42,
        )

        # Direct effect should be close to 0
        assert abs(result["direct_effect"]) < 0.2
        # Indirect effect should be non-zero
        assert result["indirect_effect"] > 0.2

    def test_no_mediation(self, no_mediation):
        """No mediation: NIE ≈ 0, NDE > 0."""
        data = no_mediation
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            method="simulation",
            n_bootstrap=200,
            n_simulations=200,
            random_state=42,
        )

        # Indirect effect should be close to 0
        assert abs(result["indirect_effect"]) < 0.2
        # Direct effect should be non-zero
        assert abs(result["direct_effect"]) > 0.2

    def test_no_treatment_mediator_path(self, no_treatment_on_mediator):
        """No T -> M: NIE ≈ 0."""
        data = no_treatment_on_mediator
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            method="simulation",
            n_bootstrap=200,
            n_simulations=200,
            random_state=42,
        )

        # Indirect effect should be small
        assert abs(result["indirect_effect"]) < 0.2


class TestProportionMediated:
    """Tests for proportion mediated."""

    def test_proportion_mediated_bounded(self, simple_linear_mediation):
        """Proportion mediated is reasonable."""
        data = simple_linear_mediation
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_bootstrap=100,
            n_simulations=100,
        )

        # For partial mediation, proportion should be between 0 and 1
        # (can exceed 1 with suppression, but unlikely in this DGP)
        pm = result["proportion_mediated"]
        assert -0.5 < pm < 1.5  # Allow some sampling error

    def test_full_mediation_high_proportion(self, full_mediation):
        """Full mediation has high proportion mediated."""
        data = full_mediation
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_bootstrap=200,
            n_simulations=200,
            random_state=42,
        )

        # Should be close to 1 (100% mediated)
        assert result["proportion_mediated"] > 0.7

    def test_no_mediation_low_proportion(self, no_mediation):
        """No mediation has low proportion mediated."""
        data = no_mediation
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_bootstrap=200,
            n_simulations=200,
            random_state=42,
        )

        # Should be close to 0
        assert abs(result["proportion_mediated"]) < 0.3


class TestNDENIEFunctions:
    """Tests for standalone NDE/NIE functions."""

    def test_natural_direct_effect_basic(self, simple_linear_mediation):
        """natural_direct_effect returns correct format."""
        data = simple_linear_mediation
        nde, se, ci = natural_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_bootstrap=100,
            n_simulations=100,
            random_state=42,
        )

        assert isinstance(nde, float)
        assert isinstance(se, float)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]

    def test_natural_indirect_effect_basic(self, simple_linear_mediation):
        """natural_indirect_effect returns correct format."""
        data = simple_linear_mediation
        nie, se, ci = natural_indirect_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_bootstrap=100,
            n_simulations=100,
            random_state=42,
        )

        assert isinstance(nie, float)
        assert isinstance(se, float)
        assert isinstance(ci, tuple)
        assert ci[0] < ci[1]

    def test_nde_recovery_standalone(self, simple_linear_mediation):
        """Standalone NDE recovers true value."""
        data = simple_linear_mediation
        nde, se, ci = natural_direct_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_bootstrap=200,
            n_simulations=200,
            random_state=42,
        )

        assert_allclose(nde, data["true_direct"], atol=0.2)

    def test_nie_recovery_standalone(self, simple_linear_mediation):
        """Standalone NIE recovers true value."""
        data = simple_linear_mediation
        nie, se, ci = natural_indirect_effect(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_bootstrap=200,
            n_simulations=200,
            random_state=42,
        )

        assert_allclose(nie, data["true_indirect"], atol=0.2)


class TestWithCovariates:
    """Tests with covariates."""

    def test_simulation_with_covariates(self, mediation_with_covariates):
        """Simulation method works with covariates."""
        data = mediation_with_covariates
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            covariates=data["covariates"],
            method="simulation",
            n_bootstrap=200,
            n_simulations=200,
            random_state=42,
        )

        # Should recover effects
        assert_allclose(result["indirect_effect"], data["true_indirect"], atol=0.25)
        assert_allclose(result["direct_effect"], data["true_direct"], atol=0.25)


class TestRandomState:
    """Tests for reproducibility."""

    def test_reproducible_with_seed(self, simple_linear_mediation):
        """Same seed gives same results."""
        data = simple_linear_mediation

        result1 = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            method="simulation",
            n_bootstrap=50,
            n_simulations=50,
            random_state=42,
        )

        result2 = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            method="simulation",
            n_bootstrap=50,
            n_simulations=50,
            random_state=42,
        )

        assert_allclose(result1["direct_effect"], result2["direct_effect"], rtol=1e-10)
        assert_allclose(result1["indirect_effect"], result2["indirect_effect"], rtol=1e-10)


class TestTreatmentControlValues:
    """Tests for treatment/control value specification."""

    def test_treatment_control_recorded(self, simple_linear_mediation):
        """Treatment/control values are recorded."""
        data = simple_linear_mediation
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            treat_value=1.0,
            control_value=0.0,
            n_bootstrap=50,
            n_simulations=50,
        )

        assert result["treatment_control"] == (0.0, 1.0)

    def test_custom_treatment_control(self, continuous_treatment):
        """Custom treatment/control values work."""
        data = continuous_treatment
        result = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            treat_value=1.0,
            control_value=-1.0,
            n_bootstrap=100,
            n_simulations=100,
        )

        # Effect should scale (2x the per-unit effect)
        # True effect per unit is ~0.98, so for 2-unit change: ~1.96
        assert result["total_effect"] > data["true_total"]


class TestMethodComparison:
    """Compare Baron-Kenny vs Simulation."""

    def test_methods_agree_linear_dgp(self, simple_linear_mediation):
        """Baron-Kenny and simulation agree on linear DGP."""
        data = simple_linear_mediation

        result_bk = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            method="baron_kenny",
            n_bootstrap=200,
        )

        result_sim = mediation_analysis(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            method="simulation",
            n_bootstrap=200,
            n_simulations=500,
            random_state=42,
        )

        # Should give similar results on linear DGP
        assert_allclose(result_bk["direct_effect"], result_sim["direct_effect"], atol=0.15)
        assert_allclose(result_bk["indirect_effect"], result_sim["indirect_effect"], atol=0.15)
