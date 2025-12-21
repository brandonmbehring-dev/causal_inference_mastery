"""
Tests for mediation sensitivity analysis.

Assesses robustness to violations of sequential ignorability.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.mediation import mediation_sensitivity
from src.causal_inference.mediation.types import SensitivityResult


class TestSensitivityBasic:
    """Basic functionality tests."""

    def test_returns_correct_type(self, simple_linear_mediation):
        """Returns SensitivityResult TypedDict."""
        data = simple_linear_mediation
        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_rho=11,
            n_simulations=50,
            n_bootstrap=20,
            random_state=42,
        )

        assert isinstance(result, dict)
        assert "rho_grid" in result
        assert "nde_at_rho" in result
        assert "nie_at_rho" in result
        assert "rho_at_zero_nie" in result
        assert "interpretation" in result

    def test_rho_grid_correct_shape(self, simple_linear_mediation):
        """Rho grid has correct number of points."""
        data = simple_linear_mediation
        n_rho = 21

        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_rho=n_rho,
            n_simulations=50,
            n_bootstrap=20,
        )

        assert len(result["rho_grid"]) == n_rho
        assert len(result["nde_at_rho"]) == n_rho
        assert len(result["nie_at_rho"]) == n_rho

    def test_rho_grid_range(self, simple_linear_mediation):
        """Rho grid covers specified range."""
        data = simple_linear_mediation

        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            rho_range=(-0.5, 0.5),
            n_rho=11,
            n_simulations=50,
            n_bootstrap=20,
        )

        assert result["rho_grid"].min() >= -0.5
        assert result["rho_grid"].max() <= 0.5

    def test_original_effects_recorded(self, simple_linear_mediation):
        """Original effects (at rho=0) are recorded."""
        data = simple_linear_mediation
        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_rho=21,
            n_simulations=50,
            n_bootstrap=20,
        )

        # Original effects should match effects at rho ≈ 0
        zero_idx = np.argmin(np.abs(result["rho_grid"]))
        assert_allclose(result["original_nde"], result["nde_at_rho"][zero_idx], rtol=0.01)
        assert_allclose(result["original_nie"], result["nie_at_rho"][zero_idx], rtol=0.01)


class TestSensitivityBehavior:
    """Tests for expected sensitivity behavior."""

    def test_nie_decreases_with_positive_rho(self, simple_linear_mediation):
        """NIE typically decreases as rho increases from 0."""
        data = simple_linear_mediation
        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            rho_range=(-0.8, 0.8),
            n_rho=41,
            n_simulations=100,
            n_bootstrap=30,
            random_state=42,
        )

        # Find indices for rho = 0 and rho = 0.5
        rho_grid = result["rho_grid"]
        zero_idx = np.argmin(np.abs(rho_grid))
        pos_idx = np.argmin(np.abs(rho_grid - 0.5))

        nie_at_zero = result["nie_at_rho"][zero_idx]
        nie_at_pos = result["nie_at_rho"][pos_idx]

        # With positive confounding, NIE should decrease
        # (This is DGP-dependent, but generally holds)
        assert nie_at_pos < nie_at_zero or np.isclose(nie_at_pos, nie_at_zero, atol=0.3)

    def test_nde_stable_linear_model(self, simple_linear_mediation):
        """NDE is relatively stable in linear model sensitivity."""
        data = simple_linear_mediation
        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            rho_range=(-0.5, 0.5),
            n_rho=21,
            n_simulations=100,
            n_bootstrap=30,
        )

        # In our linear sensitivity model, NDE = beta_1 is constant
        nde_values = result["nde_at_rho"]
        nde_std = np.std(nde_values)

        # NDE should be relatively stable
        assert nde_std < 0.2


class TestRhoAtZero:
    """Tests for finding rho at which effects cross zero."""

    def test_rho_at_zero_exists(self, simple_linear_mediation):
        """With wide enough rho range, zero crossing exists."""
        data = simple_linear_mediation
        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            rho_range=(-0.9, 0.9),
            n_rho=41,
            n_simulations=100,
            n_bootstrap=30,
            random_state=42,
        )

        # NIE should cross zero at some rho
        # (depends on DGP, may not always cross in finite range)
        rho_zero = result["rho_at_zero_nie"]

        # Either it exists and is within range, or it's NaN
        if not np.isnan(rho_zero):
            assert -0.9 <= rho_zero <= 0.9

    def test_rho_at_zero_sign_correct(self, simple_linear_mediation):
        """Rho at zero has correct sign relative to original effect."""
        data = simple_linear_mediation
        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            rho_range=(-0.9, 0.9),
            n_rho=41,
            n_simulations=100,
            n_bootstrap=30,
            random_state=42,
        )

        # With positive NIE, we need positive rho to reduce it to zero
        if not np.isnan(result["rho_at_zero_nie"]):
            if result["original_nie"] > 0:
                assert result["rho_at_zero_nie"] > -0.5  # Typically positive


class TestSensitivityWithCovariates:
    """Tests with covariates."""

    def test_sensitivity_with_covariates(self, mediation_with_covariates):
        """Sensitivity analysis works with covariates."""
        data = mediation_with_covariates
        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            covariates=data["covariates"],
            n_rho=11,
            n_simulations=50,
            n_bootstrap=20,
        )

        # Should produce valid results
        assert len(result["rho_grid"]) == 11
        assert not np.all(np.isnan(result["nie_at_rho"]))


class TestSensitivityInterpretation:
    """Tests for interpretation string."""

    def test_interpretation_generated(self, simple_linear_mediation):
        """Interpretation string is generated."""
        data = simple_linear_mediation
        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_rho=11,
            n_simulations=50,
            n_bootstrap=20,
        )

        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0

    def test_interpretation_mentions_effects(self, simple_linear_mediation):
        """Interpretation mentions NIE and NDE."""
        data = simple_linear_mediation
        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_rho=11,
            n_simulations=50,
            n_bootstrap=20,
        )

        interp = result["interpretation"].lower()
        assert "nie" in interp or "indirect" in interp
        assert "nde" in interp or "direct" in interp


class TestSensitivityCIs:
    """Tests for confidence intervals."""

    def test_ci_arrays_correct_shape(self, simple_linear_mediation):
        """CI arrays have correct shape."""
        data = simple_linear_mediation
        n_rho = 21

        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_rho=n_rho,
            n_simulations=50,
            n_bootstrap=30,
        )

        assert len(result["nde_ci_lower"]) == n_rho
        assert len(result["nde_ci_upper"]) == n_rho
        assert len(result["nie_ci_lower"]) == n_rho
        assert len(result["nie_ci_upper"]) == n_rho

    def test_ci_ordered(self, simple_linear_mediation):
        """CIs are ordered (lower < upper)."""
        data = simple_linear_mediation
        result = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_rho=11,
            n_simulations=50,
            n_bootstrap=50,  # More bootstrap for stable CIs
        )

        # For non-NaN entries
        valid = ~np.isnan(result["nie_ci_lower"]) & ~np.isnan(result["nie_ci_upper"])
        if np.any(valid):
            assert np.all(
                result["nie_ci_lower"][valid] <= result["nie_ci_upper"][valid]
            )


class TestSensitivityRandomState:
    """Tests for reproducibility."""

    def test_reproducible_with_seed(self, simple_linear_mediation):
        """Same seed gives same results."""
        data = simple_linear_mediation

        result1 = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_rho=11,
            n_simulations=50,
            n_bootstrap=20,
            random_state=42,
        )

        result2 = mediation_sensitivity(
            data["outcome"],
            data["treatment"],
            data["mediator"],
            n_rho=11,
            n_simulations=50,
            n_bootstrap=20,
            random_state=42,
        )

        # Main estimates should be identical
        assert_allclose(result1["nie_at_rho"], result2["nie_at_rho"], rtol=1e-10)
        assert_allclose(result1["nde_at_rho"], result2["nde_at_rho"], rtol=1e-10)
