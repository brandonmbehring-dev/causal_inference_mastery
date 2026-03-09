"""
Cross-Language Parity Tests: Panel QTE (Session 118)

Tests that Python and Julia Panel QTE implementations produce consistent results.

Note: These tests require juliacall to be installed and Julia to be available.
Tests are skipped if Julia is not available.
"""

import numpy as np
import pytest

from causal_inference.panel import (
    panel_rif_qte,
    panel_unconditional_qte,
    panel_rif_qte_band,
    PanelData,
)

# Try to import Julia interface
try:
    from tests.validation.cross_language.julia_interface import (
        julia_panel_rif_qte,
        julia_panel_rif_qte_band,
        julia_panel_unconditional_qte,
        JULIA_AVAILABLE,
    )
except ImportError:
    JULIA_AVAILABLE = False


# Skip all tests if Julia not available
pytestmark = pytest.mark.skipif(
    not JULIA_AVAILABLE,
    reason="Julia not available (juliacall not installed or Julia not found)",
)


def generate_panel_data(
    n_units: int = 50,
    n_periods: int = 10,
    true_qte: float = 2.0,
    seed: int = 42,
) -> tuple:
    """Generate panel data for cross-language testing."""
    np.random.seed(seed)

    n_obs = n_units * n_periods

    # Panel structure
    unit_id = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    # Covariates
    X = np.random.randn(n_obs, 2)

    # Random treatment
    D = (np.random.rand(n_obs) < 0.5).astype(float)

    # Outcome: Y = X1 + true_qte * D + epsilon
    Y = X[:, 0] + true_qte * D + np.random.randn(n_obs)

    return Y, D, X, unit_id, time


class TestPanelRIFQTEParity:
    """Test Python-Julia parity for panel_rif_qte."""

    def test_median_qte_parity(self):
        """Test that median QTE estimates match between Python and Julia."""
        Y, D, X, unit_id, time = generate_panel_data(seed=123)

        # Python
        panel_py = PanelData(outcomes=Y, treatment=D, covariates=X, unit_id=unit_id, time=time)
        result_py = panel_rif_qte(panel_py, quantile=0.5)

        # Julia
        result_jl = julia_panel_rif_qte(
            outcomes=Y,
            treatment=D,
            covariates=X,
            unit_id=unit_id,
            time=time,
            tau=0.5,
        )

        # Compare QTE estimates (allow for numerical differences)
        np.testing.assert_allclose(
            result_py.qte,
            result_jl["qte"],
            rtol=0.02,
            err_msg="QTE estimates differ between Python and Julia",
        )

        # Compare SEs (allow more tolerance due to clustered SE variations)
        np.testing.assert_allclose(
            result_py.qte_se,
            result_jl["qte_se"],
            rtol=0.10,
            err_msg="SE estimates differ between Python and Julia",
        )

    def test_quantile_25_parity(self):
        """Test 25th percentile QTE parity."""
        Y, D, X, unit_id, time = generate_panel_data(seed=456)

        panel_py = PanelData(outcomes=Y, treatment=D, covariates=X, unit_id=unit_id, time=time)
        result_py = panel_rif_qte(panel_py, quantile=0.25)
        result_jl = julia_panel_rif_qte(
            outcomes=Y, treatment=D, covariates=X, unit_id=unit_id, time=time, tau=0.25
        )

        np.testing.assert_allclose(
            result_py.qte, result_jl["qte"], rtol=0.05, err_msg="Q25 QTE mismatch"
        )

    def test_quantile_75_parity(self):
        """Test 75th percentile QTE parity."""
        Y, D, X, unit_id, time = generate_panel_data(seed=789)

        panel_py = PanelData(outcomes=Y, treatment=D, covariates=X, unit_id=unit_id, time=time)
        result_py = panel_rif_qte(panel_py, quantile=0.75)
        result_jl = julia_panel_rif_qte(
            outcomes=Y, treatment=D, covariates=X, unit_id=unit_id, time=time, tau=0.75
        )

        np.testing.assert_allclose(
            result_py.qte, result_jl["qte"], rtol=0.05, err_msg="Q75 QTE mismatch"
        )

    def test_diagnostics_parity(self):
        """Test that diagnostic quantities match."""
        Y, D, X, unit_id, time = generate_panel_data(seed=111)

        panel_py = PanelData(outcomes=Y, treatment=D, covariates=X, unit_id=unit_id, time=time)
        result_py = panel_rif_qte(panel_py, quantile=0.5)
        result_jl = julia_panel_rif_qte(
            outcomes=Y, treatment=D, covariates=X, unit_id=unit_id, time=time, tau=0.5
        )

        # n_obs and n_units should match exactly
        assert result_py.n_obs == result_jl["n_obs"], "n_obs mismatch"
        assert result_py.n_units == result_jl["n_units"], "n_units mismatch"

        # Outcome quantile and density should be close
        np.testing.assert_allclose(
            result_py.outcome_quantile,
            result_jl["outcome_quantile"],
            rtol=0.01,
            err_msg="Outcome quantile mismatch",
        )

        np.testing.assert_allclose(
            result_py.density_at_quantile,
            result_jl["density_at_quantile"],
            rtol=0.05,
            err_msg="Density estimate mismatch",
        )


class TestPanelQTEBandParity:
    """Test Python-Julia parity for panel_rif_qte_band."""

    def test_default_quantiles_parity(self):
        """Test that default band estimates match."""
        Y, D, X, unit_id, time = generate_panel_data(seed=222)

        panel_py = PanelData(outcomes=Y, treatment=D, covariates=X, unit_id=unit_id, time=time)
        result_py = panel_rif_qte_band(panel_py)
        result_jl = julia_panel_rif_qte_band(
            outcomes=Y, treatment=D, covariates=X, unit_id=unit_id, time=time
        )

        # Quantiles should match
        np.testing.assert_array_equal(
            result_py.quantiles,
            result_jl["quantiles"],
            err_msg="Quantile arrays differ",
        )

        # QTE estimates should be close
        np.testing.assert_allclose(
            result_py.qtes,
            result_jl["qtes"],
            rtol=0.05,
            err_msg="Band QTE estimates differ",
        )

    def test_custom_quantiles_parity(self):
        """Test custom quantile band parity."""
        Y, D, X, unit_id, time = generate_panel_data(seed=333)
        custom_quantiles = np.array([0.1, 0.5, 0.9])

        panel_py = PanelData(outcomes=Y, treatment=D, covariates=X, unit_id=unit_id, time=time)
        result_py = panel_rif_qte_band(panel_py, quantiles=custom_quantiles)
        result_jl = julia_panel_rif_qte_band(
            outcomes=Y,
            treatment=D,
            covariates=X,
            unit_id=unit_id,
            time=time,
            quantiles=custom_quantiles,
        )

        np.testing.assert_allclose(
            result_py.qtes, result_jl["qtes"], rtol=0.05, err_msg="Custom band mismatch"
        )


class TestPanelUnconditionalQTEParity:
    """Test Python-Julia parity for panel_unconditional_qte."""

    def test_median_unconditional_parity(self):
        """Test unconditional QTE parity with fixed seed."""
        Y, D, X, unit_id, time = generate_panel_data(seed=444)

        panel_py = PanelData(outcomes=Y, treatment=D, covariates=X, unit_id=unit_id, time=time)
        result_py = panel_unconditional_qte(
            panel_py, quantile=0.5, n_bootstrap=200, random_state=42
        )
        result_jl = julia_panel_unconditional_qte(
            outcomes=Y,
            treatment=D,
            covariates=X,
            unit_id=unit_id,
            time=time,
            tau=0.5,
            n_bootstrap=200,
            random_state=42,
        )

        # Point estimates should match exactly (same algorithm, same seed)
        np.testing.assert_allclose(
            result_py.qte,
            result_jl["qte"],
            rtol=0.01,
            err_msg="Unconditional QTE mismatch",
        )

    def test_cluster_bootstrap_effect(self):
        """Test that cluster bootstrap is implemented consistently."""
        Y, D, X, unit_id, time = generate_panel_data(n_units=30, n_periods=20, seed=555)

        panel_py = PanelData(outcomes=Y, treatment=D, covariates=X, unit_id=unit_id, time=time)

        # Cluster bootstrap
        result_cluster_py = panel_unconditional_qte(
            panel_py,
            quantile=0.5,
            n_bootstrap=100,
            cluster_bootstrap=True,
            random_state=99,
        )
        result_cluster_jl = julia_panel_unconditional_qte(
            outcomes=Y,
            treatment=D,
            covariates=X,
            unit_id=unit_id,
            time=time,
            tau=0.5,
            n_bootstrap=100,
            cluster_bootstrap=True,
            random_state=99,
        )

        # Both should produce similar SE (within 20% given bootstrap variation)
        np.testing.assert_allclose(
            result_cluster_py.qte_se,
            result_cluster_jl["qte_se"],
            rtol=0.20,
            err_msg="Cluster bootstrap SE differs too much",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
