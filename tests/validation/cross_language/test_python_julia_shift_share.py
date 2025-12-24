"""
Python ↔ Julia Shift-Share IV Parity Tests

Tests that Python and Julia shift-share IV implementations produce
numerically equivalent results for:
- Basic estimation
- First-stage diagnostics
- Rotemberg weights
- Standard errors

Session 97: Julia Shift-Share IV Implementation

References:
- Bartik (1991) - Original shift-share instrument
- Goldsmith-Pinkham et al. (2020) - Bartik instruments methodology
- Rotemberg (1983) - Weight decomposition diagnostics
"""

import numpy as np
import pytest

from causal_inference.shift_share import shift_share_iv
from tests.validation.cross_language.julia_interface import (
    is_julia_available,
    julia_shift_share_iv,
)

# Skip all tests if Julia is not available
pytestmark = pytest.mark.skipif(
    not is_julia_available(),
    reason="Julia not available for cross-language validation",
)


def generate_shift_share_data(
    n: int = 200,
    n_sectors: int = 10,
    true_beta: float = 1.5,
    first_stage_strength: float = 2.0,
    seed: int = 42,
):
    """Generate shift-share data for parity testing."""
    rng = np.random.default_rng(seed)

    # Dirichlet shares (sum to 1)
    alpha = np.ones(n_sectors)
    shares = rng.dirichlet(alpha, n)

    # Shocks
    shocks = rng.normal(0, 0.1, n_sectors)

    # Bartik instrument
    Z = shares @ shocks

    # Treatment
    D = first_stage_strength * Z + rng.normal(0, 0.5, n)

    # Outcome
    Y = true_beta * D + rng.normal(0, 1, n)

    return Y, D, shares, shocks


class TestBasicEstimationParity:
    """Test basic estimation parity between Python and Julia."""

    def test_estimate_parity(self):
        """Point estimates match within tolerance."""
        Y, D, shares, shocks = generate_shift_share_data(seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        # Estimates should match closely
        np.testing.assert_allclose(
            py_result["estimate"],
            jl_result["estimate"],
            rtol=1e-4,
            err_msg="Point estimates differ"
        )

    def test_se_parity(self):
        """Standard errors match within tolerance."""
        Y, D, shares, shocks = generate_shift_share_data(seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        # SEs should be close (HC3 can have small numerical differences)
        np.testing.assert_allclose(
            py_result["se"],
            jl_result["se"],
            rtol=0.1,  # 10% tolerance for SE
            err_msg="Standard errors differ too much"
        )

    def test_t_stat_parity(self):
        """T-statistics match."""
        Y, D, shares, shocks = generate_shift_share_data(seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        np.testing.assert_allclose(
            py_result["t_stat"],
            jl_result["t_stat"],
            rtol=0.1,
            err_msg="T-statistics differ"
        )

    def test_ci_parity(self):
        """Confidence intervals match."""
        Y, D, shares, shocks = generate_shift_share_data(seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        np.testing.assert_allclose(
            py_result["ci_lower"],
            jl_result["ci_lower"],
            rtol=0.1,
            err_msg="CI lower bounds differ"
        )
        np.testing.assert_allclose(
            py_result["ci_upper"],
            jl_result["ci_upper"],
            rtol=0.1,
            err_msg="CI upper bounds differ"
        )


class TestFirstStageParity:
    """Test first-stage diagnostics parity."""

    def test_f_statistic_parity(self):
        """F-statistics match."""
        Y, D, shares, shocks = generate_shift_share_data(seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        np.testing.assert_allclose(
            py_result["first_stage"]["f_statistic"],
            jl_result["first_stage"]["f_statistic"],
            rtol=0.1,
            err_msg="F-statistics differ"
        )

    def test_first_stage_coefficient_parity(self):
        """First-stage coefficients match."""
        Y, D, shares, shocks = generate_shift_share_data(seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        np.testing.assert_allclose(
            py_result["first_stage"]["coefficient"],
            jl_result["first_stage"]["coefficient"],
            rtol=1e-4,
            err_msg="First-stage coefficients differ"
        )

    def test_partial_r2_parity(self):
        """Partial R-squared values match."""
        Y, D, shares, shocks = generate_shift_share_data(seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        np.testing.assert_allclose(
            py_result["first_stage"]["partial_r2"],
            jl_result["first_stage"]["partial_r2"],
            rtol=0.1,
            err_msg="Partial R-squared values differ"
        )

    def test_weak_iv_warning_parity(self):
        """Weak IV warnings agree."""
        Y, D, shares, shocks = generate_shift_share_data(seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        assert py_result["first_stage"]["weak_iv_warning"] == \
               jl_result["first_stage"]["weak_iv_warning"], \
               "Weak IV warning flags differ"


class TestRotembergWeightsParity:
    """Test Rotemberg weight diagnostics parity."""

    def test_weights_parity(self):
        """Rotemberg weights match."""
        Y, D, shares, shocks = generate_shift_share_data(seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        np.testing.assert_allclose(
            py_result["rotemberg"]["weights"],
            jl_result["rotemberg"]["weights"],
            rtol=1e-4,
            err_msg="Rotemberg weights differ"
        )

    def test_negative_weight_share_parity(self):
        """Negative weight share matches."""
        Y, D, shares, shocks = generate_shift_share_data(seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        np.testing.assert_allclose(
            py_result["rotemberg"]["negative_weight_share"],
            jl_result["rotemberg"]["negative_weight_share"],
            rtol=1e-4,
            err_msg="Negative weight shares differ"
        )

    def test_herfindahl_parity(self):
        """Herfindahl index matches."""
        Y, D, shares, shocks = generate_shift_share_data(seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        np.testing.assert_allclose(
            py_result["rotemberg"]["herfindahl"],
            jl_result["rotemberg"]["herfindahl"],
            rtol=1e-4,
            err_msg="Herfindahl indices differ"
        )


class TestSampleSizeParity:
    """Test parity across different sample sizes."""

    @pytest.mark.parametrize("n", [100, 200, 500])
    def test_different_sample_sizes(self, n):
        """Estimates match across sample sizes."""
        Y, D, shares, shocks = generate_shift_share_data(n=n, seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        np.testing.assert_allclose(
            py_result["estimate"],
            jl_result["estimate"],
            rtol=1e-4,
            err_msg=f"Estimates differ for n={n}"
        )


class TestSectorCountParity:
    """Test parity across different sector counts."""

    @pytest.mark.parametrize("n_sectors", [5, 10, 20])
    def test_different_sector_counts(self, n_sectors):
        """Estimates match across sector counts."""
        Y, D, shares, shocks = generate_shift_share_data(
            n_sectors=n_sectors, seed=42
        )

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        np.testing.assert_allclose(
            py_result["estimate"],
            jl_result["estimate"],
            rtol=1e-4,
            err_msg=f"Estimates differ for n_sectors={n_sectors}"
        )


class TestMetadataParity:
    """Test metadata parity."""

    def test_n_obs_parity(self):
        """Number of observations matches."""
        Y, D, shares, shocks = generate_shift_share_data(n=200, seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        assert py_result["n_obs"] == jl_result["n_obs"]

    def test_n_sectors_parity(self):
        """Number of sectors matches."""
        Y, D, shares, shocks = generate_shift_share_data(n_sectors=15, seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        assert py_result["n_sectors"] == jl_result["n_sectors"]

    def test_share_sum_mean_parity(self):
        """Share sum mean matches."""
        Y, D, shares, shocks = generate_shift_share_data(seed=42)

        py_result = shift_share_iv(Y, D, shares, shocks)
        jl_result = julia_shift_share_iv(Y, D, shares, shocks)

        np.testing.assert_allclose(
            py_result["share_sum_mean"],
            jl_result["share_sum_mean"],
            rtol=1e-4,
            err_msg="Share sum means differ"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
