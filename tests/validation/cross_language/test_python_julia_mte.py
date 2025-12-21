"""
Cross-language validation tests for Marginal Treatment Effects.

Tests Python ↔ Julia parity for MTE estimators (Session 91).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.mte import (
    late_estimator,
    late_bounds,
    local_iv,
    ate_from_mte,
)

# Import Julia interface with skip if unavailable
try:
    from tests.validation.cross_language.julia_interface import (
        is_julia_available,
        julia_late_estimator,
        julia_late_bounds,
        julia_local_iv,
        julia_ate_from_mte,
    )
    JULIA_AVAILABLE = is_julia_available()
except ImportError:
    JULIA_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not JULIA_AVAILABLE,
    reason="Julia not available for cross-language tests"
)


# =============================================================================
# Test Fixtures
# =============================================================================


def generate_simple_binary_iv_data(
    n: int = 500,
    true_late: float = 2.0,
    complier_share: float = 0.4,
    random_seed: int = 42,
):
    """
    Generate simple binary IV data with known LATE.

    Selection: D = 1 if U < 0.3 + 0.4*Z (complier share = 0.4)
    Outcome: Y = 1 + true_late*D + noise
    """
    np.random.seed(random_seed)

    n = int(n)
    Z = np.random.binomial(1, 0.5, n).astype(float)
    U = np.random.uniform(0, 1, n)

    # Selection model
    base_prob = 0.3
    first_stage = complier_share  # 0.4
    D = (U < (base_prob + first_stage * Z)).astype(float)

    # Outcome model
    Y = 1.0 + true_late * D + 0.3 * np.random.randn(n)

    return {
        "outcome": Y,
        "treatment": D,
        "instrument": Z,
        "true_late": true_late,
        "true_complier_share": complier_share,
    }


def generate_heterogeneous_mte_data(
    n: int = 800,
    random_seed: int = 42,
):
    """
    Generate data with linearly decreasing MTE.

    MTE(u) = 3 - 2u (linearly decreasing in resistance)
    """
    np.random.seed(random_seed)

    n = int(n)
    Z = np.random.randn(n)
    U = np.random.uniform(0, 1, n)

    # Propensity: P(D=1|Z) = Phi(Z)
    from scipy.stats import norm
    propensity = norm.cdf(Z)

    # Selection
    D = (U < propensity).astype(float)

    # MTE(u) = 3 - 2u
    mte_individual = 3.0 - 2.0 * U
    Y = 1.0 + mte_individual * D + 0.5 * np.random.randn(n)

    return {
        "outcome": Y,
        "treatment": D,
        "instrument": Z,
        "true_mte_slope": -2.0,
    }


# =============================================================================
# LATE Parity Tests
# =============================================================================


class TestLATEParity:
    """Python ↔ Julia parity for LATE estimator."""

    def test_late_estimate_parity(self):
        """LATE point estimates should match."""
        data = generate_simple_binary_iv_data(n=500, random_seed=42)

        # Python
        py_result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # Julia
        jl_result = julia_late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # Point estimate should match closely (same algorithm)
        assert_allclose(py_result["late"], jl_result["late"], rtol=1e-6)

    def test_late_se_parity(self):
        """LATE standard errors should match."""
        data = generate_simple_binary_iv_data(n=500, random_seed=42)

        py_result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        jl_result = julia_late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # SE should match
        assert_allclose(py_result["se"], jl_result["se"], rtol=1e-6)

    def test_late_ci_parity(self):
        """LATE confidence intervals should match."""
        data = generate_simple_binary_iv_data(n=500, random_seed=42)

        py_result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        jl_result = julia_late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        assert_allclose(py_result["ci_lower"], jl_result["ci_lower"], rtol=1e-6)
        assert_allclose(py_result["ci_upper"], jl_result["ci_upper"], rtol=1e-6)

    def test_late_complier_share_parity(self):
        """Complier share estimates should match."""
        data = generate_simple_binary_iv_data(n=500, random_seed=42)

        py_result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        jl_result = julia_late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        assert_allclose(
            py_result["complier_share"],
            jl_result["complier_share"],
            rtol=1e-6,
        )

    def test_late_first_stage_parity(self):
        """First-stage diagnostics should match."""
        data = generate_simple_binary_iv_data(n=500, random_seed=42)

        py_result = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        jl_result = julia_late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        assert_allclose(
            py_result["first_stage_coef"],
            jl_result["first_stage_coef"],
            rtol=1e-6,
        )
        assert_allclose(
            py_result["first_stage_f"],
            jl_result["first_stage_f"],
            rtol=1e-4,
        )


# =============================================================================
# LATE Bounds Parity Tests
# =============================================================================


class TestLATEBoundsParity:
    """Python ↔ Julia parity for LATE bounds."""

    def test_bounds_parity(self):
        """LATE bounds should match."""
        data = generate_simple_binary_iv_data(n=500, random_seed=42)

        # Python
        py_result = late_bounds(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # Julia
        jl_result = julia_late_bounds(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        assert_allclose(
            py_result["bounds_lower"],
            jl_result["bounds_lower"],
            rtol=1e-6,
        )
        assert_allclose(
            py_result["bounds_upper"],
            jl_result["bounds_upper"],
            rtol=1e-6,
        )

    def test_bounds_monotonicity_estimate_parity(self):
        """LATE under monotonicity should match."""
        data = generate_simple_binary_iv_data(n=500, random_seed=42)

        py_result = late_bounds(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        jl_result = julia_late_bounds(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        assert_allclose(
            py_result["late_under_monotonicity"],
            jl_result["late_under_monotonicity"],
            rtol=1e-6,
        )


# =============================================================================
# Local IV Parity Tests
# =============================================================================


class TestLocalIVParity:
    """Python ↔ Julia parity for local IV MTE."""

    def test_mte_grid_shape_parity(self):
        """MTE grid shapes should match."""
        data = generate_heterogeneous_mte_data(n=500, random_seed=42)

        n_grid = 15

        py_result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_grid=n_grid,
            n_bootstrap=50,
            random_state=42,
        )

        jl_result = julia_local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_grid=n_grid,
            n_bootstrap=50,
            seed=42,
        )

        # Grid lengths should match
        assert len(py_result["mte_grid"]) == len(jl_result["mte_grid"])
        assert len(py_result["u_grid"]) == len(jl_result["u_grid"])

    def test_mte_u_grid_parity(self):
        """U grid values should match."""
        data = generate_heterogeneous_mte_data(n=500, random_seed=42)

        py_result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_grid=15,
            n_bootstrap=50,
            random_state=42,
        )

        jl_result = julia_local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_grid=15,
            n_bootstrap=50,
            seed=42,
        )

        # U grid should match closely (same propensity estimation)
        assert_allclose(py_result["u_grid"], jl_result["u_grid"], rtol=0.1)

    def test_mte_slope_direction_parity(self):
        """Both implementations should detect same MTE slope direction."""
        data = generate_heterogeneous_mte_data(n=800, random_seed=42)

        py_result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_grid=20,
            n_bootstrap=50,
            random_state=42,
        )

        jl_result = julia_local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_grid=20,
            n_bootstrap=50,
            seed=42,
        )

        # Both should detect negative slope
        py_valid = ~np.isnan(py_result["mte_grid"])
        jl_valid = ~np.isnan(jl_result["mte_grid"])

        if py_valid.sum() > 5 and jl_valid.sum() > 5:
            py_slope = np.polyfit(
                py_result["u_grid"][py_valid],
                py_result["mte_grid"][py_valid],
                1,
            )[0]
            jl_slope = np.polyfit(
                jl_result["u_grid"][jl_valid],
                jl_result["mte_grid"][jl_valid],
                1,
            )[0]

            # Both should be negative (true slope is -2)
            assert py_slope < 0
            assert jl_slope < 0


# =============================================================================
# Policy Parameter Parity Tests
# =============================================================================


class TestPolicyParity:
    """Python ↔ Julia parity for policy parameters."""

    def test_ate_from_mte_parity(self):
        """ATE from MTE should match."""
        # Create a simple MTE curve for testing
        u_grid = np.linspace(0.1, 0.9, 20)
        mte_grid = 3.0 - 2.0 * u_grid  # Linear MTE
        se_grid = np.full_like(mte_grid, 0.2)

        # Python
        mte_result = {
            "mte_grid": mte_grid,
            "u_grid": u_grid,
            "se_grid": se_grid,
            "ci_lower": mte_grid - 1.96 * se_grid,
            "ci_upper": mte_grid + 1.96 * se_grid,
            "propensity_support": (0.1, 0.9),
            "n_obs": 500,
            "n_trimmed": 0,
            "bandwidth": 0.1,
            "method": "local_iv",
        }

        py_result = ate_from_mte(mte_result)

        # Julia
        jl_result = julia_ate_from_mte(
            mte_grid,
            u_grid,
            se_grid,
            (0.1, 0.9),
            500,
        )

        # ATE estimates should match
        assert_allclose(py_result["estimate"], jl_result["estimate"], rtol=1e-6)


# =============================================================================
# Integration Tests
# =============================================================================


class TestMTEIntegration:
    """Integration tests across Python and Julia."""

    def test_full_pipeline_consistency(self):
        """Full MTE pipeline should give consistent results."""
        data = generate_simple_binary_iv_data(n=600, random_seed=123)

        # Both should recover approximately the true LATE
        py_late = late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        jl_late = julia_late_estimator(
            data["outcome"],
            data["treatment"],
            data["instrument"],
        )

        # Both should be close to true value
        assert abs(py_late["late"] - data["true_late"]) < 0.5
        assert abs(jl_late["late"] - data["true_late"]) < 0.5

        # And should match each other
        assert_allclose(py_late["late"], jl_late["late"], rtol=1e-6)
