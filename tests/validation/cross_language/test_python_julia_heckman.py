"""
Cross-language validation tests for Heckman Selection Model.

Tests Python ↔ Julia parity for Heckman two-step estimator (Session 85).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.selection import heckman_two_step, selection_bias_test

# Import Julia interface with skip if unavailable
try:
    from tests.validation.cross_language.julia_interface import (
        is_julia_available,
        julia_heckman_two_step,
        julia_compute_imr,
    )

    JULIA_AVAILABLE = is_julia_available()
except ImportError:
    JULIA_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not JULIA_AVAILABLE, reason="Julia not available for cross-language tests"
)


# =============================================================================
# Test Fixtures
# =============================================================================


def generate_heckman_data(
    n: int = 500,
    rho: float = 0.5,
    true_beta: float = 2.0,
    random_seed: int = 42,
):
    """
    Generate Heckman DGP for testing.

    Selection: S* = γ₀ + γ₁Z + γ₂X + v, S = 1(S* > 0)
    Outcome:   Y = β₀ + β₁X + u (observed only if S=1)
    Corr(u, v) = ρ

    Returns tuple with all data components.
    """
    np.random.seed(random_seed)

    # Covariates
    X = np.random.randn(n)
    Z = np.random.randn(n)  # Exclusion restriction

    # Correlated errors: (u, v) with correlation ρ
    u = np.random.randn(n)
    v = rho * u + np.sqrt(1 - rho**2) * np.random.randn(n)

    # Selection equation
    gamma = np.array([0.5, 1.0, 0.3])  # intercept, Z, X
    s_star = gamma[0] + gamma[1] * Z + gamma[2] * X + v
    selected = s_star > 0

    # Outcome equation
    beta = np.array([1.0, true_beta])  # intercept, X
    outcomes = beta[0] + beta[1] * X + u
    outcomes[~selected] = np.nan

    # Prepare covariates
    sel_cov = np.column_stack([X, Z])  # For selection: includes exclusion
    out_cov = X.reshape(-1, 1)  # For outcome: only X

    return {
        "outcomes": outcomes,
        "selected": selected,
        "sel_cov": sel_cov,
        "out_cov": out_cov,
        "true_beta": true_beta,
        "true_rho": rho,
        "n_selected": selected.sum(),
    }


# =============================================================================
# Heckman Two-Step Parity Tests
# =============================================================================


class TestHeckmanEstimateParity:
    """Python ↔ Julia parity for Heckman estimate."""

    def test_estimate_parity_moderate_selection(self):
        """Estimates should match within tolerance (ρ=0.5)."""
        data = generate_heckman_data(n=500, rho=0.5, true_beta=2.0, random_seed=42)

        # Python
        py_result = heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
            alpha=0.05,
        )

        # Julia
        jl_result = julia_heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
            alpha=0.05,
        )

        # Primary estimate parity
        assert_allclose(
            py_result["estimate"],
            jl_result["estimate"],
            rtol=0.05,
            err_msg=f"Estimate mismatch: Python={py_result['estimate']:.4f}, Julia={jl_result['estimate']:.4f}",
        )

    def test_se_parity(self):
        """Standard errors should match within tolerance."""
        data = generate_heckman_data(n=500, rho=0.5, true_beta=2.0, random_seed=123)

        py_result = heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        jl_result = julia_heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        # SE parity (Heckman SE can differ due to variance correction details)
        assert_allclose(
            py_result["se"],
            jl_result["se"],
            rtol=0.15,
            err_msg=f"SE mismatch: Python={py_result['se']:.4f}, Julia={jl_result['se']:.4f}",
        )

    def test_ci_parity(self):
        """Confidence intervals should match."""
        data = generate_heckman_data(n=500, rho=0.3, random_seed=456)

        py_result = heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        jl_result = julia_heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        # CI should span similar range
        assert_allclose(
            py_result["ci_lower"],
            jl_result["ci_lower"],
            rtol=0.15,
        )
        assert_allclose(
            py_result["ci_upper"],
            jl_result["ci_upper"],
            rtol=0.15,
        )


class TestSelectionParametersParity:
    """Python ↔ Julia parity for selection parameters."""

    def test_rho_parity(self):
        """Selection correlation ρ should match."""
        data = generate_heckman_data(n=500, rho=0.6, random_seed=789)

        py_result = heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        jl_result = julia_heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        # ρ should have same sign and similar magnitude
        assert np.sign(py_result["rho"]) == np.sign(jl_result["rho"]), (
            f"ρ sign mismatch: Python={py_result['rho']:.4f}, Julia={jl_result['rho']:.4f}"
        )

        assert_allclose(
            py_result["rho"],
            jl_result["rho"],
            rtol=0.20,
            err_msg=f"ρ mismatch: Python={py_result['rho']:.4f}, Julia={jl_result['rho']:.4f}",
        )

    def test_lambda_parity(self):
        """IMR coefficient λ should match."""
        data = generate_heckman_data(n=500, rho=0.5, random_seed=101)

        py_result = heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        jl_result = julia_heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        assert_allclose(
            py_result["lambda_coef"],
            jl_result["lambda_coef"],
            rtol=0.15,
            err_msg=f"λ mismatch: Python={py_result['lambda_coef']:.4f}, Julia={jl_result['lambda_coef']:.4f}",
        )

    def test_lambda_se_parity(self):
        """IMR coefficient SE should match."""
        data = generate_heckman_data(n=500, rho=0.4, random_seed=202)

        py_result = heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        jl_result = julia_heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        assert_allclose(
            py_result["lambda_se"],
            jl_result["lambda_se"],
            rtol=0.20,
        )


class TestIMRComputationParity:
    """Python ↔ Julia parity for IMR computation."""

    def test_imr_values_parity(self):
        """IMR values should match exactly (mathematical formula)."""
        # Test with standard probabilities
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Python IMR (using scipy)
        from scipy.stats import norm

        py_imr = norm.pdf(norm.ppf(probs)) / probs

        # Julia IMR
        jl_imr = julia_compute_imr(probs)

        # Should match exactly (same formula)
        assert_allclose(py_imr, jl_imr, rtol=1e-10, err_msg="IMR computation should match exactly")

    def test_imr_from_solution_parity(self):
        """IMR values from full solution should be correlated."""
        data = generate_heckman_data(n=300, rho=0.5, random_seed=303)

        py_result = heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        jl_result = julia_heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        # IMR correlation (probit may differ slightly)
        corr = np.corrcoef(py_result["imr"], jl_result["imr"])[0, 1]
        assert corr > 0.95, f"IMR correlation {corr:.3f} below threshold"


class TestSelectionProbabilityParity:
    """Python ↔ Julia parity for selection probabilities."""

    def test_selection_probs_correlation(self):
        """Selection probabilities should be highly correlated."""
        data = generate_heckman_data(n=400, rho=0.4, random_seed=404)

        py_result = heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        jl_result = julia_heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        # Probit probabilities should be similar
        corr = np.corrcoef(py_result["selection_probs"], jl_result["selection_probs"])[0, 1]
        assert corr > 0.99, f"Selection probability correlation {corr:.4f} below threshold"

    def test_sample_sizes_match(self):
        """Sample sizes should match exactly."""
        data = generate_heckman_data(n=500, random_seed=505)

        py_result = heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        jl_result = julia_heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        assert py_result["n_total"] == jl_result["n_total"]
        assert py_result["n_selected"] == jl_result["n_selected"]


class TestNoSelectionScenario:
    """Parity tests when ρ ≈ 0 (no selection bias)."""

    def test_parity_no_selection(self):
        """Both should detect ρ ≈ 0 when no selection bias."""
        data = generate_heckman_data(n=500, rho=0.0, random_seed=600)

        py_result = heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        jl_result = julia_heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        # Both should have small ρ
        assert abs(py_result["rho"]) < 0.5, f"Python ρ={py_result['rho']:.4f} unexpectedly large"
        assert abs(jl_result["rho"]) < 0.5, f"Julia ρ={jl_result['rho']:.4f} unexpectedly large"

        # Estimates should be similar
        assert_allclose(
            py_result["estimate"],
            jl_result["estimate"],
            rtol=0.10,
        )


class TestStrongSelectionScenario:
    """Parity tests when ρ is large (strong selection bias)."""

    def test_parity_strong_selection(self):
        """Both should handle strong selection similarly."""
        data = generate_heckman_data(n=500, rho=0.8, true_beta=1.5, random_seed=700)

        py_result = heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        jl_result = julia_heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        # Both should detect positive ρ
        assert py_result["rho"] > 0.3, f"Python ρ={py_result['rho']:.4f} unexpectedly small"
        assert jl_result["rho"] > 0.3, f"Julia ρ={jl_result['rho']:.4f} unexpectedly small"

        # Estimates should be reasonably close
        assert_allclose(
            py_result["estimate"],
            jl_result["estimate"],
            rtol=0.20,
        )


class TestNegativeSelectionScenario:
    """Parity tests when ρ < 0 (negative selection bias)."""

    def test_parity_negative_selection(self):
        """Both should handle negative selection similarly."""
        data = generate_heckman_data(n=500, rho=-0.6, random_seed=800)

        py_result = heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        jl_result = julia_heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        # λ coefficient should have same sign
        py_sign = np.sign(py_result["lambda_coef"])
        jl_sign = np.sign(jl_result["lambda_coef"])
        assert py_sign == jl_sign, (
            f"λ sign mismatch: Python={py_result['lambda_coef']:.4f}, Julia={jl_result['lambda_coef']:.4f}"
        )


class TestMultipleSeedsConsistency:
    """Test consistency across multiple random seeds."""

    @pytest.mark.parametrize("seed", [1, 42, 123, 456, 789])
    def test_estimate_parity_multiple_seeds(self, seed):
        """Estimates should match across different seeds."""
        data = generate_heckman_data(n=400, rho=0.5, random_seed=seed)

        py_result = heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        jl_result = julia_heckman_two_step(
            outcomes=data["outcomes"],
            selected=data["selected"],
            selection_covariates=data["sel_cov"],
            outcome_covariates=data["out_cov"],
        )

        # Relaxed tolerance for cross-seed consistency
        assert_allclose(
            py_result["estimate"],
            jl_result["estimate"],
            rtol=0.10,
            err_msg=f"Estimate mismatch at seed={seed}",
        )
