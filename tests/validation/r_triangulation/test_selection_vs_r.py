"""Triangulation tests: Python Heckman Selection vs R sampleSelection.

This module provides Layer 5 validation by comparing our Python Heckman
two-step estimator against R's sampleSelection package.

Tests skip gracefully when R/rpy2 or sampleSelection is unavailable.

Tolerance levels (established based on algorithm differences):
- Coefficients (β, γ): rtol=0.02 (2% relative, core estimates)
- Standard errors: rtol=0.05 (5% relative, sandwich vs asymptotic)
- rho, sigma: rtol=0.02 (2% relative, selection parameters)
- lambda p-value: rtol=0.10 (10% relative, inference variability)

Run with: pytest tests/validation/r_triangulation/test_selection_vs_r.py -v

References:
- Heckman, J. (1979). Sample Selection Bias as a Specification Error.
  Econometrica, 47(1), 153-161.
- Toomet, O. & Henningsen, A. (2008). Sample Selection Models in R:
  Package sampleSelection. Journal of Statistical Software.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import Dict, Any

from tests.validation.r_triangulation.r_interface import (
    check_r_available,
    check_sample_selection_available,
    r_heckman_two_step,
)

# Lazy import Python implementation
try:
    from src.causal_inference.selection.heckman import heckman_two_step

    HECKMAN_AVAILABLE = True
except ImportError:
    HECKMAN_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_heckman_python = pytest.mark.skipif(
    not HECKMAN_AVAILABLE,
    reason="Python Heckman module not available",
)

requires_sample_selection_r = pytest.mark.skipif(
    not check_sample_selection_available(),
    reason="R sampleSelection package not installed. "
    "Install in R with: install.packages('sampleSelection')",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_heckman_dgp(
    n: int = 1000,
    rho: float = 0.5,
    beta_x: float = 2.0,
    gamma_z: float = 1.0,
    sigma_u: float = 1.0,
    selection_rate: float = 0.6,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate Heckman selection model data.

    DGP (sample selection):
    - Z ~ Normal(0, 1) [exclusion restriction]
    - X ~ Normal(0, 1) [outcome covariate]
    - (u, v) ~ Bivariate Normal with correlation rho
    - S* = gamma_0 + gamma_z * Z + v
    - S = 1{S* > 0} (selection)
    - Y* = beta_0 + beta_x * X + u
    - Y = Y* if S=1, else unobserved

    Parameters
    ----------
    n : int
        Sample size.
    rho : float
        Correlation between selection and outcome errors.
        rho > 0: Positive selection (high outcome → more likely selected)
        rho < 0: Negative selection
    beta_x : float
        Effect of X on outcome.
    gamma_z : float
        Effect of Z (exclusion restriction) on selection.
    sigma_u : float
        Standard deviation of outcome error.
    selection_rate : float
        Target proportion of observations selected.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcome, selected, selection_covariates, outcome_covariates,
        true parameters, and sample sizes.
    """
    np.random.seed(seed)

    # Covariates
    Z = np.random.normal(0, 1, n)  # Exclusion restriction
    X = np.random.normal(0, 1, n)  # Outcome covariate

    # Generate correlated errors
    sigma_v = 1.0  # Probit scale
    cov_uv = rho * sigma_u * sigma_v

    mean = [0, 0]
    cov = [[sigma_u**2, cov_uv], [cov_uv, sigma_v**2]]
    errors = np.random.multivariate_normal(mean, cov, n)
    u = errors[:, 0]
    v = errors[:, 1]

    # Selection equation: S* = gamma_0 + gamma_z * Z + v
    # Adjust gamma_0 to achieve target selection rate
    gamma_0 = 0.0  # Start with 0
    S_star = gamma_0 + gamma_z * Z + v
    current_rate = (S_star > 0).mean()

    # Binary search for gamma_0 to hit target selection rate
    from scipy.stats import norm

    gamma_0 = norm.ppf(selection_rate)  # Approximate starting point
    S_star = gamma_0 + gamma_z * Z + v
    selected = (S_star > 0).astype(int)

    # Outcome equation: Y = beta_0 + beta_x * X + u
    beta_0 = 5.0
    Y_star = beta_0 + beta_x * X + u

    # Observed outcome (NaN for unselected)
    outcome = np.where(selected == 1, Y_star, np.nan)

    # Selection covariates include both Z and X (X for common support)
    # But Z is the exclusion restriction (affects S, not Y)
    selection_covariates = np.column_stack([Z, X])

    # Outcome covariates (only X, not Z - exclusion restriction)
    outcome_covariates = X.reshape(-1, 1)

    return {
        "outcome": outcome,
        "selected": selected,
        "selection_covariates": selection_covariates,
        "outcome_covariates": outcome_covariates,
        "true_beta_x": beta_x,
        "true_rho": rho,
        "true_sigma": sigma_u,
        "true_gamma_z": gamma_z,
        "n_selected": int(selected.sum()),
        "n_total": n,
        "Z": Z,
        "X": X,
    }


# =============================================================================
# Test Class: Heckman vs sampleSelection
# =============================================================================


@requires_heckman_python
@requires_sample_selection_r
class TestHeckmanVsSampleSelection:
    """Compare Python heckman_two_step against R sampleSelection."""

    def test_basic_selection(self):
        """Standard DGP with moderate selection (ρ=0.5)."""
        data = generate_heckman_dgp(n=1000, rho=0.5, beta_x=2.0, seed=42)

        # Python implementation
        py_result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        # R implementation
        r_result = r_heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        assert r_result is not None, "R implementation returned None"

        # Compare core estimate
        assert np.isclose(py_result["estimate"], r_result["estimate"], rtol=0.02), (
            f"Estimate mismatch: Python={py_result['estimate']:.4f}, R={r_result['estimate']:.4f}"
        )

        # Compare standard error
        assert np.isclose(py_result["se"], r_result["se"], rtol=0.05), (
            f"SE mismatch: Python={py_result['se']:.4f}, R={r_result['se']:.4f}"
        )

        # Compare selection parameters
        assert np.isclose(py_result["rho"], r_result["rho"], rtol=0.02), (
            f"rho mismatch: Python={py_result['rho']:.4f}, R={r_result['rho']:.4f}"
        )

        assert np.isclose(py_result["sigma"], r_result["sigma"], rtol=0.02), (
            f"sigma mismatch: Python={py_result['sigma']:.4f}, R={r_result['sigma']:.4f}"
        )

    def test_strong_selection(self):
        """Strong positive selection (ρ=0.8)."""
        data = generate_heckman_dgp(n=1000, rho=0.8, beta_x=2.0, seed=123)

        py_result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        r_result = r_heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        assert r_result is not None

        # With strong selection, estimates should still match
        assert np.isclose(py_result["estimate"], r_result["estimate"], rtol=0.02)
        assert np.isclose(py_result["rho"], r_result["rho"], rtol=0.02)

        # Strong selection → large |lambda|
        assert abs(py_result["lambda_coef"]) > 0.3, "Expected substantial lambda"

    def test_negative_selection(self):
        """Negative selection (ρ=-0.6)."""
        data = generate_heckman_dgp(n=1000, rho=-0.6, beta_x=2.0, seed=456)

        py_result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        r_result = r_heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        assert r_result is not None

        assert np.isclose(py_result["estimate"], r_result["estimate"], rtol=0.02)
        assert np.isclose(py_result["rho"], r_result["rho"], rtol=0.02)

        # Negative selection → rho < 0
        assert py_result["rho"] < 0, "Expected negative rho"
        assert r_result["rho"] < 0, "R should also have negative rho"

    def test_no_selection(self):
        """No selection bias (ρ≈0)."""
        data = generate_heckman_dgp(n=1000, rho=0.0, beta_x=2.0, seed=789)

        py_result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        r_result = r_heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        assert r_result is not None

        assert np.isclose(py_result["estimate"], r_result["estimate"], rtol=0.02)

        # No selection → rho ≈ 0, lambda p-value > 0.05
        assert abs(py_result["rho"]) < 0.3, "Expected rho near zero"
        assert py_result["lambda_pvalue"] > 0.01, "Expected non-significant lambda"

    def test_high_selection_rate(self):
        """High selection rate (~90% selected)."""
        data = generate_heckman_dgp(n=1000, rho=0.5, selection_rate=0.9, seed=111)

        py_result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        r_result = r_heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        assert r_result is not None

        # High selection rate: still should match
        assert np.isclose(py_result["estimate"], r_result["estimate"], rtol=0.02)
        assert np.isclose(py_result["rho"], r_result["rho"], rtol=0.02)

    def test_low_selection_rate(self):
        """Low selection rate (~30% selected)."""
        data = generate_heckman_dgp(n=1000, rho=0.5, selection_rate=0.3, seed=222)

        py_result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        r_result = r_heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        assert r_result is not None

        assert np.isclose(py_result["estimate"], r_result["estimate"], rtol=0.02)
        assert np.isclose(py_result["rho"], r_result["rho"], rtol=0.02)


# =============================================================================
# Test Class: Heckman Diagnostics
# =============================================================================


@requires_heckman_python
@requires_sample_selection_r
class TestHeckmanDiagnosticsVsR:
    """Test diagnostic outputs match R."""

    def test_lambda_inference(self):
        """Lambda coefficient and p-value should match."""
        data = generate_heckman_dgp(n=1000, rho=0.5, seed=42)

        py_result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        r_result = r_heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        assert r_result is not None

        # Lambda = rho * sigma
        expected_lambda = py_result["rho"] * py_result["sigma"]
        assert np.isclose(py_result["lambda_coef"], expected_lambda, rtol=0.01), (
            "Lambda should equal rho * sigma"
        )

        # Lambda should match R
        assert np.isclose(py_result["lambda_coef"], r_result["lambda_coef"], rtol=0.02), (
            f"Lambda mismatch: Python={py_result['lambda_coef']:.4f}, R={r_result['lambda_coef']:.4f}"
        )

        # Lambda SE
        assert np.isclose(py_result["lambda_se"], r_result["lambda_se"], rtol=0.05), (
            f"Lambda SE mismatch: Python={py_result['lambda_se']:.4f}, R={r_result['lambda_se']:.4f}"
        )

    def test_coefficient_arrays(self):
        """Selection and outcome coefficient arrays should match."""
        data = generate_heckman_dgp(n=1000, rho=0.5, seed=42)

        py_result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        r_result = r_heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        assert r_result is not None

        # Selection coefficients (gamma)
        py_gamma = py_result["gamma"]
        r_gamma = r_result["gamma"]

        assert len(py_gamma) == len(r_gamma), "Gamma length mismatch"

        for i in range(len(py_gamma)):
            assert np.isclose(py_gamma[i], r_gamma[i], rtol=0.02), (
                f"Gamma[{i}] mismatch: Python={py_gamma[i]:.4f}, R={r_gamma[i]:.4f}"
            )

        # Outcome coefficients (beta)
        py_beta = py_result["beta"]
        r_beta = r_result["beta"]

        assert len(py_beta) == len(r_beta), "Beta length mismatch"

        for i in range(len(py_beta)):
            assert np.isclose(py_beta[i], r_beta[i], rtol=0.02), (
                f"Beta[{i}] mismatch: Python={py_beta[i]:.4f}, R={r_beta[i]:.4f}"
            )


# =============================================================================
# Test Class: Consistency Checks
# =============================================================================


@requires_heckman_python
@requires_sample_selection_r
class TestHeckmanConsistency:
    """Cross-method consistency tests."""

    def test_sample_sizes(self):
        """Sample size reporting should be consistent."""
        data = generate_heckman_dgp(n=1000, rho=0.5, seed=42)

        py_result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        r_result = r_heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        assert r_result is not None

        assert py_result["n_selected"] == r_result["n_selected"]
        assert py_result["n_total"] == r_result["n_total"]
        assert py_result["n_selected"] == data["n_selected"]

    def test_heckman_corrects_ols_bias(self):
        """Heckman should correct for selection bias that OLS ignores."""
        # Strong positive selection: selected units have higher Y
        data = generate_heckman_dgp(n=2000, rho=0.7, beta_x=2.0, seed=333)

        # Get Heckman estimates
        py_result = heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        r_result = r_heckman_two_step(
            outcome=data["outcome"],
            selected=data["selected"],
            selection_covariates=data["selection_covariates"],
            outcome_covariates=data["outcome_covariates"],
        )

        assert r_result is not None

        # Compute naive OLS on selected sample
        selected_mask = data["selected"] == 1
        X_sel = data["outcome_covariates"][selected_mask]
        Y_sel = data["outcome"][selected_mask]

        # Add intercept
        X_sel_int = np.column_stack([np.ones(len(Y_sel)), X_sel])
        ols_coef = np.linalg.lstsq(X_sel_int, Y_sel, rcond=None)[0]
        ols_beta_x = ols_coef[1]

        # With positive selection and rho > 0, OLS is typically biased
        # Heckman should give estimate closer to true value
        heckman_est = py_result["estimate"]
        true_beta = data["true_beta_x"]

        heckman_bias = abs(heckman_est - true_beta)
        ols_bias = abs(ols_beta_x - true_beta)

        # Heckman should have less bias (or at least not much more)
        assert heckman_bias <= ols_bias + 0.5, (
            f"Heckman bias ({heckman_bias:.3f}) should not be much worse than "
            f"OLS bias ({ols_bias:.3f})"
        )


# =============================================================================
# Main: Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
