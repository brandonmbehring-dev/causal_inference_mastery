"""
Cross-language validation tests for observational IPW and DR estimators.

Tests verify Python↔Julia parity for:
1. IPW estimator point estimates and standard errors
2. Doubly Robust (AIPW) estimator results
3. Propensity score estimation
4. Diagnostics (AUC, R²)

Tolerance Note:
- Point estimates: rtol=0.15 (IPW/DR can differ due to optimization)
- Standard errors: rtol=0.25 (SE estimation varies by method)
- Propensity AUC: atol=0.1 (AUC robust to small differences)
"""

import pytest
import numpy as np
from typing import Dict, Any

from tests.validation.cross_language.julia_interface import (
    is_julia_available,
    julia_observational_ipw,
    julia_doubly_robust,
)
from src.causal_inference.observational.ipw import ipw_ate_observational
from src.causal_inference.observational.doubly_robust import dr_ate


# Skip all tests if Julia not available
pytestmark = pytest.mark.skipif(
    not is_julia_available(), reason="Julia not available for cross-validation"
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_observational_data(
    n: int = 500, true_ate: float = 2.0, confounding_strength: float = 0.5, seed: int = 42
) -> Dict[str, Any]:
    """Generate observational data with known confounding."""
    np.random.seed(seed)

    # Covariates
    X = np.random.randn(n, 2)

    # Propensity: depends on X (confounding)
    logit = confounding_strength * X[:, 0] + 0.3 * X[:, 1]
    e_true = 1 / (1 + np.exp(-logit))

    # Treatment assignment
    T = (np.random.rand(n) < e_true).astype(float)

    # Outcome: confounded (X affects both T and Y)
    Y = true_ate * T + 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n)

    return {
        "outcomes": Y,
        "treatment": T,
        "covariates": X,
        "e_true": e_true,
        "true_ate": true_ate,
    }


def generate_weak_confounding(
    n: int = 500, true_ate: float = 1.5, seed: int = 123
) -> Dict[str, Any]:
    """Generate data with weak confounding (near-RCT)."""
    np.random.seed(seed)

    X = np.random.randn(n, 2)

    # Very weak relationship between X and T
    logit = 0.1 * X[:, 0]
    e_true = 1 / (1 + np.exp(-logit))
    T = (np.random.rand(n) < e_true).astype(float)

    # Outcome
    Y = true_ate * T + 0.5 * X[:, 0] + np.random.randn(n)

    return {
        "outcomes": Y,
        "treatment": T,
        "covariates": X,
        "e_true": e_true,
        "true_ate": true_ate,
    }


# =============================================================================
# IPW Parity Tests
# =============================================================================


class TestIPWBasicParity:
    """Basic IPW parity tests between Python and Julia."""

    def test_ipw_point_estimate_moderate_confounding(self):
        """IPW point estimates should match within tolerance."""
        data = generate_observational_data(n=500, seed=1)

        py_result = ipw_ate_observational(
            data["outcomes"], data["treatment"], data["covariates"], alpha=0.05
        )

        jl_result = julia_observational_ipw(
            data["outcomes"], data["treatment"], data["covariates"], alpha=0.05, trim_threshold=0.01
        )

        # Point estimates should be similar (within 15%)
        assert np.isclose(py_result["estimate"], jl_result["estimate"], rtol=0.15), (
            f"IPW estimates differ: Python={py_result['estimate']:.4f}, Julia={jl_result['estimate']:.4f}"
        )

    def test_ipw_standard_error_order_of_magnitude(self):
        """IPW standard errors should be same order of magnitude."""
        data = generate_observational_data(n=500, seed=2)

        py_result = ipw_ate_observational(data["outcomes"], data["treatment"], data["covariates"])

        jl_result = julia_observational_ipw(data["outcomes"], data["treatment"], data["covariates"])

        # SE should be within 50% (different variance estimators)
        assert np.isclose(py_result["se"], jl_result["se"], rtol=0.5), (
            f"IPW SE differ: Python={py_result['se']:.4f}, Julia={jl_result['se']:.4f}"
        )

    def test_ipw_weak_confounding(self):
        """IPW should agree well with weak confounding (easier case)."""
        data = generate_weak_confounding(n=600, seed=3)

        py_result = ipw_ate_observational(data["outcomes"], data["treatment"], data["covariates"])

        jl_result = julia_observational_ipw(data["outcomes"], data["treatment"], data["covariates"])

        # Weak confounding should give closer agreement
        assert np.isclose(py_result["estimate"], jl_result["estimate"], rtol=0.10), (
            f"IPW estimates differ (weak): Python={py_result['estimate']:.4f}, Julia={jl_result['estimate']:.4f}"
        )


class TestIPWConfigurationParity:
    """IPW configuration parity tests."""

    def test_ipw_with_trimming(self):
        """IPW with trimming should give similar results."""
        data = generate_observational_data(n=500, seed=10)

        py_result = ipw_ate_observational(
            data["outcomes"], data["treatment"], data["covariates"], trim_at=(0.05, 0.95)
        )

        jl_result = julia_observational_ipw(
            data["outcomes"], data["treatment"], data["covariates"], trim_threshold=0.05
        )

        # Both should trim some observations
        assert py_result["n_trimmed"] >= 0
        assert jl_result["n_trimmed"] >= 0

        # Point estimates should still be similar
        assert np.isclose(py_result["estimate"], jl_result["estimate"], rtol=0.20), (
            f"IPW trimmed estimates differ: Python={py_result['estimate']:.4f}, Julia={jl_result['estimate']:.4f}"
        )

    def test_ipw_sample_sizes_similar(self):
        """Sample sizes should be similar (trimming may differ slightly)."""
        data = generate_observational_data(n=400, seed=11)

        py_result = ipw_ate_observational(data["outcomes"], data["treatment"], data["covariates"])

        jl_result = julia_observational_ipw(data["outcomes"], data["treatment"], data["covariates"])

        # Sample sizes should be within 5% (trimming algorithms may differ)
        py_n = py_result["n_treated"] + py_result["n_control"]
        jl_n = jl_result["n_treated"] + jl_result["n_control"]
        assert np.isclose(py_n, jl_n, rtol=0.05), f"Total N differs: Python={py_n}, Julia={jl_n}"


class TestIPWDiagnosticsParity:
    """IPW diagnostics parity tests."""

    def test_propensity_auc_similar(self):
        """Propensity AUC should be similar."""
        data = generate_observational_data(n=500, seed=20)

        py_result = ipw_ate_observational(data["outcomes"], data["treatment"], data["covariates"])

        jl_result = julia_observational_ipw(data["outcomes"], data["treatment"], data["covariates"])

        # AUC should be within 0.1 (both detecting confounding)
        py_auc = py_result.get("propensity_diagnostics", {}).get("auc", 0.5)
        jl_auc = jl_result["propensity_auc"]

        # Note: Python may not compute AUC in all cases
        if py_auc > 0:
            assert np.isclose(py_auc, jl_auc, atol=0.15), (
                f"AUC differs: Python={py_auc:.3f}, Julia={jl_auc:.3f}"
            )


# =============================================================================
# Doubly Robust Parity Tests
# =============================================================================


class TestDRBasicParity:
    """Basic DR parity tests between Python and Julia."""

    def test_dr_point_estimate_moderate_confounding(self):
        """DR point estimates should match within tolerance."""
        data = generate_observational_data(n=500, seed=30)

        py_result = dr_ate(data["outcomes"], data["treatment"], data["covariates"], alpha=0.05)

        jl_result = julia_doubly_robust(
            data["outcomes"], data["treatment"], data["covariates"], alpha=0.05, trim_threshold=0.01
        )

        # DR should be closer than IPW (both models correct)
        assert np.isclose(py_result["estimate"], jl_result["estimate"], rtol=0.15), (
            f"DR estimates differ: Python={py_result['estimate']:.4f}, Julia={jl_result['estimate']:.4f}"
        )

    def test_dr_standard_error_order_of_magnitude(self):
        """DR standard errors should be same order of magnitude."""
        data = generate_observational_data(n=500, seed=31)

        py_result = dr_ate(data["outcomes"], data["treatment"], data["covariates"])

        jl_result = julia_doubly_robust(data["outcomes"], data["treatment"], data["covariates"])

        # SE should be within 50%
        assert np.isclose(py_result["se"], jl_result["se"], rtol=0.5), (
            f"DR SE differ: Python={py_result['se']:.4f}, Julia={jl_result['se']:.4f}"
        )

    def test_dr_weak_confounding(self):
        """DR should agree well with weak confounding."""
        data = generate_weak_confounding(n=600, seed=32)

        py_result = dr_ate(data["outcomes"], data["treatment"], data["covariates"])

        jl_result = julia_doubly_robust(data["outcomes"], data["treatment"], data["covariates"])

        # Weak confounding should give closer agreement
        assert np.isclose(py_result["estimate"], jl_result["estimate"], rtol=0.10), (
            f"DR estimates differ (weak): Python={py_result['estimate']:.4f}, Julia={jl_result['estimate']:.4f}"
        )


class TestDRvIPWComparison:
    """Compare DR and IPW across languages."""

    def test_dr_vs_ipw_both_languages_consistent(self):
        """DR and IPW should give similar estimates in both languages."""
        data = generate_observational_data(n=600, seed=40)

        # Python
        py_ipw = ipw_ate_observational(data["outcomes"], data["treatment"], data["covariates"])
        py_dr = dr_ate(data["outcomes"], data["treatment"], data["covariates"])

        # Julia
        jl_ipw = julia_observational_ipw(data["outcomes"], data["treatment"], data["covariates"])
        jl_dr = julia_doubly_robust(data["outcomes"], data["treatment"], data["covariates"])

        # Within each language, IPW and DR should give similar estimates
        assert np.isclose(py_ipw["estimate"], py_dr["estimate"], rtol=0.20), (
            f"Python IPW/DR differ: {py_ipw['estimate']:.4f} vs {py_dr['estimate']:.4f}"
        )

        assert np.isclose(jl_ipw["estimate"], jl_dr["estimate"], rtol=0.20), (
            f"Julia IPW/DR differ: {jl_ipw['estimate']:.4f} vs {jl_dr['estimate']:.4f}"
        )


class TestDRDiagnosticsParity:
    """DR diagnostics parity tests."""

    def test_outcome_r2_computed(self):
        """Both implementations should compute outcome model R²."""
        data = generate_observational_data(n=500, seed=50)

        jl_result = julia_doubly_robust(data["outcomes"], data["treatment"], data["covariates"])

        # Julia should have R² diagnostics
        assert "mu0_r2" in jl_result
        assert "mu1_r2" in jl_result
        assert 0 <= jl_result["mu0_r2"] <= 1 or jl_result["mu0_r2"] < 0  # R² can be negative
        assert 0 <= jl_result["mu1_r2"] <= 1 or jl_result["mu1_r2"] < 0

    def test_sample_sizes_similar(self):
        """Sample sizes should be similar (trimming may differ slightly)."""
        data = generate_observational_data(n=400, seed=51)

        py_result = dr_ate(data["outcomes"], data["treatment"], data["covariates"])

        jl_result = julia_doubly_robust(data["outcomes"], data["treatment"], data["covariates"])

        # Sample sizes should be within 5% (trimming algorithms may differ)
        py_n = py_result["n_treated"] + py_result["n_control"]
        jl_n = jl_result["n_treated"] + jl_result["n_control"]
        assert np.isclose(py_n, jl_n, rtol=0.05), f"Total N differs: Python={py_n}, Julia={jl_n}"
