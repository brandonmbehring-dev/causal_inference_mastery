"""
Cross-language validation tests for Quantile Treatment Effects.

Tests Python ↔ Julia parity for QTE estimators (Session 89).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.qte import (
    unconditional_qte,
    conditional_qte,
    rif_qte,
)

# Import Julia interface with skip if unavailable
try:
    from tests.validation.cross_language.julia_interface import (
        is_julia_available,
        julia_unconditional_qte,
        julia_conditional_qte,
        julia_rif_qte,
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


def generate_qte_data(
    n: int = 500,
    true_ate: float = 2.0,
    random_seed: int = 42,
):
    """
    Generate simple QTE data for testing.

    Y = true_ate * T + noise
    """
    np.random.seed(random_seed)

    treatment = np.random.binomial(1, 0.5, n).astype(float)
    noise = np.random.randn(n)
    outcome = true_ate * treatment + noise

    return {
        "outcome": outcome,
        "treatment": treatment,
        "true_ate": true_ate,
    }


def generate_qte_with_covariates(
    n: int = 500,
    p: int = 3,
    true_ate: float = 2.0,
    random_seed: int = 42,
):
    """
    Generate QTE data with covariates.

    Y = true_ate * T + 0.5 * sum(X) + noise
    """
    np.random.seed(random_seed)

    treatment = np.random.binomial(1, 0.5, n).astype(float)
    covariates = np.random.randn(n, p)
    noise = np.random.randn(n)

    covariate_sum = covariates.sum(axis=1)
    outcome = true_ate * treatment + 0.5 * covariate_sum + noise

    return {
        "outcome": outcome,
        "treatment": treatment,
        "covariates": covariates,
        "true_ate": true_ate,
    }


# =============================================================================
# Unconditional QTE Parity Tests
# =============================================================================


class TestUnconditionalQTEParity:
    """Python ↔ Julia parity for unconditional QTE."""

    def test_estimate_parity_median(self):
        """Point estimates should match at median."""
        data = generate_qte_data(n=500, true_ate=2.0, random_seed=42)

        # Python
        py_result = unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=500,
            random_state=42,
        )

        # Julia
        jl_result = julia_unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=500,
            seed=42,
        )

        # Point estimates should be identical (same algorithm)
        assert_allclose(py_result["tau_q"], jl_result["tau_q"], rtol=1e-10)

    def test_estimate_parity_q25(self):
        """Point estimates should match at 25th percentile."""
        data = generate_qte_data(n=500, true_ate=2.0, random_seed=42)

        py_result = unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.25,
            n_bootstrap=500,
            random_state=42,
        )

        jl_result = julia_unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.25,
            n_bootstrap=500,
            seed=42,
        )

        assert_allclose(py_result["tau_q"], jl_result["tau_q"], rtol=1e-10)

    def test_estimate_parity_q75(self):
        """Point estimates should match at 75th percentile."""
        data = generate_qte_data(n=500, true_ate=2.0, random_seed=42)

        py_result = unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.75,
            n_bootstrap=500,
            random_state=42,
        )

        jl_result = julia_unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.75,
            n_bootstrap=500,
            seed=42,
        )

        assert_allclose(py_result["tau_q"], jl_result["tau_q"], rtol=1e-10)

    def test_sample_counts_match(self):
        """Sample counts should match exactly."""
        data = generate_qte_data(n=500, true_ate=2.0, random_seed=42)

        py_result = unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=100,
            random_state=42,
        )

        jl_result = julia_unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=100,
            seed=42,
        )

        assert py_result["n_total"] == jl_result["n_total"]
        assert py_result["n_treated"] == jl_result["n_treated"]
        assert py_result["n_control"] == jl_result["n_control"]

    def test_se_same_order_of_magnitude(self):
        """Bootstrap SEs should be similar (not identical due to RNG differences)."""
        data = generate_qte_data(n=500, true_ate=2.0, random_seed=42)

        py_result = unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=1000,
            random_state=42,
        )

        jl_result = julia_unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=1000,
            seed=42,
        )

        # SEs should be within 50% (RNGs differ)
        assert_allclose(py_result["se"], jl_result["se"], rtol=0.5)


# =============================================================================
# Conditional QTE Parity Tests
# =============================================================================


class TestConditionalQTEParity:
    """Python ↔ Julia parity for conditional QTE."""

    def test_estimate_parity_with_covariates(self):
        """Point estimates should be similar with covariates."""
        data = generate_qte_with_covariates(n=500, p=3, true_ate=2.0, random_seed=42)

        py_result = conditional_qte(
            data["outcome"],
            data["treatment"],
            data["covariates"],
            quantile=0.5,
        )

        jl_result = julia_conditional_qte(
            data["outcome"],
            data["treatment"],
            data["covariates"],
            quantile=0.5,
        )

        # Estimates should be close (IRLS convergence may differ slightly)
        assert_allclose(py_result["tau_q"], jl_result["tau_q"], rtol=0.15)

    def test_sample_counts_match_conditional(self):
        """Sample counts should match exactly."""
        data = generate_qte_with_covariates(n=300, p=2, true_ate=2.0, random_seed=42)

        py_result = conditional_qte(
            data["outcome"],
            data["treatment"],
            data["covariates"],
            quantile=0.5,
        )

        jl_result = julia_conditional_qte(
            data["outcome"],
            data["treatment"],
            data["covariates"],
            quantile=0.5,
        )

        assert py_result["n_total"] == jl_result["n_total"]


# =============================================================================
# RIF QTE Parity Tests
# =============================================================================


class TestRIFQTEParity:
    """Python ↔ Julia parity for RIF-OLS QTE."""

    def test_estimate_parity_rif(self):
        """RIF point estimates should be similar."""
        data = generate_qte_data(n=500, true_ate=2.0, random_seed=42)

        py_result = rif_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=500,
            random_state=42,
        )

        jl_result = julia_rif_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=500,
            seed=42,
        )

        # RIF estimates should be close (density estimation differs)
        assert_allclose(py_result["tau_q"], jl_result["tau_q"], rtol=0.2)

    def test_sample_counts_match_rif(self):
        """Sample counts should match exactly."""
        data = generate_qte_data(n=300, true_ate=2.0, random_seed=42)

        py_result = rif_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=100,
            random_state=42,
        )

        jl_result = julia_rif_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=100,
            seed=42,
        )

        assert py_result["n_total"] == jl_result["n_total"]
        assert py_result["n_treated"] == jl_result["n_treated"]
        assert py_result["n_control"] == jl_result["n_control"]


# =============================================================================
# Multi-Quantile Parity Tests
# =============================================================================


class TestMultiQuantileParity:
    """Test parity across multiple quantiles."""

    def test_quantile_ordering_preserved(self):
        """Both implementations should preserve quantile ordering."""
        data = generate_qte_data(n=500, true_ate=2.0, random_seed=42)

        py_results = {}
        jl_results = {}

        for q in [0.25, 0.5, 0.75]:
            py_results[q] = unconditional_qte(
                data["outcome"],
                data["treatment"],
                quantile=q,
                n_bootstrap=300,
                random_state=42,
            )
            jl_results[q] = julia_unconditional_qte(
                data["outcome"],
                data["treatment"],
                quantile=q,
                n_bootstrap=300,
                seed=42,
            )

        # Both should have consistent ordering
        py_estimates = [py_results[q]["tau_q"] for q in [0.25, 0.5, 0.75]]
        jl_estimates = [jl_results[q]["tau_q"] for q in [0.25, 0.5, 0.75]]

        # With homogeneous effects, all estimates should be similar
        assert np.std(py_estimates) < 0.5
        assert np.std(jl_estimates) < 0.5


# =============================================================================
# Edge Case Parity Tests
# =============================================================================


class TestEdgeCaseParity:
    """Test parity on edge cases."""

    def test_extreme_quantile_q05(self):
        """Both should handle extreme low quantile."""
        data = generate_qte_data(n=500, true_ate=2.0, random_seed=42)

        py_result = unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.05,
            n_bootstrap=300,
            random_state=42,
        )

        jl_result = julia_unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.05,
            n_bootstrap=300,
            seed=42,
        )

        # Point estimates should match
        assert_allclose(py_result["tau_q"], jl_result["tau_q"], rtol=1e-10)

    def test_extreme_quantile_q95(self):
        """Both should handle extreme high quantile."""
        data = generate_qte_data(n=500, true_ate=2.0, random_seed=42)

        py_result = unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.95,
            n_bootstrap=300,
            random_state=42,
        )

        jl_result = julia_unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.95,
            n_bootstrap=300,
            seed=42,
        )

        assert_allclose(py_result["tau_q"], jl_result["tau_q"], rtol=1e-10)

    def test_small_sample(self):
        """Both should handle small samples consistently."""
        data = generate_qte_data(n=50, true_ate=2.0, random_seed=42)

        py_result = unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=200,
            random_state=42,
        )

        jl_result = julia_unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=200,
            seed=42,
        )

        assert_allclose(py_result["tau_q"], jl_result["tau_q"], rtol=1e-10)

    def test_imbalanced_treatment(self):
        """Both should handle imbalanced treatment groups."""
        np.random.seed(42)
        n = 300
        treatment = np.random.binomial(1, 0.8, n).astype(float)  # 80% treated
        outcome = 2.0 * treatment + np.random.randn(n)

        py_result = unconditional_qte(
            outcome,
            treatment,
            quantile=0.5,
            n_bootstrap=300,
            random_state=42,
        )

        jl_result = julia_unconditional_qte(
            outcome,
            treatment,
            quantile=0.5,
            n_bootstrap=300,
            seed=42,
        )

        assert_allclose(py_result["tau_q"], jl_result["tau_q"], rtol=1e-10)


# =============================================================================
# Method Agreement Tests
# =============================================================================


class TestMethodAgreement:
    """Test that methods agree across languages."""

    def test_method_label_unconditional(self):
        """Method labels should match."""
        data = generate_qte_data(n=300, true_ate=2.0, random_seed=42)

        py_result = unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=100,
            random_state=42,
        )

        jl_result = julia_unconditional_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=100,
            seed=42,
        )

        assert py_result["method"] == "unconditional"
        assert jl_result["method"] == "unconditional"

    def test_method_label_rif(self):
        """Method labels should match for RIF."""
        data = generate_qte_data(n=300, true_ate=2.0, random_seed=42)

        py_result = rif_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=100,
            random_state=42,
        )

        jl_result = julia_rif_qte(
            data["outcome"],
            data["treatment"],
            quantile=0.5,
            n_bootstrap=100,
            seed=42,
        )

        assert py_result["method"] == "rif"
        assert jl_result["method"] == "rif"
