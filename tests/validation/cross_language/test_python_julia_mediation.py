"""
Cross-language validation tests for Mediation Analysis.

Session 95: Python ↔ Julia parity for Baron-Kenny and mediation effects.
"""

import pytest
import numpy as np
from typing import Tuple

from .julia_interface import (
    is_julia_available,
    julia_baron_kenny,
    julia_mediation_analysis,
)

# Import Python implementations
try:
    from src.causal_inference.mediation.estimators import (
        baron_kenny,
        mediation_analysis,
        controlled_direct_effect,
    )

    PYTHON_MEDIATION_AVAILABLE = True
except ImportError:
    PYTHON_MEDIATION_AVAILABLE = False


# Skip all tests if Julia not available
pytestmark = pytest.mark.skipif(
    not is_julia_available(),
    reason="Julia not available for cross-validation",
)


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================


def generate_mediation_data(
    n: int = 500,
    alpha_1: float = 0.6,  # T -> M effect
    beta_1: float = 0.5,  # Direct effect T -> Y
    beta_2: float = 0.8,  # M -> Y effect
    noise_m: float = 0.5,
    noise_y: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generate data for mediation analysis."""
    rng = np.random.default_rng(seed)

    treatment = (rng.random(n) < 0.5).astype(float)
    mediator = 0.5 + alpha_1 * treatment + noise_m * rng.standard_normal(n)
    outcome = 1.0 + beta_1 * treatment + beta_2 * mediator + noise_y * rng.standard_normal(n)

    true_effects = {
        "indirect": alpha_1 * beta_2,
        "direct": beta_1,
        "total": beta_1 + alpha_1 * beta_2,
    }

    return outcome, treatment, mediator, true_effects


def generate_full_mediation_data(
    n: int = 500,
    alpha_1: float = 1.0,
    beta_2: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data with full mediation (no direct effect)."""
    rng = np.random.default_rng(seed)

    treatment = (rng.random(n) < 0.5).astype(float)
    mediator = 0.5 + alpha_1 * treatment + 0.3 * rng.standard_normal(n)
    # No direct effect (beta_1 = 0)
    outcome = 1.0 + beta_2 * mediator + 0.3 * rng.standard_normal(n)

    return outcome, treatment, mediator


# =============================================================================
# BARON-KENNY PARITY TESTS
# =============================================================================


class TestBaronKennyParity:
    """Cross-language tests for Baron-Kenny mediation."""

    @pytest.mark.skipif(not PYTHON_MEDIATION_AVAILABLE, reason="Python mediation not available")
    def test_path_coefficients_match(self):
        """Test that Python and Julia path coefficients match."""
        outcome, treatment, mediator, _ = generate_mediation_data(n=500, seed=42)

        py_result = baron_kenny(outcome, treatment, mediator)
        jl_result = julia_baron_kenny(outcome, treatment, mediator)

        # alpha_1 should match
        assert np.isclose(py_result["alpha_1"], jl_result["alpha_1"], rtol=0.01), (
            f"alpha_1 mismatch: Python={py_result['alpha_1']:.6f}, Julia={jl_result['alpha_1']:.6f}"
        )

        # beta_1 should match
        assert np.isclose(py_result["beta_1"], jl_result["beta_1"], rtol=0.01), (
            f"beta_1 mismatch: Python={py_result['beta_1']:.6f}, Julia={jl_result['beta_1']:.6f}"
        )

        # beta_2 should match
        assert np.isclose(py_result["beta_2"], jl_result["beta_2"], rtol=0.01), (
            f"beta_2 mismatch: Python={py_result['beta_2']:.6f}, Julia={jl_result['beta_2']:.6f}"
        )

    @pytest.mark.skipif(not PYTHON_MEDIATION_AVAILABLE, reason="Python mediation not available")
    def test_effects_match(self):
        """Test that Python and Julia effects match."""
        outcome, treatment, mediator, _ = generate_mediation_data(n=500, seed=42)

        py_result = baron_kenny(outcome, treatment, mediator)
        jl_result = julia_baron_kenny(outcome, treatment, mediator)

        # Indirect effect
        assert np.isclose(py_result["indirect_effect"], jl_result["indirect_effect"], rtol=0.01)

        # Direct effect
        assert np.isclose(py_result["direct_effect"], jl_result["direct_effect"], rtol=0.01)

        # Total effect
        assert np.isclose(py_result["total_effect"], jl_result["total_effect"], rtol=0.01)

    @pytest.mark.skipif(not PYTHON_MEDIATION_AVAILABLE, reason="Python mediation not available")
    def test_sobel_test_match(self):
        """Test that Sobel test statistics match."""
        outcome, treatment, mediator, _ = generate_mediation_data(n=500, seed=42)

        py_result = baron_kenny(outcome, treatment, mediator)
        jl_result = julia_baron_kenny(outcome, treatment, mediator)

        # Sobel Z should match
        assert np.isclose(py_result["sobel_z"], jl_result["sobel_z"], rtol=0.05)

        # Sobel p-value should match
        assert np.isclose(py_result["sobel_pvalue"], jl_result["sobel_pvalue"], rtol=0.1)

    def test_julia_recovers_true_effects(self):
        """Test that Julia recovers true effects."""
        outcome, treatment, mediator, true_effects = generate_mediation_data(
            n=5000, alpha_1=0.6, beta_1=0.5, beta_2=0.8, seed=42
        )

        jl_result = julia_baron_kenny(outcome, treatment, mediator)

        # Should recover effects approximately
        assert np.isclose(jl_result["alpha_1"], 0.6, atol=0.1)
        assert np.isclose(jl_result["beta_1"], 0.5, atol=0.1)
        assert np.isclose(jl_result["beta_2"], 0.8, atol=0.1)
        assert np.isclose(jl_result["indirect_effect"], true_effects["indirect"], atol=0.15)

    def test_total_effect_decomposition(self):
        """Test total = direct + indirect."""
        outcome, treatment, mediator, _ = generate_mediation_data(n=500, seed=42)

        jl_result = julia_baron_kenny(outcome, treatment, mediator)

        computed_total = jl_result["direct_effect"] + jl_result["indirect_effect"]
        assert np.isclose(jl_result["total_effect"], computed_total, rtol=1e-10)


# =============================================================================
# FULL MEDIATION ANALYSIS PARITY TESTS
# =============================================================================


class TestMediationAnalysisParity:
    """Cross-language tests for full mediation analysis with bootstrap."""

    @pytest.mark.skipif(not PYTHON_MEDIATION_AVAILABLE, reason="Python mediation not available")
    def test_effects_match(self):
        """Test that Python and Julia effects match (with bootstrap variability)."""
        outcome, treatment, mediator, _ = generate_mediation_data(n=500, seed=42)

        py_result = mediation_analysis(
            outcome,
            treatment,
            mediator,
            n_bootstrap=200,
            random_state=42,
        )
        jl_result = julia_mediation_analysis(
            outcome,
            treatment,
            mediator,
            n_bootstrap=200,
            seed=42,
        )

        # Point estimates should be close (but not exact due to bootstrap randomness)
        assert np.isclose(py_result["direct_effect"], jl_result["direct_effect"], rtol=0.1)
        assert np.isclose(py_result["indirect_effect"], jl_result["indirect_effect"], rtol=0.1)
        assert np.isclose(py_result["total_effect"], jl_result["total_effect"], rtol=0.1)

    def test_julia_cis_computed(self):
        """Test that Julia computes valid CIs."""
        outcome, treatment, mediator, _ = generate_mediation_data(n=500, seed=42)

        jl_result = julia_mediation_analysis(
            outcome,
            treatment,
            mediator,
            n_bootstrap=200,
            seed=42,
        )

        # CIs should contain point estimates
        assert jl_result["de_ci"][0] <= jl_result["direct_effect"] <= jl_result["de_ci"][1]
        assert jl_result["ie_ci"][0] <= jl_result["indirect_effect"] <= jl_result["ie_ci"][1]
        assert jl_result["te_ci"][0] <= jl_result["total_effect"] <= jl_result["te_ci"][1]

    def test_proportion_mediated(self):
        """Test proportion mediated calculation."""
        outcome, treatment, mediator, _ = generate_mediation_data(
            n=1000, alpha_1=0.6, beta_1=0.5, beta_2=0.8, seed=42
        )

        jl_result = julia_mediation_analysis(
            outcome,
            treatment,
            mediator,
            n_bootstrap=100,
            seed=42,
        )

        # Proportion = Indirect / Total
        expected_pm = jl_result["indirect_effect"] / jl_result["total_effect"]
        assert np.isclose(jl_result["proportion_mediated"], expected_pm, rtol=0.01)

        # For this DGP, ~50% mediated
        assert 0.3 < jl_result["proportion_mediated"] < 0.7

    def test_full_mediation_detection(self):
        """Test detection of full mediation."""
        outcome, treatment, mediator = generate_full_mediation_data(n=2000, seed=42)

        jl_result = julia_mediation_analysis(
            outcome,
            treatment,
            mediator,
            n_bootstrap=100,
            seed=42,
        )

        # Direct effect should be near zero
        assert abs(jl_result["direct_effect"]) < 0.2

        # Proportion mediated should be high
        assert jl_result["proportion_mediated"] > 0.8


# =============================================================================
# R-SQUARED AND DIAGNOSTICS
# =============================================================================


class TestMediationDiagnostics:
    """Tests for mediation diagnostics."""

    def test_r_squared_values(self):
        """Test R-squared values are computed correctly."""
        outcome, treatment, mediator, _ = generate_mediation_data(n=500, seed=42)

        jl_result = julia_baron_kenny(outcome, treatment, mediator)

        # R-squared should be between 0 and 1
        assert 0 <= jl_result["r2_mediator_model"] <= 1
        assert 0 <= jl_result["r2_outcome_model"] <= 1

    def test_sample_size(self):
        """Test sample size is correct."""
        n = 500
        outcome, treatment, mediator, _ = generate_mediation_data(n=n, seed=42)

        jl_result = julia_baron_kenny(outcome, treatment, mediator)

        assert jl_result["n_obs"] == n


# =============================================================================
# EDGE CASES
# =============================================================================


class TestMediationEdgeCases:
    """Edge case tests for mediation analysis."""

    def test_binary_mediator(self):
        """Test with binary mediator."""
        rng = np.random.default_rng(42)
        n = 500
        treatment = (rng.random(n) < 0.5).astype(float)
        # Binary mediator
        mediator = (rng.random(n) < (0.3 + 0.4 * treatment)).astype(float)
        outcome = 1.0 + 0.5 * treatment + 0.8 * mediator + 0.5 * rng.standard_normal(n)

        jl_result = julia_baron_kenny(outcome, treatment, mediator)

        assert "indirect_effect" in jl_result
        assert not np.isnan(jl_result["indirect_effect"])

    def test_continuous_treatment(self):
        """Test with continuous treatment."""
        rng = np.random.default_rng(42)
        n = 500
        treatment = rng.standard_normal(n)  # Continuous treatment
        mediator = 0.5 + 0.6 * treatment + 0.5 * rng.standard_normal(n)
        outcome = 1.0 + 0.5 * treatment + 0.8 * mediator + 0.5 * rng.standard_normal(n)

        jl_result = julia_baron_kenny(outcome, treatment, mediator)

        assert np.isclose(jl_result["alpha_1"], 0.6, atol=0.15)

    def test_large_sample(self):
        """Test with large sample."""
        outcome, treatment, mediator, _ = generate_mediation_data(n=10000, seed=42)

        jl_result = julia_baron_kenny(outcome, treatment, mediator)

        assert jl_result["n_obs"] == 10000
        # Should have smaller SEs with large sample
        assert jl_result["alpha_1_se"] < 0.05
        assert jl_result["beta_1_se"] < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
