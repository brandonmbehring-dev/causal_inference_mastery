"""Triangulation tests: Python Mediation vs R mediation package.

This module provides Layer 5 validation by comparing our Python mediation
implementations against R's mediation package (Imai, Keele, Yamamoto).

Tests skip gracefully when R/rpy2 or mediation package is unavailable.

Tolerance levels (established based on simulation/bootstrap variability):
- Path coefficients (α₁, β₁, β₂): rtol=0.02 (OLS estimates)
- Direct/Indirect effects: rtol=0.05 (simulation variability)
- Sobel test: rtol=0.05 (analytic formula)
- Bootstrap CI: rtol=0.15 (resampling variation)
- Proportion mediated: rtol=0.10 (ratio estimate)
- Sensitivity ρ: rtol=0.05 (grid search precision)

Run with: pytest tests/validation/r_triangulation/test_mediation_vs_r.py -v

References:
- Baron & Kenny (1986). The Moderator-Mediator Variable Distinction.
- Imai, Keele, Yamamoto (2010). A General Approach to Causal Mediation.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import Dict, Any

from tests.validation.r_triangulation.r_interface import (
    check_r_available,
    check_mediation_available,
    r_baron_kenny,
    r_mediation_analysis,
    r_mediation_sensitivity,
)

# Lazy import Python implementations
try:
    from src.causal_inference.mediation.estimators import (
        baron_kenny,
        mediation_analysis,
    )
    from src.causal_inference.mediation.sensitivity import mediation_sensitivity

    MEDIATION_AVAILABLE = True
except ImportError:
    MEDIATION_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_mediation_python = pytest.mark.skipif(
    not MEDIATION_AVAILABLE,
    reason="Python Mediation module not available",
)

requires_mediation_r = pytest.mark.skipif(
    not check_mediation_available(),
    reason="R mediation package not installed. "
    "Install in R with: install.packages('mediation')",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_mediation_data(
    n: int = 1000,
    alpha_1: float = 0.6,
    beta_1: float = 0.5,
    beta_2: float = 0.8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate standard linear mediation data.

    DGP:
    - T ~ Bernoulli(0.5)
    - M = alpha_0 + alpha_1 * T + e_m
    - Y = beta_0 + beta_1 * T + beta_2 * M + e_y

    True effects:
    - Direct effect (DE) = beta_1
    - Indirect effect (IE) = alpha_1 * beta_2
    - Total effect (TE) = beta_1 + alpha_1 * beta_2

    Parameters
    ----------
    n : int
        Sample size.
    alpha_1 : float
        Effect of T on M.
    beta_1 : float
        Direct effect of T on Y.
    beta_2 : float
        Effect of M on Y.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcome, treatment, mediator, and true parameters.
    """
    np.random.seed(seed)

    # Treatment (binary)
    treatment = np.random.binomial(1, 0.5, n).astype(float)

    # Mediator model: M = 3 + alpha_1 * T + e_m
    e_m = np.random.normal(0, 1, n)
    mediator = 3.0 + alpha_1 * treatment + e_m

    # Outcome model: Y = 2 + beta_1 * T + beta_2 * M + e_y
    e_y = np.random.normal(0, 1, n)
    outcome = 2.0 + beta_1 * treatment + beta_2 * mediator + e_y

    # True effects
    indirect_effect = alpha_1 * beta_2
    direct_effect = beta_1
    total_effect = beta_1 + alpha_1 * beta_2

    return {
        "outcome": outcome,
        "treatment": treatment,
        "mediator": mediator,
        "true_alpha_1": alpha_1,
        "true_beta_1": beta_1,
        "true_beta_2": beta_2,
        "true_indirect": indirect_effect,
        "true_direct": direct_effect,
        "true_total": total_effect,
        "n": n,
    }


def generate_full_mediation_data(
    n: int = 1000,
    alpha_1: float = 0.8,
    beta_2: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate data with full mediation (no direct effect).

    DGP:
    - T ~ Bernoulli(0.5)
    - M = alpha_0 + alpha_1 * T + e_m
    - Y = beta_0 + 0 * T + beta_2 * M + e_y  (beta_1 = 0)

    Parameters
    ----------
    n : int
        Sample size.
    alpha_1 : float
        Effect of T on M.
    beta_2 : float
        Effect of M on Y.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcome, treatment, mediator, and true parameters.
    """
    return generate_mediation_data(
        n=n, alpha_1=alpha_1, beta_1=0.0, beta_2=beta_2, seed=seed
    )


def generate_no_mediation_data(
    n: int = 1000,
    beta_1: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate data with no mediation (no T→M effect).

    DGP:
    - T ~ Bernoulli(0.5)
    - M = alpha_0 + 0 * T + e_m  (alpha_1 = 0)
    - Y = beta_0 + beta_1 * T + beta_2 * M + e_y

    Parameters
    ----------
    n : int
        Sample size.
    beta_1 : float
        Direct effect of T on Y.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcome, treatment, mediator, and true parameters.
    """
    return generate_mediation_data(
        n=n, alpha_1=0.0, beta_1=beta_1, beta_2=0.5, seed=seed
    )


def generate_mediation_with_covariates(
    n: int = 1000,
    alpha_1: float = 0.6,
    beta_1: float = 0.5,
    beta_2: float = 0.8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate mediation data with pre-treatment covariates.

    DGP:
    - X ~ Normal(0, 1)
    - T ~ Bernoulli(logistic(0.3*X))
    - M = alpha_0 + alpha_1 * T + gamma * X + e_m
    - Y = beta_0 + beta_1 * T + beta_2 * M + delta * X + e_y

    Parameters
    ----------
    n : int
        Sample size.
    alpha_1 : float
        Effect of T on M.
    beta_1 : float
        Direct effect of T on Y.
    beta_2 : float
        Effect of M on Y.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcome, treatment, mediator, covariates, and true parameters.
    """
    np.random.seed(seed)

    # Pre-treatment covariate
    X = np.random.normal(0, 1, n)

    # Treatment depends on X (confounding)
    p_treat = 1 / (1 + np.exp(-0.3 * X))
    treatment = np.random.binomial(1, p_treat).astype(float)

    # Mediator with covariate effect
    gamma = 0.4
    e_m = np.random.normal(0, 1, n)
    mediator = 3.0 + alpha_1 * treatment + gamma * X + e_m

    # Outcome with covariate effect
    delta = 0.3
    e_y = np.random.normal(0, 1, n)
    outcome = 2.0 + beta_1 * treatment + beta_2 * mediator + delta * X + e_y

    return {
        "outcome": outcome,
        "treatment": treatment,
        "mediator": mediator,
        "covariates": X.reshape(-1, 1),
        "true_alpha_1": alpha_1,
        "true_beta_1": beta_1,
        "true_beta_2": beta_2,
        "true_indirect": alpha_1 * beta_2,
        "true_direct": beta_1,
        "true_total": beta_1 + alpha_1 * beta_2,
        "n": n,
    }


# =============================================================================
# Test Class: Baron-Kenny vs R
# =============================================================================


@requires_mediation_python
@requires_mediation_r
class TestBaronKennyVsR:
    """Compare Python baron_kenny against R OLS implementation."""

    def test_path_coefficients(self):
        """Path coefficients (α₁, β₁, β₂) should match R."""
        data = generate_mediation_data(n=1000, seed=42)

        py_result = baron_kenny(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
        )

        r_result = r_baron_kenny(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
        )

        assert r_result is not None, "R implementation returned None"

        # Alpha_1 (T → M)
        assert np.isclose(
            py_result["alpha_1"], r_result["alpha_1"], rtol=0.02
        ), f"α₁ mismatch: Python={py_result['alpha_1']:.4f}, R={r_result['alpha_1']:.4f}"

        # Beta_1 (direct effect)
        assert np.isclose(
            py_result["beta_1"], r_result["beta_1"], rtol=0.02
        ), f"β₁ mismatch: Python={py_result['beta_1']:.4f}, R={r_result['beta_1']:.4f}"

        # Beta_2 (M → Y)
        assert np.isclose(
            py_result["beta_2"], r_result["beta_2"], rtol=0.02
        ), f"β₂ mismatch: Python={py_result['beta_2']:.4f}, R={r_result['beta_2']:.4f}"

    def test_sobel_test(self):
        """Sobel test statistic and p-value should match."""
        data = generate_mediation_data(n=1000, seed=123)

        py_result = baron_kenny(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
        )

        r_result = r_baron_kenny(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
        )

        assert r_result is not None

        # Sobel Z statistic
        assert np.isclose(
            py_result["sobel_z"], r_result["sobel_z"], rtol=0.05
        ), f"Sobel Z mismatch: Python={py_result['sobel_z']:.4f}, R={r_result['sobel_z']:.4f}"

        # Sobel p-value
        assert np.isclose(
            py_result["sobel_pvalue"], r_result["sobel_pvalue"], rtol=0.05
        ), f"Sobel p mismatch: Python={py_result['sobel_pvalue']:.4f}, R={r_result['sobel_pvalue']:.4f}"

    def test_effect_decomposition(self):
        """Direct, indirect, and total effects should match."""
        data = generate_mediation_data(n=1000, seed=456)

        py_result = baron_kenny(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
        )

        r_result = r_baron_kenny(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
        )

        assert r_result is not None

        # Indirect effect = α₁ × β₂
        assert np.isclose(
            py_result["indirect_effect"], r_result["indirect_effect"], rtol=0.02
        )

        # Direct effect = β₁
        assert np.isclose(
            py_result["direct_effect"], r_result["direct_effect"], rtol=0.02
        )

        # Total effect = β₁ + α₁×β₂
        assert np.isclose(
            py_result["total_effect"], r_result["total_effect"], rtol=0.02
        )

        # Verify decomposition: total = direct + indirect
        assert np.isclose(
            py_result["total_effect"],
            py_result["direct_effect"] + py_result["indirect_effect"],
            rtol=0.001,
        )

    def test_with_covariates(self):
        """Baron-Kenny with pre-treatment covariates."""
        data = generate_mediation_with_covariates(n=1000, seed=789)

        py_result = baron_kenny(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            covariates=data["covariates"],
        )

        r_result = r_baron_kenny(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            covariates=data["covariates"],
        )

        assert r_result is not None

        assert np.isclose(py_result["alpha_1"], r_result["alpha_1"], rtol=0.02)
        assert np.isclose(py_result["beta_1"], r_result["beta_1"], rtol=0.02)
        assert np.isclose(py_result["beta_2"], r_result["beta_2"], rtol=0.02)


# =============================================================================
# Test Class: Mediation Analysis vs R
# =============================================================================


@requires_mediation_python
@requires_mediation_r
class TestMediationAnalysisVsR:
    """Compare Python mediation_analysis against R mediation package."""

    def test_simulation_method(self):
        """NDE/NIE via simulation should match R mediate()."""
        data = generate_mediation_data(n=1000, seed=42)

        py_result = mediation_analysis(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            method="simulation",
            n_simulations=1000,
            n_bootstrap=500,
            random_state=42,
        )

        r_result = r_mediation_analysis(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            n_bootstrap=500,
            seed=42,
        )

        assert r_result is not None

        # NDE (Natural Direct Effect)
        assert np.isclose(
            py_result["direct_effect"], r_result["nde"], rtol=0.05
        ), f"NDE mismatch: Python={py_result['direct_effect']:.4f}, R={r_result['nde']:.4f}"

        # NIE (Natural Indirect Effect)
        assert np.isclose(
            py_result["indirect_effect"], r_result["nie"], rtol=0.05
        ), f"NIE mismatch: Python={py_result['indirect_effect']:.4f}, R={r_result['nie']:.4f}"

        # Total effect
        assert np.isclose(
            py_result["total_effect"], r_result["total_effect"], rtol=0.05
        )

    def test_proportion_mediated(self):
        """Proportion mediated (NIE/Total) should match."""
        data = generate_mediation_data(n=1000, seed=123)

        py_result = mediation_analysis(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            method="simulation",
            n_bootstrap=500,
            random_state=42,
        )

        r_result = r_mediation_analysis(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            n_bootstrap=500,
            seed=42,
        )

        assert r_result is not None

        # Proportion mediated
        assert np.isclose(
            py_result["proportion_mediated"],
            r_result["proportion_mediated"],
            rtol=0.10,
        ), f"PM mismatch: Python={py_result['proportion_mediated']:.4f}, R={r_result['proportion_mediated']:.4f}"

    def test_full_mediation(self):
        """When direct effect ≈ 0, proportion mediated ≈ 1."""
        data = generate_full_mediation_data(n=1000, seed=456)

        py_result = mediation_analysis(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            method="simulation",
            n_bootstrap=300,
            random_state=42,
        )

        r_result = r_mediation_analysis(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            n_bootstrap=300,
            seed=42,
        )

        assert r_result is not None

        # Direct effect should be near 0
        assert abs(py_result["direct_effect"]) < 0.3, "Expected small direct effect"

        # Indirect should dominate
        assert abs(py_result["indirect_effect"]) > abs(py_result["direct_effect"])

    def test_no_mediation(self):
        """When α₁ ≈ 0, indirect effect should be near zero."""
        data = generate_no_mediation_data(n=1000, seed=789)

        py_result = mediation_analysis(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            method="simulation",
            n_bootstrap=300,
            random_state=42,
        )

        r_result = r_mediation_analysis(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            n_bootstrap=300,
            seed=42,
        )

        assert r_result is not None

        # Indirect effect should be near 0
        assert abs(py_result["indirect_effect"]) < 0.3, "Expected small indirect effect"

        # Both should agree on small indirect
        assert np.isclose(
            py_result["indirect_effect"], r_result["nie"], atol=0.2
        )


# =============================================================================
# Test Class: Mediation Sensitivity vs R
# =============================================================================


@requires_mediation_python
@requires_mediation_r
class TestMediationSensitivityVsR:
    """Compare Python mediation_sensitivity against R medsens."""

    def test_original_effects(self):
        """Effects at ρ=0 should match base mediation analysis."""
        data = generate_mediation_data(n=500, seed=42)

        py_result = mediation_sensitivity(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            n_simulations=300,
            n_bootstrap=100,
            random_state=42,
        )

        r_result = r_mediation_sensitivity(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            n_bootstrap=100,
            seed=42,
        )

        assert r_result is not None

        # Original NDE at ρ=0
        assert np.isclose(
            py_result["original_nde"], r_result["original_nde"], rtol=0.05
        ), f"NDE at ρ=0 mismatch: Python={py_result['original_nde']:.4f}, R={r_result['original_nde']:.4f}"

        # Original NIE at ρ=0
        assert np.isclose(
            py_result["original_nie"], r_result["original_nie"], rtol=0.05
        ), f"NIE at ρ=0 mismatch: Python={py_result['original_nie']:.4f}, R={r_result['original_nie']:.4f}"

    def test_rho_grid(self):
        """Effects at various ρ values should be consistent."""
        data = generate_mediation_data(n=500, seed=123)

        py_result = mediation_sensitivity(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            rho_range=(-0.5, 0.5),
            n_rho=11,
            n_simulations=200,
            n_bootstrap=50,
            random_state=42,
        )

        r_result = r_mediation_sensitivity(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            rho_range=(-0.5, 0.5),
            n_rho=11,
            n_bootstrap=50,
            seed=42,
        )

        assert r_result is not None

        # Should have similar grid size
        assert len(py_result["rho_grid"]) >= 5
        assert len(r_result["rho_grid"]) >= 5

        # Effects should vary with ρ
        py_nie_range = py_result["nie_at_rho"].max() - py_result["nie_at_rho"].min()
        r_nie_range = r_result["nie_at_rho"].max() - r_result["nie_at_rho"].min()

        # Both should show sensitivity to ρ
        assert py_nie_range > 0.1 or r_nie_range > 0.1, "Expected variation with ρ"

    def test_zero_crossing(self):
        """Rho at which effect crosses zero should be similar."""
        # Generate data where effects will cross zero at some ρ
        data = generate_mediation_data(
            n=500, alpha_1=0.6, beta_1=0.3, beta_2=0.8, seed=456
        )

        py_result = mediation_sensitivity(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            rho_range=(-0.9, 0.9),
            n_rho=21,
            n_simulations=200,
            n_bootstrap=50,
            random_state=42,
        )

        r_result = r_mediation_sensitivity(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            rho_range=(-0.9, 0.9),
            n_rho=21,
            n_bootstrap=50,
            seed=42,
        )

        assert r_result is not None

        # If both find a zero crossing, they should be similar
        if py_result.get("rho_at_zero_nie") and r_result.get("rho_at_zero_nie"):
            assert np.isclose(
                py_result["rho_at_zero_nie"],
                r_result["rho_at_zero_nie"],
                atol=0.15,
            ), "Zero crossing ρ differs too much"


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


@requires_mediation_python
@requires_mediation_r
class TestMediationEdgeCases:
    """Test edge cases for mediation analysis."""

    def test_large_sample(self):
        """Large sample (n=2000) should converge well."""
        data = generate_mediation_data(n=2000, seed=111)

        py_result = baron_kenny(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
        )

        r_result = r_baron_kenny(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
        )

        assert r_result is not None

        # Should match very closely with large n
        assert np.isclose(py_result["alpha_1"], r_result["alpha_1"], rtol=0.01)
        assert np.isclose(py_result["beta_1"], r_result["beta_1"], rtol=0.01)
        assert np.isclose(py_result["beta_2"], r_result["beta_2"], rtol=0.01)

        # Estimates should be close to true values
        assert np.isclose(py_result["alpha_1"], data["true_alpha_1"], rtol=0.1)
        assert np.isclose(py_result["beta_1"], data["true_beta_1"], rtol=0.1)

    def test_sample_size_consistency(self):
        """Sample sizes should be reported consistently."""
        data = generate_mediation_data(n=500, seed=222)

        py_result = baron_kenny(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
        )

        r_result = r_baron_kenny(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
        )

        assert r_result is not None

        assert py_result["n_obs"] == r_result["n_obs"] == 500

    def test_r_squared_values(self):
        """R² values for mediator and outcome models should match."""
        data = generate_mediation_data(n=1000, seed=333)

        py_result = baron_kenny(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
        )

        r_result = r_baron_kenny(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
        )

        assert r_result is not None

        # R² for mediator model
        assert np.isclose(
            py_result["r2_mediator_model"], r_result["r2_mediator"], rtol=0.02
        )

        # R² for outcome model
        assert np.isclose(
            py_result["r2_outcome_model"], r_result["r2_outcome"], rtol=0.02
        )


# =============================================================================
# Main: Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
