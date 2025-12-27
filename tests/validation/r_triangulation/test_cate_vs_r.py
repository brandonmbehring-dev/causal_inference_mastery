"""Triangulation tests: Python Causal Forest vs R `grf` package.

This module provides Layer 5 validation by comparing our Python implementation
of Causal Forests (via econml) against the gold-standard R `grf` package
(Athey-Wager reference implementation).

Tests skip gracefully when R/rpy2 is unavailable.

Tolerance levels (established in plan):
- ATE: rtol=0.10 (honest splitting introduces randomness)
- ATE SE: rtol=0.20 (infinitesimal jackknife implementations may differ)
- CATE correlation: r > 0.80 (individual effects are noisy)
- CI coverage: similar patterns (not exact match)

Run with: pytest tests/validation/r_triangulation/test_cate_vs_r.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from tests.validation.r_triangulation.r_interface import (
    check_grf_installed,
    check_r_available,
    r_causal_forest_grf,
)

# Lazy import to avoid errors when module paths differ
try:
    from src.causal_inference.cate.causal_forest import causal_forest

    CATE_AVAILABLE = True
except ImportError:
    CATE_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests in this module if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_cate_python = pytest.mark.skipif(
    not CATE_AVAILABLE,
    reason="Python CATE module not available",
)

requires_grf_r = pytest.mark.skipif(
    not check_grf_installed(),
    reason="R 'grf' package not installed",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_cate_dgp(
    n: int = 1000,
    p: int = 5,
    effect_type: str = "heterogeneous",
    true_ate: float = 2.0,
    propensity: str = "balanced",
    noise_sd: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate data for CATE estimation.

    Model:
        Y = μ(X) + τ(X)*W + ε
        W ~ Bernoulli(e(X))

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of covariates.
    effect_type : str
        Type of treatment effect heterogeneity:
        - "constant": τ(X) = true_ate
        - "heterogeneous": τ(X) = true_ate + X₁
        - "nonlinear": τ(X) = true_ate * (1 + sin(π*X₁))
    true_ate : float
        Average treatment effect (or baseline for heterogeneous).
    propensity : str
        Propensity score model:
        - "balanced": e(X) = 0.5 (RCT-like)
        - "confounded": e(X) = expit(X₁)
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with keys: outcome, treatment, covariates, true_cate, true_ate
    """
    rng = np.random.default_rng(seed)

    # Generate covariates (standardized)
    X = rng.normal(0, 1, (n, p))

    # Baseline outcome model μ(X)
    mu = X[:, 0] + 0.5 * X[:, 1]

    # Treatment effect τ(X)
    if effect_type == "constant":
        tau = np.full(n, true_ate)
    elif effect_type == "heterogeneous":
        tau = true_ate + X[:, 0]
    elif effect_type == "nonlinear":
        tau = true_ate * (1 + np.sin(np.pi * X[:, 0]))
    else:
        raise ValueError(f"Unknown effect_type: {effect_type}")

    # Propensity score e(X)
    if propensity == "balanced":
        e = np.full(n, 0.5)
    elif propensity == "confounded":
        from scipy.special import expit

        e = expit(X[:, 0])
    else:
        raise ValueError(f"Unknown propensity: {propensity}")

    # Generate treatment
    W = rng.binomial(1, e)

    # Generate outcome
    epsilon = rng.normal(0, noise_sd, n)
    Y = mu + tau * W + epsilon

    return {
        "outcome": Y,
        "treatment": W,
        "covariates": X,
        "true_cate": tau,
        "true_ate": np.mean(tau),
        "propensity": e,
    }


# =============================================================================
# Test Classes
# =============================================================================


@requires_cate_python
@requires_grf_r
class TestCausalForestVsGRF:
    """Compare Python Causal Forest to R grf::causal_forest()."""

    def test_ate_parity_constant_effect(self):
        """ATE should match within rtol=0.10 for constant effect."""
        data = generate_cate_dgp(
            n=1000,
            p=5,
            effect_type="constant",
            true_ate=2.0,
            seed=42,
        )

        # Python result
        py_result = causal_forest(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_estimators=500,
            min_samples_leaf=5,
        )

        # R result
        r_result = r_causal_forest_grf(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_trees=500,
            min_node_size=5,
            seed=42,
        )

        # ATE should match within tolerance
        np.testing.assert_allclose(
            py_result.ate,
            r_result["ate"],
            rtol=0.10,
            err_msg=f"ATE mismatch: Python={py_result.ate:.4f}, R={r_result['ate']:.4f}",
        )

    def test_ate_parity_heterogeneous_effect(self):
        """ATE should match within rtol=0.10 for heterogeneous effects."""
        data = generate_cate_dgp(
            n=1000,
            p=5,
            effect_type="heterogeneous",
            true_ate=2.0,
            seed=123,
        )

        py_result = causal_forest(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_estimators=500,
        )

        r_result = r_causal_forest_grf(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_trees=500,
            seed=123,
        )

        np.testing.assert_allclose(
            py_result.ate,
            r_result["ate"],
            rtol=0.10,
            err_msg=f"ATE mismatch: Python={py_result.ate:.4f}, R={r_result['ate']:.4f}",
        )

    def test_se_parity(self):
        """SE should match within rtol=0.20 (jackknife approximations differ)."""
        data = generate_cate_dgp(n=800, p=5, effect_type="constant", seed=456)

        py_result = causal_forest(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_estimators=500,
        )

        r_result = r_causal_forest_grf(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_trees=500,
            seed=456,
        )

        # SE should be in similar ballpark (not exact due to jackknife differences)
        np.testing.assert_allclose(
            py_result.ate_se,
            r_result["ate_se"],
            rtol=0.20,
            err_msg=f"SE mismatch: Python={py_result.ate_se:.4f}, R={r_result['ate_se']:.4f}",
        )

    def test_cate_correlation(self):
        """Individual CATE estimates should be correlated (r > 0.80)."""
        data = generate_cate_dgp(
            n=1000,
            p=5,
            effect_type="heterogeneous",
            true_ate=2.0,
            seed=789,
        )

        py_result = causal_forest(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_estimators=500,
        )

        r_result = r_causal_forest_grf(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_trees=500,
            seed=789,
        )

        # Compute correlation between Python and R CATE estimates
        correlation, p_value = stats.pearsonr(py_result.cate, r_result["cate"])

        assert correlation > 0.80, (
            f"CATE correlation too low: r={correlation:.3f} (expected > 0.80)"
        )

    def test_high_dimensional(self):
        """Should handle high-dimensional covariates (p=50)."""
        data = generate_cate_dgp(
            n=500,
            p=50,
            effect_type="heterogeneous",
            seed=999,
        )

        py_result = causal_forest(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_estimators=200,
        )

        r_result = r_causal_forest_grf(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_trees=200,
            seed=999,
        )

        # Should produce estimates without error
        assert np.isfinite(py_result.ate), "Python ATE not finite"
        assert np.isfinite(r_result["ate"]), "R ATE not finite"

        # Still within tolerance
        np.testing.assert_allclose(
            py_result.ate,
            r_result["ate"],
            rtol=0.15,  # Slightly looser for high-dim
            err_msg="High-dim ATE mismatch",
        )


@requires_cate_python
@requires_grf_r
class TestCATEEdgeCases:
    """Edge cases for CATE validation."""

    def test_small_sample(self):
        """Should work with small samples (n=200)."""
        data = generate_cate_dgp(n=200, p=3, effect_type="constant", seed=111)

        py_result = causal_forest(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_estimators=100,
            min_samples_leaf=10,
        )

        r_result = r_causal_forest_grf(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_trees=100,
            min_node_size=10,
            seed=111,
        )

        # Both should produce finite estimates
        assert np.isfinite(py_result.ate), "Python ATE not finite for small sample"
        assert np.isfinite(r_result["ate"]), "R ATE not finite for small sample"

        # Wider tolerance for small sample
        np.testing.assert_allclose(
            py_result.ate,
            r_result["ate"],
            rtol=0.20,
            err_msg="Small sample ATE mismatch",
        )

    def test_zero_effect(self):
        """Should detect zero treatment effect."""
        data = generate_cate_dgp(
            n=800,
            p=5,
            effect_type="constant",
            true_ate=0.0,
            seed=222,
        )

        py_result = causal_forest(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_estimators=300,
        )

        r_result = r_causal_forest_grf(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_trees=300,
            seed=222,
        )

        # Both should be near zero (absolute tolerance)
        assert abs(py_result.ate) < 0.5, f"Python ATE={py_result.ate:.3f} not near zero"
        assert abs(r_result["ate"]) < 0.5, f"R ATE={r_result['ate']:.3f} not near zero"

        # CI should cover zero for both
        assert py_result.ci_lower < 0 < py_result.ci_upper, "Python CI doesn't cover 0"
        assert r_result["ate_ci_lower"] < 0 < r_result["ate_ci_upper"], "R CI doesn't cover 0"

    def test_nonlinear_effect(self):
        """Should capture nonlinear heterogeneity."""
        data = generate_cate_dgp(
            n=1000,
            p=5,
            effect_type="nonlinear",
            true_ate=2.0,
            seed=333,
        )

        py_result = causal_forest(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_estimators=500,
        )

        r_result = r_causal_forest_grf(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_trees=500,
            seed=333,
        )

        # Both should capture mean effect reasonably
        np.testing.assert_allclose(
            py_result.ate,
            r_result["ate"],
            rtol=0.15,
            err_msg="Nonlinear ATE mismatch",
        )

        # CATE variation should be detected (not constant)
        py_cate_sd = np.std(py_result.cate)
        r_cate_sd = np.std(r_result["cate"])

        assert py_cate_sd > 0.1, "Python CATE shows no heterogeneity"
        assert r_cate_sd > 0.1, "R CATE shows no heterogeneity"


@requires_cate_python
@requires_grf_r
class TestCATEMonteCarlo:
    """Monte Carlo validation for CATE estimation."""

    @pytest.mark.slow
    def test_bias_comparison(self):
        """Compare bias distribution across 20 simulations."""
        n_sims = 20
        py_biases = []
        r_biases = []

        true_ate = 2.0

        for seed in range(n_sims):
            data = generate_cate_dgp(
                n=500,
                p=5,
                effect_type="constant",
                true_ate=true_ate,
                seed=1000 + seed,
            )

            py_result = causal_forest(
                outcomes=data["outcome"],
                treatment=data["treatment"],
                covariates=data["covariates"],
                n_estimators=200,
            )

            r_result = r_causal_forest_grf(
                outcome=data["outcome"],
                treatment=data["treatment"],
                covariates=data["covariates"],
                n_trees=200,
                seed=1000 + seed,
            )

            py_biases.append(py_result.ate - true_ate)
            r_biases.append(r_result["ate"] - true_ate)

        # Both should have similar mean bias (roughly unbiased)
        py_mean_bias = np.mean(py_biases)
        r_mean_bias = np.mean(r_biases)

        assert abs(py_mean_bias) < 0.3, f"Python bias too large: {py_mean_bias:.3f}"
        assert abs(r_mean_bias) < 0.3, f"R bias too large: {r_mean_bias:.3f}"

        # Bias should be similar between Python and R
        assert abs(py_mean_bias - r_mean_bias) < 0.2, (
            f"Bias differs: Python={py_mean_bias:.3f}, R={r_mean_bias:.3f}"
        )

    @pytest.mark.slow
    def test_coverage_comparison(self):
        """Compare CI coverage across 20 simulations."""
        n_sims = 20
        py_coverage = 0
        r_coverage = 0

        true_ate = 2.0

        for seed in range(n_sims):
            data = generate_cate_dgp(
                n=500,
                p=5,
                effect_type="constant",
                true_ate=true_ate,
                seed=2000 + seed,
            )

            py_result = causal_forest(
                outcomes=data["outcome"],
                treatment=data["treatment"],
                covariates=data["covariates"],
                n_estimators=200,
            )

            r_result = r_causal_forest_grf(
                outcome=data["outcome"],
                treatment=data["treatment"],
                covariates=data["covariates"],
                n_trees=200,
                seed=2000 + seed,
            )

            if py_result.ci_lower < true_ate < py_result.ci_upper:
                py_coverage += 1
            if r_result["ate_ci_lower"] < true_ate < r_result["ate_ci_upper"]:
                r_coverage += 1

        py_rate = py_coverage / n_sims
        r_rate = r_coverage / n_sims

        # Both should have reasonable coverage (not too far from 95%)
        assert 0.70 < py_rate < 1.0, f"Python coverage={py_rate:.1%} outside range"
        assert 0.70 < r_rate < 1.0, f"R coverage={r_rate:.1%} outside range"

        # Coverage should be similar between implementations
        assert abs(py_rate - r_rate) < 0.25, (
            f"Coverage differs: Python={py_rate:.1%}, R={r_rate:.1%}"
        )
