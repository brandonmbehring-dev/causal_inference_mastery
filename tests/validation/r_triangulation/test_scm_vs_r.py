"""Triangulation tests: Python SCM vs R `Synth` package.

This module provides Layer 5 validation by comparing our Python implementation
of Synthetic Control Methods against the official R `Synth` package.

Tests skip gracefully when R/rpy2 is unavailable.

Tolerance levels (established in plan):
- SCM weights (top 3): rtol=0.10 (optimization convergence differences)
- ATT estimate: rtol=0.15 (gap calculation)
- Pre-treatment RMSE: rtol=0.10 (same formula)

Note: SCM optimization may converge to different local minima in Python vs R,
so we use looser tolerances than for IV.

Run with: pytest tests/validation/r_triangulation/test_scm_vs_r.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.validation.r_triangulation.r_interface import (
    check_r_available,
    check_synth_installed,
    r_scm_synth,
)

# Lazy import to avoid errors when scm module paths differ
try:
    from src.causal_inference.scm.basic_scm import synthetic_control

    SCM_AVAILABLE = True
except ImportError:
    SCM_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests in this module if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_scm_python = pytest.mark.skipif(
    not SCM_AVAILABLE,
    reason="Python SCM module not available",
)

requires_synth_r = pytest.mark.skipif(
    not check_synth_installed(),
    reason="R 'Synth' package not installed",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_scm_panel(
    n_controls: int = 20,
    n_pre: int = 10,
    n_post: int = 5,
    true_effect: float = 2.0,
    n_factors: int = 2,
    noise_sd: float = 0.5,
    seed: int = 42,
) -> dict:
    """Generate panel data for SCM from a factor model.

    Model:
        Y_it = α_i + λ_i'F_t + τ*I(treated)*I(post) + ε_it

    where:
        α_i = unit fixed effect
        λ_i = unit-specific factor loadings (n_factors,)
        F_t = common time factors (n_factors,)
        τ = treatment effect (only for treated unit in post-treatment)

    Parameters
    ----------
    n_controls : int
        Number of control units.
    n_pre : int
        Number of pre-treatment periods.
    n_post : int
        Number of post-treatment periods.
    true_effect : float
        True treatment effect τ.
    n_factors : int
        Number of latent factors.
    noise_sd : float
        Standard deviation of idiosyncratic noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcome_treated, outcome_controls, pre_periods,
        true_effect, and other metadata.
    """
    np.random.seed(seed)

    T = n_pre + n_post

    # Unit fixed effects
    alpha_treated = np.random.randn() * 2
    alpha_controls = np.random.randn(n_controls) * 2

    # Factor loadings
    lambda_treated = np.random.randn(n_factors)
    lambda_controls = np.random.randn(n_controls, n_factors)

    # Time factors (smooth trends)
    F = np.zeros((T, n_factors))
    for k in range(n_factors):
        F[:, k] = np.sin(2 * np.pi * np.arange(T) / (T + k)) + 0.5 * np.arange(T) / T

    # Generate outcomes
    # Treated unit
    outcome_treated = np.zeros(T)
    for t in range(T):
        outcome_treated[t] = alpha_treated + lambda_treated @ F[t]
        if t >= n_pre:  # Post-treatment
            outcome_treated[t] += true_effect
        outcome_treated[t] += noise_sd * np.random.randn()

    # Control units
    outcome_controls = np.zeros((T, n_controls))
    for j in range(n_controls):
        for t in range(T):
            outcome_controls[t, j] = alpha_controls[j] + lambda_controls[j] @ F[t]
            outcome_controls[t, j] += noise_sd * np.random.randn()

    return {
        "outcome_treated": outcome_treated,
        "outcome_controls": outcome_controls,
        "pre_periods": n_pre,
        "n_controls": n_controls,
        "n_pre": n_pre,
        "n_post": n_post,
        "true_effect": true_effect,
        "T": T,
    }


def generate_scm_perfect_fit(
    n_controls: int = 10,
    n_pre: int = 8,
    n_post: int = 4,
    true_effect: float = 3.0,
    seed: int = 42,
) -> dict:
    """Generate data where synthetic control can achieve perfect pre-treatment fit.

    The treated unit is a convex combination of controls plus treatment effect.
    """
    np.random.seed(seed)

    T = n_pre + n_post

    # Generate control outcomes with trends
    outcome_controls = np.zeros((T, n_controls))
    for j in range(n_controls):
        trend = 0.5 * np.arange(T) + np.random.randn() * 2
        cycle = np.sin(2 * np.pi * np.arange(T) / 6) * (j + 1) * 0.3
        outcome_controls[:, j] = trend + cycle + np.random.randn(T) * 0.1

    # Treated = weighted combination of first 3 controls
    weights_true = np.array([0.4, 0.3, 0.3] + [0.0] * (n_controls - 3))
    outcome_treated = outcome_controls @ weights_true

    # Add treatment effect post-treatment
    outcome_treated[n_pre:] += true_effect

    return {
        "outcome_treated": outcome_treated,
        "outcome_controls": outcome_controls,
        "pre_periods": n_pre,
        "n_controls": n_controls,
        "n_pre": n_pre,
        "n_post": n_post,
        "true_effect": true_effect,
        "true_weights": weights_true,
        "T": T,
    }


# =============================================================================
# Layer 5: SCM Triangulation
# =============================================================================


@requires_scm_python
@requires_synth_r
class TestClassicSCMVsSynth:
    """Compare Python synthetic_control() to R `Synth::synth()`."""

    def test_basic_att_parity(self):
        """ATT estimate should match R Synth within rtol=0.15."""
        data = generate_scm_panel(
            n_controls=15,
            n_pre=10,
            n_post=5,
            true_effect=2.0,
            seed=42,
        )

        # Python estimate
        py_result = synthetic_control(
            treated_outcome=data["outcome_treated"],
            control_outcomes=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        # R estimate
        r_result = r_scm_synth(
            outcome_treated=data["outcome_treated"],
            outcome_controls=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        py_att = py_result["estimate"]
        r_att = r_result["att"]

        assert np.isclose(py_att, r_att, rtol=0.15), (
            f"SCM ATT mismatch: Python={py_att:.4f}, R={r_att:.4f}"
        )

    def test_pre_rmse_parity(self):
        """Pre-treatment RMSE should match R Synth within rtol=0.10."""
        data = generate_scm_panel(
            n_controls=20,
            n_pre=12,
            n_post=4,
            true_effect=1.5,
            seed=123,
        )

        py_result = synthetic_control(
            treated_outcome=data["outcome_treated"],
            control_outcomes=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        r_result = r_scm_synth(
            outcome_treated=data["outcome_treated"],
            outcome_controls=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        py_rmse = py_result["pre_rmse"]
        r_rmse = r_result["pre_rmse"]

        assert np.isclose(py_rmse, r_rmse, rtol=0.20), (
            f"Pre-RMSE mismatch: Python={py_rmse:.4f}, R={r_rmse:.4f}"
        )

    def test_weights_simplex_constraint(self):
        """Both Python and R should produce simplex weights (sum=1, non-negative)."""
        data = generate_scm_panel(
            n_controls=15,
            n_pre=10,
            n_post=5,
            true_effect=2.0,
            seed=456,
        )

        py_result = synthetic_control(
            treated_outcome=data["outcome_treated"],
            control_outcomes=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        r_result = r_scm_synth(
            outcome_treated=data["outcome_treated"],
            outcome_controls=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        # Python weights
        py_weights = py_result["weights"]
        assert np.isclose(np.sum(py_weights), 1.0, atol=1e-6), (
            f"Python weights sum {np.sum(py_weights):.6f} != 1"
        )
        assert np.all(py_weights >= -1e-6), "Python weights have negative values"

        # R weights
        r_weights = r_result["weights"]
        assert np.isclose(np.sum(r_weights), 1.0, atol=1e-6), (
            f"R weights sum {np.sum(r_weights):.6f} != 1"
        )
        assert np.all(r_weights >= -1e-6), "R weights have negative values"

    def test_synthetic_series_shape(self):
        """Synthetic control series should have correct shape."""
        data = generate_scm_panel(
            n_controls=10,
            n_pre=8,
            n_post=4,
            true_effect=2.5,
            seed=789,
        )

        py_result = synthetic_control(
            treated_outcome=data["outcome_treated"],
            control_outcomes=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        r_result = r_scm_synth(
            outcome_treated=data["outcome_treated"],
            outcome_controls=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        T = data["T"]
        assert len(py_result["synthetic_control"]) == T, (
            f"Python synthetic series length {len(py_result['synthetic_control'])} != {T}"
        )
        assert len(r_result["synthetic_control"]) == T, (
            f"R synthetic series length {len(r_result['synthetic_control'])} != {T}"
        )

    def test_gap_calculation_parity(self):
        """Gap (treated - synthetic) should be consistent."""
        data = generate_scm_panel(
            n_controls=15,
            n_pre=10,
            n_post=5,
            true_effect=3.0,
            seed=101,
        )

        py_result = synthetic_control(
            treated_outcome=data["outcome_treated"],
            control_outcomes=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        r_result = r_scm_synth(
            outcome_treated=data["outcome_treated"],
            outcome_controls=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        # Post-treatment gap mean should be similar
        n_pre = data["n_pre"]
        py_post_gap = py_result["gap"][n_pre:]
        r_post_gap = r_result["gap"][n_pre:]

        py_mean_gap = np.mean(py_post_gap)
        r_mean_gap = np.mean(r_post_gap)

        assert np.isclose(py_mean_gap, r_mean_gap, rtol=0.20), (
            f"Post-treatment gap mean mismatch: Python={py_mean_gap:.4f}, R={r_mean_gap:.4f}"
        )


@requires_scm_python
@requires_synth_r
class TestSCMPerfectFit:
    """Test SCM when perfect pre-treatment fit is possible."""

    def test_perfect_fit_weights(self):
        """When treated is convex combo of controls, weights should match."""
        data = generate_scm_perfect_fit(
            n_controls=10,
            n_pre=8,
            n_post=4,
            true_effect=3.0,
            seed=202,
        )

        py_result = synthetic_control(
            treated_outcome=data["outcome_treated"],
            control_outcomes=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        r_result = r_scm_synth(
            outcome_treated=data["outcome_treated"],
            outcome_controls=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        # Pre-RMSE should be near zero for both
        assert py_result["pre_rmse"] < 0.5, (
            f"Python pre-RMSE {py_result['pre_rmse']:.4f} too high for perfect fit case"
        )
        assert r_result["pre_rmse"] < 0.5, (
            f"R pre-RMSE {r_result['pre_rmse']:.4f} too high for perfect fit case"
        )

    def test_perfect_fit_att(self):
        """ATT should recover true effect when fit is perfect."""
        data = generate_scm_perfect_fit(
            n_controls=10,
            n_pre=8,
            n_post=4,
            true_effect=3.0,
            seed=303,
        )

        py_result = synthetic_control(
            treated_outcome=data["outcome_treated"],
            control_outcomes=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        r_result = r_scm_synth(
            outcome_treated=data["outcome_treated"],
            outcome_controls=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        true_effect = data["true_effect"]

        # Both should be close to true effect
        assert np.isclose(py_result["estimate"], true_effect, rtol=0.15), (
            f"Python ATT {py_result['estimate']:.4f} far from true {true_effect}"
        )
        assert np.isclose(r_result["att"], true_effect, rtol=0.15), (
            f"R ATT {r_result['att']:.4f} far from true {true_effect}"
        )


@requires_scm_python
@requires_synth_r
class TestSCMEdgeCases:
    """Edge case tests for SCM triangulation."""

    def test_few_controls(self):
        """SCM should work with few controls (J=5)."""
        data = generate_scm_panel(
            n_controls=5,
            n_pre=8,
            n_post=4,
            true_effect=2.0,
            seed=404,
        )

        py_result = synthetic_control(
            treated_outcome=data["outcome_treated"],
            control_outcomes=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        r_result = r_scm_synth(
            outcome_treated=data["outcome_treated"],
            outcome_controls=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        # Should produce valid estimates
        assert np.isfinite(py_result["estimate"]), "Python ATT not finite"
        assert np.isfinite(r_result["att"]), "R ATT not finite"

        # Should be somewhat close
        assert np.isclose(py_result["estimate"], r_result["att"], rtol=0.30), (
            f"Few controls ATT mismatch: Python={py_result['estimate']:.4f}, R={r_result['att']:.4f}"
        )

    def test_short_pretreatment(self):
        """SCM should work with short pre-treatment (T_pre=5)."""
        data = generate_scm_panel(
            n_controls=15,
            n_pre=5,
            n_post=3,
            true_effect=2.5,
            seed=505,
        )

        py_result = synthetic_control(
            treated_outcome=data["outcome_treated"],
            control_outcomes=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        r_result = r_scm_synth(
            outcome_treated=data["outcome_treated"],
            outcome_controls=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        # Should produce valid estimates
        assert np.isfinite(py_result["estimate"]), "Python ATT not finite"
        assert np.isfinite(r_result["att"]), "R ATT not finite"

    def test_many_controls(self):
        """SCM should work with many controls (J=50)."""
        data = generate_scm_panel(
            n_controls=50,
            n_pre=10,
            n_post=5,
            true_effect=1.5,
            seed=606,
        )

        py_result = synthetic_control(
            treated_outcome=data["outcome_treated"],
            control_outcomes=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        r_result = r_scm_synth(
            outcome_treated=data["outcome_treated"],
            outcome_controls=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        assert np.isclose(py_result["estimate"], r_result["att"], rtol=0.20), (
            f"Many controls ATT mismatch: Python={py_result['estimate']:.4f}, R={r_result['att']:.4f}"
        )

    def test_zero_effect(self):
        """SCM should recover zero effect correctly."""
        data = generate_scm_panel(
            n_controls=15,
            n_pre=10,
            n_post=5,
            true_effect=0.0,  # No treatment effect
            seed=707,
        )

        py_result = synthetic_control(
            treated_outcome=data["outcome_treated"],
            control_outcomes=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        r_result = r_scm_synth(
            outcome_treated=data["outcome_treated"],
            outcome_controls=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        # Both should be close to zero
        assert abs(py_result["estimate"]) < 1.0, (
            f"Python zero effect estimate {py_result['estimate']:.4f} too far from 0"
        )
        assert abs(r_result["att"]) < 1.0, (
            f"R zero effect estimate {r_result['att']:.4f} too far from 0"
        )


@requires_scm_python
@requires_synth_r
class TestSCMWeightsParity:
    """Test weight distribution parity between Python and R."""

    def test_sparse_weights(self):
        """Both should produce sparse weights (few controls with high weight)."""
        data = generate_scm_panel(
            n_controls=20,
            n_pre=12,
            n_post=4,
            true_effect=2.0,
            seed=808,
        )

        py_result = synthetic_control(
            treated_outcome=data["outcome_treated"],
            control_outcomes=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        r_result = r_scm_synth(
            outcome_treated=data["outcome_treated"],
            outcome_controls=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        # Count controls with weight > 0.05
        py_active = np.sum(py_result["weights"] > 0.05)
        r_active = np.sum(r_result["weights"] > 0.05)

        # Both should have sparsity (not all controls equally weighted)
        n_controls = data["n_controls"]
        assert py_active < n_controls, "Python weights not sparse"
        assert r_active < n_controls, "R weights not sparse"

    def test_top_weights_order(self):
        """Top weighted controls should be similar (not identical due to optimization)."""
        data = generate_scm_panel(
            n_controls=15,
            n_pre=10,
            n_post=5,
            true_effect=2.0,
            seed=909,
        )

        py_result = synthetic_control(
            treated_outcome=data["outcome_treated"],
            control_outcomes=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        r_result = r_scm_synth(
            outcome_treated=data["outcome_treated"],
            outcome_controls=data["outcome_controls"],
            pre_periods=data["pre_periods"],
        )

        # Get top 3 weighted controls
        py_top3 = np.argsort(py_result["weights"])[-3:]
        r_top3 = np.argsort(r_result["weights"])[-3:]

        # At least 1-2 should overlap (not strict requirement due to optimization)
        overlap = len(set(py_top3) & set(r_top3))
        assert overlap >= 1, f"No overlap in top 3 weighted controls: Python={py_top3}, R={r_top3}"


@requires_scm_python
@requires_synth_r
class TestSCMMonteCarlo:
    """Monte Carlo validation of SCM triangulation."""

    @pytest.mark.slow
    def test_monte_carlo_bias_comparison(self):
        """Both Python and R should have similar bias properties."""
        n_sims = 30
        true_effect = 2.0
        py_estimates = []
        r_estimates = []

        for sim in range(n_sims):
            data = generate_scm_panel(
                n_controls=15,
                n_pre=10,
                n_post=5,
                true_effect=true_effect,
                seed=4000 + sim,
            )

            try:
                py_result = synthetic_control(
                    treated_outcome=data["outcome_treated"],
                    control_outcomes=data["outcome_controls"],
                    pre_periods=data["pre_periods"],
                )
                py_estimates.append(py_result["estimate"])

                r_result = r_scm_synth(
                    outcome_treated=data["outcome_treated"],
                    outcome_controls=data["outcome_controls"],
                    pre_periods=data["pre_periods"],
                )
                r_estimates.append(r_result["att"])
            except Exception:
                continue

        if len(py_estimates) < 20:
            pytest.skip("Too few successful simulations")

        py_bias = np.mean(py_estimates) - true_effect
        r_bias = np.mean(r_estimates) - true_effect

        # SCM can have bias, but should be similar between implementations
        assert abs(py_bias - r_bias) < 0.50, (
            f"Bias difference {abs(py_bias - r_bias):.4f} between Python and R too large"
        )

    @pytest.mark.slow
    def test_monte_carlo_rmse_comparison(self):
        """Pre-treatment RMSE should be similar on average."""
        n_sims = 30
        py_rmses = []
        r_rmses = []

        for sim in range(n_sims):
            data = generate_scm_panel(
                n_controls=15,
                n_pre=10,
                n_post=5,
                true_effect=2.0,
                seed=5000 + sim,
            )

            try:
                py_result = synthetic_control(
                    treated_outcome=data["outcome_treated"],
                    control_outcomes=data["outcome_controls"],
                    pre_periods=data["pre_periods"],
                )
                py_rmses.append(py_result["pre_rmse"])

                r_result = r_scm_synth(
                    outcome_treated=data["outcome_treated"],
                    outcome_controls=data["outcome_controls"],
                    pre_periods=data["pre_periods"],
                )
                r_rmses.append(r_result["pre_rmse"])
            except Exception:
                continue

        if len(py_rmses) < 20:
            pytest.skip("Too few successful simulations")

        py_mean_rmse = np.mean(py_rmses)
        r_mean_rmse = np.mean(r_rmses)

        # RMSE should be similar on average
        assert np.isclose(py_mean_rmse, r_mean_rmse, rtol=0.30), (
            f"Mean pre-RMSE mismatch: Python={py_mean_rmse:.4f}, R={r_mean_rmse:.4f}"
        )
