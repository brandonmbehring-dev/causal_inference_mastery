"""
Cross-language validation tests for Synthetic Control Methods.

Tests Python ↔ Julia parity for SCM (Session 47).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.scm import synthetic_control, augmented_synthetic_control

# Import Julia interface with skip if unavailable
try:
    from tests.validation.cross_language.julia_interface import (
        is_julia_available,
        julia_synthetic_control,
        julia_augmented_scm,
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


def generate_scm_data(
    n_control: int = 10,
    n_periods: int = 20,
    treatment_period: int = 11,
    true_effect: float = 5.0,
    random_seed: int = 42,
):
    """
    Generate panel data for SCM testing.

    Returns tuple: (outcomes, treatment, treatment_period, true_effect)
    """
    np.random.seed(random_seed)

    # Common time trend
    time_trend = np.linspace(0, 2, n_periods)

    # Control unit outcomes with individual effects
    control_outcomes = np.zeros((n_control, n_periods))
    for i in range(n_control):
        unit_effect = 10 + i * 0.5
        control_outcomes[i, :] = unit_effect + time_trend + 0.3 * np.random.randn(n_periods)

    # Treated unit = weighted average of controls pre-treatment + effect post-treatment
    weights = np.random.dirichlet(np.ones(n_control))
    treated_outcome = control_outcomes.T @ weights  # (n_periods,)
    treated_outcome[treatment_period - 1 :] += true_effect  # Julia 1-indexed

    # Combine into panel
    outcomes = np.vstack([treated_outcome.reshape(1, -1), control_outcomes])
    treatment = np.array([True] + [False] * n_control)

    return outcomes, treatment, treatment_period, true_effect


# =============================================================================
# SyntheticControl Tests
# =============================================================================


class TestSyntheticControlParity:
    """Python ↔ Julia parity for SyntheticControl."""

    def test_estimate_parity(self):
        """ATE estimates should match within tolerance."""
        outcomes, treatment, treatment_period, true_effect = generate_scm_data()

        # Python (returns TypedDict, access with [])
        py_result = synthetic_control(
            outcomes=outcomes,
            treatment=treatment.astype(int),  # Python expects 0/1 int
            treatment_period=treatment_period - 1,  # Python is 0-indexed
            inference="none",
        )

        # Julia
        jl_result = julia_synthetic_control(
            outcomes=outcomes,
            treatment=treatment,
            treatment_period=treatment_period,  # Julia is 1-indexed
            inference="none",
        )

        assert_allclose(
            py_result["estimate"],
            jl_result["estimate"],
            rtol=0.10,
            err_msg=f"ATE mismatch: Python={py_result['estimate']:.4f}, Julia={jl_result['estimate']:.4f}",
        )

    def test_weights_correlation(self):
        """Weights should be highly correlated (same optimization objective)."""
        outcomes, treatment, treatment_period, _ = generate_scm_data(random_seed=123)

        py_result = synthetic_control(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            treatment_period=treatment_period - 1,
            inference="none",
        )

        jl_result = julia_synthetic_control(
            outcomes=outcomes,
            treatment=treatment,
            treatment_period=treatment_period,
            inference="none",
        )

        # Correlation between weight vectors
        corr = np.corrcoef(py_result["weights"], jl_result["weights"])[0, 1]
        assert corr > 0.90, f"Weight correlation {corr:.3f} below threshold 0.90"

    def test_pre_fit_metrics(self):
        """Pre-treatment fit metrics should be similar."""
        outcomes, treatment, treatment_period, _ = generate_scm_data(random_seed=456)

        py_result = synthetic_control(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            treatment_period=treatment_period - 1,
            inference="none",
        )

        jl_result = julia_synthetic_control(
            outcomes=outcomes,
            treatment=treatment,
            treatment_period=treatment_period,
            inference="none",
        )

        # Pre-RMSE should be similar
        # Note: When DGP creates perfect fit, Python may find exact weights (RMSE≈0)
        # while Julia may settle slightly differently. Use atol for small values.
        assert_allclose(
            py_result["pre_rmse"],
            jl_result["pre_rmse"],
            rtol=0.20,
            atol=0.01,  # Allow small absolute difference when both are near zero
            err_msg=f"Pre-RMSE mismatch: Python={py_result['pre_rmse']:.4f}, Julia={jl_result['pre_rmse']:.4f}",
        )

    def test_sample_sizes(self):
        """Sample size counts should match exactly."""
        outcomes, treatment, treatment_period, _ = generate_scm_data()

        py_result = synthetic_control(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            treatment_period=treatment_period - 1,
            inference="none",
        )

        jl_result = julia_synthetic_control(
            outcomes=outcomes,
            treatment=treatment,
            treatment_period=treatment_period,
            inference="none",
        )

        assert py_result["n_treated"] == jl_result["n_treated"]
        assert py_result["n_control"] == jl_result["n_control"]
        assert py_result["n_pre_periods"] == jl_result["n_pre_periods"]
        assert py_result["n_post_periods"] == jl_result["n_post_periods"]

    def test_placebo_inference(self):
        """Placebo SE should be in same ballpark."""
        outcomes, treatment, treatment_period, _ = generate_scm_data(random_seed=789)

        py_result = synthetic_control(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            treatment_period=treatment_period - 1,
            inference="placebo",
            n_placebo=20,
        )

        jl_result = julia_synthetic_control(
            outcomes=outcomes,
            treatment=treatment,
            treatment_period=treatment_period,
            inference="placebo",
            n_placebo=20,
        )

        # SE should be positive and finite in both
        assert py_result["se"] > 0
        assert jl_result["se"] is not None and jl_result["se"] > 0

        # SE within factor of 3 (high variance due to limited placebos)
        ratio = py_result["se"] / jl_result["se"]
        assert 0.33 < ratio < 3.0, f"SE ratio {ratio:.2f} outside [0.33, 3.0]"


# =============================================================================
# AugmentedSC Tests
# =============================================================================


class TestAugmentedSCMParity:
    """Python ↔ Julia parity for AugmentedSC."""

    def test_estimate_parity(self):
        """ASCM estimates should match within tolerance."""
        outcomes, treatment, treatment_period, _ = generate_scm_data(random_seed=101)

        py_result = augmented_synthetic_control(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            treatment_period=treatment_period - 1,  # Python 0-indexed
            inference="none",
        )

        jl_result = julia_augmented_scm(
            outcomes=outcomes,
            treatment=treatment,
            treatment_period=treatment_period,  # Julia 1-indexed
            inference="none",
        )

        assert_allclose(
            py_result["estimate"],
            jl_result["estimate"],
            rtol=0.15,  # Slightly higher tolerance due to ridge CV
            err_msg=f"ASCM estimate mismatch: Python={py_result['estimate']:.4f}, Julia={jl_result['estimate']:.4f}",
        )

    def test_weights_sum_to_one(self):
        """Both implementations should produce simplex-constrained weights."""
        outcomes, treatment, treatment_period, _ = generate_scm_data(random_seed=202)

        py_result = augmented_synthetic_control(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            treatment_period=treatment_period - 1,
            inference="none",
        )

        jl_result = julia_augmented_scm(
            outcomes=outcomes,
            treatment=treatment,
            treatment_period=treatment_period,
            inference="none",
        )

        assert_allclose(np.sum(py_result["weights"]), 1.0, atol=1e-6)
        assert_allclose(np.sum(jl_result["weights"]), 1.0, atol=1e-6)

    def test_jackknife_se(self):
        """Jackknife SE should be similar."""
        outcomes, treatment, treatment_period, _ = generate_scm_data(n_control=8, random_seed=303)

        py_result = augmented_synthetic_control(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            treatment_period=treatment_period - 1,
            inference="jackknife",
        )

        jl_result = julia_augmented_scm(
            outcomes=outcomes,
            treatment=treatment,
            treatment_period=treatment_period,
            inference="jackknife",
        )

        # Both should produce valid SE
        assert py_result["se"] > 0
        assert jl_result["se"] is not None and jl_result["se"] > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestSCMIntegration:
    """Integration tests across both estimators."""

    def test_scm_vs_ascm_direction(self):
        """Both Python and Julia should show similar SCM vs ASCM relationship."""
        outcomes, treatment, treatment_period, _ = generate_scm_data(random_seed=404)

        # Python
        py_scm = synthetic_control(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            treatment_period=treatment_period - 1,
            inference="none",
        )
        py_ascm = augmented_synthetic_control(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            treatment_period=treatment_period - 1,
            inference="none",
        )

        # Julia
        jl_scm = julia_synthetic_control(
            outcomes=outcomes,
            treatment=treatment,
            treatment_period=treatment_period,
            inference="none",
        )
        jl_ascm = julia_augmented_scm(
            outcomes=outcomes,
            treatment=treatment,
            treatment_period=treatment_period,
            inference="none",
        )

        # Difference direction should be consistent
        py_diff = py_ascm["estimate"] - py_scm["estimate"]
        jl_diff = jl_ascm["estimate"] - jl_scm["estimate"]

        # Both should have similar adjustment direction (can be small noise)
        if abs(py_diff) > 0.5 and abs(jl_diff) > 0.5:
            assert np.sign(py_diff) == np.sign(jl_diff), (
                f"ASCM adjustment direction mismatch: Python={py_diff:.3f}, Julia={jl_diff:.3f}"
            )

    def test_known_effect_recovery(self):
        """Both implementations should recover known effect reasonably well."""
        true_effect = 5.0
        outcomes, treatment, treatment_period, _ = generate_scm_data(
            true_effect=true_effect, random_seed=505
        )

        py_result = synthetic_control(
            outcomes=outcomes,
            treatment=treatment.astype(int),
            treatment_period=treatment_period - 1,
            inference="none",
        )

        jl_result = julia_synthetic_control(
            outcomes=outcomes,
            treatment=treatment,
            treatment_period=treatment_period,
            inference="none",
        )

        # Both should be within 30% of true effect (DGP has some noise)
        py_bias = abs(py_result["estimate"] - true_effect) / true_effect
        jl_bias = abs(jl_result["estimate"] - true_effect) / true_effect

        assert py_bias < 0.30, f"Python bias {py_bias:.1%} > 30%"
        assert jl_bias < 0.30, f"Julia bias {jl_bias:.1%} > 30%"
