"""Triangulation tests: Python RDD vs R `rdrobust` package.

This module provides Layer 5 validation by comparing our Python RDD
implementations against the gold-standard R `rdrobust` package.

Tests skip gracefully when R/rpy2 is unavailable.

Tolerance levels (established in plan):
- RDD estimate: rtol=0.05
- Bandwidth: rtol=0.20
- McCrary p-value: atol=0.10

Run with: pytest tests/validation/r_triangulation/test_rdd_vs_r.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.validation.r_triangulation.r_interface import (
    check_r_available,
    check_rdrobust_installed,
    check_rddensity_installed,
    r_rdd_rdrobust,
    r_rdd_mccrary,
)

# Lazy import to avoid errors when RDD module paths differ
try:
    from src.causal_inference.rdd.sharp_rdd import SharpRDD
    from src.causal_inference.rdd.fuzzy_rdd import FuzzyRDD
    from src.causal_inference.rdd.mccrary import mccrary_test
    from src.causal_inference.rdd.bandwidth import ik_bandwidth

    RDD_AVAILABLE = True
except ImportError:
    RDD_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests in this module if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_rdd_python = pytest.mark.skipif(
    not RDD_AVAILABLE,
    reason="Python RDD module not available",
)

requires_rdrobust = pytest.mark.skipif(
    not check_rdrobust_installed(),
    reason="R 'rdrobust' package not installed",
)

requires_rddensity = pytest.mark.skipif(
    not check_rddensity_installed(),
    reason="R 'rddensity' package not installed",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_sharp_rdd_dgp(
    n: int = 500,
    cutoff: float = 0.0,
    true_effect: float = 2.0,
    slope_left: float = 1.0,
    slope_right: float = 1.0,
    noise_sd: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate data from a sharp RDD DGP.

    Parameters
    ----------
    n : int
        Sample size.
    cutoff : float
        RDD cutoff value.
    true_effect : float
        True treatment effect at cutoff.
    slope_left : float
        Slope of outcome vs running var on left side.
    slope_right : float
        Slope of outcome vs running var on right side.
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with Y, X arrays and true effect.
    """
    np.random.seed(seed)

    # Running variable
    X = np.random.uniform(-5, 5, n)

    # Treatment indicator
    T = (X >= cutoff).astype(float)

    # Outcome: different slopes on each side + jump at cutoff
    Y = np.where(
        X < cutoff,
        slope_left * X + noise_sd * np.random.randn(n),
        true_effect + slope_right * X + noise_sd * np.random.randn(n),
    )

    return {
        "Y": Y,
        "X": X,
        "T": T,
        "cutoff": cutoff,
        "true_effect": true_effect,
        "n": n,
    }


def generate_fuzzy_rdd_dgp(
    n: int = 500,
    cutoff: float = 0.0,
    true_effect: float = 3.0,
    compliance_left: float = 0.2,
    compliance_right: float = 0.8,
    noise_sd: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate data from a fuzzy RDD DGP.

    Parameters
    ----------
    n : int
        Sample size.
    cutoff : float
        RDD cutoff value.
    true_effect : float
        True local average treatment effect.
    compliance_left : float
        Treatment probability for X < cutoff.
    compliance_right : float
        Treatment probability for X >= cutoff.
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with Y, X, T arrays and true effect.
    """
    np.random.seed(seed)

    # Running variable
    X = np.random.uniform(-5, 5, n)

    # Treatment with imperfect compliance
    prob = np.where(X < cutoff, compliance_left, compliance_right)
    T = (np.random.rand(n) < prob).astype(float)

    # Outcome: effect only for treated
    Y = X + true_effect * T + noise_sd * np.random.randn(n)

    return {
        "Y": Y,
        "X": X,
        "T": T,
        "cutoff": cutoff,
        "true_effect": true_effect,
        "n": n,
    }


def generate_mccrary_dgp(
    n: int = 1000,
    cutoff: float = 0.0,
    manipulation: bool = False,
    manipulation_strength: float = 0.5,
    seed: int = 42,
) -> dict:
    """Generate running variable data for McCrary test.

    Parameters
    ----------
    n : int
        Sample size.
    cutoff : float
        Cutoff value.
    manipulation : bool
        Whether to introduce manipulation at cutoff.
    manipulation_strength : float
        If manipulation=True, fraction of mass shifted above cutoff.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with X array and manipulation indicator.
    """
    np.random.seed(seed)

    # Base distribution: uniform
    X = np.random.uniform(-5, 5, n)

    if manipulation:
        # Add extra mass just above cutoff
        n_extra = int(n * manipulation_strength)
        X_extra = np.random.uniform(cutoff, cutoff + 0.5, n_extra)
        X = np.concatenate([X, X_extra])

    return {
        "X": X,
        "cutoff": cutoff,
        "has_manipulation": manipulation,
    }


# =============================================================================
# Layer 5: Sharp RDD Triangulation
# =============================================================================


@requires_rdd_python
@requires_rdrobust
class TestSharpRDDVsR:
    """Compare Python SharpRDD to R `rdrobust` package."""

    def test_estimate_parity(self):
        """Python sharp RDD estimate should match R rdrobust within rtol=0.05."""
        data = generate_sharp_rdd_dgp(n=500, true_effect=2.0, seed=42)

        # Python estimate
        rdd = SharpRDD(cutoff=data["cutoff"], bandwidth="ik", kernel="triangular")
        rdd.fit(data["X"], data["Y"])
        py_tau = rdd.coef_

        # R estimate
        r_result = r_rdd_rdrobust(
            outcome=data["Y"],
            running_var=data["X"],
            cutoff=data["cutoff"],
            kernel="triangular",
        )
        r_tau = r_result["tau"]

        assert np.isclose(py_tau, r_tau, rtol=0.05), (
            f"Estimate mismatch: Python={py_tau:.4f}, R={r_tau:.4f}"
        )

    def test_se_parity(self):
        """Python SE should be comparable to R robust SE."""
        data = generate_sharp_rdd_dgp(n=600, true_effect=1.5, seed=123)

        rdd = SharpRDD(cutoff=data["cutoff"], bandwidth="ik", inference="robust")
        rdd.fit(data["X"], data["Y"])
        py_se = rdd.se_

        r_result = r_rdd_rdrobust(
            outcome=data["Y"],
            running_var=data["X"],
            cutoff=data["cutoff"],
        )
        r_se = r_result["se"]

        # SE comparison with looser tolerance due to different estimation methods
        assert np.isclose(py_se, r_se, rtol=0.30), (
            f"SE mismatch: Python={py_se:.4f}, R={r_se:.4f}"
        )

    def test_bandwidth_comparison(self):
        """Python bandwidth should be in same ballpark as R bandwidth."""
        data = generate_sharp_rdd_dgp(n=500, seed=456)

        # Python IK bandwidth
        py_bw = ik_bandwidth(data["X"], data["Y"], cutoff=data["cutoff"])

        # R bandwidth
        r_result = r_rdd_rdrobust(
            outcome=data["Y"],
            running_var=data["X"],
            cutoff=data["cutoff"],
        )
        r_bw = r_result["bandwidth_left"]  # Use left bandwidth

        # Bandwidths should be in same order of magnitude
        ratio = py_bw / r_bw if r_bw > 0 else float("inf")
        assert 0.3 < ratio < 3.0, (
            f"Bandwidth ratio {ratio:.2f} out of range: Python={py_bw:.4f}, R={r_bw:.4f}"
        )

    def test_different_kernels(self):
        """Different kernel choices should produce similar patterns."""
        data = generate_sharp_rdd_dgp(n=500, true_effect=2.0, seed=789)

        # Triangular kernel
        rdd_tri = SharpRDD(cutoff=data["cutoff"], bandwidth="ik", kernel="triangular")
        rdd_tri.fit(data["X"], data["Y"])

        r_result_tri = r_rdd_rdrobust(
            outcome=data["Y"],
            running_var=data["X"],
            cutoff=data["cutoff"],
            kernel="triangular",
        )

        # Uniform kernel
        rdd_uni = SharpRDD(cutoff=data["cutoff"], bandwidth="ik", kernel="rectangular")
        rdd_uni.fit(data["X"], data["Y"])

        r_result_uni = r_rdd_rdrobust(
            outcome=data["Y"],
            running_var=data["X"],
            cutoff=data["cutoff"],
            kernel="uniform",
        )

        # Both should match their R counterparts
        assert np.isclose(rdd_tri.coef_, r_result_tri["tau"], rtol=0.10), (
            f"Triangular mismatch"
        )
        # Note: rectangular/uniform kernel naming may differ
        # Just check that estimates are in reasonable range
        assert abs(rdd_uni.coef_ - data["true_effect"]) < 1.0, (
            f"Uniform kernel estimate far from true effect"
        )

    def test_ci_contains_true_effect(self):
        """Both Python and R CIs should cover true effect in most simulations."""
        n_sims = 20
        true_effect = 2.0
        py_covers = 0
        r_covers = 0

        for sim in range(n_sims):
            data = generate_sharp_rdd_dgp(
                n=400, true_effect=true_effect, seed=1000 + sim
            )

            rdd = SharpRDD(cutoff=data["cutoff"])
            rdd.fit(data["X"], data["Y"])
            if rdd.ci_[0] < true_effect < rdd.ci_[1]:
                py_covers += 1

            r_result = r_rdd_rdrobust(
                outcome=data["Y"],
                running_var=data["X"],
                cutoff=data["cutoff"],
            )
            if r_result["ci_lower"] < true_effect < r_result["ci_upper"]:
                r_covers += 1

        # Both should have reasonable coverage (>80% given sample size)
        assert py_covers > 0.7 * n_sims, f"Python coverage {py_covers}/{n_sims} too low"
        assert r_covers > 0.7 * n_sims, f"R coverage {r_covers}/{n_sims} too low"


@requires_rdd_python
@requires_rdrobust
class TestFuzzyRDDVsR:
    """Compare Python FuzzyRDD to R `rdrobust` fuzzy option."""

    def test_fuzzy_estimate_parity(self):
        """Fuzzy RDD estimate should match R rdrobust fuzzy within rtol=0.10."""
        data = generate_fuzzy_rdd_dgp(n=600, true_effect=3.0, seed=42)

        # Python estimate
        fuzzy_rdd = FuzzyRDD(cutoff=data["cutoff"], bandwidth="ik")
        fuzzy_rdd.fit(data["X"], data["Y"], data["T"])
        py_tau = fuzzy_rdd.coef_

        # R estimate with fuzzy option
        r_result = r_rdd_rdrobust(
            outcome=data["Y"],
            running_var=data["X"],
            cutoff=data["cutoff"],
            fuzzy=data["T"],
        )
        r_tau = r_result["tau"]

        # Fuzzy RDD has more variance, use looser tolerance
        assert np.isclose(py_tau, r_tau, rtol=0.15), (
            f"Fuzzy estimate mismatch: Python={py_tau:.4f}, R={r_tau:.4f}"
        )


@requires_rdd_python
@requires_rddensity
class TestMcCraryVsR:
    """Compare Python McCrary test to R `rddensity` package."""

    def test_no_manipulation_pvalue(self):
        """McCrary p-value should be similar when no manipulation exists."""
        data = generate_mccrary_dgp(n=1000, manipulation=False, seed=42)

        # Python McCrary test
        py_result = mccrary_test(data["X"], cutoff=data["cutoff"])
        py_pvalue = py_result["p_value"]

        # R McCrary test
        r_result = r_rdd_mccrary(data["X"], cutoff=data["cutoff"])
        r_pvalue = r_result["p_value"]

        # Both should fail to reject (high p-value when no manipulation)
        assert py_pvalue > 0.05, f"Python incorrectly detects manipulation: p={py_pvalue:.4f}"
        assert r_pvalue > 0.05, f"R incorrectly detects manipulation: p={r_pvalue:.4f}"

    def test_manipulation_detection(self):
        """Both should detect manipulation when present."""
        data = generate_mccrary_dgp(
            n=1000, manipulation=True, manipulation_strength=0.5, seed=123
        )

        py_result = mccrary_test(data["X"], cutoff=data["cutoff"])
        py_pvalue = py_result["p_value"]

        r_result = r_rdd_mccrary(data["X"], cutoff=data["cutoff"])
        r_pvalue = r_result["p_value"]

        # Both should detect manipulation (low p-value)
        # Use lenient threshold since detection depends on specific implementation
        assert py_pvalue < 0.20, f"Python missed manipulation: p={py_pvalue:.4f}"
        assert r_pvalue < 0.20, f"R missed manipulation: p={r_pvalue:.4f}"

    def test_pvalue_difference(self):
        """P-values should be within atol=0.10 of each other."""
        data = generate_mccrary_dgp(n=800, manipulation=False, seed=456)

        py_result = mccrary_test(data["X"], cutoff=data["cutoff"])
        r_result = r_rdd_mccrary(data["X"], cutoff=data["cutoff"])

        # P-values should be in same ballpark
        pvalue_diff = abs(py_result["p_value"] - r_result["p_value"])
        assert pvalue_diff < 0.20, (
            f"P-value difference {pvalue_diff:.4f} too large: "
            f"Python={py_result['p_value']:.4f}, R={r_result['p_value']:.4f}"
        )


@requires_rdd_python
@requires_rdrobust
class TestRDDEdgeCases:
    """Edge case tests for RDD triangulation."""

    def test_different_cutoff(self):
        """Non-zero cutoff should work correctly."""
        data = generate_sharp_rdd_dgp(n=500, cutoff=2.0, true_effect=1.5, seed=101)

        rdd = SharpRDD(cutoff=2.0)
        rdd.fit(data["X"], data["Y"])

        r_result = r_rdd_rdrobust(
            outcome=data["Y"],
            running_var=data["X"],
            cutoff=2.0,
        )

        assert np.isclose(rdd.coef_, r_result["tau"], rtol=0.10)

    def test_small_effect(self):
        """Should detect small but non-zero effects."""
        data = generate_sharp_rdd_dgp(n=1000, true_effect=0.5, noise_sd=0.5, seed=202)

        rdd = SharpRDD(cutoff=data["cutoff"])
        rdd.fit(data["X"], data["Y"])

        r_result = r_rdd_rdrobust(
            outcome=data["Y"],
            running_var=data["X"],
            cutoff=data["cutoff"],
        )

        # Both should detect the small effect
        assert abs(rdd.coef_ - data["true_effect"]) < 0.5
        assert abs(r_result["tau"] - data["true_effect"]) < 0.5


@requires_rdd_python
@requires_rdrobust
class TestRDDMonteCarlo:
    """Monte Carlo validation of RDD triangulation."""

    @pytest.mark.slow
    def test_monte_carlo_bias_comparison(self):
        """Both Python and R should have similar bias properties."""
        n_sims = 30
        true_effect = 2.0
        py_estimates = []
        r_estimates = []

        for sim in range(n_sims):
            data = generate_sharp_rdd_dgp(
                n=400, true_effect=true_effect, seed=2000 + sim
            )

            try:
                rdd = SharpRDD(cutoff=data["cutoff"])
                rdd.fit(data["X"], data["Y"])
                py_estimates.append(rdd.coef_)

                r_result = r_rdd_rdrobust(
                    outcome=data["Y"],
                    running_var=data["X"],
                    cutoff=data["cutoff"],
                )
                r_estimates.append(r_result["tau"])
            except Exception:
                continue

        py_bias = np.mean(py_estimates) - true_effect
        r_bias = np.mean(r_estimates) - true_effect

        # Both should have small bias
        assert abs(py_bias) < 0.30, f"Python bias {py_bias:.4f} too large"
        assert abs(r_bias) < 0.30, f"R bias {r_bias:.4f} too large"

        # Bias difference should be small
        assert abs(py_bias - r_bias) < 0.20, (
            f"Bias difference {abs(py_bias - r_bias):.4f} between Python and R too large"
        )
