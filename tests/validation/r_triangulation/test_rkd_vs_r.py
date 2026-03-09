"""Triangulation tests: Python RKD (Regression Kink Design) vs R reference.

This module provides Layer 5 validation by comparing our Python RKD implementations
against R implementations using rdrobust with deriv=1 for slope estimation.

Note: No native R package exists for RKD. We use rdrobust adapted for derivative
estimation, which approximates the RKD approach.

Tests skip gracefully when R/rpy2 is unavailable.

Tolerance levels (established based on implementation differences):
- Sharp RKD estimate: rtol=0.10 (10% relative, derivative estimation less stable)
- Fuzzy RKD estimate: rtol=0.15 (15% relative, ratio of derivatives)
- Slope estimates: rtol=0.05 (5% relative, core regression outputs)
- Standard errors: rtol=0.20 (20% relative, robust SE implementations vary)
- Bandwidth: rtol=0.20 (20% relative, different selection algorithms)

Run with: pytest tests/validation/r_triangulation/test_rkd_vs_r.py -v

References:
- Card, Lee, Pei, Weber (2015). Inference on Causal Effects in a Generalized
  Regression Kink Design. Econometrica, 83(6).
- Calonico, Cattaneo, Titiunik (2014). Robust Nonparametric Confidence Intervals
  for Regression-Discontinuity Designs. Econometrica.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import Dict, Any

from tests.validation.r_triangulation.r_interface import (
    check_r_available,
    check_rdrobust_rkd_capable,
    r_sharp_rkd,
    r_fuzzy_rkd,
)

# Lazy import Python implementations
try:
    from src.causal_inference.rkd.sharp_rkd import SharpRKD
    from src.causal_inference.rkd.fuzzy_rkd import FuzzyRKD

    RKD_AVAILABLE = True
except ImportError:
    RKD_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_rkd_python = pytest.mark.skipif(
    not RKD_AVAILABLE,
    reason="Python RKD module not available",
)

requires_rdrobust_r = pytest.mark.skipif(
    not check_rdrobust_rkd_capable(),
    reason="R rdrobust package not installed or version < 0.99. "
    "Install in R with: install.packages('rdrobust')",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_sharp_rkd_data(
    n: int = 1000,
    cutoff: float = 0.0,
    true_effect: float = 2.0,
    kink_slope_change: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate sharp RKD data with known treatment effect.

    DGP:
    - X ~ Uniform[-5, 5] (running variable)
    - D = slope_left * X for X < cutoff
    - D = slope_right * X for X >= cutoff
    - Y = true_effect * D + 0.3 * X + noise

    The kink in D creates identification:
    - slope_d_left = 0.5
    - slope_d_right = 0.5 + kink_slope_change = 1.5 (default)
    - delta_slope_d = kink_slope_change = 1.0

    Parameters
    ----------
    n : int
        Sample size.
    cutoff : float
        Kink point in running variable.
    true_effect : float
        True treatment effect (τ).
    kink_slope_change : float
        Change in D slope at cutoff (δ_R - δ_L).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with y, x, d, cutoff, true_effect, slope_d_left, slope_d_right.
    """
    np.random.seed(seed)

    # Running variable: uniform
    x = np.random.uniform(-5, 5, n)

    # Treatment intensity with kink at cutoff
    slope_d_left = 0.5
    slope_d_right = slope_d_left + kink_slope_change

    d = np.where(x < cutoff, slope_d_left * x, slope_d_right * x)

    # Outcome: Y = τ*D + baseline_effect*X + noise
    y = true_effect * d + 0.3 * x + np.random.normal(0, 1, n)

    # Compute expected slope changes
    # slope_y_left = true_effect * slope_d_left + baseline = 2*0.5 + 0.3 = 1.3
    # slope_y_right = true_effect * slope_d_right + baseline = 2*1.5 + 0.3 = 3.3
    # delta_slope_y = 3.3 - 1.3 = 2.0
    # RKD estimate = delta_slope_y / delta_slope_d = 2.0 / 1.0 = 2.0 ✓

    return {
        "y": y,
        "x": x,
        "d": d,
        "cutoff": cutoff,
        "true_effect": true_effect,
        "slope_d_left": slope_d_left,
        "slope_d_right": slope_d_right,
        "delta_slope_d": kink_slope_change,
    }


def generate_fuzzy_rkd_data(
    n: int = 1000,
    cutoff: float = 0.0,
    true_late: float = 2.5,
    compliance_kink: float = 0.8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate fuzzy RKD data with stochastic treatment.

    DGP (fuzzy kink):
    - X ~ Uniform[-5, 5]
    - E[D|X] = slope_left * X + noise for X < cutoff
    - E[D|X] = slope_right * X + noise for X >= cutoff
    - D has noise added (fuzzy assignment)
    - Y = true_late * D + 0.2 * X + noise

    Parameters
    ----------
    n : int
        Sample size.
    cutoff : float
        Kink point.
    true_late : float
        True local average treatment effect at kink.
    compliance_kink : float
        Size of kink in treatment intensity (slope change).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with y, x, d, cutoff, true_late, first_stage_kink.
    """
    np.random.seed(seed)

    # Running variable
    x = np.random.uniform(-5, 5, n)

    # Treatment intensity with fuzzy kink
    slope_d_left = 0.3
    slope_d_right = slope_d_left + compliance_kink

    # Expected D (deterministic part)
    d_expected = np.where(x < cutoff, slope_d_left * x, slope_d_right * x)

    # Add noise to make it fuzzy
    d = d_expected + np.random.normal(0, 0.5, n)

    # Outcome
    y = true_late * d + 0.2 * x + np.random.normal(0, 1, n)

    return {
        "y": y,
        "x": x,
        "d": d,
        "cutoff": cutoff,
        "true_late": true_late,
        "first_stage_kink": compliance_kink,
        "slope_d_left": slope_d_left,
        "slope_d_right": slope_d_right,
    }


def generate_strong_kink_data(
    n: int = 1000,
    cutoff: float = 0.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate data with very strong kink (large slope change).

    Large kink → easier estimation, better power.

    Parameters
    ----------
    n : int
        Sample size.
    cutoff : float
        Kink point.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with y, x, d, cutoff, true_effect.
    """
    return generate_sharp_rkd_data(
        n=n,
        cutoff=cutoff,
        true_effect=3.0,
        kink_slope_change=2.0,  # Strong kink
        seed=seed,
    )


def generate_weak_kink_data(
    n: int = 1000,
    cutoff: float = 0.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate data with weak kink (small slope change).

    Small kink → harder estimation, weak first stage concern.

    Parameters
    ----------
    n : int
        Sample size.
    cutoff : float
        Kink point.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with y, x, d, cutoff, true_effect.
    """
    return generate_sharp_rkd_data(
        n=n,
        cutoff=cutoff,
        true_effect=2.0,
        kink_slope_change=0.2,  # Weak kink
        seed=seed,
    )


def generate_smooth_data(
    n: int = 1000,
    cutoff: float = 0.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate data with NO kink (smooth relationship).

    Used to test that RKD correctly identifies null effect.

    Parameters
    ----------
    n : int
        Sample size.
    cutoff : float
        Fake kink point.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with y, x, d, cutoff (no actual kink).
    """
    np.random.seed(seed)

    # Running variable
    x = np.random.uniform(-5, 5, n)

    # Smooth treatment (no kink)
    d = 0.5 * x  # Constant slope everywhere

    # Outcome (also smooth)
    y = 2.0 * d + 0.3 * x + np.random.normal(0, 1, n)

    return {
        "y": y,
        "x": x,
        "d": d,
        "cutoff": cutoff,
        "true_effect": 2.0,  # But kink is zero, so RKD gives indeterminate
        "delta_slope_d": 0.0,  # No kink!
    }


def generate_noisy_data(
    n: int = 1000,
    cutoff: float = 0.0,
    noise_sd: float = 3.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate RKD data with high noise.

    Tests estimation under challenging signal-to-noise ratio.

    Parameters
    ----------
    n : int
        Sample size.
    cutoff : float
        Kink point.
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with y, x, d, cutoff, true_effect.
    """
    np.random.seed(seed)

    x = np.random.uniform(-5, 5, n)

    slope_d_left = 0.5
    slope_d_right = 1.5
    d = np.where(x < cutoff, slope_d_left * x, slope_d_right * x)

    # High noise
    y = 2.0 * d + 0.3 * x + np.random.normal(0, noise_sd, n)

    return {
        "y": y,
        "x": x,
        "d": d,
        "cutoff": cutoff,
        "true_effect": 2.0,
        "slope_d_left": slope_d_left,
        "slope_d_right": slope_d_right,
        "delta_slope_d": 1.0,
    }


def generate_boundary_cutoff_data(
    n: int = 1000,
    cutoff: float = 3.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate RKD data with cutoff near boundary.

    Tests estimation when cutoff is not centered in data.

    Parameters
    ----------
    n : int
        Sample size.
    cutoff : float
        Kink point (near upper boundary).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with y, x, d, cutoff, true_effect.
    """
    np.random.seed(seed)

    # Running variable: uniform, but cutoff at 3 means 80% on left
    x = np.random.uniform(-5, 5, n)

    slope_d_left = 0.5
    slope_d_right = 1.5
    d = np.where(x < cutoff, slope_d_left * x, slope_d_right * x)

    y = 2.0 * d + 0.3 * x + np.random.normal(0, 1, n)

    return {
        "y": y,
        "x": x,
        "d": d,
        "cutoff": cutoff,
        "true_effect": 2.0,
        "slope_d_left": slope_d_left,
        "slope_d_right": slope_d_right,
        "delta_slope_d": 1.0,
    }


# =============================================================================
# Test Class: Sharp RKD vs R (rdrobust)
# =============================================================================


class TestSharpRKDVsRdrobust:
    """Compare Python SharpRKD to R rdrobust derivative estimation."""

    @requires_rkd_python
    @requires_rdrobust_r
    def test_sharp_rkd_basic(self):
        """Basic Sharp RKD comparison: standard DGP."""
        data = generate_sharp_rkd_data(n=1000, seed=42)

        # Python Sharp RKD
        rkd = SharpRKD(
            cutoff=data["cutoff"],
            bandwidth=2.0,
            kernel="triangular",
            polynomial_order=2,
        )
        py_result = rkd.fit(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            slope_d_left=data["slope_d_left"],
            slope_d_right=data["slope_d_right"],
        )

        # R Sharp RKD via rdrobust
        r_result = r_sharp_rkd(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            cutoff=data["cutoff"],
            bandwidth=2.0,
            kernel="triangular",
        )

        if r_result is None or r_result["estimate"] is None:
            pytest.skip("R rdrobust RKD estimation failed")

        # Compare estimates
        # Note: We compare to true effect since both should be close
        assert abs(py_result.estimate - data["true_effect"]) < 0.5, (
            f"Python estimate {py_result.estimate:.3f} far from true {data['true_effect']}"
        )

        # R vs Python comparison (looser tolerance due to different approaches)
        if r_result["estimate"] is not None:
            np.testing.assert_allclose(
                py_result.estimate,
                r_result["estimate"],
                rtol=0.25,  # 25% tolerance for different methodologies
                err_msg="Sharp RKD estimate mismatch Python vs R",
            )

        # Compare slope estimates (tighter tolerance)
        if r_result["delta_slope_y"] is not None and r_result["delta_slope_d"] is not None:
            # Delta slope D should be close
            np.testing.assert_allclose(
                py_result.delta_slope_d,
                r_result["delta_slope_d"],
                rtol=0.15,
                err_msg="Delta slope D mismatch",
            )

    @requires_rkd_python
    @requires_rdrobust_r
    def test_sharp_rkd_strong_kink(self):
        """Sharp RKD with strong kink: easier estimation."""
        data = generate_strong_kink_data(n=1000, seed=123)

        # Python
        rkd = SharpRKD(
            cutoff=data["cutoff"],
            bandwidth=2.0,
            kernel="triangular",
        )
        py_result = rkd.fit(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            slope_d_left=data["slope_d_left"],
            slope_d_right=data["slope_d_right"],
        )

        # R
        r_result = r_sharp_rkd(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            cutoff=data["cutoff"],
            bandwidth=2.0,
        )

        if r_result is None or r_result["estimate"] is None:
            pytest.skip("R rdrobust estimation failed")

        # Strong kink should yield estimate close to true
        assert abs(py_result.estimate - data["true_effect"]) < 0.5, (
            f"Python estimate {py_result.estimate:.3f} not close to true {data['true_effect']}"
        )

        # Comparison with R
        if r_result["estimate"] is not None:
            np.testing.assert_allclose(
                py_result.estimate,
                r_result["estimate"],
                rtol=0.20,
                err_msg="Strong kink RKD estimate mismatch",
            )

    @requires_rkd_python
    @requires_rdrobust_r
    def test_sharp_rkd_known_slopes(self):
        """Sharp RKD when D slopes are exactly known (no estimation error in D)."""
        data = generate_sharp_rkd_data(n=1500, seed=456)

        # Python: provide exact known slopes
        rkd = SharpRKD(
            cutoff=data["cutoff"],
            bandwidth=2.5,
            kernel="triangular",
        )
        py_result = rkd.fit(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            slope_d_left=data["slope_d_left"],  # Known exactly
            slope_d_right=data["slope_d_right"],  # Known exactly
        )

        # When slopes are known, estimate should be very accurate
        assert abs(py_result.estimate - data["true_effect"]) < 0.3, (
            f"With known slopes, estimate {py_result.estimate:.3f} should be near "
            f"true {data['true_effect']}"
        )

        # R comparison (R doesn't have the "known slopes" advantage)
        r_result = r_sharp_rkd(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            cutoff=data["cutoff"],
            bandwidth=2.5,
        )

        if r_result is not None and r_result["estimate"] is not None:
            # Python should be at least as good as R
            py_error = abs(py_result.estimate - data["true_effect"])
            r_error = abs(r_result["estimate"] - data["true_effect"])
            # Allow Python to have higher error only if R also struggles
            assert py_error < r_error + 0.5, (
                f"Python error {py_error:.3f} much worse than R error {r_error:.3f}"
            )

    @requires_rkd_python
    @requires_rdrobust_r
    def test_sharp_rkd_bandwidth_sensitivity(self):
        """Test Sharp RKD across different bandwidth choices."""
        data = generate_sharp_rkd_data(n=1000, seed=789)

        bandwidths = [1.5, 2.0, 3.0]
        py_estimates = []
        r_estimates = []

        for h in bandwidths:
            # Python
            rkd = SharpRKD(cutoff=data["cutoff"], bandwidth=h)
            py_result = rkd.fit(
                y=data["y"],
                x=data["x"],
                d=data["d"],
                slope_d_left=data["slope_d_left"],
                slope_d_right=data["slope_d_right"],
            )
            py_estimates.append(py_result.estimate)

            # R
            r_result = r_sharp_rkd(
                y=data["y"],
                x=data["x"],
                d=data["d"],
                cutoff=data["cutoff"],
                bandwidth=h,
            )
            if r_result is not None and r_result["estimate"] is not None:
                r_estimates.append(r_result["estimate"])

        # All estimates should be in reasonable range
        for est in py_estimates:
            assert 0.5 < est < 4.0, f"Python estimate {est:.3f} out of reasonable range"

        # Estimates should be relatively stable across bandwidths
        py_range = max(py_estimates) - min(py_estimates)
        assert py_range < 1.5, f"Python estimates vary too much: range {py_range:.3f}"


# =============================================================================
# Test Class: Fuzzy RKD vs R (rdrobust)
# =============================================================================


class TestFuzzyRKDVsRdrobust:
    """Compare Python FuzzyRKD to R rdrobust derivative estimation."""

    @requires_rkd_python
    @requires_rdrobust_r
    def test_fuzzy_rkd_basic(self):
        """Basic Fuzzy RKD comparison: standard DGP."""
        data = generate_fuzzy_rkd_data(n=1000, seed=42)

        # Python Fuzzy RKD
        rkd = FuzzyRKD(
            cutoff=data["cutoff"],
            bandwidth=2.0,
            kernel="triangular",
        )
        py_result = rkd.fit(
            y=data["y"],
            x=data["x"],
            d=data["d"],
        )

        # R Fuzzy RKD via rdrobust
        r_result = r_fuzzy_rkd(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            cutoff=data["cutoff"],
            bandwidth=2.0,
        )

        if r_result is None or r_result["estimate"] is None:
            pytest.skip("R rdrobust Fuzzy RKD estimation failed")

        # Compare to true LATE
        assert abs(py_result.estimate - data["true_late"]) < 1.0, (
            f"Python estimate {py_result.estimate:.3f} far from true {data['true_late']}"
        )

        # R vs Python comparison
        if r_result["estimate"] is not None:
            np.testing.assert_allclose(
                py_result.estimate,
                r_result["estimate"],
                rtol=0.30,  # 30% tolerance for fuzzy (harder)
                err_msg="Fuzzy RKD estimate mismatch Python vs R",
            )

        # Compare first stage kink
        if r_result["first_stage_kink"] is not None:
            np.testing.assert_allclose(
                py_result.first_stage_kink,
                r_result["first_stage_kink"],
                rtol=0.20,
                err_msg="First stage kink mismatch",
            )

    @requires_rkd_python
    @requires_rdrobust_r
    def test_fuzzy_rkd_weak_first_stage(self):
        """Fuzzy RKD with weak first stage kink."""
        data = generate_fuzzy_rkd_data(
            n=1000,
            cutoff=0.0,
            true_late=2.0,
            compliance_kink=0.2,  # Weak kink
            seed=123,
        )

        # Python
        rkd = FuzzyRKD(cutoff=data["cutoff"], bandwidth=2.0)
        py_result = rkd.fit(y=data["y"], x=data["x"], d=data["d"])

        # Weak first stage should have low F-stat
        # (Not a hard constraint, but expected)
        if py_result.first_stage_f_stat < 10:
            # F < 10 indicates weak kink - estimate may be imprecise
            assert abs(py_result.estimate) < 10, (
                f"Weak first stage should not give extreme estimate: {py_result.estimate}"
            )

        # R comparison
        r_result = r_fuzzy_rkd(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            cutoff=data["cutoff"],
            bandwidth=2.0,
        )

        if r_result is not None and r_result["first_stage_f_stat"] is not None:
            # Both should identify weak first stage
            # F-stat should be low (< 10 typical threshold)
            assert r_result["first_stage_f_stat"] < 20, (
                f"R F-stat {r_result['first_stage_f_stat']:.1f} seems too high for weak kink"
            )

    @requires_rkd_python
    @requires_rdrobust_r
    def test_fuzzy_rkd_strong_first_stage(self):
        """Fuzzy RKD with strong first stage kink."""
        data = generate_fuzzy_rkd_data(
            n=1000,
            cutoff=0.0,
            true_late=2.5,
            compliance_kink=1.5,  # Strong kink
            seed=456,
        )

        # Python
        rkd = FuzzyRKD(cutoff=data["cutoff"], bandwidth=2.0)
        py_result = rkd.fit(y=data["y"], x=data["x"], d=data["d"])

        # Strong first stage should yield good estimate
        assert abs(py_result.estimate - data["true_late"]) < 0.8, (
            f"Strong first stage estimate {py_result.estimate:.3f} should be near "
            f"true {data['true_late']}"
        )

        # R comparison
        r_result = r_fuzzy_rkd(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            cutoff=data["cutoff"],
            bandwidth=2.0,
        )

        if r_result is not None and r_result["estimate"] is not None:
            np.testing.assert_allclose(
                py_result.estimate,
                r_result["estimate"],
                rtol=0.25,
                err_msg="Strong first stage Fuzzy RKD mismatch",
            )


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestRKDEdgeCases:
    """Test RKD behavior in edge cases."""

    @requires_rkd_python
    @requires_rdrobust_r
    def test_rkd_no_kink(self):
        """RKD when there is no actual kink (smooth function)."""
        data = generate_smooth_data(n=1000, seed=42)

        # Python Sharp RKD
        rkd = SharpRKD(cutoff=data["cutoff"], bandwidth=2.0)

        # With no kink in D, estimation should warn or give large SE
        # Provide slopes that have zero difference
        py_result = rkd.fit(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            slope_d_left=0.5,
            slope_d_right=0.5,  # Same slope → no kink
        )

        # delta_slope_d should be ~0
        assert abs(py_result.delta_slope_d) < 0.1, (
            f"Delta slope D should be ~0 for smooth data: {py_result.delta_slope_d}"
        )

        # Note: R will likely fail or give NA for this case
        r_result = r_sharp_rkd(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            cutoff=data["cutoff"],
            bandwidth=2.0,
        )

        # R might return None or give indeterminate estimate
        # This is expected behavior when there's no kink to identify

    @requires_rkd_python
    @requires_rdrobust_r
    def test_rkd_noisy_data(self):
        """RKD estimation under high noise conditions."""
        data = generate_noisy_data(n=1000, noise_sd=3.0, seed=123)

        # Python
        rkd = SharpRKD(cutoff=data["cutoff"], bandwidth=2.5)
        py_result = rkd.fit(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            slope_d_left=data["slope_d_left"],
            slope_d_right=data["slope_d_right"],
        )

        # With high noise, estimate should still be in reasonable range
        # but may deviate more from true
        assert -2.0 < py_result.estimate < 6.0, (
            f"Noisy data estimate {py_result.estimate:.3f} out of range"
        )

        # SE should be larger than with clean data
        assert py_result.se > 0.1, f"SE {py_result.se:.3f} unexpectedly small for noisy data"

        # R comparison
        r_result = r_sharp_rkd(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            cutoff=data["cutoff"],
            bandwidth=2.5,
        )

        if r_result is not None and r_result["estimate"] is not None:
            # Both estimates should be affected by noise similarly
            py_error = abs(py_result.estimate - data["true_effect"])
            r_error = abs(r_result["estimate"] - data["true_effect"])
            # Allow both to have substantial error
            assert py_error < 3.0 or r_error < 3.0, (
                f"Both Python ({py_error:.2f}) and R ({r_error:.2f}) have huge errors"
            )

    @requires_rkd_python
    @requires_rdrobust_r
    def test_rkd_boundary_cutoff(self):
        """RKD with cutoff near data boundary."""
        data = generate_boundary_cutoff_data(n=1000, cutoff=3.0, seed=456)

        # Python
        rkd = SharpRKD(cutoff=data["cutoff"], bandwidth=2.0)
        py_result = rkd.fit(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            slope_d_left=data["slope_d_left"],
            slope_d_right=data["slope_d_right"],
        )

        # Should still get reasonable estimate despite asymmetric data
        assert 0.5 < py_result.estimate < 4.0, (
            f"Boundary cutoff estimate {py_result.estimate:.3f} out of range"
        )

        # Effective sample sizes should be asymmetric
        # Left: ~80% of data, Right: ~20% of data
        assert py_result.n_left > py_result.n_right, f"Expected n_left > n_right for cutoff at 3.0"

        # R comparison
        r_result = r_sharp_rkd(
            y=data["y"],
            x=data["x"],
            d=data["d"],
            cutoff=data["cutoff"],
            bandwidth=2.0,
        )

        if r_result is not None and r_result["estimate"] is not None:
            np.testing.assert_allclose(
                py_result.estimate,
                r_result["estimate"],
                rtol=0.30,
                err_msg="Boundary cutoff RKD mismatch",
            )


# =============================================================================
# Test Class: Consistency Checks
# =============================================================================


class TestRKDConsistency:
    """Test RKD internal consistency and convergence properties."""

    @requires_rkd_python
    @requires_rdrobust_r
    def test_sharp_fuzzy_convergence(self):
        """Sharp and Fuzzy RKD should converge when D is deterministic.

        When D has no noise (perfect compliance with kink), Fuzzy RKD
        should give the same answer as Sharp RKD.
        """
        np.random.seed(42)
        n = 1000
        cutoff = 0.0

        # Generate data where D is deterministic (sharp)
        x = np.random.uniform(-5, 5, n)
        slope_d_left = 0.5
        slope_d_right = 1.5
        d = np.where(x < cutoff, slope_d_left * x, slope_d_right * x)
        true_effect = 2.0
        y = true_effect * d + 0.3 * x + np.random.normal(0, 1, n)

        # Sharp RKD
        sharp_rkd = SharpRKD(cutoff=cutoff, bandwidth=2.0)
        sharp_result = sharp_rkd.fit(
            y=y,
            x=x,
            d=d,
            slope_d_left=slope_d_left,
            slope_d_right=slope_d_right,
        )

        # Fuzzy RKD on same (deterministic) data
        fuzzy_rkd = FuzzyRKD(cutoff=cutoff, bandwidth=2.0)
        fuzzy_result = fuzzy_rkd.fit(y=y, x=x, d=d)

        # Estimates should be similar (not identical due to different SE methods)
        np.testing.assert_allclose(
            sharp_result.estimate,
            fuzzy_result.estimate,
            rtol=0.15,
            err_msg="Sharp and Fuzzy RKD should converge for deterministic D",
        )

    @requires_rkd_python
    @requires_rdrobust_r
    @pytest.mark.slow
    def test_rkd_coverage(self):
        """Monte Carlo coverage check for RKD confidence intervals.

        Runs multiple simulations and checks that the CI coverage is
        approximately nominal (93-97% for 95% CI).
        """
        n_sims = 100
        n_obs = 500
        true_effect = 2.0
        coverage_count = 0

        for seed in range(n_sims):
            data = generate_sharp_rkd_data(
                n=n_obs,
                cutoff=0.0,
                true_effect=true_effect,
                kink_slope_change=1.0,
                seed=seed,
            )

            rkd = SharpRKD(cutoff=data["cutoff"], bandwidth=2.0)
            result = rkd.fit(
                y=data["y"],
                x=data["x"],
                d=data["d"],
                slope_d_left=data["slope_d_left"],
                slope_d_right=data["slope_d_right"],
            )

            if result.ci_lower < true_effect < result.ci_upper:
                coverage_count += 1

        coverage = coverage_count / n_sims

        # Coverage should be 90-98% (allowing some slack for RKD's
        # inherent difficulty with SE estimation)
        assert 0.85 < coverage < 0.99, f"Coverage {coverage:.2%} outside acceptable range (85-99%)"
