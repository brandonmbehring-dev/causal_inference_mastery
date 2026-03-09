"""
Python↔Julia RKD parity tests.

Validates that Python and Julia RKD implementations produce consistent results
on the same data.

Session 74: Cross-language validation for Regression Kink Design.
"""

import numpy as np
import pytest
from typing import Tuple

# Python implementations
from src.causal_inference.rkd import SharpRKD

# Julia interface
from tests.validation.cross_language.julia_interface import (
    is_julia_available,
    julia_sharp_rkd,
)

# Skip all tests if Julia not available
pytestmark = pytest.mark.skipif(
    not is_julia_available(), reason="Julia not available for cross-validation"
)


# =============================================================================
# DGP Functions
# =============================================================================


def generate_rkd_data(
    n: int = 1000,
    cutoff: float = 0.0,
    slope_left_d: float = 0.5,
    slope_right_d: float = 1.5,
    true_effect: float = 2.0,
    noise_y: float = 1.0,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Sharp RKD data with known effect.

    Creates a proper kink at the cutoff where D is continuous but
    the slope of D changes from slope_left_d to slope_right_d.
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.uniform(-5, 5, n)

    # Treatment with kink at cutoff
    # D must be continuous at cutoff: use (X - cutoff) to ensure D(cutoff) = 0 from both sides
    D = np.where(X < cutoff, slope_left_d * (X - cutoff), slope_right_d * (X - cutoff))

    # Outcome
    Y = true_effect * D + np.random.normal(0, noise_y, n)

    return Y, X, D


# =============================================================================
# Sharp RKD Parity Tests
# =============================================================================


class TestSharpRKDParity:
    """Cross-language parity tests for Sharp RKD."""

    def test_basic_rkd_parity(self):
        """Python and Julia Sharp RKD should produce similar results."""
        Y, X, D = generate_rkd_data(n=1000, true_effect=2.0, seed=42)
        bandwidth = 2.5

        # Python
        py_rkd = SharpRKD(cutoff=0.0, bandwidth=bandwidth, polynomial_order=1)
        py_result = py_rkd.fit(Y, X, D)

        # Julia
        jl_result = julia_sharp_rkd(
            Y, X, D, cutoff=0.0, bandwidth=bandwidth, polynomial_order=1, kernel="triangular"
        )

        # Compare estimates (allow some tolerance due to implementation differences)
        assert np.isclose(py_result.estimate, jl_result["estimate"], rtol=0.15), (
            f"Estimate mismatch: Python={py_result.estimate:.4f}, Julia={jl_result['estimate']:.4f}"
        )

        # Compare treatment kink detection (Python uses delta_slope_d, Julia uses treatment_kink)
        assert np.isclose(py_result.delta_slope_d, jl_result["treatment_kink"], rtol=0.15), (
            f"Treatment kink mismatch: Python={py_result.delta_slope_d:.4f}, "
            f"Julia={jl_result['treatment_kink']:.4f}"
        )

    def test_outcome_kink_parity(self):
        """Outcome kink estimates should match."""
        Y, X, D = generate_rkd_data(n=1500, true_effect=3.0, seed=123)
        bandwidth = 2.0

        py_rkd = SharpRKD(cutoff=0.0, bandwidth=bandwidth)
        py_result = py_rkd.fit(Y, X, D)

        jl_result = julia_sharp_rkd(Y, X, D, cutoff=0.0, bandwidth=bandwidth)

        # Outcome kink should be similar (Python uses delta_slope_y, Julia uses outcome_kink)
        assert np.isclose(py_result.delta_slope_y, jl_result["outcome_kink"], rtol=0.20), (
            f"Outcome kink mismatch: Python={py_result.delta_slope_y:.4f}, "
            f"Julia={jl_result['outcome_kink']:.4f}"
        )

    def test_se_parity(self):
        """Standard errors should be similar.

        Note: Python uses simplified delta method (assumes D slopes known),
        while Julia uses full delta method (accounts for D slope variance).
        This leads to systematic differences where Julia SE > Python SE.
        We test that the ratio is reasonable rather than exact match.
        """
        Y, X, D = generate_rkd_data(n=1200, seed=456)
        bandwidth = 2.5

        py_rkd = SharpRKD(cutoff=0.0, bandwidth=bandwidth)
        py_result = py_rkd.fit(Y, X, D)

        jl_result = julia_sharp_rkd(Y, X, D, cutoff=0.0, bandwidth=bandwidth)

        # SEs should be in the same order of magnitude
        # Julia SE typically larger due to accounting for D slope variance
        se_ratio = jl_result["se"] / py_result.se
        assert 0.1 < se_ratio < 10.0, (
            f"SE ratio out of bounds: Python={py_result.se:.4f}, "
            f"Julia={jl_result['se']:.4f}, ratio={se_ratio:.2f}"
        )

        # Both should give similar inference conclusions
        # (both significant or both not significant at 5% level)
        py_significant = py_result.p_value < 0.05
        jl_significant = jl_result["p_value"] < 0.05
        assert py_significant == jl_significant, (
            f"Inference mismatch: Python p={py_result.p_value:.4f}, "
            f"Julia p={jl_result['p_value']:.4f}"
        )

    def test_ci_overlap(self):
        """Confidence intervals should overlap significantly."""
        Y, X, D = generate_rkd_data(n=1000, seed=789)
        bandwidth = 2.0

        py_rkd = SharpRKD(cutoff=0.0, bandwidth=bandwidth)
        py_result = py_rkd.fit(Y, X, D)

        jl_result = julia_sharp_rkd(Y, X, D, cutoff=0.0, bandwidth=bandwidth)

        # CIs should overlap
        py_lower, py_upper = py_result.ci_lower, py_result.ci_upper
        jl_lower, jl_upper = jl_result["ci_lower"], jl_result["ci_upper"]

        # Check overlap
        overlap_lower = max(py_lower, jl_lower)
        overlap_upper = min(py_upper, jl_upper)
        has_overlap = overlap_lower < overlap_upper

        assert has_overlap, (
            f"CIs don't overlap: Python=[{py_lower:.4f}, {py_upper:.4f}], "
            f"Julia=[{jl_lower:.4f}, {jl_upper:.4f}]"
        )

    def test_quadratic_polynomial_parity(self):
        """Quadratic polynomial order should also match."""
        Y, X, D = generate_rkd_data(n=1200, seed=321)
        bandwidth = 2.5

        py_rkd = SharpRKD(cutoff=0.0, bandwidth=bandwidth, polynomial_order=2)
        py_result = py_rkd.fit(Y, X, D)

        jl_result = julia_sharp_rkd(Y, X, D, cutoff=0.0, bandwidth=bandwidth, polynomial_order=2)

        assert np.isclose(py_result.estimate, jl_result["estimate"], rtol=0.20)

    def test_negative_effect_parity(self):
        """Both should correctly estimate negative effects."""
        Y, X, D = generate_rkd_data(n=1000, true_effect=-1.5, seed=654)
        bandwidth = 2.5

        py_rkd = SharpRKD(cutoff=0.0, bandwidth=bandwidth)
        py_result = py_rkd.fit(Y, X, D)

        jl_result = julia_sharp_rkd(Y, X, D, cutoff=0.0, bandwidth=bandwidth)

        # Both should be negative
        assert py_result.estimate < 0
        assert jl_result["estimate"] < 0

        # And similar magnitude
        assert np.isclose(py_result.estimate, jl_result["estimate"], rtol=0.20)

    def test_nonzero_cutoff_parity(self):
        """Both should work with non-zero cutoff."""
        Y, X, D = generate_rkd_data(n=1000, cutoff=2.0, seed=987)
        bandwidth = 2.0

        py_rkd = SharpRKD(cutoff=2.0, bandwidth=bandwidth)
        py_result = py_rkd.fit(Y, X, D)

        jl_result = julia_sharp_rkd(Y, X, D, cutoff=2.0, bandwidth=bandwidth)

        assert np.isclose(py_result.estimate, jl_result["estimate"], rtol=0.25)

    def test_large_sample_parity(self):
        """With large samples, estimates should converge."""
        Y, X, D = generate_rkd_data(n=5000, true_effect=2.0, seed=111)
        bandwidth = 2.0

        py_rkd = SharpRKD(cutoff=0.0, bandwidth=bandwidth)
        py_result = py_rkd.fit(Y, X, D)

        jl_result = julia_sharp_rkd(Y, X, D, cutoff=0.0, bandwidth=bandwidth)

        # With more data, should be closer to true effect and to each other
        assert abs(py_result.estimate - 2.0) < 0.5
        assert abs(jl_result["estimate"] - 2.0) < 0.5
        assert np.isclose(py_result.estimate, jl_result["estimate"], rtol=0.10)

    def test_sample_sizes_match(self):
        """Effective sample sizes should be similar."""
        Y, X, D = generate_rkd_data(n=1000, seed=222)
        bandwidth = 2.0

        py_rkd = SharpRKD(cutoff=0.0, bandwidth=bandwidth)
        py_result = py_rkd.fit(Y, X, D)

        jl_result = julia_sharp_rkd(Y, X, D, cutoff=0.0, bandwidth=bandwidth)

        # Sample sizes should be the same or very close
        assert abs(py_result.n_left - jl_result["n_left"]) <= 5
        assert abs(py_result.n_right - jl_result["n_right"]) <= 5


# =============================================================================
# Slope Comparison Tests
# =============================================================================


class TestSlopeParity:
    """Tests for slope estimation parity."""

    def test_treatment_slopes_parity(self):
        """Treatment slope estimates should match."""
        np.random.seed(333)
        n = 1500
        X = np.random.uniform(-5, 5, n)
        D = np.where(X < 0, 0.4 * X, 1.6 * X)  # kink = 1.2
        Y = 2.0 * D + np.random.normal(0, 0.5, n)

        bandwidth = 2.5

        py_rkd = SharpRKD(cutoff=0.0, bandwidth=bandwidth)
        py_result = py_rkd.fit(Y, X, D)

        jl_result = julia_sharp_rkd(Y, X, D, cutoff=0.0, bandwidth=bandwidth)

        # Treatment slopes (Python uses slope_d_*, Julia uses treatment_slope_*)
        assert np.isclose(py_result.slope_d_left, jl_result["treatment_slope_left"], rtol=0.20)
        assert np.isclose(py_result.slope_d_right, jl_result["treatment_slope_right"], rtol=0.20)

    def test_outcome_slopes_parity(self):
        """Outcome slope estimates should match."""
        np.random.seed(444)
        n = 1500
        X = np.random.uniform(-5, 5, n)
        D = np.where(X < 0, 0.5 * X, 1.5 * X)
        Y = 3.0 * D + np.random.normal(0, 0.8, n)

        bandwidth = 2.5

        py_rkd = SharpRKD(cutoff=0.0, bandwidth=bandwidth)
        py_result = py_rkd.fit(Y, X, D)

        jl_result = julia_sharp_rkd(Y, X, D, cutoff=0.0, bandwidth=bandwidth)

        # Outcome slopes (Python uses slope_y_*, Julia uses outcome_slope_*)
        assert np.isclose(py_result.slope_y_left, jl_result["outcome_slope_left"], rtol=0.20)
        assert np.isclose(py_result.slope_y_right, jl_result["outcome_slope_right"], rtol=0.20)
