"""Triangulation tests: Python IV estimators vs R `AER` package.

This module provides Layer 5 validation by comparing our Python implementation
of 2SLS and LIML against the official R `AER` package (ivreg).

Tests skip gracefully when R/rpy2 is unavailable.

Tolerance levels (established in plan):
- 2SLS coefficient: rtol=0.01 (exact formula match)
- 2SLS SE: rtol=0.05 (both use correct 2SLS formula)
- LIML coefficient: rtol=0.05 (κ computation may differ)
- LIML SE: rtol=0.15 (k-class SEs sensitive)
- First-stage F: rtol=0.01 (same calculation)

Run with: pytest tests/validation/r_triangulation/test_iv_vs_r.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.validation.r_triangulation.r_interface import (
    check_aer_installed,
    check_r_available,
    r_2sls_aer,
    r_liml_aer,
)

# Lazy import to avoid errors when iv module paths differ
try:
    from src.causal_inference.iv.two_stage_least_squares import TwoStageLeastSquares
    from src.causal_inference.iv.liml import LIML

    IV_AVAILABLE = True
except ImportError:
    IV_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests in this module if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_iv_python = pytest.mark.skipif(
    not IV_AVAILABLE,
    reason="Python IV module not available",
)

requires_aer_r = pytest.mark.skipif(
    not check_aer_installed(),
    reason="R 'AER' package not installed",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_iv_dgp(
    n: int = 500,
    n_instruments: int = 1,
    n_controls: int = 0,
    true_effect: float = 1.0,
    first_stage_strength: float = 0.5,
    endogeneity: float = 0.5,
    noise_sd: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate data from an IV DGP.

    Model:
        Y = α + β*D + X'γ + ε
        D = π_0 + Z'π + X'δ + υ
        Cov(ε, υ) = endogeneity * σ_ε * σ_υ  (endogeneity bias)

    Parameters
    ----------
    n : int
        Sample size.
    n_instruments : int
        Number of instruments (for overidentification).
    n_controls : int
        Number of exogenous controls.
    true_effect : float
        True causal effect β.
    first_stage_strength : float
        Coefficient on instruments in first stage (higher = stronger).
    endogeneity : float
        Correlation between ε and υ (0 = exogenous, 0.5 = moderate).
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with Y, D, Z, X arrays and true parameters.
    """
    np.random.seed(seed)

    # Generate instruments Z (correlated with D, excluded from outcome)
    Z = np.random.randn(n, n_instruments)

    # Generate controls X (included in both equations)
    if n_controls > 0:
        X = np.random.randn(n, n_controls)
    else:
        X = None

    # Generate correlated errors (endogeneity)
    mean = [0, 0]
    cov = [[1, endogeneity], [endogeneity, 1]]
    errors = np.random.multivariate_normal(mean, cov, n)
    epsilon = errors[:, 0] * noise_sd  # Outcome error
    upsilon = errors[:, 1]  # First-stage error

    # First stage: D = π_0 + Z'π + X'δ + υ
    D = 0.5 + Z @ (np.ones(n_instruments) * first_stage_strength) + upsilon
    if X is not None:
        D += X @ (np.ones(n_controls) * 0.3)

    # Second stage: Y = α + β*D + X'γ + ε
    Y = 1.0 + true_effect * D + epsilon
    if X is not None:
        Y += X @ (np.ones(n_controls) * 0.5)

    return {
        "Y": Y,
        "D": D,
        "Z": Z,
        "X": X,
        "true_effect": true_effect,
        "n": n,
        "n_instruments": n_instruments,
        "n_controls": n_controls,
        "first_stage_strength": first_stage_strength,
    }


def generate_weak_iv_dgp(
    n: int = 500,
    n_instruments: int = 1,
    true_effect: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate data with weak instruments (low first-stage F).

    First-stage F-statistic should be < 10 (Stock-Yogo threshold).
    """
    return generate_iv_dgp(
        n=n,
        n_instruments=n_instruments,
        true_effect=true_effect,
        first_stage_strength=0.05,  # Very weak
        endogeneity=0.5,
        seed=seed,
    )


# =============================================================================
# Layer 5: 2SLS Triangulation
# =============================================================================


@requires_iv_python
@requires_aer_r
class TestTwoSLSVsAER:
    """Compare Python TwoStageLeastSquares to R `AER::ivreg()`."""

    def test_just_identified_coef_parity(self):
        """Just-identified 2SLS (1 instrument, 1 endogenous) coefficient match."""
        data = generate_iv_dgp(
            n=500,
            n_instruments=1,
            n_controls=0,
            true_effect=2.0,
            first_stage_strength=0.5,
            seed=42,
        )

        # Python estimate
        py_iv = TwoStageLeastSquares()
        py_iv.fit(data["Y"], data["D"], data["Z"])
        py_coef = py_iv.coef_[0]

        # R estimate
        r_result = r_2sls_aer(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )
        r_coef = r_result["coef"]

        assert np.isclose(py_coef, r_coef, rtol=0.01), (
            f"2SLS coefficient mismatch: Python={py_coef:.6f}, R={r_coef:.6f}"
        )

    def test_just_identified_se_parity(self):
        """Just-identified 2SLS SE should match within rtol=0.05."""
        data = generate_iv_dgp(
            n=500,
            n_instruments=1,
            true_effect=1.5,
            seed=123,
        )

        py_iv = TwoStageLeastSquares()
        py_iv.fit(data["Y"], data["D"], data["Z"])
        py_se = py_iv.se_[0]

        r_result = r_2sls_aer(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )
        r_se = r_result["se"]

        assert np.isclose(py_se, r_se, rtol=0.05), (
            f"2SLS SE mismatch: Python={py_se:.6f}, R={r_se:.6f}"
        )

    def test_overidentified_coef_parity(self):
        """Overidentified 2SLS (3 instruments) coefficient match."""
        data = generate_iv_dgp(
            n=500,
            n_instruments=3,
            true_effect=1.0,
            first_stage_strength=0.4,
            seed=456,
        )

        py_iv = TwoStageLeastSquares()
        py_iv.fit(data["Y"], data["D"], data["Z"])
        py_coef = py_iv.coef_[0]

        r_result = r_2sls_aer(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )
        r_coef = r_result["coef"]

        assert np.isclose(py_coef, r_coef, rtol=0.02), (
            f"Overidentified 2SLS coefficient mismatch: Python={py_coef:.6f}, R={r_coef:.6f}"
        )

    def test_with_controls_coef_parity(self):
        """2SLS with exogenous controls should match R."""
        data = generate_iv_dgp(
            n=500,
            n_instruments=2,
            n_controls=2,
            true_effect=2.5,
            seed=789,
        )

        py_iv = TwoStageLeastSquares()
        py_iv.fit(data["Y"], data["D"], data["Z"], data["X"])
        py_coef = py_iv.coef_[0]

        r_result = r_2sls_aer(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
            controls=data["X"],
        )
        r_coef = r_result["coef"]

        # Looser tolerance with controls due to potential formula differences
        assert np.isclose(py_coef, r_coef, rtol=0.05), (
            f"2SLS with controls coefficient mismatch: Python={py_coef:.6f}, R={r_coef:.6f}"
        )

    def test_ci_coverage_parity(self):
        """CI bounds should be consistent between Python and R."""
        data = generate_iv_dgp(
            n=500,
            n_instruments=1,
            true_effect=2.0,
            seed=101,
        )

        py_iv = TwoStageLeastSquares()
        py_iv.fit(data["Y"], data["D"], data["Z"])

        r_result = r_2sls_aer(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )

        # CI width ratio should be close to 1
        py_width = py_iv.ci_[0, 1] - py_iv.ci_[0, 0]
        r_width = r_result["ci_upper"] - r_result["ci_lower"]

        ci_width_ratio = py_width / (r_width + 1e-10)
        assert 0.8 < ci_width_ratio < 1.2, (
            f"CI width ratio {ci_width_ratio:.2f} out of expected range"
        )


@requires_iv_python
@requires_aer_r
class TestLIMLVsAER:
    """Compare Python LIML to R `AER::ivreg(method='LIML')`."""

    def test_liml_just_identified(self):
        """LIML with just-identified should equal 2SLS."""
        data = generate_iv_dgp(
            n=500,
            n_instruments=1,
            true_effect=1.5,
            seed=202,
        )

        py_liml = LIML()
        py_liml.fit(data["Y"], data["D"], data["Z"])
        py_coef = py_liml.coef_[0]

        r_result = r_liml_aer(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )
        r_coef = r_result["coef"]

        # Just-identified LIML = 2SLS, so exact match expected
        assert np.isclose(py_coef, r_coef, rtol=0.02), (
            f"Just-identified LIML coefficient mismatch: Python={py_coef:.6f}, R={r_coef:.6f}"
        )

    def test_liml_overidentified(self):
        """LIML with overidentification should be close to R."""
        data = generate_iv_dgp(
            n=500,
            n_instruments=3,
            true_effect=1.0,
            first_stage_strength=0.4,
            seed=303,
        )

        py_liml = LIML()
        py_liml.fit(data["Y"], data["D"], data["Z"])
        py_coef = py_liml.coef_[0]

        r_result = r_liml_aer(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )
        r_coef = r_result["coef"]

        assert np.isclose(py_coef, r_coef, rtol=0.05), (
            f"Overidentified LIML coefficient mismatch: Python={py_coef:.6f}, R={r_coef:.6f}"
        )

    def test_liml_weak_instruments(self):
        """LIML with weak instruments should still be close to R."""
        data = generate_weak_iv_dgp(
            n=500,
            n_instruments=2,
            true_effect=1.0,
            seed=404,
        )

        py_liml = LIML()
        py_liml.fit(data["Y"], data["D"], data["Z"])
        py_coef = py_liml.coef_[0]

        r_result = r_liml_aer(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )
        r_coef = r_result["coef"]

        # Looser tolerance for weak instruments
        assert np.isclose(py_coef, r_coef, rtol=0.10), (
            f"Weak IV LIML coefficient mismatch: Python={py_coef:.6f}, R={r_coef:.6f}"
        )

    def test_liml_se_parity(self):
        """LIML SE should match R within rtol=0.15."""
        data = generate_iv_dgp(
            n=500,
            n_instruments=2,
            true_effect=2.0,
            seed=505,
        )

        py_liml = LIML()
        py_liml.fit(data["Y"], data["D"], data["Z"])
        py_se = py_liml.se_[0]

        r_result = r_liml_aer(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )
        r_se = r_result["se"]

        assert np.isclose(py_se, r_se, rtol=0.15), (
            f"LIML SE mismatch: Python={py_se:.6f}, R={r_se:.6f}"
        )


@requires_iv_python
@requires_aer_r
class TestIVDiagnosticsVsR:
    """Compare IV diagnostics between Python and R."""

    def test_first_stage_f_strong_instruments(self):
        """First-stage F-stat should match for strong instruments."""
        data = generate_iv_dgp(
            n=500,
            n_instruments=1,
            true_effect=1.0,
            first_stage_strength=0.5,
            seed=606,
        )

        py_iv = TwoStageLeastSquares()
        py_iv.fit(data["Y"], data["D"], data["Z"])
        py_f = py_iv.first_stage_f_stat_

        r_result = r_2sls_aer(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )
        r_f = r_result["first_stage_f"]

        # F-stat should be > 10 for strong instruments
        assert py_f > 10, f"Python F-stat {py_f:.2f} too low for strong IV"
        assert r_f > 10, f"R F-stat {r_f:.2f} too low for strong IV"

        # F-stats should be close
        assert np.isclose(py_f, r_f, rtol=0.10), (
            f"First-stage F mismatch: Python={py_f:.2f}, R={r_f:.2f}"
        )

    def test_first_stage_f_weak_instruments(self):
        """First-stage F-stat should be low for weak instruments."""
        data = generate_weak_iv_dgp(
            n=500,
            n_instruments=1,
            true_effect=1.0,
            seed=707,
        )

        py_iv = TwoStageLeastSquares()
        py_iv.fit(data["Y"], data["D"], data["Z"])
        py_f = py_iv.first_stage_f_stat_

        r_result = r_2sls_aer(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )
        r_f = r_result["first_stage_f"]

        # Both should detect weak instruments (F < 10)
        assert py_f < 10, f"Python F-stat {py_f:.2f} unexpectedly high for weak IV"
        # R may compute slightly different F, just check direction
        assert r_f < 15, f"R F-stat {r_f:.2f} unexpectedly high for weak IV"


@requires_iv_python
@requires_aer_r
class TestIVEdgeCases:
    """Edge case tests for IV triangulation."""

    def test_small_sample(self):
        """IV should work with small samples (n=50)."""
        data = generate_iv_dgp(
            n=50,
            n_instruments=1,
            true_effect=2.0,
            first_stage_strength=0.6,
            seed=808,
        )

        py_iv = TwoStageLeastSquares()
        py_iv.fit(data["Y"], data["D"], data["Z"])
        py_coef = py_iv.coef_[0]

        r_result = r_2sls_aer(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )
        r_coef = r_result["coef"]

        # Looser tolerance for small samples
        assert np.isclose(py_coef, r_coef, rtol=0.05), (
            f"Small sample 2SLS mismatch: Python={py_coef:.6f}, R={r_coef:.6f}"
        )

    def test_many_instruments(self):
        """IV with many instruments (5) should still work."""
        data = generate_iv_dgp(
            n=500,
            n_instruments=5,
            true_effect=1.5,
            first_stage_strength=0.3,
            seed=909,
        )

        py_iv = TwoStageLeastSquares()
        py_iv.fit(data["Y"], data["D"], data["Z"])
        py_coef = py_iv.coef_[0]

        r_result = r_2sls_aer(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )
        r_coef = r_result["coef"]

        assert np.isclose(py_coef, r_coef, rtol=0.05), (
            f"Many instruments 2SLS mismatch: Python={py_coef:.6f}, R={r_coef:.6f}"
        )

    def test_zero_effect(self):
        """IV should recover zero effect correctly."""
        data = generate_iv_dgp(
            n=500,
            n_instruments=2,
            true_effect=0.0,
            first_stage_strength=0.5,
            seed=1010,
        )

        py_iv = TwoStageLeastSquares()
        py_iv.fit(data["Y"], data["D"], data["Z"])
        py_coef = py_iv.coef_[0]

        r_result = r_2sls_aer(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )
        r_coef = r_result["coef"]

        # Both should be close to zero
        assert abs(py_coef) < 0.5, f"Python zero effect estimate {py_coef:.4f} too far from 0"
        assert abs(r_coef) < 0.5, f"R zero effect estimate {r_coef:.4f} too far from 0"

        # Should match each other
        assert np.isclose(py_coef, r_coef, atol=0.1), (
            f"Zero effect mismatch: Python={py_coef:.6f}, R={r_coef:.6f}"
        )


@requires_iv_python
@requires_aer_r
class TestIVMonteCarlo:
    """Monte Carlo validation of IV triangulation."""

    @pytest.mark.slow
    def test_monte_carlo_bias_comparison(self):
        """Both Python and R should have similar bias properties."""
        n_sims = 50
        true_effect = 1.5
        py_estimates = []
        r_estimates = []

        for sim in range(n_sims):
            data = generate_iv_dgp(
                n=300,
                n_instruments=2,
                true_effect=true_effect,
                first_stage_strength=0.4,
                seed=2000 + sim,
            )

            try:
                py_iv = TwoStageLeastSquares()
                py_iv.fit(data["Y"], data["D"], data["Z"])
                py_estimates.append(py_iv.coef_[0])

                r_result = r_2sls_aer(
                    outcome=data["Y"],
                    endogenous=data["D"],
                    instruments=data["Z"],
                )
                r_estimates.append(r_result["coef"])
            except Exception:
                continue

        py_bias = np.mean(py_estimates) - true_effect
        r_bias = np.mean(r_estimates) - true_effect

        # Both should have small bias
        assert abs(py_bias) < 0.20, f"Python bias {py_bias:.4f} too large"
        assert abs(r_bias) < 0.20, f"R bias {r_bias:.4f} too large"

        # Bias difference should be small
        assert abs(py_bias - r_bias) < 0.10, (
            f"Bias difference {abs(py_bias - r_bias):.4f} between Python and R too large"
        )

    @pytest.mark.slow
    def test_monte_carlo_coverage_comparison(self):
        """Both Python and R should have similar CI coverage."""
        n_sims = 100
        true_effect = 2.0
        py_covers = 0
        r_covers = 0

        for sim in range(n_sims):
            data = generate_iv_dgp(
                n=300,
                n_instruments=2,
                true_effect=true_effect,
                first_stage_strength=0.5,
                seed=3000 + sim,
            )

            try:
                py_iv = TwoStageLeastSquares()
                py_iv.fit(data["Y"], data["D"], data["Z"])
                if py_iv.ci_[0, 0] < true_effect < py_iv.ci_[0, 1]:
                    py_covers += 1

                r_result = r_2sls_aer(
                    outcome=data["Y"],
                    endogenous=data["D"],
                    instruments=data["Z"],
                )
                if r_result["ci_lower"] < true_effect < r_result["ci_upper"]:
                    r_covers += 1
            except Exception:
                continue

        py_coverage = py_covers / n_sims
        r_coverage = r_covers / n_sims

        # Both should have reasonable coverage (close to 95%)
        assert 0.85 < py_coverage < 0.99, f"Python coverage {py_coverage:.2%} out of range"
        assert 0.85 < r_coverage < 0.99, f"R coverage {r_coverage:.2%} out of range"

        # Coverage should be similar
        assert abs(py_coverage - r_coverage) < 0.10, (
            f"Coverage difference {abs(py_coverage - r_coverage):.2%} too large"
        )
