"""Triangulation tests: Python Control Function vs R 2-step OLS implementation.

This module provides Layer 5 validation by comparing our Python implementation
of Control Function against manual 2-step OLS in R.

Control Function is numerically equivalent to 2SLS for linear models,
but provides an explicit endogeneity test via the control coefficient (rho).

Tolerance levels (established in plan):
- Treatment coefficient: rtol=0.01 (exact formula match, both are OLS)
- SE (naive): rtol=0.05 (same calculation)
- Control coefficient: rtol=0.01 (exact match)
- First-stage F: rtol=0.05 (R uses slightly different F formula)
- Endogeneity test conclusion: Exact match (same null hypothesis)

Run with: pytest tests/validation/r_triangulation/test_control_function_vs_r.py -v

Created: Session 181 (2026-01-02)
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.validation.r_triangulation.r_interface import (
    check_r_available,
    r_control_function_manual,
)

# Lazy import to avoid errors when module paths differ
try:
    from src.causal_inference.control_function import ControlFunction

    CF_AVAILABLE = True
except ImportError:
    CF_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests in this module if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_cf_python = pytest.mark.skipif(
    not CF_AVAILABLE,
    reason="Python Control Function module not available",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_cf_dgp(
    n: int = 500,
    true_effect: float = 2.0,
    first_stage_strength: float = 0.5,
    endogeneity: float = 0.5,
    noise_sd: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate data from a Control Function DGP.

    Model:
        Y = alpha + beta*D + epsilon
        D = pi_0 + pi*Z + nu
        Cov(epsilon, nu) = endogeneity  (creates endogeneity bias)

    Parameters
    ----------
    n : int
        Sample size.
    true_effect : float
        True causal effect beta.
    first_stage_strength : float
        Coefficient on instrument in first stage (higher = stronger).
    endogeneity : float
        Correlation between epsilon and nu (0 = exogenous).
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with keys: Y, D, Z, true_effect, endogeneity
    """
    rng = np.random.default_rng(seed)

    # Generate correlated errors
    cov_matrix = np.array([[1.0, endogeneity], [endogeneity, 1.0]])
    errors = rng.multivariate_normal([0, 0], cov_matrix, size=n)
    nu = errors[:, 0]  # First-stage error
    epsilon = errors[:, 1] * noise_sd  # Outcome error

    # Generate instrument (exogenous)
    Z = rng.normal(0, 1, n)

    # First stage: D = pi*Z + nu
    D = first_stage_strength * Z + nu

    # Outcome: Y = beta*D + epsilon
    Y = true_effect * D + epsilon

    return {
        "Y": Y,
        "D": D,
        "Z": Z,
        "true_effect": true_effect,
        "endogeneity": endogeneity,
        "first_stage_strength": first_stage_strength,
    }


def generate_cf_dgp_with_controls(
    n: int = 500,
    n_controls: int = 2,
    true_effect: float = 2.0,
    first_stage_strength: float = 0.5,
    endogeneity: float = 0.5,
    seed: int = 42,
) -> dict:
    """Generate DGP with exogenous control variables."""
    rng = np.random.default_rng(seed)

    # Generate correlated errors
    cov_matrix = np.array([[1.0, endogeneity], [endogeneity, 1.0]])
    errors = rng.multivariate_normal([0, 0], cov_matrix, size=n)
    nu = errors[:, 0]
    epsilon = errors[:, 1]

    # Generate controls
    X = rng.normal(0, 1, (n, n_controls))
    control_effects_Y = rng.uniform(0.5, 1.5, n_controls)
    control_effects_D = rng.uniform(0.2, 0.5, n_controls)

    # Generate instrument
    Z = rng.normal(0, 1, n)

    # First stage: D = pi*Z + X'delta + nu
    D = first_stage_strength * Z + X @ control_effects_D + nu

    # Outcome: Y = beta*D + X'gamma + epsilon
    Y = true_effect * D + X @ control_effects_Y + epsilon

    return {
        "Y": Y,
        "D": D,
        "Z": Z,
        "X": X,
        "true_effect": true_effect,
        "endogeneity": endogeneity,
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestControlFunctionCoefficientVsR:
    """Test treatment effect coefficient matches R."""

    @requires_cf_python
    def test_basic_endogenous_treatment(self):
        """Basic DGP with moderate endogeneity: coefficient matches R."""
        data = generate_cf_dgp(n=500, endogeneity=0.5, seed=42)

        # Python Control Function (analytical SE for faster test)
        cf = ControlFunction(inference="analytical")
        py_result = cf.fit(data["Y"], data["D"], data["Z"])

        # R Control Function
        r_result = r_control_function_manual(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )

        # Treatment coefficient should match exactly (both are OLS)
        assert np.isclose(py_result["estimate"], r_result["estimate"], rtol=0.01), (
            f"Coefficient mismatch: Python={py_result['estimate']:.6f}, "
            f"R={r_result['estimate']:.6f}"
        )

    @requires_cf_python
    def test_strong_endogeneity(self):
        """Strong endogeneity (rho=0.8): coefficient matches R."""
        data = generate_cf_dgp(n=500, endogeneity=0.8, seed=123)

        cf = ControlFunction(inference="analytical")
        py_result = cf.fit(data["Y"], data["D"], data["Z"])

        r_result = r_control_function_manual(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )

        assert np.isclose(py_result["estimate"], r_result["estimate"], rtol=0.01), (
            f"Coefficient mismatch with strong endogeneity: "
            f"Python={py_result['estimate']:.6f}, R={r_result['estimate']:.6f}"
        )

    @requires_cf_python
    def test_no_endogeneity(self):
        """Exogenous treatment (rho=0): coefficient matches R."""
        data = generate_cf_dgp(n=500, endogeneity=0.0, seed=456)

        cf = ControlFunction(inference="analytical")
        py_result = cf.fit(data["Y"], data["D"], data["Z"])

        r_result = r_control_function_manual(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )

        assert np.isclose(py_result["estimate"], r_result["estimate"], rtol=0.01), (
            f"Coefficient mismatch with no endogeneity: "
            f"Python={py_result['estimate']:.6f}, R={r_result['estimate']:.6f}"
        )

    @requires_cf_python
    def test_with_controls(self):
        """Include exogenous control variables: coefficient matches R."""
        data = generate_cf_dgp_with_controls(n=500, n_controls=2, seed=789)

        cf = ControlFunction(inference="analytical")
        py_result = cf.fit(data["Y"], data["D"], data["Z"], data["X"])

        r_result = r_control_function_manual(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
            controls=data["X"],
        )

        assert np.isclose(py_result["estimate"], r_result["estimate"], rtol=0.01), (
            f"Coefficient mismatch with controls: "
            f"Python={py_result['estimate']:.6f}, R={r_result['estimate']:.6f}"
        )

    @requires_cf_python
    def test_large_sample(self):
        """Large sample (n=2000): coefficient matches R."""
        data = generate_cf_dgp(n=2000, endogeneity=0.5, seed=999)

        cf = ControlFunction(inference="analytical")
        py_result = cf.fit(data["Y"], data["D"], data["Z"])

        r_result = r_control_function_manual(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )

        # Larger sample should give even closer match
        assert np.isclose(py_result["estimate"], r_result["estimate"], rtol=0.005), (
            f"Coefficient mismatch with large sample: "
            f"Python={py_result['estimate']:.6f}, R={r_result['estimate']:.6f}"
        )


class TestControlFunctionEndogeneityTestVsR:
    """Test endogeneity test (H0: rho = 0) matches R."""

    @requires_cf_python
    def test_control_coefficient_matches_r(self):
        """Control coefficient (rho) matches R."""
        data = generate_cf_dgp(n=500, endogeneity=0.5, seed=42)

        cf = ControlFunction(inference="analytical")
        py_result = cf.fit(data["Y"], data["D"], data["Z"])

        r_result = r_control_function_manual(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )

        assert np.isclose(py_result["control_coef"], r_result["control_coef"], rtol=0.01), (
            f"Control coefficient mismatch: "
            f"Python={py_result['control_coef']:.6f}, R={r_result['control_coef']:.6f}"
        )

    @requires_cf_python
    def test_endogeneity_detected_match(self):
        """Both Python and R detect endogeneity when present."""
        data = generate_cf_dgp(n=500, endogeneity=0.7, seed=111)

        cf = ControlFunction(inference="analytical")
        py_result = cf.fit(data["Y"], data["D"], data["Z"])

        r_result = r_control_function_manual(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )

        # Both should detect endogeneity
        assert py_result["endogeneity_detected"] == r_result["endogeneity_detected"], (
            f"Endogeneity detection mismatch: "
            f"Python={py_result['endogeneity_detected']}, R={r_result['endogeneity_detected']}"
        )
        # And both should be True (endogeneity is present)
        assert py_result["endogeneity_detected"], "Endogeneity should be detected"

    @requires_cf_python
    def test_endogeneity_not_detected_match(self):
        """Both Python and R fail to detect when exogenous."""
        data = generate_cf_dgp(n=500, endogeneity=0.0, seed=222)

        cf = ControlFunction(inference="analytical")
        py_result = cf.fit(data["Y"], data["D"], data["Z"])

        r_result = r_control_function_manual(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )

        # Both should NOT detect endogeneity
        assert py_result["endogeneity_detected"] == r_result["endogeneity_detected"], (
            f"Endogeneity detection mismatch: "
            f"Python={py_result['endogeneity_detected']}, R={r_result['endogeneity_detected']}"
        )

    @requires_cf_python
    def test_control_se_naive_matches_r(self):
        """Naive SE of control coefficient matches R."""
        data = generate_cf_dgp(n=500, endogeneity=0.5, seed=333)

        cf = ControlFunction(inference="analytical")
        py_result = cf.fit(data["Y"], data["D"], data["Z"])

        r_result = r_control_function_manual(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )

        # Note: Python uses corrected SE, but we can compare control SE
        # which has similar correction issues
        # Use looser tolerance for SE comparison
        assert np.isclose(py_result["se_naive"], r_result["se_naive"], rtol=0.05), (
            f"Naive SE mismatch: Python={py_result['se_naive']:.6f}, R={r_result['se_naive']:.6f}"
        )


class TestControlFunctionFirstStageVsR:
    """Test first-stage diagnostics match R."""

    @requires_cf_python
    def test_first_stage_f_stat(self):
        """F-statistic matches R's first-stage F."""
        data = generate_cf_dgp(n=500, first_stage_strength=0.5, seed=42)

        cf = ControlFunction(inference="analytical")
        cf.fit(data["Y"], data["D"], data["Z"])
        py_f = cf.first_stage_["f_statistic"]

        r_result = r_control_function_manual(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )

        # F-statistic should be close (slight differences in formula)
        assert np.isclose(py_f, r_result["first_stage_f"], rtol=0.05), (
            f"First-stage F mismatch: Python={py_f:.2f}, R={r_result['first_stage_f']:.2f}"
        )

    @requires_cf_python
    def test_weak_iv_warning_consistency(self):
        """Weak IV warning (F < 10) is consistent."""
        # Weak instrument case
        data_weak = generate_cf_dgp(n=500, first_stage_strength=0.1, seed=444)

        cf = ControlFunction(inference="analytical")
        cf.fit(data_weak["Y"], data_weak["D"], data_weak["Z"])
        py_weak_warning = cf.first_stage_["weak_iv_warning"]

        r_result = r_control_function_manual(
            outcome=data_weak["Y"],
            endogenous=data_weak["D"],
            instruments=data_weak["Z"],
        )
        r_weak_warning = r_result["first_stage_f"] < 10

        assert py_weak_warning == r_weak_warning, (
            f"Weak IV warning inconsistent: Python={py_weak_warning}, R={r_weak_warning}"
        )

    @requires_cf_python
    def test_strong_iv_consistency(self):
        """Strong instrument (F > 10) is consistent."""
        # Strong instrument case
        data_strong = generate_cf_dgp(n=500, first_stage_strength=1.0, seed=555)

        cf = ControlFunction(inference="analytical")
        cf.fit(data_strong["Y"], data_strong["D"], data_strong["Z"])
        py_weak_warning = cf.first_stage_["weak_iv_warning"]

        r_result = r_control_function_manual(
            outcome=data_strong["Y"],
            endogenous=data_strong["D"],
            instruments=data_strong["Z"],
        )
        r_weak_warning = r_result["first_stage_f"] < 10

        # Both should NOT issue weak IV warning
        assert py_weak_warning == r_weak_warning, (
            f"Strong IV classification inconsistent: "
            f"Python weak={py_weak_warning}, R weak={r_weak_warning}"
        )
        assert not py_weak_warning, "Strong IV should not trigger weak warning"


class TestControlFunctionEquivalence2SLS:
    """Verify CF ≈ 2SLS for linear models."""

    @requires_cf_python
    def test_coefficient_matches_2sls(self):
        """Control Function coefficient equals 2SLS coefficient."""
        from src.causal_inference.iv import TwoStageLeastSquares

        data = generate_cf_dgp(n=500, endogeneity=0.5, seed=42)

        # Control Function
        cf = ControlFunction(inference="analytical")
        cf_result = cf.fit(data["Y"], data["D"], data["Z"])

        # 2SLS
        tsls = TwoStageLeastSquares(inference="robust")
        tsls.fit(data["Y"], data["D"].reshape(-1, 1), data["Z"].reshape(-1, 1))

        # Should be numerically equivalent (up to floating point)
        assert np.isclose(cf_result["estimate"], tsls.coef_[0], rtol=1e-10), (
            f"CF vs 2SLS mismatch: CF={cf_result['estimate']:.10f}, 2SLS={tsls.coef_[0]:.10f}"
        )


class TestControlFunctionMonteCarlo:
    """Monte Carlo validation of CF vs R."""

    @requires_cf_python
    @pytest.mark.slow
    def test_monte_carlo_coefficient_parity(self):
        """Monte Carlo: CF coefficient matches R across 10 runs."""
        discrepancies = []

        for seed in range(10):
            data = generate_cf_dgp(n=300, endogeneity=0.5, seed=seed)

            cf = ControlFunction(inference="analytical")
            py_result = cf.fit(data["Y"], data["D"], data["Z"])

            r_result = r_control_function_manual(
                outcome=data["Y"],
                endogenous=data["D"],
                instruments=data["Z"],
            )

            discrepancy = abs(py_result["estimate"] - r_result["estimate"])
            discrepancies.append(discrepancy)

        mean_discrepancy = np.mean(discrepancies)
        max_discrepancy = np.max(discrepancies)

        assert mean_discrepancy < 0.001, f"Mean discrepancy too large: {mean_discrepancy:.6f}"
        assert max_discrepancy < 0.01, f"Max discrepancy too large: {max_discrepancy:.6f}"

    @requires_cf_python
    @pytest.mark.slow
    def test_monte_carlo_endogeneity_detection_agreement(self):
        """Monte Carlo: Endogeneity detection agrees across 10 runs."""
        disagreements = 0

        for seed in range(10):
            data = generate_cf_dgp(n=300, endogeneity=0.6, seed=seed)

            cf = ControlFunction(inference="analytical")
            py_result = cf.fit(data["Y"], data["D"], data["Z"])

            r_result = r_control_function_manual(
                outcome=data["Y"],
                endogenous=data["D"],
                instruments=data["Z"],
            )

            if py_result["endogeneity_detected"] != r_result["endogeneity_detected"]:
                disagreements += 1

        # Allow at most 1 disagreement (borderline cases)
        assert disagreements <= 1, (
            f"Too many endogeneity detection disagreements: {disagreements}/10"
        )


# =============================================================================
# Edge Cases
# =============================================================================


class TestControlFunctionEdgeCases:
    """Edge case tests for robustness."""

    @requires_cf_python
    def test_minimal_sample_size(self):
        """Minimal sample size (n=50) still matches R."""
        data = generate_cf_dgp(n=50, seed=666)

        cf = ControlFunction(inference="analytical")
        py_result = cf.fit(data["Y"], data["D"], data["Z"])

        r_result = r_control_function_manual(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )

        # Looser tolerance for small samples
        assert np.isclose(py_result["estimate"], r_result["estimate"], rtol=0.02), (
            f"Small sample coefficient mismatch: "
            f"Python={py_result['estimate']:.6f}, R={r_result['estimate']:.6f}"
        )

    @requires_cf_python
    def test_high_noise(self):
        """High outcome noise still matches R."""
        data = generate_cf_dgp(n=500, noise_sd=3.0, seed=777)

        cf = ControlFunction(inference="analytical")
        py_result = cf.fit(data["Y"], data["D"], data["Z"])

        r_result = r_control_function_manual(
            outcome=data["Y"],
            endogenous=data["D"],
            instruments=data["Z"],
        )

        assert np.isclose(py_result["estimate"], r_result["estimate"], rtol=0.01), (
            f"High noise coefficient mismatch: "
            f"Python={py_result['estimate']:.6f}, R={r_result['estimate']:.6f}"
        )
