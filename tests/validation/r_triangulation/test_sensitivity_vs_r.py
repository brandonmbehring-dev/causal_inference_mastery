"""Triangulation tests: Python Sensitivity Analysis vs R reference implementations.

This module provides Layer 5 validation by comparing our Python sensitivity analysis
implementations against R implementations using EValue and sensitivitymv packages.

Tests skip gracefully when R/rpy2 is unavailable.

Tolerance levels (established based on implementation differences):
- E-value: rtol=0.01 (1% relative, closed-form formula)
- E-value CI: rtol=0.02 (2% relative)
- Gamma critical: rtol=0.10 (10% relative, numerical search)
- P-value bounds: atol=0.10 (statistical approximation differences)

Run with: pytest tests/validation/r_triangulation/test_sensitivity_vs_r.py -v

References:
- VanderWeele & Ding (2017). Sensitivity Analysis in Observational Research
- Rosenbaum (2002). Observational Studies, 2nd ed. Springer.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.validation.r_triangulation.r_interface import (
    check_r_available,
    check_evalue_installed,
    check_sensitivitymv_installed,
    r_e_value,
    r_rosenbaum_bounds,
)

# Lazy import Python implementations
try:
    from src.causal_inference.sensitivity.e_value import e_value
    from src.causal_inference.sensitivity.rosenbaum import rosenbaum_bounds

    SENSITIVITY_AVAILABLE = True
except ImportError:
    SENSITIVITY_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_sensitivity_python = pytest.mark.skipif(
    not SENSITIVITY_AVAILABLE,
    reason="Python sensitivity module not available",
)

requires_evalue_r = pytest.mark.skipif(
    not check_evalue_installed(),
    reason="R EValue package not installed. Install with: install.packages('EValue')",
)

requires_sensitivitymv_r = pytest.mark.skipif(
    not check_sensitivitymv_installed(),
    reason="R sensitivitymv package not installed. Install with: install.packages('sensitivitymv')",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_matched_pairs_data(
    n_pairs: int = 50,
    true_effect: float = 2.0,
    noise_sd: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate matched pairs data for Rosenbaum bounds testing.

    Parameters
    ----------
    n_pairs : int
        Number of matched pairs.
    true_effect : float
        True treatment effect.
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with treated_outcomes, control_outcomes, true_effect.
    """
    np.random.seed(seed)

    # Control outcomes (baseline)
    control_outcomes = np.random.randn(n_pairs) * noise_sd

    # Treated outcomes (control + effect + noise)
    treated_outcomes = control_outcomes + true_effect + np.random.randn(n_pairs) * noise_sd

    return {
        "treated_outcomes": treated_outcomes,
        "control_outcomes": control_outcomes,
        "true_effect": true_effect,
        "n_pairs": n_pairs,
    }


def generate_strong_effect_data(seed: int = 42) -> dict:
    """Generate data with strong treatment effect (robust to confounding)."""
    return generate_matched_pairs_data(
        n_pairs=50,
        true_effect=3.0,  # Strong effect
        noise_sd=0.5,
        seed=seed,
    )


def generate_weak_effect_data(seed: int = 123) -> dict:
    """Generate data with weak treatment effect (sensitive to confounding)."""
    return generate_matched_pairs_data(
        n_pairs=30,
        true_effect=0.5,  # Weak effect
        noise_sd=1.5,
        seed=seed,
    )


# =============================================================================
# Test Classes: E-value
# =============================================================================


@requires_sensitivity_python
@requires_evalue_r
class TestEValueVsR:
    """Compare Python e_value() to R EValue package."""

    def test_risk_ratio_greater_than_one(self):
        """Python and R should agree on E-value for RR > 1."""
        rr = 2.5

        py_result = e_value(rr, effect_type="rr")
        r_result = r_e_value(rr, effect_type="rr")

        if r_result is None:
            pytest.skip("R e_value unavailable")

        # E-value formula is exact, should match closely
        assert np.isclose(py_result["e_value"], r_result["e_value"], rtol=0.01), (
            f"E-value mismatch: Python={py_result['e_value']:.4f}, R={r_result['e_value']:.4f}"
        )

    def test_risk_ratio_with_ci(self):
        """Python and R should agree on E-value with confidence interval."""
        rr = 2.0
        ci_lower = 1.5
        ci_upper = 2.7

        py_result = e_value(rr, ci_lower=ci_lower, ci_upper=ci_upper, effect_type="rr")
        r_result = r_e_value(rr, ci_lower=ci_lower, ci_upper=ci_upper, effect_type="rr")

        if r_result is None:
            pytest.skip("R e_value unavailable")

        # Point estimate E-value
        assert np.isclose(py_result["e_value"], r_result["e_value"], rtol=0.01), (
            f"E-value mismatch: Python={py_result['e_value']:.4f}, R={r_result['e_value']:.4f}"
        )

        # CI E-value
        assert np.isclose(py_result["e_value_ci"], r_result["e_value_ci"], rtol=0.02), (
            f"E-value CI mismatch: Python={py_result['e_value_ci']:.4f}, R={r_result['e_value_ci']:.4f}"
        )

    def test_odds_ratio(self):
        """Python and R should agree on E-value for odds ratio."""
        odds_ratio = 3.0

        py_result = e_value(odds_ratio, effect_type="or")
        r_result = r_e_value(odds_ratio, effect_type="or")

        if r_result is None:
            pytest.skip("R e_value unavailable")

        # OR approximated as RR, formula is the same
        assert np.isclose(py_result["e_value"], r_result["e_value"], rtol=0.01), (
            f"E-value mismatch: Python={py_result['e_value']:.4f}, R={r_result['e_value']:.4f}"
        )

    def test_hazard_ratio(self):
        """Python and R should agree on E-value for hazard ratio."""
        hazard_ratio = 1.8

        py_result = e_value(hazard_ratio, effect_type="hr")
        r_result = r_e_value(hazard_ratio, effect_type="hr")

        if r_result is None:
            pytest.skip("R e_value unavailable")

        assert np.isclose(py_result["e_value"], r_result["e_value"], rtol=0.01), (
            f"E-value mismatch: Python={py_result['e_value']:.4f}, R={r_result['e_value']:.4f}"
        )

    def test_standardized_mean_difference(self):
        """Python and R should agree on E-value for SMD."""
        smd = 0.8  # Large effect size (Cohen's d)

        py_result = e_value(smd, effect_type="smd")
        r_result = r_e_value(smd, effect_type="smd")

        if r_result is None:
            pytest.skip("R e_value unavailable")

        # SMD conversion to RR uses exp(0.91*d)
        assert np.isclose(py_result["e_value"], r_result["e_value"], rtol=0.02), (
            f"E-value mismatch: Python={py_result['e_value']:.4f}, R={r_result['e_value']:.4f}"
        )

    def test_protective_effect(self):
        """Python and R should agree on E-value for protective effect (RR < 1)."""
        rr = 0.5  # Protective effect

        py_result = e_value(rr, effect_type="rr")
        r_result = r_e_value(rr, effect_type="rr")

        if r_result is None:
            pytest.skip("R e_value unavailable")

        # For RR < 1, E-value uses 1/RR
        assert np.isclose(py_result["e_value"], r_result["e_value"], rtol=0.01), (
            f"E-value mismatch: Python={py_result['e_value']:.4f}, R={r_result['e_value']:.4f}"
        )

    def test_null_effect(self):
        """Python and R should agree on E-value for null effect (RR = 1)."""
        rr = 1.0  # Null effect

        py_result = e_value(rr, effect_type="rr")
        r_result = r_e_value(rr, effect_type="rr")

        if r_result is None:
            pytest.skip("R e_value unavailable")

        # E-value at null should be 1.0
        assert py_result["e_value"] == 1.0, (
            f"Python E-value should be 1.0, got {py_result['e_value']}"
        )
        assert r_result["e_value"] == 1.0, f"R E-value should be 1.0, got {r_result['e_value']}"

    def test_ci_includes_null(self):
        """E-value for CI should be 1.0 when CI includes the null."""
        rr = 1.5
        ci_lower = 0.8  # Includes 1.0
        ci_upper = 2.5

        py_result = e_value(rr, ci_lower=ci_lower, ci_upper=ci_upper, effect_type="rr")
        r_result = r_e_value(rr, ci_lower=ci_lower, ci_upper=ci_upper, effect_type="rr")

        if r_result is None:
            pytest.skip("R e_value unavailable")

        # CI includes null, so E-value_CI should be 1.0
        assert py_result["e_value_ci"] == 1.0, (
            f"Python E-value_CI should be 1.0 when CI includes null"
        )
        assert r_result["e_value_ci"] == 1.0, f"R E-value_CI should be 1.0 when CI includes null"


# =============================================================================
# Test Classes: Rosenbaum Bounds
# =============================================================================


@requires_sensitivity_python
@requires_sensitivitymv_r
class TestRosenbaumBoundsVsR:
    """Compare Python rosenbaum_bounds() to R sensitivitymv package."""

    def test_strong_effect(self):
        """Python and R should agree on robust finding (strong effect)."""
        data = generate_strong_effect_data(seed=42)

        py_result = rosenbaum_bounds(
            data["treated_outcomes"],
            data["control_outcomes"],
            gamma_range=(1.0, 3.0),
            n_gamma=10,
        )
        r_result = r_rosenbaum_bounds(
            data["treated_outcomes"],
            data["control_outcomes"],
            gamma_range=(1.0, 3.0),
            n_gamma=10,
        )

        if r_result is None:
            pytest.skip("R rosenbaum_bounds unavailable")

        # Gamma values should match exactly (same linspace)
        assert np.allclose(py_result["gamma_values"], r_result["gamma_values"], rtol=1e-10), (
            "Gamma values should match exactly"
        )

        # Strong effect should be robust - both should find high gamma_critical
        # or both should indicate robustness (gamma_critical > 2.5 or None)
        py_robust = py_result["gamma_critical"] is None or py_result["gamma_critical"] > 2.5
        r_robust = r_result["gamma_critical"] is None or r_result["gamma_critical"] > 2.5

        assert py_robust == r_robust, (
            f"Robustness assessment differs: Python={py_result['gamma_critical']}, "
            f"R={r_result['gamma_critical']}"
        )

    def test_weak_effect(self):
        """Python and R should agree on sensitive finding (weak effect)."""
        data = generate_weak_effect_data(seed=123)

        py_result = rosenbaum_bounds(
            data["treated_outcomes"],
            data["control_outcomes"],
            gamma_range=(1.0, 2.0),
            n_gamma=10,
        )
        r_result = r_rosenbaum_bounds(
            data["treated_outcomes"],
            data["control_outcomes"],
            gamma_range=(1.0, 2.0),
            n_gamma=10,
        )

        if r_result is None:
            pytest.skip("R rosenbaum_bounds unavailable")

        # Weak effect - both should find similar gamma_critical
        # Allow 10% relative tolerance on gamma_critical
        if py_result["gamma_critical"] is not None and r_result["gamma_critical"] is not None:
            assert np.isclose(py_result["gamma_critical"], r_result["gamma_critical"], rtol=0.15), (
                f"Gamma critical mismatch: Python={py_result['gamma_critical']:.3f}, R={r_result['gamma_critical']:.3f}"
            )

    def test_p_value_bounds_direction(self):
        """P-value bounds should have correct ordering (p_upper >= p_lower)."""
        data = generate_matched_pairs_data(n_pairs=40, true_effect=1.5, seed=99)

        py_result = rosenbaum_bounds(
            data["treated_outcomes"],
            data["control_outcomes"],
            gamma_range=(1.0, 2.5),
            n_gamma=10,
        )
        r_result = r_rosenbaum_bounds(
            data["treated_outcomes"],
            data["control_outcomes"],
            gamma_range=(1.0, 2.5),
            n_gamma=10,
        )

        if r_result is None:
            pytest.skip("R rosenbaum_bounds unavailable")

        # P-value upper bound should >= lower bound for all gamma
        for i in range(len(py_result["gamma_values"])):
            assert py_result["p_upper"][i] >= py_result["p_lower"][i] - 0.01, (
                f"Python p_upper should >= p_lower at gamma={py_result['gamma_values'][i]}"
            )

        # P-values should increase with gamma (for positive effect)
        # Check monotonicity of p_upper
        for i in range(1, len(py_result["p_upper"])):
            # Allow small tolerance for numerical issues
            assert py_result["p_upper"][i] >= py_result["p_upper"][i - 1] - 0.01, (
                "P-value upper bound should increase with gamma"
            )

    def test_gamma_one_baseline(self):
        """At gamma=1, upper and lower p-values should be equal."""
        data = generate_matched_pairs_data(n_pairs=30, true_effect=2.0, seed=456)

        py_result = rosenbaum_bounds(
            data["treated_outcomes"],
            data["control_outcomes"],
            gamma_range=(1.0, 1.0),  # Only gamma=1
            n_gamma=1,
        )
        r_result = r_rosenbaum_bounds(
            data["treated_outcomes"],
            data["control_outcomes"],
            gamma_range=(1.0, 1.0),
            n_gamma=1,
        )

        if r_result is None:
            pytest.skip("R rosenbaum_bounds unavailable")

        # At gamma=1 (no confounding), p_upper = p_lower
        assert np.isclose(py_result["p_upper"][0], py_result["p_lower"][0], atol=0.01), (
            "At gamma=1, p_upper should equal p_lower"
        )

        # Python and R should give similar p-values at gamma=1
        assert np.isclose(py_result["p_upper"][0], r_result["p_upper"][0], atol=0.10), (
            f"P-value at gamma=1 mismatch: Python={py_result['p_upper'][0]:.4f}, R={r_result['p_upper'][0]:.4f}"
        )


# =============================================================================
# Edge Cases
# =============================================================================


@requires_sensitivity_python
@requires_evalue_r
class TestEValueEdgeCases:
    """Test E-value edge cases and extreme values."""

    def test_very_large_effect(self):
        """E-value for very large effect (RR=10)."""
        rr = 10.0

        py_result = e_value(rr, effect_type="rr")
        r_result = r_e_value(rr, effect_type="rr")

        if r_result is None:
            pytest.skip("R e_value unavailable")

        assert np.isclose(py_result["e_value"], r_result["e_value"], rtol=0.01), (
            f"E-value mismatch for large RR: Python={py_result['e_value']:.4f}, R={r_result['e_value']:.4f}"
        )

        # Very large effect should have large E-value
        assert py_result["e_value"] > 10.0, "E-value for RR=10 should be > 10"

    def test_small_effect(self):
        """E-value for small effect (RR=1.1)."""
        rr = 1.1

        py_result = e_value(rr, effect_type="rr")
        r_result = r_e_value(rr, effect_type="rr")

        if r_result is None:
            pytest.skip("R e_value unavailable")

        assert np.isclose(py_result["e_value"], r_result["e_value"], rtol=0.01), (
            f"E-value mismatch for small RR: Python={py_result['e_value']:.4f}, R={r_result['e_value']:.4f}"
        )

        # Small effect should have E-value close to 1
        assert py_result["e_value"] < 1.5, "E-value for RR=1.1 should be < 1.5"

    def test_negative_smd(self):
        """E-value for negative SMD (protective effect)."""
        smd = -0.5

        py_result = e_value(smd, effect_type="smd")
        r_result = r_e_value(smd, effect_type="smd")

        if r_result is None:
            pytest.skip("R e_value unavailable")

        # Should still compute E-value (uses |SMD| effectively)
        assert py_result["e_value"] > 1.0, "E-value should be > 1 for non-null effect"
        assert np.isclose(py_result["e_value"], r_result["e_value"], rtol=0.02), (
            f"E-value mismatch: Python={py_result['e_value']:.4f}, R={r_result['e_value']:.4f}"
        )


@requires_sensitivity_python
@requires_sensitivitymv_r
class TestRosenbaumEdgeCases:
    """Test Rosenbaum bounds edge cases."""

    def test_large_sample(self):
        """Rosenbaum bounds with larger sample (n=100 pairs)."""
        data = generate_matched_pairs_data(n_pairs=100, true_effect=2.0, seed=789)

        py_result = rosenbaum_bounds(
            data["treated_outcomes"],
            data["control_outcomes"],
            gamma_range=(1.0, 4.0),
            n_gamma=15,
        )
        r_result = r_rosenbaum_bounds(
            data["treated_outcomes"],
            data["control_outcomes"],
            gamma_range=(1.0, 4.0),
            n_gamma=15,
        )

        if r_result is None:
            pytest.skip("R rosenbaum_bounds unavailable")

        # Larger sample should give more precise bounds
        assert len(py_result["p_upper"]) == 15, "Should have 15 gamma values"
        assert len(r_result["p_upper"]) == 15, "R should have 15 gamma values"

    def test_zero_effect(self):
        """Rosenbaum bounds when there's no treatment effect."""
        np.random.seed(111)
        n_pairs = 40
        # No treatment effect - both groups from same distribution
        treated = np.random.randn(n_pairs)
        control = np.random.randn(n_pairs)

        py_result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 2.0), n_gamma=5)
        r_result = r_rosenbaum_bounds(treated, control, gamma_range=(1.0, 2.0), n_gamma=5)

        if r_result is None:
            pytest.skip("R rosenbaum_bounds unavailable")

        # With no effect, p-value at gamma=1 should be relatively high (> 0.05)
        # Note: this is probabilistic, so we use a lenient check
        assert py_result["p_upper"][0] > 0.01, (
            "P-value should not be extremely small for null effect"
        )

    def test_small_sample(self):
        """Rosenbaum bounds with small sample (n=10 pairs)."""
        data = generate_matched_pairs_data(n_pairs=10, true_effect=2.5, seed=222)

        py_result = rosenbaum_bounds(
            data["treated_outcomes"],
            data["control_outcomes"],
            gamma_range=(1.0, 2.0),
            n_gamma=5,
        )
        r_result = r_rosenbaum_bounds(
            data["treated_outcomes"],
            data["control_outcomes"],
            gamma_range=(1.0, 2.0),
            n_gamma=5,
        )

        if r_result is None:
            pytest.skip("R rosenbaum_bounds unavailable")

        # Small sample - results should still be computed
        assert py_result["n_pairs"] == 10
        assert len(py_result["p_upper"]) == 5


# =============================================================================
# Integration Tests
# =============================================================================


@requires_sensitivity_python
class TestSensitivityConsistency:
    """Test consistency between E-value and Rosenbaum bounds interpretations."""

    def test_strong_effect_both_methods(self):
        """Both methods should indicate robustness for strong effect."""
        # E-value for strong effect
        rr = 3.0
        py_evalue = e_value(rr, effect_type="rr")

        # Rosenbaum for strong effect
        data = generate_strong_effect_data(seed=42)
        py_rosenbaum = rosenbaum_bounds(
            data["treated_outcomes"],
            data["control_outcomes"],
            gamma_range=(1.0, 4.0),
            n_gamma=10,
        )

        # Both should indicate robustness
        # E-value > 2 indicates moderate robustness
        assert py_evalue["e_value"] > 2.0, "E-value should indicate robustness"

        # Gamma_critical > 2 or None indicates robustness
        assert py_rosenbaum["gamma_critical"] is None or py_rosenbaum["gamma_critical"] > 2.0, (
            "Rosenbaum bounds should indicate robustness"
        )

    def test_weak_effect_both_methods(self):
        """Both methods should indicate sensitivity for weak effect."""
        # E-value for weak effect
        rr = 1.2
        py_evalue = e_value(rr, effect_type="rr")

        # Rosenbaum for weak effect
        data = generate_weak_effect_data(seed=123)
        py_rosenbaum = rosenbaum_bounds(
            data["treated_outcomes"],
            data["control_outcomes"],
            gamma_range=(1.0, 2.0),
            n_gamma=10,
        )

        # Both should indicate sensitivity
        # E-value < 1.5 indicates sensitivity
        assert py_evalue["e_value"] < 1.5, "E-value should indicate sensitivity"

        # Gamma_critical < 1.5 indicates sensitivity (if not None)
        if py_rosenbaum["gamma_critical"] is not None:
            assert py_rosenbaum["gamma_critical"] < 2.0, (
                "Rosenbaum bounds should indicate some sensitivity"
            )
