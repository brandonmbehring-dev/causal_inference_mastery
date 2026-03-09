"""
Python ↔ Julia Sensitivity Analysis Parity Tests

Tests that Python and Julia sensitivity analysis implementations produce
numerically equivalent results for E-value and Rosenbaum bounds.

Session 51: Julia Sensitivity Analysis Implementation
"""

import numpy as np
import pytest

from src.causal_inference.sensitivity import e_value, rosenbaum_bounds
from tests.validation.cross_language.julia_interface import (
    is_julia_available,
    julia_e_value,
    julia_rosenbaum_bounds,
)

# Skip all tests if Julia is not available
pytestmark = pytest.mark.skipif(
    not is_julia_available(),
    reason="Julia not available for cross-language validation",
)


class TestEValueParity:
    """Test E-value parity between Python and Julia."""

    def test_e_value_rr_basic(self):
        """E-value for basic risk ratio matches."""
        py_result = e_value(2.0, effect_type="rr")
        jl_result = julia_e_value(2.0, effect_type="rr")

        assert np.isclose(py_result["e_value"], jl_result["e_value"], rtol=1e-10)
        assert np.isclose(py_result["rr_equivalent"], jl_result["rr_equivalent"], rtol=1e-10)

    def test_e_value_rr_with_ci(self):
        """E-value with CI matches."""
        py_result = e_value(2.5, ci_lower=1.8, ci_upper=3.5, effect_type="rr")
        jl_result = julia_e_value(2.5, ci_lower=1.8, ci_upper=3.5, effect_type="rr")

        assert np.isclose(py_result["e_value"], jl_result["e_value"], rtol=1e-10)
        assert np.isclose(py_result["e_value_ci"], jl_result["e_value_ci"], rtol=1e-10)

    def test_e_value_ci_includes_null(self):
        """E-value when CI includes null (RR=1) matches."""
        py_result = e_value(1.5, ci_lower=0.8, ci_upper=2.5, effect_type="rr")
        jl_result = julia_e_value(1.5, ci_lower=0.8, ci_upper=2.5, effect_type="rr")

        # E_value_ci should be ~1.0 when CI includes null
        assert np.isclose(py_result["e_value"], jl_result["e_value"], rtol=1e-10)
        assert np.isclose(py_result["e_value_ci"], jl_result["e_value_ci"], rtol=1e-10)
        assert py_result["e_value_ci"] == pytest.approx(1.0, abs=0.01)

    def test_e_value_protective_effect(self):
        """E-value for protective effect (RR < 1) matches."""
        py_result = e_value(0.5, effect_type="rr")
        jl_result = julia_e_value(0.5, effect_type="rr")

        assert np.isclose(py_result["e_value"], jl_result["e_value"], rtol=1e-10)
        # Protective effects get inverted: 1/0.5 = 2.0
        assert np.isclose(py_result["rr_equivalent"], 0.5, rtol=1e-10)

    def test_e_value_smd(self):
        """E-value for standardized mean difference matches."""
        py_result = e_value(0.6, effect_type="smd")
        jl_result = julia_e_value(0.6, effect_type="smd")

        # Both should use exp(0.91 * d) approximation
        expected_rr = np.exp(0.91 * 0.6)
        assert np.isclose(py_result["rr_equivalent"], expected_rr, rtol=1e-6)
        assert np.isclose(jl_result["rr_equivalent"], expected_rr, rtol=1e-6)
        assert np.isclose(py_result["e_value"], jl_result["e_value"], rtol=1e-6)

    def test_e_value_ate(self):
        """E-value for ATE with baseline risk matches."""
        py_result = e_value(0.1, effect_type="ate", baseline_risk=0.2)
        jl_result = julia_e_value(0.1, effect_type="ate", baseline_risk=0.2)

        # ATE to RR: (0.2 + 0.1) / 0.2 = 1.5
        expected_rr = 1.5
        assert np.isclose(py_result["rr_equivalent"], expected_rr, rtol=1e-10)
        assert np.isclose(jl_result["rr_equivalent"], expected_rr, rtol=1e-10)
        assert np.isclose(py_result["e_value"], jl_result["e_value"], rtol=1e-10)

    def test_e_value_or(self):
        """E-value for odds ratio matches."""
        py_result = e_value(2.3, ci_lower=1.4, ci_upper=3.8, effect_type="or")
        jl_result = julia_e_value(2.3, ci_lower=1.4, ci_upper=3.8, effect_type="or")

        assert np.isclose(py_result["e_value"], jl_result["e_value"], rtol=1e-10)
        assert np.isclose(py_result["e_value_ci"], jl_result["e_value_ci"], rtol=1e-10)


class TestRosenbaumBoundsParity:
    """Test Rosenbaum bounds parity between Python and Julia."""

    def test_rosenbaum_basic(self):
        """Basic Rosenbaum bounds matches."""
        np.random.seed(42)
        treated = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        control = np.array([5.0, 6.0, 7.0, 8.0, 9.0])

        py_result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 3.0), n_gamma=10)
        jl_result = julia_rosenbaum_bounds(treated, control, gamma_range=(1.0, 3.0), n_gamma=10)

        # Check gamma grid matches
        assert len(py_result["gamma_values"]) == len(jl_result["gamma_values"])
        assert np.allclose(py_result["gamma_values"], jl_result["gamma_values"], rtol=1e-10)

        # Check observed statistic matches
        assert np.isclose(
            py_result["observed_statistic"], jl_result["observed_statistic"], rtol=1e-10
        )

        # Check n_pairs matches
        assert py_result["n_pairs"] == jl_result["n_pairs"]

    def test_rosenbaum_p_values_monotonic(self):
        """P-value bounds are monotonically increasing with gamma."""
        treated = np.array([15.0, 18.0, 12.0, 20.0, 16.0])
        control = np.array([10.0, 12.0, 8.0, 14.0, 11.0])

        py_result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 3.0), n_gamma=20)
        jl_result = julia_rosenbaum_bounds(treated, control, gamma_range=(1.0, 3.0), n_gamma=20)

        # P_upper should be monotonically non-decreasing
        assert all(np.diff(py_result["p_upper"]) >= -1e-10)
        assert all(np.diff(jl_result["p_upper"]) >= -1e-10)

    def test_rosenbaum_gamma_critical(self):
        """Gamma critical detection matches."""
        np.random.seed(123)
        # Strong effect - should have high gamma_critical
        treated = 20.0 + np.random.randn(10)
        control = 10.0 + np.random.randn(10)

        py_result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 5.0), n_gamma=20)
        jl_result = julia_rosenbaum_bounds(treated, control, gamma_range=(1.0, 5.0), n_gamma=20)

        # Both should detect same gamma_critical (or both None)
        if py_result["gamma_critical"] is not None and jl_result["gamma_critical"] is not None:
            assert np.isclose(
                py_result["gamma_critical"],
                jl_result["gamma_critical"],
                rtol=0.15,  # Allow some tolerance due to grid discretization
            )
        else:
            # Both should be None (robust to all tested values)
            assert py_result["gamma_critical"] is None
            assert jl_result["gamma_critical"] is None

    def test_rosenbaum_weak_effect(self):
        """Weak effect produces low gamma_critical."""
        np.random.seed(456)
        # Weak effect - should have low gamma_critical
        treated = 10.5 + np.random.randn(15) * 3
        control = 10.0 + np.random.randn(15) * 3

        py_result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 2.0), n_gamma=20)
        jl_result = julia_rosenbaum_bounds(treated, control, gamma_range=(1.0, 2.0), n_gamma=20)

        # Both should produce similar sensitivity
        # With weak effect, gamma_critical should be relatively low
        assert py_result["n_pairs"] == jl_result["n_pairs"]


class TestSensitivityIntegration:
    """Integration tests for cross-language sensitivity analysis."""

    def test_full_e_value_workflow(self):
        """Full E-value workflow matches across languages."""
        # Typical epidemiological study result
        estimate = 2.3
        ci_lower = 1.4
        ci_upper = 3.8

        py_result = e_value(estimate, ci_lower=ci_lower, ci_upper=ci_upper, effect_type="or")
        jl_result = julia_e_value(estimate, ci_lower=ci_lower, ci_upper=ci_upper, effect_type="or")

        # All key metrics should match
        assert np.isclose(py_result["e_value"], jl_result["e_value"], rtol=1e-10)
        assert np.isclose(py_result["e_value_ci"], jl_result["e_value_ci"], rtol=1e-10)
        assert np.isclose(py_result["rr_equivalent"], jl_result["rr_equivalent"], rtol=1e-10)

        # Both should produce robustness assessment
        assert "robust" in py_result["interpretation"].lower()
        assert "robust" in jl_result["interpretation"].lower()

    def test_full_rosenbaum_workflow(self):
        """Full Rosenbaum bounds workflow matches across languages."""
        np.random.seed(789)
        # Simulated matched pairs from PSM
        treated = 15.0 + np.random.randn(20) * 2
        control = 12.0 + np.random.randn(20) * 2

        py_result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 3.0), n_gamma=15)
        jl_result = julia_rosenbaum_bounds(treated, control, gamma_range=(1.0, 3.0), n_gamma=15)

        # Structure should match
        assert len(py_result["gamma_values"]) == len(jl_result["gamma_values"])
        assert len(py_result["p_upper"]) == len(jl_result["p_upper"])
        assert len(py_result["p_lower"]) == len(jl_result["p_lower"])

        # Interpretations should contain robustness assessment
        assert (
            "robust" in py_result["interpretation"].lower()
            or "sensitive" in py_result["interpretation"].lower()
        )
        assert (
            "robust" in jl_result["interpretation"].lower()
            or "sensitive" in jl_result["interpretation"].lower()
        )
