"""Tests for E-value sensitivity analysis.

Tests are organized by layer:
1. Known-Answer: Verify E-value formula against published values
2. Effect Type Conversions: Test SMD, ATE, OR, HR conversions
3. Edge Cases: Null effects, extreme values, CI crossing null
4. Input Validation: Error handling for invalid inputs
"""

import pytest
import numpy as np

from src.causal_inference.sensitivity import e_value


# ============================================================================
# Layer 1: Known-Answer Tests
# ============================================================================


class TestEValueKnownAnswer:
    """Tests with known E-values from VanderWeele & Ding (2017)."""

    def test_rr_2_0(self):
        """E-value for RR=2.0 should be ~3.41.

        Formula: E = RR + sqrt(RR * (RR - 1))
                 E = 2.0 + sqrt(2.0 * 1.0) = 2.0 + 1.414 = 3.414
        """
        result = e_value(2.0, effect_type="rr")

        expected = 2.0 + np.sqrt(2.0 * 1.0)
        assert np.isclose(result["e_value"], expected, rtol=1e-10), (
            f"E-value {result['e_value']:.4f} != expected {expected:.4f}"
        )

    def test_rr_1_5(self):
        """E-value for RR=1.5 should be ~2.37.

        E = 1.5 + sqrt(1.5 * 0.5) = 1.5 + 0.866 = 2.366
        """
        result = e_value(1.5, effect_type="rr")

        expected = 1.5 + np.sqrt(1.5 * 0.5)
        assert np.isclose(result["e_value"], expected, rtol=1e-10)

    def test_rr_3_0(self):
        """E-value for RR=3.0 should be ~5.45.

        E = 3.0 + sqrt(3.0 * 2.0) = 3.0 + 2.449 = 5.449
        """
        result = e_value(3.0, effect_type="rr")

        expected = 3.0 + np.sqrt(3.0 * 2.0)
        assert np.isclose(result["e_value"], expected, rtol=1e-10)

    def test_protective_effect_rr_0_5(self):
        """E-value for protective effect (RR=0.5) uses 1/RR=2.0.

        Should give same E-value as RR=2.0.
        """
        result = e_value(0.5, effect_type="rr")

        # Should use 1/0.5 = 2.0
        expected = 2.0 + np.sqrt(2.0 * 1.0)
        assert np.isclose(result["e_value"], expected, rtol=1e-10)

    def test_rr_equivalent_stored(self):
        """Result should store the RR equivalent."""
        result = e_value(2.0, effect_type="rr")
        assert result["rr_equivalent"] == 2.0

    def test_effect_type_stored(self):
        """Result should store the effect type."""
        result = e_value(2.0, effect_type="rr")
        assert result["effect_type"] == "rr"


class TestEValueWithCI:
    """Tests for E-value computation with confidence intervals."""

    def test_ci_not_including_null(self):
        """E-value for CI when CI doesn't include null."""
        # RR=2.0 with CI [1.5, 2.7]
        result = e_value(2.0, ci_lower=1.5, ci_upper=2.7, effect_type="rr")

        # E-value for point estimate
        expected_point = 2.0 + np.sqrt(2.0 * 1.0)
        assert np.isclose(result["e_value"], expected_point, rtol=1e-10)

        # E-value for CI: use lower bound (1.5) since it's closest to null
        expected_ci = 1.5 + np.sqrt(1.5 * 0.5)
        assert np.isclose(result["e_value_ci"], expected_ci, rtol=1e-10)

    def test_ci_including_null_returns_1(self):
        """E-value_CI should be 1.0 when CI includes null (RR=1)."""
        # RR=1.5 with CI [0.8, 2.5] - includes 1.0
        result = e_value(1.5, ci_lower=0.8, ci_upper=2.5, effect_type="rr")

        assert result["e_value_ci"] == 1.0, (
            f"E-value_CI should be 1.0 when CI includes null, got {result['e_value_ci']}"
        )

    def test_protective_effect_ci(self):
        """E-value for protective effect CI."""
        # RR=0.5 with CI [0.3, 0.7]
        result = e_value(0.5, ci_lower=0.3, ci_upper=0.7, effect_type="rr")

        # Point estimate: use 1/0.5 = 2.0
        expected_point = 2.0 + np.sqrt(2.0 * 1.0)
        assert np.isclose(result["e_value"], expected_point, rtol=1e-10)

        # CI: upper bound 0.7 is closest to null, use 1/0.7 ≈ 1.43
        ci_rr = 1 / 0.7
        expected_ci = ci_rr + np.sqrt(ci_rr * (ci_rr - 1))
        assert np.isclose(result["e_value_ci"], expected_ci, rtol=1e-10)


# ============================================================================
# Layer 2: Effect Type Conversion Tests
# ============================================================================


class TestEValueConversions:
    """Tests for converting different effect types to RR."""

    def test_odds_ratio_approximation(self):
        """OR approximates RR (for rare outcomes)."""
        result = e_value(2.0, effect_type="or")

        # OR treated same as RR
        expected = 2.0 + np.sqrt(2.0 * 1.0)
        assert np.isclose(result["e_value"], expected, rtol=1e-10)
        assert result["effect_type"] == "or"

    def test_hazard_ratio_approximation(self):
        """HR approximates RR."""
        result = e_value(2.0, effect_type="hr")

        # HR treated same as RR
        expected = 2.0 + np.sqrt(2.0 * 1.0)
        assert np.isclose(result["e_value"], expected, rtol=1e-10)
        assert result["effect_type"] == "hr"

    def test_smd_conversion(self):
        """SMD converts to RR via exp(0.91 * d)."""
        # d = 0.5 -> RR = exp(0.91 * 0.5) = exp(0.455) ≈ 1.576
        result = e_value(0.5, effect_type="smd")

        rr = np.exp(0.91 * 0.5)
        expected = rr + np.sqrt(rr * (rr - 1))
        assert np.isclose(result["e_value"], expected, rtol=1e-10)
        assert np.isclose(result["rr_equivalent"], rr, rtol=1e-10)

    def test_smd_negative(self):
        """Negative SMD (protective) converts correctly."""
        # d = -0.5 -> RR = exp(-0.455) ≈ 0.634, use 1/RR ≈ 1.576
        result = e_value(-0.5, effect_type="smd")

        rr = np.exp(0.91 * (-0.5))  # ~0.634
        rr_for_evalue = 1 / rr  # ~1.576
        expected = rr_for_evalue + np.sqrt(rr_for_evalue * (rr_for_evalue - 1))
        assert np.isclose(result["e_value"], expected, rtol=1e-10)

    def test_ate_conversion(self):
        """ATE converts to RR via baseline risk."""
        # Baseline risk = 0.2, ATE = 0.1 -> treated risk = 0.3
        # RR = 0.3 / 0.2 = 1.5
        result = e_value(0.1, effect_type="ate", baseline_risk=0.2)

        rr = 0.3 / 0.2
        expected = rr + np.sqrt(rr * (rr - 1))
        assert np.isclose(result["e_value"], expected, rtol=1e-10)
        assert np.isclose(result["rr_equivalent"], 1.5, rtol=1e-10)

    def test_ate_protective(self):
        """ATE negative (protective) converts correctly."""
        # Baseline risk = 0.4, ATE = -0.2 -> treated risk = 0.2
        # RR = 0.2 / 0.4 = 0.5, use 1/RR = 2.0
        result = e_value(-0.2, effect_type="ate", baseline_risk=0.4)

        rr = 0.2 / 0.4  # 0.5
        rr_for_evalue = 1 / rr  # 2.0
        expected = rr_for_evalue + np.sqrt(rr_for_evalue * (rr_for_evalue - 1))
        assert np.isclose(result["e_value"], expected, rtol=1e-10)


# ============================================================================
# Layer 3: Edge Cases
# ============================================================================


class TestEValueEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_null_effect_rr_1(self):
        """RR=1.0 (null effect) should give E-value=1.0."""
        result = e_value(1.0, effect_type="rr")
        assert result["e_value"] == 1.0

    def test_near_null_effect(self):
        """RR very close to 1 should give E-value close to 1."""
        result = e_value(1.001, effect_type="rr")
        assert result["e_value"] < 1.1  # Should be very close to 1

    def test_very_large_rr(self):
        """Very large RR should give large E-value."""
        result = e_value(10.0, effect_type="rr")

        expected = 10.0 + np.sqrt(10.0 * 9.0)  # ~19.49
        assert np.isclose(result["e_value"], expected, rtol=1e-10)
        assert result["e_value"] > 15  # Should be substantial

    def test_smd_zero(self):
        """SMD=0 (null effect) should give E-value=1.0."""
        result = e_value(0.0, effect_type="smd")

        # d=0 -> RR = exp(0) = 1.0 -> E-value = 1.0
        assert result["e_value"] == 1.0

    def test_interpretation_included(self):
        """Result should include interpretation string."""
        result = e_value(2.0, effect_type="rr")

        assert "interpretation" in result
        assert len(result["interpretation"]) > 0
        assert "E-value" in result["interpretation"]


# ============================================================================
# Layer 4: Input Validation
# ============================================================================


class TestEValueInputValidation:
    """Tests for input validation and error handling."""

    def test_ate_requires_baseline_risk(self):
        """ATE effect type requires baseline_risk parameter."""
        with pytest.raises(ValueError, match="baseline_risk is required"):
            e_value(0.1, effect_type="ate")

    def test_ate_baseline_risk_bounds(self):
        """baseline_risk must be in (0, 1)."""
        with pytest.raises(ValueError, match="baseline_risk must be in"):
            e_value(0.1, effect_type="ate", baseline_risk=0.0)

        with pytest.raises(ValueError, match="baseline_risk must be in"):
            e_value(0.1, effect_type="ate", baseline_risk=1.0)

        with pytest.raises(ValueError, match="baseline_risk must be in"):
            e_value(0.1, effect_type="ate", baseline_risk=-0.1)

    def test_ate_invalid_treated_risk(self):
        """ATE that results in invalid treated risk should error."""
        # Treated risk would be 0.2 - 0.3 = -0.1
        with pytest.raises(ValueError, match="Treated risk"):
            e_value(-0.3, effect_type="ate", baseline_risk=0.2)

        # Treated risk would be 0.8 + 0.3 = 1.1
        with pytest.raises(ValueError, match="Treated risk"):
            e_value(0.3, effect_type="ate", baseline_risk=0.8)

    def test_invalid_effect_type(self):
        """Unknown effect type should raise error."""
        with pytest.raises(ValueError, match="Unknown effect_type"):
            e_value(2.0, effect_type="invalid")

    def test_negative_rr_error(self):
        """Negative RR should raise error."""
        with pytest.raises(ValueError, match="must be positive"):
            e_value(-2.0, effect_type="rr")

    def test_zero_rr_error(self):
        """Zero RR should raise error."""
        with pytest.raises(ValueError, match="must be positive"):
            e_value(0.0, effect_type="rr")


# ============================================================================
# Integration Tests
# ============================================================================


class TestEValueIntegration:
    """Integration tests with typical use cases."""

    def test_typical_observational_study(self):
        """Typical observational study result."""
        # Hypothetical: OR=1.8 with 95% CI [1.2, 2.7]
        result = e_value(1.8, ci_lower=1.2, ci_upper=2.7, effect_type="or")

        # Should have reasonable E-values
        assert result["e_value"] > 1.5
        assert result["e_value_ci"] > 1.0
        assert (
            "moderately robust" in result["interpretation"].lower()
            or "strongly robust" in result["interpretation"].lower()
            or "weakly robust" in result["interpretation"].lower()
        )

    def test_non_significant_result(self):
        """Non-significant result (CI includes null)."""
        # OR=1.3 with 95% CI [0.9, 1.8]
        result = e_value(1.3, ci_lower=0.9, ci_upper=1.8, effect_type="or")

        # E-value_CI should be 1.0 since CI includes null
        assert result["e_value_ci"] == 1.0
        assert (
            "includes the null" in result["interpretation"].lower()
            or "no confounding is needed" in result["interpretation"].lower()
        )

    def test_clinical_trial_smd(self):
        """Clinical trial with standardized mean difference."""
        # d=0.8 (large effect) with CI [0.5, 1.1]
        result = e_value(0.8, ci_lower=0.5, ci_upper=1.1, effect_type="smd")

        # Large effect should have large E-value
        assert result["e_value"] > 3.0
        assert result["e_value_ci"] > 1.5
