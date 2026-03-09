"""
Adversarial Tests for Sensitivity Analysis Module.

Tests edge cases, boundary conditions, and error handling for:
1. E-value: Invalid inputs, extreme values, edge case conversions
2. Rosenbaum bounds: Invalid inputs, degenerate data, numerical stability

Session 67: Python Sensitivity Adversarial Tests
"""

import numpy as np
import pytest

from src.causal_inference.sensitivity import e_value, rosenbaum_bounds


# =============================================================================
# E-Value Input Validation Tests
# =============================================================================


class TestEValueInputValidation:
    """Test E-value input validation and error handling."""

    def test_invalid_effect_type(self):
        """Invalid effect_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown effect_type"):
            e_value(2.0, effect_type="invalid")

        with pytest.raises(ValueError, match="Unknown effect_type"):
            e_value(2.0, effect_type="")

        with pytest.raises(ValueError, match="Unknown effect_type"):
            e_value(2.0, effect_type="RR")  # Case sensitive

    def test_nan_estimate(self):
        """NaN estimate should propagate or raise error."""
        result = e_value(np.nan, effect_type="rr")
        assert np.isnan(result["e_value"]) or result["e_value"] == 1.0

    def test_inf_estimate(self):
        """Infinite estimate should be handled."""
        # Positive infinity
        result_pos = e_value(np.inf, effect_type="rr")
        assert np.isinf(result_pos["e_value"]) or result_pos["e_value"] > 1e10

        # Negative infinity should fail for RR
        with pytest.raises((ValueError, RuntimeWarning)):
            e_value(-np.inf, effect_type="rr")

    def test_zero_rr(self):
        """RR = 0 should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            e_value(0.0, effect_type="rr")

    def test_negative_rr(self):
        """Negative RR should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            e_value(-1.0, effect_type="rr")

        with pytest.raises(ValueError, match="positive"):
            e_value(-2.5, effect_type="rr")

    def test_ate_missing_baseline_risk(self):
        """ATE without baseline_risk should raise ValueError."""
        with pytest.raises(ValueError, match="baseline_risk is required"):
            e_value(0.1, effect_type="ate")

        with pytest.raises(ValueError, match="baseline_risk is required"):
            e_value(0.1, effect_type="ate", baseline_risk=None)

    def test_ate_invalid_baseline_risk(self):
        """Invalid baseline_risk values should raise ValueError."""
        # baseline_risk = 0
        with pytest.raises(ValueError, match="baseline_risk must be in"):
            e_value(0.1, effect_type="ate", baseline_risk=0.0)

        # baseline_risk = 1
        with pytest.raises(ValueError, match="baseline_risk must be in"):
            e_value(0.1, effect_type="ate", baseline_risk=1.0)

        # baseline_risk < 0
        with pytest.raises(ValueError, match="baseline_risk must be in"):
            e_value(0.1, effect_type="ate", baseline_risk=-0.1)

        # baseline_risk > 1
        with pytest.raises(ValueError, match="baseline_risk must be in"):
            e_value(0.1, effect_type="ate", baseline_risk=1.5)

    def test_ate_invalid_treated_risk(self):
        """ATE producing invalid treated_risk should raise ValueError."""
        # ATE too negative: treated_risk = 0.2 + (-0.3) = -0.1
        with pytest.raises(ValueError, match="Treated risk.*must be > 0"):
            e_value(-0.3, effect_type="ate", baseline_risk=0.2)

        # ATE too positive: treated_risk = 0.8 + 0.3 = 1.1
        with pytest.raises(ValueError, match="Treated risk.*must be <= 1"):
            e_value(0.3, effect_type="ate", baseline_risk=0.8)


class TestEValueCIValidation:
    """Test E-value confidence interval handling."""

    def test_ci_lower_greater_than_upper(self):
        """CI with lower > upper should produce reasonable result."""
        # The function doesn't explicitly validate CI order
        # This tests current behavior
        result = e_value(2.0, ci_lower=2.5, ci_upper=1.5, effect_type="rr")
        # Should still produce some result
        assert "e_value" in result
        assert "e_value_ci" in result

    def test_ci_with_negative_values_rr(self):
        """Negative CI bounds for RR should produce e_value_ci=1.0 or handle gracefully."""
        result = e_value(2.0, ci_lower=-0.5, ci_upper=3.0, effect_type="rr")
        # CI crosses null (1.0) so e_value_ci should be 1.0
        assert result["e_value_ci"] == pytest.approx(1.0, abs=0.01)

    def test_ci_nan_values(self):
        """NaN in CI should be handled."""
        result1 = e_value(2.0, ci_lower=np.nan, ci_upper=3.0, effect_type="rr")
        result2 = e_value(2.0, ci_lower=1.5, ci_upper=np.nan, effect_type="rr")
        # Should produce some result without crashing
        assert "e_value" in result1
        assert "e_value" in result2

    def test_ci_inf_values(self):
        """Infinite CI bounds should be handled."""
        result1 = e_value(2.0, ci_lower=1.5, ci_upper=np.inf, effect_type="rr")
        result2 = e_value(2.0, ci_lower=-np.inf, ci_upper=3.0, effect_type="rr")
        # Should not crash
        assert "e_value" in result1
        assert "e_value" in result2


# =============================================================================
# E-Value Edge Cases
# =============================================================================


class TestEValueEdgeCases:
    """Test E-value behavior at edge cases."""

    def test_rr_exactly_one(self):
        """RR = 1.0 should give E-value = 1.0."""
        result = e_value(1.0, effect_type="rr")
        assert result["e_value"] == pytest.approx(1.0, abs=1e-10)

    def test_rr_very_close_to_one(self):
        """RR very close to 1.0 should give E-value close to 1.0."""
        result1 = e_value(1.0 + 1e-10, effect_type="rr")
        result2 = e_value(1.0 - 1e-10, effect_type="rr")

        assert result1["e_value"] == pytest.approx(1.0, abs=0.01)
        assert result2["e_value"] == pytest.approx(1.0, abs=0.01)

    def test_very_large_rr(self):
        """Very large RR should produce very large E-value."""
        result = e_value(100.0, effect_type="rr")
        # E = RR + sqrt(RR*(RR-1)) ≈ 100 + sqrt(100*99) ≈ 199.5
        expected = 100 + np.sqrt(100 * 99)
        assert result["e_value"] == pytest.approx(expected, rel=1e-6)

    def test_very_small_rr(self):
        """Very small RR (protective) should use 1/RR."""
        result = e_value(0.01, effect_type="rr")
        # Should use 1/RR = 100
        expected = 100 + np.sqrt(100 * 99)
        assert result["e_value"] == pytest.approx(expected, rel=1e-6)

    def test_smd_zero(self):
        """SMD = 0 should give E-value = 1.0."""
        result = e_value(0.0, effect_type="smd")
        # exp(0.91 * 0) = 1.0
        assert result["e_value"] == pytest.approx(1.0, abs=0.01)

    def test_smd_very_large_positive(self):
        """Very large positive SMD should produce large E-value."""
        result = e_value(5.0, effect_type="smd")
        # exp(0.91 * 5) ≈ 93.7
        expected_rr = np.exp(0.91 * 5)
        expected_e = expected_rr + np.sqrt(expected_rr * (expected_rr - 1))
        assert result["e_value"] == pytest.approx(expected_e, rel=0.01)

    def test_smd_very_large_negative(self):
        """Very large negative SMD (protective) should produce large E-value."""
        result = e_value(-5.0, effect_type="smd")
        # exp(0.91 * -5) ≈ 0.01, so use 1/0.01 = 100
        rr = np.exp(0.91 * -5)
        expected_e = (1 / rr) + np.sqrt((1 / rr) * (1 / rr - 1))
        assert result["e_value"] == pytest.approx(expected_e, rel=0.01)

    def test_ate_zero(self):
        """ATE = 0 should give RR = 1, E-value = 1.0."""
        result = e_value(0.0, effect_type="ate", baseline_risk=0.5)
        # treated_risk = 0.5, RR = 0.5/0.5 = 1.0
        assert result["e_value"] == pytest.approx(1.0, abs=0.01)
        assert result["rr_equivalent"] == pytest.approx(1.0, abs=0.01)

    def test_ate_very_small_baseline(self):
        """Very small baseline_risk should produce large RR."""
        result = e_value(0.01, effect_type="ate", baseline_risk=0.01)
        # treated_risk = 0.02, RR = 0.02/0.01 = 2.0
        assert result["rr_equivalent"] == pytest.approx(2.0, abs=0.01)


class TestEValueInterpretation:
    """Test E-value interpretation generation."""

    def test_interpretation_present(self):
        """Interpretation is always present and non-empty."""
        test_cases = [
            (1.0, "rr"),
            (2.0, "rr"),
            (0.5, "rr"),
            (0.5, "smd"),
            (0.1, "ate"),
        ]

        for estimate, effect_type in test_cases:
            kwargs = {}
            if effect_type == "ate":
                kwargs["baseline_risk"] = 0.3
            result = e_value(estimate, effect_type=effect_type, **kwargs)
            assert result["interpretation"], f"Empty interpretation for {effect_type}"
            assert len(result["interpretation"]) > 20

    def test_interpretation_harmful_vs_protective(self):
        """Interpretation correctly identifies harmful vs protective effects."""
        harmful = e_value(2.0, effect_type="rr")
        protective = e_value(0.5, effect_type="rr")

        assert "harmful" in harmful["interpretation"].lower()
        assert "protective" in protective["interpretation"].lower()


# =============================================================================
# Rosenbaum Bounds Input Validation Tests
# =============================================================================


class TestRosenbaumInputValidation:
    """Test Rosenbaum bounds input validation and error handling."""

    def test_mismatched_array_lengths(self):
        """Mismatched array lengths should raise ValueError."""
        treated = np.array([1.0, 2.0, 3.0])
        control = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="same length"):
            rosenbaum_bounds(treated, control)

    def test_too_few_pairs(self):
        """Less than 2 pairs should raise ValueError."""
        treated = np.array([1.0])
        control = np.array([0.0])

        with pytest.raises(ValueError, match="at least 2 pairs"):
            rosenbaum_bounds(treated, control)

    def test_empty_arrays(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 pairs"):
            rosenbaum_bounds(np.array([]), np.array([]))

    def test_invalid_gamma_range_lower_bound(self):
        """Gamma lower bound < 1 should raise ValueError."""
        treated = np.array([1.0, 2.0, 3.0])
        control = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="Gamma must be >= 1.0"):
            rosenbaum_bounds(treated, control, gamma_range=(0.5, 2.0))

        with pytest.raises(ValueError, match="Gamma must be >= 1.0"):
            rosenbaum_bounds(treated, control, gamma_range=(0.0, 2.0))

        with pytest.raises(ValueError, match="Gamma must be >= 1.0"):
            rosenbaum_bounds(treated, control, gamma_range=(-1.0, 2.0))

    def test_invalid_gamma_range_inverted(self):
        """Gamma upper < lower should raise ValueError."""
        treated = np.array([1.0, 2.0, 3.0])
        control = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="upper.*must be >= lower"):
            rosenbaum_bounds(treated, control, gamma_range=(3.0, 2.0))

    def test_nan_in_treated(self):
        """NaN in treated outcomes should be handled."""
        treated = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        control = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Should either handle gracefully or produce result with NaN
        result = rosenbaum_bounds(treated, control)
        assert "gamma_values" in result

    def test_nan_in_control(self):
        """NaN in control outcomes should be handled."""
        treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        control = np.array([0.0, np.nan, 0.0, 0.0, 0.0])

        result = rosenbaum_bounds(treated, control)
        assert "gamma_values" in result

    def test_inf_in_treated(self):
        """Infinite values in treated should be handled."""
        treated = np.array([1.0, np.inf, 3.0, 4.0, 5.0])
        control = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        result = rosenbaum_bounds(treated, control)
        assert "gamma_values" in result

    def test_inf_in_control(self):
        """Infinite values in control should be handled."""
        treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        control = np.array([0.0, -np.inf, 0.0, 0.0, 0.0])

        result = rosenbaum_bounds(treated, control)
        assert "gamma_values" in result


class TestRosenbaumAlphaValidation:
    """Test Rosenbaum bounds alpha parameter validation."""

    def test_alpha_zero(self):
        """Alpha = 0 should be handled."""
        np.random.seed(42)
        treated = np.random.randn(20) + 2.0
        control = np.random.randn(20)

        result = rosenbaum_bounds(treated, control, alpha=0.0)
        # With alpha=0, gamma_critical should be None (nothing significant)
        assert result["alpha"] == 0.0

    def test_alpha_one(self):
        """Alpha = 1 should be handled."""
        np.random.seed(42)
        treated = np.random.randn(20) + 2.0
        control = np.random.randn(20)

        result = rosenbaum_bounds(treated, control, alpha=1.0)
        assert result["alpha"] == 1.0

    def test_alpha_negative(self):
        """Negative alpha should produce result (no explicit validation)."""
        np.random.seed(42)
        treated = np.random.randn(20) + 2.0
        control = np.random.randn(20)

        # Current implementation doesn't validate alpha bounds
        result = rosenbaum_bounds(treated, control, alpha=-0.1)
        assert "gamma_values" in result

    def test_alpha_greater_than_one(self):
        """Alpha > 1 should produce result (no explicit validation)."""
        np.random.seed(42)
        treated = np.random.randn(20) + 2.0
        control = np.random.randn(20)

        result = rosenbaum_bounds(treated, control, alpha=1.5)
        assert "gamma_values" in result


# =============================================================================
# Rosenbaum Bounds Edge Cases
# =============================================================================


class TestRosenbaumEdgeCases:
    """Test Rosenbaum bounds behavior at edge cases."""

    def test_exactly_two_pairs(self):
        """Minimum viable: exactly 2 pairs."""
        treated = np.array([2.0, 3.0])
        control = np.array([0.0, 0.0])

        result = rosenbaum_bounds(treated, control)
        assert result["n_pairs"] == 2
        assert len(result["gamma_values"]) > 0

    def test_all_zero_differences(self):
        """All pairs with zero difference should be handled."""
        treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        control = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = rosenbaum_bounds(treated, control)
        # With all zero differences, result should indicate no effect
        assert "Too few non-zero differences" in result["interpretation"]

    def test_one_nonzero_difference(self):
        """Only one non-zero difference should be handled."""
        treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        control = np.array([1.0, 2.0, 3.0, 4.0, 4.0])  # Only last differs

        result = rosenbaum_bounds(treated, control)
        assert "Too few non-zero differences" in result["interpretation"]

    def test_constant_treated(self):
        """Constant treated outcomes should be handled."""
        treated = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        control = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = rosenbaum_bounds(treated, control)
        assert "gamma_values" in result
        assert result["n_pairs"] == 5

    def test_constant_control(self):
        """Constant control outcomes should be handled."""
        treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        control = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        result = rosenbaum_bounds(treated, control)
        assert "gamma_values" in result
        assert result["n_pairs"] == 5

    def test_all_positive_differences(self):
        """All positive differences (strong effect)."""
        np.random.seed(42)
        n = 30
        treated = np.random.randn(n) + 10.0  # Large positive effect
        control = np.random.randn(n)

        result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 5.0))
        # Should be robust with such strong effect
        assert result["gamma_critical"] is None or result["gamma_critical"] > 3.0

    def test_all_negative_differences(self):
        """All negative differences (protective effect)."""
        np.random.seed(42)
        n = 30
        treated = np.random.randn(n)
        control = np.random.randn(n) + 10.0  # Control much higher

        result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 5.0))
        # With one-sided test for positive effect, this should be non-significant
        assert result["p_upper"][0] > 0.5  # Upper p-value at gamma=1 should be high

    def test_mixed_sign_differences(self):
        """Mixed positive/negative differences."""
        treated = np.array([5.0, 1.0, 5.0, 1.0, 5.0, 1.0])
        control = np.array([0.0, 6.0, 0.0, 6.0, 0.0, 6.0])

        result = rosenbaum_bounds(treated, control)
        # Cancellation should make effect less robust
        assert "gamma_values" in result


class TestRosenbaumNumericalStability:
    """Test Rosenbaum bounds numerical stability."""

    def test_very_large_outcomes(self):
        """Very large outcome values should be handled."""
        scale = 1e8
        np.random.seed(42)
        treated = np.random.randn(20) * scale + scale
        control = np.random.randn(20) * scale

        result = rosenbaum_bounds(treated, control)
        assert "gamma_values" in result
        assert not np.any(np.isnan(result["p_upper"]))
        assert not np.any(np.isnan(result["p_lower"]))

    def test_very_small_outcomes(self):
        """Very small outcome values should be handled."""
        scale = 1e-8
        np.random.seed(42)
        treated = np.random.randn(20) * scale + 2 * scale
        control = np.random.randn(20) * scale

        result = rosenbaum_bounds(treated, control)
        assert "gamma_values" in result
        assert not np.any(np.isnan(result["p_upper"]))

    def test_mixed_scale_outcomes(self):
        """Mix of large and small values should be handled."""
        treated = np.array([1e10, 1e-10, 1e5, 1e-5, 1.0] * 4)
        control = np.array([0.0] * 20)

        result = rosenbaum_bounds(treated, control)
        assert "gamma_values" in result

    def test_nearly_identical_pairs(self):
        """Very small differences should be handled."""
        eps = 1e-15
        treated = np.array([1.0 + eps, 2.0 + eps, 3.0 + eps, 4.0 + eps, 5.0 + eps])
        control = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = rosenbaum_bounds(treated, control)
        assert "gamma_values" in result


class TestRosenbaumGammaRange:
    """Test Rosenbaum bounds gamma range handling."""

    def test_single_gamma_value(self):
        """gamma_range with same lower and upper."""
        np.random.seed(42)
        treated = np.random.randn(20) + 2.0
        control = np.random.randn(20)

        result = rosenbaum_bounds(treated, control, gamma_range=(2.0, 2.0), n_gamma=1)
        assert len(result["gamma_values"]) == 1
        assert result["gamma_values"][0] == pytest.approx(2.0)

    def test_wide_gamma_range(self):
        """Very wide gamma range should be handled."""
        np.random.seed(42)
        treated = np.random.randn(20) + 2.0
        control = np.random.randn(20)

        result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 100.0), n_gamma=50)
        assert len(result["gamma_values"]) == 50
        assert result["gamma_values"][0] == pytest.approx(1.0)
        assert result["gamma_values"][-1] == pytest.approx(100.0)

    def test_narrow_gamma_range(self):
        """Very narrow gamma range should be handled."""
        np.random.seed(42)
        treated = np.random.randn(20) + 2.0
        control = np.random.randn(20)

        result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 1.01), n_gamma=10)
        assert len(result["gamma_values"]) == 10

    def test_n_gamma_one(self):
        """n_gamma=1 should work."""
        np.random.seed(42)
        treated = np.random.randn(20) + 2.0
        control = np.random.randn(20)

        result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 3.0), n_gamma=1)
        assert len(result["gamma_values"]) == 1

    def test_large_n_gamma(self):
        """Large n_gamma should work."""
        np.random.seed(42)
        treated = np.random.randn(20) + 2.0
        control = np.random.randn(20)

        result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 3.0), n_gamma=100)
        assert len(result["gamma_values"]) == 100


class TestRosenbaumInterpretation:
    """Test Rosenbaum bounds interpretation generation."""

    def test_interpretation_always_present(self):
        """Interpretation is always present."""
        test_cases = [
            (np.array([1, 2, 3, 4, 5]) + 10.0, np.array([1, 2, 3, 4, 5])),  # Strong
            (np.array([1, 2, 3, 4, 5]) + 0.1, np.array([1, 2, 3, 4, 5])),  # Weak
            (np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])),  # Null
        ]

        for treated, control in test_cases:
            result = rosenbaum_bounds(treated, control)
            assert result["interpretation"], "Empty interpretation"
            assert len(result["interpretation"]) > 20

    def test_interpretation_mentions_pairs(self):
        """Interpretation mentions number of pairs."""
        np.random.seed(42)
        treated = np.random.randn(25) + 2.0
        control = np.random.randn(25)

        result = rosenbaum_bounds(treated, control)
        assert "25" in result["interpretation"] or "pair" in result["interpretation"].lower()


# =============================================================================
# Data Type Tests
# =============================================================================


class TestEValueDataTypes:
    """Test E-value with various data types."""

    def test_integer_estimate(self):
        """Integer estimate should work."""
        result = e_value(2, effect_type="rr")
        assert result["e_value"] == pytest.approx(3.414, rel=0.01)

    def test_float32_estimate(self):
        """Float32 estimate should work."""
        result = e_value(np.float32(2.0), effect_type="rr")
        assert result["e_value"] == pytest.approx(3.414, rel=0.01)

    def test_numpy_scalar(self):
        """NumPy scalar should work."""
        result = e_value(np.float64(2.0), effect_type="rr")
        assert result["e_value"] == pytest.approx(3.414, rel=0.01)


class TestRosenbaumDataTypes:
    """Test Rosenbaum bounds with various data types."""

    def test_integer_arrays(self):
        """Integer arrays should work."""
        treated = np.array([5, 6, 7, 8, 9])
        control = np.array([1, 2, 3, 4, 5])

        result = rosenbaum_bounds(treated, control)
        assert result["n_pairs"] == 5

    def test_float32_arrays(self):
        """Float32 arrays should work."""
        treated = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
        control = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        result = rosenbaum_bounds(treated, control)
        assert result["n_pairs"] == 5

    def test_list_inputs(self):
        """List inputs should work."""
        treated = [5.0, 6.0, 7.0, 8.0, 9.0]
        control = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = rosenbaum_bounds(treated, control)
        assert result["n_pairs"] == 5

    def test_mixed_types(self):
        """Mixed array types should work."""
        treated = np.array([5, 6, 7, 8, 9], dtype=np.int32)
        control = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = rosenbaum_bounds(treated, control)
        assert result["n_pairs"] == 5


# =============================================================================
# Result Structure Tests
# =============================================================================


class TestEValueResultStructure:
    """Test E-value result dictionary structure."""

    def test_all_keys_present(self):
        """All expected keys are present in result."""
        result = e_value(2.0, ci_lower=1.5, ci_upper=2.5, effect_type="rr")

        expected_keys = ["e_value", "e_value_ci", "rr_equivalent", "effect_type", "interpretation"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_correct_effect_type_returned(self):
        """Effect type is correctly recorded in result."""
        for effect_type in ["rr", "or", "hr", "smd"]:
            result = e_value(2.0, effect_type=effect_type)
            assert result["effect_type"] == effect_type

        result_ate = e_value(0.1, effect_type="ate", baseline_risk=0.3)
        assert result_ate["effect_type"] == "ate"


class TestRosenbaumResultStructure:
    """Test Rosenbaum bounds result dictionary structure."""

    def test_all_keys_present(self):
        """All expected keys are present in result."""
        np.random.seed(42)
        treated = np.random.randn(20) + 2.0
        control = np.random.randn(20)

        result = rosenbaum_bounds(treated, control)

        expected_keys = [
            "gamma_values",
            "p_upper",
            "p_lower",
            "gamma_critical",
            "observed_statistic",
            "n_pairs",
            "alpha",
            "interpretation",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_array_lengths_match(self):
        """Gamma values and p-value arrays have same length."""
        np.random.seed(42)
        treated = np.random.randn(20) + 2.0
        control = np.random.randn(20)

        result = rosenbaum_bounds(treated, control, n_gamma=25)

        assert len(result["gamma_values"]) == 25
        assert len(result["p_upper"]) == 25
        assert len(result["p_lower"]) == 25

    def test_n_pairs_correct(self):
        """n_pairs matches input array length."""
        for n in [5, 10, 50, 100]:
            np.random.seed(42)
            treated = np.random.randn(n) + 2.0
            control = np.random.randn(n)

            result = rosenbaum_bounds(treated, control)
            assert result["n_pairs"] == n

    def test_alpha_recorded(self):
        """Alpha parameter is recorded in result."""
        np.random.seed(42)
        treated = np.random.randn(20) + 2.0
        control = np.random.randn(20)

        for alpha in [0.01, 0.05, 0.10]:
            result = rosenbaum_bounds(treated, control, alpha=alpha)
            assert result["alpha"] == alpha
