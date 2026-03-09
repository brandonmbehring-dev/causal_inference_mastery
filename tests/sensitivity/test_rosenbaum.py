"""Tests for Rosenbaum bounds sensitivity analysis.

Tests are organized by layer:
1. Known-Answer: Verify bounds computation for known scenarios
2. Robustness Properties: Strong vs weak effects
3. Edge Cases: Small samples, ties, all same direction
4. Input Validation: Error handling for invalid inputs
"""

import pytest
import numpy as np

from src.causal_inference.sensitivity import rosenbaum_bounds


# ============================================================================
# Layer 1: Known-Answer Tests
# ============================================================================


class TestRosenbaumKnownAnswer:
    """Tests with known properties of Rosenbaum bounds."""

    def test_gamma_1_equals_no_confounding(self):
        """At Gamma=1, should give standard test p-value."""
        np.random.seed(42)
        n = 30
        treated = np.random.randn(n) + 1.5
        control = np.random.randn(n)

        result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 1.0), n_gamma=1)

        # At Gamma=1, upper and lower p-values should be equal
        assert np.isclose(result["p_upper"][0], result["p_lower"][0], rtol=0.01)

    def test_p_values_increase_with_gamma(self):
        """Upper bound p-values should increase as Gamma increases."""
        np.random.seed(42)
        n = 50
        treated = np.random.randn(n) + 1.0
        control = np.random.randn(n)

        result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 3.0), n_gamma=10)

        # Upper p-values should be monotonically increasing
        for i in range(1, len(result["p_upper"])):
            assert result["p_upper"][i] >= result["p_upper"][i - 1] - 1e-10, (
                f"p_upper should increase with Gamma: "
                f"{result['p_upper'][i]} < {result['p_upper'][i - 1]}"
            )

    def test_lower_p_values_decrease_with_gamma(self):
        """Lower bound p-values should decrease as Gamma increases."""
        np.random.seed(42)
        n = 50
        treated = np.random.randn(n) + 1.0
        control = np.random.randn(n)

        result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 3.0), n_gamma=10)

        # Lower p-values should be monotonically decreasing
        for i in range(1, len(result["p_lower"])):
            assert result["p_lower"][i] <= result["p_lower"][i - 1] + 1e-10, (
                f"p_lower should decrease with Gamma"
            )

    def test_n_pairs_correct(self):
        """n_pairs should match input length."""
        treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        control = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

        result = rosenbaum_bounds(treated, control)

        assert result["n_pairs"] == 5

    def test_alpha_stored(self):
        """Alpha should be stored in result."""
        treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        control = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

        result = rosenbaum_bounds(treated, control, alpha=0.01)

        assert result["alpha"] == 0.01


# ============================================================================
# Layer 2: Robustness Property Tests
# ============================================================================


class TestRosenbaumRobustness:
    """Tests for sensitivity analysis robustness properties."""

    def test_strong_effect_large_gamma_critical(self):
        """Strong treatment effect should have large critical Gamma."""
        np.random.seed(123)
        n = 50
        # Very strong effect: all treated > control
        treated = np.random.randn(n) + 3.0  # Large shift
        control = np.random.randn(n)

        result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 5.0), n_gamma=30)

        # Strong effect should be robust
        # Either gamma_critical is large or None (robust to all tested)
        if result["gamma_critical"] is not None:
            assert result["gamma_critical"] > 2.0, (
                f"Strong effect should have gamma_critical > 2.0, got {result['gamma_critical']}"
            )

    def test_weak_effect_small_gamma_critical(self):
        """Weak treatment effect should have small critical Gamma."""
        np.random.seed(456)
        n = 50
        # Weak effect
        treated = np.random.randn(n) + 0.3
        control = np.random.randn(n)

        result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 2.0), n_gamma=20)

        # Weak effect should be sensitive
        assert result["gamma_critical"] is not None, (
            "Weak effect should have a critical Gamma within tested range"
        )
        assert result["gamma_critical"] < 1.8, (
            f"Weak effect should have small gamma_critical, got {result['gamma_critical']}"
        )

    def test_no_effect_very_sensitive(self):
        """No treatment effect should be very sensitive (gamma_critical near 1)."""
        np.random.seed(789)
        n = 30
        # No effect: treated and control same distribution
        treated = np.random.randn(n)
        control = np.random.randn(n)

        result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 1.5), n_gamma=10)

        # No effect: should be at null or gamma_critical = 1.0
        # p_upper at gamma=1 should be > 0.5 roughly
        assert result["p_upper"][0] > 0.05, "With no effect, p-value should be non-significant"

    def test_larger_sample_more_robust(self):
        """Larger samples should be more robust (larger gamma_critical)."""
        np.random.seed(101)
        effect_size = 0.7

        # Small sample
        n_small = 20
        treated_small = np.random.randn(n_small) + effect_size
        control_small = np.random.randn(n_small)
        result_small = rosenbaum_bounds(
            treated_small, control_small, gamma_range=(1.0, 3.0), n_gamma=20
        )

        # Large sample
        np.random.seed(101)  # Reset seed for comparable draws
        n_large = 100
        treated_large = np.random.randn(n_large) + effect_size
        control_large = np.random.randn(n_large)
        result_large = rosenbaum_bounds(
            treated_large, control_large, gamma_range=(1.0, 3.0), n_gamma=20
        )

        # Larger sample should be more robust (or both robust to max gamma)
        gamma_small = result_small["gamma_critical"] or 3.1
        gamma_large = result_large["gamma_critical"] or 3.1

        # Allow for some randomness, but generally larger sample more robust
        # This is a probabilistic test
        assert gamma_large >= gamma_small * 0.8, (
            f"Larger sample ({gamma_large}) should be roughly as robust as smaller ({gamma_small})"
        )


# ============================================================================
# Layer 3: Edge Cases
# ============================================================================


class TestRosenbaumEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_positive_differences(self):
        """All pairs have positive differences (treated > control)."""
        np.random.seed(42)
        n = 20
        # Ensure all treated > control
        control = np.random.randn(n)
        treated = control + np.abs(np.random.randn(n)) + 0.5

        result = rosenbaum_bounds(treated, control)

        # Should be very robust
        assert result["p_lower"][0] < 0.01, "All positive should give small p-value"

    def test_mixed_differences(self):
        """Mix of positive and negative differences."""
        np.random.seed(42)
        n = 20
        treated = np.random.randn(n) + 0.5
        control = np.random.randn(n)

        result = rosenbaum_bounds(treated, control)

        # Should produce valid results
        assert len(result["gamma_values"]) == 20  # Default n_gamma
        assert all(0 <= p <= 1 for p in result["p_upper"])

    def test_some_zero_differences(self):
        """Some pairs have zero difference (ties)."""
        treated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        control = np.array([0.5, 2.0, 2.5, 4.0, 4.5])  # Two ties at indices 1, 3

        result = rosenbaum_bounds(treated, control)

        # Should handle ties gracefully
        assert result["n_pairs"] == 5
        # Non-zero pairs: 3
        assert np.isfinite(result["observed_statistic"])

    def test_small_sample(self):
        """Small sample (n=5 pairs)."""
        treated = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        control = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = rosenbaum_bounds(treated, control)

        # Should work but be less robust
        assert result["n_pairs"] == 5
        assert "interpretation" in result

    def test_minimum_pairs(self):
        """Minimum number of pairs (n=2)."""
        treated = np.array([2.0, 3.0])
        control = np.array([1.0, 2.0])

        result = rosenbaum_bounds(treated, control)

        assert result["n_pairs"] == 2

    def test_gamma_range_custom(self):
        """Custom gamma range."""
        np.random.seed(42)
        treated = np.random.randn(30) + 1.0
        control = np.random.randn(30)

        result = rosenbaum_bounds(treated, control, gamma_range=(1.5, 4.0), n_gamma=15)

        assert result["gamma_values"][0] == 1.5
        assert result["gamma_values"][-1] == 4.0
        assert len(result["gamma_values"]) == 15


# ============================================================================
# Layer 4: Input Validation
# ============================================================================


class TestRosenbaumInputValidation:
    """Tests for input validation and error handling."""

    def test_length_mismatch(self):
        """Treated and control must have same length."""
        treated = np.array([1.0, 2.0, 3.0])
        control = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="same length"):
            rosenbaum_bounds(treated, control)

    def test_too_few_pairs(self):
        """Need at least 2 pairs."""
        treated = np.array([1.0])
        control = np.array([0.5])

        with pytest.raises(ValueError, match="at least 2 pairs"):
            rosenbaum_bounds(treated, control)

    def test_gamma_below_1(self):
        """Gamma must be >= 1."""
        treated = np.array([1.0, 2.0, 3.0])
        control = np.array([0.5, 1.5, 2.5])

        with pytest.raises(ValueError, match="Gamma must be >= 1.0"):
            rosenbaum_bounds(treated, control, gamma_range=(0.5, 2.0))

    def test_gamma_range_inverted(self):
        """Gamma range upper must be >= lower."""
        treated = np.array([1.0, 2.0, 3.0])
        control = np.array([0.5, 1.5, 2.5])

        with pytest.raises(ValueError, match="upper.*must be >= lower"):
            rosenbaum_bounds(treated, control, gamma_range=(2.0, 1.5))


# ============================================================================
# Integration Tests
# ============================================================================


class TestRosenbaumIntegration:
    """Integration tests with typical PSM workflow."""

    def test_interpretation_present(self):
        """Result should include interpretation."""
        np.random.seed(42)
        treated = np.random.randn(30) + 1.0
        control = np.random.randn(30)

        result = rosenbaum_bounds(treated, control)

        assert "interpretation" in result
        assert len(result["interpretation"]) > 0
        assert "Rosenbaum" in result["interpretation"]

    def test_typical_psm_workflow(self):
        """Typical PSM outcome analysis."""
        np.random.seed(42)
        n = 50

        # Simulated matched pairs with moderate effect
        treated = np.random.randn(n) + 0.8  # Effect size ~0.8 SD
        control = np.random.randn(n)

        result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 3.0), n_gamma=20)

        # Should have reasonable output
        assert result["n_pairs"] == 50
        assert len(result["gamma_values"]) == 20
        assert len(result["p_upper"]) == 20
        assert len(result["p_lower"]) == 20

        # Should be at least somewhat robust with effect ~0.8
        if result["gamma_critical"] is not None:
            assert result["gamma_critical"] > 1.2

    def test_robust_result_no_critical_gamma(self):
        """Very robust result may have no critical gamma in range."""
        np.random.seed(42)
        n = 100

        # Very strong effect
        treated = np.random.randn(n) + 4.0
        control = np.random.randn(n)

        result = rosenbaum_bounds(treated, control, gamma_range=(1.0, 3.0), n_gamma=10)

        # Either gamma_critical is None or > 2.5
        if result["gamma_critical"] is not None:
            assert result["gamma_critical"] > 2.5
        # If None, interpretation should mention robust
        else:
            assert "robust" in result["interpretation"].lower()


# ============================================================================
# Statistical Property Tests
# ============================================================================


class TestRosenbaumStatisticalProperties:
    """Tests for statistical properties of bounds."""

    def test_p_values_bounded_0_1(self):
        """All p-values should be in [0, 1]."""
        np.random.seed(42)
        treated = np.random.randn(40) + 1.0
        control = np.random.randn(40)

        result = rosenbaum_bounds(treated, control)

        assert all(0 <= p <= 1 for p in result["p_upper"]), "p_upper out of bounds"
        assert all(0 <= p <= 1 for p in result["p_lower"]), "p_lower out of bounds"

    def test_upper_geq_lower_p(self):
        """Upper bound p-value >= lower bound p-value at each Gamma."""
        np.random.seed(42)
        treated = np.random.randn(50) + 0.8
        control = np.random.randn(50)

        result = rosenbaum_bounds(treated, control, n_gamma=20)

        for i in range(len(result["gamma_values"])):
            assert result["p_upper"][i] >= result["p_lower"][i] - 1e-10, (
                f"p_upper ({result['p_upper'][i]}) should >= "
                f"p_lower ({result['p_lower'][i]}) at gamma={result['gamma_values'][i]}"
            )

    def test_observed_statistic_positive(self):
        """Observed T+ statistic should be non-negative."""
        np.random.seed(42)
        treated = np.random.randn(30) + 0.5
        control = np.random.randn(30)

        result = rosenbaum_bounds(treated, control)

        assert result["observed_statistic"] >= 0
