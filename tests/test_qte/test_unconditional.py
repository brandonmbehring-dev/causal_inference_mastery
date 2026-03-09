"""
Tests for unconditional QTE estimation.

Tests cover:
- Known-answer validation with hand-calculated values
- Property tests (SE positive, CI contains estimate, etc.)
- Edge cases and input validation
"""

import numpy as np
import pytest

from src.causal_inference.qte import unconditional_qte, unconditional_qte_band


class TestUnconditionalQTEKnownAnswers:
    """Test unconditional QTE with known expected values."""

    def test_median_effect_basic_case(self, known_quantile_data):
        """
        Test median treatment effect with hand-calculated expected values.

        Treated: [3, 4, 5, 6, 7] -> median = 5
        Control: [1, 2, 3, 4, 5] -> median = 3
        Expected QTE(0.5) = 5 - 3 = 2.0
        """
        outcome, treatment = known_quantile_data

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=500, random_state=42
        )

        # Point estimate should be exactly 2.0 (or very close due to interpolation)
        assert np.isclose(result["tau_q"], 2.0, atol=0.1), f"Expected ~2.0, got {result['tau_q']}"

        # Check result structure
        assert result["quantile"] == 0.5
        assert result["method"] == "unconditional"
        assert result["inference"] == "bootstrap"
        assert result["n_treated"] == 5
        assert result["n_control"] == 5
        assert result["n_total"] == 10

    def test_homogeneous_effect_across_quantiles(self, simple_rct_data):
        """With homogeneous effect, QTE should be similar across quantiles."""
        outcome, treatment = simple_rct_data

        qte_25 = unconditional_qte(outcome, treatment, quantile=0.25, random_state=42)
        qte_50 = unconditional_qte(outcome, treatment, quantile=0.50, random_state=42)
        qte_75 = unconditional_qte(outcome, treatment, quantile=0.75, random_state=42)

        # All should be close to true effect of 2.0
        assert np.isclose(qte_25["tau_q"], 2.0, atol=0.5)
        assert np.isclose(qte_50["tau_q"], 2.0, atol=0.5)
        assert np.isclose(qte_75["tau_q"], 2.0, atol=0.5)

        # And similar to each other (within 0.5)
        assert abs(qte_25["tau_q"] - qte_75["tau_q"]) < 0.5

    def test_zero_effect(self, zero_effect_data):
        """Test estimation when true effect is zero."""
        outcome, treatment = zero_effect_data

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=500, random_state=42
        )

        # Should be close to 0 (within 2 SEs)
        assert abs(result["tau_q"]) < 2 * result["se"] + 0.2

    def test_negative_effect(self, negative_effect_data):
        """Test estimation with negative true effect."""
        outcome, treatment = negative_effect_data

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=500, random_state=42
        )

        # Should be negative and close to -1.5
        assert result["tau_q"] < 0
        assert np.isclose(result["tau_q"], -1.5, atol=0.5)


class TestUnconditionalQTEProperties:
    """Property-based tests for unconditional QTE."""

    def test_se_positive(self, simple_rct_data):
        """Standard error should always be positive."""
        outcome, treatment = simple_rct_data

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=500, random_state=42
        )

        assert result["se"] > 0, f"SE should be positive, got {result['se']}"

    def test_ci_contains_estimate(self, simple_rct_data):
        """Confidence interval should contain point estimate."""
        outcome, treatment = simple_rct_data

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=500, random_state=42
        )

        assert result["ci_lower"] < result["tau_q"] < result["ci_upper"]

    def test_ci_lower_less_than_upper(self, simple_rct_data):
        """CI lower bound should be less than upper bound."""
        outcome, treatment = simple_rct_data

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=500, random_state=42
        )

        assert result["ci_lower"] < result["ci_upper"]

    def test_outcome_support_correct(self, simple_rct_data):
        """Outcome support should match data range."""
        outcome, treatment = simple_rct_data

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=100, random_state=42
        )

        assert result["outcome_support"][0] == outcome.min()
        assert result["outcome_support"][1] == outcome.max()

    def test_sample_sizes_correct(self, simple_rct_data):
        """Sample sizes should be correct."""
        outcome, treatment = simple_rct_data

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=100, random_state=42
        )

        expected_treated = int(np.sum(treatment == 1))
        expected_control = int(np.sum(treatment == 0))

        assert result["n_treated"] == expected_treated
        assert result["n_control"] == expected_control
        assert result["n_total"] == len(outcome)

    def test_reproducibility_with_seed(self, simple_rct_data):
        """Same seed should give identical results."""
        outcome, treatment = simple_rct_data

        result1 = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=500, random_state=123
        )
        result2 = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=500, random_state=123
        )

        assert result1["tau_q"] == result2["tau_q"]
        assert result1["se"] == result2["se"]
        assert result1["ci_lower"] == result2["ci_lower"]
        assert result1["ci_upper"] == result2["ci_upper"]

    def test_larger_sample_smaller_se(self, simple_rct_data, large_rct_data):
        """Larger sample should have smaller standard error."""
        outcome_small, treatment_small = simple_rct_data
        outcome_large, treatment_large = large_rct_data

        result_small = unconditional_qte(
            outcome_small, treatment_small, quantile=0.5, n_bootstrap=500, random_state=42
        )
        result_large = unconditional_qte(
            outcome_large, treatment_large, quantile=0.5, n_bootstrap=500, random_state=42
        )

        assert result_large["se"] < result_small["se"]


class TestUnconditionalQTEBand:
    """Tests for unconditional_qte_band function."""

    def test_band_returns_all_quantiles(self, simple_rct_data):
        """Band should return estimates at all requested quantiles."""
        outcome, treatment = simple_rct_data

        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        result = unconditional_qte_band(
            outcome, treatment, quantiles=quantiles, n_bootstrap=200, random_state=42
        )

        assert len(result["quantiles"]) == len(quantiles)
        assert len(result["qte_estimates"]) == len(quantiles)
        assert len(result["se_estimates"]) == len(quantiles)
        assert len(result["ci_lower"]) == len(quantiles)
        assert len(result["ci_upper"]) == len(quantiles)

    def test_band_default_quantiles(self, simple_rct_data):
        """Default quantiles should be [0.1, 0.25, 0.5, 0.75, 0.9]."""
        outcome, treatment = simple_rct_data

        result = unconditional_qte_band(outcome, treatment, n_bootstrap=200, random_state=42)

        expected = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        np.testing.assert_array_almost_equal(result["quantiles"], expected)

    def test_band_joint_ci(self, simple_rct_data):
        """Joint CI should be wider than pointwise CI."""
        outcome, treatment = simple_rct_data

        result = unconditional_qte_band(
            outcome, treatment, n_bootstrap=500, joint=True, random_state=42
        )

        # Joint CI should be computed
        assert result["joint_ci_lower"] is not None
        assert result["joint_ci_upper"] is not None

        # Joint should be wider (or equal) at each quantile
        joint_widths = result["joint_ci_upper"] - result["joint_ci_lower"]
        pointwise_widths = result["ci_upper"] - result["ci_lower"]

        assert np.all(joint_widths >= pointwise_widths - 1e-10)

    def test_band_no_joint_by_default(self, simple_rct_data):
        """Joint CI should not be computed by default."""
        outcome, treatment = simple_rct_data

        result = unconditional_qte_band(outcome, treatment, n_bootstrap=200, random_state=42)

        assert result["joint_ci_lower"] is None
        assert result["joint_ci_upper"] is None


class TestUnconditionalQTEEdgeCases:
    """Edge case tests for robustness."""

    def test_extreme_quantile_low(self, large_rct_data):
        """Low quantile (0.05) should still work."""
        outcome, treatment = large_rct_data

        result = unconditional_qte(
            outcome, treatment, quantile=0.05, n_bootstrap=500, random_state=42
        )

        assert np.isfinite(result["tau_q"])
        assert np.isfinite(result["se"])
        assert result["se"] > 0

    def test_extreme_quantile_high(self, large_rct_data):
        """High quantile (0.95) should still work."""
        outcome, treatment = large_rct_data

        result = unconditional_qte(
            outcome, treatment, quantile=0.95, n_bootstrap=500, random_state=42
        )

        assert np.isfinite(result["tau_q"])
        assert np.isfinite(result["se"])
        assert result["se"] > 0

    def test_small_sample(self, minimal_sample):
        """Should work with minimal sample (2 per group)."""
        outcome, treatment = minimal_sample

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=100, random_state=42
        )

        assert np.isfinite(result["tau_q"])
        # SE should be large for small samples
        assert result["se"] > 0

    def test_imbalanced_treatment(self, imbalanced_treatment_data):
        """Should work with imbalanced treatment groups."""
        outcome, treatment = imbalanced_treatment_data

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=500, random_state=42
        )

        assert np.isfinite(result["tau_q"])
        assert result["se"] > 0

    def test_high_noise(self, high_noise_data):
        """Should work with high noise data."""
        outcome, treatment = high_noise_data

        result = unconditional_qte(
            outcome, treatment, quantile=0.5, n_bootstrap=500, random_state=42
        )

        assert np.isfinite(result["tau_q"])
        # SE should be larger with high noise
        assert result["se"] > 0

    def test_list_input(self, known_quantile_data):
        """Should accept list inputs."""
        outcome, treatment = known_quantile_data

        result = unconditional_qte(
            list(outcome), list(treatment), quantile=0.5, n_bootstrap=100, random_state=42
        )

        assert np.isfinite(result["tau_q"])


class TestUnconditionalQTEInputValidation:
    """Input validation tests - should raise appropriate errors."""

    def test_empty_arrays(self, empty_arrays):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError, match="CRITICAL ERROR.*Empty"):
            unconditional_qte(empty_arrays["outcome"], empty_arrays["treatment"], quantile=0.5)

    def test_nan_in_outcome(self, nan_in_outcome):
        """NaN values should raise ValueError."""
        outcome, treatment = nan_in_outcome

        with pytest.raises(ValueError, match="CRITICAL ERROR.*NaN"):
            unconditional_qte(outcome, treatment, quantile=0.5)

    def test_inf_in_outcome(self, inf_in_outcome):
        """Infinite values should raise ValueError."""
        outcome, treatment = inf_in_outcome

        with pytest.raises(ValueError, match="CRITICAL ERROR.*Infinite"):
            unconditional_qte(outcome, treatment, quantile=0.5)

    def test_non_binary_treatment(self, non_binary_treatment):
        """Non-binary treatment should raise ValueError."""
        outcome, treatment = non_binary_treatment

        with pytest.raises(ValueError, match="CRITICAL ERROR.*binary"):
            unconditional_qte(outcome, treatment, quantile=0.5)

    def test_no_variation_treatment(self, no_variation_treatment):
        """No treatment variation should raise ValueError."""
        outcome, treatment = no_variation_treatment

        with pytest.raises(ValueError, match="CRITICAL ERROR.*variation"):
            unconditional_qte(outcome, treatment, quantile=0.5)

    def test_length_mismatch(self, length_mismatch):
        """Mismatched array lengths should raise ValueError."""
        outcome, treatment = length_mismatch

        with pytest.raises(ValueError, match="CRITICAL ERROR.*lengths"):
            unconditional_qte(outcome, treatment, quantile=0.5)

    def test_invalid_quantile_too_low(self, simple_rct_data):
        """Quantile <= 0 should raise ValueError."""
        outcome, treatment = simple_rct_data

        with pytest.raises(ValueError, match="CRITICAL ERROR.*quantile"):
            unconditional_qte(outcome, treatment, quantile=0.0)

        with pytest.raises(ValueError, match="CRITICAL ERROR.*quantile"):
            unconditional_qte(outcome, treatment, quantile=-0.1)

    def test_invalid_quantile_too_high(self, simple_rct_data):
        """Quantile >= 1 should raise ValueError."""
        outcome, treatment = simple_rct_data

        with pytest.raises(ValueError, match="CRITICAL ERROR.*quantile"):
            unconditional_qte(outcome, treatment, quantile=1.0)

        with pytest.raises(ValueError, match="CRITICAL ERROR.*quantile"):
            unconditional_qte(outcome, treatment, quantile=1.5)

    def test_invalid_alpha(self, simple_rct_data):
        """Invalid alpha should raise ValueError."""
        outcome, treatment = simple_rct_data

        with pytest.raises(ValueError, match="CRITICAL ERROR.*alpha"):
            unconditional_qte(outcome, treatment, quantile=0.5, alpha=0.0)

        with pytest.raises(ValueError, match="CRITICAL ERROR.*alpha"):
            unconditional_qte(outcome, treatment, quantile=0.5, alpha=1.0)
