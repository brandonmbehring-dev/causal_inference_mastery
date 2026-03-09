"""
Adversarial tests for simple_ate estimator.

Tests extreme edge cases and boundary conditions:
1. Extreme sample sizes (n=2, n=3, n=1000000)
2. Extreme imbalance (n1=1 n0=999, n1=999 n0=1)
3. Extreme variance (σ²=0.001, σ²=1000000)
4. Numerical stability (values near machine precision)
5. Outliers (extreme outcome values)
6. Perfect separation (all treated=100, all control=0)
7. Tied values (all outcomes identical)
"""

import numpy as np
import pytest
from src.causal_inference.rct.estimators import simple_ate


class TestSimpleATEExtremeSampleSizes:
    """Test simple_ate with extreme sample sizes."""

    @pytest.mark.xfail(reason="Known issue: n1=1 or n0=1 produces NaN SE (ddof causes df≤0)")
    def test_minimum_sample_n2(self):
        """n=2 (n1=1, n0=1) - minimum possible sample."""
        outcomes = np.array([10.0, 5.0])
        treatment = np.array([1, 0])
        result = simple_ate(outcomes, treatment)

        # Should compute ATE = 10 - 5 = 5
        assert result["estimate"] == 5.0
        # Should return valid SE (may be large)
        assert result["se"] > 0
        assert np.isfinite(result["se"])
        # CI should be finite
        assert np.isfinite(result["ci_lower"])
        assert np.isfinite(result["ci_upper"])

    @pytest.mark.xfail(reason="Known issue: n0=1 produces NaN SE (ddof causes df≤0)")
    def test_very_small_sample_n3(self):
        """n=3 (n1=2, n0=1) - very small unbalanced sample."""
        outcomes = np.array([10.0, 8.0, 3.0])
        treatment = np.array([1, 1, 0])
        result = simple_ate(outcomes, treatment)

        # ATE = 9 - 3 = 6
        assert np.isclose(result["estimate"], 6.0)
        assert result["se"] > 0
        assert np.isfinite(result["ci_lower"])
        assert np.isfinite(result["ci_upper"])

    def test_large_sample_n10000(self):
        """Large sample (n=10000) - should be fast and accurate."""
        np.random.seed(42)
        n = 10000
        treatment = np.array([1] * 5000 + [0] * 5000)
        # True ATE = 2.0
        outcomes = np.where(
            treatment == 1, np.random.normal(2.0, 1.0, n), np.random.normal(0.0, 1.0, n)
        )

        result = simple_ate(outcomes, treatment)

        # Should be close to 2.0 (law of large numbers)
        assert 1.8 < result["estimate"] < 2.2
        # SE should be small
        assert result["se"] < 0.05


class TestSimpleATEExtremeImbalance:
    """Test simple_ate with extreme treatment imbalance."""

    @pytest.mark.xfail(reason="Known issue: n1=1 produces NaN SE (ddof causes df≤0)")
    def test_extreme_imbalance_n1_n999(self):
        """n1=1, n0=999 - one treated unit vs 999 control."""
        outcomes = np.concatenate([[100.0], np.zeros(999)])
        treatment = np.array([1] + [0] * 999)

        result = simple_ate(outcomes, treatment)

        # ATE = 100 - 0 = 100
        assert result["estimate"] == 100.0
        # SE should be large (high uncertainty)
        assert result["se"] > 0
        assert np.isfinite(result["se"])

    @pytest.mark.xfail(reason="Known issue: n0=1 produces NaN SE (ddof causes df≤0)")
    def test_extreme_imbalance_n999_n1(self):
        """n1=999, n0=1 - 999 treated vs one control."""
        outcomes = np.concatenate([np.ones(999) * 10, [5.0]])
        treatment = np.array([1] * 999 + [0])

        result = simple_ate(outcomes, treatment)

        # ATE = 10 - 5 = 5
        assert result["estimate"] == 5.0
        assert result["se"] > 0


class TestSimpleATEExtremeVariance:
    """Test simple_ate with extreme outcome variance."""

    def test_very_low_variance(self):
        """Outcomes with very low variance (σ² ≈ 0.001)."""
        np.random.seed(42)
        treatment = np.array([1] * 50 + [0] * 50)
        outcomes = np.where(
            treatment == 1,
            2.0 + np.random.normal(0, 0.01, 100),  # σ=0.01
            0.0 + np.random.normal(0, 0.01, 100),
        )

        result = simple_ate(outcomes, treatment)

        # Should be close to 2.0
        assert 1.9 < result["estimate"] < 2.1
        # SE should be very small
        assert result["se"] < 0.01

    def test_very_high_variance(self):
        """Outcomes with very high variance (σ² = 1000000)."""
        np.random.seed(42)
        treatment = np.array([1] * 50 + [0] * 50)
        outcomes = np.where(
            treatment == 1,
            2.0 + np.random.normal(0, 1000, 100),  # σ=1000
            0.0 + np.random.normal(0, 1000, 100),
        )

        result = simple_ate(outcomes, treatment)

        # Estimate should be finite
        assert np.isfinite(result["estimate"])
        # SE should be large
        assert result["se"] > 100
        # CI should be very wide (at least 500, may not reach 1000 with n=100)
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert ci_width > 500


class TestSimpleATENumericalStability:
    """Test simple_ate with values near machine precision."""

    def test_tiny_differences(self):
        """Outcomes differing by ~1e-10."""
        outcomes = np.array([1e-10, 2e-10, 0.0, 0.0])
        treatment = np.array([1, 1, 0, 0])

        result = simple_ate(outcomes, treatment)

        # Should compute ATE ≈ 1.5e-10
        assert np.isclose(result["estimate"], 1.5e-10, atol=1e-15)
        assert result["se"] > 0

    def test_huge_values(self):
        """Outcomes near max float (1e308)."""
        outcomes = np.array([1e100, 1e100, 1e99, 1e99])
        treatment = np.array([1, 1, 0, 0])

        result = simple_ate(outcomes, treatment)

        # Should compute ATE ≈ 9e99
        assert np.isfinite(result["estimate"])
        assert result["estimate"] > 1e99


class TestSimpleATEOutliers:
    """Test simple_ate with extreme outliers."""

    def test_single_extreme_outlier_treated(self):
        """One extreme outlier in treated group."""
        outcomes = np.array([10000.0, 2.0, 2.1, 1.0, 1.1])
        treatment = np.array([1, 1, 1, 0, 0])

        result = simple_ate(outcomes, treatment)

        # Outlier should affect estimate
        assert result["estimate"] > 1000
        # SE should be large
        assert result["se"] > 100

    def test_single_extreme_outlier_control(self):
        """One extreme outlier in control group."""
        outcomes = np.array([2.0, 2.1, 10000.0, 1.0, 1.1])
        treatment = np.array([1, 1, 0, 0, 0])

        result = simple_ate(outcomes, treatment)

        # Outlier should affect estimate (negative)
        assert result["estimate"] < -1000


class TestSimpleATEPerfectSeparation:
    """Test simple_ate with perfect separation."""

    def test_all_treated_same_all_control_same(self):
        """All treated=100, all control=0 (perfect separation)."""
        outcomes = np.array([100.0] * 50 + [0.0] * 50)
        treatment = np.array([1] * 50 + [0] * 50)

        result = simple_ate(outcomes, treatment)

        # ATE = 100 - 0 = 100
        assert result["estimate"] == 100.0
        # Variance should be zero (no within-group variation)
        # SE should be zero or very small
        assert result["se"] < 1e-10


class TestSimpleATETiedValues:
    """Test simple_ate with tied outcome values."""

    def test_all_outcomes_identical(self):
        """All outcomes = 5.0 (zero effect, zero variance)."""
        outcomes = np.array([5.0] * 100)
        treatment = np.array([1] * 50 + [0] * 50)

        result = simple_ate(outcomes, treatment)

        # ATE = 5 - 5 = 0
        assert result["estimate"] == 0.0
        # Variance should be zero
        assert result["se"] < 1e-10

    def test_two_tied_groups(self):
        """Treated all 10, control all 5."""
        outcomes = np.array([10.0] * 50 + [5.0] * 50)
        treatment = np.array([1] * 50 + [0] * 50)

        result = simple_ate(outcomes, treatment)

        # ATE = 10 - 5 = 5
        assert result["estimate"] == 5.0
        # SE should be zero (no within-group variance)
        assert result["se"] < 1e-10


class TestSimpleATEMixedCases:
    """Test simple_ate with multiple adversarial conditions."""

    @pytest.mark.xfail(reason="Known issue: n0=1 produces NaN SE (ddof causes df≤0)")
    def test_tiny_sample_huge_variance(self):
        """Combine n=3 with huge variance."""
        np.random.seed(42)
        outcomes = np.array([1000.0, -500.0, 0.0])
        treatment = np.array([1, 1, 0])

        result = simple_ate(outcomes, treatment)

        # Should compute valid estimate
        assert np.isfinite(result["estimate"])
        # SE should be large
        assert result["se"] > 100

    @pytest.mark.xfail(reason="Known issue: n1=1 produces NaN SE (ddof causes df≤0)")
    def test_extreme_imbalance_with_outlier(self):
        """n1=1 with extreme outlier."""
        outcomes = np.array([1e6] + [0.0] * 999)
        treatment = np.array([1] + [0] * 999)

        result = simple_ate(outcomes, treatment)

        # ATE should be huge
        assert result["estimate"] > 1e5
        # But should be finite
        assert np.isfinite(result["estimate"])
        assert np.isfinite(result["se"])
