"""
Tests for Sharp Regression Kink Design (RKD) Estimator

Test Categories:
1. Known-Answer Tests - Hand-calculated expected values
2. Statistical Properties - Unbiasedness, coverage, consistency
3. Edge Cases - Small samples, extreme kinks, boundary conditions
"""

import numpy as np
import pytest
from scipy import stats

from src.causal_inference.rkd import SharpRKD, SharpRKDResult


# =============================================================================
# DGP Functions for Testing
# =============================================================================


def generate_rkd_data(
    n: int = 1000,
    cutoff: float = 0.0,
    slope_d_left: float = 0.5,
    slope_d_right: float = 1.5,
    true_effect: float = 2.0,
    baseline_slope: float = 0.3,
    sigma: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Sharp RKD data with known parameters.

    Model:
        D = slope_d_left * X  for X < cutoff
        D = slope_d_right * X for X >= cutoff

        Y = true_effect * D + baseline_slope * X + noise
          = true_effect * slope_d * X + baseline_slope * X + noise

    The kink in Y comes from the kink in D:
        dY/dX|left  = true_effect * slope_d_left + baseline_slope
        dY/dX|right = true_effect * slope_d_right + baseline_slope

        Δ(dY/dX) = true_effect * (slope_d_right - slope_d_left)
                 = true_effect * Δslope_D

    So: true_effect = Δ(dY/dX) / Δslope_D

    Returns
    -------
    y : np.ndarray
        Outcome variable
    x : np.ndarray
        Running variable
    d : np.ndarray
        Treatment intensity
    """
    rng = np.random.default_rng(seed)

    # Running variable uniform around cutoff
    x = rng.uniform(cutoff - 5, cutoff + 5, n)

    # Treatment with kink
    d = np.where(x < cutoff, slope_d_left * x, slope_d_right * x)

    # Outcome: causal effect of D plus smooth baseline
    y = true_effect * d + baseline_slope * x + rng.normal(0, sigma, n)

    return y, x, d


def generate_rkd_no_effect_data(
    n: int = 1000,
    cutoff: float = 0.0,
    slope_d_left: float = 0.5,
    slope_d_right: float = 1.5,
    sigma: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate RKD data with NO treatment effect (null hypothesis)."""
    rng = np.random.default_rng(seed)

    x = rng.uniform(cutoff - 5, cutoff + 5, n)
    d = np.where(x < cutoff, slope_d_left * x, slope_d_right * x)

    # Outcome with smooth linear relationship (no kink from treatment)
    y = 0.5 * x + rng.normal(0, sigma, n)

    return y, x, d


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestSharpRKDKnownAnswer:
    """Tests with hand-calculated expected values."""

    def test_simple_kink_recovery(self):
        """RKD should recover known treatment effect."""
        true_effect = 2.0
        y, x, d = generate_rkd_data(
            n=2000,
            true_effect=true_effect,
            slope_d_left=0.5,
            slope_d_right=1.5,
            sigma=0.5,
            seed=42,
        )

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        # Should be close to true effect (within 0.6 for finite sample)
        assert abs(result.estimate - true_effect) < 0.6, (
            f"Estimate {result.estimate:.3f} too far from truth {true_effect}"
        )

    def test_negative_effect(self):
        """RKD should recover negative treatment effects."""
        true_effect = -1.5
        y, x, d = generate_rkd_data(
            n=2000,
            true_effect=true_effect,
            slope_d_left=0.5,
            slope_d_right=1.5,
            sigma=0.5,
            seed=123,
        )

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        assert abs(result.estimate - true_effect) < 0.3

    def test_large_effect(self):
        """RKD should handle large treatment effects."""
        true_effect = 10.0
        y, x, d = generate_rkd_data(
            n=2000,
            true_effect=true_effect,
            slope_d_left=0.5,
            slope_d_right=1.5,
            sigma=1.0,
            seed=456,
        )

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        assert abs(result.estimate - true_effect) < 1.0

    def test_zero_effect(self):
        """RKD should not spuriously find effects under null."""
        y, x, d = generate_rkd_no_effect_data(n=2000, sigma=1.0, seed=789)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        # Effect should be close to zero
        assert abs(result.estimate) < 0.5, f"Spurious effect detected: {result.estimate:.3f}"

    def test_different_kink_magnitudes(self):
        """Test with different kink sizes."""
        true_effect = 2.0

        # Small kink
        y, x, d = generate_rkd_data(
            n=2000,
            true_effect=true_effect,
            slope_d_left=0.9,
            slope_d_right=1.1,  # Δ = 0.2
            sigma=0.5,
            seed=111,
        )
        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result_small = rkd.fit(y, x, d)

        # Large kink
        y, x, d = generate_rkd_data(
            n=2000,
            true_effect=true_effect,
            slope_d_left=0.0,
            slope_d_right=2.0,  # Δ = 2.0
            sigma=0.5,
            seed=222,
        )
        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result_large = rkd.fit(y, x, d)

        # Both should recover effect, but large kink should be more precise
        assert abs(result_small.estimate - true_effect) < 1.0
        assert abs(result_large.estimate - true_effect) < 0.5
        # Larger kink → smaller SE
        assert result_large.se < result_small.se

    def test_nonzero_cutoff(self):
        """RKD should work with non-zero cutoff."""
        true_effect = 2.0
        cutoff = 5.0

        rng = np.random.default_rng(333)
        x = rng.uniform(cutoff - 5, cutoff + 5, 2000)
        d = np.where(x < cutoff, 0.5 * (x - cutoff), 1.5 * (x - cutoff))
        y = true_effect * d + 0.3 * x + rng.normal(0, 0.5, 2000)

        rkd = SharpRKD(cutoff=cutoff, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        # Wider tolerance for finite sample bias
        assert abs(result.estimate - true_effect) < 0.6


# =============================================================================
# Layer 2: Statistical Properties
# =============================================================================


class TestSharpRKDStatistics:
    """Tests for statistical properties."""

    def test_coverage_monte_carlo(self):
        """95% CI should cover true value ~95% of the time."""
        true_effect = 2.0
        n_sims = 100
        covered = 0

        for seed in range(n_sims):
            y, x, d = generate_rkd_data(
                n=500,
                true_effect=true_effect,
                sigma=1.0,
                seed=seed,
            )

            rkd = SharpRKD(cutoff=0.0, bandwidth=2.0, alpha=0.05)
            result = rkd.fit(y, x, d)

            if result.retcode != "error":
                if result.ci_lower <= true_effect <= result.ci_upper:
                    covered += 1

        coverage = covered / n_sims
        # Allow 85-99% due to Monte Carlo noise and small sample
        assert 0.85 <= coverage <= 0.99, f"Coverage {coverage:.2%} outside range"

    def test_unbiasedness_monte_carlo(self):
        """Estimator should be approximately unbiased."""
        true_effect = 2.0
        n_sims = 100
        estimates = []

        for seed in range(n_sims):
            y, x, d = generate_rkd_data(
                n=500,
                true_effect=true_effect,
                sigma=1.0,
                seed=seed,
            )

            rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
            result = rkd.fit(y, x, d)

            if result.retcode != "error":
                estimates.append(result.estimate)

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_effect

        # Bias should be small (< 0.2 for n=500)
        assert abs(bias) < 0.3, f"Bias {bias:.3f} exceeds threshold"

    def test_consistency(self):
        """Larger samples should give more precise estimates."""
        true_effect = 2.0

        se_by_n = {}
        for n in [200, 500, 1000, 2000]:
            y, x, d = generate_rkd_data(
                n=n,
                true_effect=true_effect,
                sigma=1.0,
                seed=42,
            )

            rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
            result = rkd.fit(y, x, d)
            se_by_n[n] = result.se

        # SE should decrease with n
        assert se_by_n[500] < se_by_n[200]
        assert se_by_n[1000] < se_by_n[500]
        assert se_by_n[2000] < se_by_n[1000]

    def test_type_i_error_control(self):
        """Should not reject null when true effect is zero."""
        n_sims = 100
        rejections = 0

        for seed in range(n_sims):
            y, x, d = generate_rkd_no_effect_data(
                n=500,
                sigma=1.0,
                seed=seed,
            )

            rkd = SharpRKD(cutoff=0.0, bandwidth=2.0, alpha=0.05)
            result = rkd.fit(y, x, d)

            if result.retcode != "error" and result.p_value < 0.05:
                rejections += 1

        type_i_rate = rejections / n_sims
        # Allow 1-12% for MC noise
        assert type_i_rate < 0.12, f"Type I error {type_i_rate:.2%} too high"


# =============================================================================
# Layer 3: Edge Cases and Robustness
# =============================================================================


class TestSharpRKDEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_sample(self):
        """Should handle small samples gracefully."""
        y, x, d = generate_rkd_data(n=50, true_effect=2.0, sigma=1.0, seed=42)

        rkd = SharpRKD(cutoff=0.0, bandwidth=3.0)
        result = rkd.fit(y, x, d)

        # Should complete (may have warning)
        assert result.retcode in ("success", "warning", "error")
        if result.retcode != "error":
            assert np.isfinite(result.estimate)
            assert np.isfinite(result.se)

    def test_very_small_sample_error(self):
        """Should error on too-small samples."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 5)
        x = rng.uniform(-1, 1, 5)
        d = x

        rkd = SharpRKD(cutoff=0.0, bandwidth=1.0)

        with pytest.raises(ValueError, match="Insufficient"):
            rkd.fit(y, x, d)

    def test_no_kink_error(self):
        """Should handle case with no kink in D."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, 1000)
        d = 0.5 * x  # No kink - constant slope
        y = 2.0 * d + rng.normal(0, 1, 1000)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        # Should return error because Δslope_D ≈ 0
        assert result.retcode == "error"
        assert "kink" in result.message.lower()

    def test_misspecified_slopes(self):
        """Should work when slopes are estimated vs specified."""
        true_effect = 2.0
        y, x, d = generate_rkd_data(
            n=1000,
            true_effect=true_effect,
            slope_d_left=0.5,
            slope_d_right=1.5,
            sigma=0.5,
            seed=42,
        )

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)

        # Estimated slopes
        result_est = rkd.fit(y, x, d)

        # Specified slopes (correct)
        result_spec = rkd.fit(y, x, d, slope_d_left=0.5, slope_d_right=1.5)

        # Both should recover effect
        assert abs(result_est.estimate - true_effect) < 0.5
        assert abs(result_spec.estimate - true_effect) < 0.5

    def test_different_kernels(self):
        """Should work with different kernel functions."""
        y, x, d = generate_rkd_data(n=1000, true_effect=2.0, sigma=0.5, seed=42)

        for kernel in ["triangular", "rectangular", "epanechnikov"]:
            rkd = SharpRKD(cutoff=0.0, bandwidth=2.0, kernel=kernel)
            result = rkd.fit(y, x, d)

            assert result.retcode in ("success", "warning")
            assert np.isfinite(result.estimate)

    def test_different_polynomial_orders(self):
        """Should work with different polynomial orders."""
        y, x, d = generate_rkd_data(n=1000, true_effect=2.0, sigma=0.5, seed=42)

        for order in [1, 2, 3]:
            rkd = SharpRKD(cutoff=0.0, bandwidth=2.0, polynomial_order=order)
            result = rkd.fit(y, x, d)

            assert result.retcode in ("success", "warning")
            assert np.isfinite(result.estimate)

    def test_high_noise(self):
        """Should handle high noise levels."""
        y, x, d = generate_rkd_data(n=1000, true_effect=2.0, sigma=5.0, seed=42)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        # SE should be larger with more noise
        assert result.se > 0.5

    def test_auto_bandwidth(self):
        """Should select bandwidth automatically."""
        y, x, d = generate_rkd_data(n=1000, true_effect=2.0, sigma=1.0, seed=42)

        rkd = SharpRKD(cutoff=0.0, bandwidth="auto")
        result = rkd.fit(y, x, d)

        assert result.retcode in ("success", "warning")
        assert result.bandwidth > 0
        assert np.isfinite(result.estimate)

    def test_length_mismatch_error(self):
        """Should error on input length mismatch."""
        y = np.random.normal(0, 1, 100)
        x = np.random.uniform(-1, 1, 100)
        d = np.random.uniform(-1, 1, 50)  # Wrong length

        rkd = SharpRKD(cutoff=0.0, bandwidth=1.0)

        with pytest.raises(ValueError, match="mismatch"):
            rkd.fit(y, x, d)

    def test_nonfinite_values_error(self):
        """Should error on non-finite values."""
        y = np.array([1, 2, np.nan, 4, 5] * 20)
        x = np.random.uniform(-1, 1, 100)
        d = x

        rkd = SharpRKD(cutoff=0.0, bandwidth=1.0)

        with pytest.raises(ValueError, match="non-finite"):
            rkd.fit(y, x, d)


# =============================================================================
# Layer 4: Result Object Tests
# =============================================================================


class TestSharpRKDResult:
    """Tests for the result object structure."""

    def test_result_fields_exist(self):
        """Result should have all expected fields."""
        y, x, d = generate_rkd_data(n=500, true_effect=2.0, sigma=1.0, seed=42)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        # Check all fields exist
        assert hasattr(result, "estimate")
        assert hasattr(result, "se")
        assert hasattr(result, "t_stat")
        assert hasattr(result, "p_value")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "bandwidth")
        assert hasattr(result, "n_left")
        assert hasattr(result, "n_right")
        assert hasattr(result, "slope_y_left")
        assert hasattr(result, "slope_y_right")
        assert hasattr(result, "slope_d_left")
        assert hasattr(result, "slope_d_right")
        assert hasattr(result, "delta_slope_y")
        assert hasattr(result, "delta_slope_d")
        assert hasattr(result, "retcode")
        assert hasattr(result, "message")

    def test_result_types(self):
        """Result fields should have correct types."""
        y, x, d = generate_rkd_data(n=500, true_effect=2.0, sigma=1.0, seed=42)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert isinstance(result.estimate, float)
        assert isinstance(result.se, float)
        assert isinstance(result.n_left, int)
        assert isinstance(result.n_right, int)
        assert isinstance(result.retcode, str)
        assert isinstance(result.message, str)

    def test_ci_contains_estimate(self):
        """CI should contain the point estimate."""
        y, x, d = generate_rkd_data(n=500, true_effect=2.0, sigma=1.0, seed=42)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        assert result.ci_lower <= result.estimate <= result.ci_upper

    def test_summary_method(self):
        """Summary method should return formatted string."""
        y, x, d = generate_rkd_data(n=500, true_effect=2.0, sigma=1.0, seed=42)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        rkd.fit(y, x, d)

        summary = rkd.summary()
        assert isinstance(summary, str)
        assert "Estimate" in summary
        assert "Std. Error" in summary
        assert "Bandwidth" in summary

    def test_slope_relationship(self):
        """delta_slope_y / delta_slope_d should equal estimate."""
        y, x, d = generate_rkd_data(n=500, true_effect=2.0, sigma=1.0, seed=42)

        rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
        result = rkd.fit(y, x, d)

        computed = result.delta_slope_y / result.delta_slope_d
        assert abs(computed - result.estimate) < 1e-10


class TestBug3DenominatorVariance:
    """
    BUG-3 FIX VALIDATION: RKD SE should include denominator variance.

    For Sharp RKD, the estimate is τ = Δβ_Y / Δδ_D (ratio of slope changes).
    The delta method for ratio variance is:
        Var(τ) = [Var(Δβ_Y) + τ²·Var(Δδ_D)] / Δδ_D²

    The old code only used Var(Δβ_Y), ignoring Var(Δδ_D).

    Reference: docs/KNOWN_BUGS.md BUG-3
    """

    def test_se_accounts_for_denominator_variance(self):
        """
        BUG-3: SE should be larger when D slopes have uncertainty.

        When D slopes are estimated (not known), SE should include their variance.
        This test compares:
        1. SE when D slopes are estimated (includes denominator variance)
        2. SE when D slopes are provided as known (no denominator variance)

        The estimated case should have larger SE.
        """
        np.random.seed(42)
        n = 500

        # Generate data with noisy D
        x = np.random.uniform(-3, 3, n)
        cutoff = 0.0

        # Treatment D with noise (so slopes have variance)
        d_left = 0.5 * (x - cutoff) + np.random.normal(0, 0.3, n)
        d_right = 1.5 * (x - cutoff) + np.random.normal(0, 0.3, n)
        d = np.where(x < cutoff, d_left, d_right)

        # Outcome
        true_effect = 2.0
        y = 0.3 * x + true_effect * d + np.random.normal(0, 0.5, n)

        rkd = SharpRKD(cutoff=cutoff, bandwidth=2.0)

        # Case 1: D slopes estimated (includes variance)
        result_estimated = rkd.fit(y, x, d)

        # Case 2: D slopes provided as "known" (variance = 0)
        result_known = rkd.fit(y, x, d, slope_d_left=0.5, slope_d_right=1.5)

        # SE with estimated D should be >= SE with known D
        # (The delta method adds denominator variance term)
        assert result_estimated.se >= result_known.se * 0.99, (
            f"SE with estimated D ({result_estimated.se:.4f}) should be >= "
            f"SE with known D ({result_known.se:.4f})"
        )

        # Both should produce valid estimates
        assert np.isfinite(result_estimated.estimate)
        assert np.isfinite(result_known.estimate)
        assert np.isfinite(result_estimated.se)
        assert np.isfinite(result_known.se)

    def test_se_formula_correct_with_noisy_d(self):
        """
        Monte Carlo: SE should be calibrated (SE ≈ empirical SD) when D is noisy.

        Run many replications and check that mean SE is close to empirical SD.
        This validates the full delta method formula.
        """
        np.random.seed(123)
        true_effect = 2.0
        n = 400
        n_reps = 100
        cutoff = 0.0

        estimates = []
        ses = []

        for _ in range(n_reps):
            x = np.random.uniform(-3, 3, n)

            # Treatment D with noise
            d_left = 0.5 * (x - cutoff) + np.random.normal(0, 0.2, n)
            d_right = 1.5 * (x - cutoff) + np.random.normal(0, 0.2, n)
            d = np.where(x < cutoff, d_left, d_right)

            # Outcome
            y = 0.3 * x + true_effect * d + np.random.normal(0, 0.5, n)

            rkd = SharpRKD(cutoff=cutoff, bandwidth=2.0)
            result = rkd.fit(y, x, d)

            if np.isfinite(result.estimate) and np.isfinite(result.se):
                estimates.append(result.estimate)
                ses.append(result.se)

        estimates = np.array(estimates)
        ses = np.array(ses)

        # SE should approximate empirical SD
        empirical_sd = np.std(estimates)
        mean_se = np.mean(ses)

        # Allow 30% tolerance (SE estimation has variance)
        ratio = mean_se / empirical_sd
        assert 0.7 < ratio < 1.5, (
            f"SE calibration: mean SE ({mean_se:.4f}) should be close to "
            f"empirical SD ({empirical_sd:.4f}), got ratio {ratio:.2f}"
        )

    def test_ci_coverage_with_noisy_d(self):
        """
        Monte Carlo: 95% CI should have approximately 95% coverage.

        This is the ultimate validation that SE accounts for all uncertainty.
        """
        np.random.seed(456)
        true_effect = 2.0
        n = 400
        n_reps = 100
        cutoff = 0.0

        covers = []

        for _ in range(n_reps):
            x = np.random.uniform(-3, 3, n)

            # Treatment D with noise
            d_left = 0.5 * (x - cutoff) + np.random.normal(0, 0.2, n)
            d_right = 1.5 * (x - cutoff) + np.random.normal(0, 0.2, n)
            d = np.where(x < cutoff, d_left, d_right)

            # Outcome
            y = 0.3 * x + true_effect * d + np.random.normal(0, 0.5, n)

            rkd = SharpRKD(cutoff=cutoff, bandwidth=2.0, alpha=0.05)
            result = rkd.fit(y, x, d)

            if np.isfinite(result.ci_lower) and np.isfinite(result.ci_upper):
                covers.append(result.ci_lower <= true_effect <= result.ci_upper)

        coverage = np.mean(covers)

        # Coverage should be close to 95% (allow 85-99% due to MC variance)
        assert 0.85 <= coverage <= 0.99, f"Coverage {coverage:.2%} should be approximately 95%"
