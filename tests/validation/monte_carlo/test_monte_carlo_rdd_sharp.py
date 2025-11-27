"""
Monte Carlo validation for Sharp RDD estimator.

Key properties validated:
- Unbiasedness: E[τ̂] ≈ τ with linear DGP (bias < 0.05)
- Coverage: 95% CI contains true τ in 93-97% of simulations
- SE accuracy: Estimated SE ≈ empirical SD (within 15%)
- Bandwidth robustness: Estimates stable across 0.5h to 2h
- Kernel insensitivity: Triangular and rectangular agree

Key References:
    - Imbens & Lemieux (2008). "Regression Discontinuity Designs: A Guide to Practice"
    - Lee & Lemieux (2010). "Regression Discontinuity Designs in Economics"
    - Calonico, Cattaneo & Titiunik (2014). "Robust Nonparametric CI for RDD"

The key insight: Local linear regression is approximately unbiased for the
treatment effect at the cutoff, with bias determined by curvature and bandwidth.
"""

import numpy as np
import pytest
import warnings
from src.causal_inference.rdd import SharpRDD
from tests.validation.monte_carlo.dgp_rdd import (
    dgp_rdd_linear,
    dgp_rdd_quadratic,
    dgp_rdd_zero_effect,
    dgp_rdd_heteroskedastic,
    dgp_rdd_high_noise,
    dgp_rdd_different_slopes,
    dgp_rdd_small_sample,
    dgp_rdd_large_sample,
)
from tests.validation.utils import validate_monte_carlo_results


class TestSharpRDDUnbiasedness:
    """Test Sharp RDD unbiasedness across DGPs."""

    @pytest.mark.slow
    def test_sharp_rdd_unbiased_linear_dgp(self):
        """
        Sharp RDD should be unbiased with linear DGP.

        With E[Y|X] linear, local linear regression exactly recovers
        the true discontinuity regardless of bandwidth.
        """
        n_runs = 2000
        true_tau = 2.0

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_rdd_linear(n=500, true_tau=true_tau, random_state=seed)

            rdd = SharpRDD(cutoff=data.cutoff, bandwidth="ik", inference="robust")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd.fit(data.Y, data.X)

            estimates.append(rdd.coef_)
            standard_errors.append(rdd.se_)
            ci_lowers.append(rdd.ci_[0])
            ci_uppers.append(rdd.ci_[1])

        validation = validate_monte_carlo_results(
            estimates,
            standard_errors,
            ci_lowers,
            ci_uppers,
            true_tau,
            bias_threshold=0.05,
            coverage_lower=0.93,
            coverage_upper=0.97,
            se_accuracy_threshold=0.15,
        )

        assert validation["bias_ok"], (
            f"Sharp RDD bias {validation['bias']:.4f} exceeds 0.05 with linear DGP. "
            f"Mean estimate: {np.mean(estimates):.4f}, true τ: {true_tau}"
        )

    @pytest.mark.slow
    def test_sharp_rdd_unbiased_quadratic_dgp(self):
        """
        Sharp RDD should have small bias with quadratic DGP.

        Local linear regression has O(h²) bias with curvature,
        but IK bandwidth minimizes MSE.
        """
        n_runs = 2000
        true_tau = 2.0

        estimates = []

        for seed in range(n_runs):
            data = dgp_rdd_quadratic(n=500, true_tau=true_tau, random_state=seed)

            rdd = SharpRDD(cutoff=data.cutoff, bandwidth="ik", inference="robust")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd.fit(data.Y, data.X)

            estimates.append(rdd.coef_)

        bias = abs(np.mean(estimates) - true_tau)

        # Bias should be small even with curvature (local linear handles it)
        assert bias < 0.10, (
            f"Sharp RDD bias {bias:.4f} too large with quadratic DGP. "
            f"Expected < 0.10. Mean estimate: {np.mean(estimates):.4f}"
        )

    @pytest.mark.slow
    def test_sharp_rdd_zero_effect(self):
        """
        Sharp RDD should correctly identify zero effect.

        Estimate should be ≈0, and 95% CI should contain 0 in ~95% of runs.
        """
        n_runs = 2000
        true_tau = 0.0

        estimates = []
        ci_lowers = []
        ci_uppers = []
        p_values = []

        for seed in range(n_runs):
            data = dgp_rdd_zero_effect(n=500, random_state=seed)

            rdd = SharpRDD(cutoff=data.cutoff, bandwidth="ik", inference="robust")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd.fit(data.Y, data.X)

            estimates.append(rdd.coef_)
            ci_lowers.append(rdd.ci_[0])
            ci_uppers.append(rdd.ci_[1])
            p_values.append(rdd.p_value_)

        bias = abs(np.mean(estimates) - true_tau)
        coverage = np.mean(
            (np.array(ci_lowers) <= true_tau) & (true_tau <= np.array(ci_uppers))
        )

        # Type I error: reject rate when H₀ is true
        type1_error = np.mean(np.array(p_values) < 0.05)

        assert bias < 0.05, f"Mean estimate {np.mean(estimates):.4f} should be ≈0"
        assert 0.93 < coverage < 0.97, f"Coverage {coverage:.2%} should be ~95%"
        assert type1_error < 0.07, f"Type I error {type1_error:.2%} should be ≤5%"


class TestSharpRDDCoverage:
    """Test Sharp RDD confidence interval coverage."""

    @pytest.mark.slow
    def test_sharp_rdd_coverage_linear_dgp(self):
        """
        95% CI should contain true τ in 93-97% of simulations.
        """
        n_runs = 2000
        true_tau = 2.0

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_rdd_linear(n=500, true_tau=true_tau, random_state=seed)

            rdd = SharpRDD(cutoff=data.cutoff, bandwidth="ik", inference="robust")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd.fit(data.Y, data.X)

            estimates.append(rdd.coef_)
            standard_errors.append(rdd.se_)
            ci_lowers.append(rdd.ci_[0])
            ci_uppers.append(rdd.ci_[1])

        validation = validate_monte_carlo_results(
            estimates,
            standard_errors,
            ci_lowers,
            ci_uppers,
            true_tau,
            bias_threshold=0.10,
            coverage_lower=0.93,
            coverage_upper=0.97,
            se_accuracy_threshold=0.15,
        )

        assert validation["coverage_ok"], (
            f"Sharp RDD coverage {validation['coverage']:.2%} outside [93%, 97%]"
        )

    @pytest.mark.slow
    def test_sharp_rdd_coverage_robust_vs_standard(self):
        """
        Robust SEs should provide better coverage with heteroskedasticity.
        """
        n_runs = 1500
        true_tau = 2.0

        robust_covers = []
        standard_covers = []

        for seed in range(n_runs):
            data = dgp_rdd_heteroskedastic(
                n=500, true_tau=true_tau, random_state=seed
            )

            # Robust inference
            rdd_robust = SharpRDD(
                cutoff=data.cutoff, bandwidth="ik", inference="robust"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd_robust.fit(data.Y, data.X)

            robust_covered = rdd_robust.ci_[0] <= true_tau <= rdd_robust.ci_[1]
            robust_covers.append(robust_covered)

            # Standard inference
            rdd_standard = SharpRDD(
                cutoff=data.cutoff, bandwidth="ik", inference="standard"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd_standard.fit(data.Y, data.X)

            standard_covered = rdd_standard.ci_[0] <= true_tau <= rdd_standard.ci_[1]
            standard_covers.append(standard_covered)

        robust_coverage = np.mean(robust_covers)
        standard_coverage = np.mean(standard_covers)

        # Robust should have reasonable coverage
        assert 0.90 < robust_coverage < 0.98, (
            f"Robust coverage {robust_coverage:.2%} outside acceptable range"
        )

        # Document difference (robust typically better with heteroskedasticity)
        # Standard SEs may undercover
        assert robust_coverage >= standard_coverage - 0.05, (
            f"Robust coverage ({robust_coverage:.2%}) should be >= "
            f"standard coverage ({standard_coverage:.2%}) with heteroskedasticity"
        )


class TestSharpRDDSEAccuracy:
    """Test Sharp RDD standard error accuracy."""

    @pytest.mark.slow
    def test_sharp_rdd_se_accuracy_linear_dgp(self):
        """
        Estimated SE should be close to empirical SD of estimates.
        """
        n_runs = 2000
        true_tau = 2.0

        estimates = []
        standard_errors = []

        for seed in range(n_runs):
            data = dgp_rdd_linear(n=500, true_tau=true_tau, random_state=seed)

            rdd = SharpRDD(cutoff=data.cutoff, bandwidth="ik", inference="robust")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd.fit(data.Y, data.X)

            estimates.append(rdd.coef_)
            standard_errors.append(rdd.se_)

        empirical_sd = np.std(estimates)
        mean_se = np.mean(standard_errors)
        se_ratio = mean_se / empirical_sd

        # SE should be within 15% of empirical SD
        assert 0.85 < se_ratio < 1.15, (
            f"SE ratio {se_ratio:.2f} outside [0.85, 1.15]. "
            f"Empirical SD: {empirical_sd:.4f}, Mean SE: {mean_se:.4f}"
        )

    @pytest.mark.slow
    def test_sharp_rdd_se_scales_with_noise(self):
        """
        SE should scale proportionally with error variance.
        """
        n_runs = 1000
        true_tau = 2.0

        ses_low_noise = []
        ses_high_noise = []

        for seed in range(n_runs):
            # Low noise (error_sd = 1.0)
            data_low = dgp_rdd_linear(
                n=500, true_tau=true_tau, error_sd=1.0, random_state=seed
            )
            rdd_low = SharpRDD(cutoff=data_low.cutoff, bandwidth="ik", inference="robust")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd_low.fit(data_low.Y, data_low.X)
            ses_low_noise.append(rdd_low.se_)

            # High noise (error_sd = 3.0)
            data_high = dgp_rdd_high_noise(n=500, true_tau=true_tau, random_state=seed)
            rdd_high = SharpRDD(cutoff=data_high.cutoff, bandwidth="ik", inference="robust")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd_high.fit(data_high.Y, data_high.X)
            ses_high_noise.append(rdd_high.se_)

        mean_se_low = np.mean(ses_low_noise)
        mean_se_high = np.mean(ses_high_noise)

        # SE should scale roughly proportionally with error SD
        # High noise has 3x error SD, so SE should be ~3x larger
        se_ratio = mean_se_high / mean_se_low

        assert 2.0 < se_ratio < 4.0, (
            f"SE ratio {se_ratio:.2f} should be ~3.0 (proportional to error SD ratio). "
            f"Low noise SE: {mean_se_low:.4f}, High noise SE: {mean_se_high:.4f}"
        )


class TestSharpRDDBandwidthRobustness:
    """Test Sharp RDD robustness to bandwidth choice."""

    @pytest.mark.slow
    def test_sharp_rdd_bandwidth_sensitivity(self):
        """
        Estimates should be stable across reasonable bandwidth choices.

        Test 0.5×h_IK, h_IK, 2×h_IK bandwidths.
        """
        n_runs = 1500
        true_tau = 2.0

        estimates_ik = []
        estimates_narrow = []
        estimates_wide = []

        for seed in range(n_runs):
            data = dgp_rdd_linear(n=500, true_tau=true_tau, random_state=seed)

            # IK bandwidth
            rdd_ik = SharpRDD(cutoff=data.cutoff, bandwidth="ik", inference="robust")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd_ik.fit(data.Y, data.X)

            h_ik = rdd_ik.bandwidth_left_
            estimates_ik.append(rdd_ik.coef_)

            # Narrow bandwidth (0.5 × h_IK)
            rdd_narrow = SharpRDD(
                cutoff=data.cutoff, bandwidth=0.5 * h_ik, inference="robust"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd_narrow.fit(data.Y, data.X)
            estimates_narrow.append(rdd_narrow.coef_)

            # Wide bandwidth (2.0 × h_IK)
            rdd_wide = SharpRDD(
                cutoff=data.cutoff, bandwidth=2.0 * h_ik, inference="robust"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd_wide.fit(data.Y, data.X)
            estimates_wide.append(rdd_wide.coef_)

        # All should be unbiased
        bias_ik = abs(np.mean(estimates_ik) - true_tau)
        bias_narrow = abs(np.mean(estimates_narrow) - true_tau)
        bias_wide = abs(np.mean(estimates_wide) - true_tau)

        assert bias_ik < 0.10, f"IK bandwidth bias {bias_ik:.4f} too large"
        assert bias_narrow < 0.10, f"Narrow bandwidth bias {bias_narrow:.4f} too large"
        assert bias_wide < 0.10, f"Wide bandwidth bias {bias_wide:.4f} too large"

        # Stability: estimates shouldn't vary too much across bandwidths
        mean_diff = np.mean(np.abs(np.array(estimates_narrow) - np.array(estimates_wide)))
        assert mean_diff < 0.30, (
            f"Estimates unstable across bandwidths: |narrow - wide| = {mean_diff:.4f}"
        )


class TestSharpRDDKernelComparison:
    """Test Sharp RDD kernel comparison."""

    @pytest.mark.slow
    def test_triangular_vs_rectangular_kernel(self):
        """
        Triangular and rectangular kernels should give similar results.

        Triangular is preferred (efficient), but results shouldn't differ much.
        """
        n_runs = 1500
        true_tau = 2.0

        estimates_triangular = []
        estimates_rectangular = []

        for seed in range(n_runs):
            data = dgp_rdd_linear(n=500, true_tau=true_tau, random_state=seed)

            # Triangular kernel
            rdd_tri = SharpRDD(
                cutoff=data.cutoff, bandwidth="ik", kernel="triangular", inference="robust"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd_tri.fit(data.Y, data.X)
            estimates_triangular.append(rdd_tri.coef_)

            # Rectangular kernel (same bandwidth for fair comparison)
            h = rdd_tri.bandwidth_left_
            rdd_rect = SharpRDD(
                cutoff=data.cutoff, bandwidth=h, kernel="rectangular", inference="robust"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd_rect.fit(data.Y, data.X)
            estimates_rectangular.append(rdd_rect.coef_)

        # Both should be unbiased
        bias_tri = abs(np.mean(estimates_triangular) - true_tau)
        bias_rect = abs(np.mean(estimates_rectangular) - true_tau)

        assert bias_tri < 0.10, f"Triangular kernel bias {bias_tri:.4f} too large"
        assert bias_rect < 0.10, f"Rectangular kernel bias {bias_rect:.4f} too large"

        # Results should be similar (within 30% on average)
        mean_diff = np.mean(
            np.abs(np.array(estimates_triangular) - np.array(estimates_rectangular))
        )
        assert mean_diff < 0.30, (
            f"Kernels disagree too much: |tri - rect| = {mean_diff:.4f}"
        )


class TestSharpRDDSampleSize:
    """Test Sharp RDD behavior across sample sizes."""

    @pytest.mark.slow
    def test_sharp_rdd_small_sample(self):
        """
        Sharp RDD should still work with small samples (n=100).

        Coverage may be slightly lower, SEs larger.
        """
        n_runs = 2000
        true_tau = 2.0

        estimates = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_rdd_small_sample(n=100, true_tau=true_tau, random_state=seed)

            rdd = SharpRDD(cutoff=data.cutoff, bandwidth="ik", inference="robust")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd.fit(data.Y, data.X)

            estimates.append(rdd.coef_)
            ci_lowers.append(rdd.ci_[0])
            ci_uppers.append(rdd.ci_[1])

        bias = abs(np.mean(estimates) - true_tau)
        coverage = np.mean(
            (np.array(ci_lowers) <= true_tau) & (true_tau <= np.array(ci_uppers))
        )

        # Small sample: bias acceptable, coverage may be slightly lower
        assert bias < 0.15, f"Small sample bias {bias:.4f} too large"
        assert coverage > 0.88, f"Small sample coverage {coverage:.2%} too low"

    @pytest.mark.slow
    def test_sharp_rdd_se_decreases_with_n(self):
        """
        SE should decrease proportionally to 1/√n.
        """
        n_runs = 1000
        true_tau = 2.0

        ses_small = []  # n = 200
        ses_large = []  # n = 800

        for seed in range(n_runs):
            # Small sample
            data_small = dgp_rdd_linear(n=200, true_tau=true_tau, random_state=seed)
            rdd_small = SharpRDD(cutoff=data_small.cutoff, bandwidth="ik", inference="robust")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd_small.fit(data_small.Y, data_small.X)
            ses_small.append(rdd_small.se_)

            # Large sample
            data_large = dgp_rdd_linear(n=800, true_tau=true_tau, random_state=seed)
            rdd_large = SharpRDD(cutoff=data_large.cutoff, bandwidth="ik", inference="robust")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd_large.fit(data_large.Y, data_large.X)
            ses_large.append(rdd_large.se_)

        mean_se_small = np.mean(ses_small)
        mean_se_large = np.mean(ses_large)

        # SE ratio should be ~√(800/200) = 2.0
        # But RDD SE scales differently due to bandwidth effects
        se_ratio = mean_se_small / mean_se_large

        # SE should decrease substantially with sample size
        assert se_ratio > 1.3, (
            f"SE ratio {se_ratio:.2f} should be > 1.3 (SE decreases with n). "
            f"Small n SE: {mean_se_small:.4f}, Large n SE: {mean_se_large:.4f}"
        )


class TestSharpRDDDifferentSlopes:
    """Test Sharp RDD with different slopes on each side."""

    @pytest.mark.slow
    def test_sharp_rdd_different_slopes(self):
        """
        Sharp RDD should recover treatment effect with different slopes.

        Local linear regression handles asymmetric slopes naturally.
        """
        n_runs = 2000
        true_tau = 2.0

        estimates = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            data = dgp_rdd_different_slopes(
                n=500,
                true_tau=true_tau,
                slope_left=1.0,
                slope_right=0.5,
                random_state=seed,
            )

            rdd = SharpRDD(cutoff=data.cutoff, bandwidth="ik", inference="robust")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rdd.fit(data.Y, data.X)

            estimates.append(rdd.coef_)
            ci_lowers.append(rdd.ci_[0])
            ci_uppers.append(rdd.ci_[1])

        bias = abs(np.mean(estimates) - true_tau)
        coverage = np.mean(
            (np.array(ci_lowers) <= true_tau) & (true_tau <= np.array(ci_uppers))
        )

        assert bias < 0.10, (
            f"Different slopes bias {bias:.4f} too large. "
            f"Mean estimate: {np.mean(estimates):.4f}"
        )
        assert 0.90 < coverage < 0.98, (
            f"Different slopes coverage {coverage:.2%} outside acceptable range"
        )
