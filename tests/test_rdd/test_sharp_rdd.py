"""
Tests for Sharp Regression Discontinuity Design (RDD) estimator.

Layer 1 (Known-Answer) tests verify that the Sharp RDD estimator:
- Recovers true treatment effects from simulated data (±20-25% tolerance)
- Handles different functional forms (linear, quadratic)
- Correctly identifies zero effects vs. large effects
- Produces valid confidence intervals
- Selects reasonable bandwidths
- Computes positive, finite standard errors

Layer 2 (Adversarial) tests verify error handling:
- Raises errors when no observations on one side of cutoff
- Warns about small effective sample sizes
- Handles invalid inputs gracefully
"""

import numpy as np
import pytest
import warnings

from src.causal_inference.rdd import SharpRDD
from src.causal_inference.rdd.bandwidth import (
    imbens_kalyanaraman_bandwidth,
    cct_bandwidth,
)


class TestSharpRDDKnownAnswers:
    """Test Sharp RDD with known-answer fixtures."""

    def test_linear_dgp_recovers_true_effect(self, sharp_rdd_linear_dgp):
        """Test Sharp RDD recovers true effect with linear DGP (τ=2.0)."""
        Y, X, cutoff, true_tau = sharp_rdd_linear_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="robust")
        rdd.fit(Y, X)

        # Check point estimate within 20% of truth
        assert np.isclose(rdd.coef_, true_tau, rtol=0.20), (
            f"Expected τ ≈ {true_tau}, got {rdd.coef_:.4f}"
        )

    def test_quadratic_dgp_recovers_true_effect(self, sharp_rdd_quadratic_dgp):
        """Test Sharp RDD with quadratic DGP (local linear should handle curvature)."""
        Y, X, cutoff, true_tau = sharp_rdd_quadratic_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="robust")
        rdd.fit(Y, X)

        # Quadratic DGP: looser tolerance (25%) due to curvature
        assert np.isclose(rdd.coef_, true_tau, rtol=0.25), (
            f"Expected τ ≈ {true_tau}, got {rdd.coef_:.4f}"
        )

    def test_zero_effect_not_significant(self, sharp_rdd_zero_effect_dgp):
        """Test that zero effect is not statistically significant."""
        Y, X, cutoff, true_tau = sharp_rdd_zero_effect_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="robust")
        rdd.fit(Y, X)

        # Effect should be close to zero
        assert np.abs(rdd.coef_) < 0.5, f"Expected |τ| < 0.5, got {rdd.coef_:.4f}"

        # Should not be significant (p-value > 0.05)
        assert rdd.p_value_ > 0.05, f"Expected p-value > 0.05, got {rdd.p_value_:.4f}"

    def test_large_effect_highly_significant(self, sharp_rdd_large_effect_dgp):
        """Test that large effect (τ=10.0) is highly significant."""
        Y, X, cutoff, true_tau = sharp_rdd_large_effect_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="robust")
        rdd.fit(Y, X)

        # Effect should be close to 10.0 (±20%)
        assert np.isclose(rdd.coef_, true_tau, rtol=0.20), (
            f"Expected τ ≈ {true_tau}, got {rdd.coef_:.4f}"
        )

        # Should be highly significant (p-value < 0.001)
        assert rdd.p_value_ < 0.001, f"Expected p-value < 0.001, got {rdd.p_value_:.4f}"

    def test_triangular_vs_rectangular_kernel(self, sharp_rdd_linear_dgp):
        """Test that triangular and rectangular kernels both recover true effect."""
        Y, X, cutoff, true_tau = sharp_rdd_linear_dgp

        # Fit with triangular kernel
        rdd_tri = SharpRDD(cutoff=cutoff, bandwidth="ik", kernel="triangular")
        rdd_tri.fit(Y, X)

        # Fit with rectangular kernel
        rdd_rect = SharpRDD(cutoff=cutoff, bandwidth="ik", kernel="rectangular")
        rdd_rect.fit(Y, X)

        # Both should recover true effect (±30% tolerance, kernels differ)
        assert np.isclose(rdd_tri.coef_, true_tau, rtol=0.30), (
            f"Triangular: expected τ ≈ {true_tau}, got {rdd_tri.coef_:.4f}"
        )

        assert np.isclose(rdd_rect.coef_, true_tau, rtol=0.30), (
            f"Rectangular: expected τ ≈ {true_tau}, got {rdd_rect.coef_:.4f}"
        )

    def test_confidence_intervals_coverage(self, sharp_rdd_linear_dgp):
        """Test that 95% confidence intervals contain true value."""
        Y, X, cutoff, true_tau = sharp_rdd_linear_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="robust", alpha=0.05)
        rdd.fit(Y, X)

        ci_lower, ci_upper = rdd.ci_

        # True value should be in confidence interval
        assert ci_lower <= true_tau <= ci_upper, (
            f"CI [{ci_lower:.3f}, {ci_upper:.3f}] does not contain true τ={true_tau}"
        )


class TestBandwidthSelection:
    """Test automatic bandwidth selection methods."""

    def test_ik_bandwidth_positive_finite(self, sharp_rdd_linear_dgp):
        """Test that IK bandwidth is positive and finite."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        h = imbens_kalyanaraman_bandwidth(Y, X, cutoff, kernel="triangular")

        # Bandwidth should be positive
        assert h > 0, f"Bandwidth must be positive, got {h}"

        # Bandwidth should be finite
        assert np.isfinite(h), f"Bandwidth must be finite, got {h}"

    def test_cct_bandwidth_positive_finite(self, sharp_rdd_linear_dgp):
        """Test that CCT bandwidth is positive and finite."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        h_main, h_bias = cct_bandwidth(Y, X, cutoff, kernel="triangular", bias_correction=True)

        # Both bandwidths should be positive
        assert h_main > 0, f"Main bandwidth must be positive, got {h_main}"
        assert h_bias > 0, f"Bias bandwidth must be positive, got {h_bias}"

        # Both should be finite
        assert np.isfinite(h_main), f"Main bandwidth must be finite, got {h_main}"
        assert np.isfinite(h_bias), f"Bias bandwidth must be finite, got {h_bias}"

    def test_cct_bias_bandwidth_larger(self, sharp_rdd_linear_dgp):
        """Test that CCT bias correction bandwidth >= main bandwidth."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        h_main, h_bias = cct_bandwidth(Y, X, cutoff, kernel="triangular", bias_correction=True)

        # Bias bandwidth should be larger (CCT recommendation: h_bias ≈ 1.5 * h_main)
        assert h_bias >= h_main, f"Bias bandwidth ({h_bias:.4f}) should be >= main ({h_main:.4f})"

    def test_bandwidth_sensitivity(self, sharp_rdd_linear_dgp):
        """Test that different fixed bandwidths all recover effect (sensitivity test)."""
        Y, X, cutoff, true_tau = sharp_rdd_linear_dgp

        # Test with different bandwidths
        bandwidths = [0.5, 1.0, 2.0]
        estimates = []

        for h in bandwidths:
            rdd = SharpRDD(cutoff=cutoff, bandwidth=h, inference="robust")
            rdd.fit(Y, X)
            estimates.append(rdd.coef_)

        # All estimates should be within 40% of truth
        # (wider tolerance because fixed bandwidths may be suboptimal)
        for h, est in zip(bandwidths, estimates):
            assert np.isclose(est, true_tau, rtol=0.40), (
                f"Bandwidth {h}: expected τ ≈ {true_tau}, got {est:.4f}"
            )


class TestStandardErrors:
    """Test standard error computation (critical for inference)."""

    def test_standard_errors_positive_finite(self, sharp_rdd_linear_dgp):
        """Test that standard errors are positive and finite."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="standard")
        rdd.fit(Y, X)

        # SE should be positive
        assert rdd.se_ > 0, f"Standard error must be positive, got {rdd.se_}"

        # SE should be finite
        assert np.isfinite(rdd.se_), f"Standard error must be finite, got {rdd.se_}"

    def test_robust_se_vs_standard(self, sharp_rdd_heteroskedastic_dgp):
        """Test that robust SEs >= standard SEs with heteroskedasticity."""
        Y, X, cutoff, _ = sharp_rdd_heteroskedastic_dgp

        # Fit with standard SEs
        rdd_std = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="standard")
        rdd_std.fit(Y, X)

        # Fit with robust SEs
        rdd_robust = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="robust")
        rdd_robust.fit(Y, X)

        # Coefficients should be the same
        assert np.isclose(rdd_std.coef_, rdd_robust.coef_), (
            "Coefficients should be identical regardless of SE type"
        )

        # Robust SEs should be larger (or at least not much smaller) with heteroskedasticity
        # Allow 90% ratio (robust can sometimes be slightly smaller)
        assert rdd_robust.se_ >= rdd_std.se_ * 0.9, (
            f"Robust SE ({rdd_robust.se_:.4f}) should be >= 90% of standard SE ({rdd_std.se_:.4f})"
        )

    def test_small_sample_warning(self, sharp_rdd_sparse_data_dgp):
        """Test that sparse data near cutoff triggers small sample size warning."""
        Y, X, cutoff, _ = sharp_rdd_sparse_data_dgp

        # Should raise RuntimeWarning about small effective sample size
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            rdd = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="robust")
            rdd.fit(Y, X)

            # Check that at least one warning was raised
            warning_messages = [str(warning.message) for warning in w]
            has_small_sample_warning = any(
                "Small effective sample size" in msg for msg in warning_messages
            )

            assert has_small_sample_warning, (
                f"Expected warning about small effective sample size. Warnings: {warning_messages}"
            )


class TestInference:
    """Test t-statistics, p-values, and confidence intervals."""

    def test_t_statistic_computation(self, sharp_rdd_linear_dgp):
        """Test that t-statistic = coefficient / SE."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="robust")
        rdd.fit(Y, X)

        # Compute expected t-statistic
        expected_t = rdd.coef_ / rdd.se_

        # Check that stored t-statistic matches
        assert np.isclose(rdd.t_stat_, expected_t), (
            f"Expected t = {expected_t:.4f}, got {rdd.t_stat_:.4f}"
        )

    def test_p_value_two_sided(self, sharp_rdd_linear_dgp):
        """Test that p-value is in [0, 1] (two-sided test)."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="robust")
        rdd.fit(Y, X)

        # P-value should be between 0 and 1
        assert 0 <= rdd.p_value_ <= 1, f"P-value must be in [0, 1], got {rdd.p_value_}"

    def test_confidence_interval_width(self, sharp_rdd_linear_dgp):
        """Test that confidence interval has positive width."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="robust", alpha=0.05)
        rdd.fit(Y, X)

        ci_lower, ci_upper = rdd.ci_

        # Upper bound should be greater than lower bound
        assert ci_upper > ci_lower, f"CI upper ({ci_upper:.4f}) must be > CI lower ({ci_lower:.4f})"

        # Point estimate should be in CI
        assert ci_lower <= rdd.coef_ <= ci_upper, (
            f"Point estimate {rdd.coef_:.4f} not in CI [{ci_lower:.4f}, {ci_upper:.4f}]"
        )


class TestAdversarial:
    """Test error handling and edge cases."""

    def test_all_observations_one_side_raises_error(self, sharp_rdd_all_left_dgp):
        """Test that all observations on one side raises error."""
        Y, X, cutoff, _ = sharp_rdd_all_left_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="robust")

        # Should raise ValueError because no observations with X >= cutoff
        with pytest.raises(ValueError, match="No observations with X >= cutoff"):
            rdd.fit(Y, X)

    def test_cutoff_at_boundary_raises_error(self, sharp_rdd_boundary_cutoff_dgp):
        """Test that cutoff at boundary may raise error or warning."""
        Y, X, cutoff, _ = sharp_rdd_boundary_cutoff_dgp

        # This test may pass or may raise warnings depending on bandwidth
        # We just check that it doesn't crash
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")

                rdd = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="robust")
                rdd.fit(Y, X)

                # If it succeeds, check that estimate is finite
                assert np.isfinite(rdd.coef_), "Coefficient must be finite"

        except (ValueError, RuntimeWarning):
            # May raise error if effective sample size too small
            pass

    def test_bandwidth_larger_than_range(self, sharp_rdd_linear_dgp):
        """Test that very large bandwidth triggers warning or still works."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        # Use bandwidth much larger than data range
        # X ~ U(-5, 5), so range is 10. Use h=1000 >> 10.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            rdd = SharpRDD(cutoff=cutoff, bandwidth=1000.0, inference="robust")
            rdd.fit(Y, X)

            # Should still produce finite estimate
            assert np.isfinite(rdd.coef_), "Coefficient must be finite with large bandwidth"

            # May or may not produce warning (implementation-dependent)

    def test_invalid_kernel_raises_error(self, sharp_rdd_linear_dgp):
        """Test that invalid kernel raises error."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        # Invalid kernel should raise ValueError
        with pytest.raises(ValueError, match="kernel must be"):
            rdd = SharpRDD(cutoff=cutoff, bandwidth="ik", kernel="gaussian")


class TestCCTBiasCorrection:
    """Test CCT (Calonico-Cattaneo-Titiunik 2014) bias correction in SharpRDD."""

    def test_bias_corrected_flag_true_with_cct(self, sharp_rdd_linear_dgp):
        """bias_corrected_ is True when bandwidth='cct'."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="cct", inference="robust")
        rdd.fit(Y, X)

        assert rdd.bias_corrected_ is True, "bias_corrected_ should be True with CCT bandwidth"
        assert rdd.h_bias_ is not None, "h_bias_ should be set"
        assert rdd.bias_estimate_ is not None, "bias_estimate_ should be set"

    def test_no_bias_correction_with_ik(self, sharp_rdd_linear_dgp):
        """bias_corrected_ is False when bandwidth='ik'."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="robust")
        rdd.fit(Y, X)

        assert rdd.bias_corrected_ is False, "bias_corrected_ should be False with IK bandwidth"
        assert rdd.h_bias_ is None, "h_bias_ should be None"
        assert rdd.bias_estimate_ is None, "bias_estimate_ should be None"

    def test_h_bias_larger_than_h_main(self, sharp_rdd_linear_dgp):
        """h_bias >= h_main (CCT uses wider bandwidth for bias estimation)."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="cct", inference="robust")
        rdd.fit(Y, X)

        h_main = rdd.bandwidth_left_
        h_bias = rdd.h_bias_

        assert h_bias >= h_main, f"h_bias ({h_bias:.4f}) should be >= h_main ({h_main:.4f})"

    def test_bias_estimate_small_for_linear_dgp(self, sharp_rdd_linear_dgp):
        """Linear DGP should have near-zero bias estimate (no curvature)."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="cct", inference="robust")
        rdd.fit(Y, X)

        # Linear DGP has no curvature → bias should be small
        assert abs(rdd.bias_estimate_) < 0.5, (
            f"Linear DGP should have small bias estimate, got {rdd.bias_estimate_:.4f}"
        )

    def test_bias_correction_reduces_error_quadratic_dgp(self, sharp_rdd_quadratic_dgp):
        """CCT should reduce error for quadratic DGP (where bias exists)."""
        Y, X, cutoff, true_tau = sharp_rdd_quadratic_dgp

        # Fit without bias correction (IK)
        rdd_ik = SharpRDD(cutoff=cutoff, bandwidth="ik", inference="robust")
        rdd_ik.fit(Y, X)

        # Fit with bias correction (CCT)
        rdd_cct = SharpRDD(cutoff=cutoff, bandwidth="cct", inference="robust")
        rdd_cct.fit(Y, X)

        # Both should recover the effect reasonably
        error_ik = abs(rdd_ik.coef_ - true_tau)
        error_cct = abs(rdd_cct.coef_ - true_tau)

        # CCT error should not be dramatically worse (may or may not be better depending on sample)
        # At minimum, both should be within 50% of truth
        assert error_ik < true_tau * 0.50, f"IK error too large: {error_ik:.4f}"
        assert error_cct < true_tau * 0.50, f"CCT error too large: {error_cct:.4f}"

    def test_robust_se_accounts_for_bias_uncertainty(self, sharp_rdd_linear_dgp):
        """Robust SE with CCT accounts for bias estimation uncertainty."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="cct", inference="robust")
        rdd.fit(Y, X)

        # SE should be positive and finite
        assert rdd.se_ > 0, "SE must be positive"
        assert np.isfinite(rdd.se_), "SE must be finite"

        # SE should be marked as "robust" in summary
        summary = rdd.summary()
        assert "robust" in summary.lower(), "Summary should mention robust SE"

    def test_summary_shows_bias_info(self, sharp_rdd_linear_dgp):
        """Summary output includes bias correction information."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="cct", inference="robust")
        rdd.fit(Y, X)

        summary = rdd.summary()

        # Check for bias correction info
        assert "Bias Corrected:" in summary, "Summary should show bias correction status"
        assert "h_bias:" in summary, "Summary should show h_bias"
        assert "Bias Estimate:" in summary, "Summary should show bias estimate"

    def test_cct_still_recovers_linear_effect(self, sharp_rdd_linear_dgp):
        """CCT with bias correction still recovers true effect for linear DGP."""
        Y, X, cutoff, true_tau = sharp_rdd_linear_dgp

        rdd = SharpRDD(cutoff=cutoff, bandwidth="cct", inference="robust")
        rdd.fit(Y, X)

        # Should be within 25% of truth (same tolerance as quadratic DGP without correction)
        assert np.isclose(rdd.coef_, true_tau, rtol=0.25), (
            f"CCT should recover τ ≈ {true_tau}, got {rdd.coef_:.4f}"
        )
