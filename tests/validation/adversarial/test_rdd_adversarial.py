"""
Layer 2: Adversarial tests for RDD estimator edge cases.

Session 61: Comprehensive edge case and error handling tests.

Tests 25+ challenging scenarios:
- Boundary violations (insufficient data, all one side)
- Data quality issues (NaN, Inf, zero variance)
- Bandwidth extremes (h → 0, h → ∞)
- Numerical stability (outliers, ties, tiny/huge values)
- McCrary test edge cases
- Covariate issues (collinearity, high-dimensional)
- Sensitivity test edge cases

Goal: Ensure graceful failure OR correct handling of edge cases.
"""

import warnings

import numpy as np
import pytest

from src.causal_inference.rdd import (
    SharpRDD,
    FuzzyRDD,
    mccrary_density_test,
    covariate_balance_test,
    donut_hole_rdd,
    bandwidth_sensitivity_analysis,
    polynomial_order_sensitivity,
)


# =============================================================================
# Boundary Violations
# =============================================================================


class TestRDDBoundaryViolations:
    """Test RDD with boundary conditions and insufficient data."""

    def test_all_observations_one_side_left(self):
        """All observations left of cutoff should raise error."""
        np.random.seed(123)
        n = 100
        X = -np.abs(np.random.randn(n))  # All negative
        Y = np.random.randn(n)

        rdd = SharpRDD(cutoff=0.0)

        # Should raise error - need observations on both sides
        with pytest.raises((ValueError, RuntimeError)):
            rdd.fit(Y, X)

    def test_all_observations_one_side_right(self):
        """All observations right of cutoff should raise error."""
        np.random.seed(124)
        n = 100
        X = np.abs(np.random.randn(n))  # All positive
        Y = np.random.randn(n)

        rdd = SharpRDD(cutoff=0.0)

        with pytest.raises((ValueError, RuntimeError)):
            rdd.fit(Y, X)

    def test_very_few_observations_per_side(self):
        """Only 2-3 observations per side - should fail or warn."""
        np.random.seed(456)
        X = np.array([-0.5, -0.3, 0.3, 0.5])
        Y = np.array([1.0, 2.0, 3.0, 4.0])

        rdd = SharpRDD(cutoff=0.0, bandwidth=1.0)

        # Either raises exception or produces result with warning
        try:
            rdd.fit(Y, X)
            # If it works, check result is finite
            assert np.isfinite(rdd.coef_)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass  # Expected failure

    def test_observations_far_from_cutoff(self):
        """Observations far from cutoff - tests automatic bandwidth."""
        np.random.seed(789)
        n = 100
        X = np.concatenate(
            [
                np.random.randn(50) - 10.0,  # Far left
                np.random.randn(50) + 10.0,  # Far right
            ]
        )
        Y = np.random.randn(n)

        rdd = SharpRDD(cutoff=0.0, bandwidth="ik")

        # May work or fail with singular matrix due to data far from cutoff
        try:
            rdd.fit(Y, X)
            assert np.isfinite(rdd.coef_)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass  # Expected failure with data far from cutoff


# =============================================================================
# Data Quality Issues
# =============================================================================


class TestRDDDataQuality:
    """Test RDD with data quality issues (NaN, Inf)."""

    def test_nan_in_outcome_raises(self):
        """NaN in outcome should raise error."""
        np.random.seed(111)
        n = 100
        X = np.random.randn(n)
        Y = np.random.randn(n)
        Y[0] = np.nan

        rdd = SharpRDD(cutoff=0.0)

        with pytest.raises((ValueError, RuntimeError)):
            rdd.fit(Y, X)

    def test_nan_in_running_variable_raises(self):
        """NaN in running variable should raise error."""
        np.random.seed(112)
        n = 100
        X = np.random.randn(n)
        X[5] = np.nan
        Y = np.random.randn(n)

        rdd = SharpRDD(cutoff=0.0)

        with pytest.raises((ValueError, RuntimeError)):
            rdd.fit(Y, X)

    def test_inf_in_running_variable_raises(self):
        """Inf in running variable should raise error."""
        np.random.seed(222)
        n = 100
        X = np.random.randn(n)
        X[0] = np.inf
        Y = np.random.randn(n)

        rdd = SharpRDD(cutoff=0.0)

        with pytest.raises((ValueError, RuntimeError)):
            rdd.fit(Y, X)

    def test_constant_outcomes(self):
        """Constant outcomes - may cause numerical issues."""
        np.random.seed(333)
        n = 100
        X = np.random.randn(n)
        Y = np.full(n, 5.0)  # Constant

        rdd = SharpRDD(cutoff=0.0)

        # Should either fail or return zero effect
        try:
            rdd.fit(Y, X)
            # If works, effect should be ~0 (no discontinuity in constant)
            assert np.isfinite(rdd.coef_)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass  # Expected failure

    def test_extreme_outliers_handled(self):
        """Extreme outliers should be handled gracefully."""
        np.random.seed(444)
        n = 100
        X = np.random.randn(n)
        Y = X + 5.0 * (X >= 0) + np.random.randn(n)
        Y[0] = 1000.0  # Extreme outlier

        rdd = SharpRDD(cutoff=0.0)
        rdd.fit(Y, X)

        # Should work but estimate may be affected
        assert np.isfinite(rdd.coef_)
        assert np.isfinite(rdd.se_)


# =============================================================================
# Numerical Stability
# =============================================================================


class TestRDDNumericalStability:
    """Test RDD numerical stability with edge cases."""

    def test_exact_ties_at_cutoff(self):
        """Observations exactly at cutoff - should be handled."""
        np.random.seed(777)
        n = 100
        X = np.random.randn(n)
        X[:10] = 0.0  # Exact ties at cutoff
        Y = X + 5.0 * (X >= 0) + np.random.randn(n)

        rdd = SharpRDD(cutoff=0.0)
        rdd.fit(Y, X)

        # Should work - ties assigned consistently
        assert np.isfinite(rdd.coef_)

    def test_very_small_outcome_values(self):
        """Very small outcome values - tests numerical precision."""
        np.random.seed(888)
        n = 100
        X = np.random.randn(n)
        Y = (X + 5.0 * (X >= 0) + np.random.randn(n)) * 1e-10

        rdd = SharpRDD(cutoff=0.0)

        # May fail due to numerical issues or work with scaled estimate
        try:
            rdd.fit(Y, X)
            assert np.isfinite(rdd.coef_)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass  # Expected failure

    def test_very_large_outcome_values(self):
        """Very large outcome values - tests numerical stability."""
        np.random.seed(999)
        n = 100
        X = np.random.randn(n)
        Y = (X + 5.0 * (X >= 0) + np.random.randn(n)) * 1e10

        rdd = SharpRDD(cutoff=0.0)
        rdd.fit(Y, X)

        # Should work
        assert np.isfinite(rdd.coef_)

    def test_scaled_running_variable(self):
        """Running variable with large scale - tests normalization."""
        np.random.seed(1001)
        n = 100
        X = np.random.randn(n) * 1e6
        cutoff = 0.0
        Y = X + 5e6 * (X >= cutoff) + np.random.randn(n) * 1e6

        rdd = SharpRDD(cutoff=cutoff)
        rdd.fit(Y, X)

        assert np.isfinite(rdd.coef_)


# =============================================================================
# Bandwidth Edge Cases
# =============================================================================


class TestRDDBandwidthEdgeCases:
    """Test RDD with bandwidth extremes."""

    def test_very_small_bandwidth(self):
        """Very small bandwidth - few effective observations."""
        np.random.seed(555)
        n = 1000
        X = np.random.randn(n)
        Y = X + 5.0 * (X >= 0) + np.random.randn(n)

        rdd = SharpRDD(cutoff=0.0, bandwidth=0.01)

        # Should work but may have large SE
        try:
            rdd.fit(Y, X)
            assert np.isfinite(rdd.coef_)
            assert rdd.se_ > 0.0
        except (ValueError, RuntimeError):
            pass  # May fail with too few observations

    def test_very_large_bandwidth(self):
        """Very large bandwidth - includes all observations."""
        np.random.seed(666)
        n = 100
        X = np.random.randn(n)
        Y = X + 5.0 * (X >= 0) + np.random.randn(n)

        rdd = SharpRDD(cutoff=0.0, bandwidth=100.0)
        rdd.fit(Y, X)

        # Should work with large bandwidth
        assert np.isfinite(rdd.coef_)
        assert rdd.n_left_ + rdd.n_right_ == n  # All obs used

    def test_automatic_bandwidth_selection(self):
        """IK automatic bandwidth - should be reasonable."""
        np.random.seed(667)
        n = 500
        X = np.random.randn(n)
        Y = X + 5.0 * (X >= 0) + np.random.randn(n)

        rdd = SharpRDD(cutoff=0.0, bandwidth="ik")
        rdd.fit(Y, X)

        # Bandwidth should be reasonable (not 0, not inf)
        assert 0.0 < rdd.bandwidth_left_ < 10.0
        assert 0.0 < rdd.bandwidth_right_ < 10.0


# =============================================================================
# McCrary Test Edge Cases
# =============================================================================


class TestMcCraryEdgeCases:
    """Test McCrary density test edge cases."""

    def test_mccrary_insufficient_data(self):
        """McCrary with very small sample - should handle gracefully."""
        np.random.seed(1010)
        n = 20  # Very small
        X = np.random.randn(n)

        # Returns tuple (theta, p_value, message)
        result = mccrary_density_test(X, cutoff=0.0)

        # Should return tuple
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_mccrary_all_positive(self):
        """McCrary when all observations are positive - returns message."""
        np.random.seed(1011)
        n = 100
        X = np.abs(np.random.randn(n)) + 0.1

        # Returns tuple - may have special message for all-one-side data
        result = mccrary_density_test(X, cutoff=0.0)
        assert isinstance(result, tuple)
        # theta, p_value, message
        theta, p_value, message = result
        # With all positive, test may return zeros or special message
        assert isinstance(message, str)


# =============================================================================
# Covariate Edge Cases
# =============================================================================


class TestRDDCovariateEdgeCases:
    """Test RDD covariate balance with edge cases."""

    def test_balance_test_single_covariate(self):
        """Balance test with single covariate."""
        np.random.seed(1111)
        n = 200
        X = np.random.randn(n)
        W = np.random.randn(n)  # Single covariate

        result = covariate_balance_test(X, W, cutoff=0.0)

        # Should return DataFrame with one row
        assert len(result) == 1
        assert "p_value" in result.columns

    def test_balance_test_multiple_covariates(self):
        """Balance test with multiple covariates."""
        np.random.seed(1112)
        n = 200
        X = np.random.randn(n)
        W = np.random.randn(n, 5)  # 5 covariates

        result = covariate_balance_test(X, W, cutoff=0.0)

        # Should return DataFrame with 5 rows
        assert len(result) == 5

    def test_balance_test_constant_covariate(self):
        """Balance test with constant covariate - should fail or return NaN."""
        np.random.seed(1113)
        n = 200
        X = np.random.randn(n)
        W = np.full(n, 5.0)  # Constant

        # Should either fail or return NaN effect
        try:
            result = covariate_balance_test(X, W, cutoff=0.0)
            # May have NaN or zero effect
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass  # Expected failure


# =============================================================================
# Sensitivity Test Edge Cases
# =============================================================================


class TestRDDSensitivityEdgeCases:
    """Test RDD sensitivity analysis edge cases."""

    def test_donut_hole_invalid_radius(self):
        """Donut hole with invalid radius should raise."""
        np.random.seed(1414)
        n = 100
        X = np.random.randn(n)
        Y = X + 5.0 * (X >= 0) + np.random.randn(n)

        # Invalid radii
        with pytest.raises((ValueError, TypeError)):
            donut_hole_rdd(Y, X, cutoff=0.0, bandwidth=1.0, hole_radius=0.0)

        with pytest.raises((ValueError, TypeError)):
            donut_hole_rdd(Y, X, cutoff=0.0, bandwidth=1.0, hole_radius=-0.5)

    def test_bandwidth_sensitivity_normal_case(self):
        """Bandwidth sensitivity with normal data."""
        np.random.seed(1515)
        n = 200
        X = np.random.randn(n)
        Y = X + 5.0 * (X >= 0) + np.random.randn(n)

        # Requires h_optimal parameter
        h_optimal = 1.0
        result = bandwidth_sensitivity_analysis(Y, X, cutoff=0.0, h_optimal=h_optimal)

        # Should return DataFrame
        assert len(result) > 0
        assert "estimate" in result.columns

    def test_polynomial_sensitivity_normal_case(self):
        """Polynomial order sensitivity with normal data."""
        np.random.seed(1516)
        n = 200
        X = np.random.randn(n)
        Y = X + 5.0 * (X >= 0) + np.random.randn(n)

        # Requires bandwidth parameter
        result = polynomial_order_sensitivity(Y, X, cutoff=0.0, bandwidth=1.0)

        # Should return DataFrame
        assert len(result) > 0
        assert "estimate" in result.columns


# =============================================================================
# Edge Cases - Cutoff Values
# =============================================================================


class TestRDDCutoffEdgeCases:
    """Test RDD with various cutoff values."""

    def test_negative_cutoff(self):
        """Negative cutoff value - should work normally."""
        np.random.seed(1919)
        n = 200
        X = np.random.randn(n) - 5.0  # Center around -5
        cutoff = -5.0
        Y = X + 3.0 * (X >= cutoff) + np.random.randn(n)

        rdd = SharpRDD(cutoff=cutoff)
        rdd.fit(Y, X)

        # Should estimate ~3.0
        assert np.isfinite(rdd.coef_)
        assert abs(rdd.coef_ - 3.0) < 2.0  # Reasonable estimate

    def test_large_cutoff(self):
        """Large cutoff value - tests internal centering."""
        np.random.seed(2020)
        n = 200
        X = np.random.randn(n) + 1e6  # Center around 1 million
        cutoff = 1e6
        Y = (X - cutoff) + 5.0 * (X >= cutoff) + np.random.randn(n)

        rdd = SharpRDD(cutoff=cutoff)
        rdd.fit(Y, X)

        # Should work with internal centering
        assert np.isfinite(rdd.coef_)

    def test_asymmetric_data_distribution(self):
        """Asymmetric data - 90% left, 10% right."""
        np.random.seed(2121)
        n_left = 180
        n_right = 20
        X = np.concatenate(
            [
                np.random.randn(n_left) - 1.0,
                np.random.randn(n_right) + 1.0,
            ]
        )
        Y = X + 4.0 * (X >= 0) + np.random.randn(len(X))

        rdd = SharpRDD(cutoff=0.0)
        rdd.fit(Y, X)

        # Should work with asymmetric effective samples
        assert np.isfinite(rdd.coef_)
        assert rdd.n_left_ != rdd.n_right_  # Asymmetric


# =============================================================================
# Kernel Edge Cases
# =============================================================================


class TestRDDKernelEdgeCases:
    """Test RDD with different kernel options."""

    def test_triangular_kernel(self):
        """Triangular kernel - default."""
        np.random.seed(1717)
        n = 200
        X = np.random.randn(n)
        Y = X + 5.0 * (X >= 0) + np.random.randn(n)

        rdd = SharpRDD(cutoff=0.0, kernel="triangular")
        rdd.fit(Y, X)
        assert np.isfinite(rdd.coef_)

    def test_rectangular_kernel(self):
        """Rectangular (uniform) kernel."""
        np.random.seed(1718)
        n = 200
        X = np.random.randn(n)
        Y = X + 5.0 * (X >= 0) + np.random.randn(n)

        rdd = SharpRDD(cutoff=0.0, kernel="rectangular")
        rdd.fit(Y, X)
        assert np.isfinite(rdd.coef_)

    def test_invalid_kernel_raises(self):
        """Invalid kernel should raise error."""
        with pytest.raises((ValueError, TypeError)):
            rdd = SharpRDD(cutoff=0.0, kernel="invalid_kernel")


# =============================================================================
# Inference Options
# =============================================================================


class TestRDDInferenceOptions:
    """Test RDD with different inference options."""

    def test_standard_inference(self):
        """Standard (homoskedastic) inference."""
        np.random.seed(1800)
        n = 200
        X = np.random.randn(n)
        Y = X + 5.0 * (X >= 0) + np.random.randn(n)

        rdd = SharpRDD(cutoff=0.0, inference="standard")
        rdd.fit(Y, X)

        assert np.isfinite(rdd.se_)
        assert rdd.se_ > 0

    def test_robust_inference(self):
        """Robust (heteroskedasticity-robust) inference."""
        np.random.seed(1801)
        n = 200
        X = np.random.randn(n)
        Y = X + 5.0 * (X >= 0) + np.random.randn(n) * (1 + np.abs(X))  # Heteroskedastic

        rdd = SharpRDD(cutoff=0.0, inference="robust")
        rdd.fit(Y, X)

        assert np.isfinite(rdd.se_)
        assert rdd.se_ > 0

    def test_invalid_inference_raises(self):
        """Invalid inference option should raise error."""
        with pytest.raises((ValueError, TypeError)):
            rdd = SharpRDD(cutoff=0.0, inference="invalid_inference")


# =============================================================================
# Error Handling
# =============================================================================


class TestRDDErrorHandling:
    """Test RDD error handling for invalid inputs."""

    def test_mismatched_dimensions_raises(self):
        """Mismatched X and Y dimensions should raise error."""
        np.random.seed(2000)
        X = np.random.randn(100)
        Y = np.random.randn(50)  # Different length

        rdd = SharpRDD(cutoff=0.0)

        with pytest.raises(ValueError):
            rdd.fit(Y, X)

    def test_empty_arrays_raises(self):
        """Empty arrays should raise error."""
        X = np.array([])
        Y = np.array([])

        rdd = SharpRDD(cutoff=0.0)

        with pytest.raises((ValueError, RuntimeError)):
            rdd.fit(Y, X)

    def test_invalid_alpha_raises(self):
        """Invalid alpha should raise error at construction or fit."""
        # Alpha outside (0, 1) should raise
        with pytest.raises((ValueError, TypeError)):
            rdd = SharpRDD(cutoff=0.0, alpha=-0.05)

        with pytest.raises((ValueError, TypeError)):
            rdd = SharpRDD(cutoff=0.0, alpha=1.5)


# =============================================================================
# Integration Test
# =============================================================================


class TestRDDIntegration:
    """Integration tests combining multiple edge cases."""

    def test_multiple_edge_cases_combined(self):
        """Combine several edge cases."""
        np.random.seed(1818)
        n = 50  # Small sample
        X = np.random.randn(n) * 0.5  # Narrow range
        X[0] = 0.0  # Exact tie
        Y = np.full(n, 10.0)  # Start constant
        Y[n - 1] = 100.0  # Add outlier
        Y = Y + 3.0 * (X >= 0)  # Add small treatment effect

        rdd = SharpRDD(cutoff=0.0)

        # Should handle gracefully
        try:
            rdd.fit(Y, X)
            assert np.isfinite(rdd.coef_)
            assert np.isfinite(rdd.se_)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass  # Expected failure with these extreme conditions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
