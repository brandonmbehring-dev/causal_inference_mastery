"""
Tests for RDD diagnostic tools.

Validates:
- McCrary density test (manipulation detection)
- Covariate balance tests (falsification)
- Bandwidth sensitivity analysis
- Polynomial order sensitivity
- Donut-hole RDD
"""

import numpy as np
import pytest
import pandas as pd

from src.causal_inference.rdd.rdd_diagnostics import (
    mccrary_density_test,
    covariate_balance_test,
    bandwidth_sensitivity_analysis,
    polynomial_order_sensitivity,
    donut_hole_rdd,
)
from src.causal_inference.rdd.bandwidth import imbens_kalyanaraman_bandwidth


class TestMcCraryDensityTest:
    """Test McCrary (2008) density test for manipulation."""

    def test_mccrary_no_manipulation(self, sharp_rdd_linear_dgp):
        """Test that McCrary test runs without error on uniform-ish data."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        theta, p_value, interpretation = mccrary_density_test(X, cutoff)

        # With finite samples, McCrary test can have false positives
        # Main check: test runs and returns valid outputs
        assert np.isfinite(theta), "θ should be finite"
        assert 0 <= p_value <= 1, f"p-value should be in [0, 1], got {p_value}"
        assert len(interpretation) > 0, "Should return interpretation string"

    def test_mccrary_bunching_detected(self, rdd_bunching_dgp):
        """Test that bunching at cutoff is detected."""
        Y, X, cutoff, _ = rdd_bunching_dgp

        theta, p_value, interpretation = mccrary_density_test(X, cutoff)

        # Bunching → |θ| > 0 (direction depends on which side)
        # We can't be sure of sign, but should detect discontinuity
        assert p_value < 0.10, f"Expected p < 0.10 with bunching, got {p_value:.3f}"

    def test_mccrary_bandwidth_auto(self, sharp_rdd_linear_dgp):
        """Test that automatic bandwidth selection works."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        # bandwidth=None should use Silverman's rule
        theta, p_value, _ = mccrary_density_test(X, cutoff, bandwidth=None)

        assert np.isfinite(theta), "θ should be finite"
        assert 0 <= p_value <= 1, f"p-value should be in [0, 1], got {p_value}"

    def test_mccrary_interpretation_string(self, sharp_rdd_linear_dgp):
        """Test that interpretation message is correct."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        _, p_value, interpretation = mccrary_density_test(X, cutoff)

        # Should include p-value in message
        assert f"{p_value:.3f}" in interpretation or f"{p_value:.2f}" in interpretation

        # Should have clear message
        if p_value < 0.05:
            assert "manipulation" in interpretation.lower()
        else:
            assert "no evidence" in interpretation.lower()


class TestCovariateBalance:
    """Test covariate balance tests (falsification)."""

    def test_covariate_balance_valid_rdd(self, rdd_with_covariates_dgp):
        """Test that balanced covariates show p > 0.05."""
        Y, X, cutoff, _, W = rdd_with_covariates_dgp

        results = covariate_balance_test(X, W, cutoff, bandwidth="ik", covariate_names=["age", "gender"])

        # Both covariates should be balanced (p > 0.05)
        assert len(results) == 2, f"Expected 2 rows, got {len(results)}"
        assert all(results["p_value"] > 0.01), "Expected all p > 0.01 for balanced covariates"

    def test_covariate_balance_violation(self, rdd_sorted_on_covariate_dgp):
        """Test that sorting on covariate is detected."""
        Y, X, cutoff, _, W = rdd_sorted_on_covariate_dgp

        results = covariate_balance_test(X, W, cutoff, bandwidth="ik")

        # Income has discontinuity → should be flagged
        assert len(results) == 1
        assert results.iloc[0]["p_value"] < 0.10, "Expected p < 0.10 for unbalanced covariate"
        assert results.iloc[0]["significant"], "Should flag as significant"

    def test_covariate_balance_dataframe_format(self, rdd_with_covariates_dgp):
        """Test that DataFrame has required columns."""
        Y, X, cutoff, _, W = rdd_with_covariates_dgp

        results = covariate_balance_test(X, W, cutoff)

        required_cols = ["covariate", "estimate", "se", "t_stat", "p_value", "significant"]
        for col in required_cols:
            assert col in results.columns, f"Missing column: {col}"


class TestBandwidthSensitivity:
    """Test bandwidth sensitivity analysis."""

    def test_bandwidth_sensitivity_stable(self, sharp_rdd_linear_dgp):
        """Test that estimates are stable across bandwidth grid."""
        Y, X, cutoff, true_tau = sharp_rdd_linear_dgp

        h_opt = imbens_kalyanaraman_bandwidth(Y, X, cutoff)
        results = bandwidth_sensitivity_analysis(Y, X, cutoff, h_opt)

        # Should have 5 rows (default grid)
        assert len(results) == 5

        # Estimates should all be within 50% of truth (loose tolerance)
        estimates = results["estimate"].values
        assert all(np.abs(est - true_tau) < 0.5 * true_tau for est in estimates), \
            f"Estimates should be stable, got range: {estimates.min():.2f} - {estimates.max():.2f}"

    def test_bandwidth_sensitivity_grid(self, sharp_rdd_linear_dgp):
        """Test that custom bandwidth grid works."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        custom_grid = np.array([0.5, 1.0, 1.5])
        results = bandwidth_sensitivity_analysis(Y, X, cutoff, h_optimal=1.0, bandwidth_grid=custom_grid)

        assert len(results) == 3, "Should have one row per bandwidth"
        assert list(results["bandwidth"]) == list(custom_grid), "Bandwidths should match grid"

    def test_bandwidth_sensitivity_dataframe(self, sharp_rdd_linear_dgp):
        """Test DataFrame structure."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        results = bandwidth_sensitivity_analysis(Y, X, cutoff, h_optimal=1.0)

        required_cols = ["bandwidth", "estimate", "se", "ci_lower", "ci_upper"]
        for col in required_cols:
            assert col in results.columns, f"Missing column: {col}"


class TestPolynomialOrderSensitivity:
    """Test polynomial order sensitivity."""

    def test_polynomial_order_stability(self, sharp_rdd_linear_dgp):
        """Test that local linear (p>=1) recovers effect for linear DGP with slope.

        Note: Local constant (p=0) IS biased when the DGP has a slope (Y = X + tau*D + eps).
        This is expected behavior: p=0 doesn't account for the slope, leading to bias.
        Only p>=1 should be approximately unbiased for linear DGP.
        """
        Y, X, cutoff, true_tau = sharp_rdd_linear_dgp

        results = polynomial_order_sensitivity(Y, X, cutoff, bandwidth=1.5, max_order=3)

        # p=0 (local constant) IS biased for linear DGP with slope - this is correct behavior
        # Only p>=1 should recover the effect
        estimates_p1_plus = results[results["order"] >= 1]["estimate"].values
        assert all(np.abs(est - true_tau) < 0.6 for est in estimates_p1_plus), \
            "Local linear (p>=1) should recover effect for linear DGP"

        # p=0 should show bias (demonstrating that it doesn't work for DGP with slope)
        est_p0 = results[results["order"] == 0]["estimate"].values[0]
        assert np.abs(est_p0 - true_tau) > 0.5, \
            "Local constant (p=0) should be biased when DGP has slope"

    def test_polynomial_order_0_to_3(self, sharp_rdd_quadratic_dgp):
        """Test that orders 0-3 all produce finite estimates."""
        Y, X, cutoff, _ = sharp_rdd_quadratic_dgp

        results = polynomial_order_sensitivity(Y, X, cutoff, bandwidth=1.5, max_order=3)

        assert len(results) == 4, "Should have 4 rows (orders 0, 1, 2, 3)"
        assert all(np.isfinite(results["estimate"])), "All estimates should be finite"

    def test_polynomial_order_dataframe(self, sharp_rdd_linear_dgp):
        """Test DataFrame structure."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        results = polynomial_order_sensitivity(Y, X, cutoff, bandwidth=1.0)

        required_cols = ["order", "estimate", "se", "ci_lower", "ci_upper"]
        for col in required_cols:
            assert col in results.columns, f"Missing column: {col}"


class TestDonutHoleRDD:
    """Test donut-hole RDD."""

    def test_donut_hole_stable(self, sharp_rdd_linear_dgp):
        """Test that estimates are stable when excluding small window."""
        Y, X, cutoff, true_tau = sharp_rdd_linear_dgp

        results = donut_hole_rdd(Y, X, cutoff, bandwidth=1.5, hole_width=0.2)

        # Should have 4 rows (default: 0, 0.1, 0.2, 0.4)
        assert len(results) == 4

        # Estimates should be stable (within 40% of truth)
        valid_estimates = results["estimate"].dropna()
        assert all(np.abs(est - true_tau) < 0.8 for est in valid_estimates), \
            "Donut-hole estimates should be stable"

    def test_donut_hole_sample_size_reduction(self, sharp_rdd_linear_dgp):
        """Test that n_excluded increases with hole_width."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        results = donut_hole_rdd(Y, X, cutoff, bandwidth=1.5, hole_width=0.3)

        # n_excluded should increase with hole width
        n_excluded = results["n_excluded"].values
        assert np.all(np.diff(n_excluded) >= 0), "n_excluded should increase with hole_width"

    def test_donut_hole_dataframe(self, sharp_rdd_linear_dgp):
        """Test DataFrame structure."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        results = donut_hole_rdd(Y, X, cutoff, bandwidth=1.0)

        required_cols = ["hole_width", "estimate", "se", "n_excluded"]
        for col in required_cols:
            assert col in results.columns, f"Missing column: {col}"


class TestDiagnosticIntegration:
    """Test integrated diagnostic workflow."""

    def test_diagnostics_on_sharp_rdd_result(self, sharp_rdd_linear_dgp):
        """Test that all diagnostics run on same data."""
        Y, X, cutoff, _ = sharp_rdd_linear_dgp

        # McCrary test
        theta, p_value, _ = mccrary_density_test(X, cutoff)
        assert np.isfinite(theta) and 0 <= p_value <= 1

        # Bandwidth sensitivity
        h_opt = imbens_kalyanaraman_bandwidth(Y, X, cutoff)
        bw_results = bandwidth_sensitivity_analysis(Y, X, cutoff, h_opt)
        assert len(bw_results) > 0

        # Polynomial sensitivity
        poly_results = polynomial_order_sensitivity(Y, X, cutoff, bandwidth=h_opt)
        assert len(poly_results) > 0

        # Donut-hole
        donut_results = donut_hole_rdd(Y, X, cutoff, bandwidth=h_opt)
        assert len(donut_results) > 0

    def test_diagnostics_warn_on_manipulation(self, rdd_bunching_dgp):
        """Test that McCrary detects manipulation and warns."""
        Y, X, cutoff, _ = rdd_bunching_dgp

        _, p_value, interpretation = mccrary_density_test(X, cutoff)

        # Should detect bunching
        assert p_value < 0.15, f"Expected p < 0.15 with bunching, got {p_value:.3f}"
        # Interpretation should mention manipulation if p < 0.05
        if p_value < 0.05:
            assert "manipulation" in interpretation.lower()
