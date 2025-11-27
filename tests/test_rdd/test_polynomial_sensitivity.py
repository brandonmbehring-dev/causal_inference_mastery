"""TDD Step 1: Polynomial order sensitivity tests.

These tests verify that polynomial_order_sensitivity() actually fits
different polynomial orders, not just p=1 for all.

Written BEFORE fixes per TDD protocol - should expose the stub issue.
"""

import numpy as np
import pytest

from causal_inference.rdd import polynomial_order_sensitivity


class TestPolynomialOrderSensitivity:
    """Tests for polynomial_order_sensitivity function."""

    @pytest.fixture
    def quadratic_dgp(self):
        """DGP with curvature where p=0 should differ from p=1,2."""
        np.random.seed(42)
        n = 1000

        # Running variable centered at cutoff
        X = np.random.uniform(-2, 2, n)

        # Outcome with strong curvature + treatment effect at cutoff
        # Y = X^2 + tau * I(X >= 0) + noise
        tau = 2.0  # True treatment effect
        Y = X**2 + tau * (X >= 0) + np.random.normal(0, 0.5, n)

        return X, Y, tau

    @pytest.fixture
    def linear_dgp(self):
        """DGP that is truly linear - all orders should agree."""
        np.random.seed(42)
        n = 1000

        X = np.random.uniform(-2, 2, n)

        # Linear DGP: Y = X + tau * I(X >= 0) + noise
        tau = 2.0
        Y = X + tau * (X >= 0) + np.random.normal(0, 0.5, n)

        return X, Y, tau

    def test_returns_all_polynomial_orders(self, linear_dgp):
        """Should return results for orders 0, 1, 2, 3."""
        X, Y, _ = linear_dgp

        results = polynomial_order_sensitivity(Y, X, cutoff=0.0, bandwidth=1.0, max_order=3)

        assert len(results) == 4, f"Expected 4 rows, got {len(results)}"
        assert list(results["order"]) == [0, 1, 2, 3]

    def test_order_0_differs_from_order_1_with_curvature(self, quadratic_dgp):
        """Local constant (p=0) should be biased when there's curvature."""
        X, Y, tau = quadratic_dgp

        results = polynomial_order_sensitivity(Y, X, cutoff=0.0, bandwidth=1.0, max_order=2)

        est_p0 = results[results["order"] == 0]["estimate"].values[0]
        est_p1 = results[results["order"] == 1]["estimate"].values[0]

        # With curvature, p=0 should show noticeable bias (estimate further from tau)
        # Local linear (p=1) should be closer to true effect
        bias_p0 = abs(est_p0 - tau)
        bias_p1 = abs(est_p1 - tau)

        # The key test: p=0 and p=1 should give DIFFERENT estimates
        # if polynomial fitting is actually working.
        # Threshold of 0.01 ensures estimates are numerically different
        # (larger differences require stronger curvature or wider bandwidth)
        assert abs(est_p0 - est_p1) > 0.01, (
            f"p=0 ({est_p0:.3f}) and p=1 ({est_p1:.3f}) should differ with curvature. "
            f"This suggests polynomial fitting is not actually implemented."
        )

    def test_estimates_change_with_polynomial_order(self, quadratic_dgp):
        """Different polynomial orders should produce measurably different estimates."""
        X, Y, tau = quadratic_dgp

        results = polynomial_order_sensitivity(Y, X, cutoff=0.0, bandwidth=1.0, max_order=3)

        estimates = results["estimate"].values

        # At least some orders should produce different estimates
        # If all estimates are identical, the function is a stub
        unique_estimates = np.unique(np.round(estimates, 4))

        assert len(unique_estimates) > 1, (
            f"All polynomial orders produced identical estimates: {estimates}. "
            f"This indicates polynomial_order_sensitivity is a stub that doesn't "
            f"actually fit different polynomial orders."
        )

    def test_local_constant_is_naive_difference_in_means(self, linear_dgp):
        """Order p=0 (local constant) should approximate difference in means near cutoff."""
        X, Y, tau = linear_dgp

        bandwidth = 0.5  # Narrow bandwidth
        results = polynomial_order_sensitivity(Y, X, cutoff=0.0, bandwidth=bandwidth, max_order=0)

        est_p0 = results[results["order"] == 0]["estimate"].values[0]

        # Manual calculation: difference in means within bandwidth
        in_band = np.abs(X) <= bandwidth
        above = X >= 0
        manual_est = np.mean(Y[in_band & above]) - np.mean(Y[in_band & ~above])

        # p=0 estimate should be close to manual difference in means
        # (not exactly equal due to kernel weighting)
        assert np.isclose(est_p0, manual_est, rtol=0.3), (
            f"p=0 estimate ({est_p0:.3f}) should approximate "
            f"difference in means ({manual_est:.3f})"
        )

    def test_local_linear_matches_sharprdd(self, linear_dgp):
        """Order p=1 should match SharpRDD (which uses local linear)."""
        from causal_inference.rdd import SharpRDD

        X, Y, tau = linear_dgp
        bandwidth = 1.0

        # Get p=1 estimate from polynomial sensitivity
        results = polynomial_order_sensitivity(Y, X, cutoff=0.0, bandwidth=bandwidth, max_order=1)
        est_p1 = results[results["order"] == 1]["estimate"].values[0]

        # Get SharpRDD estimate (local linear)
        rdd = SharpRDD(cutoff=0.0, bandwidth=bandwidth, inference="robust")
        rdd.fit(Y, X)
        est_rdd = rdd.coef_

        # These should match closely
        assert np.isclose(est_p1, est_rdd, rtol=0.01), (
            f"p=1 ({est_p1:.4f}) should match SharpRDD ({est_rdd:.4f})"
        )

    @pytest.mark.parametrize("poly_order", [0, 1, 2, 3])
    def test_all_orders_produce_finite_estimates(self, linear_dgp, poly_order):
        """All polynomial orders should produce finite estimates and SEs."""
        X, Y, _ = linear_dgp

        results = polynomial_order_sensitivity(
            Y, X, cutoff=0.0, bandwidth=1.0, max_order=poly_order
        )

        row = results[results["order"] == poly_order].iloc[0]

        assert np.isfinite(row["estimate"]), f"p={poly_order} estimate is not finite"
        assert np.isfinite(row["se"]), f"p={poly_order} SE is not finite"
        assert row["se"] > 0, f"p={poly_order} SE should be positive"
        assert row["ci_lower"] < row["ci_upper"], "CI lower should be < upper"


class TestPolynomialSensitivityStatisticalProperties:
    """Statistical validation of polynomial order estimates."""

    @pytest.mark.slow
    def test_local_linear_unbiased_for_linear_dgp(self):
        """p=1 should be unbiased when DGP is linear (Monte Carlo)."""
        np.random.seed(42)
        n_sims = 500
        tau = 2.0
        estimates = []

        for seed in range(n_sims):
            rng = np.random.default_rng(seed)
            n = 500
            X = rng.uniform(-2, 2, n)
            Y = X + tau * (X >= 0) + rng.normal(0, 0.5, n)

            results = polynomial_order_sensitivity(Y, X, cutoff=0.0, bandwidth=1.0, max_order=1)
            est_p1 = results[results["order"] == 1]["estimate"].values[0]
            estimates.append(est_p1)

        bias = np.mean(estimates) - tau
        assert abs(bias) < 0.1, f"p=1 bias {bias:.4f} exceeds 0.1 for linear DGP"

    @pytest.mark.slow
    def test_local_constant_biased_with_slope(self):
        """p=0 should be biased when there's a slope at cutoff."""
        np.random.seed(42)
        n_sims = 500
        tau = 2.0
        slope = 1.0  # Nonzero slope
        estimates = []

        for seed in range(n_sims):
            rng = np.random.default_rng(seed)
            n = 500
            X = rng.uniform(-2, 2, n)
            Y = slope * X + tau * (X >= 0) + rng.normal(0, 0.5, n)

            results = polynomial_order_sensitivity(Y, X, cutoff=0.0, bandwidth=1.0, max_order=0)
            est_p0 = results[results["order"] == 0]["estimate"].values[0]
            estimates.append(est_p0)

        bias = np.mean(estimates) - tau
        # p=0 should show bias due to slope
        assert abs(bias) > 0.05, (
            f"p=0 should be biased with slope, but bias={bias:.4f}"
        )
