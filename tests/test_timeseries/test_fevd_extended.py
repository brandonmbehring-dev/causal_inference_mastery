"""
Tests for Extended FEVD Functions (Bootstrap Inference).

Session 146: Tests for bootstrap FEVD confidence intervals.

Test layers:
- Layer 1: Known-answer tests (structure, basic properties)
- Layer 2: Adversarial tests (edge cases)
- Layer 3: Monte Carlo validation (coverage) @slow
"""

import numpy as np
import pytest

from causal_inference.timeseries.fevd import (
    bootstrap_fevd,
    compute_fevd,
)
from causal_inference.timeseries.svar_types import FEVDBootstrapResult
from causal_inference.timeseries.svar import cholesky_svar
from causal_inference.timeseries.var import var_estimate


# ============================================================================
# Helper Functions
# ============================================================================


def generate_var1_data(n=200, seed=42):
    """Generate data from stable VAR(1) process."""
    np.random.seed(seed)
    k = 2
    A1 = np.array([[0.5, 0.1], [0.2, 0.4]])
    data = np.zeros((n, k))

    for t in range(1, n):
        data[t, :] = A1 @ data[t - 1, :] + np.random.randn(k) * 0.5

    return data


def generate_var2_data(n=200, seed=42):
    """Generate data from stable VAR(2) process."""
    np.random.seed(seed)
    k = 3
    A1 = np.array([[0.4, 0.1, 0.05], [0.1, 0.3, 0.1], [0.05, 0.1, 0.35]])
    A2 = np.array([[0.2, 0.05, 0.02], [0.05, 0.15, 0.05], [0.02, 0.05, 0.2]])

    data = np.zeros((n, k))

    for t in range(2, n):
        data[t, :] = A1 @ data[t - 1, :] + A2 @ data[t - 2, :] + np.random.randn(k) * 0.5

    return data


# ============================================================================
# Layer 1: Known-Answer Tests - Bootstrap FEVD
# ============================================================================


class TestBootstrapFEVDKnownAnswer:
    """Known-answer tests for bootstrap FEVD."""

    def test_bootstrap_fevd_returns_correct_type(self, seed):
        """Bootstrap FEVD should return FEVDBootstrapResult."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        fevd_ci = bootstrap_fevd(data, svar_result, horizons=10, n_bootstrap=50, seed=seed)

        assert isinstance(fevd_ci, FEVDBootstrapResult)

    def test_bootstrap_fevd_has_correct_structure(self, seed):
        """Bootstrap FEVD should have correct array shapes."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        fevd_ci = bootstrap_fevd(data, svar_result, horizons=10, n_bootstrap=50, seed=seed)

        assert fevd_ci.fevd.shape == (2, 2, 11)
        assert fevd_ci.fevd_lower.shape == (2, 2, 11)
        assert fevd_ci.fevd_upper.shape == (2, 2, 11)

    def test_bootstrap_fevd_has_confidence_bands(self, seed):
        """Bootstrap FEVD should have confidence bands."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        fevd_ci = bootstrap_fevd(data, svar_result, horizons=10, n_bootstrap=50, seed=seed)

        assert fevd_ci.has_confidence_bands
        # Lower should be <= upper
        assert np.all(fevd_ci.fevd_lower <= fevd_ci.fevd_upper)

    def test_bootstrap_fevd_bounds_valid(self, seed):
        """Bootstrap FEVD bounds should be in [0, 1]."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        fevd_ci = bootstrap_fevd(data, svar_result, horizons=10, n_bootstrap=50, seed=seed)

        # All values should be between 0 and 1
        assert np.all(fevd_ci.fevd >= 0)
        assert np.all(fevd_ci.fevd <= 1)
        assert np.all(fevd_ci.fevd_lower >= 0)
        assert np.all(fevd_ci.fevd_upper <= 1)

    def test_bootstrap_fevd_rows_sum_to_one(self, seed):
        """Bootstrap FEVD point estimates should have rows summing to 1."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        fevd_ci = bootstrap_fevd(data, svar_result, horizons=10, n_bootstrap=50, seed=seed)

        assert fevd_ci.validate_rows_sum_to_one()

    def test_bootstrap_fevd_point_matches_compute_fevd(self, seed):
        """Bootstrap FEVD point estimate should match compute_fevd."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        fevd_point = compute_fevd(svar_result, horizons=10)
        fevd_ci = bootstrap_fevd(data, svar_result, horizons=10, n_bootstrap=50, seed=seed)

        np.testing.assert_allclose(fevd_ci.fevd, fevd_point.fevd, rtol=1e-10)

    def test_bootstrap_fevd_ci_contains_point(self, seed):
        """Bootstrap FEVD CI should contain point estimate."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        fevd_ci = bootstrap_fevd(data, svar_result, horizons=10, n_bootstrap=50, seed=seed)

        # Point estimate should be within CI (or very close due to bootstrap variability)
        # Using relaxed check since bootstrap median may differ slightly from point estimate
        within_ci = (fevd_ci.fevd_lower <= fevd_ci.fevd) & (fevd_ci.fevd <= fevd_ci.fevd_upper)
        assert np.mean(within_ci) > 0.95  # At least 95% should be within CI


# ============================================================================
# Layer 1: Known-Answer Tests - Bootstrap Methods
# ============================================================================


class TestBootstrapFEVDMethods:
    """Tests for different bootstrap methods in FEVD."""

    def test_residual_bootstrap(self, seed):
        """Residual bootstrap should work correctly."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        fevd_ci = bootstrap_fevd(
            data, svar_result, horizons=10, n_bootstrap=50, method="residual", seed=seed
        )

        assert fevd_ci.method == "residual"
        assert fevd_ci.has_confidence_bands

    def test_wild_bootstrap(self, seed):
        """Wild bootstrap should work correctly."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        fevd_ci = bootstrap_fevd(
            data, svar_result, horizons=10, n_bootstrap=50, method="wild", seed=seed
        )

        assert fevd_ci.method == "wild"
        assert fevd_ci.has_confidence_bands

    def test_block_bootstrap(self, seed):
        """Block bootstrap should work correctly."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        fevd_ci = bootstrap_fevd(
            data, svar_result, horizons=10, n_bootstrap=50, method="block", seed=seed
        )

        assert fevd_ci.method == "block"
        assert fevd_ci.has_confidence_bands


# ============================================================================
# Layer 1: Known-Answer Tests - Result Methods
# ============================================================================


class TestFEVDBootstrapResultMethods:
    """Tests for FEVDBootstrapResult methods."""

    def test_get_decomposition_with_ci(self, seed):
        """get_decomposition_with_ci should return correct dict."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        fevd_ci = bootstrap_fevd(data, svar_result, horizons=10, n_bootstrap=50, seed=seed)

        result = fevd_ci.get_decomposition_with_ci(0, horizon=5)

        assert "fevd" in result
        assert "lower" in result
        assert "upper" in result
        assert result["fevd"].shape == (2,)  # n_vars shocks
        assert np.sum(result["fevd"]) == pytest.approx(1.0, abs=1e-10)

    def test_get_contribution_with_ci(self, seed):
        """get_contribution_with_ci should return correct dict."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        fevd_ci = bootstrap_fevd(data, svar_result, horizons=10, n_bootstrap=50, seed=seed)

        result = fevd_ci.get_contribution_with_ci(response_var=0, shock_var=1)

        assert "fevd" in result
        assert "lower" in result
        assert "upper" in result
        assert "horizon" in result
        assert result["fevd"].shape == (11,)  # horizons + 1


# ============================================================================
# Layer 2: Adversarial Tests
# ============================================================================


class TestBootstrapFEVDAdversarial:
    """Adversarial tests for bootstrap FEVD."""

    def test_short_series(self, seed):
        """Bootstrap FEVD should handle short series."""
        np.random.seed(seed)
        data = generate_var1_data(n=50, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        fevd_ci = bootstrap_fevd(data, svar_result, horizons=5, n_bootstrap=30, seed=seed)

        assert fevd_ci.has_confidence_bands

    def test_var2(self, seed):
        """Bootstrap FEVD should work with VAR(2)."""
        np.random.seed(seed)
        data = generate_var2_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=2)
        svar_result = cholesky_svar(var_result)

        fevd_ci = bootstrap_fevd(data, svar_result, horizons=10, n_bootstrap=50, seed=seed)

        assert fevd_ci.fevd.shape == (3, 3, 11)

    def test_invalid_n_bootstrap(self, seed):
        """Should reject n_bootstrap < 2."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        with pytest.raises(ValueError, match="n_bootstrap"):
            bootstrap_fevd(data, svar_result, horizons=10, n_bootstrap=1, seed=seed)

    def test_invalid_alpha(self, seed):
        """Should reject invalid alpha."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        with pytest.raises(ValueError, match="alpha"):
            bootstrap_fevd(data, svar_result, horizons=10, n_bootstrap=50, alpha=0, seed=seed)

        with pytest.raises(ValueError, match="alpha"):
            bootstrap_fevd(data, svar_result, horizons=10, n_bootstrap=50, alpha=1, seed=seed)

    def test_invalid_method(self, seed):
        """Should reject invalid method."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        with pytest.raises(ValueError, match="method must be"):
            bootstrap_fevd(
                data, svar_result, horizons=10, n_bootstrap=50, method="invalid", seed=seed
            )


# ============================================================================
# Layer 3: Monte Carlo Tests
# ============================================================================


class TestBootstrapFEVDMonteCarlo:
    """Monte Carlo validation for bootstrap FEVD."""

    @pytest.mark.slow
    def test_coverage(self):
        """Bootstrap FEVD CI should have approximately correct coverage."""
        n_runs = 100
        alpha = 0.10  # 90% CI
        n_obs = 200
        horizons = 10
        target_horizon = 5
        target_response = 0
        target_shock = 0

        coverage_count = 0

        for run in range(n_runs):
            np.random.seed(run)

            # Generate data
            A1 = np.array([[0.5, 0.1], [0.2, 0.4]])
            k = 2
            data = np.zeros((n_obs, k))
            for t in range(1, n_obs):
                data[t, :] = A1 @ data[t - 1, :] + np.random.randn(k) * 0.5

            var_result = var_estimate(data, lags=1)
            svar_result = cholesky_svar(var_result)

            # Point estimate as "truth"
            true_fevd = compute_fevd(svar_result, horizons=horizons)
            true_value = true_fevd.fevd[target_response, target_shock, target_horizon]

            # Bootstrap CI
            fevd_ci = bootstrap_fevd(
                data,
                svar_result,
                horizons=horizons,
                n_bootstrap=200,
                alpha=alpha,
                seed=run,
            )

            lower = fevd_ci.fevd_lower[target_response, target_shock, target_horizon]
            upper = fevd_ci.fevd_upper[target_response, target_shock, target_horizon]

            if lower <= true_value <= upper:
                coverage_count += 1

        coverage = coverage_count / n_runs

        # Coverage should be approximately 1 - alpha (allow 80-99%)
        assert 0.80 < coverage <= 0.99, f"FEVD coverage {coverage:.2%} outside bounds"

    @pytest.mark.slow
    def test_wild_vs_residual_coverage(self):
        """Wild and residual bootstrap should give similar coverage."""
        n_runs = 50
        alpha = 0.10
        n_obs = 200
        horizons = 10

        residual_coverage = 0
        wild_coverage = 0

        for run in range(n_runs):
            np.random.seed(run)

            # Generate data
            A1 = np.array([[0.5, 0.1], [0.2, 0.4]])
            k = 2
            data = np.zeros((n_obs, k))
            for t in range(1, n_obs):
                data[t, :] = A1 @ data[t - 1, :] + np.random.randn(k) * 0.5

            var_result = var_estimate(data, lags=1)
            svar_result = cholesky_svar(var_result)

            true_fevd = compute_fevd(svar_result, horizons=horizons)
            true_value = true_fevd.fevd[0, 0, 5]

            # Residual bootstrap
            fevd_resid = bootstrap_fevd(
                data,
                svar_result,
                horizons=horizons,
                n_bootstrap=100,
                alpha=alpha,
                method="residual",
                seed=run,
            )
            if fevd_resid.fevd_lower[0, 0, 5] <= true_value <= fevd_resid.fevd_upper[0, 0, 5]:
                residual_coverage += 1

            # Wild bootstrap
            fevd_wild = bootstrap_fevd(
                data,
                svar_result,
                horizons=horizons,
                n_bootstrap=100,
                alpha=alpha,
                method="wild",
                seed=run,
            )
            if fevd_wild.fevd_lower[0, 0, 5] <= true_value <= fevd_wild.fevd_upper[0, 0, 5]:
                wild_coverage += 1

        resid_cov = residual_coverage / n_runs
        wild_cov = wild_coverage / n_runs

        # Both should have reasonable coverage
        assert resid_cov > 0.75, f"Residual coverage {resid_cov:.2%} too low"
        assert wild_cov > 0.75, f"Wild coverage {wild_cov:.2%} too low"

        # Difference should not be too large
        assert abs(resid_cov - wild_cov) < 0.25, (
            f"Coverage difference {abs(resid_cov - wild_cov):.2%} too large"
        )

    @pytest.mark.slow
    def test_ci_width_decreases_with_sample_size(self):
        """Bootstrap CI width should decrease with larger samples."""
        horizons = 10
        n_bootstrap = 100
        alpha = 0.05

        widths = {}

        for n_obs in [100, 200, 400]:
            np.random.seed(42)

            A1 = np.array([[0.5, 0.1], [0.2, 0.4]])
            k = 2
            data = np.zeros((n_obs, k))
            for t in range(1, n_obs):
                data[t, :] = A1 @ data[t - 1, :] + np.random.randn(k) * 0.5

            var_result = var_estimate(data, lags=1)
            svar_result = cholesky_svar(var_result)

            fevd_ci = bootstrap_fevd(
                data, svar_result, horizons=horizons, n_bootstrap=n_bootstrap, alpha=alpha, seed=42
            )

            # Average CI width
            width = np.mean(fevd_ci.fevd_upper - fevd_ci.fevd_lower)
            widths[n_obs] = width

        # Width should generally decrease with sample size
        assert widths[400] < widths[100], (
            f"Width at n=400 ({widths[400]:.4f}) not smaller than n=100 ({widths[100]:.4f})"
        )
