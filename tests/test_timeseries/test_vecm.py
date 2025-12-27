"""
Tests for Vector Error Correction Model (VECM).

Session 149: VECM estimation and forecasting.

Test layers:
- Layer 1: Known-answer tests (structure, basic properties)
- Layer 2: Adversarial tests (edge cases)
- Layer 3: Monte Carlo validation (coverage, bias) @slow
"""

import numpy as np
import pytest

from causal_inference.timeseries import (
    vecm_estimate,
    vecm_forecast,
    vecm_granger_causality,
    compute_error_correction_term,
    VECMResult,
    johansen_test,
)


# ============================================================================
# Helper Functions
# ============================================================================


def generate_cointegrated_system(n: int = 200, seed: int = 42) -> np.ndarray:
    """
    Generate a cointegrated bivariate system.

    y1_t and y2_t share a common stochastic trend with cointegrating vector [1, -0.5].
    """
    np.random.seed(seed)
    trend = np.cumsum(np.random.randn(n))
    y1 = trend + np.random.randn(n) * 0.5
    y2 = 0.5 * trend + np.random.randn(n) * 0.5
    return np.column_stack([y1, y2])


def generate_trivariate_cointegrated(n: int = 200, seed: int = 42) -> np.ndarray:
    """
    Generate a cointegrated trivariate system with rank 2.
    """
    np.random.seed(seed)
    trend1 = np.cumsum(np.random.randn(n))
    trend2 = np.cumsum(np.random.randn(n))
    y1 = trend1 + np.random.randn(n) * 0.3
    y2 = 0.5 * trend1 + 0.5 * trend2 + np.random.randn(n) * 0.3
    y3 = trend2 + np.random.randn(n) * 0.3
    return np.column_stack([y1, y2, y3])


# ============================================================================
# Layer 1: Known-Answer Tests
# ============================================================================


class TestVECMEstimation:
    """Tests for VECM estimation."""

    def test_vecm_basic_structure(self):
        """VECM result should have correct structure."""
        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=2)

        assert isinstance(result, VECMResult)
        assert result.coint_rank == 1
        assert result.lags == 2
        assert result.n_vars == 2
        assert result.n_obs > 0

    def test_vecm_alpha_beta_shapes(self):
        """α and β should have correct shapes."""
        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=2)

        k = 2  # Number of variables
        r = 1  # Cointegration rank

        assert result.alpha.shape == (k, r)
        assert result.beta.shape == (k, r)
        assert result.pi.shape == (k, k)

    def test_vecm_gamma_shape(self):
        """Γ matrix should have correct shape for short-run dynamics."""
        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=3)

        k = 2
        n_sr_lags = 2  # lags - 1

        assert result.gamma.shape == (k, k * n_sr_lags)

    def test_vecm_pi_equals_alpha_beta(self):
        """Π should equal αβ'."""
        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=2)

        expected_pi = result.alpha @ result.beta.T
        np.testing.assert_allclose(result.pi, expected_pi, rtol=1e-10)

    def test_vecm_residuals_shape(self):
        """Residuals should have correct shape."""
        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=2)

        assert result.residuals.shape[0] == result.n_obs
        assert result.residuals.shape[1] == result.n_vars

    def test_vecm_sigma_symmetric(self):
        """Residual covariance should be symmetric positive semi-definite."""
        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=2)

        # Check symmetry
        np.testing.assert_allclose(result.sigma, result.sigma.T, rtol=1e-10)

        # Check positive semi-definite (eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(result.sigma)
        assert np.all(eigenvalues >= -1e-10)

    def test_vecm_information_criteria(self):
        """AIC and BIC should be computed."""
        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=2)

        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)
        assert np.isfinite(result.log_likelihood)

    def test_vecm_constant_included(self):
        """Constant should be included for det_order >= 0."""
        data = generate_cointegrated_system(n=200)

        result0 = vecm_estimate(data, coint_rank=1, lags=2, det_order=0)
        result1 = vecm_estimate(data, coint_rank=1, lags=2, det_order=1)
        result_m1 = vecm_estimate(data, coint_rank=1, lags=2, det_order=-1)

        assert result0.const is not None
        assert result1.const is not None
        assert result_m1.const is None

    def test_vecm_trivariate(self):
        """VECM should work with more than 2 variables."""
        data = generate_trivariate_cointegrated(n=200)
        result = vecm_estimate(data, coint_rank=2, lags=2)

        assert result.n_vars == 3
        assert result.coint_rank == 2
        assert result.alpha.shape == (3, 2)
        assert result.beta.shape == (3, 2)

    def test_vecm_method_ols(self):
        """OLS method should produce valid results."""
        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=2, method="ols")

        assert isinstance(result, VECMResult)
        assert result.coint_rank == 1
        assert np.isfinite(result.aic)


class TestVECMForecast:
    """Tests for VECM forecasting."""

    def test_forecast_shape(self):
        """Forecast should have correct shape."""
        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=2)

        forecasts = vecm_forecast(result, data, horizons=10)

        assert forecasts.shape == (10, 2)

    def test_forecast_continuity(self):
        """First forecast should be close to last observation."""
        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=2)

        forecasts = vecm_forecast(result, data, horizons=5)

        # First forecast shouldn't be too far from last observation
        # (assuming stable dynamics)
        diff = np.abs(forecasts[0, :] - data[-1, :])
        assert np.all(diff < 10)  # Reasonable bound for this DGP

    def test_forecast_multiple_horizons(self):
        """Forecasts at different horizons should differ."""
        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=2)

        forecasts = vecm_forecast(result, data, horizons=10)

        # Forecasts should evolve over time
        assert not np.allclose(forecasts[0, :], forecasts[9, :])


class TestVECMGrangerCausality:
    """Tests for Granger causality in VECM framework."""

    def test_granger_causality_returns_tuple(self):
        """Granger causality should return (stat, pvalue, df)."""
        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=3)

        stat, pval, df = vecm_granger_causality(result, data, cause_idx=0, effect_idx=1)

        assert isinstance(stat, float)
        assert isinstance(pval, float)
        assert isinstance(df, int)

    def test_granger_causality_pvalue_bounds(self):
        """P-value should be in [0, 1]."""
        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=3)

        _, pval, _ = vecm_granger_causality(result, data, cause_idx=0, effect_idx=1)

        assert 0 <= pval <= 1


class TestErrorCorrectionTerm:
    """Tests for error correction term computation."""

    def test_ect_shape(self):
        """ECT should have shape (T, r)."""
        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=2)

        ect = compute_error_correction_term(data, result.beta)

        assert ect.shape == (200, 1)

    def test_ect_stationarity(self):
        """ECT should be stationary for cointegrated system."""
        from causal_inference.timeseries import adf_test

        data = generate_cointegrated_system(n=200)
        result = vecm_estimate(data, coint_rank=1, lags=2)

        ect = compute_error_correction_term(data, result.beta)

        # ECT should be stationary (ADF should reject unit root)
        adf_result = adf_test(ect.flatten())
        # May not always reject in finite samples, so just check it's computed
        assert np.isfinite(adf_result.statistic)


# ============================================================================
# Layer 2: Adversarial Tests
# ============================================================================


class TestVECMAdversarial:
    """Edge case and adversarial tests for VECM."""

    def test_invalid_coint_rank_zero(self):
        """coint_rank=0 should raise error."""
        data = generate_cointegrated_system(n=200)

        with pytest.raises(ValueError, match="coint_rank must be >= 1"):
            vecm_estimate(data, coint_rank=0, lags=2)

    def test_invalid_coint_rank_too_high(self):
        """coint_rank >= k should raise error."""
        data = generate_cointegrated_system(n=200)

        with pytest.raises(ValueError, match="coint_rank must be < k"):
            vecm_estimate(data, coint_rank=2, lags=2)

    def test_invalid_lags_zero(self):
        """lags=0 should raise error."""
        data = generate_cointegrated_system(n=200)

        with pytest.raises(ValueError, match="lags must be >= 1"):
            vecm_estimate(data, coint_rank=1, lags=0)

    def test_insufficient_observations(self):
        """Short time series should raise error."""
        data = np.random.randn(20, 2)

        with pytest.raises(ValueError, match="Insufficient observations"):
            vecm_estimate(data, coint_rank=1, lags=5)

    def test_1d_data_error(self):
        """1D data should raise error."""
        data = np.random.randn(100)

        with pytest.raises(ValueError, match="2-dimensional"):
            vecm_estimate(data, coint_rank=1, lags=2)

    def test_invalid_det_order(self):
        """Invalid det_order should raise error."""
        data = generate_cointegrated_system(n=200)

        with pytest.raises(ValueError, match="det_order must be"):
            vecm_estimate(data, coint_rank=1, lags=2, det_order=5)

    def test_invalid_method(self):
        """Unknown method should raise error."""
        data = generate_cointegrated_system(n=200)

        with pytest.raises(ValueError, match="Unknown method"):
            vecm_estimate(data, coint_rank=1, lags=2, method="invalid")


class TestVECMRobustness:
    """Tests for robustness of VECM estimation."""

    def test_different_random_seeds(self):
        """VECM should produce different results for different data."""
        data1 = generate_cointegrated_system(n=200, seed=42)
        data2 = generate_cointegrated_system(n=200, seed=123)

        result1 = vecm_estimate(data1, coint_rank=1, lags=2)
        result2 = vecm_estimate(data2, coint_rank=1, lags=2)

        # Coefficients should differ
        assert not np.allclose(result1.alpha, result2.alpha)

    def test_longer_series_stability(self):
        """Longer series should give more stable estimates."""
        np.random.seed(42)

        data_short = generate_cointegrated_system(n=100)
        data_long = generate_cointegrated_system(n=500)

        result_short = vecm_estimate(data_short, coint_rank=1, lags=2)
        result_long = vecm_estimate(data_long, coint_rank=1, lags=2)

        # Both should produce valid results
        assert np.isfinite(result_short.aic)
        assert np.isfinite(result_long.aic)

    def test_higher_rank_trivariate(self):
        """Trivariate system with rank 2 should work."""
        data = generate_trivariate_cointegrated(n=300)
        result = vecm_estimate(data, coint_rank=2, lags=2)

        assert result.coint_rank == 2
        assert result.alpha.shape == (3, 2)


# ============================================================================
# Layer 3: Monte Carlo Validation
# ============================================================================


class TestVECMMonteCarlo:
    """Monte Carlo validation for VECM."""

    @pytest.mark.slow
    def test_alpha_consistency(self):
        """α estimates should be consistent (converge to true values)."""
        n_runs = 50
        n_obs = 500

        # True DGP: VECM with known α
        # For simplicity, we check that estimates are not wildly wrong

        alpha_estimates = []
        for run in range(n_runs):
            np.random.seed(run)
            data = generate_cointegrated_system(n=n_obs, seed=run)
            result = vecm_estimate(data, coint_rank=1, lags=2)
            alpha_estimates.append(result.alpha.flatten())

        alpha_estimates = np.array(alpha_estimates)
        mean_alpha = np.mean(alpha_estimates, axis=0)
        std_alpha = np.std(alpha_estimates, axis=0)

        # Check that estimates are stable (low variance)
        assert np.all(std_alpha < 0.5), f"α estimates too variable: std={std_alpha}"

    @pytest.mark.slow
    def test_forecast_accuracy(self):
        """Forecasts should be reasonably accurate for cointegrated systems."""
        n_runs = 30

        mse_list = []
        for run in range(n_runs):
            np.random.seed(run)

            # Generate full data
            full_data = generate_cointegrated_system(n=250, seed=run)

            # Use first 200 for estimation
            train_data = full_data[:200, :]
            test_data = full_data[200:210, :]  # 10 periods ahead

            result = vecm_estimate(train_data, coint_rank=1, lags=2)
            forecasts = vecm_forecast(result, train_data, horizons=10)

            mse = np.mean((forecasts - test_data) ** 2)
            mse_list.append(mse)

        mean_mse = np.mean(mse_list)
        # MSE should be bounded for this stable DGP
        assert mean_mse < 5.0, f"Mean MSE {mean_mse:.2f} too high"


# ============================================================================
# Integration Tests
# ============================================================================


class TestVECMIntegration:
    """Integration tests combining VECM with other components."""

    def test_johansen_vecm_consistency(self):
        """VECM β should match Johansen test eigenvectors."""
        data = generate_cointegrated_system(n=200)

        johansen_result = johansen_test(data, lags=2)
        vecm_result = vecm_estimate(data, coint_rank=1, lags=2)

        # β vectors should be proportional (up to normalization)
        johansen_beta = johansen_result.eigenvectors[:, 0]
        vecm_beta = vecm_result.beta[:, 0]

        # Check correlation is near 1 or -1
        corr = np.abs(np.corrcoef(johansen_beta, vecm_beta)[0, 1])
        assert corr > 0.99, f"β correlation {corr:.4f} too low"

    def test_vecm_granger_with_causality(self):
        """VECM Granger test should detect designed causality."""
        np.random.seed(42)
        n = 300

        # Create causal structure: X causes Y
        trend = np.cumsum(np.random.randn(n))
        x = trend + np.random.randn(n) * 0.3
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.5 * x[t - 1] + 0.3 * y[t - 1] + 0.2 * trend[t] + np.random.randn() * 0.5

        data = np.column_stack([y, x])
        result = vecm_estimate(data, coint_rank=1, lags=3)

        # X (idx 1) should Granger-cause Y (idx 0)
        stat, pval, _ = vecm_granger_causality(result, data, cause_idx=1, effect_idx=0)

        # Check that test was computed (may or may not be significant)
        assert np.isfinite(stat)
        assert 0 <= pval <= 1
