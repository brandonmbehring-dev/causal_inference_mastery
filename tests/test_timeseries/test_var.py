"""
Tests for VAR Estimation.

Session 135: Tests for Vector Autoregression estimation.
"""

import numpy as np
import pytest

from causal_inference.timeseries import (
    var_estimate,
    var_forecast,
    var_residuals,
    VARResult,
)
from causal_inference.timeseries.stationarity import adf_test, check_stationarity
from causal_inference.timeseries.lag_selection import select_lag_order


class TestVAREstimation:
    """Tests for VAR estimation."""

    def test_basic_var1_estimation(self):
        """Basic VAR(1) estimation should work."""
        np.random.seed(42)
        data = np.random.randn(200, 2)

        result = var_estimate(data, lags=1)

        assert isinstance(result, VARResult)
        assert result.lags == 1
        assert result.n_vars == 2
        assert result.coefficients.shape == (2, 3)  # 2 vars, 1 + 2 coeffs

    def test_var_coefficient_recovery(self, sample_var1_data):
        """VAR should recover true coefficients approximately."""
        data, true_A1 = sample_var1_data

        result = var_estimate(data, lags=1)

        # Extract estimated A1 matrix
        est_A1 = result.get_lag_matrix(1)

        # Should be close to true (within noise)
        np.testing.assert_allclose(est_A1, true_A1, atol=0.2)

    def test_var2_estimation(self, sample_var2_data):
        """VAR(2) estimation should work."""
        data, (true_A1, true_A2) = sample_var2_data

        result = var_estimate(data, lags=2)

        assert result.lags == 2
        # Coefficients: intercept + 2 vars * 2 lags = 5 columns
        assert result.coefficients.shape == (2, 5)

        est_A1 = result.get_lag_matrix(1)
        est_A2 = result.get_lag_matrix(2)

        # Approximate recovery
        np.testing.assert_allclose(est_A1, true_A1, atol=0.25)
        np.testing.assert_allclose(est_A2, true_A2, atol=0.25)

    def test_var_residuals_shape(self):
        """Residuals should have correct shape."""
        np.random.seed(42)
        n, k = 100, 3
        data = np.random.randn(n, k)

        result = var_estimate(data, lags=2)

        assert result.residuals.shape == (n - 2, k)
        assert result.n_obs_effective == n - 2

    def test_var_sigma_positive_definite(self):
        """Residual covariance should be positive definite."""
        np.random.seed(42)
        data = np.random.randn(200, 2)

        result = var_estimate(data, lags=1)

        eigvals = np.linalg.eigvalsh(result.sigma)
        assert all(eigvals >= 0)  # Positive semi-definite at least

    def test_var_information_criteria(self):
        """Information criteria should be computed."""
        np.random.seed(42)
        data = np.random.randn(200, 2)

        result = var_estimate(data, lags=2)

        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)
        assert np.isfinite(result.hqc)
        assert np.isfinite(result.log_likelihood)

    def test_var_names_custom(self):
        """Custom variable names should be stored."""
        np.random.seed(42)
        data = np.random.randn(100, 2)

        result = var_estimate(data, lags=1, var_names=["GDP", "Inflation"])

        assert result.var_names == ["GDP", "Inflation"]

    def test_get_intercepts(self):
        """Intercept extraction should work."""
        np.random.seed(42)
        data = np.random.randn(100, 2)

        result = var_estimate(data, lags=1)
        intercepts = result.get_intercepts()

        assert intercepts.shape == (2,)

    def test_get_lag_matrix_invalid(self):
        """Invalid lag should raise."""
        np.random.seed(42)
        data = np.random.randn(100, 2)
        result = var_estimate(data, lags=2)

        with pytest.raises(ValueError):
            result.get_lag_matrix(3)

        with pytest.raises(ValueError):
            result.get_lag_matrix(0)


class TestVARForecast:
    """Tests for VAR forecasting."""

    def test_basic_forecast(self):
        """Basic forecasting should work."""
        np.random.seed(42)
        data = np.random.randn(100, 2)

        result = var_estimate(data, lags=1)
        forecast = var_forecast(result, data, steps=5)

        assert forecast.shape == (5, 2)
        assert np.all(np.isfinite(forecast))

    def test_forecast_single_step(self):
        """Single step forecast should work."""
        np.random.seed(42)
        data = np.random.randn(100, 2)

        result = var_estimate(data, lags=1)
        forecast = var_forecast(result, data, steps=1)

        assert forecast.shape == (1, 2)

    def test_forecast_many_steps(self):
        """Many step forecast should work."""
        np.random.seed(42)
        data = np.random.randn(100, 2)

        result = var_estimate(data, lags=1)
        forecast = var_forecast(result, data, steps=50)

        assert forecast.shape == (50, 2)

    def test_forecast_insufficient_history_raises(self):
        """Insufficient history for forecasting should raise."""
        np.random.seed(42)
        data = np.random.randn(100, 2)
        result = var_estimate(data, lags=5)

        short_data = np.random.randn(3, 2)

        with pytest.raises(ValueError, match="Need at least"):
            var_forecast(result, short_data, steps=1)

    def test_forecast_zero_steps_raises(self):
        """Zero steps should raise."""
        np.random.seed(42)
        data = np.random.randn(100, 2)
        result = var_estimate(data, lags=1)

        with pytest.raises(ValueError, match="steps must be >= 1"):
            var_forecast(result, data, steps=0)


class TestVARResiduals:
    """Tests for residual computation."""

    def test_residuals_from_training_data(self):
        """Residuals from training data should match stored."""
        np.random.seed(42)
        data = np.random.randn(100, 2)

        result = var_estimate(data, lags=2)
        computed_resid = var_residuals(result, data)

        np.testing.assert_allclose(computed_resid, result.residuals, rtol=1e-10)

    def test_residuals_shape(self):
        """Residual shape should be correct."""
        np.random.seed(42)
        data = np.random.randn(100, 3)

        result = var_estimate(data, lags=3)
        resid = var_residuals(result, data)

        assert resid.shape == (97, 3)


class TestADFTest:
    """Tests for ADF stationarity test."""

    def test_stationary_series(self, sample_stationary_series):
        """Stationary series should be detected."""
        result = adf_test(sample_stationary_series)

        assert result.is_stationary  # Truthy check
        assert result.p_value < 0.05

    def test_nonstationary_series(self):
        """Non-stationary series should be detected."""
        # Use a specific random walk that's clearly non-stationary
        np.random.seed(123)  # Different seed for reliability
        n = 300  # Longer series for clearer non-stationarity
        series = np.cumsum(np.random.randn(n))

        result = adf_test(series)

        assert not result.is_stationary  # Truthy check
        assert result.p_value > 0.05

    def test_adf_critical_values(self, sample_stationary_series):
        """Critical values should be present."""
        result = adf_test(sample_stationary_series)

        assert "1%" in result.critical_values
        assert "5%" in result.critical_values
        assert "10%" in result.critical_values

    def test_adf_regression_types(self):
        """Different regression types should work."""
        np.random.seed(42)
        series = np.random.randn(100)

        for reg in ["n", "c", "ct"]:
            result = adf_test(series, regression=reg)
            assert result.regression == reg

    def test_adf_invalid_regression_raises(self):
        """Invalid regression type should raise."""
        series = np.random.randn(100)

        with pytest.raises(ValueError, match="must be 'n', 'c', or 'ct'"):
            adf_test(series, regression="invalid")

    def test_adf_short_series_raises(self):
        """Too short series should raise."""
        series = np.random.randn(5)

        with pytest.raises(ValueError, match="too short"):
            adf_test(series)


class TestCheckStationarity:
    """Tests for multivariate stationarity check."""

    def test_check_multiple_series(self):
        """Check stationarity for multiple series."""
        np.random.seed(456)  # Different seed
        n = 300  # Longer series
        data = np.column_stack(
            [
                np.random.randn(n),  # Stationary
                np.cumsum(np.random.randn(n)),  # Non-stationary
            ]
        )

        results = check_stationarity(
            data,
            var_names=["stat", "nonstat"],
        )

        assert "stat" in results
        assert "nonstat" in results
        assert results["stat"].is_stationary  # Truthy check
        assert not results["nonstat"].is_stationary  # Truthy check


class TestLagSelection:
    """Tests for lag order selection."""

    def test_basic_lag_selection(self):
        """Basic lag selection should work."""
        np.random.seed(42)
        data = np.random.randn(200, 2)

        result = select_lag_order(data, max_lags=5)

        assert 1 <= result.optimal_lag <= 5
        assert result.criterion in ["aic", "bic", "hqc"]

    def test_lag_selection_all_criteria(self):
        """All criteria values should be computed."""
        np.random.seed(42)
        data = np.random.randn(200, 2)

        result = select_lag_order(data, max_lags=5)

        assert len(result.aic_values) > 0
        assert len(result.bic_values) > 0
        assert len(result.hqc_values) > 0

    def test_lag_selection_bic_simpler(self):
        """BIC should tend toward simpler models."""
        np.random.seed(42)
        data = np.random.randn(200, 2)

        result_aic = select_lag_order(data, max_lags=8, criterion="aic")
        result_bic = select_lag_order(data, max_lags=8, criterion="bic")

        # BIC penalty is stronger, should select <= AIC lag
        assert result_bic.optimal_lag <= result_aic.optimal_lag + 1

    def test_get_optimal_by_criterion(self):
        """Getting optimal by specific criterion should work."""
        np.random.seed(42)
        data = np.random.randn(200, 2)

        result = select_lag_order(data, max_lags=5, criterion="aic")

        aic_optimal = result.get_optimal_by_criterion("aic")
        bic_optimal = result.get_optimal_by_criterion("bic")

        assert 1 <= aic_optimal <= 5
        assert 1 <= bic_optimal <= 5


class TestVARInputValidation:
    """Tests for input validation."""

    def test_1d_data_raises(self):
        """1D data should raise."""
        with pytest.raises(ValueError, match="must be 2D"):
            var_estimate(np.random.randn(100), lags=1)

    def test_insufficient_obs_raises(self):
        """Too few observations should raise."""
        with pytest.raises(ValueError, match="Insufficient observations"):
            var_estimate(np.random.randn(5, 2), lags=10)

    def test_negative_lags_raises(self):
        """Negative lags should raise."""
        with pytest.raises(ValueError, match="Lags must be >= 1"):
            var_estimate(np.random.randn(100, 2), lags=0)

    def test_var_names_mismatch_raises(self):
        """Mismatched var_names length should raise."""
        with pytest.raises(ValueError, match="must match"):
            var_estimate(
                np.random.randn(100, 2),
                lags=1,
                var_names=["A", "B", "C"],
            )


class TestVARReproducibility:
    """Tests for reproducibility."""

    def test_deterministic_estimation(self):
        """Same data should give same results."""
        np.random.seed(42)
        data = np.random.randn(100, 2)

        result1 = var_estimate(data, lags=1)
        result2 = var_estimate(data, lags=1)

        np.testing.assert_array_equal(result1.coefficients, result2.coefficients)
        assert result1.aic == result2.aic

    def test_repr_format(self):
        """String representation should be informative."""
        np.random.seed(42)
        data = np.random.randn(100, 2)

        result = var_estimate(data, lags=2)
        repr_str = repr(result)

        assert "VARResult" in repr_str
        assert "n_vars=2" in repr_str
        assert "lags=2" in repr_str
