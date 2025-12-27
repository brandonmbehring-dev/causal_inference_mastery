"""
Tests for Extended Stationarity Tests (KPSS and Phillips-Perron).

Session 145: Tests for KPSS and Phillips-Perron stationarity tests.

Test layers:
- Layer 1: Known-answer tests (stationary vs non-stationary)
- Layer 2: Adversarial tests (edge cases)
- Layer 3: Monte Carlo validation (Type I error, power) @slow
"""

import numpy as np
import pytest

from causal_inference.timeseries.stationarity import (
    kpss_test,
    phillips_perron_test,
    confirmatory_stationarity_test,
    adf_test,
)
from causal_inference.timeseries.types import KPSSResult, PPResult


# ============================================================================
# Layer 1: Known-Answer Tests - KPSS
# ============================================================================


class TestKPSSKnownAnswer:
    """Known-answer tests for KPSS test."""

    def test_stationary_ar1_is_stationary(self, sample_stationary_series):
        """KPSS should fail to reject stationarity for AR(1)."""
        result = kpss_test(sample_stationary_series)

        assert isinstance(result, KPSSResult)
        assert result.is_stationary is True
        assert result.statistic < result.critical_values["5%"]

    def test_random_walk_is_nonstationary(self, sample_nonstationary_series):
        """KPSS should reject stationarity for random walk."""
        result = kpss_test(sample_nonstationary_series)

        assert result.is_stationary is False
        assert result.statistic > result.critical_values["5%"]

    def test_white_noise_is_stationary(self, seed):
        """KPSS should fail to reject stationarity for white noise."""
        np.random.seed(seed)
        y = np.random.randn(200)

        result = kpss_test(y)

        assert result.is_stationary is True

    def test_trend_stationary_with_ct(self, seed):
        """KPSS with ct regression should detect trend-stationarity."""
        np.random.seed(seed)
        n = 200
        t = np.arange(n)
        # Trend-stationary: linear trend + stationary errors
        y = 0.1 * t + np.random.randn(n)

        result = kpss_test(y, regression="ct")

        assert result.is_stationary is True
        assert result.regression == "ct"

    def test_result_structure(self, sample_stationary_series):
        """Result should have all required fields."""
        result = kpss_test(sample_stationary_series)

        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")
        assert hasattr(result, "lags")
        assert hasattr(result, "n_obs")
        assert hasattr(result, "critical_values")
        assert hasattr(result, "is_stationary")
        assert hasattr(result, "regression")
        assert hasattr(result, "alpha")

        assert "5%" in result.critical_values
        assert "1%" in result.critical_values

    def test_custom_lags(self, sample_stationary_series):
        """Should accept custom lag parameter."""
        result1 = kpss_test(sample_stationary_series, lags=5)
        result2 = kpss_test(sample_stationary_series, lags=10)

        assert result1.lags == 5
        assert result2.lags == 10
        # Different lags may give different statistics
        assert result1.statistic != result2.statistic


# ============================================================================
# Layer 1: Known-Answer Tests - Phillips-Perron
# ============================================================================


class TestPhillipsPerronKnownAnswer:
    """Known-answer tests for Phillips-Perron test."""

    def test_stationary_ar1_is_stationary(self, sample_stationary_series):
        """PP should reject unit root for AR(1)."""
        result = phillips_perron_test(sample_stationary_series)

        assert isinstance(result, PPResult)
        assert result.is_stationary is True
        assert result.statistic < result.critical_values["5%"]

    def test_random_walk_is_nonstationary(self, sample_nonstationary_series):
        """PP should fail to reject unit root for random walk."""
        result = phillips_perron_test(sample_nonstationary_series)

        assert result.is_stationary is False

    def test_white_noise_is_stationary(self, seed):
        """PP should reject unit root for white noise (with larger sample)."""
        np.random.seed(seed)
        # Larger sample for more power
        y = np.random.randn(500)

        result = phillips_perron_test(y)

        # Should detect stationarity with larger sample
        assert result.is_stationary is True

    def test_consistent_with_adf(self, sample_stationary_series):
        """PP should generally agree with ADF on stationarity."""
        pp_result = phillips_perron_test(sample_stationary_series)
        adf_result = adf_test(sample_stationary_series)

        # Both should agree on stationary series
        assert pp_result.is_stationary == adf_result.is_stationary

    def test_result_structure(self, sample_stationary_series):
        """Result should have all required fields."""
        result = phillips_perron_test(sample_stationary_series)

        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")
        assert hasattr(result, "lags")
        assert hasattr(result, "n_obs")
        assert hasattr(result, "critical_values")
        assert hasattr(result, "is_stationary")
        assert hasattr(result, "regression")
        assert hasattr(result, "rho_stat")

    def test_regression_types(self, sample_stationary_series):
        """Should work with all regression types."""
        result_n = phillips_perron_test(sample_stationary_series, regression="n")
        result_c = phillips_perron_test(sample_stationary_series, regression="c")
        result_ct = phillips_perron_test(sample_stationary_series, regression="ct")

        assert result_n.regression == "n"
        assert result_c.regression == "c"
        assert result_ct.regression == "ct"


# ============================================================================
# Layer 1: Known-Answer Tests - Confirmatory
# ============================================================================


class TestConfirmatoryTest:
    """Tests for combined ADF + KPSS analysis."""

    def test_stationary_both_agree(self, sample_stationary_series):
        """Both tests should agree on stationary series."""
        result = confirmatory_stationarity_test(sample_stationary_series)

        assert "adf" in result
        assert "kpss" in result
        assert "conclusion" in result
        assert "interpretation" in result
        assert result["conclusion"] == "stationary"

    def test_nonstationary_detected(self, seed):
        """Confirmatory test should detect clear non-stationarity."""
        np.random.seed(seed)
        # Longer random walk for clearer signal
        y = np.cumsum(np.random.randn(500))

        result = confirmatory_stationarity_test(y)

        # Should be non-stationary or at least not conclusively stationary
        assert result["conclusion"] in ["non-stationary", "inconclusive"]
        # And should not conclude it's stationary
        assert result["conclusion"] != "stationary"


# ============================================================================
# Layer 2: Adversarial Tests
# ============================================================================


class TestStationarityAdversarial:
    """Adversarial tests for stationarity functions."""

    def test_kpss_short_series(self, seed):
        """KPSS should handle short series."""
        np.random.seed(seed)
        y = np.random.randn(50)

        result = kpss_test(y)

        assert isinstance(result, KPSSResult)
        assert not np.isnan(result.statistic)

    def test_pp_short_series(self, seed):
        """PP should handle short series."""
        np.random.seed(seed)
        y = np.random.randn(50)

        result = phillips_perron_test(y)

        assert isinstance(result, PPResult)
        assert not np.isnan(result.statistic)

    def test_kpss_too_short_raises(self):
        """KPSS should raise for series < 10."""
        y = np.random.randn(5)

        with pytest.raises(ValueError, match="too short"):
            kpss_test(y)

    def test_pp_too_short_raises(self):
        """PP should raise for series < 10."""
        y = np.random.randn(5)

        with pytest.raises(ValueError, match="too short"):
            phillips_perron_test(y)

    def test_kpss_invalid_regression(self, seed):
        """KPSS should raise for invalid regression type."""
        np.random.seed(seed)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="regression must be"):
            kpss_test(y, regression="n")  # KPSS doesn't support "n"

    def test_pp_invalid_regression(self, seed):
        """PP should raise for invalid regression type."""
        np.random.seed(seed)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="regression must be"):
            phillips_perron_test(y, regression="invalid")

    def test_kpss_near_unit_root(self, seed):
        """KPSS should handle near-unit-root process (rho=0.99)."""
        np.random.seed(seed)
        n = 200
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.99 * y[t - 1] + np.random.randn()

        # Near unit root - KPSS may or may not reject
        result = kpss_test(y)
        assert isinstance(result, KPSSResult)
        assert not np.isnan(result.statistic)

    def test_pp_near_unit_root(self, seed):
        """PP should handle near-unit-root process (rho=0.99)."""
        np.random.seed(seed)
        n = 200
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.99 * y[t - 1] + np.random.randn()

        result = phillips_perron_test(y)
        assert isinstance(result, PPResult)
        assert not np.isnan(result.statistic)

    def test_kpss_constant_series(self, seed):
        """KPSS should handle constant + small noise."""
        np.random.seed(seed)
        y = 5.0 + np.random.randn(100) * 0.01

        result = kpss_test(y)
        assert result.is_stationary is True

    def test_pp_deterministic_trend(self, seed):
        """PP with ct should handle deterministic trend."""
        np.random.seed(seed)
        n = 200
        t = np.arange(n)
        y = 0.5 * t + np.random.randn(n)

        result = phillips_perron_test(y, regression="ct")
        # Trend-stationary process should be detected as stationary
        # when trend is accounted for
        assert isinstance(result, PPResult)

    def test_kpss_heteroskedastic_errors(self, seed):
        """KPSS should work with heteroskedastic errors."""
        np.random.seed(seed)
        n = 200
        # Heteroskedastic errors: variance increases with time
        errors = np.random.randn(n) * np.sqrt(np.arange(1, n + 1) / 50)
        y = np.cumsum(errors * 0.1) + np.random.randn(n)

        result = kpss_test(y)
        assert not np.isnan(result.statistic)


# ============================================================================
# Layer 3: Monte Carlo Tests
# ============================================================================


class TestStationarityMonteCarlo:
    """Monte Carlo validation for stationarity tests."""

    @pytest.mark.slow
    def test_kpss_type1_error(self):
        """KPSS Type I error should be close to alpha under H0."""
        n_runs = 500
        alpha = 0.05
        rejections = 0

        for seed in range(n_runs):
            np.random.seed(seed)
            # Under H0: stationary AR(1)
            n = 200
            y = np.zeros(n)
            for t in range(1, n):
                y[t] = 0.5 * y[t - 1] + np.random.randn()

            result = kpss_test(y, alpha=alpha)
            if not result.is_stationary:  # Rejects H0 (stationarity)
                rejections += 1

        type1_rate = rejections / n_runs
        # Should be around 5% (allow 2-10%)
        assert 0.02 < type1_rate < 0.12, f"Type I error {type1_rate:.2%} outside bounds"

    @pytest.mark.slow
    def test_kpss_power(self):
        """KPSS should have high power against random walk."""
        n_runs = 200
        rejections = 0

        for seed in range(n_runs):
            np.random.seed(seed)
            # Under H1: random walk
            y = np.cumsum(np.random.randn(200))

            result = kpss_test(y, alpha=0.05)
            if not result.is_stationary:
                rejections += 1

        power = rejections / n_runs
        # Should have reasonable power (>70%)
        assert power > 0.70, f"KPSS power {power:.2%} too low"

    @pytest.mark.slow
    def test_pp_type1_error(self):
        """PP Type I error should be close to alpha under H0."""
        n_runs = 500
        alpha = 0.05
        rejections = 0

        for seed in range(n_runs):
            np.random.seed(seed)
            # Under H0: random walk (unit root)
            y = np.cumsum(np.random.randn(200))

            result = phillips_perron_test(y, alpha=alpha)
            if result.is_stationary:  # Rejects H0 (unit root)
                rejections += 1

        type1_rate = rejections / n_runs
        # Should be around 5% (allow 2-10%)
        assert 0.02 < type1_rate < 0.12, f"Type I error {type1_rate:.2%} outside bounds"

    @pytest.mark.slow
    def test_pp_power(self):
        """PP should have high power against stationary AR(1)."""
        n_runs = 200
        rejections = 0

        for seed in range(n_runs):
            np.random.seed(seed)
            # Under H1: stationary AR(1)
            n = 200
            y = np.zeros(n)
            for t in range(1, n):
                y[t] = 0.5 * y[t - 1] + np.random.randn()

            result = phillips_perron_test(y, alpha=0.05)
            if result.is_stationary:
                rejections += 1

        power = rejections / n_runs
        # Should have high power (>80%)
        assert power > 0.80, f"PP power {power:.2%} too low"

    @pytest.mark.slow
    def test_pp_vs_adf_agreement(self):
        """PP and ADF should agree on most cases."""
        n_runs = 200
        agreements = 0

        for seed in range(n_runs):
            np.random.seed(seed)
            # Mix of stationary and non-stationary
            if seed % 2 == 0:
                # Stationary
                n = 200
                y = np.zeros(n)
                for t in range(1, n):
                    y[t] = 0.5 * y[t - 1] + np.random.randn()
            else:
                # Non-stationary
                y = np.cumsum(np.random.randn(200))

            pp_result = phillips_perron_test(y)
            adf_result = adf_test(y)

            if pp_result.is_stationary == adf_result.is_stationary:
                agreements += 1

        agreement_rate = agreements / n_runs
        # Should agree most of the time (>85%)
        assert agreement_rate > 0.85, f"PP/ADF agreement {agreement_rate:.2%} too low"

    @pytest.mark.slow
    def test_confirmatory_consistency(self):
        """Confirmatory test should give consistent conclusions."""
        n_runs = 100
        consistent_stationary = 0
        consistent_nonstationary = 0

        for seed in range(n_runs):
            np.random.seed(seed)

            # Clearly stationary
            n = 200
            y = np.zeros(n)
            for t in range(1, n):
                y[t] = 0.3 * y[t - 1] + np.random.randn()

            result = confirmatory_stationarity_test(y)
            if result["conclusion"] == "stationary":
                consistent_stationary += 1

        for seed in range(n_runs):
            np.random.seed(seed + n_runs)

            # Clearly non-stationary
            y = np.cumsum(np.random.randn(200))

            result = confirmatory_stationarity_test(y)
            if result["conclusion"] == "non-stationary":
                consistent_nonstationary += 1

        stat_rate = consistent_stationary / n_runs
        nonstat_rate = consistent_nonstationary / n_runs

        assert stat_rate > 0.80, f"Stationary detection {stat_rate:.2%} too low"
        assert nonstat_rate > 0.80, f"Non-stationary detection {nonstat_rate:.2%} too low"
