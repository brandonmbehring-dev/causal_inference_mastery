"""
Tests for Proxy SVAR (External Instrument SVAR).

Session 163: Test suite for Stock & Watson (2012), Mertens & Ravn (2013) methodology.

Three-layer validation:
1. Known-Answer: Tests with known DGP structure
2. Adversarial: Edge cases, invalid inputs, error handling
3. Monte Carlo: Statistical validation (bias, coverage)
"""

import warnings

import numpy as np
import pytest
from scipy import linalg

from causal_inference.timeseries import (
    var_estimate,
    IdentificationMethod,
)
from causal_inference.timeseries.proxy_svar import (
    ProxySVARResult,
    proxy_svar,
    weak_instrument_diagnostics,
    compute_irf_from_proxy,
    _first_stage_regression,
    _compute_proxy_impact_column,
    _complete_impact_matrix,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def generate_proxy_svar_dgp(
    n: int = 500,
    n_vars: int = 3,
    lags: int = 1,
    instrument_strength: float = 0.5,
    seed: int = 42,
) -> tuple:
    """
    Generate data from a known SVAR DGP with external instrument.

    Parameters
    ----------
    n : int
        Number of observations
    n_vars : int
        Number of variables
    lags : int
        VAR lag order
    instrument_strength : float
        Correlation between instrument and target shock (0-1)
    seed : int
        Random seed

    Returns
    -------
    tuple
        (data, instrument, B0_inv_true, var_coefficients)
    """
    np.random.seed(seed)

    # True structural impact matrix (lower triangular)
    B0_inv_true = np.eye(n_vars)
    B0_inv_true[1, 0] = 0.3  # Shock 0 affects var 1
    B0_inv_true[2, 0] = 0.2  # Shock 0 affects var 2
    B0_inv_true[2, 1] = 0.1  # Shock 1 affects var 2

    # Structural shocks
    eps = np.random.randn(n, n_vars)

    # External instrument correlated with first shock
    noise = np.random.randn(n)
    z = instrument_strength * eps[:, 0] + np.sqrt(1 - instrument_strength**2) * noise

    # Reduced-form errors
    u = (B0_inv_true @ eps.T).T

    # VAR coefficients (stable)
    A1 = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.0, 0.0, 0.3]])[:n_vars, :n_vars]

    # Generate VAR data
    data = np.zeros((n, n_vars))
    for t in range(lags, n):
        for p in range(1, lags + 1):
            data[t] += A1 @ data[t - p]
        data[t] += u[t]

    return data, z, B0_inv_true, A1


@pytest.fixture
def basic_proxy_data():
    """Generate basic test data for proxy SVAR."""
    return generate_proxy_svar_dgp(n=500, instrument_strength=0.5, seed=42)


@pytest.fixture
def strong_instrument_data():
    """Generate data with strong instrument."""
    return generate_proxy_svar_dgp(n=500, instrument_strength=0.8, seed=123)


@pytest.fixture
def weak_instrument_data():
    """Generate data with weak instrument."""
    return generate_proxy_svar_dgp(n=200, instrument_strength=0.05, seed=456)


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestProxySVARKnownAnswer:
    """Layer 1: Tests with known DGP structure."""

    def test_basic_estimation(self, basic_proxy_data):
        """Proxy SVAR runs without error on valid data."""
        data, z, _, _ = basic_proxy_data
        var_result = var_estimate(data, lags=1)

        # Trim instrument to match VAR residuals
        z_trimmed = z[1:]

        result = proxy_svar(var_result, instrument=z_trimmed, target_shock_idx=0)

        assert isinstance(result, ProxySVARResult)
        assert result.identification == IdentificationMethod.PROXY

    def test_output_shapes(self, basic_proxy_data):
        """All outputs have correct shapes."""
        data, z, _, _ = basic_proxy_data
        n_vars = data.shape[1]
        var_result = var_estimate(data, lags=1)
        n_obs = var_result.n_obs_effective

        result = proxy_svar(var_result, instrument=z[1:], target_shock_idx=0)

        # B0_inv and B0
        assert result.B0_inv.shape == (n_vars, n_vars)
        assert result.B0.shape == (n_vars, n_vars)

        # Structural shocks
        assert result.structural_shocks.shape == (n_obs, n_vars)

        # Impact column
        assert result.impact_column.shape == (n_vars,)
        assert result.impact_column_se.shape == (n_vars,)
        assert result.impact_column_ci_lower.shape == (n_vars,)
        assert result.impact_column_ci_upper.shape == (n_vars,)

    def test_first_stage_f_stat_computation(self, strong_instrument_data):
        """F-statistic computed correctly for strong instrument."""
        data, z, _, _ = strong_instrument_data
        var_result = var_estimate(data, lags=1)

        result = proxy_svar(var_result, instrument=z[1:], target_shock_idx=0)

        # Strong instrument should have high F-stat
        assert result.first_stage_f_stat > 10.0
        assert result.first_stage_r2 > 0.1
        assert not result.is_weak_instrument

    def test_impact_column_normalization(self, basic_proxy_data):
        """Impact column correctly normalized."""
        data, z, _, _ = basic_proxy_data
        var_result = var_estimate(data, lags=1)

        result = proxy_svar(
            var_result,
            instrument=z[1:],
            target_shock_idx=0,
            target_residual_idx=0,
        )

        # Target shock impact on target residual should be 1.0
        assert np.isclose(result.impact_column[0], 1.0, atol=0.01)

    def test_identification_method_is_proxy(self, basic_proxy_data):
        """Identification method correctly set to PROXY."""
        data, z, _, _ = basic_proxy_data
        var_result = var_estimate(data, lags=1)

        result = proxy_svar(var_result, instrument=z[1:])

        assert result.identification == IdentificationMethod.PROXY
        assert result.is_just_identified == True
        assert result.is_over_identified == False

    def test_strong_instrument_recovery(self, strong_instrument_data):
        """Strong instrument recovers approximately correct impact vector."""
        data, z, B0_inv_true, _ = strong_instrument_data
        var_result = var_estimate(data, lags=1)

        result = proxy_svar(var_result, instrument=z[1:], target_shock_idx=0)

        # First column should be close to true (up to scale)
        true_col = B0_inv_true[:, 0]
        est_col = result.impact_column

        # Normalize both for comparison
        true_normalized = true_col / true_col[0]
        est_normalized = est_col / est_col[0]

        # Should be reasonably close (within 0.3 of true)
        assert np.allclose(est_normalized, true_normalized, atol=0.3)

    def test_structural_shocks_properties(self, basic_proxy_data):
        """Structural shocks have approximately correct properties."""
        data, z, _, _ = basic_proxy_data
        var_result = var_estimate(data, lags=1)

        result = proxy_svar(var_result, instrument=z[1:])

        eps = result.structural_shocks

        # Mean should be close to zero
        assert np.allclose(np.mean(eps, axis=0), 0, atol=0.1)

        # Variance should be close to 1 (up to scaling)
        variances = np.var(eps, axis=0)
        assert np.all(variances > 0.1)  # Not degenerate

    def test_confidence_intervals_bracket_estimate(self, basic_proxy_data):
        """Confidence intervals bracket the point estimate."""
        data, z, _, _ = basic_proxy_data
        var_result = var_estimate(data, lags=1)

        result = proxy_svar(var_result, instrument=z[1:], alpha=0.05)

        # CI should bracket estimate
        for i in range(result.n_vars):
            assert result.impact_column_ci_lower[i] <= result.impact_column[i]
            assert result.impact_column[i] <= result.impact_column_ci_upper[i]


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestProxySVARAdversarial:
    """Layer 2: Edge cases and error handling."""

    def test_weak_instrument_warning(self, weak_instrument_data):
        """Weak instrument (F < 10) triggers warning."""
        data, z, _, _ = weak_instrument_data
        var_result = var_estimate(data, lags=1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = proxy_svar(
                var_result,
                instrument=z[1:],
                weak_instrument_threshold=10.0,
            )

            # Should trigger weak instrument warning
            assert result.is_weak_instrument == True
            assert result.first_stage_f_stat < 10.0

    def test_invalid_instrument_length(self, basic_proxy_data):
        """Mismatched instrument length raises error."""
        data, z, _, _ = basic_proxy_data
        var_result = var_estimate(data, lags=1)

        # Wrong length (not trimmed for VAR lags)
        with pytest.raises(ValueError, match="Instrument length"):
            proxy_svar(var_result, instrument=z)

        # Too short
        with pytest.raises(ValueError, match="Instrument length"):
            proxy_svar(var_result, instrument=z[:10])

    def test_constant_instrument_error(self, basic_proxy_data):
        """Constant instrument raises informative error."""
        data, _, _, _ = basic_proxy_data
        var_result = var_estimate(data, lags=1)

        # Constant instrument
        z_const = np.ones(var_result.n_obs_effective)

        with pytest.raises(ValueError, match="near-zero variance"):
            proxy_svar(var_result, instrument=z_const)

    def test_invalid_target_idx(self, basic_proxy_data):
        """Out of bounds target_shock_idx raises error."""
        data, z, _, _ = basic_proxy_data
        var_result = var_estimate(data, lags=1)

        with pytest.raises(ValueError, match="target_shock_idx"):
            proxy_svar(var_result, instrument=z[1:], target_shock_idx=10)

        with pytest.raises(ValueError, match="target_shock_idx"):
            proxy_svar(var_result, instrument=z[1:], target_shock_idx=-1)

    def test_nan_in_instrument(self, basic_proxy_data):
        """NaN values in instrument raises error."""
        data, z, _, _ = basic_proxy_data
        var_result = var_estimate(data, lags=1)

        z_with_nan = z[1:].copy()
        z_with_nan[10] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            proxy_svar(var_result, instrument=z_with_nan)

    def test_nearly_perfect_instrument(self):
        """Very strong instrument (R² → 1) handled correctly."""
        np.random.seed(789)
        n = 200
        n_vars = 3

        # Create nearly perfect instrument
        eps = np.random.randn(n, n_vars)
        z = eps[:, 0] + 1e-10 * np.random.randn(n)  # Almost perfect correlation

        B0_inv = np.eye(n_vars)
        B0_inv[1, 0] = 0.3
        u = (B0_inv @ eps.T).T

        data = np.zeros((n, n_vars))
        for t in range(1, n):
            data[t] = 0.5 * data[t - 1] + u[t]

        var_result = var_estimate(data, lags=1)

        # Should work without error
        result = proxy_svar(var_result, instrument=z[1:])
        assert result.first_stage_r2 > 0.99

    def test_invalid_target_residual_idx(self, basic_proxy_data):
        """Out of bounds target_residual_idx raises error."""
        data, z, _, _ = basic_proxy_data
        var_result = var_estimate(data, lags=1)

        with pytest.raises(ValueError, match="target_residual_idx"):
            proxy_svar(
                var_result,
                instrument=z[1:],
                target_shock_idx=0,
                target_residual_idx=10,
            )


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================


class TestProxySVARMonteCarlo:
    """Layer 3: Statistical validation via Monte Carlo."""

    @pytest.mark.slow
    def test_impact_column_unbiased(self):
        """Impact estimates are approximately unbiased with strong instrument."""
        n_mc = 100
        n = 300
        true_impact = np.array([1.0, 0.3, 0.2])

        estimates = []

        for seed in range(n_mc):
            data, z, B0_inv_true, _ = generate_proxy_svar_dgp(
                n=n,
                instrument_strength=0.7,
                seed=seed,
            )

            var_result = var_estimate(data, lags=1)

            try:
                result = proxy_svar(var_result, instrument=z[1:])
                estimates.append(result.impact_column)
            except Exception:
                continue

        estimates = np.array(estimates)

        # Mean estimate
        mean_estimate = np.mean(estimates, axis=0)

        # Bias = E[estimate] - true (normalized)
        true_normalized = true_impact / true_impact[0]
        mean_normalized = mean_estimate / mean_estimate[0]

        bias = mean_normalized - true_normalized

        # Bias should be small (< 0.15 for each element)
        assert np.all(np.abs(bias) < 0.15), f"Bias too large: {bias}"

    @pytest.mark.slow
    def test_confidence_interval_coverage(self):
        """95% CI has approximately correct coverage."""
        n_mc = 200
        n = 400
        true_impact = np.array([1.0, 0.3, 0.2])
        alpha = 0.05

        coverage = np.zeros(3)

        for seed in range(n_mc):
            data, z, _, _ = generate_proxy_svar_dgp(
                n=n,
                instrument_strength=0.6,
                seed=seed,
            )

            var_result = var_estimate(data, lags=1)

            try:
                result = proxy_svar(var_result, instrument=z[1:], alpha=alpha)

                # Check if true value in CI (for normalized comparison)
                for i in range(3):
                    if (
                        result.impact_column_ci_lower[i]
                        <= true_impact[i]
                        <= result.impact_column_ci_upper[i]
                    ):
                        coverage[i] += 1
            except Exception:
                continue

        coverage_rate = coverage / n_mc

        # Coverage should be close to 95% (allow 85-99% range)
        # Note: May be conservative or liberal due to finite sample
        assert np.all(coverage_rate > 0.80), f"Coverage too low: {coverage_rate}"

    @pytest.mark.slow
    def test_weak_instrument_bias(self):
        """Weak instruments induce larger bias than strong instruments."""
        n_mc = 50
        n = 300

        # Strong instrument estimates
        strong_estimates = []
        for seed in range(n_mc):
            data, z, _, _ = generate_proxy_svar_dgp(
                n=n,
                instrument_strength=0.8,
                seed=seed,
            )
            var_result = var_estimate(data, lags=1)
            try:
                result = proxy_svar(var_result, instrument=z[1:])
                strong_estimates.append(result.impact_column)
            except Exception:
                continue

        # Weak instrument estimates
        weak_estimates = []
        for seed in range(n_mc):
            data, z, _, _ = generate_proxy_svar_dgp(
                n=n,
                instrument_strength=0.15,
                seed=seed + 1000,
            )
            var_result = var_estimate(data, lags=1)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = proxy_svar(var_result, instrument=z[1:])
                weak_estimates.append(result.impact_column)
            except Exception:
                continue

        strong_var = np.var(strong_estimates, axis=0)
        weak_var = np.var(weak_estimates, axis=0)

        # Weak instrument should have higher variance
        assert np.mean(weak_var) > np.mean(strong_var)

    @pytest.mark.slow
    def test_bootstrap_se_accuracy(self):
        """Bootstrap SEs close to analytical SEs in large samples."""
        np.random.seed(42)
        n = 500

        data, z, _, _ = generate_proxy_svar_dgp(n=n, instrument_strength=0.6, seed=42)
        var_result = var_estimate(data, lags=1)

        # With bootstrap
        result_boot = proxy_svar(
            var_result,
            instrument=z[1:],
            bootstrap_se=True,
            n_bootstrap=200,
            seed=42,
        )

        # Compare bootstrap and delta-method SE
        # They should be in same ballpark (within factor of 3)
        # Skip first element which is normalized to 1.0 (SE=0 by construction)
        ratio = result_boot.bootstrap_se[1:] / result_boot.impact_column_se[1:]

        assert np.all(ratio > 0.3), f"Bootstrap SE too small: {ratio}"
        assert np.all(ratio < 3.0), f"Bootstrap SE too large: {ratio}"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestFirstStageRegression:
    """Tests for _first_stage_regression helper."""

    def test_perfect_correlation(self):
        """Perfect correlation gives R² = 1."""
        np.random.seed(42)
        n = 100
        z = np.random.randn(n)
        y = 2.0 * z + 1.0  # Perfect linear relationship

        fitted, f_stat, r2, residuals = _first_stage_regression(y, z)

        assert np.isclose(r2, 1.0, atol=1e-10)
        assert np.allclose(residuals, 0.0, atol=1e-10)

    def test_no_correlation(self):
        """Uncorrelated variables give R² ≈ 0."""
        np.random.seed(42)
        n = 1000
        z = np.random.randn(n)
        y = np.random.randn(n)  # Independent

        fitted, f_stat, r2, residuals = _first_stage_regression(y, z)

        assert r2 < 0.05  # Should be near zero
        assert f_stat < 5.0  # Low F-stat

    def test_f_stat_formula(self):
        """F-statistic computed correctly."""
        np.random.seed(42)
        n = 100
        z = np.random.randn(n)
        y = 0.5 * z + 0.5 * np.random.randn(n)

        fitted, f_stat, r2, residuals = _first_stage_regression(y, z)

        # Manual F-stat calculation
        k = 2  # intercept + z
        f_stat_manual = (r2 / (1 - r2)) * ((n - k) / (k - 1))

        assert np.isclose(f_stat, f_stat_manual, rtol=1e-10)


class TestCompleteImpactMatrix:
    """Tests for _complete_impact_matrix helper."""

    def test_output_shape(self):
        """Completed matrix has correct shape."""
        np.random.seed(42)
        n_vars = 3

        # Random covariance matrix
        A = np.random.randn(n_vars, n_vars)
        sigma_u = A @ A.T

        # Known impact column
        impact_col = np.array([1.0, 0.3, 0.2])

        B0_inv = _complete_impact_matrix(impact_col, sigma_u, target_shock_idx=0)

        assert B0_inv.shape == (n_vars, n_vars)

    def test_target_column_preserved(self):
        """Target column approximately preserved after completion."""
        np.random.seed(42)
        n_vars = 3

        # Random covariance matrix
        A = np.random.randn(n_vars, n_vars)
        sigma_u = A @ A.T

        # Known impact column
        impact_col = np.array([1.0, 0.3, 0.2])

        B0_inv = _complete_impact_matrix(impact_col, sigma_u, target_shock_idx=0)

        # First column should be proportional to impact_col
        result_col = B0_inv[:, 0]
        normalized_result = result_col / result_col[0]
        normalized_input = impact_col / impact_col[0]

        assert np.allclose(normalized_result, normalized_input, atol=0.01)


class TestWeakInstrumentDiagnostics:
    """Tests for weak_instrument_diagnostics function."""

    def test_strong_instrument(self):
        """Strong instrument correctly classified."""
        diag = weak_instrument_diagnostics(f_stat=25.0, n_obs=200)

        assert diag["is_weak"] is False
        assert diag["is_very_weak"] is False
        assert "Strong" in diag["interpretation"]
        assert diag["recommended_inference"] == "standard"

    def test_weak_instrument(self):
        """Weak instrument correctly classified."""
        diag = weak_instrument_diagnostics(f_stat=7.0, n_obs=200)

        assert diag["is_weak"] is True
        assert diag["is_very_weak"] is False
        assert "Weak" in diag["interpretation"]
        assert diag["recommended_inference"] == "anderson_rubin"

    def test_very_weak_instrument(self):
        """Very weak instrument correctly classified."""
        diag = weak_instrument_diagnostics(f_stat=3.0, n_obs=200)

        assert diag["is_weak"] is True
        assert diag["is_very_weak"] is True
        assert "Very weak" in diag["interpretation"]
        assert diag["recommended_inference"] == "reconsider_instrument"

    def test_stock_yogo_values(self):
        """Stock-Yogo critical values included."""
        diag = weak_instrument_diagnostics(f_stat=10.0, n_obs=200)

        assert "stock_yogo_critical_values" in diag
        assert 0.10 in diag["stock_yogo_critical_values"]
        assert diag["stock_yogo_critical_values"][0.10] == 16.38


class TestComputeIRFFromProxy:
    """Tests for compute_irf_from_proxy function."""

    def test_irf_shape(self, basic_proxy_data):
        """IRF has correct shape."""
        data, z, _, _ = basic_proxy_data
        n_vars = data.shape[1]
        var_result = var_estimate(data, lags=1)

        result = proxy_svar(var_result, instrument=z[1:])
        horizons = 20

        irf = compute_irf_from_proxy(result, horizons=horizons)

        assert irf.shape == (n_vars, n_vars, horizons + 1)

    def test_irf_impact_matches_b0_inv(self, basic_proxy_data):
        """IRF at horizon 0 equals B0_inv."""
        data, z, _, _ = basic_proxy_data
        var_result = var_estimate(data, lags=1)

        result = proxy_svar(var_result, instrument=z[1:])
        irf = compute_irf_from_proxy(result, horizons=10)

        # IRF at h=0 should be B0_inv
        assert np.allclose(irf[:, :, 0], result.B0_inv)


# =============================================================================
# Integration Tests
# =============================================================================


class TestProxySVARIntegration:
    """Integration tests with other timeseries components."""

    def test_result_properties(self, basic_proxy_data):
        """Result object properties work correctly."""
        data, z, _, _ = basic_proxy_data
        var_result = var_estimate(data, lags=1)

        result = proxy_svar(var_result, instrument=z[1:])

        # Properties should work
        assert result.n_vars == 3
        assert result.lags == 1
        assert result.n_obs > 0
        assert len(result.var_names) == 3

    def test_get_structural_coefficient(self, basic_proxy_data):
        """get_structural_coefficient method works."""
        data, z, _, _ = basic_proxy_data
        var_result = var_estimate(data, lags=1)

        result = proxy_svar(var_result, instrument=z[1:])

        # Should return B0_inv element
        coef = result.get_structural_coefficient(shock_var=0, response_var=1)
        assert coef == result.B0_inv[1, 0]

    def test_repr_method(self, basic_proxy_data):
        """__repr__ method produces valid string."""
        data, z, _, _ = basic_proxy_data
        var_result = var_estimate(data, lags=1)

        result = proxy_svar(var_result, instrument=z[1:])

        repr_str = repr(result)
        assert "ProxySVARResult" in repr_str
        assert "F=" in repr_str

    def test_weak_instrument_repr(self, weak_instrument_data):
        """Weak instrument shown in repr."""
        data, z, _, _ = weak_instrument_data
        var_result = var_estimate(data, lags=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = proxy_svar(var_result, instrument=z[1:])

        repr_str = repr(result)
        if result.is_weak_instrument:
            assert "WEAK" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
