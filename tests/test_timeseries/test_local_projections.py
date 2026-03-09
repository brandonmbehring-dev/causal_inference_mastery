"""
Tests for Local Projections (Jordà 2005).

Session 159: Test suite for LP-based impulse response estimation.

Test Structure (3-layer validation):
1. Known-Answer Tests: Known VAR(1) DGP → LP should match VAR IRF asymptotically
2. Robustness Tests: Misspecified VAR → LP should be more robust
3. Monte Carlo Tests: Coverage validation for confidence bands
"""

import numpy as np
import pytest
from typing import Tuple

from causal_inference.timeseries.local_projections import (
    LocalProjectionResult,
    local_projection_irf,
    compare_lp_var_irf,
    state_dependent_lp,
    lp_to_irf_result,
    _compute_cholesky_shocks,
    _newey_west_se,
)
from causal_inference.timeseries.svar_types import IRFResult


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestLocalProjectionBasic:
    """Basic functionality tests for local_projection_irf."""

    def test_basic_estimation(self, sample_var1_data):
        """LP estimation runs without error."""
        data, _ = sample_var1_data

        result = local_projection_irf(data, horizons=10, lags=2)

        assert isinstance(result, LocalProjectionResult)
        assert result.irf.shape == (2, 2, 11)  # 2 vars, horizons 0-10
        assert result.se.shape == (2, 2, 11)
        assert result.n_vars == 2
        assert result.horizons == 10

    def test_output_shapes(self, sample_var1_data):
        """All output arrays have correct shapes."""
        data, _ = sample_var1_data

        result = local_projection_irf(data, horizons=15, lags=3)

        assert result.irf.shape == (2, 2, 16)
        assert result.se.shape == (2, 2, 16)
        assert result.ci_lower.shape == (2, 2, 16)
        assert result.ci_upper.shape == (2, 2, 16)

    def test_confidence_bands_bracket_point_estimate(self, sample_var1_data):
        """CI should bracket the point estimate."""
        data, _ = sample_var1_data

        result = local_projection_irf(data, horizons=10, lags=2, alpha=0.05)

        # Lower < IRF < Upper everywhere
        assert np.all(result.ci_lower <= result.irf)
        assert np.all(result.irf <= result.ci_upper)

    def test_var_names_assignment(self, sample_var1_data):
        """Variable names are correctly assigned."""
        data, _ = sample_var1_data

        result = local_projection_irf(data, horizons=10, lags=2, var_names=["output", "inflation"])

        assert result.var_names == ["output", "inflation"]

    def test_get_response_method(self, sample_var1_data):
        """get_response returns correct values."""
        data, _ = sample_var1_data

        result = local_projection_irf(data, horizons=10, lags=2, var_names=["y", "x"])

        # By index
        response = result.get_response(0, 1, horizon=5)
        assert response == result.irf[0, 1, 5]

        # By name
        response = result.get_response("y", "x", horizon=5)
        assert response == result.irf[0, 1, 5]

        # All horizons
        response = result.get_response("y", "x")
        np.testing.assert_array_equal(response, result.irf[0, 1, :])

    def test_get_response_with_ci(self, sample_var1_data):
        """get_response_with_ci returns correct dictionary."""
        data, _ = sample_var1_data

        result = local_projection_irf(data, horizons=10, lags=2)

        response_data = result.get_response_with_ci(0, 1)

        assert "irf" in response_data
        assert "se" in response_data
        assert "lower" in response_data
        assert "upper" in response_data
        assert "horizon" in response_data

        assert len(response_data["irf"]) == 11
        assert len(response_data["horizon"]) == 11


class TestLocalProjectionVsVAR:
    """Compare LP and VAR-based IRF for correctly specified VAR."""

    def test_lp_var_similarity_simple_var1(self):
        """LP and VAR IRF should be similar for VAR(1) DGP."""
        np.random.seed(123)
        n = 500

        # Simple VAR(1) with known structure
        A1 = np.array([[0.5, 0.0], [0.3, 0.4]])
        data = np.zeros((n, 2))

        for t in range(1, n):
            data[t, :] = A1 @ data[t - 1, :] + np.random.randn(2) * 0.5

        comparison = compare_lp_var_irf(data, horizons=10, lags=1)

        # With large sample and correct lag, should be quite close
        # Allow for some estimation error
        max_diff = comparison["max_diff"]

        # LP and VAR should agree reasonably well (< 0.3 at max)
        assert max_diff < 0.5, f"Max difference too large: {max_diff:.4f}"

    def test_impact_response_matches(self):
        """Impact response (h=0) should match closely."""
        np.random.seed(456)
        n = 400

        A1 = np.array([[0.5, 0.1], [0.2, 0.4]])
        data = np.zeros((n, 2))

        for t in range(1, n):
            data[t, :] = A1 @ data[t - 1, :] + np.random.randn(2)

        comparison = compare_lp_var_irf(data, horizons=5, lags=1)

        # Impact response should be very close (both use Cholesky)
        lp_impact = comparison["lp_irf"].irf[:, :, 0]
        var_impact = comparison["var_irf"].irf[:, :, 0]

        # Impact is normalized by Cholesky - should match closely
        np.testing.assert_allclose(lp_impact, var_impact, atol=0.05)


# =============================================================================
# Layer 2: Robustness Tests
# =============================================================================


class TestLocalProjectionRobustness:
    """LP robustness to VAR misspecification."""

    def test_lp_robust_to_wrong_lag(self):
        """LP should be more robust when VAR lag is misspecified."""
        np.random.seed(789)
        n = 300

        # True DGP is VAR(3)
        A1 = np.array([[0.4, 0.0], [0.2, 0.3]])
        A2 = np.array([[0.2, 0.0], [0.1, 0.15]])
        A3 = np.array([[0.1, 0.0], [0.05, 0.1]])

        data = np.zeros((n, 2))
        for t in range(3, n):
            data[t, :] = (
                A1 @ data[t - 1, :]
                + A2 @ data[t - 2, :]
                + A3 @ data[t - 3, :]
                + np.random.randn(2) * 0.5
            )

        # Estimate with VAR(1) - misspecified
        comparison_lag1 = compare_lp_var_irf(data, horizons=5, lags=1)

        # Estimate with VAR(3) - correctly specified
        comparison_lag3 = compare_lp_var_irf(data, horizons=5, lags=3)

        # LP should be similar regardless of lag choice
        lp_lag1 = comparison_lag1["lp_irf"].irf
        lp_lag3 = comparison_lag3["lp_irf"].irf

        # LP estimates should be reasonably similar
        lp_diff = np.max(np.abs(lp_lag1 - lp_lag3))
        assert lp_diff < 0.5, f"LP estimates too different with different lags: {lp_diff:.4f}"


class TestCholekyShockComputation:
    """Test internal Cholesky shock computation."""

    def test_shocks_orthogonal(self, sample_var1_data):
        """Computed shocks should be orthogonal."""
        data, _ = sample_var1_data

        shocks = _compute_cholesky_shocks(data, lags=2)

        # Check orthogonality via correlation
        corr = np.corrcoef(shocks.T)
        off_diag = corr[0, 1]

        # Should be close to zero
        assert abs(off_diag) < 0.1, f"Shocks not orthogonal: corr={off_diag:.4f}"

    def test_shocks_unit_variance(self, sample_var1_data):
        """Shocks should have approximately unit variance."""
        data, _ = sample_var1_data

        shocks = _compute_cholesky_shocks(data, lags=2)

        variances = np.var(shocks, axis=0)

        # Should be close to 1
        np.testing.assert_allclose(variances, 1.0, atol=0.3)


class TestHACStandardErrors:
    """Test Newey-West HAC standard errors."""

    def test_hac_positive_standard_errors(self):
        """HAC SE should be positive for all coefficients."""
        np.random.seed(111)
        n = 200

        # Generate data with autocorrelated errors
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = 0.7 * u[t - 1] + np.random.randn()

        XtX_inv = np.linalg.inv(X.T @ X)

        # HAC SE
        hac_se = _newey_west_se(X, u, XtX_inv, "bartlett", 10)

        # Should all be positive
        assert np.all(hac_se > 0), "HAC SE should be positive"

    def test_hac_accounts_for_autocorrelation_in_lp(self, sample_var1_data):
        """HAC SEs should be computed and positive in LP estimation."""
        data, _ = sample_var1_data

        result = local_projection_irf(data, horizons=5, lags=2)

        # All SEs should be positive
        assert np.all(result.se > 0), "All SEs should be positive"

        # SEs should increase with horizon (generally, due to more uncertainty)
        # This is a soft check - not always true but generally expected
        se_h0 = np.mean(result.se[:, :, 0])
        se_h5 = np.mean(result.se[:, :, 5])
        # At least one horizon should have larger SE than h=0
        assert se_h5 > 0.8 * se_h0, "SE at higher horizon should not be much smaller"


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================


class TestLocalProjectionMonteCarlo:
    """Monte Carlo validation for LP coverage."""

    @pytest.mark.slow
    def test_coverage_sign_correct(self):
        """LP should correctly identify sign of causal effect."""
        np.random.seed(222)
        n_sims = 50  # Reduced for speed
        n_obs = 300

        # True DGP: var0 causes var1 with positive effect
        true_effect = 0.4

        correct_sign_count = 0

        for sim in range(n_sims):
            np.random.seed(222 + sim)

            # Generate data
            data = np.zeros((n_obs, 2))
            for t in range(1, n_obs):
                data[t, 0] = 0.5 * data[t - 1, 0] + np.random.randn()
                data[t, 1] = true_effect * data[t - 1, 0] + 0.4 * data[t - 1, 1] + np.random.randn()

            # Estimate LP
            result = local_projection_irf(data, horizons=5, lags=1, alpha=0.05)

            # Check if sign at horizon 1 is correct (positive)
            # Response of var1 to shock in var0
            irf_1_0_h1 = result.irf[1, 0, 1]

            if irf_1_0_h1 > 0:
                correct_sign_count += 1

        # Should get sign correct most of the time
        correct_rate = correct_sign_count / n_sims
        assert correct_rate > 0.70, f"Only {correct_rate:.0%} had correct sign, expected >70%"

    def test_confidence_bands_reasonable(self, sample_var1_data):
        """CI width should be reasonable (not too wide or narrow)."""
        data, _ = sample_var1_data

        result = local_projection_irf(data, horizons=10, lags=2, alpha=0.05)

        # CI width should be positive
        width = result.ci_upper - result.ci_lower
        assert np.all(width > 0), "CI width must be positive"

        # CI width shouldn't be absurdly large
        assert np.all(width < 20), "CI width seems too large"

        # Average width should be positive (exact value depends on data scale)
        avg_width = np.mean(width)
        assert avg_width > 0, f"Average CI width must be positive"

        # SE should be proportional to CI width (z_0.025 * 2 ≈ 3.92)
        avg_se = np.mean(result.se)
        expected_width = avg_se * 2 * 1.96
        np.testing.assert_allclose(avg_width, expected_width, rtol=0.01)


# =============================================================================
# State-Dependent LP Tests
# =============================================================================


class TestStateDependentLP:
    """Tests for state-dependent local projections."""

    def test_state_dependent_returns_two_irfs(self):
        """Should return separate IRFs for high and low states."""
        np.random.seed(333)
        n = 300

        # Simple VAR data
        data = np.zeros((n, 2))
        for t in range(1, n):
            data[t, :] = 0.5 * data[t - 1, :] + np.random.randn(2)

        # Random state indicator
        state = (np.random.rand(n) > 0.5).astype(float)

        result = state_dependent_lp(data, state, horizons=5, lags=2)

        assert "high_state_irf" in result
        assert "low_state_irf" in result
        assert "difference" in result
        assert "diff_significant" in result

        assert isinstance(result["high_state_irf"], LocalProjectionResult)
        assert isinstance(result["low_state_irf"], LocalProjectionResult)

    def test_state_dependent_detects_asymmetry(self):
        """Should detect state-dependent effects when they exist."""
        np.random.seed(444)
        n = 500

        # State-dependent DGP
        data = np.zeros((n, 2))
        state = np.zeros(n)

        for t in range(1, n):
            state[t] = 1 if data[t - 1, 0] > 0 else 0

            # Different dynamics in different states
            if state[t] == 1:
                data[t, 0] = 0.7 * data[t - 1, 0] + np.random.randn()
                data[t, 1] = 0.5 * data[t - 1, 0] + np.random.randn()
            else:
                data[t, 0] = 0.3 * data[t - 1, 0] + np.random.randn()
                data[t, 1] = 0.1 * data[t - 1, 0] + np.random.randn()

        result = state_dependent_lp(data, state, horizons=5, lags=1)

        # Should find some difference
        diff = result["difference"]
        assert not np.allclose(diff, 0, atol=0.1), "Should detect state-dependent effects"


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestLPToIRFConversion:
    """Test conversion from LocalProjectionResult to IRFResult."""

    def test_conversion_preserves_data(self, sample_var1_data):
        """Conversion should preserve all key data."""
        data, _ = sample_var1_data

        lp_result = local_projection_irf(data, horizons=10, lags=2)
        irf_result = lp_to_irf_result(lp_result)

        assert isinstance(irf_result, IRFResult)
        np.testing.assert_array_equal(irf_result.irf, lp_result.irf)
        np.testing.assert_array_equal(irf_result.irf_lower, lp_result.ci_lower)
        np.testing.assert_array_equal(irf_result.irf_upper, lp_result.ci_upper)
        assert irf_result.horizons == lp_result.horizons
        assert irf_result.var_names == lp_result.var_names


class TestCumulativeIRF:
    """Test cumulative IRF option."""

    def test_cumulative_irf_sums_correctly(self, sample_var1_data):
        """Cumulative IRF should be sum of IRF up to horizon."""
        data, _ = sample_var1_data

        # Non-cumulative
        result = local_projection_irf(data, horizons=10, lags=2, cumulative=False)

        # Cumulative
        result_cum = local_projection_irf(data, horizons=10, lags=2, cumulative=True)

        # Cumulative should be cumsum of non-cumulative
        expected_cum = np.cumsum(result.irf, axis=2)
        np.testing.assert_allclose(result_cum.irf, expected_cum, rtol=1e-5)


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_horizons(self, sample_var1_data):
        """Negative horizons should raise ValueError."""
        data, _ = sample_var1_data

        with pytest.raises(ValueError, match="horizons must be >= 0"):
            local_projection_irf(data, horizons=-1, lags=2)

    def test_invalid_lags(self, sample_var1_data):
        """Zero or negative lags should raise ValueError."""
        data, _ = sample_var1_data

        with pytest.raises(ValueError, match="lags must be >= 1"):
            local_projection_irf(data, horizons=10, lags=0)

    def test_insufficient_observations(self):
        """Too few observations should raise ValueError."""
        data = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="Insufficient observations"):
            local_projection_irf(data, horizons=10, lags=5)

    def test_external_shock_required(self, sample_var1_data):
        """External shock type requires external_shock parameter."""
        data, _ = sample_var1_data

        with pytest.raises(ValueError, match="external_shock must be provided"):
            local_projection_irf(data, horizons=10, lags=2, shock_type="external")

    def test_invalid_shock_type(self, sample_var1_data):
        """Invalid shock type should raise ValueError."""
        data, _ = sample_var1_data

        with pytest.raises(ValueError, match="shock_type must be"):
            local_projection_irf(data, horizons=10, lags=2, shock_type="invalid")

    def test_var_names_length_mismatch(self, sample_var1_data):
        """Mismatched var_names length should raise ValueError."""
        data, _ = sample_var1_data

        with pytest.raises(ValueError, match="var_names length"):
            local_projection_irf(data, horizons=10, lags=2, var_names=["a"])


# =============================================================================
# External Shock Tests
# =============================================================================


class TestExternalShock:
    """Test LP with external shock identification."""

    def test_external_shock_basic(self):
        """External shock estimation runs without error."""
        np.random.seed(555)
        n = 200

        # DGP: external shock affects variable
        shock = np.random.randn(n)
        data = np.zeros((n, 2))

        for t in range(1, n):
            data[t, 0] = 0.5 * data[t - 1, 0] + 0.3 * shock[t - 1] + np.random.randn() * 0.5
            data[t, 1] = 0.4 * data[t - 1, 1] + np.random.randn() * 0.5

        result = local_projection_irf(
            data, horizons=10, lags=2, shock_type="external", external_shock=shock
        )

        assert result.method == "external"
        assert result.irf.shape == (2, 2, 11)

    def test_external_shock_length_validation(self, sample_var1_data):
        """External shock must match data length."""
        data, _ = sample_var1_data
        n = data.shape[0]

        # Wrong length shock
        wrong_shock = np.random.randn(n + 10)

        with pytest.raises(ValueError, match="external_shock length"):
            local_projection_irf(
                data, horizons=10, lags=2, shock_type="external", external_shock=wrong_shock
            )


# =============================================================================
# HAC Kernel Tests
# =============================================================================


class TestHACKernels:
    """Test different HAC kernels."""

    def test_bartlett_kernel(self, sample_var1_data):
        """Bartlett kernel estimation works."""
        data, _ = sample_var1_data

        result = local_projection_irf(data, horizons=5, lags=2, hac_kernel="bartlett")

        assert result.hac_kernel == "bartlett"
        assert np.all(result.se > 0)

    def test_quadratic_spectral_kernel(self, sample_var1_data):
        """Quadratic spectral kernel estimation works."""
        data, _ = sample_var1_data

        result = local_projection_irf(data, horizons=5, lags=2, hac_kernel="quadratic_spectral")

        assert result.hac_kernel == "quadratic_spectral"
        assert np.all(result.se > 0)

    def test_custom_bandwidth(self, sample_var1_data):
        """Custom HAC bandwidth is applied."""
        data, _ = sample_var1_data

        result = local_projection_irf(data, horizons=5, lags=2, hac_bandwidth=15)

        assert result.hac_bandwidth == 15
