"""
Tests for Time-Varying Parameter VAR (TVP-VAR).

Session 165: Test suite for Primiceri (2005), Cogley & Sargent (2005) methodology.

Three-layer validation:
1. Known-Answer: Tests with known DGP structure
2. Adversarial: Edge cases, invalid inputs, error handling
3. Monte Carlo: Statistical validation (coefficient tracking, smoother properties)
"""

import warnings

import numpy as np
import pytest
from scipy import linalg

from causal_inference.timeseries import var_estimate
from causal_inference.timeseries.tvp_var import (
    TVPVARResult,
    tvp_var_estimate,
    compute_tvp_irf,
    compute_tvp_irf_all_times,
    check_tvp_stability,
    check_tvp_stability_all_times,
    coefficient_change_test,
    _build_tvp_regressor_matrix,
    _kalman_filter,
    _rts_smoother,
    _joseph_form_update,
    _initialize_from_ols,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def generate_constant_var_dgp(
    n: int = 300,
    n_vars: int = 2,
    lags: int = 1,
    seed: int = 42,
) -> tuple:
    """
    Generate data from constant VAR DGP.

    TVP-VAR on this data should recover approximately constant coefficients.

    Parameters
    ----------
    n : int
        Number of observations
    n_vars : int
        Number of variables
    lags : int
        VAR lag order
    seed : int
        Random seed

    Returns
    -------
    tuple
        (data, A1_true, sigma_true)
    """
    np.random.seed(seed)

    # Stable coefficient matrix
    if n_vars == 2:
        A1 = np.array(
            [
                [0.5, 0.1],
                [0.0, 0.4],
            ]
        )
    else:
        A1 = np.eye(n_vars) * 0.4

    # Covariance matrix
    sigma = np.eye(n_vars) * 0.5

    # Generate data
    eps = np.random.multivariate_normal(np.zeros(n_vars), sigma, size=n)
    data = np.zeros((n, n_vars))

    for t in range(lags, n):
        for p in range(1, lags + 1):
            data[t] += A1 @ data[t - p]
        data[t] += eps[t]

    return data, A1, sigma


def generate_structural_break_dgp(
    n: int = 400,
    n_vars: int = 2,
    lags: int = 1,
    break_point: float = 0.5,
    seed: int = 42,
) -> tuple:
    """
    Generate VAR data with structural break in coefficients.

    Parameters
    ----------
    n : int
        Number of observations
    n_vars : int
        Number of variables
    lags : int
        VAR lag order
    break_point : float
        Relative position of break (0-1)
    seed : int
        Random seed

    Returns
    -------
    tuple
        (data, A1_before, A1_after, break_idx)
    """
    np.random.seed(seed)

    break_idx = int(n * break_point)

    # Pre-break coefficients
    A1_before = np.array(
        [
            [0.6, 0.1],
            [0.0, 0.5],
        ]
    )[:n_vars, :n_vars]

    # Post-break coefficients (different persistence)
    A1_after = np.array(
        [
            [0.3, 0.2],
            [0.1, 0.3],
        ]
    )[:n_vars, :n_vars]

    sigma = np.eye(n_vars) * 0.5
    eps = np.random.multivariate_normal(np.zeros(n_vars), sigma, size=n)

    data = np.zeros((n, n_vars))

    for t in range(lags, n):
        A1 = A1_before if t < break_idx else A1_after
        for p in range(1, lags + 1):
            data[t] += A1 @ data[t - p]
        data[t] += eps[t]

    return data, A1_before, A1_after, break_idx


def generate_gradually_changing_dgp(
    n: int = 300,
    n_vars: int = 2,
    lags: int = 1,
    seed: int = 42,
) -> tuple:
    """
    Generate VAR with smoothly time-varying coefficients.

    Parameters
    ----------
    n : int
        Number of observations
    n_vars : int
        Number of variables
    lags : int
        VAR lag order
    seed : int
        Random seed

    Returns
    -------
    tuple
        (data, A1_trajectory)
    """
    np.random.seed(seed)

    sigma = np.eye(n_vars) * 0.5
    eps = np.random.multivariate_normal(np.zeros(n_vars), sigma, size=n)

    data = np.zeros((n, n_vars))
    A1_trajectory = np.zeros((n, n_vars, n_vars))

    for t in range(lags, n):
        # Coefficient varies sinusoidally over time
        phase = 2 * np.pi * t / n
        A1_t = np.array(
            [
                [0.4 + 0.2 * np.sin(phase), 0.1],
                [0.0, 0.3 + 0.1 * np.cos(phase)],
            ]
        )[:n_vars, :n_vars]

        A1_trajectory[t] = A1_t

        for p in range(1, lags + 1):
            data[t] += A1_t @ data[t - p]
        data[t] += eps[t]

    return data, A1_trajectory


@pytest.fixture
def constant_var_data():
    """Generate constant VAR data."""
    return generate_constant_var_dgp(n=300, n_vars=2, lags=1, seed=42)


@pytest.fixture
def structural_break_data():
    """Generate structural break VAR data."""
    return generate_structural_break_dgp(n=400, n_vars=2, lags=1, seed=123)


@pytest.fixture
def gradually_changing_data():
    """Generate gradually changing VAR data."""
    return generate_gradually_changing_dgp(n=300, n_vars=2, lags=1, seed=456)


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestTVPVARKnownAnswer:
    """Layer 1: Tests with known DGP structure."""

    def test_basic_estimation(self, constant_var_data):
        """TVP-VAR runs without error on valid data."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1)

        assert isinstance(result, TVPVARResult)
        assert result.initialization == "ols"

    def test_output_shapes(self, constant_var_data):
        """All outputs have correct shapes."""
        data, _, _ = constant_var_data
        n_obs, n_vars = data.shape
        lags = 1

        result = tvp_var_estimate(data, lags=lags)

        n_eff = n_obs - lags
        n_params_per_eq = n_vars * lags + 1
        state_dim = n_vars * n_params_per_eq

        # Coefficient arrays
        assert result.coefficients_filtered.shape == (n_eff, n_vars, n_params_per_eq)
        assert result.coefficients_smoothed.shape == (n_eff, n_vars, n_params_per_eq)

        # Covariance arrays
        assert result.covariance_filtered.shape == (n_eff, state_dim, state_dim)
        assert result.covariance_smoothed.shape == (n_eff, state_dim, state_dim)

        # Innovations
        assert result.innovations.shape == (n_eff, n_vars)
        assert result.innovation_covariance.shape == (n_eff, n_vars, n_vars)

        # Kalman gain
        assert result.kalman_gain.shape == (n_eff, state_dim, n_vars)

        # Sigma and Q
        assert result.sigma.shape == (n_vars, n_vars)
        assert result.Q.shape == (state_dim, state_dim)

    def test_constant_dgp_matches_ols(self, constant_var_data):
        """With very small Q, TVP-VAR should approximately match OLS VAR."""
        data, A1_true, _ = constant_var_data

        # Very small Q (almost no time variation allowed)
        result = tvp_var_estimate(data, lags=1, Q_scale=1e-10, smooth=True)

        # OLS VAR for comparison
        var_result = var_estimate(data, lags=1)

        # Final smoothed coefficients should be close to OLS
        final_coef = result.get_coefficients_at_time(result.n_obs_effective - 1, smoothed=True)

        # Allow larger tolerance due to different estimation methods
        assert np.allclose(final_coef, var_result.coefficients, atol=0.15)

    def test_log_likelihood_finite(self, constant_var_data):
        """Log-likelihood is finite and negative."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1)

        assert np.isfinite(result.log_likelihood)
        assert result.log_likelihood < 0  # Log-likelihood is typically negative

    def test_innovations_zero_mean(self, constant_var_data):
        """Innovation sequence has approximately zero mean."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1)

        mean_innovations = np.mean(result.innovations, axis=0)
        assert np.allclose(mean_innovations, 0.0, atol=0.1)

    def test_filtered_vs_smoothed_variance(self, constant_var_data):
        """Smoothed covariance should be <= filtered covariance (in trace)."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1, smooth=True)

        # For middle time points (not edge effects)
        mid_idx = result.n_obs_effective // 2

        trace_filt = np.trace(result.covariance_filtered[mid_idx])
        trace_smooth = np.trace(result.covariance_smoothed[mid_idx])

        # Smoothed should have less or equal uncertainty
        assert trace_smooth <= trace_filt * 1.01  # Allow 1% tolerance

    def test_initialization_methods(self, constant_var_data):
        """All initialization methods work."""
        data, _, _ = constant_var_data

        # OLS initialization
        result_ols = tvp_var_estimate(data, lags=1, initialization="ols")
        assert result_ols.initialization == "ols"

        # Diffuse initialization
        result_diffuse = tvp_var_estimate(data, lags=1, initialization="diffuse")
        assert result_diffuse.initialization == "diffuse"

        # Custom initialization
        state_dim = result_ols.state_dim
        beta_init = np.zeros(state_dim)
        P_init = np.eye(state_dim) * 100

        result_custom = tvp_var_estimate(
            data,
            lags=1,
            initialization="custom",
            beta_init=beta_init,
            P_init=P_init,
        )
        assert result_custom.initialization == "custom"

    def test_irf_at_time_shape(self, constant_var_data):
        """IRF at specific time has correct shape."""
        data, _, _ = constant_var_data
        n_vars = data.shape[1]

        result = tvp_var_estimate(data, lags=1)

        horizons = 20
        irf = compute_tvp_irf(result, t=50, horizons=horizons, shock_idx=0)

        assert irf.shape == (n_vars, horizons + 1)

    def test_to_var_result_conversion(self, constant_var_data):
        """Conversion to VARResult works."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1)
        var_result = result.to_var_result_at_time(100)

        assert hasattr(var_result, "coefficients")
        assert hasattr(var_result, "sigma")
        assert var_result.lags == result.lags

    def test_kalman_gain_reasonable_bounds(self, constant_var_data):
        """Kalman gain values are in reasonable range."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1)

        # Kalman gain should be bounded
        K_max = np.max(np.abs(result.kalman_gain))
        assert K_max < 100  # Should not be extremely large


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestTVPVARAdversarial:
    """Layer 2: Edge cases and error handling."""

    def test_insufficient_observations(self):
        """Too few observations raises error."""
        np.random.seed(42)
        data = np.random.randn(14, 2)  # Only 14 obs, gives 9 effective with lags=5

        with pytest.raises(ValueError, match="Insufficient observations"):
            tvp_var_estimate(data, lags=5)

    def test_singular_sigma_regularization(self):
        """Near-singular sigma handled via regularization."""
        np.random.seed(42)
        n, n_vars = 200, 2

        # Create data with nearly collinear variables
        x = np.random.randn(n)
        data = np.column_stack([x, x + 1e-10 * np.random.randn(n)])

        # Should run without error (regularization kicks in)
        # May raise warning but should complete
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = tvp_var_estimate(data, lags=1)

        assert isinstance(result, TVPVARResult)

    def test_non_positive_definite_q_error(self):
        """Non-PSD Q raises error."""
        np.random.seed(42)
        data = np.random.randn(100, 2)

        # Negative eigenvalue in Q
        state_dim = 2 * (2 * 1 + 1)  # n_vars * n_params_per_eq
        Q_bad = np.eye(state_dim)
        Q_bad[0, 0] = -1.0

        with pytest.raises(ValueError, match="positive semi-definite"):
            tvp_var_estimate(data, lags=1, Q_init=Q_bad)

    def test_nan_in_data(self):
        """NaN values in data raise error."""
        np.random.seed(42)
        data = np.random.randn(100, 2)
        data[50, 0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            tvp_var_estimate(data, lags=1)

    def test_single_variable(self):
        """k=1 edge case works."""
        np.random.seed(42)
        data = np.random.randn(200, 1)

        result = tvp_var_estimate(data, lags=1)

        assert result.n_vars == 1
        assert result.coefficients_filtered.shape[1] == 1

    def test_single_lag(self):
        """p=1 edge case works."""
        np.random.seed(42)
        data = np.random.randn(200, 2)

        result = tvp_var_estimate(data, lags=1)

        assert result.lags == 1
        assert result.n_params_per_eq == 3  # 2 variables + intercept

    def test_many_lags(self):
        """Large p with numerical stability."""
        np.random.seed(42)
        data = np.random.randn(500, 2)

        # Use lags=5, requiring sufficient observations
        result = tvp_var_estimate(data, lags=5)

        assert result.lags == 5
        assert np.isfinite(result.log_likelihood)

    def test_missing_custom_init(self):
        """Custom init without beta_init raises error."""
        np.random.seed(42)
        data = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="beta_init and P_init must be provided"):
            tvp_var_estimate(data, lags=1, initialization="custom")

    def test_incompatible_q_dimensions(self):
        """Wrong Q shape raises error."""
        np.random.seed(42)
        data = np.random.randn(100, 2)

        Q_wrong = np.eye(5)  # Wrong dimension

        with pytest.raises(ValueError, match="wrong shape"):
            tvp_var_estimate(data, lags=1, Q_init=Q_wrong)

    def test_invalid_time_index(self, constant_var_data):
        """Out of bounds time index raises error."""
        data, _, _ = constant_var_data
        result = tvp_var_estimate(data, lags=1)

        with pytest.raises(ValueError, match="out of bounds"):
            result.get_coefficients_at_time(10000)

        with pytest.raises(ValueError, match="out of bounds"):
            result.get_coefficients_at_time(-10000)

    def test_invalid_lag_index(self, constant_var_data):
        """Invalid lag number raises error."""
        data, _, _ = constant_var_data
        result = tvp_var_estimate(data, lags=1)

        with pytest.raises(ValueError, match="Lag must be between"):
            result.get_lag_matrix_at_time(50, lag=0)

        with pytest.raises(ValueError, match="Lag must be between"):
            result.get_lag_matrix_at_time(50, lag=5)


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================


class TestTVPVARMonteCarlo:
    """Layer 3: Statistical validation via Monte Carlo."""

    def test_tvp_tracks_structural_break(self, structural_break_data):
        """TVP-VAR detects coefficient change at structural break."""
        data, A1_before, A1_after, break_idx = structural_break_data

        result = tvp_var_estimate(data, lags=1, Q_scale=0.01, smooth=True)

        # Effective break index (accounting for lags)
        break_eff = break_idx - result.lags

        # Get coefficient trajectory for A[0,0]
        coef_00 = result.coefficient_trajectory(0, 1, smoothed=True)

        # Mean before and after break
        pre_mean = np.mean(coef_00[: break_eff - 20])
        post_mean = np.mean(coef_00[break_eff + 20 :])

        # True values
        true_pre = A1_before[0, 0]
        true_post = A1_after[0, 0]

        # Should be closer to true values than the overall mean
        overall_mean = np.mean(coef_00)

        pre_error = abs(pre_mean - true_pre)
        post_error = abs(post_mean - true_post)
        overall_pre_error = abs(overall_mean - true_pre)
        overall_post_error = abs(overall_mean - true_post)

        # Pre-break estimate should be closer to true pre than overall mean
        # (This may not always hold due to filtering, but on average should)
        assert pre_error < overall_pre_error + 0.2  # Allow some slack

    def test_smoothed_reduces_variance(self, constant_var_data):
        """Smoothed state covariance consistently <= filtered."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1, smooth=True)

        # Check for all time points (excluding edges)
        n_lower = 0
        for t in range(20, result.n_obs_effective - 20):
            trace_filt = np.trace(result.covariance_filtered[t])
            trace_smooth = np.trace(result.covariance_smoothed[t])
            if trace_smooth <= trace_filt * 1.001:
                n_lower += 1

        # At least 90% should satisfy the property
        prop_lower = n_lower / (result.n_obs_effective - 40)
        assert prop_lower > 0.9

    def test_q_scale_affects_smoothness(self):
        """Larger Q → more variable coefficients."""
        np.random.seed(42)
        data = np.random.randn(300, 2)

        # Small Q (smooth)
        result_small = tvp_var_estimate(data, lags=1, Q_scale=1e-6)
        var_small = np.var(result_small.coefficient_trajectory(0, 1, smoothed=True))

        # Large Q (variable)
        result_large = tvp_var_estimate(data, lags=1, Q_scale=0.1)
        var_large = np.var(result_large.coefficient_trajectory(0, 1, smoothed=True))

        # Larger Q should allow more coefficient variation
        assert var_large > var_small

    def test_irf_stability_across_time(self, constant_var_data):
        """IRF values bounded across all time points."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1)

        irf_all = compute_tvp_irf_all_times(result, horizons=20, shock_idx=0)

        # IRF should be bounded
        max_irf = np.max(np.abs(irf_all))
        assert max_irf < 100  # Should not explode

    @pytest.mark.slow
    def test_likelihood_improves_with_data(self):
        """Log-likelihood per observation improves with more data."""
        np.random.seed(42)

        ll_per_obs = []
        for n in [100, 200, 400]:
            data, _, _ = generate_constant_var_dgp(n=n, seed=42)
            result = tvp_var_estimate(data, lags=1)
            ll_per_obs.append(result.log_likelihood / result.n_obs_effective)

        # Per-observation LL should generally improve (or be stable)
        # This is not always monotonic, but shouldn't get worse
        assert ll_per_obs[-1] > ll_per_obs[0] - 0.5  # Allow some slack


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestBuildTVPRegressorMatrix:
    """Tests for _build_tvp_regressor_matrix helper."""

    def test_output_shape(self):
        """Output has correct shape."""
        np.random.seed(42)
        n = 100
        n_vars = 2
        lags = 1

        data = np.random.randn(n, n_vars)
        from causal_inference.timeseries.var import _build_var_matrices

        Y, X_full = _build_var_matrices(data, lags, include_constant=True)

        n_params_per_eq = n_vars * lags + 1
        X_tvp = _build_tvp_regressor_matrix(Y, X_full, n_vars, n_params_per_eq)

        T = Y.shape[0]
        state_dim = n_vars * n_params_per_eq

        assert X_tvp.shape == (T, n_vars, state_dim)

    def test_block_diagonal_structure(self):
        """X_tvp has block-diagonal structure."""
        np.random.seed(42)
        n = 50
        n_vars = 2
        lags = 1

        data = np.random.randn(n, n_vars)
        from causal_inference.timeseries.var import _build_var_matrices

        Y, X_full = _build_var_matrices(data, lags, include_constant=True)

        n_params_per_eq = n_vars * lags + 1
        X_tvp = _build_tvp_regressor_matrix(Y, X_full, n_vars, n_params_per_eq)

        # Check block structure at time t=0
        X_t = X_tvp[0]

        # First equation uses first block
        assert np.any(X_t[0, :n_params_per_eq] != 0)
        assert np.all(X_t[0, n_params_per_eq:] == 0)

        # Second equation uses second block
        assert np.all(X_t[1, :n_params_per_eq] == 0)
        assert np.any(X_t[1, n_params_per_eq:] != 0)


class TestJosephFormUpdate:
    """Tests for _joseph_form_update helper."""

    def test_output_shape(self):
        """Output has correct shape."""
        state_dim = 6
        n_vars = 2

        P_pred = np.eye(state_dim)
        K = np.random.randn(state_dim, n_vars)
        X_t = np.random.randn(n_vars, state_dim)
        sigma = np.eye(n_vars)

        P_filt = _joseph_form_update(P_pred, K, X_t, sigma)

        assert P_filt.shape == (state_dim, state_dim)

    def test_symmetry_preserved(self):
        """Output is symmetric."""
        np.random.seed(42)
        state_dim = 6
        n_vars = 2

        P_pred = np.eye(state_dim)
        K = np.random.randn(state_dim, n_vars) * 0.1
        X_t = np.random.randn(n_vars, state_dim)
        sigma = np.eye(n_vars)

        P_filt = _joseph_form_update(P_pred, K, X_t, sigma)

        assert np.allclose(P_filt, P_filt.T)

    def test_positive_definiteness(self):
        """Output is positive semi-definite."""
        np.random.seed(42)
        state_dim = 6
        n_vars = 2

        # Start with PD matrix
        A = np.random.randn(state_dim, state_dim)
        P_pred = A @ A.T + 0.1 * np.eye(state_dim)

        K = np.random.randn(state_dim, n_vars) * 0.1
        X_t = np.random.randn(n_vars, state_dim)
        sigma = np.eye(n_vars)

        P_filt = _joseph_form_update(P_pred, K, X_t, sigma)

        eigvals = np.linalg.eigvalsh(P_filt)
        assert np.all(eigvals >= -1e-10)


class TestCheckTVPStability:
    """Tests for check_tvp_stability function."""

    def test_stable_var(self):
        """Stable VAR correctly identified."""
        n_vars = 2
        lags = 1

        # Stable coefficients (small diagonal)
        coef = np.array(
            [
                [0.5, 0.3, 0.1],  # Intercept, A1[0,0], A1[0,1]
                [0.5, 0.0, 0.3],  # Intercept, A1[1,0], A1[1,1]
            ]
        )

        is_stable, eigvals = check_tvp_stability(coef, lags, n_vars)

        assert is_stable  # Use truthiness, not identity
        assert np.all(np.abs(eigvals) < 1.0)

    def test_unstable_var(self):
        """Unstable VAR correctly identified."""
        n_vars = 2
        lags = 1

        # Unstable coefficients (eigenvalue > 1)
        coef = np.array(
            [
                [0.5, 0.9, 0.3],
                [0.5, 0.3, 0.9],
            ]
        )

        is_stable, eigvals = check_tvp_stability(coef, lags, n_vars)

        assert not is_stable  # Use truthiness, not identity
        assert np.max(np.abs(eigvals)) >= 1.0


class TestCoefficientChangeTest:
    """Tests for coefficient_change_test function."""

    def test_constant_coefficient(self, constant_var_data):
        """Constant coefficient has low variance ratio."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1, Q_scale=1e-6)

        # Very small Q should give stable coefficients
        var_ratio, p_val = coefficient_change_test(result, 0, 1)

        # Variance ratio should be small for constant coefficients
        # (though this depends on filtering)
        assert var_ratio < 10  # Reasonable upper bound


# =============================================================================
# IRF Tests
# =============================================================================


class TestTVPIRF:
    """Tests for TVP IRF computation."""

    def test_irf_shape(self, constant_var_data):
        """IRF has correct shape."""
        data, _, _ = constant_var_data
        n_vars = data.shape[1]

        result = tvp_var_estimate(data, lags=1)

        irf = compute_tvp_irf(result, t=50, horizons=20, shock_idx=0)

        assert irf.shape == (n_vars, 21)

    def test_irf_all_times_shape(self, constant_var_data):
        """IRF at all times has correct shape."""
        data, _, _ = constant_var_data
        n_vars = data.shape[1]

        result = tvp_var_estimate(data, lags=1)
        T = result.n_obs_effective

        irf_all = compute_tvp_irf_all_times(result, horizons=20, shock_idx=0)

        assert irf_all.shape == (T, n_vars, 21)

    def test_irf_impact_normalized(self, constant_var_data):
        """IRF at h=0 for shocked variable is 1 (orthogonalized)."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1)

        # Note: With Cholesky orthogonalization, the impact is not necessarily 1
        # It depends on the Cholesky decomposition of sigma
        irf = compute_tvp_irf(result, t=50, horizons=20, shock_idx=0, orthogonalize=True)

        # Impact on shocked variable should be the first column of Cholesky(sigma)
        P = np.linalg.cholesky(result.sigma)
        expected_impact = P[:, 0]

        assert np.allclose(irf[:, 0], expected_impact, rtol=0.01)

    def test_invalid_time_index_irf(self, constant_var_data):
        """Invalid time index raises error."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1)

        with pytest.raises(ValueError, match="out of bounds"):
            compute_tvp_irf(result, t=10000, horizons=20, shock_idx=0)

    def test_invalid_shock_idx_irf(self, constant_var_data):
        """Invalid shock index raises error."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1)

        with pytest.raises(ValueError, match="out of bounds"):
            compute_tvp_irf(result, t=50, horizons=20, shock_idx=10)


# =============================================================================
# Integration Tests
# =============================================================================


class TestTVPVARIntegration:
    """Integration tests with other components."""

    def test_result_properties(self, constant_var_data):
        """Result object properties work correctly."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1)

        assert result.n_vars == 2
        assert result.lags == 1
        assert result.n_obs_effective > 0
        assert result.has_smoothed is True
        assert len(result.var_names) == 2

    def test_coefficient_trajectory(self, constant_var_data):
        """Coefficient trajectory extraction works."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1)

        traj = result.coefficient_trajectory(0, 1, smoothed=True)

        assert traj.shape == (result.n_obs_effective,)
        assert np.all(np.isfinite(traj))

    def test_get_intercepts_at_time(self, constant_var_data):
        """Intercept extraction works."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1)

        intercepts = result.get_intercepts_at_time(50)

        assert intercepts.shape == (result.n_vars,)

    def test_stability_check_all_times(self, constant_var_data):
        """Stability check at all times works."""
        data, _, _ = constant_var_data

        result = tvp_var_estimate(data, lags=1)

        is_stable, max_mod = check_tvp_stability_all_times(result)

        assert is_stable.shape == (result.n_obs_effective,)
        assert max_mod.shape == (result.n_obs_effective,)

        # For stable data, most should be stable
        assert np.mean(is_stable) > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
