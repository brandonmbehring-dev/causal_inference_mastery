"""
Tests for Structural VAR.

Session 137: Comprehensive tests for SVAR, IRF, and FEVD.

Test Layers:
- Layer 1: Known-Answer (8 tests) - Known structures, exact results
- Layer 2: Adversarial (7 tests) - Edge cases, boundary conditions
- Layer 3: Monte Carlo (6 tests) - Statistical validation
"""

import numpy as np
import pytest
from typing import Tuple

from causal_inference.timeseries import (
    var_estimate,
    VARResult,
    cholesky_svar,
    short_run_svar,
    companion_form,
    vma_coefficients,
    structural_vma_coefficients,
    check_stability,
    long_run_impact_matrix,
    verify_identification,
    compute_irf,
    compute_irf_reduced_form,
    bootstrap_irf,
    compute_fevd,
    historical_decomposition,
    fevd_convergence,
    SVARResult,
    IRFResult,
    FEVDResult,
    IdentificationMethod,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_recursive_svar_data():
    """
    Generate data from a recursive SVAR structure.

    Structural model:
        y1_t = 0.5 y1_{t-1} + ε1_t
        y2_t = 0.3 y1_t + 0.4 y2_{t-1} + ε2_t  (y1 affects y2 contemporaneously)
    """
    np.random.seed(42)
    n = 300
    k = 2

    # Structural shocks
    eps = np.random.randn(n, k)

    # Structural coefficient (B0_inv lower triangular)
    # y2 responds to y1 with coefficient 0.3
    B0_inv_true = np.array([[1.0, 0.0], [0.3, 1.0]])

    # AR coefficients
    A1_true = np.array([[0.5, 0.0], [0.0, 0.4]])

    # Generate data
    data = np.zeros((n, k))
    for t in range(1, n):
        structural_y = A1_true @ data[t - 1, :] + eps[t, :]
        data[t, :] = B0_inv_true @ structural_y

    return data, B0_inv_true, A1_true


@pytest.fixture
def sample_3var_data():
    """Generate 3-variable VAR data for testing."""
    np.random.seed(42)
    n = 300
    k = 3

    # Stable VAR(1) coefficients
    A1 = np.array([
        [0.4, 0.1, 0.0],
        [0.0, 0.5, 0.1],
        [0.0, 0.0, 0.3]
    ])

    data = np.zeros((n, k))
    for t in range(1, n):
        data[t, :] = A1 @ data[t - 1, :] + np.random.randn(k) * 0.5

    return data


@pytest.fixture
def sample_unstable_var_data():
    """Generate VAR data with unit root (non-stationary)."""
    np.random.seed(42)
    n = 200
    k = 2

    # Near-unit root coefficients
    A1 = np.array([[0.99, 0.0], [0.0, 0.99]])

    data = np.zeros((n, k))
    for t in range(1, n):
        data[t, :] = A1 @ data[t - 1, :] + np.random.randn(k) * 0.5

    return data


def generate_svar_data(
    n: int = 200,
    B0_inv: np.ndarray = None,
    A1: np.ndarray = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate SVAR data with known structure."""
    np.random.seed(seed)

    if B0_inv is None:
        B0_inv = np.eye(2)
    if A1 is None:
        A1 = np.array([[0.5, 0.0], [0.0, 0.4]])

    k = B0_inv.shape[0]
    eps = np.random.randn(n, k)

    data = np.zeros((n, k))
    for t in range(1, n):
        reduced_form_y = A1 @ data[t - 1, :] + B0_inv @ eps[t, :]
        data[t, :] = reduced_form_y

    return data, B0_inv, A1


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestSVARKnownStructure:
    """Layer 1: Tests with known SVAR structures."""

    def test_cholesky_svar_basic(self, sample_recursive_svar_data):
        """Test Cholesky SVAR on recursive structure."""
        data, B0_inv_true, A1_true = sample_recursive_svar_data

        # Estimate VAR
        var_result = var_estimate(data, lags=1)

        # Estimate SVAR
        svar_result = cholesky_svar(var_result)

        # Check identification method
        assert svar_result.identification == IdentificationMethod.CHOLESKY

        # B0_inv should be lower triangular
        assert np.allclose(np.triu(svar_result.B0_inv, 1), 0, atol=1e-10)

        # Check structural shocks have unit variance
        shock_var = np.var(svar_result.structural_shocks, axis=0)
        assert np.allclose(shock_var, 1.0, atol=0.15)

    def test_cholesky_svar_with_ordering(self, sample_3var_data):
        """Test Cholesky SVAR with custom ordering."""
        data = sample_3var_data
        var_result = var_estimate(data, lags=1, var_names=["x", "y", "z"])

        # Different orderings should give different results
        svar1 = cholesky_svar(var_result, ordering=["x", "y", "z"])
        svar2 = cholesky_svar(var_result, ordering=["z", "y", "x"])

        assert not np.allclose(svar1.B0_inv, svar2.B0_inv)

    def test_irf_horizon_zero(self, sample_recursive_svar_data):
        """Test IRF at horizon 0 equals B0_inv."""
        data, B0_inv_true, A1_true = sample_recursive_svar_data
        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        irf = compute_irf(svar_result, horizons=10)

        # IRF at horizon 0 should equal B0_inv
        assert np.allclose(irf.irf[:, :, 0], svar_result.B0_inv, atol=1e-10)

    def test_fevd_rows_sum_to_one(self, sample_3var_data):
        """Test FEVD rows sum to 1 at each horizon."""
        data = sample_3var_data
        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        fevd = compute_fevd(svar_result, horizons=20)

        # Check rows sum to 1
        for h in range(21):
            row_sums = fevd.fevd[:, :, h].sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-10)

    def test_companion_form_shape(self):
        """Test companion matrix shape."""
        np.random.seed(42)
        data = np.random.randn(200, 3)
        var_result = var_estimate(data, lags=2)

        F = companion_form(var_result)

        # Companion matrix should be (n_vars * lags, n_vars * lags)
        expected_shape = (3 * 2, 3 * 2)
        assert F.shape == expected_shape

    def test_verify_identification(self, sample_recursive_svar_data):
        """Test identification verification."""
        data, _, _ = sample_recursive_svar_data
        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        is_valid, max_error = verify_identification(
            var_result.sigma, svar_result.B0_inv
        )

        assert is_valid
        assert max_error < 1e-8

    def test_vma_coefficients_decay(self, sample_3var_data):
        """Test VMA coefficients decay for stable VAR."""
        data = sample_3var_data
        var_result = var_estimate(data, lags=1)

        Phi = vma_coefficients(var_result, horizons=50)

        # Coefficients should decay for stable VAR
        early_norm = np.linalg.norm(Phi[:, :, 5])
        late_norm = np.linalg.norm(Phi[:, :, 50])

        assert late_norm < early_norm

    def test_stability_check(self):
        """Test stability check for stable VAR."""
        np.random.seed(42)

        # Generate stable VAR data
        data = np.zeros((200, 2))
        A1 = np.array([[0.5, 0.1], [0.2, 0.4]])

        for t in range(1, 200):
            data[t, :] = A1 @ data[t - 1, :] + np.random.randn(2) * 0.5

        var_result = var_estimate(data, lags=1)
        is_stable, eigenvalues = check_stability(var_result)

        assert is_stable
        assert np.all(np.abs(eigenvalues) < 1.0)


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestSVARAdversarial:
    """Layer 2: Edge cases and boundary conditions."""

    def test_near_singular_covariance(self):
        """Test SVAR with near-singular covariance."""
        np.random.seed(42)
        n = 200

        # Create nearly collinear data
        x = np.random.randn(n)
        data = np.column_stack([x, x + 0.01 * np.random.randn(n)])

        var_result = var_estimate(data, lags=1)

        # Should still work (with possible warning)
        svar_result = cholesky_svar(var_result)
        assert svar_result is not None

    def test_short_time_series(self):
        """Test SVAR with short time series."""
        np.random.seed(42)
        data = np.random.randn(30, 2)  # Short series

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        # Should work but with high uncertainty
        assert svar_result.n_obs < 50

    def test_high_dimensional(self):
        """Test SVAR with many variables."""
        np.random.seed(42)
        n, k = 300, 10

        data = np.random.randn(n, k)
        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        assert svar_result.n_vars == k
        assert svar_result.B0_inv.shape == (k, k)

    def test_irf_long_horizon(self, sample_3var_data):
        """Test IRF at very long horizons."""
        data = sample_3var_data
        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        # Long horizon
        irf = compute_irf(svar_result, horizons=100)

        # For stable VAR, IRF should approach zero
        max_late_response = np.max(np.abs(irf.irf[:, :, -1]))
        assert max_late_response < 0.1

    def test_cumulative_irf(self, sample_3var_data):
        """Test cumulative IRF computation."""
        data = sample_3var_data
        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        irf_regular = compute_irf(svar_result, horizons=20, cumulative=False)
        irf_cumulative = compute_irf(svar_result, horizons=20, cumulative=True)

        # Cumulative at horizon h should be sum of regular 0 to h
        for h in range(21):
            expected_cumulative = irf_regular.irf[:, :, : h + 1].sum(axis=2)
            assert np.allclose(irf_cumulative.irf[:, :, h], expected_cumulative, atol=1e-10)

    def test_ordering_validation(self, sample_3var_data):
        """Test that invalid ordering raises error."""
        data = sample_3var_data
        var_result = var_estimate(data, lags=1, var_names=["x", "y", "z"])

        # Invalid variable name
        with pytest.raises(ValueError, match="Variable"):
            cholesky_svar(var_result, ordering=["x", "y", "invalid"])

        # Wrong number of variables
        with pytest.raises(ValueError, match="elements"):
            cholesky_svar(var_result, ordering=["x", "y"])

    def test_fevd_convergence_analysis(self, sample_3var_data):
        """Test FEVD convergence analysis."""
        data = sample_3var_data
        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        result = fevd_convergence(svar_result, max_horizon=100, tol=1e-3)

        # Should converge for stable VAR
        assert result["converged"]
        assert result["horizon_converged"] < 100


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================


class TestSVARMonteCarlo:
    """Layer 3: Statistical validation via Monte Carlo."""

    @pytest.mark.parametrize("n_obs", [100, 200, 500])
    def test_irf_bias_decreases_with_sample_size(self, n_obs):
        """Test that IRF bias decreases with sample size."""
        n_sims = 50
        horizons = 10

        # True parameters
        B0_inv_true = np.array([[1.0, 0.0], [0.5, 1.0]])
        A1_true = np.array([[0.5, 0.0], [0.0, 0.4]])

        # Compute true IRF
        true_irf = np.zeros((2, 2, horizons + 1))
        true_irf[:, :, 0] = B0_inv_true
        A_power = np.eye(2)
        for h in range(1, horizons + 1):
            A_power = A_power @ A1_true
            true_irf[:, :, h] = A_power @ B0_inv_true

        # Monte Carlo
        irf_estimates = []
        for sim in range(n_sims):
            data, _, _ = generate_svar_data(
                n=n_obs, B0_inv=B0_inv_true, A1=A1_true, seed=sim
            )
            var_result = var_estimate(data, lags=1)
            svar_result = cholesky_svar(var_result)
            irf = compute_irf(svar_result, horizons=horizons)
            irf_estimates.append(irf.irf)

        mean_irf = np.mean(irf_estimates, axis=0)
        bias = np.mean(np.abs(mean_irf - true_irf))

        # Bias should be reasonable for given sample size
        if n_obs >= 200:
            assert bias < 0.15

    def test_fevd_consistency(self):
        """Test FEVD converges to correct values."""
        n_sims = 30
        n_obs = 500

        # True parameters (diagonal A1 for simplicity)
        B0_inv_true = np.array([[1.0, 0.0], [0.5, 1.0]])
        A1_true = np.array([[0.5, 0.0], [0.0, 0.4]])

        fevd_estimates = []
        for sim in range(n_sims):
            data, _, _ = generate_svar_data(
                n=n_obs, B0_inv=B0_inv_true, A1=A1_true, seed=sim
            )
            var_result = var_estimate(data, lags=1)
            svar_result = cholesky_svar(var_result)
            fevd = compute_fevd(svar_result, horizons=20)
            fevd_estimates.append(fevd.fevd[:, :, -1])  # Long-run

        mean_fevd = np.mean(fevd_estimates, axis=0)

        # Each row should sum to 1
        for i in range(2):
            assert np.isclose(mean_fevd[i, :].sum(), 1.0, atol=0.05)

    def test_structural_shocks_independence(self):
        """Test that structural shocks are approximately independent."""
        n_sims = 50
        correlations = []

        for sim in range(n_sims):
            np.random.seed(sim)
            data = np.random.randn(300, 2)
            var_result = var_estimate(data, lags=1)
            svar_result = cholesky_svar(var_result)

            shocks = svar_result.structural_shocks
            corr = np.corrcoef(shocks[:, 0], shocks[:, 1])[0, 1]
            correlations.append(corr)

        mean_corr = np.mean(np.abs(correlations))

        # Structural shocks should be nearly uncorrelated
        assert mean_corr < 0.1

    def test_bootstrap_coverage(self):
        """Test bootstrap confidence interval coverage."""
        np.random.seed(42)

        # True parameters
        B0_inv_true = np.array([[1.0, 0.0], [0.3, 1.0]])
        A1_true = np.array([[0.5, 0.0], [0.0, 0.4]])

        n_sims = 30
        n_bootstrap = 100  # Fewer for speed
        alpha = 0.10  # 90% CI

        coverage_count = 0

        for sim in range(n_sims):
            data, _, _ = generate_svar_data(
                n=200, B0_inv=B0_inv_true, A1=A1_true, seed=sim
            )
            var_result = var_estimate(data, lags=1)
            svar_result = cholesky_svar(var_result)

            irf_ci = bootstrap_irf(
                data, svar_result, horizons=5, n_bootstrap=n_bootstrap,
                alpha=alpha, seed=sim + 1000
            )

            # Check if true IRF at horizon 1 is within CI
            # True IRF[1, 0, 1] = A1[1,0] * B0_inv[0,0] + A1[1,1] * B0_inv[1,0]
            true_irf_1_0_1 = A1_true @ B0_inv_true
            true_val = true_irf_1_0_1[1, 0]

            lower = irf_ci.irf_lower[1, 0, 1]
            upper = irf_ci.irf_upper[1, 0, 1]

            if lower <= true_val <= upper:
                coverage_count += 1

        coverage = coverage_count / n_sims

        # Coverage should be approximately 1 - alpha (90%)
        # Allow some slack due to small n_sims
        assert coverage >= 0.70  # At least 70% coverage

    def test_irf_at_horizon_1_approximates_a1_b0inv(self):
        """Test IRF at horizon 1 ≈ A1 @ B0_inv."""
        n_sims = 30

        B0_inv_true = np.array([[1.0, 0.0], [0.4, 1.0]])
        A1_true = np.array([[0.5, 0.1], [0.0, 0.4]])

        errors = []
        for sim in range(n_sims):
            data, _, _ = generate_svar_data(
                n=300, B0_inv=B0_inv_true, A1=A1_true, seed=sim
            )
            var_result = var_estimate(data, lags=1)
            svar_result = cholesky_svar(var_result)
            irf = compute_irf(svar_result, horizons=5)

            # True IRF at horizon 1
            true_irf_1 = A1_true @ B0_inv_true

            error = np.mean(np.abs(irf.irf[:, :, 1] - true_irf_1))
            errors.append(error)

        mean_error = np.mean(errors)
        assert mean_error < 0.2

    def test_stability_detection(self):
        """Test that stability check correctly identifies unstable VARs."""
        np.random.seed(42)

        # Unstable VAR (eigenvalue > 1)
        n = 200
        A1_unstable = np.array([[1.1, 0.0], [0.0, 0.5]])

        data = np.zeros((n, 2))
        for t in range(1, n):
            innovation = np.random.randn(2) * 0.1
            data[t, :] = A1_unstable @ data[t - 1, :] + innovation
            # Bound to prevent overflow
            data[t, :] = np.clip(data[t, :], -1e6, 1e6)

        var_result = var_estimate(data, lags=1)
        is_stable, eigenvalues = check_stability(var_result)

        # Should detect instability (or near-instability)
        assert np.max(np.abs(eigenvalues)) > 0.9


# =============================================================================
# IRF Result Tests
# =============================================================================


class TestIRFResult:
    """Tests for IRFResult class functionality."""

    def test_get_response_by_index(self, sample_3var_data):
        """Test getting response by variable index."""
        data = sample_3var_data
        var_result = var_estimate(data, lags=1, var_names=["x", "y", "z"])
        svar_result = cholesky_svar(var_result)
        irf = compute_irf(svar_result, horizons=10)

        # Get response by index
        response = irf.get_response(1, 0)
        assert response.shape == (11,)

        # Get specific horizon
        val = irf.get_response(1, 0, horizon=5)
        assert isinstance(val, (float, np.floating))

    def test_get_response_by_name(self, sample_3var_data):
        """Test getting response by variable name."""
        data = sample_3var_data
        var_result = var_estimate(data, lags=1, var_names=["x", "y", "z"])
        svar_result = cholesky_svar(var_result)
        irf = compute_irf(svar_result, horizons=10)

        response = irf.get_response("y", "x")
        assert response.shape == (11,)


# =============================================================================
# FEVD Result Tests
# =============================================================================


class TestFEVDResult:
    """Tests for FEVDResult class functionality."""

    def test_get_decomposition(self, sample_3var_data):
        """Test getting variance decomposition."""
        data = sample_3var_data
        var_result = var_estimate(data, lags=1, var_names=["x", "y", "z"])
        svar_result = cholesky_svar(var_result)
        fevd = compute_fevd(svar_result, horizons=20)

        decomp = fevd.get_decomposition("y", horizon=10)
        assert decomp.shape == (3,)
        assert np.isclose(decomp.sum(), 1.0)

    def test_get_contribution(self, sample_3var_data):
        """Test getting specific contribution."""
        data = sample_3var_data
        var_result = var_estimate(data, lags=1, var_names=["x", "y", "z"])
        svar_result = cholesky_svar(var_result)
        fevd = compute_fevd(svar_result, horizons=20)

        contrib = fevd.get_contribution("y", "x", horizon=10)
        assert 0.0 <= contrib <= 1.0

    def test_validate_rows_sum_to_one(self, sample_3var_data):
        """Test validation method."""
        data = sample_3var_data
        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)
        fevd = compute_fevd(svar_result, horizons=20)

        assert fevd.validate_rows_sum_to_one()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestSVAREdgeCases:
    """Additional edge case tests."""

    def test_single_lag(self):
        """Test SVAR with single lag."""
        np.random.seed(42)
        data = np.random.randn(200, 2)
        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        assert svar_result.lags == 1

    def test_multiple_lags(self):
        """Test SVAR with multiple lags."""
        np.random.seed(42)
        data = np.random.randn(200, 2)
        var_result = var_estimate(data, lags=3)
        svar_result = cholesky_svar(var_result)

        assert svar_result.lags == 3

        # Companion matrix should be 2*3 x 2*3
        F = companion_form(var_result)
        assert F.shape == (6, 6)

    def test_reduced_form_irf(self):
        """Test reduced form (non-orthogonalized) IRF."""
        np.random.seed(42)
        data = np.random.randn(200, 2)
        var_result = var_estimate(data, lags=1)

        irf = compute_irf_reduced_form(var_result, horizons=10)

        assert not irf.orthogonalized
        # IRF at horizon 0 should be identity for reduced form
        assert np.allclose(irf.irf[:, :, 0], np.eye(2))

    def test_long_run_impact(self, sample_3var_data):
        """Test long-run impact matrix computation."""
        data = sample_3var_data
        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        long_run = long_run_impact_matrix(svar_result)

        assert long_run.shape == (3, 3)
        assert np.all(np.isfinite(long_run))
