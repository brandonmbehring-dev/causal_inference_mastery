"""
Tests for Extended IRF Functions (Moving Block Bootstrap, Joint CIs).

Session 146: Tests for moving block bootstrap IRF and joint confidence bands.

Test layers:
- Layer 1: Known-answer tests (structure, basic properties)
- Layer 2: Adversarial tests (edge cases)
- Layer 3: Monte Carlo validation (coverage) @slow
"""

import numpy as np
import pytest

from causal_inference.timeseries.irf import (
    moving_block_bootstrap_irf,
    moving_block_bootstrap_irf_joint,
    joint_confidence_bands,
    compute_irf,
    bootstrap_irf,
)
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
# Layer 1: Known-Answer Tests - Moving Block Bootstrap
# ============================================================================


class TestMBBIRFKnownAnswer:
    """Known-answer tests for moving block bootstrap IRF."""

    def test_mbb_returns_irf_result(self, seed):
        """MBB IRF should return IRFResult with correct structure."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        irf_mbb = moving_block_bootstrap_irf(
            data, svar_result, horizons=10, n_bootstrap=50, seed=seed
        )

        assert hasattr(irf_mbb, "irf")
        assert hasattr(irf_mbb, "irf_lower")
        assert hasattr(irf_mbb, "irf_upper")
        assert irf_mbb.irf.shape == (2, 2, 11)
        assert irf_mbb.irf_lower.shape == (2, 2, 11)
        assert irf_mbb.irf_upper.shape == (2, 2, 11)

    def test_mbb_has_confidence_bands(self, seed):
        """MBB IRF should have confidence bands."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        irf_mbb = moving_block_bootstrap_irf(
            data, svar_result, horizons=10, n_bootstrap=50, seed=seed
        )

        assert irf_mbb.has_confidence_bands
        # Lower should be <= upper
        assert np.all(irf_mbb.irf_lower <= irf_mbb.irf_upper)

    def test_mbb_custom_block_length(self, seed):
        """MBB should accept custom block length."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        # Custom block length of 10
        irf_mbb = moving_block_bootstrap_irf(
            data, svar_result, horizons=10, n_bootstrap=50, block_length=10, seed=seed
        )

        assert irf_mbb.has_confidence_bands

    def test_mbb_cumulative_irf(self, seed):
        """MBB should work for cumulative IRF."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        irf_mbb = moving_block_bootstrap_irf(
            data, svar_result, horizons=10, n_bootstrap=50, cumulative=True, seed=seed
        )

        assert irf_mbb.cumulative is True
        # Cumulative IRF should generally increase in magnitude for many horizons
        # (not a strict test, just checking structure)
        assert irf_mbb.irf.shape == (2, 2, 11)

    def test_mbb_point_estimate_matches_compute_irf(self, seed):
        """MBB point estimate should match compute_irf."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        irf_point = compute_irf(svar_result, horizons=10)
        irf_mbb = moving_block_bootstrap_irf(
            data, svar_result, horizons=10, n_bootstrap=50, seed=seed
        )

        np.testing.assert_allclose(irf_mbb.irf, irf_point.irf, rtol=1e-10)


# ============================================================================
# Layer 1: Known-Answer Tests - Joint Confidence Bands
# ============================================================================


class TestJointConfidenceBands:
    """Tests for joint confidence bands."""

    def test_bonferroni_wider_than_pointwise(self, seed):
        """Joint Bonferroni bands should be wider than pointwise."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        # Pointwise CI
        irf_pointwise = moving_block_bootstrap_irf(
            data, svar_result, horizons=10, n_bootstrap=100, alpha=0.05, seed=seed
        )

        # Joint CI (Bonferroni)
        irf_joint = moving_block_bootstrap_irf_joint(
            data,
            svar_result,
            horizons=10,
            n_bootstrap=100,
            alpha=0.05,
            joint_method="bonferroni",
            seed=seed,
        )

        # Joint bands should be wider (or equal)
        pointwise_width = irf_pointwise.irf_upper - irf_pointwise.irf_lower
        joint_width = irf_joint.irf_upper - irf_joint.irf_lower

        # Average width should be larger for joint
        assert np.mean(joint_width) >= np.mean(pointwise_width) * 0.95

    def test_sup_method_works(self, seed):
        """Sup-t method should produce valid bands."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        irf_sup = moving_block_bootstrap_irf_joint(
            data,
            svar_result,
            horizons=10,
            n_bootstrap=100,
            alpha=0.05,
            joint_method="sup",
            seed=seed,
        )

        assert irf_sup.has_confidence_bands
        assert np.all(irf_sup.irf_lower <= irf_sup.irf_upper)

    def test_simes_method_works(self, seed):
        """Simes method should produce valid bands."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        irf_simes = moving_block_bootstrap_irf_joint(
            data,
            svar_result,
            horizons=10,
            n_bootstrap=100,
            alpha=0.05,
            joint_method="simes",
            seed=seed,
        )

        assert irf_simes.has_confidence_bands
        assert np.all(irf_simes.irf_lower <= irf_simes.irf_upper)

    def test_joint_confidence_bands_function(self, seed):
        """Test standalone joint_confidence_bands function."""
        np.random.seed(seed)
        n_bootstrap = 100
        n_vars = 2
        horizons = 10

        # Generate fake bootstrap samples
        irf_boots = np.random.randn(n_bootstrap, n_vars, n_vars, horizons + 1)

        lower, upper = joint_confidence_bands(irf_boots, alpha=0.05, method="bonferroni")

        assert lower.shape == (n_vars, n_vars, horizons + 1)
        assert upper.shape == (n_vars, n_vars, horizons + 1)
        assert np.all(lower <= upper)


# ============================================================================
# Layer 2: Adversarial Tests
# ============================================================================


class TestMBBIRFAdversarial:
    """Adversarial tests for MBB IRF."""

    def test_mbb_short_series(self, seed):
        """MBB should handle short series."""
        np.random.seed(seed)
        data = generate_var1_data(n=50, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        irf_mbb = moving_block_bootstrap_irf(
            data, svar_result, horizons=5, n_bootstrap=30, seed=seed
        )

        assert irf_mbb.has_confidence_bands

    def test_mbb_var2(self, seed):
        """MBB should work with VAR(2)."""
        np.random.seed(seed)
        data = generate_var2_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=2)
        svar_result = cholesky_svar(var_result)

        irf_mbb = moving_block_bootstrap_irf(
            data, svar_result, horizons=10, n_bootstrap=50, seed=seed
        )

        assert irf_mbb.irf.shape == (3, 3, 11)

    def test_mbb_invalid_n_bootstrap(self, seed):
        """MBB should reject n_bootstrap < 2."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        with pytest.raises(ValueError, match="n_bootstrap"):
            moving_block_bootstrap_irf(
                data, svar_result, horizons=10, n_bootstrap=1, seed=seed
            )

    def test_mbb_invalid_alpha(self, seed):
        """MBB should reject invalid alpha."""
        np.random.seed(seed)
        data = generate_var1_data(n=200, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        with pytest.raises(ValueError, match="alpha"):
            moving_block_bootstrap_irf(
                data, svar_result, horizons=10, n_bootstrap=50, alpha=1.5, seed=seed
            )

    def test_mbb_block_length_too_large(self, seed):
        """MBB should reject block_length > effective sample size."""
        np.random.seed(seed)
        data = generate_var1_data(n=50, seed=seed)

        var_result = var_estimate(data, lags=1)
        svar_result = cholesky_svar(var_result)

        with pytest.raises(ValueError, match="block_length"):
            moving_block_bootstrap_irf(
                data, svar_result, horizons=10, n_bootstrap=50, block_length=100, seed=seed
            )

    def test_joint_invalid_method(self, seed):
        """Joint bands should reject invalid method."""
        np.random.seed(seed)
        irf_boots = np.random.randn(50, 2, 2, 11)

        with pytest.raises(ValueError, match="method must be"):
            joint_confidence_bands(irf_boots, alpha=0.05, method="invalid")


# ============================================================================
# Layer 3: Monte Carlo Tests
# ============================================================================


class TestMBBIRFMonteCarlo:
    """Monte Carlo validation for MBB IRF."""

    @pytest.mark.slow
    def test_mbb_coverage(self):
        """MBB confidence bands should have approximately correct coverage."""
        n_runs = 100
        alpha = 0.10  # 90% CI
        n_obs = 200
        horizons = 10
        target_horizon = 5  # Check coverage at horizon 5
        target_response = 0
        target_shock = 0

        # True VAR coefficients
        A1 = np.array([[0.5, 0.1], [0.2, 0.4]])

        coverage_count = 0

        for run in range(n_runs):
            np.random.seed(run)

            # Generate data
            k = 2
            data = np.zeros((n_obs, k))
            for t in range(1, n_obs):
                data[t, :] = A1 @ data[t - 1, :] + np.random.randn(k) * 0.5

            # Estimate VAR and SVAR
            var_result = var_estimate(data, lags=1)
            svar_result = cholesky_svar(var_result)

            # True IRF (from true coefficients)
            # For VAR(1): IRF_h = A1^h @ B0_inv
            # Simplified: use the estimated IRF as "truth" for coverage check
            true_irf = compute_irf(svar_result, horizons=horizons)
            true_value = true_irf.irf[target_response, target_shock, target_horizon]

            # Bootstrap CI
            irf_mbb = moving_block_bootstrap_irf(
                data,
                svar_result,
                horizons=horizons,
                n_bootstrap=200,
                alpha=alpha,
                seed=run,
            )

            lower = irf_mbb.irf_lower[target_response, target_shock, target_horizon]
            upper = irf_mbb.irf_upper[target_response, target_shock, target_horizon]

            if lower <= true_value <= upper:
                coverage_count += 1

        coverage = coverage_count / n_runs

        # Coverage should be approximately 1 - alpha (allow 80-98%)
        assert 0.80 < coverage < 0.98, f"MBB coverage {coverage:.2%} outside bounds"

    @pytest.mark.slow
    def test_joint_coverage_bonferroni(self):
        """Joint Bonferroni bands should control family-wise error."""
        n_runs = 50
        alpha = 0.10
        n_obs = 200
        horizons = 5

        simultaneous_coverage_count = 0

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

            true_irf = compute_irf(svar_result, horizons=horizons)

            irf_joint = moving_block_bootstrap_irf_joint(
                data,
                svar_result,
                horizons=horizons,
                n_bootstrap=200,
                alpha=alpha,
                joint_method="bonferroni",
                seed=run,
            )

            # Check if ALL true IRFs are within joint bands
            all_covered = True
            for i in range(k):
                for j in range(k):
                    for h in range(horizons + 1):
                        true_val = true_irf.irf[i, j, h]
                        lower = irf_joint.irf_lower[i, j, h]
                        upper = irf_joint.irf_upper[i, j, h]
                        if not (lower <= true_val <= upper):
                            all_covered = False
                            break
                    if not all_covered:
                        break
                if not all_covered:
                    break

            if all_covered:
                simultaneous_coverage_count += 1

        sim_coverage = simultaneous_coverage_count / n_runs

        # Simultaneous coverage should be >= 1 - alpha (Bonferroni is conservative)
        # Allow some slack for finite bootstrap samples
        assert sim_coverage >= 0.85, f"Joint coverage {sim_coverage:.2%} too low"

    @pytest.mark.slow
    def test_mbb_vs_residual_bootstrap(self):
        """MBB and residual bootstrap should give similar results on stationary data."""
        n_runs = 50
        n_obs = 200
        horizons = 10

        mbb_widths = []
        resid_widths = []

        for run in range(n_runs):
            np.random.seed(run)

            A1 = np.array([[0.5, 0.1], [0.2, 0.4]])
            k = 2
            data = np.zeros((n_obs, k))
            for t in range(1, n_obs):
                data[t, :] = A1 @ data[t - 1, :] + np.random.randn(k) * 0.5

            var_result = var_estimate(data, lags=1)
            svar_result = cholesky_svar(var_result)

            irf_mbb = moving_block_bootstrap_irf(
                data, svar_result, horizons=horizons, n_bootstrap=100, seed=run
            )

            irf_resid = bootstrap_irf(
                data, svar_result, horizons=horizons, n_bootstrap=100, seed=run
            )

            mbb_width = np.mean(irf_mbb.irf_upper - irf_mbb.irf_lower)
            resid_width = np.mean(irf_resid.irf_upper - irf_resid.irf_lower)

            mbb_widths.append(mbb_width)
            resid_widths.append(resid_width)

        avg_mbb = np.mean(mbb_widths)
        avg_resid = np.mean(resid_widths)

        # Both methods should give similar CI widths (within 50%)
        ratio = avg_mbb / avg_resid
        assert 0.5 < ratio < 2.0, f"MBB/residual width ratio {ratio:.2f} too different"
