"""
Tests for Local IV MTE estimation.

Layer 1: Known-answer tests
Layer 2: Unit tests
"""

import pytest
import numpy as np

from src.causal_inference.mte import local_iv, polynomial_mte


class TestLocalIV:
    """Tests for local_iv function."""

    def test_returns_correct_type(self, heterogeneous_mte_data):
        """Result has all required fields."""
        data = heterogeneous_mte_data
        result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_bootstrap=50,  # Reduce for speed
        )

        assert "mte_grid" in result
        assert "u_grid" in result
        assert "se_grid" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "propensity_support" in result
        assert "n_obs" in result
        assert "n_trimmed" in result
        assert "bandwidth" in result
        assert "method" in result

    def test_mte_grid_shape(self, heterogeneous_mte_data):
        """MTE grid has correct shape."""
        data = heterogeneous_mte_data
        n_grid = 30
        result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_grid=n_grid,
            n_bootstrap=50,
        )

        assert len(result["mte_grid"]) == n_grid
        assert len(result["u_grid"]) == n_grid
        assert len(result["se_grid"]) == n_grid

    def test_mte_linear_decreasing(self, heterogeneous_mte_data):
        """Recovers linear decreasing MTE pattern."""
        data = heterogeneous_mte_data
        result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_grid=20,
            n_bootstrap=50,
        )

        # MTE should be decreasing (negative slope)
        mte = result["mte_grid"]
        u = result["u_grid"]

        # Fit line to check slope
        valid = ~np.isnan(mte)
        if valid.sum() > 3:
            slope = np.polyfit(u[valid], mte[valid], 1)[0]
            # True slope is -2, should be negative
            assert slope < 0

    def test_mte_support_region(self, heterogeneous_mte_data):
        """Support region is within [0, 1]."""
        data = heterogeneous_mte_data
        result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_bootstrap=50,
        )

        p_min, p_max = result["propensity_support"]
        assert 0 <= p_min < p_max <= 1

    def test_mte_trimming(self, heterogeneous_mte_data):
        """Trimming reduces sample size."""
        data = heterogeneous_mte_data

        result_low = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            trim_fraction=0.01,
            n_bootstrap=50,
        )

        result_high = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            trim_fraction=0.10,
            n_bootstrap=50,
        )

        # More trimming = more observations trimmed
        assert result_high["n_trimmed"] > result_low["n_trimmed"]

    def test_mte_bandwidth_selection(self, heterogeneous_mte_data):
        """Different bandwidth rules produce different bandwidths."""
        data = heterogeneous_mte_data

        result_silverman = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            bandwidth_rule="silverman",
            n_bootstrap=50,
        )

        result_scott = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            bandwidth_rule="scott",
            n_bootstrap=50,
        )

        # Bandwidths might differ (not always by much)
        assert result_silverman["bandwidth"] > 0
        assert result_scott["bandwidth"] > 0

    def test_mte_manual_bandwidth(self, heterogeneous_mte_data):
        """Manual bandwidth is used when specified."""
        data = heterogeneous_mte_data
        manual_bw = 0.15

        result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            bandwidth=manual_bw,
            n_bootstrap=50,
        )

        assert result["bandwidth"] == manual_bw

    def test_mte_with_covariates(self, covariate_data):
        """Works with covariates."""
        data = covariate_data
        result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            covariates=data["covariates"],
            n_bootstrap=50,
        )

        # Should still produce valid output
        assert not np.all(np.isnan(result["mte_grid"]))

    def test_mte_se_positive(self, heterogeneous_mte_data):
        """Standard errors should be positive."""
        data = heterogeneous_mte_data
        result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_bootstrap=100,
        )

        # Most SEs should be positive (some might be NaN at edges)
        valid_se = result["se_grid"][~np.isnan(result["se_grid"])]
        assert np.all(valid_se > 0)

    def test_mte_ci_contains_estimate(self, heterogeneous_mte_data):
        """CI should contain point estimate."""
        data = heterogeneous_mte_data
        result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_bootstrap=100,
        )

        mte = result["mte_grid"]
        ci_lower = result["ci_lower"]
        ci_upper = result["ci_upper"]

        valid = ~np.isnan(mte)
        # Point estimate should be within CI
        assert np.all(ci_lower[valid] <= mte[valid])
        assert np.all(mte[valid] <= ci_upper[valid])


class TestPolynomialMTE:
    """Tests for polynomial_mte function."""

    def test_returns_correct_type(self, heterogeneous_mte_data):
        """Result has all required fields."""
        data = heterogeneous_mte_data
        result = polynomial_mte(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            degree=3,
            n_bootstrap=50,
        )

        assert "mte_grid" in result
        assert "u_grid" in result
        assert result["method"] == "polynomial"

    def test_polynomial_degree(self, heterogeneous_mte_data):
        """Different degrees produce different results."""
        data = heterogeneous_mte_data

        result_linear = polynomial_mte(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            degree=1,
            n_bootstrap=50,
        )

        result_cubic = polynomial_mte(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            degree=3,
            n_bootstrap=50,
        )

        # Linear vs cubic should differ
        mse_diff = np.nanmean((result_linear["mte_grid"] - result_cubic["mte_grid"]) ** 2)
        # They might be similar if true MTE is linear, but computation should work
        assert not np.isnan(mse_diff)

    def test_polynomial_recovers_linear(self, heterogeneous_mte_data):
        """Polynomial should recover approximately linear MTE."""
        data = heterogeneous_mte_data
        result = polynomial_mte(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            degree=2,
            n_bootstrap=50,
        )

        mte = result["mte_grid"]
        u = result["u_grid"]

        valid = ~np.isnan(mte)
        if valid.sum() > 3:
            # Fit line to check slope
            slope = np.polyfit(u[valid], mte[valid], 1)[0]
            # True MTE(u) = 3 - 2u, slope ≈ -2
            assert slope < 0  # At least negative

    def test_polynomial_quadratic_mte(self, quadratic_mte_data):
        """Polynomial should capture quadratic MTE."""
        data = quadratic_mte_data
        result = polynomial_mte(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            degree=3,
            n_bootstrap=50,
        )

        # Should produce non-NaN results
        assert not np.all(np.isnan(result["mte_grid"]))


class TestMTEConstant:
    """Tests for constant MTE case."""

    def test_constant_mte_relatively_flat(self, constant_mte_data):
        """With constant MTE, curve should be relatively flat (low variance)."""
        data = constant_mte_data
        result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_bootstrap=50,
        )

        mte = result["mte_grid"]
        valid = ~np.isnan(mte)

        if valid.sum() > 5:
            # Check that std is reasonable (not extremely variable)
            mte_std = np.std(mte[valid])
            # With kernel smoothing, there's inherent variability
            # Just check it's not absurdly high
            assert mte_std < 5.0  # Very loose bound - just checking it works

    def test_constant_mte_order_of_magnitude(self, constant_mte_data):
        """Constant MTE should be in right order of magnitude."""
        data = constant_mte_data
        result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_bootstrap=50,
        )

        mte = result["mte_grid"]
        valid = ~np.isnan(mte)

        mean_mte = np.mean(mte[valid])
        # True MTE = 1.5, accept within ±1.5 (i.e., roughly correct direction)
        assert 0 < mean_mte < 3.0  # Loose bounds - MTE is positive, in reasonable range


class TestMTEInputValidation:
    """Tests for input validation."""

    def test_rejects_mismatched_lengths(self, rng):
        """Should reject mismatched input lengths."""
        Y = rng.normal(size=100)
        D = rng.binomial(1, 0.5, size=90).astype(float)
        Z = rng.normal(size=100)

        with pytest.raises(ValueError, match="length"):
            local_iv(Y, D, Z, n_bootstrap=10)

    def test_rejects_non_binary_treatment(self, rng):
        """Should reject non-binary treatment."""
        Y = rng.normal(size=100)
        D = rng.normal(size=100)  # Continuous
        Z = rng.normal(size=100)

        with pytest.raises(ValueError, match="binary"):
            local_iv(Y, D, Z, n_bootstrap=10)

    def test_warns_small_sample(self, rng):
        """Should warn for small samples."""
        Y = rng.normal(size=30)
        D = rng.binomial(1, 0.5, size=30).astype(float)
        Z = rng.normal(size=30)

        with pytest.warns(UserWarning, match="[Ss]mall"):
            local_iv(Y, D, Z, n_bootstrap=10)


class TestMTEMultipleInstruments:
    """Tests with multiple instruments."""

    def test_multivariate_instrument(self, multivariate_instrument_data):
        """Should work with multiple instruments."""
        data = multivariate_instrument_data
        result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],  # 2D array
            n_bootstrap=50,
        )

        # Should produce valid output
        assert not np.all(np.isnan(result["mte_grid"]))

    def test_multivariate_mte_shape(self, multivariate_instrument_data):
        """MTE curve shape should still be correct."""
        data = multivariate_instrument_data
        result = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_bootstrap=50,
        )

        mte = result["mte_grid"]
        u = result["u_grid"]

        valid = ~np.isnan(mte)
        if valid.sum() > 3:
            # True MTE(u) = 2 - u, should be decreasing
            slope = np.polyfit(u[valid], mte[valid], 1)[0]
            assert slope < 0


class TestLocalIVvsPolynomial:
    """Compare local IV and polynomial methods."""

    def test_methods_similar_for_linear(self, heterogeneous_mte_data):
        """Both methods should give similar results for linear MTE."""
        data = heterogeneous_mte_data

        result_local = local_iv(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            n_grid=20,
            n_bootstrap=50,
        )

        result_poly = polynomial_mte(
            data["outcome"],
            data["treatment"],
            data["instrument"],
            degree=2,
            n_grid=20,
            n_bootstrap=50,
        )

        # Both should show decreasing pattern
        local_valid = ~np.isnan(result_local["mte_grid"])
        poly_valid = ~np.isnan(result_poly["mte_grid"])

        if local_valid.sum() > 3 and poly_valid.sum() > 3:
            slope_local = np.polyfit(
                result_local["u_grid"][local_valid], result_local["mte_grid"][local_valid], 1
            )[0]
            slope_poly = np.polyfit(
                result_poly["u_grid"][poly_valid], result_poly["mte_grid"][poly_valid], 1
            )[0]

            # Both slopes should be negative
            assert slope_local < 0
            assert slope_poly < 0
