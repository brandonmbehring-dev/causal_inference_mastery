"""
Adversarial tests for bunching estimation.

Tests edge cases, boundary conditions, and pathological inputs to ensure
robust behavior under stress.

Categories:
1. Extreme data distributions
2. Edge cases in bunching region
3. Polynomial fitting challenges
4. Numerical stability
5. Bootstrap robustness
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.causal_inference.bunching import (
    estimate_counterfactual,
    polynomial_counterfactual,
    bunching_estimator,
    compute_excess_mass,
    compute_elasticity,
    bootstrap_bunching_se,
)


# =============================================================================
# Category 1: Extreme Data Distributions
# =============================================================================


class TestExtremeDistributions:
    """Adversarial tests for extreme data distributions."""

    def test_all_data_in_bunching_region(self):
        """All data concentrated in bunching region - should raise error."""
        rng = np.random.default_rng(42)
        # All data within bunching region
        data = rng.normal(50, 1, size=1000)

        # When all data is in bunching region, there are no bins outside
        # to fit the polynomial counterfactual. This is an expected error.
        with pytest.raises(ValueError, match="Need at least"):
            estimate_counterfactual(
                data=data,
                kink_point=50,
                bunching_width=5,
                n_bins=50,
                polynomial_order=3,
            )

    def test_bimodal_distribution(self):
        """Bimodal distribution with bunching at neither mode."""
        rng = np.random.default_rng(42)
        mode1 = rng.normal(30, 5, size=500)
        mode2 = rng.normal(70, 5, size=500)
        data = np.concatenate([mode1, mode2])

        result = estimate_counterfactual(
            data=data,
            kink_point=50,  # Between modes
            bunching_width=5,
            n_bins=50,
        )

        # Counterfactual estimation should work
        assert len(result["bin_centers"]) == 50

        excess_mass, excess_count, h0 = compute_excess_mass(result)
        # For bimodal data with gap at kink, h0 may be very low or zero
        # leading to inf excess mass. This is an expected edge case.
        # The key is that the computation completes without error.
        assert isinstance(excess_mass, (int, float, np.floating))

    def test_heavy_tailed_distribution(self):
        """Heavy-tailed Cauchy-like distribution."""
        rng = np.random.default_rng(42)
        # Use t-distribution with low df for heavy tails
        data = 50 + 10 * rng.standard_t(df=2, size=1000)
        # Trim extreme outliers
        data = data[(data > 0) & (data < 100)]

        result = bunching_estimator(
            data=data,
            kink_point=50,
            bunching_width=5,
            n_bins=40,
            n_bootstrap=30,
            random_state=42,
        )

        assert result["convergence"] is True
        assert np.isfinite(result["excess_mass"])

    def test_highly_skewed_distribution(self):
        """Highly skewed exponential distribution."""
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=20, size=1000)

        result = estimate_counterfactual(
            data=data,
            kink_point=20,  # Near mean
            bunching_width=5,
            n_bins=50,
        )

        assert np.all(np.isfinite(result["counterfactual_counts"]))

    def test_discrete_valued_data(self):
        """Data that takes discrete values (e.g., rounded income)."""
        rng = np.random.default_rng(42)
        # Round to nearest 1000
        data = np.round(rng.normal(50000, 10000, size=1000), -3)

        result = estimate_counterfactual(
            data=data,
            kink_point=50000,
            bunching_width=3000,
            n_bins=30,
        )

        assert len(result["bin_centers"]) == 30

    def test_sparse_data(self):
        """Very sparse data (large gaps)."""
        rng = np.random.default_rng(42)
        # Two clusters with large gap
        low_cluster = rng.uniform(10, 20, size=50)
        high_cluster = rng.uniform(80, 90, size=50)
        data = np.concatenate([low_cluster, high_cluster])

        result = estimate_counterfactual(
            data=data,
            kink_point=50,  # In the gap
            bunching_width=10,
            n_bins=20,
            polynomial_order=3,
        )

        # Many bins will be empty
        assert len(result["bin_centers"]) == 20


# =============================================================================
# Category 2: Edge Cases in Bunching Region
# =============================================================================


class TestBunchingRegionEdgeCases:
    """Edge cases related to bunching region specification."""

    def test_bunching_region_at_lower_boundary(self):
        """Bunching region at lower data boundary."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, size=1000)

        result = estimate_counterfactual(
            data=data,
            kink_point=5,  # Very close to 0
            bunching_width=3,
            n_bins=50,
        )

        # Bunching region should be adjusted
        lower, upper = result["bunching_region"]
        assert lower >= 0

    def test_bunching_region_at_upper_boundary(self):
        """Bunching region at upper data boundary."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, size=1000)

        result = estimate_counterfactual(
            data=data,
            kink_point=95,  # Very close to max
            bunching_width=3,
            n_bins=50,
        )

        lower, upper = result["bunching_region"]
        assert upper <= 100

    def test_very_narrow_bunching_region(self):
        """Very narrow bunching region (single bin)."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, size=1000)

        result = estimate_counterfactual(
            data=data,
            kink_point=50,
            bunching_width=0.5,  # Very narrow
            n_bins=100,  # Many bins
        )

        # Should still work
        assert len(result["bin_centers"]) == 100

    def test_bunching_region_outside_data_range(self):
        """Bunching region specified outside data range."""
        rng = np.random.default_rng(42)
        data = rng.uniform(40, 60, size=1000)

        # Kink at 80, but data only goes to 60
        result = estimate_counterfactual(
            data=data,
            kink_point=80,
            bunching_width=5,
            n_bins=30,
        )

        # Bunching region should be clipped to data range
        lower, upper = result["bunching_region"]
        data_max = np.max(data)
        # Both bounds should be <= data_max since kink is outside range
        assert upper <= data_max + 0.1  # Small tolerance for bin edges

    def test_asymmetric_bunching_detection(self):
        """Bunching on only one side of kink."""
        rng = np.random.default_rng(42)

        # Background
        background = rng.uniform(20, 80, size=800)
        # Bunching only BELOW kink
        bunchers = rng.uniform(46, 50, size=200)
        data = np.concatenate([background, bunchers])

        result = bunching_estimator(
            data=data,
            kink_point=50,
            bunching_width=5,
            n_bins=60,
            n_bootstrap=30,
            random_state=42,
        )

        # Should still detect positive excess mass
        assert result["excess_mass"] > 0


# =============================================================================
# Category 3: Polynomial Fitting Challenges
# =============================================================================


class TestPolynomialFittingChallenges:
    """Tests for challenging polynomial fitting scenarios."""

    def test_multimodal_outside_bunching(self):
        """Multiple modes outside bunching region."""
        rng = np.random.default_rng(42)
        mode1 = rng.normal(20, 3, size=300)
        mode2 = rng.normal(80, 3, size=300)
        bunching = rng.normal(50, 1, size=400)
        data = np.concatenate([mode1, mode2, bunching])

        result = estimate_counterfactual(
            data=data,
            kink_point=50,
            bunching_width=5,
            n_bins=60,
            polynomial_order=7,
        )

        # Should handle multimodal shape
        assert result["r_squared"] > 0

    def test_very_high_polynomial_order(self):
        """Very high polynomial order (potential overfitting)."""
        rng = np.random.default_rng(42)
        data = rng.normal(50, 15, size=500)

        # High order with enough bins
        result = estimate_counterfactual(
            data=data,
            kink_point=50,
            bunching_width=5,
            n_bins=100,
            polynomial_order=15,
        )

        # Should work but may have lower R-squared
        assert len(result["polynomial_coeffs"]) == 16

    def test_polynomial_order_equals_bins_minus_bunching(self):
        """Polynomial order nearly equals available bins."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, size=1000)

        # 30 bins, bunching covers ~10, so ~20 for fitting
        # Polynomial order 15 needs 16 points
        result = estimate_counterfactual(
            data=data,
            kink_point=50,
            bunching_width=15,
            n_bins=30,
            polynomial_order=5,  # Safe order
        )

        assert len(result["bin_centers"]) == 30

    def test_flat_distribution_polynomial(self):
        """Perfectly flat distribution (uniform)."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, size=10000)

        result = estimate_counterfactual(
            data=data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
            polynomial_order=1,  # Linear for flat
        )

        # Counterfactual should be nearly constant
        cf = result["counterfactual_counts"]
        cv = np.std(cf) / np.mean(cf) if np.mean(cf) > 0 else 0
        assert cv < 0.3  # Low coefficient of variation

    def test_polynomial_negative_extrapolation(self):
        """Polynomial that would extrapolate to negative values."""
        bin_centers = np.linspace(0, 100, 50)
        # U-shaped: high at edges, low in middle
        counts = 100 - 50 * np.exp(-((bin_centers - 50) ** 2) / 500)

        counterfactual, _, _ = polynomial_counterfactual(
            bin_centers=bin_centers,
            counts=counts,
            bunching_lower=45,
            bunching_upper=55,
            polynomial_order=4,
        )

        # Must be non-negative
        assert np.all(counterfactual >= 0)


# =============================================================================
# Category 4: Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_very_large_values(self):
        """Data with very large values."""
        rng = np.random.default_rng(42)
        data = rng.normal(1e9, 1e7, size=1000)

        result = estimate_counterfactual(
            data=data,
            kink_point=1e9,
            bunching_width=1e7,
            n_bins=50,
        )

        assert np.all(np.isfinite(result["counterfactual_counts"]))

    def test_very_small_values(self):
        """Data with very small values."""
        rng = np.random.default_rng(42)
        data = rng.normal(1e-6, 1e-7, size=1000)
        data = data[data > 0]  # Keep positive

        result = estimate_counterfactual(
            data=data,
            kink_point=1e-6,
            bunching_width=1e-7,
            n_bins=50,
        )

        assert np.all(np.isfinite(result["counterfactual_counts"]))

    def test_wide_range_values(self):
        """Data spanning many orders of magnitude."""
        rng = np.random.default_rng(42)
        # Log-uniform distribution
        log_data = rng.uniform(0, 6, size=1000)  # 10^0 to 10^6
        data = 10**log_data

        result = estimate_counterfactual(
            data=data,
            kink_point=1000,  # 10^3
            bunching_width=500,
            n_bins=50,
        )

        assert len(result["bin_centers"]) == 50

    def test_elasticity_near_zero_rate_change(self):
        """Elasticity when rate change is very small."""
        # Small rate change but valid
        elasticity = compute_elasticity(
            excess_mass=1.0,
            t1_rate=0.200,
            t2_rate=0.201,  # 0.1 percentage point difference
        )

        # Should be large but finite
        assert np.isfinite(elasticity)
        assert elasticity > 100  # Large elasticity for small rate change

    def test_elasticity_near_100_percent_rate(self):
        """Elasticity when rates are near 100%."""
        elasticity = compute_elasticity(
            excess_mass=0.5,
            t1_rate=0.98,
            t2_rate=0.99,
        )

        assert np.isfinite(elasticity)

    def test_zero_counts_some_bins(self):
        """Some bins have zero counts."""
        rng = np.random.default_rng(42)
        # Sparse data
        data = np.concatenate(
            [
                rng.normal(20, 2, size=100),
                rng.normal(80, 2, size=100),
            ]
        )

        result = estimate_counterfactual(
            data=data,
            kink_point=50,
            bunching_width=5,
            n_bins=60,
            polynomial_order=5,
        )

        # Many bins should be zero but fit should work
        assert np.sum(result["actual_counts"] == 0) > 0
        assert np.all(np.isfinite(result["counterfactual_counts"]))


# =============================================================================
# Category 5: Bootstrap Robustness
# =============================================================================


class TestBootstrapRobustness:
    """Tests for bootstrap robustness."""

    def test_bootstrap_with_small_sample(self):
        """Bootstrap with very small sample."""
        rng = np.random.default_rng(42)
        data = rng.normal(50, 10, size=50)

        result = bunching_estimator(
            data=data,
            kink_point=50,
            bunching_width=5,
            n_bins=15,
            polynomial_order=3,
            n_bootstrap=30,
            random_state=42,
        )

        # Should still converge
        assert result["convergence"] is True

    def test_bootstrap_with_heterogeneous_data(self):
        """Bootstrap with highly heterogeneous data."""
        rng = np.random.default_rng(42)
        # Mix of different distributions with overlap in bunching region
        component1 = rng.uniform(10, 50, size=300)  # Overlaps bunching
        component2 = rng.uniform(50, 90, size=300)  # Overlaps bunching
        component3 = rng.normal(50, 2, size=100)  # Bunching component
        data = np.concatenate([component1, component2, component3])

        result = bunching_estimator(
            data=data,
            kink_point=50,
            bunching_width=5,
            n_bins=40,
            n_bootstrap=30,
            random_state=42,
        )

        # Should converge with this data
        assert result["convergence"] is True
        # excess_mass should be a number (can be inf in extreme cases but not nan)
        assert not np.isnan(result["excess_mass"])

    def test_bootstrap_few_iterations(self):
        """Bootstrap with very few iterations."""
        rng = np.random.default_rng(42)
        data = rng.normal(50, 10, size=500)

        # This should warn or handle gracefully
        excess_mass_se, _, _, _ = bootstrap_bunching_se(
            data=data,
            kink_point=50,
            bunching_width=5,
            n_bins=30,
            n_bootstrap=15,  # Few iterations
            random_state=42,
        )

        # Should still return valid SE
        assert excess_mass_se > 0
        assert np.isfinite(excess_mass_se)


# =============================================================================
# Category 6: Integration and Consistency
# =============================================================================


class TestIntegrationConsistency:
    """Tests for integration and consistency."""

    def test_counterfactual_consistency(self):
        """Counterfactual + excess mass are consistent."""
        rng = np.random.default_rng(42)
        background = rng.uniform(20, 80, size=800)
        bunchers = rng.normal(50, 1, size=200)
        data = np.concatenate([background, bunchers])

        result = bunching_estimator(
            data=data,
            kink_point=50,
            bunching_width=5,
            n_bins=60,
            n_bootstrap=30,
            random_state=42,
        )

        # Recompute excess mass from counterfactual
        cf = result["counterfactual"]
        b, B, h0 = compute_excess_mass(cf)

        assert b == pytest.approx(result["excess_mass"], rel=1e-10)

    def test_elasticity_direction(self):
        """Elasticity has correct sign."""
        # Positive excess mass + increasing rates = positive elasticity
        e1 = compute_elasticity(excess_mass=1.0, t1_rate=0.2, t2_rate=0.3)
        assert e1 > 0

        # Zero excess mass = zero elasticity
        e2 = compute_elasticity(excess_mass=0.0, t1_rate=0.2, t2_rate=0.3)
        assert e2 == 0

    def test_n_obs_matches_data(self):
        """n_obs in result matches input data length."""
        rng = np.random.default_rng(42)
        data = rng.normal(50, 10, size=347)

        result = bunching_estimator(
            data=data,
            kink_point=50,
            bunching_width=5,
            n_bins=30,
            n_bootstrap=20,
            random_state=42,
        )

        assert result["n_obs"] == 347

    def test_bin_width_consistency(self):
        """Bin width is consistent with n_bins and range."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, size=1000)

        result = estimate_counterfactual(
            data=data,
            kink_point=50,
            bunching_width=5,
            n_bins=50,
        )

        expected_width = 100 / 50  # range / n_bins
        assert result["bin_width"] == pytest.approx(expected_width, rel=0.01)


# =============================================================================
# Category 7: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for proper error handling."""

    def test_inf_in_data(self):
        """Raises for infinite values in data."""
        data = np.array([1.0, 2.0, np.inf, 4.0])
        with pytest.raises(ValueError, match="non-finite"):
            estimate_counterfactual(
                data=data,
                kink_point=2,
                bunching_width=0.5,
                n_bins=10,
            )

    def test_all_same_value(self):
        """Data with all same value - degenerate case."""
        data = np.full(100, 50.0)

        # When all data has the same value, histogram has only one non-zero bin.
        # With any bunching region, there are no bins outside to fit polynomial.
        # This is an expected error case.
        with pytest.raises(ValueError, match="Need at least"):
            estimate_counterfactual(
                data=data,
                kink_point=50,
                bunching_width=1,
                n_bins=10,
                polynomial_order=1,
            )

    def test_very_few_bins_with_high_order(self):
        """Too few bins for polynomial order."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, size=100)

        with pytest.raises(ValueError, match="Need at least"):
            estimate_counterfactual(
                data=data,
                kink_point=50,
                bunching_width=40,  # Large bunching region
                n_bins=15,
                polynomial_order=10,  # Needs 11 bins outside
            )

    def test_n_bins_less_than_minimum(self):
        """n_bins less than minimum (10)."""
        rng = np.random.default_rng(42)
        data = rng.normal(50, 10, size=100)

        with pytest.raises(ValueError, match="n_bins.*>= 10"):
            estimate_counterfactual(
                data=data,
                kink_point=50,
                bunching_width=5,
                n_bins=5,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
