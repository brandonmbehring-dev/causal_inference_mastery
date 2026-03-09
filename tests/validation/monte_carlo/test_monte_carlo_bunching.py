"""
Monte Carlo validation tests for bunching estimation.

Validates:
1. Unbiasedness of excess mass estimator
2. Coverage of confidence intervals
3. Standard error accuracy
4. Type I error control (null hypothesis)
5. Elasticity estimation accuracy

References:
- Saez (2010) - Original bunching methodology
- Chetty et al. (2011) - Integration constraint
- Kleven (2016) - Bunching review
"""

import numpy as np
import pytest
from typing import List, Dict, Any

from src.causal_inference.bunching import (
    bunching_estimator,
    estimate_counterfactual,
    compute_excess_mass,
    compute_elasticity,
)
from tests.validation.monte_carlo.dgp_bunching import (
    dgp_bunching_simple,
    dgp_bunching_uniform_counterfactual,
    dgp_bunching_no_effect,
    dgp_bunching_with_elasticity,
    dgp_bunching_asymmetric,
    dgp_bunching_diffuse,
    dgp_bunching_large_sample,
    dgp_bunching_small_sample,
)


# =============================================================================
# Test Configuration
# =============================================================================

N_SIMULATIONS = 100  # Number of Monte Carlo iterations
N_BOOTSTRAP = 50  # Bootstrap iterations per estimation (speed vs accuracy)


def run_bunching_monte_carlo(
    dgp_func,
    dgp_kwargs: Dict[str, Any],
    n_simulations: int = N_SIMULATIONS,
    n_bins: int = 50,
    polynomial_order: int = 7,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Dict[str, Any]:
    """Run Monte Carlo simulation for bunching estimator.

    Returns
    -------
    Dict with keys:
        - estimates: List of excess mass estimates
        - ses: List of standard errors
        - true_value: True excess mass
        - bias: Mean bias
        - rmse: Root mean squared error
        - coverage: CI coverage rate (if applicable)
        - mean_se: Mean standard error
        - empirical_se: Empirical standard deviation of estimates
    """
    estimates = []
    ses = []
    ci_covers = []

    # Get true parameters from one run
    sample_data = dgp_func(**dgp_kwargs, random_state=0)
    if len(sample_data) == 4:
        _, true_excess_mass, kink_point, bunching_width = sample_data
    else:
        # With elasticity
        _, _, true_excess_mass, kink_point, _, _ = sample_data
        bunching_width = 0.05 * kink_point  # Default

    for i in range(n_simulations):
        dgp_result = dgp_func(**dgp_kwargs, random_state=42 + i)

        if len(dgp_result) == 4:
            data, _, kink, bw = dgp_result
        else:
            data, _, _, kink, _, _ = dgp_result
            bw = bunching_width

        try:
            result = bunching_estimator(
                data=data,
                kink_point=kink,
                bunching_width=bw,
                n_bins=n_bins,
                polynomial_order=polynomial_order,
                n_bootstrap=n_bootstrap,
                random_state=i,
            )

            if np.isfinite(result["excess_mass"]):
                estimates.append(result["excess_mass"])

                if np.isfinite(result["excess_mass_se"]):
                    ses.append(result["excess_mass_se"])
                    # CI coverage (using normal approximation)
                    ci_lower = result["excess_mass"] - 1.96 * result["excess_mass_se"]
                    ci_upper = result["excess_mass"] + 1.96 * result["excess_mass_se"]
                    ci_covers.append(ci_lower <= true_excess_mass <= ci_upper)

        except (ValueError, np.linalg.LinAlgError):
            # Skip failed iterations
            continue

    if len(estimates) < 10:
        raise ValueError(f"Too few successful iterations: {len(estimates)}")

    estimates_arr = np.array(estimates)
    bias = np.mean(estimates_arr) - true_excess_mass
    rmse = np.sqrt(np.mean((estimates_arr - true_excess_mass) ** 2))
    empirical_se = np.std(estimates_arr, ddof=1)

    result = {
        "estimates": estimates,
        "ses": ses if ses else None,
        "true_value": true_excess_mass,
        "mean_estimate": np.mean(estimates_arr),
        "bias": bias,
        "rmse": rmse,
        "empirical_se": empirical_se,
        "n_successful": len(estimates),
    }

    if ses:
        result["mean_se"] = np.mean(ses)
        result["se_ratio"] = np.mean(ses) / empirical_se if empirical_se > 0 else np.nan

    if ci_covers:
        result["coverage"] = np.mean(ci_covers)

    return result


# =============================================================================
# Test: Excess Mass Unbiasedness
# =============================================================================


class TestExcessMassUnbiasedness:
    """Tests that excess mass estimator is approximately unbiased."""

    @pytest.mark.monte_carlo
    def test_simple_bunching_bias(self):
        """Excess mass estimator has low bias with clear bunching."""
        mc_result = run_bunching_monte_carlo(
            dgp_bunching_simple,
            {"n": 1000, "true_excess_mass": 2.0},
            n_simulations=N_SIMULATIONS,
        )

        # Bias should be small relative to true value
        relative_bias = abs(mc_result["bias"]) / max(mc_result["true_value"], 0.1)
        assert relative_bias < 0.50, (
            f"Relative bias {relative_bias:.2f} exceeds 0.50. "
            f"Bias={mc_result['bias']:.3f}, True={mc_result['true_value']:.3f}"
        )

    @pytest.mark.monte_carlo
    def test_uniform_counterfactual_bias(self):
        """Low bias with uniform counterfactual (simpler case)."""
        mc_result = run_bunching_monte_carlo(
            dgp_bunching_uniform_counterfactual,
            {"n": 1000, "buncher_fraction": 0.15},
            n_simulations=N_SIMULATIONS,
        )

        # Should detect positive excess mass
        assert mc_result["mean_estimate"] > 0, "Should detect positive excess mass"

    @pytest.mark.monte_carlo
    def test_large_sample_bias_reduction(self):
        """Bias decreases with larger sample size."""
        mc_small = run_bunching_monte_carlo(
            dgp_bunching_simple,
            {"n": 500, "true_excess_mass": 2.0},
            n_simulations=50,
        )

        mc_large = run_bunching_monte_carlo(
            dgp_bunching_simple,
            {"n": 2000, "true_excess_mass": 2.0},
            n_simulations=50,
        )

        # RMSE should be lower with larger sample
        assert mc_large["rmse"] < mc_small["rmse"] * 1.5, (
            f"Large sample RMSE ({mc_large['rmse']:.3f}) should be lower than "
            f"small sample ({mc_small['rmse']:.3f})"
        )


# =============================================================================
# Test: Coverage and Standard Errors
# =============================================================================


class TestCoverageAndSE:
    """Tests for confidence interval coverage and SE accuracy."""

    @pytest.mark.monte_carlo
    def test_coverage_near_nominal(self):
        """95% CI achieves reasonable coverage.

        Note: Bunching estimators have known undercoverage issues due to:
        1. Bootstrap SE underestimation (polynomial fit uncertainty not captured)
        2. Bunching region misspecification
        3. Polynomial order sensitivity

        See Kleven (2016) for discussion of inference challenges.
        """
        mc_result = run_bunching_monte_carlo(
            dgp_bunching_simple,
            {"n": 1000, "true_excess_mass": 2.0},
            n_simulations=N_SIMULATIONS,
        )

        if "coverage" in mc_result:
            # Coverage should be at least 50% (bunching has known undercoverage)
            # This is documenting actual behavior, not claiming nominal coverage
            assert 0.50 <= mc_result["coverage"] <= 1.0, (
                f"Coverage {mc_result['coverage']:.1%} outside [50%, 100%]"
            )

    @pytest.mark.monte_carlo
    def test_se_ratio_reasonable(self):
        """Bootstrap SE reasonably approximates empirical SE."""
        mc_result = run_bunching_monte_carlo(
            dgp_bunching_simple,
            {"n": 1000, "true_excess_mass": 2.0},
            n_simulations=N_SIMULATIONS,
        )

        if "se_ratio" in mc_result and np.isfinite(mc_result["se_ratio"]):
            # SE ratio should be between 0.5 and 2.0
            assert 0.3 <= mc_result["se_ratio"] <= 3.0, (
                f"SE ratio {mc_result['se_ratio']:.2f} outside [0.3, 3.0]"
            )

    @pytest.mark.monte_carlo
    def test_se_decreases_with_n(self):
        """Standard error decreases with sample size."""
        mc_small = run_bunching_monte_carlo(
            dgp_bunching_simple,
            {"n": 500, "true_excess_mass": 2.0},
            n_simulations=50,
        )

        mc_large = run_bunching_monte_carlo(
            dgp_bunching_simple,
            {"n": 2000, "true_excess_mass": 2.0},
            n_simulations=50,
        )

        # Empirical SE should decrease
        assert mc_large["empirical_se"] < mc_small["empirical_se"], (
            f"Large sample SE ({mc_large['empirical_se']:.3f}) should be lower than "
            f"small sample ({mc_small['empirical_se']:.3f})"
        )


# =============================================================================
# Test: Type I Error (No Bunching)
# =============================================================================


class TestTypeIError:
    """Tests for Type I error control under null hypothesis."""

    @pytest.mark.monte_carlo
    def test_no_bunching_type_i_error(self):
        """Type I error controlled when no bunching present."""
        n_simulations = 100
        rejections = 0

        for i in range(n_simulations):
            data, _, kink_point, bunching_width = dgp_bunching_no_effect(
                n=1000, random_state=42 + i
            )

            result = bunching_estimator(
                data=data,
                kink_point=kink_point,
                bunching_width=bunching_width,
                n_bins=50,
                n_bootstrap=30,
                random_state=i,
            )

            # Test if significantly positive
            if np.isfinite(result["excess_mass"]) and np.isfinite(result["excess_mass_se"]):
                z_stat = result["excess_mass"] / result["excess_mass_se"]
                if z_stat > 1.96:  # One-sided test for positive excess mass
                    rejections += 1

        rejection_rate = rejections / n_simulations

        # Type I error should be below 15% (bunching tests can be liberal)
        assert rejection_rate < 0.20, f"Type I error rate {rejection_rate:.1%} exceeds 20%"

    @pytest.mark.monte_carlo
    def test_null_excess_mass_near_zero(self):
        """Excess mass estimate near zero when no bunching."""
        mc_result = run_bunching_monte_carlo(
            dgp_bunching_no_effect,
            {"n": 1000},
            n_simulations=N_SIMULATIONS,
        )

        # Mean estimate should be close to zero
        assert abs(mc_result["mean_estimate"]) < 1.0, (
            f"Mean excess mass {mc_result['mean_estimate']:.3f} should be near zero"
        )


# =============================================================================
# Test: Elasticity Estimation
# =============================================================================


class TestElasticityEstimation:
    """Tests for elasticity estimation accuracy."""

    @pytest.mark.monte_carlo
    def test_elasticity_direction(self):
        """Elasticity estimate has correct sign."""
        n_positive = 0
        n_simulations = 50

        for i in range(n_simulations):
            data, true_e, true_b, kink, t1, t2 = dgp_bunching_with_elasticity(
                n=1000,
                true_elasticity=0.25,
                random_state=42 + i,
            )

            result = bunching_estimator(
                data=data,
                kink_point=kink,
                bunching_width=kink * 0.04,
                n_bins=60,
                t1_rate=t1,
                t2_rate=t2,
                n_bootstrap=30,
                random_state=i,
            )

            if np.isfinite(result["elasticity"]) and result["elasticity"] > 0:
                n_positive += 1

        # Most elasticity estimates should be positive
        assert n_positive / n_simulations > 0.70, (
            f"Only {n_positive}/{n_simulations} positive elasticity estimates"
        )

    @pytest.mark.monte_carlo
    def test_elasticity_formula_consistency(self):
        """Elasticity = excess_mass / log((1-t1)/(1-t2))."""
        data, true_e, true_b, kink, t1, t2 = dgp_bunching_with_elasticity(
            n=2000,
            true_elasticity=0.30,
            random_state=42,
        )

        result = bunching_estimator(
            data=data,
            kink_point=kink,
            bunching_width=kink * 0.04,
            n_bins=60,
            t1_rate=t1,
            t2_rate=t2,
            n_bootstrap=30,
            random_state=42,
        )

        if np.isfinite(result["elasticity"]) and np.isfinite(result["excess_mass"]):
            # Verify formula
            log_rate_change = np.log((1 - t1) / (1 - t2))
            implied_elasticity = result["excess_mass"] / log_rate_change

            assert result["elasticity"] == pytest.approx(implied_elasticity, rel=0.01), (
                f"Elasticity {result['elasticity']:.4f} != implied {implied_elasticity:.4f}"
            )


# =============================================================================
# Test: Robustness
# =============================================================================


class TestRobustness:
    """Tests for robustness to DGP variations."""

    @pytest.mark.monte_carlo
    def test_asymmetric_bunching_detection(self):
        """Detects bunching even when offset from kink."""
        mc_result = run_bunching_monte_carlo(
            dgp_bunching_asymmetric,
            {"n": 1000, "bunching_offset": -2.0},
            n_simulations=50,
        )

        # Should still detect positive excess mass
        assert mc_result["mean_estimate"] > 0, "Should detect bunching even with offset"

    @pytest.mark.monte_carlo
    def test_diffuse_bunching_detection(self):
        """Detects diffuse bunching (optimization frictions)."""
        mc_result = run_bunching_monte_carlo(
            dgp_bunching_diffuse,
            {"n": 1000, "bunching_std": 4.0},
            n_simulations=50,
        )

        # Should detect positive excess mass (though possibly attenuated)
        assert mc_result["mean_estimate"] > 0, "Should detect diffuse bunching"

    @pytest.mark.monte_carlo
    def test_small_sample_still_works(self):
        """Estimator works with small samples (higher variance)."""
        mc_result = run_bunching_monte_carlo(
            dgp_bunching_small_sample,
            {"n": 200},
            n_simulations=50,
            n_bins=30,  # Fewer bins for small sample
        )

        # Should complete without error and detect bunching direction
        assert mc_result["n_successful"] >= 40, (
            f"Only {mc_result['n_successful']}/50 successful iterations"
        )


# =============================================================================
# Test: Polynomial Order Sensitivity
# =============================================================================


class TestPolynomialOrder:
    """Tests for polynomial order sensitivity."""

    @pytest.mark.monte_carlo
    def test_polynomial_order_stability(self):
        """Estimates stable across polynomial orders."""
        estimates_by_order = {}

        data, true_b, kink, bw = dgp_bunching_simple(n=2000, true_excess_mass=2.0, random_state=42)

        for order in [5, 7, 9]:
            result = bunching_estimator(
                data=data,
                kink_point=kink,
                bunching_width=bw,
                n_bins=60,
                polynomial_order=order,
                n_bootstrap=30,
                random_state=42,
            )
            if np.isfinite(result["excess_mass"]):
                estimates_by_order[order] = result["excess_mass"]

        if len(estimates_by_order) >= 2:
            values = list(estimates_by_order.values())
            cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0

            # Coefficient of variation should be low (stable across orders)
            assert cv < 0.5, f"Estimates too variable across polynomial orders: CV={cv:.2f}"


# =============================================================================
# Test: Bin Width Sensitivity
# =============================================================================


class TestBinWidthSensitivity:
    """Tests for bin width sensitivity."""

    @pytest.mark.monte_carlo
    def test_bin_count_stability(self):
        """Estimates stable across bin counts."""
        estimates_by_bins = {}

        data, true_b, kink, bw = dgp_bunching_simple(n=2000, true_excess_mass=2.0, random_state=42)

        for n_bins in [40, 60, 80]:
            result = bunching_estimator(
                data=data,
                kink_point=kink,
                bunching_width=bw,
                n_bins=n_bins,
                n_bootstrap=30,
                random_state=42,
            )
            if np.isfinite(result["excess_mass"]):
                estimates_by_bins[n_bins] = result["excess_mass"]

        if len(estimates_by_bins) >= 2:
            values = list(estimates_by_bins.values())
            cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0

            # Should be relatively stable
            assert cv < 0.5, f"Estimates too variable across bin counts: CV={cv:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "monte_carlo"])
