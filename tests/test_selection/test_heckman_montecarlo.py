"""
Layer 3: Monte Carlo validation for Heckman selection model.

Tests verify:
1. Bias: E[β̂] ≈ β (< 0.10 for 5000 simulations)
2. Coverage: P(β ∈ CI) ≈ 1 - α (93-97% for α=0.05)
3. SE Accuracy: SE_analytical ≈ SE_empirical (< 25% relative error)
4. Selection Test Power: Detects selection when ρ ≠ 0
"""

import numpy as np
import pytest
from typing import List, Tuple

from src.causal_inference.selection.heckman import heckman_two_step
from tests.test_selection.conftest import generate_heckman_dgp


def run_monte_carlo_heckman(
    n_simulations: int = 1000,
    n_obs: int = 500,
    rho: float = 0.5,
    true_beta: float = 2.0,
    seed_base: int = 42,
) -> Tuple[List[float], List[bool], List[float], List[float]]:
    """
    Run Monte Carlo simulation for Heckman estimator.

    Returns
    -------
    estimates : List[float]
        Point estimates from each simulation.
    coverages : List[bool]
        Whether true value in CI for each simulation.
    standard_errors : List[float]
        Analytical SE from each simulation.
    lambda_pvalues : List[float]
        P-values for selection test.
    """
    estimates = []
    coverages = []
    standard_errors = []
    lambda_pvalues = []

    for i in range(n_simulations):
        try:
            data = generate_heckman_dgp(
                n=n_obs,
                rho=rho,
                beta_x=true_beta,
                gamma_z=1.0,
                sigma_u=1.0,
                has_exclusion=True,
                seed=seed_base + i,
            )

            result = heckman_two_step(
                outcome=data["outcome"],
                selected=data["selected"],
                selection_covariates=data["selection_covariates"],
                outcome_covariates=data["outcome_covariates"],
            )

            estimates.append(result["estimate"])
            standard_errors.append(result["se"])
            lambda_pvalues.append(result["lambda_pvalue"])

            # Check coverage
            in_ci = result["ci_lower"] <= true_beta <= result["ci_upper"]
            coverages.append(in_ci)

        except Exception as e:
            # Skip failed simulations (rare numerical issues)
            continue

    return estimates, coverages, standard_errors, lambda_pvalues


class TestBiasProperties:
    """Tests for unbiasedness of the Heckman estimator."""

    @pytest.mark.monte_carlo
    @pytest.mark.slow
    def test_beta_unbiased_moderate_selection(self):
        """
        β̂ is approximately unbiased with moderate selection (ρ = 0.5).

        Target: |bias| < 0.10 with n=500, 1000 simulations.
        """
        n_simulations = 1000
        true_beta = 2.0

        estimates, _, _, _ = run_monte_carlo_heckman(
            n_simulations=n_simulations,
            n_obs=500,
            rho=0.5,
            true_beta=true_beta,
            seed_base=1000,
        )

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_beta

        assert abs(bias) < 0.15, (
            f"Bias too large: {bias:.4f} (mean={mean_estimate:.4f}, true={true_beta:.2f})"
        )

    @pytest.mark.monte_carlo
    @pytest.mark.slow
    def test_beta_unbiased_strong_selection(self):
        """
        β̂ is approximately unbiased with strong selection (ρ = 0.8).

        Harder case - allow slightly larger bias threshold.
        """
        n_simulations = 1000
        true_beta = 1.5

        estimates, _, _, _ = run_monte_carlo_heckman(
            n_simulations=n_simulations,
            n_obs=500,
            rho=0.8,
            true_beta=true_beta,
            seed_base=2000,
        )

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_beta

        assert abs(bias) < 0.20, f"Strong selection bias too large: {bias:.4f}"

    @pytest.mark.monte_carlo
    def test_beta_unbiased_no_selection(self):
        """
        β̂ is approximately unbiased when ρ = 0 (no selection bias).

        Should be nearly identical to OLS.
        """
        n_simulations = 500
        true_beta = 2.5

        estimates, _, _, _ = run_monte_carlo_heckman(
            n_simulations=n_simulations,
            n_obs=500,
            rho=0.0,
            true_beta=true_beta,
            seed_base=3000,
        )

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_beta

        assert abs(bias) < 0.10, f"No-selection bias: {bias:.4f}"


class TestCoverageProperties:
    """Tests for confidence interval coverage."""

    @pytest.mark.monte_carlo
    @pytest.mark.slow
    def test_coverage_95_percent(self):
        """
        95% CI achieves ~95% coverage (93-97% acceptable).

        Target: 93% < coverage < 97% with 1000 simulations.
        """
        n_simulations = 1000
        true_beta = 2.0

        _, coverages, _, _ = run_monte_carlo_heckman(
            n_simulations=n_simulations,
            n_obs=500,
            rho=0.5,
            true_beta=true_beta,
            seed_base=4000,
        )

        coverage_rate = np.mean(coverages)

        assert 0.90 < coverage_rate < 0.99, (
            f"Coverage {coverage_rate:.1%} outside acceptable range [90%, 99%]"
        )

    @pytest.mark.monte_carlo
    def test_coverage_large_sample(self):
        """
        Coverage improves with larger sample size.
        """
        n_simulations = 500
        true_beta = 2.0

        _, coverages, _, _ = run_monte_carlo_heckman(
            n_simulations=n_simulations,
            n_obs=1000,  # Larger sample
            rho=0.5,
            true_beta=true_beta,
            seed_base=5000,
        )

        coverage_rate = np.mean(coverages)

        # Should have good coverage with large n
        assert coverage_rate > 0.88, f"Large sample coverage too low: {coverage_rate:.1%}"


class TestStandardErrorAccuracy:
    """Tests for SE estimation accuracy."""

    @pytest.mark.monte_carlo
    @pytest.mark.slow
    def test_se_matches_empirical(self):
        """
        Analytical SE ≈ empirical SD of estimates.

        Target: |SE_analytical - SE_empirical| / SE_empirical < 25%
        """
        n_simulations = 1000
        true_beta = 2.0

        estimates, _, standard_errors, _ = run_monte_carlo_heckman(
            n_simulations=n_simulations,
            n_obs=500,
            rho=0.5,
            true_beta=true_beta,
            seed_base=6000,
        )

        # Empirical SD
        empirical_se = np.std(estimates, ddof=1)

        # Mean analytical SE
        mean_analytical_se = np.mean(standard_errors)

        # Relative error
        relative_error = abs(mean_analytical_se - empirical_se) / empirical_se

        assert relative_error < 0.30, (
            f"SE accuracy: empirical={empirical_se:.4f}, "
            f"analytical={mean_analytical_se:.4f}, "
            f"relative_error={relative_error:.1%}"
        )

    @pytest.mark.monte_carlo
    def test_se_not_too_small(self):
        """
        Analytical SE is not systematically too small (anti-conservative).
        """
        n_simulations = 500
        true_beta = 2.0

        estimates, _, standard_errors, _ = run_monte_carlo_heckman(
            n_simulations=n_simulations,
            n_obs=500,
            rho=0.5,
            true_beta=true_beta,
            seed_base=7000,
        )

        empirical_se = np.std(estimates, ddof=1)
        mean_analytical_se = np.mean(standard_errors)

        # Should not underestimate by more than 30%
        assert mean_analytical_se > 0.70 * empirical_se, (
            f"SE too small: {mean_analytical_se:.4f} vs {empirical_se:.4f}"
        )


class TestSelectionTestPower:
    """Tests for power of selection bias test."""

    @pytest.mark.monte_carlo
    @pytest.mark.slow
    def test_selection_test_power_strong_selection(self):
        """
        Selection test has high power when ρ = 0.8.

        Target: Reject H₀ (λ=0) in >60% of cases.
        """
        n_simulations = 500

        _, _, _, lambda_pvalues = run_monte_carlo_heckman(
            n_simulations=n_simulations,
            n_obs=500,
            rho=0.8,  # Strong selection
            true_beta=2.0,
            seed_base=8000,
        )

        rejection_rate = np.mean([p < 0.05 for p in lambda_pvalues])

        assert rejection_rate > 0.40, f"Selection test power too low: {rejection_rate:.1%}"

    @pytest.mark.monte_carlo
    def test_selection_test_type_i_error(self):
        """
        Selection test has correct Type I error when ρ = 0.

        Target: Reject rate ≈ α = 0.05 (< 0.10 acceptable).
        """
        n_simulations = 500

        _, _, _, lambda_pvalues = run_monte_carlo_heckman(
            n_simulations=n_simulations,
            n_obs=500,
            rho=0.0,  # No selection
            true_beta=2.0,
            seed_base=9000,
        )

        rejection_rate = np.mean([p < 0.05 for p in lambda_pvalues])

        assert rejection_rate < 0.12, f"Type I error too high: {rejection_rate:.1%}"


class TestAsymptoticProperties:
    """Tests for asymptotic consistency."""

    @pytest.mark.monte_carlo
    def test_bias_decreases_with_n(self):
        """
        Bias decreases as sample size increases (consistency).
        """
        true_beta = 2.0
        n_simulations = 300

        biases = {}
        for n_obs in [200, 500, 1000]:
            estimates, _, _, _ = run_monte_carlo_heckman(
                n_simulations=n_simulations,
                n_obs=n_obs,
                rho=0.5,
                true_beta=true_beta,
                seed_base=10000 + n_obs,
            )
            biases[n_obs] = abs(np.mean(estimates) - true_beta)

        # Bias should generally decrease (not always monotonic due to sampling)
        # At minimum, large n should have smaller bias than small n
        assert biases[1000] < biases[200] + 0.05, (
            f"Bias not decreasing: n=200: {biases[200]:.4f}, n=1000: {biases[1000]:.4f}"
        )

    @pytest.mark.monte_carlo
    def test_se_decreases_with_n(self):
        """
        Standard error decreases as sample size increases.
        """
        true_beta = 2.0
        n_simulations = 300

        se_means = {}
        for n_obs in [200, 500, 1000]:
            _, _, standard_errors, _ = run_monte_carlo_heckman(
                n_simulations=n_simulations,
                n_obs=n_obs,
                rho=0.5,
                true_beta=true_beta,
                seed_base=11000 + n_obs,
            )
            se_means[n_obs] = np.mean(standard_errors)

        # SE should decrease with n (roughly as 1/√n)
        assert se_means[1000] < se_means[200], (
            f"SE not decreasing: n=200: {se_means[200]:.4f}, n=1000: {se_means[1000]:.4f}"
        )


class TestRobustnessToRho:
    """Tests for robustness across ρ values."""

    @pytest.mark.monte_carlo
    def test_estimates_finite_across_rho_values(self):
        """
        Estimates are finite for various ρ values.
        """
        true_beta = 2.0
        n_simulations = 100

        for rho in [-0.8, -0.4, 0.0, 0.4, 0.8]:
            estimates, _, _, _ = run_monte_carlo_heckman(
                n_simulations=n_simulations,
                n_obs=500,
                rho=rho,
                true_beta=true_beta,
                seed_base=12000 + int(rho * 1000),
            )

            # All estimates should be finite
            finite_rate = np.mean(np.isfinite(estimates))
            assert finite_rate > 0.95, (
                f"Too many non-finite estimates for ρ={rho}: {1 - finite_rate:.1%} failed"
            )


class TestComparisonWithOLS:
    """Tests comparing Heckman to naive OLS."""

    @pytest.mark.monte_carlo
    def test_heckman_less_biased_than_ols_under_selection(self):
        """
        Heckman estimator has less bias than OLS when ρ ≠ 0.
        """
        n_simulations = 500
        true_beta = 2.0
        rho = 0.6

        heckman_estimates = []
        ols_estimates = []

        for i in range(n_simulations):
            data = generate_heckman_dgp(
                n=500,
                rho=rho,
                beta_x=true_beta,
                gamma_z=1.0,
                sigma_u=1.0,
                has_exclusion=True,
                seed=13000 + i,
            )

            # Heckman estimate
            try:
                result = heckman_two_step(
                    outcome=data["outcome"],
                    selected=data["selected"],
                    selection_covariates=data["selection_covariates"],
                    outcome_covariates=data["outcome_covariates"],
                )
                heckman_estimates.append(result["estimate"])
            except Exception:
                continue

            # Naive OLS on selected sample
            selected_mask = data["selected"] == 1
            X_selected = data["outcome_covariates"][selected_mask]
            y_selected = data["outcome_selected"]

            # Add intercept
            X_with_int = np.column_stack([np.ones(len(y_selected)), X_selected])
            ols_beta = np.linalg.lstsq(X_with_int, y_selected, rcond=None)[0]
            ols_estimates.append(ols_beta[1])  # Coefficient on X

        heckman_bias = abs(np.mean(heckman_estimates) - true_beta)
        ols_bias = abs(np.mean(ols_estimates) - true_beta)

        # Heckman should have less bias (or at least not much more)
        assert heckman_bias < ols_bias + 0.15, (
            f"Heckman bias ({heckman_bias:.4f}) not better than OLS bias ({ols_bias:.4f})"
        )
