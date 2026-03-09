"""
Monte Carlo validation tests for Quantile Treatment Effects estimators.

Validates:
1. Bias < threshold across 5,000 simulations
2. CI coverage in [93%, 97%] range
3. SE accuracy within 15% of empirical SE
"""

import numpy as np
import pytest
from typing import Callable, Dict, Any

from tests.validation.monte_carlo.dgp_qte import (
    generate_homogeneous_qte_dgp,
    generate_heterogeneous_qte_dgp,
    generate_qte_with_covariates_dgp,
    generate_location_scale_shift_dgp,
    generate_extreme_quantile_dgp,
)

from src.causal_inference.qte import (
    unconditional_qte,
    unconditional_qte_band,
    conditional_qte,
    rif_qte,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def run_monte_carlo_qte(
    dgp_fn: Callable,
    estimator_fn: Callable,
    n_simulations: int = 5000,
    true_qte: float = 2.0,
    quantile: float = 0.5,
    n_per_sim: int = 500,
    **estimator_kwargs,
) -> Dict[str, float]:
    """
    Run Monte Carlo simulation for QTE estimator.

    Returns dictionary with bias, coverage, se_ratio.
    """
    estimates = []
    covers_true = []
    reported_ses = []

    for sim in range(n_simulations):
        seed = sim + 1
        outcome, treatment, _ = dgp_fn(n=n_per_sim, seed=seed)

        result = estimator_fn(
            outcome,
            treatment,
            quantile=quantile,
            random_state=seed,
            **estimator_kwargs,
        )

        estimates.append(result["tau_q"])
        covers_true.append(result["ci_lower"] <= true_qte <= result["ci_upper"])
        reported_ses.append(result["se"])

    estimates = np.array(estimates)
    covers_true = np.array(covers_true)
    reported_ses = np.array(reported_ses)

    empirical_se = np.std(estimates, ddof=1)
    mean_reported_se = np.mean(reported_ses)

    return {
        "bias": np.mean(estimates) - true_qte,
        "abs_bias": np.abs(np.mean(estimates) - true_qte),
        "coverage": np.mean(covers_true),
        "se_ratio": mean_reported_se / empirical_se if empirical_se > 0 else np.inf,
        "empirical_se": empirical_se,
        "mean_reported_se": mean_reported_se,
        "mean_estimate": np.mean(estimates),
    }


# =============================================================================
# UNCONDITIONAL QTE TESTS
# =============================================================================


class TestUnconditionalQTEMonteCarlo:
    """Monte Carlo validation for unconditional QTE estimator."""

    @pytest.mark.slow
    @pytest.mark.monte_carlo
    def test_median_unbiased_homogeneous(self):
        """Unconditional QTE at median is unbiased with homogeneous effects."""
        results = run_monte_carlo_qte(
            dgp_fn=generate_homogeneous_qte_dgp,
            estimator_fn=unconditional_qte,
            n_simulations=5000,
            true_qte=2.0,
            quantile=0.5,
            n_per_sim=500,
            n_bootstrap=500,
        )

        assert results["abs_bias"] < 0.05, f"Bias {results['bias']:.4f} exceeds 0.05"
        assert 0.93 < results["coverage"] < 0.97, (
            f"Coverage {results['coverage']:.2%} outside [93%, 97%]"
        )

    @pytest.mark.slow
    @pytest.mark.monte_carlo
    def test_q25_unbiased_homogeneous(self):
        """Unconditional QTE at 25th percentile is unbiased."""
        results = run_monte_carlo_qte(
            dgp_fn=generate_homogeneous_qte_dgp,
            estimator_fn=unconditional_qte,
            n_simulations=5000,
            true_qte=2.0,
            quantile=0.25,
            n_per_sim=500,
            n_bootstrap=500,
        )

        assert results["abs_bias"] < 0.05, f"Bias {results['bias']:.4f} exceeds 0.05"
        assert 0.93 < results["coverage"] < 0.97, (
            f"Coverage {results['coverage']:.2%} outside [93%, 97%]"
        )

    @pytest.mark.slow
    @pytest.mark.monte_carlo
    def test_q75_unbiased_homogeneous(self):
        """Unconditional QTE at 75th percentile is unbiased."""
        results = run_monte_carlo_qte(
            dgp_fn=generate_homogeneous_qte_dgp,
            estimator_fn=unconditional_qte,
            n_simulations=5000,
            true_qte=2.0,
            quantile=0.75,
            n_per_sim=500,
            n_bootstrap=500,
        )

        assert results["abs_bias"] < 0.05, f"Bias {results['bias']:.4f} exceeds 0.05"
        assert 0.93 < results["coverage"] < 0.97, (
            f"Coverage {results['coverage']:.2%} outside [93%, 97%]"
        )

    @pytest.mark.slow
    @pytest.mark.monte_carlo
    def test_extreme_quantile_q10(self):
        """Extreme quantile (10th percentile) has wider tolerance."""
        results = run_monte_carlo_qte(
            dgp_fn=generate_homogeneous_qte_dgp,
            estimator_fn=unconditional_qte,
            n_simulations=5000,
            true_qte=2.0,
            quantile=0.10,
            n_per_sim=500,
            n_bootstrap=500,
        )

        # Wider tolerance for extreme quantiles
        assert results["abs_bias"] < 0.10, f"Bias {results['bias']:.4f} exceeds 0.10"
        assert 0.90 < results["coverage"] < 0.98, (
            f"Coverage {results['coverage']:.2%} outside [90%, 98%]"
        )

    @pytest.mark.slow
    @pytest.mark.monte_carlo
    def test_extreme_quantile_q90(self):
        """Extreme quantile (90th percentile) has wider tolerance."""
        results = run_monte_carlo_qte(
            dgp_fn=generate_homogeneous_qte_dgp,
            estimator_fn=unconditional_qte,
            n_simulations=5000,
            true_qte=2.0,
            quantile=0.90,
            n_per_sim=500,
            n_bootstrap=500,
        )

        assert results["abs_bias"] < 0.10, f"Bias {results['bias']:.4f} exceeds 0.10"
        assert 0.90 < results["coverage"] < 0.98, (
            f"Coverage {results['coverage']:.2%} outside [90%, 98%]"
        )


# =============================================================================
# CONDITIONAL QTE TESTS
# =============================================================================


class TestConditionalQTEMonteCarlo:
    """Monte Carlo validation for conditional QTE estimator."""

    @pytest.mark.slow
    @pytest.mark.monte_carlo
    def test_median_with_covariates_unbiased(self):
        """Conditional QTE at median is unbiased when controlling for covariates."""
        estimates = []
        covers_true = []
        true_qte = 2.0

        for sim in range(2000):  # Fewer sims for conditional (slower)
            seed = sim + 1
            outcome, treatment, covariates, _ = generate_qte_with_covariates_dgp(
                n=500, p=3, true_ate=2.0, seed=seed
            )

            result = conditional_qte(
                outcome,
                treatment,
                covariates,
                quantile=0.5,
            )

            estimates.append(result["tau_q"])
            covers_true.append(result["ci_lower"] <= true_qte <= result["ci_upper"])

        estimates = np.array(estimates)
        bias = np.mean(estimates) - true_qte
        coverage = np.mean(covers_true)

        assert abs(bias) < 0.10, f"Bias {bias:.4f} exceeds 0.10"
        assert 0.92 < coverage < 0.98, f"Coverage {coverage:.2%} outside [92%, 98%]"


# =============================================================================
# RIF QTE TESTS
# =============================================================================


class TestRIFQTEMonteCarlo:
    """Monte Carlo validation for RIF-OLS QTE estimator."""

    @pytest.mark.slow
    @pytest.mark.monte_carlo
    def test_median_rif_unbiased(self):
        """RIF-OLS QTE at median is unbiased."""
        results = run_monte_carlo_qte(
            dgp_fn=generate_homogeneous_qte_dgp,
            estimator_fn=rif_qte,
            n_simulations=2000,  # Fewer sims (RIF is slower)
            true_qte=2.0,
            quantile=0.5,
            n_per_sim=500,
            n_bootstrap=300,
        )

        # RIF has higher variance, use wider tolerance
        assert results["abs_bias"] < 0.15, f"Bias {results['bias']:.4f} exceeds 0.15"
        assert 0.90 < results["coverage"] < 0.98, (
            f"Coverage {results['coverage']:.2%} outside [90%, 98%]"
        )


# =============================================================================
# HETEROGENEOUS EFFECTS TESTS
# =============================================================================


class TestHeterogeneousQTEMonteCarlo:
    """Monte Carlo validation with heterogeneous treatment effects."""

    @pytest.mark.slow
    @pytest.mark.monte_carlo
    def test_location_scale_shift_median(self):
        """Test QTE estimation with location-scale treatment effect at median."""
        # With location_shift=2.0, scale_ratio=1.5:
        # QTE(0.5) = 2.0 + (1.5 - 1) * 0 = 2.0 (median of standard normal is 0)
        true_qte_median = 2.0

        estimates = []
        covers_true = []

        for sim in range(2000):
            seed = sim + 1
            outcome, treatment, true_qtes = generate_location_scale_shift_dgp(
                n=500, location_shift=2.0, scale_ratio=1.5, seed=seed
            )

            result = unconditional_qte(
                outcome,
                treatment,
                quantile=0.5,
                n_bootstrap=500,
                random_state=seed,
            )

            estimates.append(result["tau_q"])
            covers_true.append(result["ci_lower"] <= true_qte_median <= result["ci_upper"])

        estimates = np.array(estimates)
        bias = np.mean(estimates) - true_qte_median
        coverage = np.mean(covers_true)

        assert abs(bias) < 0.10, f"Bias {bias:.4f} exceeds 0.10"
        assert 0.92 < coverage < 0.98, f"Coverage {coverage:.2%} outside [92%, 98%]"

    @pytest.mark.slow
    @pytest.mark.monte_carlo
    def test_heterogeneous_detects_gradient(self):
        """
        With heterogeneous effects, QTE should increase across quantiles.
        """
        qte_25_estimates = []
        qte_75_estimates = []

        for sim in range(1000):
            seed = sim + 1
            outcome, treatment, true_qtes = generate_heterogeneous_qte_dgp(
                n=500, base_effect=2.0, heterogeneity=2.0, seed=seed
            )

            result_25 = unconditional_qte(
                outcome, treatment, quantile=0.25, n_bootstrap=300, random_state=seed
            )
            result_75 = unconditional_qte(
                outcome, treatment, quantile=0.75, n_bootstrap=300, random_state=seed
            )

            qte_25_estimates.append(result_25["tau_q"])
            qte_75_estimates.append(result_75["tau_q"])

        mean_qte_25 = np.mean(qte_25_estimates)
        mean_qte_75 = np.mean(qte_75_estimates)

        # With heterogeneity=2.0:
        # True QTE(0.25) = 2.0 + 2.0 * 0.25 = 2.5
        # True QTE(0.75) = 2.0 + 2.0 * 0.75 = 3.5
        # Difference should be ~1.0
        assert mean_qte_75 > mean_qte_25, "QTE should increase with quantile"
        gradient = mean_qte_75 - mean_qte_25
        assert gradient > 0.5, f"Gradient {gradient:.3f} too small to detect heterogeneity"


# =============================================================================
# SE ACCURACY TESTS
# =============================================================================


class TestSEAccuracy:
    """Test that reported SEs match empirical variability."""

    @pytest.mark.slow
    @pytest.mark.monte_carlo
    def test_unconditional_se_accuracy(self):
        """Reported SE should be within 15% of empirical SE."""
        results = run_monte_carlo_qte(
            dgp_fn=generate_homogeneous_qte_dgp,
            estimator_fn=unconditional_qte,
            n_simulations=3000,
            true_qte=2.0,
            quantile=0.5,
            n_per_sim=500,
            n_bootstrap=500,
        )

        # SE ratio should be close to 1.0 (within 15%)
        assert 0.85 < results["se_ratio"] < 1.15, (
            f"SE ratio {results['se_ratio']:.3f} outside [0.85, 1.15]"
        )
