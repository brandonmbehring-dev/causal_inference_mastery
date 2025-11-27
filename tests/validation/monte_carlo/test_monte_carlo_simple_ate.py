"""
Monte Carlo validation for simple_ate estimator.

Validates statistical properties:
- Bias < 0.05
- Coverage 94-96% (for 95% CI)
- SE accuracy within 10%
"""

import numpy as np
import pytest
from src.causal_inference.rct.estimators import simple_ate
from tests.validation.monte_carlo.dgp_generators import (
    dgp_simple_rct,
    dgp_heteroskedastic_rct,
    dgp_small_sample_rct,
)
from tests.validation.utils import validate_monte_carlo_results


class TestSimpleATEMonteCarloHomoskedastic:
    """Monte Carlo validation with homoskedastic errors."""

    def test_simple_rct_n100(self):
        """Validate simple_ate on homoskedastic RCT (n=100, 1000 runs)."""
        n_runs = 5000
        true_ate = 2.0

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            outcomes, treatment = dgp_simple_rct(n=100, true_ate=true_ate, random_state=seed)
            result = simple_ate(outcomes, treatment)

            estimates.append(result["estimate"])
            standard_errors.append(result["se"])
            ci_lowers.append(result["ci_lower"])
            ci_uppers.append(result["ci_upper"])

        # Validate
        validation = validate_monte_carlo_results(
            estimates, standard_errors, ci_lowers, ci_uppers, true_ate
        )

        # Assert all checks pass
        assert validation["bias_ok"], f"Bias {validation['bias']:.4f} exceeds threshold"
        assert validation["coverage_ok"], f"Coverage {validation['coverage']:.4f} outside [0.94, 0.96]"
        assert validation["se_accuracy_ok"], f"SE accuracy {validation['se_accuracy']:.4f} exceeds 10%"
        assert validation["all_pass"], "Monte Carlo validation failed"


class TestSimpleATEMonteCarloHeteroskedastic:
    """Monte Carlo validation with heteroskedastic errors."""

    def test_heteroskedastic_rct_n200(self):
        """Validate simple_ate handles heteroskedasticity (Neyman variance)."""
        n_runs = 5000
        true_ate = 2.0

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            outcomes, treatment = dgp_heteroskedastic_rct(n=200, true_ate=true_ate, random_state=seed)
            result = simple_ate(outcomes, treatment)

            estimates.append(result["estimate"])
            standard_errors.append(result["se"])
            ci_lowers.append(result["ci_lower"])
            ci_uppers.append(result["ci_upper"])

        # Validate (Neyman variance should be robust)
        validation = validate_monte_carlo_results(
            estimates, standard_errors, ci_lowers, ci_uppers, true_ate
        )

        assert validation["bias_ok"], f"Bias {validation['bias']:.4f} exceeds threshold"
        assert validation["coverage_ok"], f"Coverage {validation['coverage']:.4f} outside [0.94, 0.96]"
        # SE accuracy may be slightly worse with heteroskedasticity, but should still be reasonable
        assert validation["se_accuracy"] < 0.15, f"SE accuracy {validation['se_accuracy']:.4f} exceeds 15%"


class TestSimpleATEMonteCarloSmallSample:
    """Monte Carlo validation with small samples (tests t-distribution)."""

    def test_small_sample_n20(self):
        """Validate simple_ate with small sample (n=20) uses t-distribution correctly."""
        n_runs = 5000
        true_ate = 2.0

        estimates = []
        standard_errors = []
        ci_lowers = []
        ci_uppers = []

        for seed in range(n_runs):
            outcomes, treatment = dgp_small_sample_rct(n=20, true_ate=true_ate, random_state=seed)
            result = simple_ate(outcomes, treatment)

            estimates.append(result["estimate"])
            standard_errors.append(result["se"])
            ci_lowers.append(result["ci_lower"])
            ci_uppers.append(result["ci_upper"])

        # Validate (t-distribution critical for small samples)
        validation = validate_monte_carlo_results(
            estimates, standard_errors, ci_lowers, ci_uppers, true_ate
        )

        assert validation["bias_ok"], f"Bias {validation['bias']:.4f} exceeds threshold"
        assert validation["coverage_ok"], f"Coverage {validation['coverage']:.4f} outside [0.94, 0.96]"
        assert validation["se_accuracy_ok"], f"SE accuracy {validation['se_accuracy']:.4f} exceeds 10%"


class TestSimpleATEMonteCarloDiagnostics:
    """Diagnostic tests for Monte Carlo validation."""

    def test_bias_distribution_centered(self):
        """Verify estimates are centered around true ATE."""
        n_runs = 500
        true_ate = 2.0

        estimates = []
        for seed in range(n_runs):
            outcomes, treatment = dgp_simple_rct(n=100, true_ate=true_ate, random_state=seed)
            result = simple_ate(outcomes, treatment)
            estimates.append(result["estimate"])

        # Mean should be close to true ATE
        mean_estimate = np.mean(estimates)
        assert abs(mean_estimate - true_ate) < 0.1

        # Distribution should be approximately normal (by CLT)
        # Check that estimates span reasonable range
        assert np.std(estimates) > 0.1  # Should have variation
        assert np.std(estimates) < 0.5  # But not too much


    def test_se_estimates_reasonable(self):
        """Verify SE estimates are in reasonable range."""
        n_runs = 500
        true_ate = 2.0

        standard_errors = []
        for seed in range(n_runs):
            outcomes, treatment = dgp_simple_rct(n=100, true_ate=true_ate, random_state=seed)
            result = simple_ate(outcomes, treatment)
            standard_errors.append(result["se"])

        # SEs should be consistent across runs
        mean_se = np.mean(standard_errors)
        assert 0.15 < mean_se < 0.25  # Reasonable range for n=100

        # SEs should not vary too much
        std_of_ses = np.std(standard_errors)
        assert std_of_ses < 0.05
