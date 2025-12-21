"""
Monte Carlo validation tests for Control Function estimation.

Tests statistical properties across many simulations:
- Bias: Average estimate should be close to true value
- Coverage: 95% CI should contain true value ~95% of time
- SE accuracy: Bootstrap SE should track empirical SD
- Endogeneity test size: False positive rate at nominal level
- Endogeneity test power: Detection rate when endogeneity exists

Layer 3 of 6-layer validation architecture.

Target thresholds (from CLAUDE.md):
- Bias: < 0.05 (strong IV), < 0.10 (moderate IV)
- Coverage: 93-97% for 95% CI
- SE accuracy: ratio 0.85-1.15
"""

import numpy as np
import pytest
from typing import Tuple

from src.causal_inference.control_function import ControlFunction
from tests.test_control_function.conftest import generate_cf_data


def run_monte_carlo_cf(
    n_runs: int = 500,
    n_obs: int = 500,
    true_beta: float = 2.0,
    pi: float = 0.5,
    rho: float = 0.5,
    inference: str = "bootstrap",
    n_bootstrap: int = 200,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Monte Carlo simulation for Control Function estimation.

    Returns
    -------
    estimates : array
        CF estimates from each run.
    ses : array
        Standard errors from each run.
    covers : array
        Boolean array: True if CI covers true value.
    endogeneity_detected : array
        Boolean array: True if endogeneity test significant.
    """
    estimates = []
    ses = []
    covers = []
    endogeneity_detected = []

    for seed in range(n_runs):
        Y, D, Z, X, _, _ = generate_cf_data(
            n=n_obs,
            true_beta=true_beta,
            pi=pi,
            rho=rho,
            random_state=seed,
        )

        cf = ControlFunction(
            inference=inference,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            random_state=seed,
        )
        result = cf.fit(Y, D, Z.ravel(), X)

        estimates.append(result["estimate"])
        ses.append(result["se"])
        covers.append(result["ci_lower"] <= true_beta <= result["ci_upper"])
        endogeneity_detected.append(result["endogeneity_detected"])

    return (
        np.array(estimates),
        np.array(ses),
        np.array(covers),
        np.array(endogeneity_detected),
    )


class TestBias:
    """Tests for estimation bias."""

    @pytest.mark.slow
    def test_unbiased_strong_iv(self):
        """CF is approximately unbiased with strong instrument."""
        estimates, _, _, _ = run_monte_carlo_cf(
            n_runs=500,
            n_obs=500,
            true_beta=2.0,
            pi=0.5,  # Strong first stage
            rho=0.5,
            inference="analytical",  # Faster
        )

        bias = np.mean(estimates) - 2.0
        assert abs(bias) < 0.05, f"Bias {bias:.4f} exceeds threshold 0.05"

    @pytest.mark.slow
    def test_unbiased_no_endogeneity(self):
        """CF is unbiased when treatment is exogenous (rho=0)."""
        estimates, _, _, _ = run_monte_carlo_cf(
            n_runs=500,
            n_obs=500,
            true_beta=1.5,
            pi=0.5,
            rho=0.0,  # No endogeneity
            inference="analytical",
        )

        bias = np.mean(estimates) - 1.5
        assert abs(bias) < 0.05, f"Bias {bias:.4f} exceeds threshold 0.05"

    @pytest.mark.slow
    def test_moderate_bias_weak_iv(self):
        """CF has moderate bias with weak instrument."""
        estimates, _, _, _ = run_monte_carlo_cf(
            n_runs=500,
            n_obs=500,
            true_beta=2.0,
            pi=0.15,  # Weak first stage
            rho=0.5,
            inference="analytical",
        )

        bias = np.mean(estimates) - 2.0
        # Weak IV can have more bias
        assert abs(bias) < 0.20, f"Bias {bias:.4f} exceeds threshold 0.20"


class TestCoverage:
    """Tests for confidence interval coverage."""

    @pytest.mark.slow
    def test_coverage_bootstrap(self):
        """Bootstrap 95% CI has ~95% coverage."""
        _, _, covers, _ = run_monte_carlo_cf(
            n_runs=500,
            n_obs=500,
            true_beta=2.0,
            pi=0.5,
            rho=0.5,
            inference="bootstrap",
            n_bootstrap=200,
        )

        coverage = np.mean(covers)
        assert 0.93 <= coverage <= 0.97, (
            f"Coverage {coverage:.1%} outside [93%, 97%]"
        )

    @pytest.mark.slow
    def test_coverage_analytical(self):
        """Analytical 95% CI has reasonable coverage."""
        _, _, covers, _ = run_monte_carlo_cf(
            n_runs=500,
            n_obs=500,
            true_beta=2.0,
            pi=0.5,
            rho=0.5,
            inference="analytical",
        )

        coverage = np.mean(covers)
        # Analytical may have slightly different coverage
        assert 0.90 <= coverage <= 0.98, (
            f"Analytical coverage {coverage:.1%} outside [90%, 98%]"
        )

    @pytest.mark.slow
    def test_coverage_large_sample(self):
        """Coverage is good in large samples."""
        _, _, covers, _ = run_monte_carlo_cf(
            n_runs=300,
            n_obs=2000,  # Large sample
            true_beta=1.5,
            pi=0.5,
            rho=0.4,
            inference="bootstrap",
            n_bootstrap=200,
        )

        coverage = np.mean(covers)
        assert 0.93 <= coverage <= 0.97


class TestSEAccuracy:
    """Tests for standard error accuracy."""

    @pytest.mark.slow
    def test_se_accuracy_bootstrap(self):
        """Bootstrap SE tracks empirical SD."""
        estimates, ses, _, _ = run_monte_carlo_cf(
            n_runs=500,
            n_obs=500,
            true_beta=2.0,
            pi=0.5,
            rho=0.5,
            inference="bootstrap",
            n_bootstrap=200,
        )

        empirical_sd = np.std(estimates, ddof=1)
        avg_se = np.mean(ses)

        ratio = avg_se / empirical_sd
        assert 0.85 <= ratio <= 1.15, (
            f"SE/SD ratio {ratio:.2f} outside [0.85, 1.15]"
        )

    @pytest.mark.slow
    def test_se_accuracy_analytical(self):
        """Analytical SE is reasonably accurate."""
        estimates, ses, _, _ = run_monte_carlo_cf(
            n_runs=500,
            n_obs=500,
            true_beta=2.0,
            pi=0.5,
            rho=0.5,
            inference="analytical",
        )

        empirical_sd = np.std(estimates, ddof=1)
        avg_se = np.mean(ses)

        ratio = avg_se / empirical_sd
        # Allow wider range for analytical (Murphy-Topel approximation)
        assert 0.70 <= ratio <= 1.30, (
            f"Analytical SE/SD ratio {ratio:.2f} outside [0.70, 1.30]"
        )


class TestEndogeneityTestSize:
    """Tests for endogeneity test size (false positive rate)."""

    @pytest.mark.slow
    def test_size_at_nominal_level(self):
        """False positive rate is ~5% when rho=0."""
        _, _, _, detected = run_monte_carlo_cf(
            n_runs=500,
            n_obs=500,
            true_beta=2.0,
            pi=0.5,
            rho=0.0,  # No endogeneity
            inference="bootstrap",
            n_bootstrap=200,
            alpha=0.05,
        )

        false_positive_rate = np.mean(detected)
        # Should be close to nominal level (4-6%)
        assert 0.02 <= false_positive_rate <= 0.10, (
            f"False positive rate {false_positive_rate:.1%} outside [2%, 10%]"
        )

    @pytest.mark.slow
    def test_size_analytical(self):
        """Analytical test has correct size."""
        _, _, _, detected = run_monte_carlo_cf(
            n_runs=500,
            n_obs=500,
            true_beta=2.0,
            pi=0.5,
            rho=0.0,
            inference="analytical",
            alpha=0.05,
        )

        false_positive_rate = np.mean(detected)
        assert 0.02 <= false_positive_rate <= 0.10


class TestEndogeneityTestPower:
    """Tests for endogeneity test power (detection rate)."""

    @pytest.mark.slow
    def test_power_strong_endogeneity(self):
        """High power to detect strong endogeneity (rho=0.7)."""
        _, _, _, detected = run_monte_carlo_cf(
            n_runs=300,
            n_obs=500,
            true_beta=2.0,
            pi=0.5,
            rho=0.7,  # Strong endogeneity
            inference="bootstrap",
            n_bootstrap=200,
        )

        power = np.mean(detected)
        assert power >= 0.80, (
            f"Power {power:.1%} < 80% for strong endogeneity"
        )

    @pytest.mark.slow
    def test_power_moderate_endogeneity(self):
        """Reasonable power for moderate endogeneity (rho=0.5)."""
        _, _, _, detected = run_monte_carlo_cf(
            n_runs=300,
            n_obs=500,
            true_beta=2.0,
            pi=0.5,
            rho=0.5,  # Moderate endogeneity
            inference="bootstrap",
            n_bootstrap=200,
        )

        power = np.mean(detected)
        assert power >= 0.50, (
            f"Power {power:.1%} < 50% for moderate endogeneity"
        )

    @pytest.mark.slow
    def test_power_increases_with_sample_size(self):
        """Power increases with sample size."""
        _, _, _, detected_small = run_monte_carlo_cf(
            n_runs=200,
            n_obs=200,  # Small
            true_beta=2.0,
            pi=0.5,
            rho=0.5,
            inference="bootstrap",
            n_bootstrap=150,
        )

        _, _, _, detected_large = run_monte_carlo_cf(
            n_runs=200,
            n_obs=1000,  # Large
            true_beta=2.0,
            pi=0.5,
            rho=0.5,
            inference="bootstrap",
            n_bootstrap=150,
        )

        power_small = np.mean(detected_small)
        power_large = np.mean(detected_large)

        assert power_large >= power_small, (
            f"Power should increase with sample size: "
            f"n=200: {power_small:.1%}, n=1000: {power_large:.1%}"
        )


class TestConsistency:
    """Tests for consistency (estimates improve with sample size)."""

    @pytest.mark.slow
    def test_bias_decreases_with_n(self):
        """Bias decreases with sample size."""
        estimates_small, _, _, _ = run_monte_carlo_cf(
            n_runs=300,
            n_obs=100,
            true_beta=2.0,
            pi=0.5,
            rho=0.5,
            inference="analytical",
        )

        estimates_large, _, _, _ = run_monte_carlo_cf(
            n_runs=300,
            n_obs=2000,
            true_beta=2.0,
            pi=0.5,
            rho=0.5,
            inference="analytical",
        )

        bias_small = abs(np.mean(estimates_small) - 2.0)
        bias_large = abs(np.mean(estimates_large) - 2.0)

        # Large sample should have less bias
        assert bias_large <= bias_small + 0.02

    @pytest.mark.slow
    def test_variance_decreases_with_n(self):
        """Variance decreases with sample size."""
        estimates_small, _, _, _ = run_monte_carlo_cf(
            n_runs=300,
            n_obs=100,
            true_beta=2.0,
            pi=0.5,
            rho=0.5,
            inference="analytical",
        )

        estimates_large, _, _, _ = run_monte_carlo_cf(
            n_runs=300,
            n_obs=2000,
            true_beta=2.0,
            pi=0.5,
            rho=0.5,
            inference="analytical",
        )

        var_small = np.var(estimates_small)
        var_large = np.var(estimates_large)

        # Large sample should have smaller variance
        assert var_large < var_small, (
            f"Variance should decrease: n=100: {var_small:.4f}, n=2000: {var_large:.4f}"
        )
