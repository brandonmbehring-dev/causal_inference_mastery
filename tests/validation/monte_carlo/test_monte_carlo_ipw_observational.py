"""Monte Carlo validation for observational IPW ATE estimator.

Tests validate statistical properties across repeated samples:
- Bias: < 0.10 (relaxed vs RCT's 0.05 due to confounding)
- Coverage: 93-97% (nominal 95%)
- SE accuracy: < 15%

Layer 3 testing: 5000 runs per DGP.
"""

import numpy as np
import pytest
from src.causal_inference.observational.ipw import ipw_ate_observational


class TestIPWObservationalMonteCarloLinearConfounding:
    """Monte Carlo validation with linear confounding."""

    def test_linear_confounding_n200(self):
        """
        DGP: Linear confounding with moderate sample size.

        Setup:
            X ~ N(0, 1)
            logit(P(T=1|X)) = 0 + 0.8*X  # Strong confounding
            Y = 3.0*T + 0.5*X + N(0,1)   # True ATE = 3.0

        Expected:
            - Bias < 0.10
            - Coverage 93-97%
            - SE accuracy < 15%
        """
        np.random.seed(42)
        n_runs = 5000
        n = 200
        true_ate = 3.0
        alpha = 0.05

        estimates = []
        ses = []
        coverage = 0

        for run in range(n_runs):
            # Generate data
            X = np.random.normal(0, 1, n)
            logit = 0.8 * X
            T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
            Y = true_ate * T + 0.5 * X + np.random.normal(0, 1, n)

            # Estimate ATE
            result = ipw_ate_observational(Y, T, X, alpha=alpha)

            estimates.append(result["estimate"])
            ses.append(result["se"])

            # Check coverage
            if result["ci_lower"] <= true_ate <= result["ci_upper"]:
                coverage += 1

        estimates = np.array(estimates)
        ses = np.array(ses)

        # Compute metrics
        bias = np.mean(estimates) - true_ate
        coverage_rate = coverage / n_runs
        se_empirical = np.std(estimates)
        se_mean = np.mean(ses)
        se_accuracy = np.abs(se_empirical - se_mean) / se_empirical

        # Assertions
        assert np.abs(bias) < 0.10, f"Bias too large: {bias:.4f}"
        assert 0.93 <= coverage_rate <= 0.975, f"Coverage rate: {coverage_rate:.4f}"
        assert se_accuracy < 0.15, f"SE accuracy: {se_accuracy:.4f}"

    def test_linear_confounding_n500(self):
        """
        DGP: Linear confounding with large sample size.

        Larger n → lower bias, better coverage.
        """
        np.random.seed(123)
        n_runs = 5000
        n = 500
        true_ate = 2.5
        alpha = 0.05

        estimates = []
        ses = []
        coverage = 0

        for run in range(n_runs):
            X = np.random.normal(0, 1, n)
            logit = 0.6 * X
            T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
            Y = true_ate * T + 0.4 * X + np.random.normal(0, 1, n)

            result = ipw_ate_observational(Y, T, X, alpha=alpha)

            estimates.append(result["estimate"])
            ses.append(result["se"])

            if result["ci_lower"] <= true_ate <= result["ci_upper"]:
                coverage += 1

        estimates = np.array(estimates)
        ses = np.array(ses)

        bias = np.mean(estimates) - true_ate
        coverage_rate = coverage / n_runs
        se_empirical = np.std(estimates)
        se_mean = np.mean(ses)
        se_accuracy = np.abs(se_empirical - se_mean) / se_empirical

        # With larger n, should have better performance
        assert np.abs(bias) < 0.08, f"Bias: {bias:.4f}"
        assert 0.93 <= coverage_rate <= 0.975, f"Coverage: {coverage_rate:.4f}"
        assert se_accuracy < 0.12, f"SE accuracy: {se_accuracy:.4f}"


class TestIPWObservationalMonteCarloMultipleConfounders:
    """Monte Carlo validation with multiple confounders."""

    def test_three_confounders_n300(self):
        """
        DGP: Three confounders affecting both treatment and outcome.

        Setup:
            X1, X2, X3 ~ N(0, 1)
            logit(P(T=1|X)) = 0 + 0.5*X1 + 0.3*X2 + 0.2*X3
            Y = 3.5*T + 0.4*X1 + 0.3*X2 + 0.2*X3 + N(0,1)
            True ATE = 3.5

        Expected:
            - Bias < 0.10
            - Coverage 93-97%
            - SE accuracy < 15%
        """
        np.random.seed(456)
        n_runs = 5000
        n = 300
        true_ate = 3.5
        alpha = 0.05

        estimates = []
        ses = []
        coverage = 0

        for run in range(n_runs):
            # Three confounders
            X1 = np.random.normal(0, 1, n)
            X2 = np.random.normal(0, 1, n)
            X3 = np.random.normal(0, 1, n)
            X = np.column_stack([X1, X2, X3])

            # All three affect treatment
            logit = 0.5 * X1 + 0.3 * X2 + 0.2 * X3
            T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)

            # All three affect outcome
            Y = true_ate * T + 0.4 * X1 + 0.3 * X2 + 0.2 * X3 + np.random.normal(0, 1, n)

            result = ipw_ate_observational(Y, T, X, alpha=alpha)

            estimates.append(result["estimate"])
            ses.append(result["se"])

            if result["ci_lower"] <= true_ate <= result["ci_upper"]:
                coverage += 1

        estimates = np.array(estimates)
        ses = np.array(ses)

        bias = np.mean(estimates) - true_ate
        coverage_rate = coverage / n_runs
        se_empirical = np.std(estimates)
        se_mean = np.mean(ses)
        se_accuracy = np.abs(se_empirical - se_mean) / se_empirical

        assert np.abs(bias) < 0.10, f"Bias: {bias:.4f}"
        assert 0.93 <= coverage_rate <= 0.975, f"Coverage: {coverage_rate:.4f}"
        assert se_accuracy < 0.15, f"SE accuracy: {se_accuracy:.4f}"


class TestIPWObservationalMonteCarloWeakConfounding:
    """Monte Carlo validation with weak confounding."""

    def test_weak_confounding_n200(self):
        """
        DGP: Weak confounding (T nearly independent of X).

        Setup:
            X ~ N(0, 1)
            logit(P(T=1|X)) = 0 + 0.2*X  # Weak confounding
            Y = 4.0*T + 0.1*X + N(0,1)
            True ATE = 4.0

        Expected:
            - Should still work (IPW handles weak confounding gracefully)
            - Bias < 0.08
            - Coverage 93-97%
        """
        np.random.seed(789)
        n_runs = 5000
        n = 200
        true_ate = 4.0
        alpha = 0.05

        estimates = []
        ses = []
        coverage = 0

        for run in range(n_runs):
            X = np.random.normal(0, 1, n)
            logit = 0.2 * X  # Weak confounding
            T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
            Y = true_ate * T + 0.1 * X + np.random.normal(0, 1, n)

            result = ipw_ate_observational(Y, T, X, alpha=alpha)

            estimates.append(result["estimate"])
            ses.append(result["se"])

            if result["ci_lower"] <= true_ate <= result["ci_upper"]:
                coverage += 1

        estimates = np.array(estimates)
        ses = np.array(ses)

        bias = np.mean(estimates) - true_ate
        coverage_rate = coverage / n_runs
        se_empirical = np.std(estimates)
        se_mean = np.mean(ses)
        se_accuracy = np.abs(se_empirical - se_mean) / se_empirical

        # Weak confounding should give better performance
        assert np.abs(bias) < 0.08, f"Bias: {bias:.4f}"
        assert 0.93 <= coverage_rate <= 0.975, f"Coverage: {coverage_rate:.4f}"
        assert se_accuracy < 0.12, f"SE accuracy: {se_accuracy:.4f}"


class TestIPWObservationalMonteCarloWithTrimming:
    """Monte Carlo validation with weight trimming."""

    def test_strong_confounding_with_trimming_n300(self):
        """
        DGP: Strong confounding with trimming at 1st/99th percentile.

        Setup:
            X ~ N(0, 1)
            logit(P(T=1|X)) = 0 + 1.2*X  # Strong confounding -> extreme propensities
            Y = 3.0*T + 0.6*X + N(0,1)
            True ATE = 3.0
            Trim at (0.01, 0.99)

        Expected:
            - Trimming introduces slight bias but reduces variance
            - Bias < 0.15 (relaxed due to trimming)
            - Coverage 92-98% (may be wider due to bias-variance tradeoff)
        """
        np.random.seed(101)
        n_runs = 5000
        n = 300
        true_ate = 3.0
        alpha = 0.05

        estimates = []
        ses = []
        coverage = 0

        for run in range(n_runs):
            X = np.random.normal(0, 1, n)
            logit = 1.2 * X  # Strong confounding
            T = (np.random.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
            Y = true_ate * T + 0.6 * X + np.random.normal(0, 1, n)

            # Trim at 1st/99th percentile
            result = ipw_ate_observational(Y, T, X, trim_at=(0.01, 0.99), alpha=alpha)

            estimates.append(result["estimate"])
            ses.append(result["se"])

            if result["ci_lower"] <= true_ate <= result["ci_upper"]:
                coverage += 1

        estimates = np.array(estimates)
        ses = np.array(ses)

        bias = np.mean(estimates) - true_ate
        coverage_rate = coverage / n_runs
        se_empirical = np.std(estimates)
        se_mean = np.mean(ses)
        se_accuracy = np.abs(se_empirical - se_mean) / se_empirical

        # Trimming may introduce bias but reduces variance
        assert np.abs(bias) < 0.15, f"Bias with trimming: {bias:.4f}"
        assert 0.92 <= coverage_rate <= 0.98, f"Coverage with trimming: {coverage_rate:.4f}"
        assert se_accuracy < 0.18, f"SE accuracy with trimming: {se_accuracy:.4f}"
