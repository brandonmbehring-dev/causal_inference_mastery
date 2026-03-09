"""Monte Carlo validation for doubly robust (DR) estimation.

Tests validate statistical properties across 5000 replications:
- Bias control (mean estimate ≈ true ATE)
- Coverage (95% CI contains true ATE in 93-97.5% of runs)
- SE accuracy (empirical SD ≈ mean estimated SE)

Four DGPs test double robustness property:
1. Both models correct → Lowest variance, bias < 0.05
2. Propensity correct, outcome wrong → IPW protects, bias < 0.10
3. Outcome correct, propensity wrong → Regression protects, bias < 0.10
4. Both wrong → May be biased (no protection guarantee)
"""

import numpy as np
import pytest
from src.causal_inference.observational.doubly_robust import dr_ate


class TestMonteCarloDoublyRobustBothCorrect:
    """Monte Carlo test when both propensity and outcome models are correct."""

    def test_both_models_correct_linear_n300(self):
        """
        Monte Carlo validation with both models correctly specified (n=300).

        DGP:
            X ~ N(0, 1)
            e(X) = logistic(0.8*X)  # Propensity (correctly modeled)
            μ(X) = 2 + 0.5*X        # Outcome (correctly modeled)
            Y = τ*T + μ(X) + ε      # True ATE = 3.0

        Expected:
        - Bias < 0.05 (both models correct → ideal case)
        - Coverage 94-96%
        - SE accuracy < 10%
        - Lowest variance among all scenarios
        """
        np.random.seed(2001)
        n_sims = 5000
        n = 300
        true_ate = 3.0

        estimates = []
        ses = []
        cis_cover = []

        for sim in range(n_sims):
            X = np.random.normal(0, 1, n)

            # Propensity: linear in X (correctly modeled)
            logit_prop = 0.8 * X
            e_X = 1 / (1 + np.exp(-logit_prop))
            T = np.random.binomial(1, e_X)

            # Outcome: linear in X (correctly modeled)
            Y = true_ate * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

            result = dr_ate(Y, T, X)

            estimates.append(result["estimate"])
            ses.append(result["se"])
            cis_cover.append(result["ci_lower"] <= true_ate <= result["ci_upper"])

        # Compute statistics
        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_ate
        empirical_sd = np.std(estimates, ddof=1)
        mean_se = np.mean(ses)
        coverage = np.mean(cis_cover)

        # Bias < 0.05 (both models correct)
        assert np.abs(bias) < 0.05, f"Bias = {bias:.3f}, expected < 0.05"

        # Coverage 94-96%
        assert 0.94 <= coverage <= 0.96, f"Coverage = {coverage:.3f}, expected 94-96%"

        # SE accuracy < 10%
        se_accuracy = np.abs(empirical_sd - mean_se) / empirical_sd
        assert se_accuracy < 0.10, f"SE accuracy = {se_accuracy:.3f}, expected < 0.10"

        # Should have lowest variance (save empirical_sd for comparison)
        # In practice, compare to IPW-only and regression-only (not tested here)
        assert empirical_sd > 0  # Sanity check

    def test_both_models_correct_multiple_covariates_n400(self):
        """
        Monte Carlo validation with multiple covariates, both models correct (n=400).

        DGP:
            X1, X2, X3 ~ N(0, 1)
            e(X) = logistic(0.5*X1 + 0.3*X2 - 0.4*X3)
            μ(X) = 2 + 0.4*X1 + 0.3*X2 + 0.2*X3
            Y = τ*T + μ(X) + ε  # True ATE = 2.5

        Expected: Same as above (both models correct).
        """
        np.random.seed(2002)
        n_sims = 5000
        n = 400
        true_ate = 2.5

        estimates = []
        ses = []
        cis_cover = []

        for sim in range(n_sims):
            X1 = np.random.normal(0, 1, n)
            X2 = np.random.normal(0, 1, n)
            X3 = np.random.normal(0, 1, n)
            X = np.column_stack([X1, X2, X3])

            # Propensity: linear in covariates
            logit_prop = 0.5 * X1 + 0.3 * X2 - 0.4 * X3
            e_X = 1 / (1 + np.exp(-logit_prop))
            T = np.random.binomial(1, e_X)

            # Outcome: linear in covariates
            Y = true_ate * T + (2 + 0.4 * X1 + 0.3 * X2 + 0.2 * X3) + np.random.normal(0, 0.5, n)

            result = dr_ate(Y, T, X)

            estimates.append(result["estimate"])
            ses.append(result["se"])
            cis_cover.append(result["ci_lower"] <= true_ate <= result["ci_upper"])

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_ate
        coverage = np.mean(cis_cover)
        empirical_sd = np.std(estimates, ddof=1)
        mean_se = np.mean(ses)
        se_accuracy = np.abs(empirical_sd - mean_se) / empirical_sd

        # Bias < 0.05
        assert np.abs(bias) < 0.05, f"Bias = {bias:.3f}, expected < 0.05"

        # Coverage 94-96%
        assert 0.94 <= coverage <= 0.96, f"Coverage = {coverage:.3f}, expected 94-96%"

        # SE accuracy < 10%
        assert se_accuracy < 0.10, f"SE accuracy = {se_accuracy:.3f}, expected < 0.10"


class TestMonteCarloDoublyRobustPropensityCorrect:
    """Monte Carlo test when propensity correct but outcome model misspecified."""

    def test_propensity_correct_outcome_wrong_quadratic(self):
        """
        Monte Carlo validation with propensity correct, outcome misspecified.

        DGP:
            e(X) = logistic(0.8*X)      # Correct (linear in X)
            μ_true(X) = 2 + 0.5*X²      # True (quadratic)
            μ_fit(X) = β₀ + β₁*X        # Fitted (linear, WRONG!)
            Y = τ*T + μ_true(X) + ε     # True ATE = 3.0

        Expected:
        - Bias < 0.10 (propensity correct → IPW protects)
        - Coverage 93-97.5% (conservative OK for misspecified outcome)
        - SE accuracy < 15%
        """
        np.random.seed(2003)
        n_sims = 5000
        n = 400
        true_ate = 3.0

        estimates = []
        ses = []
        cis_cover = []

        for sim in range(n_sims):
            X = np.random.normal(0, 1, n)

            # Propensity: linear (correctly modeled)
            logit_prop = 0.8 * X
            e_X = 1 / (1 + np.exp(-logit_prop))
            T = np.random.binomial(1, e_X)

            # Outcome: QUADRATIC (misspecified by linear model)
            Y = true_ate * T + (2 + 0.5 * X**2) + np.random.normal(0, 0.5, n)

            result = dr_ate(Y, T, X)

            estimates.append(result["estimate"])
            ses.append(result["se"])
            cis_cover.append(result["ci_lower"] <= true_ate <= result["ci_upper"])

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_ate
        coverage = np.mean(cis_cover)
        empirical_sd = np.std(estimates, ddof=1)
        mean_se = np.mean(ses)
        se_accuracy = np.abs(empirical_sd - mean_se) / empirical_sd

        # Bias < 0.10 (DR still consistent via IPW protection)
        assert np.abs(bias) < 0.10, f"Bias = {bias:.3f}, expected < 0.10"

        # Coverage 93-97.5% (relaxed upper bound for misspecification)
        assert 0.93 <= coverage <= 0.975, f"Coverage = {coverage:.3f}, expected 93-97.5%"

        # SE accuracy < 15%
        assert se_accuracy < 0.15, f"SE accuracy = {se_accuracy:.3f}, expected < 0.15"


class TestMonteCarloDoublyRobustOutcomeCorrect:
    """Monte Carlo test when outcome correct but propensity model misspecified."""

    def test_outcome_correct_propensity_wrong_quadratic(self):
        """
        Monte Carlo validation with outcome correct, propensity misspecified.

        DGP:
            e_true(X) = logistic(0.8*X²)    # True (quadratic)
            e_fit(X) = logistic(β₀ + β₁*X)  # Fitted (linear, WRONG!)
            μ(X) = 2 + 0.5*X                # Correct (linear)
            Y = τ*T + μ(X) + ε              # True ATE = 2.5

        Expected:
        - Bias < 0.10 (outcome correct → regression protects)
        - Coverage 93-97.5%
        - SE accuracy < 15%
        """
        np.random.seed(2004)
        n_sims = 5000
        n = 400
        true_ate = 2.5

        estimates = []
        ses = []
        cis_cover = []

        for sim in range(n_sims):
            X = np.random.normal(0, 1, n)

            # Propensity: QUADRATIC (misspecified by linear logistic)
            logit_prop = 0.8 * X**2 - 1.0  # Shift to avoid extremes
            e_X = 1 / (1 + np.exp(-logit_prop))
            e_X = np.clip(e_X, 0.1, 0.9)  # Clip to avoid perfect separation
            T = np.random.binomial(1, e_X)

            # Outcome: LINEAR (correctly modeled)
            Y = true_ate * T + (2 + 0.5 * X) + np.random.normal(0, 0.5, n)

            result = dr_ate(Y, T, X)

            estimates.append(result["estimate"])
            ses.append(result["se"])
            cis_cover.append(result["ci_lower"] <= true_ate <= result["ci_upper"])

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_ate
        coverage = np.mean(cis_cover)
        empirical_sd = np.std(estimates, ddof=1)
        mean_se = np.mean(ses)
        se_accuracy = np.abs(empirical_sd - mean_se) / empirical_sd

        # Bias < 0.10 (DR still consistent via regression protection)
        assert np.abs(bias) < 0.10, f"Bias = {bias:.3f}, expected < 0.10"

        # Coverage 93-97.5%
        assert 0.93 <= coverage <= 0.975, f"Coverage = {coverage:.3f}, expected 93-97.5%"

        # SE accuracy < 15%
        assert se_accuracy < 0.15, f"SE accuracy = {se_accuracy:.3f}, expected < 0.15"


class TestMonteCarloDoublyRobustBothWrong:
    """Monte Carlo test when both models are misspecified (no protection)."""

    def test_both_models_wrong_runs_successfully(self):
        """
        Monte Carlo validation with both models misspecified.

        DGP:
            e_true(X) = logistic(0.8*X²)    # Quadratic
            e_fit(X) = logistic(β₀ + β₁*X)  # Linear (WRONG!)
            μ_true(X) = 2 + 0.5*X²          # Quadratic
            μ_fit(X) = β₀ + β₁*X            # Linear (WRONG!)
            Y = τ*T + μ_true(X) + ε         # True ATE = 3.0

        Expected:
        - No bias guarantee (both models wrong)
        - Test runs successfully (5000 simulations complete)
        - All estimates finite and reasonable
        - SEs positive
        """
        np.random.seed(2005)
        n_sims = 5000
        n = 300
        true_ate = 3.0

        estimates = []
        ses = []

        for sim in range(n_sims):
            X = np.random.normal(0, 1, n)

            # Propensity: quadratic (misspecified as linear)
            logit_prop = 0.8 * X**2 - 1.0
            e_X = 1 / (1 + np.exp(-logit_prop))
            e_X = np.clip(e_X, 0.15, 0.85)
            T = np.random.binomial(1, e_X)

            # Outcome: quadratic (misspecified as linear)
            Y = true_ate * T + (2 + 0.5 * X**2) + np.random.normal(0, 0.5, n)

            result = dr_ate(Y, T, X)

            estimates.append(result["estimate"])
            ses.append(result["se"])

        # Test runs successfully
        assert len(estimates) == n_sims

        # All estimates finite
        assert np.all(np.isfinite(estimates))

        # All SEs positive and finite
        assert np.all(np.array(ses) > 0)
        assert np.all(np.isfinite(ses))

        # Estimates should be in a reasonable range (not absurd)
        mean_estimate = np.mean(estimates)
        assert -5.0 < mean_estimate < 10.0  # Broad range (no bias guarantee)

        # Note: Bias may be present (no protection with both models wrong)
        # This test only verifies the estimator runs without crashing
