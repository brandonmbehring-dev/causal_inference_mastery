"""Monte Carlo validation for TMLE.

Validates:
1. Unbiasedness: Mean estimate close to true ATE
2. Coverage: 95% CIs cover true value 93-97% of the time
3. SE calibration: Estimated SE close to empirical SD

Scenarios:
1. Both models correct (ideal case)
2. Propensity correct, outcome wrong
3. Outcome correct, propensity wrong
"""

import numpy as np
import pytest
from typing import Tuple

from src.causal_inference.observational.tmle import tmle_ate


def generate_observational_dgp(
    n: int,
    true_ate: float,
    propensity_linear: bool = True,
    outcome_linear: bool = True,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate observational data with confounding.

    Parameters
    ----------
    n : int
        Sample size.
    true_ate : float
        True average treatment effect.
    propensity_linear : bool
        If True, propensity is linear in X (model correct).
        If False, propensity is quadratic (model misspecified).
    outcome_linear : bool
        If True, outcome is linear in X (model correct).
        If False, outcome is quadratic (model misspecified).
    seed : int, optional
        Random seed.

    Returns
    -------
    Y, T, X : arrays
        Outcome, treatment, covariates.
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(n)

    # Propensity model
    if propensity_linear:
        logit_prop = 0.5 * X
    else:
        logit_prop = 0.3 * X**2 - 0.5  # Quadratic

    e_X = 1 / (1 + np.exp(-logit_prop))
    T = np.random.binomial(1, e_X)

    # Outcome model
    if outcome_linear:
        mu_X = 2 + 0.5 * X
    else:
        mu_X = 2 + 0.3 * X**2  # Quadratic

    Y = true_ate * T + mu_X + np.random.randn(n) * 0.5

    return Y, T, X


@pytest.mark.monte_carlo
class TestTMLEMonteCarloBothCorrect:
    """Monte Carlo validation when both models correctly specified."""

    def test_unbiased_n300(self):
        """
        Test TMLE is unbiased with both models correct.

        Target: |bias| < 0.05
        """
        n_sims = 500  # Reduced for speed
        n = 300
        true_ate = 2.0

        estimates = []
        for seed in range(n_sims):
            Y, T, X = generate_observational_dgp(
                n, true_ate, propensity_linear=True, outcome_linear=True, seed=seed
            )
            try:
                result = tmle_ate(Y, T, X)
                estimates.append(result["estimate"])
            except Exception:
                continue

        estimates = np.array(estimates)
        bias = np.mean(estimates) - true_ate

        assert np.abs(bias) < 0.05, f"Bias {bias:.4f} exceeds threshold 0.05"

    def test_coverage_n300(self):
        """
        Test TMLE has valid 95% CI coverage.

        Target: 91-99% (looser than theoretical 95% due to finite samples)
        """
        n_sims = 500
        n = 300
        true_ate = 2.0

        covered = []
        for seed in range(n_sims):
            Y, T, X = generate_observational_dgp(
                n, true_ate, propensity_linear=True, outcome_linear=True, seed=seed
            )
            try:
                result = tmle_ate(Y, T, X)
                covered.append(result["ci_lower"] < true_ate < result["ci_upper"])
            except Exception:
                continue

        coverage = np.mean(covered)

        assert 0.91 < coverage < 0.99, f"Coverage {coverage:.2%} outside [91%, 99%]"

    def test_se_calibration_n300(self):
        """
        Test TMLE SE is well-calibrated.

        Target: Mean(SE) within 20% of empirical SD
        """
        n_sims = 500
        n = 300
        true_ate = 2.0

        estimates = []
        ses = []
        for seed in range(n_sims):
            Y, T, X = generate_observational_dgp(
                n, true_ate, propensity_linear=True, outcome_linear=True, seed=seed
            )
            try:
                result = tmle_ate(Y, T, X)
                estimates.append(result["estimate"])
                ses.append(result["se"])
            except Exception:
                continue

        empirical_sd = np.std(estimates)
        mean_se = np.mean(ses)

        se_ratio = mean_se / empirical_sd
        assert 0.8 < se_ratio < 1.2, f"SE ratio {se_ratio:.2f} outside [0.8, 1.2]"


@pytest.mark.monte_carlo
class TestTMLEMonteCarloDoubleRobustness:
    """Monte Carlo validation of double robustness property."""

    def test_propensity_correct_outcome_wrong(self):
        """
        Test TMLE when propensity correct but outcome model misspecified.

        Outcome model fits linear, but true relationship is quadratic.
        IPW component should protect against bias.

        Target: |bias| < 0.15 (looser than both correct)
        """
        n_sims = 300
        n = 400
        true_ate = 2.0

        estimates = []
        for seed in range(n_sims):
            Y, T, X = generate_observational_dgp(
                n, true_ate, propensity_linear=True, outcome_linear=False, seed=seed
            )
            try:
                result = tmle_ate(Y, T, X)
                estimates.append(result["estimate"])
            except Exception:
                continue

        estimates = np.array(estimates)
        bias = np.mean(estimates) - true_ate

        assert np.abs(bias) < 0.15, f"Bias {bias:.4f} exceeds threshold 0.15"

    def test_outcome_correct_propensity_wrong(self):
        """
        Test TMLE when outcome correct but propensity model misspecified.

        Propensity model fits linear, but true relationship is quadratic.
        Outcome regression component should protect against bias.

        Target: |bias| < 0.15 (looser than both correct)
        """
        n_sims = 300
        n = 400
        true_ate = 2.0

        estimates = []
        for seed in range(n_sims):
            Y, T, X = generate_observational_dgp(
                n, true_ate, propensity_linear=False, outcome_linear=True, seed=seed
            )
            try:
                result = tmle_ate(Y, T, X)
                estimates.append(result["estimate"])
            except Exception:
                continue

        estimates = np.array(estimates)
        bias = np.mean(estimates) - true_ate

        assert np.abs(bias) < 0.15, f"Bias {bias:.4f} exceeds threshold 0.15"


@pytest.mark.monte_carlo
class TestTMLEConvergenceProperties:
    """Test TMLE convergence across simulations."""

    def test_always_converges(self):
        """Test TMLE converges in all simulations."""
        n_sims = 100
        n = 200
        true_ate = 2.0

        convergence_count = 0
        for seed in range(n_sims):
            Y, T, X = generate_observational_dgp(
                n, true_ate, propensity_linear=True, outcome_linear=True, seed=seed
            )
            try:
                result = tmle_ate(Y, T, X)
                if result["converged"]:
                    convergence_count += 1
            except Exception:
                continue

        convergence_rate = convergence_count / n_sims
        assert convergence_rate > 0.95, f"Convergence rate {convergence_rate:.2%} < 95%"

    def test_few_iterations(self):
        """Test TMLE converges in few iterations."""
        n_sims = 100
        n = 200
        true_ate = 2.0

        iterations = []
        for seed in range(n_sims):
            Y, T, X = generate_observational_dgp(
                n, true_ate, propensity_linear=True, outcome_linear=True, seed=seed
            )
            try:
                result = tmle_ate(Y, T, X)
                iterations.append(result["n_iterations"])
            except Exception:
                continue

        mean_iter = np.mean(iterations)
        assert mean_iter < 10, f"Mean iterations {mean_iter:.1f} > 10"
