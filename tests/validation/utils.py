"""Helper functions for validation tests."""

import numpy as np
from typing import List, Tuple, Dict, Any


def compute_monte_carlo_bias(estimates: List[float], true_value: float) -> float:
    """
    Compute bias from Monte Carlo estimates.

    Parameters
    ----------
    estimates : list of float
        Monte Carlo estimates (e.g., 1000 ATE estimates)
    true_value : float
        True parameter value (e.g., true ATE = 2.0)

    Returns
    -------
    float
        Absolute bias: |mean(estimates) - true_value|
    """
    return abs(np.mean(estimates) - true_value)


def compute_monte_carlo_coverage(
    ci_lower: List[float], ci_upper: List[float], true_value: float
) -> float:
    """
    Compute coverage rate from confidence intervals.

    Parameters
    ----------
    ci_lower : list of float
        Lower bounds of confidence intervals
    ci_upper : list of float
        Upper bounds of confidence intervals
    true_value : float
        True parameter value

    Returns
    -------
    float
        Coverage rate: proportion of CIs containing true value
    """
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)
    contains_true = (ci_lower <= true_value) & (true_value <= ci_upper)
    return np.mean(contains_true)


def compute_se_accuracy(estimates: List[float], standard_errors: List[float]) -> float:
    """
    Compute SE accuracy: how close standard errors are to empirical SD.

    Parameters
    ----------
    estimates : list of float
        Monte Carlo estimates
    standard_errors : list of float
        Estimated standard errors

    Returns
    -------
    float
        Relative error: |std(estimates) - mean(SE)| / std(estimates)
    """
    empirical_sd = np.std(estimates, ddof=1)
    mean_se = np.mean(standard_errors)
    return abs(empirical_sd - mean_se) / empirical_sd


def generate_dgp_simple_rct(
    n: int,
    true_ate: float,
    sigma1: float = 1.0,
    sigma0: float = 1.0,
    balanced: bool = True,
    random_state: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data from simple RCT data generating process.

    Parameters
    ----------
    n : int
        Total sample size
    true_ate : float
        True average treatment effect
    sigma1 : float, default=1.0
        Standard deviation for treated units
    sigma0 : float, default=1.0
        Standard deviation for control units
    balanced : bool, default=True
        If True, 50/50 split. If False, random assignment.
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    outcomes : np.ndarray
        Observed outcomes
    treatment : np.ndarray
        Treatment assignments (0/1)

    Notes
    -----
    DGP:
        Y(1) ~ N(true_ate, sigma1^2)
        Y(0) ~ N(0, sigma0^2)
        T ~ Bernoulli(0.5) if balanced
        Y = T*Y(1) + (1-T)*Y(0)
    """
    rng = np.random.RandomState(random_state)

    if balanced:
        n1 = n // 2
        n0 = n - n1
        treatment = np.array([1] * n1 + [0] * n0)
        rng.shuffle(treatment)
    else:
        treatment = rng.binomial(1, 0.5, n)

    # Generate potential outcomes
    y1 = rng.normal(true_ate, sigma1, n)
    y0 = rng.normal(0.0, sigma0, n)

    # Observed outcome: Y = T*Y(1) + (1-T)*Y(0)
    outcomes = treatment * y1 + (1 - treatment) * y0

    return outcomes, treatment


def generate_dgp_stratified_rct(
    n_per_stratum: int,
    n_strata: int,
    true_ate: float,
    baseline_effects: List[float] = None,
    random_state: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data from stratified RCT.

    Parameters
    ----------
    n_per_stratum : int
        Sample size per stratum (must be even for balance)
    n_strata : int
        Number of strata
    true_ate : float
        True ATE (same in all strata)
    baseline_effects : list of float, optional
        Baseline outcome for each stratum. If None, use [0, 5, 10, ...]
    random_state : int, optional
        Random seed

    Returns
    -------
    outcomes, treatment, strata : tuple of np.ndarray
    """
    rng = np.random.RandomState(random_state)

    if baseline_effects is None:
        baseline_effects = [i * 5.0 for i in range(n_strata)]

    outcomes = []
    treatment = []
    strata = []

    for s, baseline in enumerate(baseline_effects[:n_strata]):
        n1 = n_per_stratum // 2
        n0 = n_per_stratum - n1

        t = np.array([1] * n1 + [0] * n0)
        rng.shuffle(t)

        y = np.where(
            t == 1,
            rng.normal(baseline + true_ate, 1.0, n_per_stratum),
            rng.normal(baseline, 1.0, n_per_stratum),
        )

        outcomes.extend(y)
        treatment.extend(t)
        strata.extend([s] * n_per_stratum)

    return np.array(outcomes), np.array(treatment), np.array(strata)


def generate_dgp_regression_rct(
    n: int, true_ate: float, covariate_effect: float = 3.0, random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data from RCT with covariates.

    Parameters
    ----------
    n : int
        Sample size
    true_ate : float
        True ATE
    covariate_effect : float, default=3.0
        Effect of covariate on outcome
    random_state : int, optional
        Random seed

    Returns
    -------
    outcomes, treatment, covariates : tuple of np.ndarray

    Notes
    -----
    DGP:
        X ~ N(0, 1)
        T ~ Bernoulli(0.5)
        Y = true_ate*T + covariate_effect*X + epsilon
        epsilon ~ N(0, 1)
    """
    rng = np.random.RandomState(random_state)

    X = rng.normal(0, 1, n)
    treatment = np.array([1] * (n // 2) + [0] * (n - n // 2))
    rng.shuffle(treatment)

    outcomes = true_ate * treatment + covariate_effect * X + rng.normal(0, 1, n)

    return outcomes, treatment, X


def validate_monte_carlo_results(
    estimates: List[float],
    standard_errors: List[float],
    ci_lower: List[float],
    ci_upper: List[float],
    true_ate: float,
    bias_threshold: float = 0.05,
    coverage_lower: float = 0.93,
    coverage_upper: float = 0.97,
    se_accuracy_threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    Validate Monte Carlo simulation results.

    Parameters
    ----------
    estimates : list of float
        Monte Carlo ATE estimates
    standard_errors : list of float
        Estimated standard errors
    ci_lower, ci_upper : list of float
        Confidence interval bounds
    true_ate : float
        True ATE value
    bias_threshold : float, default=0.05
        Maximum acceptable bias
    coverage_lower, coverage_upper : float
        Acceptable coverage range
    se_accuracy_threshold : float, default=0.10
        Maximum acceptable SE relative error

    Returns
    -------
    dict
        Validation results with:
        - bias: float
        - bias_ok: bool
        - coverage: float
        - coverage_ok: bool
        - se_accuracy: float
        - se_accuracy_ok: bool
        - all_pass: bool
    """
    bias = compute_monte_carlo_bias(estimates, true_ate)
    coverage = compute_monte_carlo_coverage(ci_lower, ci_upper, true_ate)
    se_acc = compute_se_accuracy(estimates, standard_errors)

    bias_ok = bias < bias_threshold
    coverage_ok = coverage_lower <= coverage <= coverage_upper
    se_accuracy_ok = se_acc < se_accuracy_threshold

    return {
        "bias": bias,
        "bias_ok": bias_ok,
        "coverage": coverage,
        "coverage_ok": coverage_ok,
        "se_accuracy": se_acc,
        "se_accuracy_ok": se_accuracy_ok,
        "all_pass": bias_ok and coverage_ok and se_accuracy_ok,
        "n_runs": len(estimates),
        "true_ate": true_ate,
    }
