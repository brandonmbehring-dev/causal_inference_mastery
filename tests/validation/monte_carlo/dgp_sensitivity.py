"""
Data Generating Processes for Sensitivity Analysis Monte Carlo Validation.

Provides DGPs for validating:
1. E-value interpretation accuracy
2. Rosenbaum bounds gamma_critical accuracy

Following project conventions: function-based generators, no dataclasses.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.special import expit  # Logistic function


# =============================================================================
# E-Value DGPs
# =============================================================================


def dgp_evalue_known_rr(
    n: int = 500,
    true_rr: float = 2.0,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate binary outcome data with known risk ratio.

    DGP:
        T ~ Bernoulli(0.5)  (treatment assignment)
        p0 = 0.2  (baseline risk in control)
        p1 = p0 * true_rr  (risk in treated)
        Y ~ Bernoulli(p0 + (p1 - p0) * T)

    Parameters
    ----------
    n : int
        Sample size
    true_rr : float
        True risk ratio (RR = p1/p0)
    random_state : int, optional
        Random seed

    Returns
    -------
    outcomes : np.ndarray
        Binary outcomes (n,)
    treatment : np.ndarray
        Treatment indicator (n,)
    true_rr : float
        The true risk ratio (returned for validation)

    Notes
    -----
    E-value for this RR should be: E = RR + sqrt(RR * (RR - 1))
    """
    rng = np.random.RandomState(random_state)

    # Treatment assignment (RCT-like)
    treatment = rng.binomial(1, 0.5, n)

    # Outcome probabilities
    p0 = 0.2  # Baseline risk
    p1 = min(p0 * true_rr, 0.99)  # Cap at 0.99

    # Generate outcomes
    probs = np.where(treatment == 1, p1, p0)
    outcomes = rng.binomial(1, probs)

    return outcomes.astype(float), treatment.astype(float), true_rr


def dgp_evalue_confounded(
    n: int = 500,
    true_ate: float = 0.15,
    confounder_strength: float = 1.5,
    baseline_risk: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate confounded observational data for E-value validation.

    DGP:
        U ~ N(0, 1)  (unmeasured confounder)
        propensity = logistic(confounder_strength * U)
        T ~ Bernoulli(propensity)
        Y_prob = baseline_risk + true_ate * T + 0.1 * U
        Y ~ Bernoulli(clip(Y_prob, 0.01, 0.99))

    Parameters
    ----------
    n : int
        Sample size
    true_ate : float
        True average treatment effect (risk difference)
    confounder_strength : float
        Strength of confounding (log-odds scale)
    baseline_risk : float
        Baseline risk in unexposed
    random_state : int, optional
        Random seed

    Returns
    -------
    outcomes : np.ndarray
        Binary outcomes (n,)
    treatment : np.ndarray
        Treatment indicator (n,)
    confounder : np.ndarray
        Unmeasured confounder (n,)
    true_rr : float
        True risk ratio

    Notes
    -----
    The observed association will be confounded. The true RR can be
    compared against naive estimates to understand bias.
    """
    rng = np.random.RandomState(random_state)

    # Unmeasured confounder
    U = rng.normal(0, 1, n)

    # Confounded treatment assignment
    propensity = expit(confounder_strength * U)
    treatment = rng.binomial(1, propensity)

    # Outcome with confounding
    y_prob = baseline_risk + true_ate * treatment + 0.1 * U
    y_prob = np.clip(y_prob, 0.01, 0.99)
    outcomes = rng.binomial(1, y_prob)

    # True RR (under no confounding)
    true_rr = (baseline_risk + true_ate) / baseline_risk

    return outcomes.astype(float), treatment.astype(float), U, true_rr


def dgp_evalue_smd(
    n: int = 500,
    true_smd: float = 0.5,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate continuous outcome data with known standardized mean difference.

    DGP:
        T ~ Bernoulli(0.5)
        Y | T=0 ~ N(0, 1)
        Y | T=1 ~ N(true_smd, 1)

    Parameters
    ----------
    n : int
        Sample size
    true_smd : float
        True Cohen's d (standardized mean difference)
    random_state : int, optional
        Random seed

    Returns
    -------
    outcomes : np.ndarray
        Continuous outcomes (n,)
    treatment : np.ndarray
        Treatment indicator (n,)
    true_smd : float
        The true SMD (returned for validation)

    Notes
    -----
    E-value conversion: RR ≈ exp(0.91 * d)
    """
    rng = np.random.RandomState(random_state)

    treatment = rng.binomial(1, 0.5, n)

    # Generate outcomes
    outcomes = np.where(
        treatment == 1,
        rng.normal(true_smd, 1, n),
        rng.normal(0, 1, n),
    )

    return outcomes, treatment.astype(float), true_smd


# =============================================================================
# Rosenbaum Bounds DGPs
# =============================================================================


def dgp_matched_pairs_no_confounding(
    n_pairs: int = 50,
    true_effect: float = 2.0,
    noise_sd: float = 1.0,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate matched pairs with no hidden confounding.

    DGP:
        For each pair i:
            Y_treated = true_effect + N(0, noise_sd)
            Y_control = N(0, noise_sd)

    Parameters
    ----------
    n_pairs : int
        Number of matched pairs
    true_effect : float
        True treatment effect
    noise_sd : float
        Standard deviation of noise
    random_state : int, optional
        Random seed

    Returns
    -------
    treated_outcomes : np.ndarray
        Outcomes for treated units (n_pairs,)
    control_outcomes : np.ndarray
        Outcomes for control units (n_pairs,)
    true_effect : float
        The true effect (returned for validation)

    Notes
    -----
    With no confounding, gamma_critical should be high (or None if
    effect is strong enough to survive all tested gamma values).
    At gamma=1, p-value should reflect true effect significance.
    """
    rng = np.random.RandomState(random_state)

    control_outcomes = rng.normal(0, noise_sd, n_pairs)
    treated_outcomes = control_outcomes + true_effect + rng.normal(0, noise_sd * 0.5, n_pairs)

    return treated_outcomes, control_outcomes, true_effect


def dgp_matched_pairs_with_confounding(
    n_pairs: int = 50,
    true_effect: float = 2.0,
    gamma_confounding: float = 1.5,
    noise_sd: float = 1.0,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Generate matched pairs with known hidden confounding strength.

    DGP:
        For each pair i:
            U_i ~ Uniform(0, 1)  (hidden confounder)
            Confounding bias = log(gamma) * (2*U - 1) * noise_sd
            Y_treated = true_effect + confounding_bias + N(0, noise_sd)
            Y_control = N(0, noise_sd)

    Parameters
    ----------
    n_pairs : int
        Number of matched pairs
    true_effect : float
        True treatment effect (without confounding)
    gamma_confounding : float
        True confounding strength (Gamma parameter)
    noise_sd : float
        Standard deviation of noise
    random_state : int, optional
        Random seed

    Returns
    -------
    treated_outcomes : np.ndarray
        Outcomes for treated units (n_pairs,)
    control_outcomes : np.ndarray
        Outcomes for control units (n_pairs,)
    true_effect : float
        The true effect
    gamma_confounding : float
        The confounding strength (returned for validation)

    Notes
    -----
    The gamma_critical from Rosenbaum bounds should approximately
    equal gamma_confounding when the effect is borderline significant.
    """
    rng = np.random.RandomState(random_state)

    # Hidden confounder
    U = rng.uniform(0, 1, n_pairs)

    # Confounding bias proportional to log(gamma)
    # This creates differential treatment probability
    confounding_bias = np.log(gamma_confounding) * (2 * U - 1) * noise_sd

    control_outcomes = rng.normal(0, noise_sd, n_pairs)
    treated_outcomes = (
        control_outcomes + true_effect + confounding_bias + rng.normal(0, noise_sd * 0.5, n_pairs)
    )

    return treated_outcomes, control_outcomes, true_effect, gamma_confounding


def dgp_matched_pairs_weak_effect(
    n_pairs: int = 50,
    true_effect: float = 0.3,
    noise_sd: float = 2.0,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate matched pairs with weak treatment effect (high noise).

    DGP:
        Y_treated = true_effect + N(0, noise_sd)
        Y_control = N(0, noise_sd)

    Parameters
    ----------
    n_pairs : int
        Number of matched pairs
    true_effect : float
        True (weak) treatment effect
    noise_sd : float
        High noise standard deviation
    random_state : int, optional
        Random seed

    Returns
    -------
    treated_outcomes : np.ndarray
        Outcomes for treated units (n_pairs,)
    control_outcomes : np.ndarray
        Outcomes for control units (n_pairs,)
    true_effect : float
        The true effect

    Notes
    -----
    With weak effect, gamma_critical should be low (sensitive to confounding).
    """
    rng = np.random.RandomState(random_state)

    control_outcomes = rng.normal(0, noise_sd, n_pairs)
    treated_outcomes = control_outcomes + true_effect + rng.normal(0, noise_sd * 0.5, n_pairs)

    return treated_outcomes, control_outcomes, true_effect


def dgp_matched_pairs_strong_effect(
    n_pairs: int = 50,
    true_effect: float = 5.0,
    noise_sd: float = 1.0,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate matched pairs with strong treatment effect.

    DGP:
        Y_treated = true_effect + N(0, noise_sd)
        Y_control = N(0, noise_sd)

    Parameters
    ----------
    n_pairs : int
        Number of matched pairs
    true_effect : float
        True (strong) treatment effect
    noise_sd : float
        Noise standard deviation
    random_state : int, optional
        Random seed

    Returns
    -------
    treated_outcomes : np.ndarray
        Outcomes for treated units (n_pairs,)
    control_outcomes : np.ndarray
        Outcomes for control units (n_pairs,)
    true_effect : float
        The true effect

    Notes
    -----
    With strong effect, gamma_critical should be high or None
    (robust to substantial confounding).
    """
    rng = np.random.RandomState(random_state)

    control_outcomes = rng.normal(0, noise_sd, n_pairs)
    treated_outcomes = control_outcomes + true_effect + rng.normal(0, noise_sd * 0.5, n_pairs)

    return treated_outcomes, control_outcomes, true_effect


def dgp_matched_pairs_null_effect(
    n_pairs: int = 50,
    noise_sd: float = 1.0,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate matched pairs with no treatment effect.

    DGP:
        Y_treated = N(0, noise_sd)
        Y_control = N(0, noise_sd)

    Parameters
    ----------
    n_pairs : int
        Number of matched pairs
    noise_sd : float
        Noise standard deviation
    random_state : int, optional
        Random seed

    Returns
    -------
    treated_outcomes : np.ndarray
        Outcomes for treated units (n_pairs,)
    control_outcomes : np.ndarray
        Outcomes for control units (n_pairs,)
    true_effect : float
        Always 0.0

    Notes
    -----
    With null effect, Rosenbaum bounds should show very low gamma_critical
    (immediately sensitive) or non-significant at gamma=1.
    """
    rng = np.random.RandomState(random_state)

    control_outcomes = rng.normal(0, noise_sd, n_pairs)
    treated_outcomes = rng.normal(0, noise_sd, n_pairs)  # No effect

    return treated_outcomes, control_outcomes, 0.0
