"""Unified Data Generating Processes for benchmarking.

Provides consistent DGP functions across all method families,
ensuring reproducibility and fair cross-language comparison.

All functions use explicit seeds for reproducibility.
Same seed in Python and Julia produces identical datasets.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np


def generate_rct_data(
    n: int,
    effect_size: float = 0.5,
    n_covariates: int = 3,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate RCT (randomized controlled trial) data.

    Model:
        Y = mu + tau*T + X @ beta + epsilon
        T ~ Bernoulli(0.5)

    Parameters
    ----------
    n : int
        Sample size.
    effect_size : float
        True ATE (default 0.5).
    n_covariates : int
        Number of covariates (default 3).
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, treatment, covariates, strata, true_ate.
    """
    rng = np.random.default_rng(seed)

    # Covariates
    X = rng.normal(0, 1, (n, n_covariates))

    # Treatment (randomized)
    T = rng.binomial(1, 0.5, n)

    # Strata for stratified estimator (quartiles of X[:,0])
    strata = np.digitize(X[:, 0], np.percentile(X[:, 0], [25, 50, 75]))

    # Outcome
    mu = 1.0
    beta = np.array([0.5, -0.3, 0.2])[:n_covariates]
    epsilon = rng.normal(0, 1, n)
    Y = mu + effect_size * T + X @ beta + epsilon

    return {
        "outcome": Y,
        "treatment": T,
        "covariates": X,
        "strata": strata,
        "true_ate": effect_size,
    }


def generate_observational_data(
    n: int,
    effect_size: float = 2.0,
    confounding: float = 0.5,
    n_covariates: int = 5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate observational data with confounding.

    Model:
        propensity = logit(confounding * X[:,0])
        T ~ Bernoulli(propensity)
        Y = baseline(X) + tau*T + epsilon

    Parameters
    ----------
    n : int
        Sample size.
    effect_size : float
        True ATE (default 2.0).
    confounding : float
        Confounding strength (default 0.5).
    n_covariates : int
        Number of covariates (default 5).
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, treatment, covariates, propensity, true_ate.
    """
    rng = np.random.default_rng(seed)

    # Covariates
    X = rng.normal(0, 1, (n, n_covariates))

    # Propensity (confounded by X[:,0])
    logit_p = confounding * X[:, 0]
    propensity = 1 / (1 + np.exp(-logit_p))
    T = rng.binomial(1, propensity)

    # Baseline outcome (depends on X)
    baseline = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]

    # Outcome
    epsilon = rng.normal(0, 1, n)
    Y = baseline + effect_size * T + epsilon

    return {
        "outcome": Y,
        "treatment": T,
        "covariates": X,
        "propensity": propensity,
        "true_ate": effect_size,
    }


def generate_psm_data(
    n: int,
    effect_size: float = 1.5,
    overlap: str = "good",
    n_covariates: int = 5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate data for propensity score matching.

    Parameters
    ----------
    n : int
        Sample size.
    effect_size : float
        True ATT (default 1.5).
    overlap : str
        "good" (balanced propensities) or "poor" (extreme separation).
    n_covariates : int
        Number of covariates.
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, treatment, covariates, propensity, true_att.
    """
    rng = np.random.default_rng(seed)

    # Covariates
    X = rng.normal(0, 1, (n, n_covariates))

    # Propensity based on overlap quality
    if overlap == "good":
        logit_p = 0.3 * X[:, 0] + 0.2 * X[:, 1]
    else:  # poor overlap
        logit_p = 1.5 * X[:, 0] + 1.0 * X[:, 1]

    propensity = 1 / (1 + np.exp(-logit_p))
    T = rng.binomial(1, propensity)

    # Baseline outcome
    baseline = X[:, 0] + 0.5 * X[:, 1]

    # Outcome
    epsilon = rng.normal(0, 1, n)
    Y = baseline + effect_size * T + epsilon

    return {
        "outcome": Y,
        "treatment": T,
        "covariates": X,
        "propensity": propensity,
        "true_att": effect_size,
    }


def generate_did_data(
    n_units: int,
    n_periods: int = 10,
    treatment_period: int = 5,
    effect_size: float = 2.0,
    n_treated_frac: float = 0.5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate panel data for difference-in-differences.

    Model:
        Y_it = alpha_i + gamma_t + tau * D_it + epsilon_it

    Parameters
    ----------
    n_units : int
        Number of units (individuals/firms).
    n_periods : int
        Number of time periods (default 10).
    treatment_period : int
        Period when treatment starts (default 5).
    effect_size : float
        True ATT (default 2.0).
    n_treated_frac : float
        Fraction of units treated (default 0.5).
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, treatment, unit, time, true_att.
    """
    rng = np.random.default_rng(seed)

    n_obs = n_units * n_periods

    # Unit and time indices
    unit = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    # Unit fixed effects
    alpha = rng.normal(0, 1, n_units)
    alpha_expanded = alpha[unit]

    # Time fixed effects
    gamma = rng.normal(0, 0.5, n_periods)
    gamma_expanded = gamma[time]

    # Treatment assignment
    n_treated = int(n_units * n_treated_frac)
    treated_units = set(range(n_treated))
    treatment = np.array([
        1 if u in treated_units and t >= treatment_period else 0
        for u, t in zip(unit, time)
    ])

    # Outcome
    epsilon = rng.normal(0, 1, n_obs)
    Y = alpha_expanded + gamma_expanded + effect_size * treatment + epsilon

    return {
        "outcome": Y,
        "treatment": treatment,
        "unit": unit,
        "time": time,
        "true_att": effect_size,
    }


def generate_iv_data(
    n: int,
    effect_size: float = 1.0,
    instrument_strength: str = "strong",
    n_instruments: int = 1,
    n_controls: int = 3,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate instrumental variable data.

    Model:
        D = gamma * Z + delta * X + nu
        Y = beta * D + alpha * X + epsilon

    Parameters
    ----------
    n : int
        Sample size.
    effect_size : float
        True causal effect (default 1.0).
    instrument_strength : str
        "strong" (F>10) or "weak" (F~5).
    n_instruments : int
        Number of instruments (default 1).
    n_controls : int
        Number of control variables (default 3).
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, endogenous, instruments, controls, true_effect.
    """
    rng = np.random.default_rng(seed)

    # Controls
    X = rng.normal(0, 1, (n, n_controls))

    # Instruments
    Z = rng.normal(0, 1, (n, n_instruments))

    # Instrument strength
    gamma = 0.5 if instrument_strength == "strong" else 0.15

    # Unobserved confounder
    U = rng.normal(0, 1, n)

    # First stage: endogenous variable
    nu = rng.normal(0, 0.5, n)
    D = gamma * Z.sum(axis=1) + 0.3 * X[:, 0] + 0.5 * U + nu

    # Second stage: outcome
    epsilon = rng.normal(0, 1, n)
    Y = effect_size * D + 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.5 * U + epsilon

    return {
        "outcome": Y,
        "endogenous": D,
        "instruments": Z,
        "controls": X,
        "true_effect": effect_size,
    }


def generate_rdd_data(
    n: int,
    effect_size: float = 2.0,
    cutoff: float = 0.0,
    bandwidth: float = 1.0,
    design: str = "sharp",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate regression discontinuity data.

    Parameters
    ----------
    n : int
        Sample size.
    effect_size : float
        True RDD effect (default 2.0).
    cutoff : float
        RDD cutoff (default 0.0).
    bandwidth : float
        Running variable spread (default 1.0).
    design : str
        "sharp" or "fuzzy".
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, treatment, running_variable, cutoff, true_effect.
    """
    rng = np.random.default_rng(seed)

    # Running variable (centered around cutoff)
    X = rng.uniform(-bandwidth, bandwidth, n) + cutoff

    # Treatment assignment
    if design == "sharp":
        T = (X >= cutoff).astype(int)
    else:  # fuzzy
        prob = 0.8 * (X >= cutoff) + 0.2 * (X < cutoff)
        T = rng.binomial(1, prob)

    # Outcome with polynomial in X
    baseline = 0.5 * (X - cutoff) + 0.2 * (X - cutoff) ** 2
    epsilon = rng.normal(0, 0.5, n)
    Y = baseline + effect_size * T + epsilon

    return {
        "outcome": Y,
        "treatment": T,
        "running_variable": X,
        "cutoff": cutoff,
        "true_effect": effect_size,
    }


def generate_scm_data(
    n_control: int = 20,
    n_periods: int = 20,
    treatment_period: int = 10,
    effect_size: float = 3.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate synthetic control data.

    Parameters
    ----------
    n_control : int
        Number of control units (default 20).
    n_periods : int
        Number of time periods (default 20).
    treatment_period : int
        Period when treatment starts (default 10).
    effect_size : float
        True treatment effect (default 3.0).
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcomes, treatment_period, true_weights, true_effect.
        outcomes is (n_control+1) x n_periods matrix.
    """
    rng = np.random.default_rng(seed)

    n_total = n_control + 1  # 1 treated unit

    # Common factors
    factors = rng.normal(0, 1, (3, n_periods))

    # Factor loadings for each unit
    loadings = rng.normal(0, 1, (n_total, 3))

    # Base outcomes
    outcomes = loadings @ factors

    # Add unit-specific noise
    outcomes += rng.normal(0, 0.5, (n_total, n_periods))

    # Add trend
    trend = np.linspace(0, 2, n_periods)
    outcomes += trend

    # Add treatment effect to first unit (treated)
    outcomes[0, treatment_period:] += effect_size

    # True weights (sparse, only a few controls matter)
    true_weights = np.zeros(n_control)
    true_weights[:3] = [0.4, 0.35, 0.25]

    return {
        "outcomes": outcomes,
        "treatment_period": treatment_period,
        "true_weights": true_weights,
        "true_effect": effect_size,
    }


def generate_cate_data(
    n: int,
    n_covariates: int = 5,
    effect_type: str = "linear",
    base_effect: float = 2.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate data with heterogeneous treatment effects.

    Parameters
    ----------
    n : int
        Sample size.
    n_covariates : int
        Number of covariates (default 5).
    effect_type : str
        "constant", "linear", or "nonlinear".
    base_effect : float
        Base treatment effect (default 2.0).
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, treatment, covariates, true_cate, true_ate.
    """
    rng = np.random.default_rng(seed)

    # Covariates
    X = rng.normal(0, 1, (n, n_covariates))

    # Treatment (randomized for clean CATE estimation)
    T = rng.binomial(1, 0.5, n)

    # Heterogeneous treatment effect
    if effect_type == "constant":
        tau = np.full(n, base_effect)
    elif effect_type == "linear":
        tau = base_effect + X[:, 0]
    else:  # nonlinear
        tau = base_effect + np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2

    # Baseline outcome
    baseline = X[:, 0] + 0.5 * X[:, 1]

    # Outcome
    epsilon = rng.normal(0, 1, n)
    Y = baseline + tau * T + epsilon

    return {
        "outcome": Y,
        "treatment": T,
        "covariates": X,
        "true_cate": tau,
        "true_ate": np.mean(tau),
    }


def generate_panel_data(
    n_units: int,
    n_periods: int = 8,
    effect_size: float = 2.0,
    n_covariates: int = 3,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate panel data for DML-CRE and Panel QTE.

    Parameters
    ----------
    n_units : int
        Number of units.
    n_periods : int
        Number of periods per unit.
    effect_size : float
        True ATE.
    n_covariates : int
        Number of time-varying covariates.
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, treatment, covariates, unit_id, time_id, true_ate.
    """
    rng = np.random.default_rng(seed)

    n_obs = n_units * n_periods

    # IDs
    unit_id = np.repeat(np.arange(n_units), n_periods)
    time_id = np.tile(np.arange(n_periods), n_units)

    # Time-varying covariates
    X = rng.normal(0, 1, (n_obs, n_covariates))

    # Unit fixed effects (correlated with X)
    unit_means = X.reshape(n_units, n_periods, n_covariates).mean(axis=1)
    alpha = 0.5 * unit_means[:, 0] + rng.normal(0, 0.5, n_units)
    alpha_expanded = alpha[unit_id]

    # Treatment (time-varying)
    propensity = 1 / (1 + np.exp(-0.3 * X[:, 0]))
    T = rng.binomial(1, propensity)

    # Outcome
    epsilon = rng.normal(0, 1, n_obs)
    Y = alpha_expanded + 0.5 * X[:, 0] + effect_size * T + epsilon

    return {
        "outcome": Y,
        "treatment": T,
        "covariates": X,
        "unit_id": unit_id,
        "time_id": time_id,
        "true_ate": effect_size,
    }


def generate_dtr_data(
    n: int,
    n_covariates: int = 3,
    true_blip: Tuple[float, float] = (1.0, 2.0),
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate data for dynamic treatment regimes.

    Single-stage DTR with linear blip function.

    Parameters
    ----------
    n : int
        Sample size.
    n_covariates : int
        Number of covariates.
    true_blip : Tuple[float, float]
        True blip coefficients (intercept, X[0] coefficient).
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, treatment, covariates, true_blip, optimal_regime.
    """
    rng = np.random.default_rng(seed)

    # Covariates
    X = rng.normal(0, 1, (n, n_covariates))

    # Treatment
    propensity = 1 / (1 + np.exp(-0.3 * X[:, 0]))
    A = rng.binomial(1, propensity)

    # Blip function: gamma(X) = psi0 + psi1 * X[0]
    psi0, psi1 = true_blip
    blip = psi0 + psi1 * X[:, 0]

    # Optimal regime
    optimal = (blip > 0).astype(int)

    # Baseline outcome
    baseline = X[:, 0] + 0.5 * X[:, 1]

    # Outcome
    epsilon = rng.normal(0, 1, n)
    Y = baseline + blip * A + epsilon

    return {
        "outcome": Y,
        "treatment": A,
        "covariates": X,
        "true_blip": np.array(true_blip),
        "optimal_regime": optimal,
    }


def generate_principal_strat_data(
    n: int,
    cace: float = 2.0,
    compliance_rate: float = 0.6,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate data with noncompliance for principal stratification.

    Parameters
    ----------
    n : int
        Sample size.
    cace : float
        True CACE (complier average causal effect).
    compliance_rate : float
        Fraction of compliers.
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, assignment, received, true_cace, strata.
    """
    rng = np.random.default_rng(seed)

    # Strata: 0=never-taker, 1=complier, 2=always-taker
    # (no defiers under monotonicity)
    n_complier = int(n * compliance_rate)
    n_always = int(n * 0.2)
    n_never = n - n_complier - n_always

    strata = np.concatenate([
        np.zeros(n_never),
        np.ones(n_complier),
        np.full(n_always, 2),
    ]).astype(int)
    rng.shuffle(strata)

    # Random assignment
    Z = rng.binomial(1, 0.5, n)

    # Actual treatment received
    D = np.zeros(n, dtype=int)
    D[strata == 1] = Z[strata == 1]  # compliers follow assignment
    D[strata == 2] = 1  # always-takers always take

    # Potential outcomes
    # Y(0) ~ N(0, 1) for all
    # Y(1) = Y(0) + CACE for compliers, Y(0) for others
    Y0 = rng.normal(0, 1, n)
    Y1 = Y0.copy()
    Y1[strata == 1] += cace  # only compliers benefit

    # Observed outcome
    Y = np.where(D == 1, Y1, Y0)

    return {
        "outcome": Y,
        "assignment": Z,
        "received": D,
        "true_cace": cace,
        "strata": strata,
    }


def generate_rkd_data(
    n: int,
    effect_size: float = 1.5,
    kink_point: float = 0.0,
    bandwidth: float = 2.0,
    design: str = "sharp",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate regression kink design data.

    Model:
        Y = f(X) + tau * max(X - kink, 0) + epsilon
        where tau is the change in slope at the kink.

    Parameters
    ----------
    n : int
        Sample size.
    effect_size : float
        True kink effect (slope change, default 1.5).
    kink_point : float
        Location of kink (default 0.0).
    bandwidth : float
        Spread of running variable.
    design : str
        "sharp" or "fuzzy".
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, running_variable, kink_point, treatment_intensity, true_effect.
    """
    rng = np.random.default_rng(seed)

    # Running variable centered around kink
    X = rng.uniform(-bandwidth, bandwidth, n) + kink_point

    # Treatment intensity (distance above kink)
    D = np.maximum(X - kink_point, 0)

    if design == "fuzzy":
        # Add noise to treatment intensity
        D = D * (0.8 + 0.4 * rng.random(n))

    # Outcome: linear below kink, steeper above
    slope_below = 0.5
    baseline = slope_below * (X - kink_point)
    epsilon = rng.normal(0, 0.5, n)
    Y = baseline + effect_size * D + epsilon

    return {
        "outcome": Y,
        "running_variable": X,
        "kink_point": kink_point,
        "treatment_intensity": D,
        "true_effect": effect_size,
    }


def generate_bunching_data(
    n: int,
    threshold: float = 10000.0,
    bunching_mass: float = 0.15,
    counterfactual_density: str = "uniform",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate bunching estimator data.

    Model: Individuals bunch at threshold due to notch/kink.

    Parameters
    ----------
    n : int
        Sample size.
    threshold : float
        Bunching threshold (default 10000).
    bunching_mass : float
        Fraction of obs that bunch (default 0.15).
    counterfactual_density : str
        "uniform" or "normal".
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: observed_value, threshold, bunching_mass.
    """
    rng = np.random.default_rng(seed)

    # Counterfactual distribution (what would happen without bunching)
    spread = threshold * 0.3
    if counterfactual_density == "uniform":
        Z_star = rng.uniform(threshold - spread, threshold + spread, n)
    else:  # normal
        Z_star = rng.normal(threshold, spread / 2, n)

    # Bunching: individuals above threshold but within dominated region
    # move to threshold
    dominated_region = threshold * 0.1
    bunchers = (Z_star > threshold) & (Z_star < threshold + dominated_region)

    Z = Z_star.copy()
    Z[bunchers] = threshold  # bunch at threshold

    return {
        "observed_value": Z,
        "counterfactual_value": Z_star,
        "threshold": threshold,
        "bunching_mass": bunchers.mean(),
    }


def generate_selection_data(
    n: int,
    effect_size: float = 2.0,
    selection_strength: float = 0.7,
    n_covariates: int = 3,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate data with sample selection (Heckman model).

    Model:
        Selection: S* = gamma * Z + u, S = 1[S* > 0]
        Outcome: Y* = beta * X + tau * T + epsilon (only observed if S=1)

    Parameters
    ----------
    n : int
        Sample size (before selection).
    effect_size : float
        True treatment effect.
    selection_strength : float
        Correlation between selection and outcome errors.
    n_covariates : int
        Number of covariates.
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, treatment, covariates, selection_indicator,
              exclusion_restriction, true_effect.
    """
    rng = np.random.default_rng(seed)

    # Covariates
    X = rng.normal(0, 1, (n, n_covariates))

    # Exclusion restriction (affects selection but not outcome)
    Z = rng.normal(0, 1, n)

    # Correlated errors for selection bias
    u = rng.normal(0, 1, n)
    epsilon = selection_strength * u + np.sqrt(1 - selection_strength**2) * rng.normal(0, 1, n)

    # Selection equation
    S_star = 0.5 + 0.3 * X[:, 0] + 0.5 * Z + u
    S = (S_star > 0).astype(int)

    # Treatment
    T = rng.binomial(1, 0.5, n)

    # Outcome (latent for non-selected)
    Y_star = X[:, 0] + 0.5 * X[:, 1] + effect_size * T + epsilon

    # Observed outcome (NaN for non-selected)
    Y = np.where(S == 1, Y_star, np.nan)

    return {
        "outcome": Y,
        "treatment": T,
        "covariates": X,
        "selection_indicator": S,
        "exclusion_restriction": Z,
        "true_effect": effect_size,
    }


def generate_bounds_data(
    n: int,
    effect_size: float = 2.0,
    missing_rate: float = 0.3,
    missing_not_at_random: bool = True,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate data with missing outcomes for partial identification.

    Model: Outcomes missing potentially correlated with treatment.

    Parameters
    ----------
    n : int
        Sample size.
    effect_size : float
        True ATE.
    missing_rate : float
        Overall missing rate.
    missing_not_at_random : bool
        If True, missingness correlates with treatment.
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, treatment, observed, true_ate.
    """
    rng = np.random.default_rng(seed)

    # Treatment
    T = rng.binomial(1, 0.5, n)

    # True outcomes
    Y0 = rng.normal(0, 1, n)
    Y1 = Y0 + effect_size + rng.normal(0, 0.5, n)
    Y_true = np.where(T == 1, Y1, Y0)

    # Missingness
    if missing_not_at_random:
        # Higher outcomes more likely missing in treatment group
        miss_prob = missing_rate + 0.2 * T * (Y_true > np.median(Y_true))
        miss_prob = np.clip(miss_prob, 0, 0.8)
    else:
        miss_prob = np.full(n, missing_rate)

    observed = rng.binomial(1, 1 - miss_prob)

    # Observed outcome (NaN for missing)
    Y = np.where(observed == 1, Y_true, np.nan)

    return {
        "outcome": Y,
        "treatment": T,
        "observed": observed,
        "true_ate": effect_size,
    }


def generate_qte_data(
    n: int,
    effect_location: float = 1.0,
    effect_scale: float = 0.5,
    distribution: str = "heavy_tailed",
    n_covariates: int = 3,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate data for quantile treatment effects.

    Treatment affects both location and scale of outcome distribution.

    Parameters
    ----------
    n : int
        Sample size.
    effect_location : float
        Location shift from treatment.
    effect_scale : float
        Scale change from treatment (multiplicative).
    distribution : str
        "normal", "heavy_tailed" (t-distribution), or "skewed" (lognormal).
    n_covariates : int
        Number of covariates.
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, treatment, covariates, true_qte_median.
    """
    rng = np.random.default_rng(seed)

    # Covariates
    X = rng.normal(0, 1, (n, n_covariates))

    # Treatment
    T = rng.binomial(1, 0.5, n)

    # Error distribution
    if distribution == "normal":
        epsilon = rng.normal(0, 1, n)
    elif distribution == "heavy_tailed":
        epsilon = rng.standard_t(df=3, size=n)
    else:  # skewed
        epsilon = rng.lognormal(0, 0.5, n) - np.exp(0.5 * 0.5**2)  # centered

    # Baseline
    baseline = X[:, 0] + 0.5 * X[:, 1]

    # Outcome with location-scale treatment effect
    scale = 1 + effect_scale * T
    Y = baseline + effect_location * T + scale * epsilon

    return {
        "outcome": Y,
        "treatment": T,
        "covariates": X,
        "true_qte_median": effect_location,  # median shift
    }


def generate_mte_data(
    n: int,
    mte_intercept: float = 2.0,
    mte_slope: float = -1.5,
    n_covariates: int = 3,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate data for marginal treatment effects.

    Model with essential heterogeneity:
        Y = X'beta + T * (alpha + U_D * mte_slope) + epsilon
        Selection: D = 1[Z > V]

    Parameters
    ----------
    n : int
        Sample size.
    mte_intercept : float
        MTE at U_D = 0.
    mte_slope : float
        How MTE changes with U_D (negative = diminishing returns).
    n_covariates : int
        Number of covariates.
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, treatment, instrument, covariates, propensity,
              true_mte_intercept, true_mte_slope.
    """
    rng = np.random.default_rng(seed)

    # Covariates
    X = rng.normal(0, 1, (n, n_covariates))

    # Instrument (continuous)
    Z = rng.uniform(0, 1, n)

    # Unobserved heterogeneity in selection
    V = rng.uniform(0, 1, n)

    # Selection (treatment based on Z vs V)
    # Propensity P(Z) affects who is on margin
    T = (Z > V).astype(int)
    propensity = Z  # propensity score equals instrument value

    # MTE: returns vary by where individual is on selection margin
    # U_D = V for treated, representing their "type"
    U_D = V  # normalized unobserved type

    # Individual treatment effect
    individual_te = mte_intercept + mte_slope * U_D

    # Baseline
    baseline = X[:, 0] + 0.5 * X[:, 1]

    # Outcome
    epsilon = rng.normal(0, 1, n)
    Y = baseline + T * individual_te + epsilon

    return {
        "outcome": Y,
        "treatment": T,
        "instrument": Z,
        "covariates": X,
        "propensity": propensity,
        "true_mte_intercept": mte_intercept,
        "true_mte_slope": mte_slope,
    }


def generate_mediation_data(
    n: int,
    direct_effect: float = 1.0,
    indirect_effect: float = 0.8,
    n_covariates: int = 3,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate data for mediation analysis.

    Model:
        M = alpha * T + X'gamma + epsilon_m
        Y = direct * T + beta * M + X'delta + epsilon_y

    Parameters
    ----------
    n : int
        Sample size.
    direct_effect : float
        Direct effect of T on Y (not through M).
    indirect_effect : float
        Indirect effect (effect of T on M times effect of M on Y).
    n_covariates : int
        Number of covariates.
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys: outcome, treatment, mediator, covariates,
              true_direct, true_indirect, true_total.
    """
    rng = np.random.default_rng(seed)

    # Covariates
    X = rng.normal(0, 1, (n, n_covariates))

    # Treatment (randomized)
    T = rng.binomial(1, 0.5, n)

    # Decompose indirect = a * b where a=T->M, b=M->Y
    a = np.sqrt(abs(indirect_effect)) * np.sign(indirect_effect)
    b = np.sqrt(abs(indirect_effect))

    # Mediator equation
    epsilon_m = rng.normal(0, 0.5, n)
    M = a * T + 0.3 * X[:, 0] + epsilon_m

    # Outcome equation
    epsilon_y = rng.normal(0, 1, n)
    Y = direct_effect * T + b * M + 0.5 * X[:, 0] + epsilon_y

    return {
        "outcome": Y,
        "treatment": T,
        "mediator": M,
        "covariates": X,
        "true_direct": direct_effect,
        "true_indirect": indirect_effect,
        "true_total": direct_effect + indirect_effect,
    }


# Registry of all DGP functions
DGP_REGISTRY: Dict[str, callable] = {
    "rct": generate_rct_data,
    "observational": generate_observational_data,
    "psm": generate_psm_data,
    "did": generate_did_data,
    "iv": generate_iv_data,
    "rdd": generate_rdd_data,
    "scm": generate_scm_data,
    "cate": generate_cate_data,
    "panel": generate_panel_data,
    "dtr": generate_dtr_data,
    "principal_strat": generate_principal_strat_data,
    "rkd": generate_rkd_data,
    "bunching": generate_bunching_data,
    "selection": generate_selection_data,
    "bounds": generate_bounds_data,
    "qte": generate_qte_data,
    "mte": generate_mte_data,
    "mediation": generate_mediation_data,
}


def get_dgp(family: str) -> callable:
    """Get DGP function for a method family.

    Parameters
    ----------
    family : str
        Method family name.

    Returns
    -------
    callable
        DGP function.

    Raises
    ------
    ValueError
        If family not found.
    """
    if family not in DGP_REGISTRY:
        raise ValueError(
            f"Unknown family '{family}'. Available: {list(DGP_REGISTRY.keys())}"
        )
    return DGP_REGISTRY[family]
