"""
Data Generating Processes (DGPs) for CATE Monte Carlo validation.

This module provides DGPs for validating heterogeneous treatment effect estimators:
- S-Learner, T-Learner, X-Learner, R-Learner
- Double Machine Learning (DML)
- Causal Forests

All DGPs have known true ATE and CATE functions for validation.

Key References:
    - Kunzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"
    - Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects"
    - Wager & Athey (2018). "Estimation and inference of heterogeneous treatment effects"
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class CATEData:
    """Container for CATE simulation data with known ground truth.

    Attributes
    ----------
    Y : np.ndarray
        Outcome variable (n,)
    T : np.ndarray
        Binary treatment indicator (n,)
    X : np.ndarray
        Covariates (n, p)
    true_ate : float
        True average treatment effect E[τ(X)]
    true_cate : np.ndarray
        True conditional treatment effects τ(X) for each observation (n,)
    cate_function : Callable
        Function τ(X) that computes true CATE
    propensity : np.ndarray
        True propensity scores P(T=1|X) (n,)
    n : int
        Sample size
    p : int
        Number of covariates
    """

    Y: np.ndarray
    T: np.ndarray
    X: np.ndarray
    true_ate: float
    true_cate: np.ndarray
    cate_function: Callable
    propensity: np.ndarray
    n: int
    p: int


# =============================================================================
# Constant Effect DGP (Homogeneous)
# =============================================================================


def dgp_constant_effect(
    n: int = 1000,
    true_ate: float = 2.0,
    p: int = 5,
    propensity_strength: float = 0.5,
    random_state: Optional[int] = None,
) -> CATEData:
    """
    CATE DGP with constant treatment effect (no heterogeneity).

    Model:
        Y(0) = X @ beta + epsilon
        Y(1) = Y(0) + tau (constant)
        T ~ Bernoulli(expit(propensity_strength * X[:, 0]))

    Parameters
    ----------
    n : int, default=1000
        Sample size
    true_ate : float, default=2.0
        True constant treatment effect
    p : int, default=5
        Number of covariates
    propensity_strength : float, default=0.5
        Strength of propensity model (higher = more selection on X)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    CATEData
        Simulation data with known ground truth
    """
    rng = np.random.default_rng(random_state)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Propensity model: logistic on X[:, 0]
    logit = propensity_strength * X[:, 0]
    propensity = 1 / (1 + np.exp(-logit))
    T = rng.binomial(1, propensity)

    # Outcome model: linear in X
    beta = rng.standard_normal(p) * 0.5
    Y0 = X @ beta + rng.standard_normal(n)

    # Constant CATE
    def cate_function(x):
        return np.full(x.shape[0], true_ate)

    true_cate = cate_function(X)

    # Observed outcome
    Y = Y0 + T * true_ate

    return CATEData(
        Y=Y,
        T=T,
        X=X,
        true_ate=true_ate,
        true_cate=true_cate,
        cate_function=cate_function,
        propensity=propensity,
        n=n,
        p=p,
    )


# =============================================================================
# Linear Heterogeneity DGP
# =============================================================================


def dgp_linear_heterogeneity(
    n: int = 1000,
    base_effect: float = 2.0,
    het_coef: float = 1.0,
    p: int = 5,
    propensity_strength: float = 0.5,
    random_state: Optional[int] = None,
) -> CATEData:
    """
    CATE DGP with linear heterogeneity in first covariate.

    Model:
        tau(X) = base_effect + het_coef * X[:, 0]
        Y(0) = X @ beta + epsilon
        Y(1) = Y(0) + tau(X)

    Parameters
    ----------
    n : int, default=1000
        Sample size
    base_effect : float, default=2.0
        Intercept of CATE function
    het_coef : float, default=1.0
        Coefficient on X[:, 0] for heterogeneity
    p : int, default=5
        Number of covariates
    propensity_strength : float, default=0.5
        Strength of propensity model
    random_state : int, optional
        Random seed

    Returns
    -------
    CATEData
        Simulation data with known ground truth
    """
    rng = np.random.default_rng(random_state)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Propensity model
    logit = propensity_strength * X[:, 0]
    propensity = 1 / (1 + np.exp(-logit))
    T = rng.binomial(1, propensity)

    # Outcome model
    beta = rng.standard_normal(p) * 0.5
    Y0 = X @ beta + rng.standard_normal(n)

    # Linear CATE
    def cate_function(x):
        return base_effect + het_coef * x[:, 0]

    true_cate = cate_function(X)
    true_ate = np.mean(true_cate)

    # Observed outcome
    Y = Y0 + T * true_cate

    return CATEData(
        Y=Y,
        T=T,
        X=X,
        true_ate=true_ate,
        true_cate=true_cate,
        cate_function=cate_function,
        propensity=propensity,
        n=n,
        p=p,
    )


# =============================================================================
# Nonlinear Heterogeneity DGP
# =============================================================================


def dgp_nonlinear_heterogeneity(
    n: int = 1000,
    base_effect: float = 2.0,
    amplitude: float = 0.5,
    p: int = 5,
    propensity_strength: float = 0.5,
    random_state: Optional[int] = None,
) -> CATEData:
    """
    CATE DGP with nonlinear (sinusoidal) heterogeneity.

    Model:
        tau(X) = base_effect * (1 + amplitude * sin(X[:, 0]))
        Y(0) = X @ beta + epsilon
        Y(1) = Y(0) + tau(X)

    Parameters
    ----------
    n : int, default=1000
        Sample size
    base_effect : float, default=2.0
        Base treatment effect
    amplitude : float, default=0.5
        Amplitude of sinusoidal modulation (in [0, 1])
    p : int, default=5
        Number of covariates
    propensity_strength : float, default=0.5
        Strength of propensity model
    random_state : int, optional
        Random seed

    Returns
    -------
    CATEData
        Simulation data with known ground truth
    """
    rng = np.random.default_rng(random_state)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Propensity model
    logit = propensity_strength * X[:, 0]
    propensity = 1 / (1 + np.exp(-logit))
    T = rng.binomial(1, propensity)

    # Outcome model
    beta = rng.standard_normal(p) * 0.5
    Y0 = X @ beta + rng.standard_normal(n)

    # Nonlinear CATE
    def cate_function(x):
        return base_effect * (1 + amplitude * np.sin(x[:, 0]))

    true_cate = cate_function(X)
    true_ate = np.mean(true_cate)

    # Observed outcome
    Y = Y0 + T * true_cate

    return CATEData(
        Y=Y,
        T=T,
        X=X,
        true_ate=true_ate,
        true_cate=true_cate,
        cate_function=cate_function,
        propensity=propensity,
        n=n,
        p=p,
    )


# =============================================================================
# Complex Heterogeneity DGP (Step + Linear)
# =============================================================================


def dgp_complex_heterogeneity(
    n: int = 1000,
    base_effect: float = 1.0,
    step_effect: float = 2.0,
    linear_coef: float = 0.5,
    p: int = 5,
    propensity_strength: float = 0.5,
    random_state: Optional[int] = None,
) -> CATEData:
    """
    CATE DGP with complex heterogeneity (step function + linear).

    Model:
        tau(X) = base_effect + step_effect * I(X[:, 0] > 0) + linear_coef * X[:, 1]
        Y(0) = X @ beta + epsilon
        Y(1) = Y(0) + tau(X)

    This tests ability to detect both discrete and continuous heterogeneity.

    Parameters
    ----------
    n : int, default=1000
        Sample size
    base_effect : float, default=1.0
        Baseline effect
    step_effect : float, default=2.0
        Additional effect when X[:, 0] > 0
    linear_coef : float, default=0.5
        Coefficient on X[:, 1]
    p : int, default=5
        Number of covariates (must be >= 2)
    propensity_strength : float, default=0.5
        Strength of propensity model
    random_state : int, optional
        Random seed

    Returns
    -------
    CATEData
        Simulation data with known ground truth
    """
    if p < 2:
        raise ValueError("p must be >= 2 for complex heterogeneity DGP")

    rng = np.random.default_rng(random_state)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Propensity model
    logit = propensity_strength * X[:, 0]
    propensity = 1 / (1 + np.exp(-logit))
    T = rng.binomial(1, propensity)

    # Outcome model
    beta = rng.standard_normal(p) * 0.5
    Y0 = X @ beta + rng.standard_normal(n)

    # Complex CATE
    def cate_function(x):
        return base_effect + step_effect * (x[:, 0] > 0) + linear_coef * x[:, 1]

    true_cate = cate_function(X)
    true_ate = np.mean(true_cate)

    # Observed outcome
    Y = Y0 + T * true_cate

    return CATEData(
        Y=Y,
        T=T,
        X=X,
        true_ate=true_ate,
        true_cate=true_cate,
        cate_function=cate_function,
        propensity=propensity,
        n=n,
        p=p,
    )


# =============================================================================
# High-Dimensional DGP
# =============================================================================


def dgp_high_dimensional(
    n: int = 500,
    p: int = 50,
    true_ate: float = 2.0,
    n_relevant: int = 5,
    propensity_strength: float = 0.3,
    random_state: Optional[int] = None,
) -> CATEData:
    """
    CATE DGP with high-dimensional covariates (sparse true model).

    Model:
        tau(X) = true_ate + X[:, :n_relevant] @ het_coefs
        Y(0) = X[:, :n_relevant] @ beta + epsilon
        T ~ Bernoulli(expit(propensity_strength * X[:, 0]))

    Only first n_relevant covariates matter; rest are noise.

    Parameters
    ----------
    n : int, default=500
        Sample size
    p : int, default=50
        Total number of covariates
    true_ate : float, default=2.0
        Average treatment effect
    n_relevant : int, default=5
        Number of relevant covariates
    propensity_strength : float, default=0.3
        Strength of propensity model
    random_state : int, optional
        Random seed

    Returns
    -------
    CATEData
        Simulation data with known ground truth
    """
    if n_relevant > p:
        raise ValueError("n_relevant must be <= p")

    rng = np.random.default_rng(random_state)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Propensity model (only depends on X[:, 0])
    logit = propensity_strength * X[:, 0]
    propensity = 1 / (1 + np.exp(-logit))
    T = rng.binomial(1, propensity)

    # Outcome model (sparse)
    beta = np.zeros(p)
    beta[:n_relevant] = rng.standard_normal(n_relevant) * 0.5
    Y0 = X @ beta + rng.standard_normal(n)

    # CATE with sparse heterogeneity
    het_coefs = np.zeros(p)
    het_coefs[:n_relevant] = rng.standard_normal(n_relevant) * 0.3

    def cate_function(x):
        return true_ate + x @ het_coefs

    true_cate = cate_function(X)
    actual_ate = np.mean(true_cate)

    # Observed outcome
    Y = Y0 + T * true_cate

    return CATEData(
        Y=Y,
        T=T,
        X=X,
        true_ate=actual_ate,
        true_cate=true_cate,
        cate_function=cate_function,
        propensity=propensity,
        n=n,
        p=p,
    )


# =============================================================================
# Imbalanced Treatment DGP
# =============================================================================


def dgp_imbalanced_treatment(
    n: int = 1000,
    true_ate: float = 2.0,
    treatment_prob: float = 0.1,
    p: int = 5,
    random_state: Optional[int] = None,
) -> CATEData:
    """
    CATE DGP with imbalanced treatment assignment (few treated).

    Model:
        tau(X) = true_ate (constant)
        T ~ Bernoulli(treatment_prob)  # Independent of X
        Y(0) = X @ beta + epsilon
        Y(1) = Y(0) + tau

    Parameters
    ----------
    n : int, default=1000
        Sample size
    true_ate : float, default=2.0
        True treatment effect
    treatment_prob : float, default=0.1
        Probability of treatment (should be small or large for imbalance)
    p : int, default=5
        Number of covariates
    random_state : int, optional
        Random seed

    Returns
    -------
    CATEData
        Simulation data with known ground truth
    """
    rng = np.random.default_rng(random_state)

    # Generate covariates
    X = rng.standard_normal((n, p))

    # Constant propensity (imbalanced)
    propensity = np.full(n, treatment_prob)
    T = rng.binomial(1, propensity)

    # Outcome model
    beta = rng.standard_normal(p) * 0.5
    Y0 = X @ beta + rng.standard_normal(n)

    # Constant CATE
    def cate_function(x):
        return np.full(x.shape[0], true_ate)

    true_cate = cate_function(X)

    # Observed outcome
    Y = Y0 + T * true_ate

    return CATEData(
        Y=Y,
        T=T,
        X=X,
        true_ate=true_ate,
        true_cate=true_cate,
        cate_function=cate_function,
        propensity=propensity,
        n=n,
        p=p,
    )
