"""
Test fixtures for mediation analysis.

Provides DGPs with known true effects for validation.
"""

import numpy as np
import pytest
from typing import Dict, Any


@pytest.fixture
def simple_linear_mediation():
    """
    Simple linear mediation with known effects.

    DGP:
        T ~ Bernoulli(0.5)
        M = alpha_0 + alpha_1 * T + e_m,  alpha_1 = 0.6
        Y = beta_0 + beta_1 * T + beta_2 * M + e_y,  beta_1 = 0.5, beta_2 = 0.8

    True effects:
        Indirect effect = alpha_1 * beta_2 = 0.6 * 0.8 = 0.48
        Direct effect = beta_1 = 0.5
        Total effect = 0.5 + 0.48 = 0.98
    """
    np.random.seed(42)
    n = 500

    # Treatment
    T = np.random.binomial(1, 0.5, n).astype(float)

    # Mediator: M = 0.5 + 0.6*T + e_m
    alpha_0, alpha_1 = 0.5, 0.6
    M = alpha_0 + alpha_1 * T + np.random.randn(n) * 0.5

    # Outcome: Y = 1.0 + 0.5*T + 0.8*M + e_y
    beta_0, beta_1, beta_2 = 1.0, 0.5, 0.8
    Y = beta_0 + beta_1 * T + beta_2 * M + np.random.randn(n) * 0.5

    return {
        "outcome": Y,
        "treatment": T,
        "mediator": M,
        "true_indirect": alpha_1 * beta_2,  # 0.48
        "true_direct": beta_1,  # 0.5
        "true_total": beta_1 + alpha_1 * beta_2,  # 0.98
        "true_alpha_1": alpha_1,
        "true_beta_1": beta_1,
        "true_beta_2": beta_2,
        "n": n,
    }


@pytest.fixture
def full_mediation():
    """
    Full mediation: direct effect = 0, all effect through mediator.

    DGP:
        T ~ Bernoulli(0.5)
        M = 0.5 + 0.6 * T + e_m
        Y = 1.0 + 0*T + 0.8*M + e_y  (no direct effect!)

    True effects:
        Indirect = 0.6 * 0.8 = 0.48
        Direct = 0
        Total = 0.48
    """
    np.random.seed(123)
    n = 500

    T = np.random.binomial(1, 0.5, n).astype(float)
    M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
    Y = 1.0 + 0.0 * T + 0.8 * M + np.random.randn(n) * 0.5

    return {
        "outcome": Y,
        "treatment": T,
        "mediator": M,
        "true_indirect": 0.48,
        "true_direct": 0.0,
        "true_total": 0.48,
        "n": n,
    }


@pytest.fixture
def no_mediation():
    """
    No mediation: M has no effect on Y.

    DGP:
        T ~ Bernoulli(0.5)
        M = 0.5 + 0.6 * T + e_m  (T affects M)
        Y = 1.0 + 0.5*T + 0*M + e_y  (M doesn't affect Y)

    True effects:
        Indirect = 0.6 * 0 = 0
        Direct = 0.5
        Total = 0.5
    """
    np.random.seed(456)
    n = 500

    T = np.random.binomial(1, 0.5, n).astype(float)
    M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
    Y = 1.0 + 0.5 * T + 0.0 * M + np.random.randn(n) * 0.5

    return {
        "outcome": Y,
        "treatment": T,
        "mediator": M,
        "true_indirect": 0.0,
        "true_direct": 0.5,
        "true_total": 0.5,
        "n": n,
    }


@pytest.fixture
def no_treatment_on_mediator():
    """
    Treatment doesn't affect mediator (alpha_1 = 0).

    DGP:
        T ~ Bernoulli(0.5)
        M = 0.5 + 0*T + e_m  (T doesn't affect M)
        Y = 1.0 + 0.5*T + 0.8*M + e_y

    True effects:
        Indirect = 0 * 0.8 = 0
        Direct = 0.5
        Total = 0.5
    """
    np.random.seed(789)
    n = 500

    T = np.random.binomial(1, 0.5, n).astype(float)
    M = 0.5 + 0.0 * T + np.random.randn(n) * 0.5
    Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5

    return {
        "outcome": Y,
        "treatment": T,
        "mediator": M,
        "true_indirect": 0.0,
        "true_direct": 0.5,
        "true_total": 0.5,
        "n": n,
    }


@pytest.fixture
def mediation_with_covariates():
    """
    Mediation with pre-treatment covariates.

    DGP:
        X ~ N(0, 1)
        T ~ Bernoulli(expit(0.5*X))
        M = 0.5 + 0.6*T + 0.3*X + e_m
        Y = 1.0 + 0.5*T + 0.8*M + 0.4*X + e_y

    True effects (conditional on X):
        Indirect = 0.48
        Direct = 0.5
        Total = 0.98
    """
    np.random.seed(321)
    n = 600

    X = np.random.randn(n)
    prob = 1 / (1 + np.exp(-0.5 * X))
    T = (np.random.rand(n) < prob).astype(float)
    M = 0.5 + 0.6 * T + 0.3 * X + np.random.randn(n) * 0.5
    Y = 1.0 + 0.5 * T + 0.8 * M + 0.4 * X + np.random.randn(n) * 0.5

    return {
        "outcome": Y,
        "treatment": T,
        "mediator": M,
        "covariates": X.reshape(-1, 1),
        "true_indirect": 0.48,
        "true_direct": 0.5,
        "true_total": 0.98,
        "n": n,
    }


@pytest.fixture
def binary_mediator():
    """
    Binary mediator (logistic mediator model needed).

    DGP:
        T ~ Bernoulli(0.5)
        M ~ Bernoulli(expit(-0.5 + 1.2*T))  (binary mediator)
        Y = 1.0 + 0.5*T + 0.8*M + e_y

    True effects (approximate):
        P(M=1|T=1) - P(M=1|T=0) ≈ 0.25 (rough first stage)
        Indirect ≈ 0.25 * 0.8 = 0.20
        Direct = 0.5
    """
    np.random.seed(654)
    n = 600

    T = np.random.binomial(1, 0.5, n).astype(float)
    prob_m = 1 / (1 + np.exp(-(-0.5 + 1.2 * T)))
    M = (np.random.rand(n) < prob_m).astype(float)
    Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5

    return {
        "outcome": Y,
        "treatment": T,
        "mediator": M,
        "true_direct": 0.5,
        "is_binary_mediator": True,
        "n": n,
    }


@pytest.fixture
def continuous_treatment():
    """
    Continuous treatment variable.

    DGP:
        T ~ N(0, 1)
        M = 0.5 + 0.6*T + e_m
        Y = 1.0 + 0.5*T + 0.8*M + e_y

    True effects (per unit change in T):
        Indirect = 0.48
        Direct = 0.5
        Total = 0.98
    """
    np.random.seed(987)
    n = 500

    T = np.random.randn(n)
    M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
    Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5

    return {
        "outcome": Y,
        "treatment": T,
        "mediator": M,
        "true_indirect": 0.48,
        "true_direct": 0.5,
        "true_total": 0.98,
        "n": n,
    }


@pytest.fixture
def large_sample():
    """Large sample for Monte Carlo precision."""
    np.random.seed(111)
    n = 2000

    T = np.random.binomial(1, 0.5, n).astype(float)
    M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
    Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5

    return {
        "outcome": Y,
        "treatment": T,
        "mediator": M,
        "true_indirect": 0.48,
        "true_direct": 0.5,
        "true_total": 0.98,
        "n": n,
    }


@pytest.fixture
def small_sample():
    """Small sample for edge case testing."""
    np.random.seed(222)
    n = 50

    T = np.random.binomial(1, 0.5, n).astype(float)
    M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
    Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5

    return {
        "outcome": Y,
        "treatment": T,
        "mediator": M,
        "n": n,
    }


def generate_mediation_dgp(
    n: int = 500,
    alpha_1: float = 0.6,
    beta_1: float = 0.5,
    beta_2: float = 0.8,
    sigma_m: float = 0.5,
    sigma_y: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate mediation DGP with custom parameters.

    Parameters
    ----------
    n : int
        Sample size
    alpha_1 : float
        T -> M effect
    beta_1 : float
        T -> Y direct effect
    beta_2 : float
        M -> Y effect
    sigma_m : float
        Mediator noise SD
    sigma_y : float
        Outcome noise SD
    seed : int
        Random seed

    Returns
    -------
    dict
        DGP data and true effects
    """
    np.random.seed(seed)

    T = np.random.binomial(1, 0.5, n).astype(float)
    M = 0.5 + alpha_1 * T + np.random.randn(n) * sigma_m
    Y = 1.0 + beta_1 * T + beta_2 * M + np.random.randn(n) * sigma_y

    return {
        "outcome": Y,
        "treatment": T,
        "mediator": M,
        "true_indirect": alpha_1 * beta_2,
        "true_direct": beta_1,
        "true_total": beta_1 + alpha_1 * beta_2,
        "n": n,
        "seed": seed,
    }
