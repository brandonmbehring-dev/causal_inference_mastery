"""
Shared fixtures for PSM tests.

Provides known-answer test data with documented true ATEs.
"""

import numpy as np
import pytest


@pytest.fixture
def simple_psm_data():
    """
    Simple PSM example with known ATE.

    Setup:
    - 100 units total (50 treated, 50 control)
    - 2 covariates: X1 ~ N(0, 1), X2 ~ N(0, 1)
    - Propensity: P(T=1|X) = logit(0.5*X1 + 0.3*X2)
    - Outcome: Y = 2.0*T + 0.5*X1 + 0.3*X2 + ε, ε ~ N(0, 0.5)
    - True ATE = 2.0

    Returns:
        dict with keys: outcomes, treatment, covariates, true_ate
    """
    np.random.seed(42)
    n = 100

    # Covariates
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    covariates = np.column_stack([X1, X2])

    # Treatment assignment via propensity score
    logit_p = 0.5 * X1 + 0.3 * X2
    propensity = 1 / (1 + np.exp(-logit_p))
    treatment = np.random.binomial(1, propensity, n).astype(bool)

    # Outcome (linear response with treatment effect = 2.0)
    noise = np.random.normal(0, 0.5, n)
    outcomes = 2.0 * treatment + 0.5 * X1 + 0.3 * X2 + noise

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "covariates": covariates,
        "true_ate": 2.0,
    }


@pytest.fixture
def perfect_overlap_data():
    """
    PSM with perfect common support (full overlap).

    Setup:
    - 80 units (40 treated, 40 control)
    - 1 covariate: X ~ N(0, 1)
    - Propensity: P(T=1|X) = 0.5 (constant, randomized)
    - Outcome: Y = 3.0*T + 0.8*X + ε, ε ~ N(0, 1)
    - True ATE = 3.0

    Perfect overlap ensures all units matchable.

    Returns:
        dict with keys: outcomes, treatment, covariates, true_ate
    """
    np.random.seed(123)
    n = 80

    # Single covariate
    X = np.random.normal(0, 1, n)
    covariates = X.reshape(-1, 1)

    # Randomized treatment (constant propensity = 0.5)
    treatment = np.random.binomial(1, 0.5, n).astype(bool)

    # Outcome (treatment effect = 3.0)
    noise = np.random.normal(0, 1, n)
    outcomes = 3.0 * treatment + 0.8 * X + noise

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "covariates": covariates,
        "true_ate": 3.0,
    }


@pytest.fixture
def limited_overlap_data():
    """
    PSM with limited common support (partial overlap).

    Setup:
    - 100 units (30 treated, 70 control)
    - 1 covariate: X_treated ~ N(2, 1), X_control ~ N(0, 1)
    - Limited overlap at X ∈ [1, 3]
    - Outcome: Y = 1.5*T + 0.6*X + ε, ε ~ N(0, 0.8)
    - True ATE on overlap region ≈ 1.5

    Tests matching with units outside common support.

    Returns:
        dict with keys: outcomes, treatment, covariates, true_ate
    """
    np.random.seed(456)

    # Treated units: higher covariate values
    n_treated = 30
    X_treated = np.random.normal(2, 1, n_treated)
    treatment_treated = np.ones(n_treated, dtype=bool)

    # Control units: lower covariate values
    n_control = 70
    X_control = np.random.normal(0, 1, n_control)
    treatment_control = np.zeros(n_control, dtype=bool)

    # Combine
    X = np.concatenate([X_treated, X_control])
    covariates = X.reshape(-1, 1)
    treatment = np.concatenate([treatment_treated, treatment_control])

    # Outcome (treatment effect = 1.5)
    noise = np.random.normal(0, 0.8, len(X))
    outcomes = 1.5 * treatment + 0.6 * X + noise

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "covariates": covariates,
        "true_ate": 1.5,  # ATE on overlap region
    }


@pytest.fixture
def binary_covariate_data():
    """
    PSM with binary covariate (stratification scenario).

    Setup:
    - 60 units (30 treated, 30 control)
    - 1 binary covariate: X ∈ {0, 1} with P(X=1) = 0.5
    - Propensity: P(T=1|X) ≈ 0.5 (balanced within strata)
    - Outcome: Y = 2.5*T + 1.0*X + ε, ε ~ N(0, 0.5)
    - True ATE = 2.5

    Tests matching with discrete covariates (exact ties).

    Returns:
        dict with keys: outcomes, treatment, covariates, true_ate
    """
    np.random.seed(789)
    n = 60

    # Binary covariate
    X = np.random.binomial(1, 0.5, n).astype(float)
    covariates = X.reshape(-1, 1)

    # Balanced treatment within strata
    treatment = np.zeros(n, dtype=bool)
    for x_val in [0, 1]:
        idx = np.where(X == x_val)[0]
        n_stratum = len(idx)
        n_treat = n_stratum // 2
        treatment[idx[:n_treat]] = True

    # Outcome (treatment effect = 2.5)
    noise = np.random.normal(0, 0.5, n)
    outcomes = 2.5 * treatment + 1.0 * X + noise

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "covariates": covariates,
        "true_ate": 2.5,
    }


@pytest.fixture
def high_dimensional_data():
    """
    PSM with many covariates (tests propensity estimation).

    Setup:
    - 150 units (75 treated, 75 control)
    - 10 covariates: X_j ~ N(0, 1) for j=1..10
    - Propensity: P(T=1|X) = logit(∑ⱼ 0.2*X_j)
    - Outcome: Y = 1.8*T + ∑ⱼ 0.1*X_j + ε, ε ~ N(0, 0.6)
    - True ATE = 1.8

    Tests propensity estimation and matching with many dimensions.

    Returns:
        dict with keys: outcomes, treatment, covariates, true_ate
    """
    np.random.seed(321)
    n = 150
    p = 10

    # 10 covariates
    covariates = np.random.normal(0, 1, (n, p))

    # Treatment via propensity score
    logit_p = np.sum(0.2 * covariates, axis=1)
    propensity = 1 / (1 + np.exp(-logit_p))
    treatment = np.random.binomial(1, propensity, n).astype(bool)

    # Outcome (treatment effect = 1.8)
    noise = np.random.normal(0, 0.6, n)
    outcomes = 1.8 * treatment + np.sum(0.1 * covariates, axis=1) + noise

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "covariates": covariates,
        "true_ate": 1.8,
    }
