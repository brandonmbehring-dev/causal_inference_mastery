"""
Fixtures for Bayesian causal inference tests.

Session 101: Initial fixtures for Bayesian ATE tests.
"""

import numpy as np
import pytest


@pytest.fixture
def simple_bayesian_data() -> dict:
    """
    Simple RCT data for Bayesian testing.

    Returns
    -------
    dict
        outcomes, treatment, true_ate=2.0
    """
    np.random.seed(42)
    n = 200
    treatment = np.random.binomial(1, 0.5, n)
    true_ate = 2.0
    outcomes = true_ate * treatment + np.random.normal(0, 1, n)
    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "true_ate": true_ate,
    }


@pytest.fixture
def bayesian_data_with_covariates() -> dict:
    """
    Data with covariates for covariate-adjusted Bayesian estimation.

    Returns
    -------
    dict
        outcomes, treatment, covariates, true_ate=3.0
    """
    np.random.seed(123)
    n = 300
    X = np.random.randn(n, 2)
    treatment = (X[:, 0] + np.random.randn(n) > 0).astype(float)
    true_ate = 3.0
    outcomes = true_ate * treatment + 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.normal(0, 1, n)
    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "covariates": X,
        "true_ate": true_ate,
    }


@pytest.fixture
def small_sample_data() -> dict:
    """
    Small sample data where prior should influence posterior.

    Returns
    -------
    dict
        outcomes, treatment (n=20)
    """
    np.random.seed(456)
    n = 20
    treatment = np.random.binomial(1, 0.5, n)
    true_ate = 2.0
    outcomes = true_ate * treatment + np.random.normal(0, 1, n)
    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "true_ate": true_ate,
    }


@pytest.fixture
def large_sample_data() -> dict:
    """
    Large sample data where data should dominate prior.

    Returns
    -------
    dict
        outcomes, treatment (n=2000)
    """
    np.random.seed(789)
    n = 2000
    treatment = np.random.binomial(1, 0.5, n)
    true_ate = 2.0
    outcomes = true_ate * treatment + np.random.normal(0, 1, n)
    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "true_ate": true_ate,
    }


@pytest.fixture
def zero_effect_data() -> dict:
    """
    Data with zero true treatment effect.

    Returns
    -------
    dict
        outcomes, treatment, true_ate=0.0
    """
    np.random.seed(101)
    n = 200
    treatment = np.random.binomial(1, 0.5, n)
    outcomes = np.random.normal(0, 1, n)  # No treatment effect
    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "true_ate": 0.0,
    }
