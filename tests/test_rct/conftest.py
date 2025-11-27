"""
Pytest fixtures for RCT estimator tests.

Provides reusable test data fixtures to reduce code duplication across test files.
All fixtures use fixed random seeds for reproducibility.
"""

import numpy as np
import pytest


@pytest.fixture
def simple_rct_data():
    """
    Basic balanced RCT data (n=100, true ATE=4.0).

    Returns
    -------
    dict with keys:
        - outcomes: np.ndarray (n=100)
        - treatment: np.ndarray (n=100, 50 treated, 50 control)
        - true_ate: float (4.0)
    """
    np.random.seed(42)
    treatment = np.array([1] * 50 + [0] * 50)
    outcomes = np.where(
        treatment == 1, np.random.normal(6.0, 2.0, 100), np.random.normal(2.0, 2.0, 100)
    )
    return {"outcomes": outcomes, "treatment": treatment, "true_ate": 4.0}


@pytest.fixture
def unbalanced_rct_data():
    """
    Unbalanced RCT data (n=100, n1=30, n0=70, true ATE=3.0).

    Returns
    -------
    dict with keys:
        - outcomes: np.ndarray
        - treatment: np.ndarray (30 treated, 70 control)
        - true_ate: float
    """
    np.random.seed(123)
    treatment = np.array([1] * 30 + [0] * 70)
    outcomes = np.where(
        treatment == 1, np.random.normal(5.0, 1.5, 100), np.random.normal(2.0, 1.5, 100)
    )
    return {"outcomes": outcomes, "treatment": treatment, "true_ate": 3.0}


@pytest.fixture
def small_sample_data():
    """
    Small sample RCT (n=20, true ATE=5.0).

    Useful for testing small-sample behavior (t-distribution).

    Returns
    -------
    dict
    """
    np.random.seed(789)
    treatment = np.array([1] * 10 + [0] * 10)
    outcomes = np.where(
        treatment == 1, np.random.normal(7.0, 2.0, 20), np.random.normal(2.0, 2.0, 20)
    )
    return {"outcomes": outcomes, "treatment": treatment, "true_ate": 5.0}


@pytest.fixture
def large_sample_data():
    """
    Large sample RCT (n=1000, true ATE=2.0).

    Useful for testing asymptotic properties.

    Returns
    -------
    dict
    """
    np.random.seed(456)
    treatment = np.array([1] * 500 + [0] * 500)
    outcomes = np.where(
        treatment == 1, np.random.normal(2.0, 1.0, 1000), np.random.normal(0.0, 1.0, 1000)
    )
    return {"outcomes": outcomes, "treatment": treatment, "true_ate": 2.0}


@pytest.fixture
def heteroskedastic_data():
    """
    RCT with heteroskedastic errors (different variances by group).

    Treated: sigma=3.0
    Control: sigma=1.0
    True ATE: 4.0

    Returns
    -------
    dict
    """
    np.random.seed(111)
    treatment = np.array([1] * 50 + [0] * 50)
    outcomes = np.where(
        treatment == 1, np.random.normal(6.0, 3.0, 100), np.random.normal(2.0, 1.0, 100)
    )
    return {"outcomes": outcomes, "treatment": treatment, "true_ate": 4.0}


@pytest.fixture
def zero_effect_data():
    """
    RCT with zero treatment effect (true ATE=0.0).

    Returns
    -------
    dict
    """
    np.random.seed(222)
    treatment = np.array([1] * 50 + [0] * 50)
    outcomes = np.where(
        treatment == 1, np.random.normal(5.0, 2.0, 100), np.random.normal(5.0, 2.0, 100)
    )
    return {"outcomes": outcomes, "treatment": treatment, "true_ate": 0.0}


@pytest.fixture
def negative_effect_data():
    """
    RCT with negative treatment effect (true ATE=-3.0).

    Treatment harms outcomes.

    Returns
    -------
    dict
    """
    np.random.seed(333)
    treatment = np.array([1] * 50 + [0] * 50)
    outcomes = np.where(
        treatment == 1, np.random.normal(2.0, 1.5, 100), np.random.normal(5.0, 1.5, 100)
    )
    return {"outcomes": outcomes, "treatment": treatment, "true_ate": -3.0}


@pytest.fixture
def constant_propensity_data():
    """
    RCT data with constant propensity scores (all P(T=1)=0.5).

    Useful for IPW tests where weights should be constant.

    Returns
    -------
    dict with keys:
        - outcomes, treatment, propensity, true_ate
    """
    np.random.seed(444)
    treatment = np.array([1] * 50 + [0] * 50)
    propensity = np.ones(100) * 0.5
    outcomes = np.where(
        treatment == 1, np.random.normal(4.0, 1.0, 100), np.random.normal(2.0, 1.0, 100)
    )
    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "propensity": propensity,
        "true_ate": 2.0,
    }


@pytest.fixture
def varying_propensity_data():
    """
    RCT data with varying propensity scores.

    Simulates blocked randomization with varying assignment probabilities.

    Returns
    -------
    dict
    """
    np.random.seed(555)
    n = 100
    # Propensities vary from 0.2 to 0.8
    propensity = np.linspace(0.2, 0.8, n)
    treatment = (np.random.uniform(0, 1, n) < propensity).astype(float)
    outcomes = np.where(
        treatment == 1, np.random.normal(3.0, 1.0, n), np.random.normal(1.0, 1.0, n)
    )
    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "propensity": propensity,
        "true_ate": 2.0,
    }


@pytest.fixture
def single_covariate_data():
    """
    RCT with single covariate X ~ N(0, 1).

    Useful for regression adjustment tests.

    Returns
    -------
    dict with keys:
        - outcomes, treatment, X, true_ate
    """
    np.random.seed(666)
    n = 100
    X = np.random.normal(0, 1, n)
    treatment = np.array([1] * 50 + [0] * 50)
    # Y = 2*T + 3*X + epsilon
    outcomes = 2.0 * treatment + 3.0 * X + np.random.normal(0, 1, n)
    return {"outcomes": outcomes, "treatment": treatment, "X": X, "true_ate": 2.0}


@pytest.fixture
def multi_covariate_data():
    """
    RCT with multiple covariates (X1, X2, X3).

    Useful for testing multi-covariate regression adjustment.

    Returns
    -------
    dict
    """
    np.random.seed(777)
    n = 100
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(5, 2, n)
    X3 = np.random.uniform(-1, 1, n)
    X = np.column_stack([X1, X2, X3])
    treatment = np.array([1] * 50 + [0] * 50)
    # Y = 3*T + 2*X1 + 1*X2 + 0.5*X3 + epsilon
    outcomes = (
        3.0 * treatment + 2.0 * X1 + 1.0 * X2 + 0.5 * X3 + np.random.normal(0, 1, n)
    )
    return {"outcomes": outcomes, "treatment": treatment, "X": X, "true_ate": 3.0}


@pytest.fixture
def stratified_data():
    """
    Stratified RCT with 3 strata (n=120, 40 per stratum).

    Each stratum has different baseline but same treatment effect (ATE=4.0).

    Returns
    -------
    dict
    """
    np.random.seed(888)
    outcomes = []
    treatment = []
    strata = []

    baselines = [0, 10, 20]  # Different baseline per stratum
    for s, baseline in enumerate(baselines):
        t = np.array([1] * 20 + [0] * 20)
        np.random.shuffle(t)
        y = np.where(
            t == 1,
            np.random.normal(baseline + 4.0, 1.5, 40),
            np.random.normal(baseline, 1.5, 40),
        )
        outcomes.extend(y)
        treatment.extend(t)
        strata.extend([s] * 40)

    return {
        "outcomes": np.array(outcomes),
        "treatment": np.array(treatment),
        "strata": np.array(strata),
        "true_ate": 4.0,
    }
