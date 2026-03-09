"""Shared fixtures for validation tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Fixed random number generator for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def basic_rct_data():
    """
    Basic RCT dataset for cross-validation.

    True ATE = 2.0
    n = 100, balanced
    """
    np.random.seed(42)
    n = 100
    treatment = np.array([1] * 50 + [0] * 50)
    outcomes = np.where(
        treatment == 1,
        np.random.normal(2.0, 1.0, n),  # Treated: mean=2
        np.random.normal(0.0, 1.0, n),
    )  # Control: mean=0
    return outcomes, treatment


@pytest.fixture
def stratified_rct_data():
    """
    Stratified RCT dataset for cross-validation.

    True ATE = 2.0 (same in all strata)
    3 strata with different baselines
    """
    np.random.seed(42)
    outcomes = []
    treatment = []
    strata = []

    for stratum_id, baseline in enumerate([0, 5, 10]):  # 3 strata
        n_stratum = 40
        t = np.array([1] * 20 + [0] * 20)
        y = np.where(
            t == 1,
            np.random.normal(baseline + 2.0, 1.0, n_stratum),  # ATE=2
            np.random.normal(baseline, 1.0, n_stratum),
        )
        outcomes.extend(y)
        treatment.extend(t)
        strata.extend([stratum_id] * n_stratum)

    return np.array(outcomes), np.array(treatment), np.array(strata)


@pytest.fixture
def regression_rct_data():
    """
    RCT with covariates for regression adjustment.

    True ATE = 2.0 after controlling for X
    """
    np.random.seed(42)
    n = 100
    X = np.random.normal(0, 1, n)  # Covariate
    treatment = np.array([1] * 50 + [0] * 50)

    # Y = 2*T + 3*X + noise (ATE = 2)
    outcomes = 2.0 * treatment + 3.0 * X + np.random.normal(0, 1, n)

    return outcomes, treatment, X


@pytest.fixture
def small_sample_rct_data():
    """
    Small sample RCT (n=20) to test t-distribution.

    True ATE = 2.0
    """
    np.random.seed(42)
    n = 20
    treatment = np.array([1] * 10 + [0] * 10)
    outcomes = np.where(
        treatment == 1, np.random.normal(2.0, 1.0, n), np.random.normal(0.0, 1.0, n)
    )
    return outcomes, treatment


@pytest.fixture
def ipw_rct_data():
    """
    RCT with known propensity scores for IPW.

    True ATE = 2.0
    Propensity varies by covariate
    """
    np.random.seed(42)
    n = 100
    X = np.random.normal(0, 1, n)

    # Propensity depends on X
    propensity = 1 / (1 + np.exp(-0.5 * X))  # Logistic
    treatment = (np.random.uniform(0, 1, n) < propensity).astype(float)

    # Outcomes: ATE = 2.0
    outcomes = 2.0 * treatment + X + np.random.normal(0, 1, n)

    return outcomes, treatment, propensity


@pytest.fixture
def validation_tolerance():
    """
    Tolerance levels for cross-validation.

    Returns dict with different tolerance levels for different checks.
    """
    return {
        "rtol_cross_language": 1e-10,  # Python vs Julia: near machine precision
        "bias_monte_carlo": 0.05,  # Monte Carlo bias threshold
        "coverage_lower": 0.93,  # Coverage lower bound (93% - accounts for MC variation)
        "coverage_upper": 0.97,  # Coverage upper bound (97% - accounts for MC variation)
        "se_accuracy": 0.10,  # SE accuracy (within 10%)
    }
