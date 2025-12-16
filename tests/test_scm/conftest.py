"""
Test fixtures for Synthetic Control Methods.

Provides panel data generators with known treatment effects for testing.
"""

import numpy as np
import pytest


@pytest.fixture
def simple_panel():
    """
    Simple 3-unit panel with known effect.

    Setup:
    - 3 units: 1 treated, 2 controls (minimum for SCM validation)
    - 10 periods: 5 pre, 5 post
    - Treated and control_0 identical pre-treatment
    - control_1 adds small noise to break degeneracy
    - Treatment adds 2.0 to post-treatment outcomes
    - Expected ATT = 2.0, weights ≈ [1.0, 0.0]
    """
    np.random.seed(42)
    n_periods = 10
    treatment_period = 5

    # Base series
    base = np.cumsum(np.random.randn(n_periods) * 0.5) + 10.0

    # Treated = base + effect after treatment
    treated = base.copy()
    treated[treatment_period:] += 2.0

    # Control_0 = base (perfect match)
    control_0 = base.copy()

    # Control_1 = base + offset (should get weight ≈ 0)
    control_1 = base.copy() + 5.0

    outcomes = np.vstack([treated, control_0, control_1])
    treatment = np.array([1, 0, 0])

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "treatment_period": treatment_period,
        "true_effect": 2.0,
        "expected_weights": np.array([1.0, 0.0]),
    }


@pytest.fixture
def multi_control_panel():
    """
    Panel with 1 treated, 5 controls.

    Setup:
    - 6 units: 1 treated, 5 controls
    - 20 periods: 10 pre, 10 post
    - Treated is 50% control_0 + 50% control_1 pre-treatment
    - Treatment adds 3.0 to post-treatment
    - Expected weights close to [0.5, 0.5, 0, 0, 0]
    """
    np.random.seed(123)
    n_periods = 20
    treatment_period = 10
    n_controls = 5

    # Control series
    controls = np.zeros((n_controls, n_periods))
    for i in range(n_controls):
        trend = np.linspace(0, i * 2, n_periods)
        noise = np.random.randn(n_periods) * 0.3
        controls[i, :] = 10 + trend + noise

    # Treated = 0.5 * control_0 + 0.5 * control_1 + effect after treatment
    treated = 0.5 * controls[0, :] + 0.5 * controls[1, :] + np.random.randn(n_periods) * 0.1
    treated[treatment_period:] += 3.0

    outcomes = np.vstack([treated.reshape(1, -1), controls])
    treatment = np.array([1, 0, 0, 0, 0, 0])

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "treatment_period": treatment_period,
        "true_effect": 3.0,
        "n_controls": n_controls,
    }


@pytest.fixture
def balanced_panel():
    """
    Balanced panel with 1 treated, 9 controls.

    Used for placebo inference validation.
    """
    np.random.seed(456)
    n_units = 10
    n_periods = 15
    treatment_period = 8

    # Common trend
    common_trend = np.linspace(0, 5, n_periods)

    outcomes = np.zeros((n_units, n_periods))
    for i in range(n_units):
        unit_effect = np.random.randn() * 2
        noise = np.random.randn(n_periods) * 0.5
        outcomes[i, :] = 10 + common_trend + unit_effect + noise

    # Add treatment effect to unit 0
    true_effect = 2.5
    outcomes[0, treatment_period:] += true_effect

    treatment = np.zeros(n_units)
    treatment[0] = 1

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "treatment_period": treatment_period,
        "true_effect": true_effect,
    }


@pytest.fixture
def panel_with_covariates():
    """
    Panel with pre-treatment covariates.
    """
    np.random.seed(789)
    n_units = 8
    n_periods = 12
    treatment_period = 6
    n_covariates = 3

    # Covariates
    covariates = np.random.randn(n_units, n_covariates)

    # Outcomes depend on covariates
    outcomes = np.zeros((n_units, n_periods))
    for i in range(n_units):
        base = 10 + covariates[i, 0] * 2 + covariates[i, 1]
        trend = np.linspace(0, 3, n_periods)
        noise = np.random.randn(n_periods) * 0.3
        outcomes[i, :] = base + trend + noise

    # Treatment effect
    true_effect = 1.5
    outcomes[0, treatment_period:] += true_effect

    treatment = np.zeros(n_units)
    treatment[0] = 1

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "treatment_period": treatment_period,
        "covariates": covariates,
        "true_effect": true_effect,
    }


@pytest.fixture
def no_effect_panel():
    """
    Panel with zero treatment effect (for null hypothesis testing).
    """
    np.random.seed(999)
    n_units = 6
    n_periods = 14
    treatment_period = 7

    outcomes = np.zeros((n_units, n_periods))
    for i in range(n_units):
        trend = np.linspace(0, 4, n_periods)
        noise = np.random.randn(n_periods) * 0.5
        outcomes[i, :] = 10 + trend + noise

    treatment = np.zeros(n_units)
    treatment[0] = 1

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "treatment_period": treatment_period,
        "true_effect": 0.0,
    }
