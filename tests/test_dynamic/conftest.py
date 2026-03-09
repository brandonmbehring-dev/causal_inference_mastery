"""Shared fixtures for Dynamic DML tests.

Provides DGP generators, known-answer data, and panel data fixtures.
"""

import numpy as np
import pytest


@pytest.fixture
def simple_dgp():
    """Simple DGP with known contemporaneous effect only."""
    np.random.seed(42)
    n = 300

    # Exogenous covariates
    X = np.random.randn(n, 3)

    # Independent treatment (no confounding)
    D = np.random.binomial(1, 0.5, n).astype(float)

    # Outcome: only contemporaneous effect (theta_0 = 2.0)
    Y = 2.0 * D + X @ [1.0, 0.5, 0.2] + np.random.randn(n)

    return Y, D, X, np.array([2.0])


@pytest.fixture
def lagged_dgp():
    """DGP with contemporaneous and lagged effects."""
    np.random.seed(123)
    n = 500

    # Autocorrelated covariates
    X = np.zeros((n, 3))
    X[0] = np.random.randn(3)
    for t in range(1, n):
        X[t] = 0.3 * X[t - 1] + np.sqrt(0.91) * np.random.randn(3)

    # Treatment with mild confounding
    propensity = 0.5 + 0.2 * X[:, 0]
    propensity = np.clip(propensity, 0.2, 0.8)
    D = (np.random.rand(n) < propensity).astype(float)

    # True effects: theta = [2.0, 1.0, 0.5]
    true_effects = np.array([2.0, 1.0, 0.5])

    Y = np.zeros(n)
    for t in range(n):
        Y[t] = X[t] @ [1.0, 0.5, 0.2]
        for h, theta_h in enumerate(true_effects):
            if t >= h:
                Y[t] += theta_h * D[t - h]
        Y[t] += np.random.randn()

    return Y, D, X, true_effects


@pytest.fixture
def panel_dgp():
    """Panel data DGP with multiple units."""
    np.random.seed(456)
    n_units = 50
    n_periods = 20
    n_obs = n_units * n_periods

    # Unit and time IDs
    unit_id = np.repeat(np.arange(n_units), n_periods)
    time_id = np.tile(np.arange(n_periods), n_units)

    # Unit fixed effects
    unit_effects = np.random.randn(n_units)

    # Time-varying covariates
    X = np.random.randn(n_obs, 3)

    # Treatment (varies by unit propensity)
    unit_propensity = 0.5 + 0.3 * np.random.randn(n_units)
    unit_propensity = np.clip(unit_propensity, 0.2, 0.8)
    D = np.zeros(n_obs)
    for i, u in enumerate(np.arange(n_units)):
        mask = unit_id == u
        D[mask] = np.random.binomial(1, unit_propensity[i], n_periods)

    # True effects
    true_effects = np.array([1.5, 0.8])

    # Outcome with unit FE and dynamic effects
    Y = np.zeros(n_obs)
    for obs in range(n_obs):
        u = unit_id[obs]
        t = time_id[obs]
        Y[obs] = unit_effects[u] + X[obs] @ [0.5, 0.3, 0.2]

        # Add lagged treatment effects (within unit)
        unit_mask = unit_id == u
        unit_D = D[unit_mask]
        for h, theta_h in enumerate(true_effects):
            if t >= h:
                Y[obs] += theta_h * unit_D[t - h]

        Y[obs] += 0.5 * np.random.randn()

    return Y, D, X, unit_id, true_effects


@pytest.fixture
def zero_effect_dgp():
    """DGP with zero treatment effect."""
    np.random.seed(789)
    n = 300

    X = np.random.randn(n, 3)
    D = np.random.binomial(1, 0.5, n).astype(float)

    # No treatment effect
    Y = X @ [1.0, 0.5, 0.2] + np.random.randn(n)

    return Y, D, X, np.array([0.0, 0.0])


@pytest.fixture
def adversarial_confounded_dgp():
    """Strongly confounded DGP (adversarial test)."""
    np.random.seed(321)
    n = 500

    # Confounder
    U = np.random.randn(n)

    # Observed covariates
    X = np.column_stack(
        [
            U + 0.5 * np.random.randn(n),  # Proxies for confounder
            np.random.randn(n),
            np.random.randn(n),
        ]
    )

    # Treatment strongly depends on confounder
    propensity = 1 / (1 + np.exp(-0.8 * U))
    D = (np.random.rand(n) < propensity).astype(float)

    # Outcome also depends on confounder
    true_effect = 2.0
    Y = true_effect * D + 1.5 * U + X[:, 1] * 0.3 + np.random.randn(n)

    return Y, D, X, np.array([true_effect])


@pytest.fixture
def sparse_treatment_dgp():
    """DGP with sparse treatment (low treatment probability)."""
    np.random.seed(654)
    n = 500

    X = np.random.randn(n, 3)

    # Only 10% treated
    D = np.random.binomial(1, 0.1, n).astype(float)

    Y = 2.0 * D + X @ [1.0, 0.5, 0.2] + np.random.randn(n)

    return Y, D, X, np.array([2.0])


@pytest.fixture
def autocorrelated_dgp():
    """DGP with heavily autocorrelated errors."""
    np.random.seed(987)
    n = 500

    X = np.random.randn(n, 3)
    D = np.random.binomial(1, 0.5, n).astype(float)

    # AR(1) errors with rho=0.8
    epsilon = np.zeros(n)
    epsilon[0] = np.random.randn()
    rho = 0.8
    for t in range(1, n):
        epsilon[t] = rho * epsilon[t - 1] + np.sqrt(1 - rho**2) * np.random.randn()

    Y = 2.0 * D + X @ [1.0, 0.5, 0.2] + epsilon

    return Y, D, X, np.array([2.0])
