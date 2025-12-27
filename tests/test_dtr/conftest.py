"""Test fixtures and DGP generators for DTR module.

Provides data generating processes with known optimal treatment regimes
for validating Q-learning and other DTR methods.
"""

import numpy as np
import pytest
from typing import Optional

from causal_inference.dtr import DTRData


def generate_dtr_dgp(
    n: int = 500,
    n_stages: int = 1,
    n_covariates: int = 3,
    true_blip: float = 2.0,
    propensity: float = 0.5,
    noise_sd: float = 1.0,
    seed: Optional[int] = 42,
) -> tuple[DTRData, dict]:
    """Generate DTR test data with known optimal regime.

    Data Generating Process
    -----------------------
    Single-stage (K=1):
        X ~ N(0, I_p)
        A ~ Bernoulli(propensity)
        Y = X[:, 0] + true_blip * A + eps, eps ~ N(0, noise_sd^2)

        Optimal regime: d*(X) = 1 if true_blip > 0 else 0 (constant)

    Two-stage (K=2):
        Stage 1:
            X_1 ~ N(0, I_p)
            A_1 ~ Bernoulli(propensity)
            Y_1 = X_1[:, 0] + blip_1 * A_1 + eps_1

        Stage 2:
            H_2 = (X_1, A_1, Y_1, X_2) where X_2 ~ N(0, I_p)
            A_2 ~ Bernoulli(propensity)
            Y_2 = Y_1 + X_2[:, 0] + blip_2 * A_2 + eps_2

        Optimal regime: d*_k(H_k) = 1 if blip_k > 0

    Parameters
    ----------
    n : int
        Number of observations.
    n_stages : int
        Number of decision stages (1, 2, or more).
    n_covariates : int
        Number of covariates at each stage.
    true_blip : float
        True treatment effect (constant across stages).
    propensity : float
        Treatment probability.
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    tuple[DTRData, dict]
        (data, true_params) where true_params includes:
        - "true_blip": the true blip value
        - "optimal_value": expected outcome under optimal regime
        - "optimal_regime": callable d*(X) -> {0, 1}
    """
    if seed is not None:
        np.random.seed(seed)

    outcomes = []
    treatments = []
    covariates = []

    # Track cumulative outcome for multi-stage
    cumulative_Y = np.zeros(n)

    for k in range(n_stages):
        # Generate covariates for this stage
        X_k = np.random.randn(n, n_covariates)

        # Generate treatment
        A_k = np.random.binomial(1, propensity, n).astype(float)

        # Generate outcome
        # Y_k = baseline + blip * A_k + noise
        # baseline includes prior Y for k > 0
        baseline = X_k[:, 0]
        if k > 0:
            baseline = baseline + cumulative_Y

        Y_k = baseline + true_blip * A_k + noise_sd * np.random.randn(n)

        # Update cumulative outcome
        cumulative_Y = Y_k

        # Store
        outcomes.append(Y_k)
        treatments.append(A_k)

        # For multi-stage, history includes prior A and Y
        if k == 0:
            covariates.append(X_k)
        else:
            # H_k = (X_1, A_1, Y_1, ..., X_k)
            # But DTRData.get_history handles this, so we just store X_k
            # Actually, let's store the augmented history
            prior_parts = []
            for j in range(k):
                prior_parts.append(covariates[j])
                prior_parts.append(treatments[j].reshape(-1, 1))
                prior_parts.append(outcomes[j].reshape(-1, 1))
            prior_parts.append(X_k)
            H_k = np.hstack(prior_parts)
            covariates.append(H_k)

    data = DTRData(outcomes=outcomes, treatments=treatments, covariates=covariates)

    # Compute optimal value analytically
    # Under optimal regime, everyone gets A=1 if blip > 0
    optimal_A = 1 if true_blip > 0 else 0

    # E[Y^{d*}] for single stage: E[X[:, 0]] + blip * optimal_A = 0 + blip * optimal_A
    optimal_value = n_stages * true_blip * optimal_A  # Simplified for constant blip

    def optimal_regime(X: np.ndarray, stage: int = 1) -> int:
        """Optimal treatment rule (constant for this DGP)."""
        return int(true_blip > 0)

    true_params = {
        "true_blip": true_blip,
        "optimal_value": optimal_value,
        "optimal_regime": optimal_regime,
        "propensity": propensity,
        "noise_sd": noise_sd,
    }

    return data, true_params


def generate_heterogeneous_dtr_dgp(
    n: int = 500,
    n_covariates: int = 3,
    seed: Optional[int] = 42,
) -> tuple[DTRData, dict]:
    """Generate DTR data with heterogeneous treatment effects.

    The blip depends on covariates:
        gamma(X) = 2.0 * X[:, 0]

    So optimal regime is:
        d*(X) = 1 if X[:, 0] > 0 else 0

    Parameters
    ----------
    n : int
        Number of observations.
    n_covariates : int
        Number of covariates.
    seed : int or None
        Random seed.

    Returns
    -------
    tuple[DTRData, dict]
        Data and true parameters.
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(n, n_covariates)
    A = np.random.binomial(1, 0.5, n).astype(float)

    # Heterogeneous blip: gamma(X) = 2 * X[:, 0]
    true_blip_values = 2.0 * X[:, 0]
    Y = X[:, 1] + A * true_blip_values + np.random.randn(n)

    data = DTRData(outcomes=[Y], treatments=[A], covariates=[X])

    def optimal_regime(X_new: np.ndarray, stage: int = 1) -> int:
        """Optimal treatment: treat if X[:, 0] > 0."""
        X_new = np.atleast_1d(X_new)
        return int(X_new[0] > 0)

    true_params = {
        "blip_function": lambda x: 2.0 * x[0],
        "optimal_regime": optimal_regime,
    }

    return data, true_params


# Pytest fixtures
@pytest.fixture
def single_stage_constant_data():
    """Single-stage data with constant treatment effect."""
    return generate_dtr_dgp(n=500, n_stages=1, true_blip=2.0, seed=42)


@pytest.fixture
def single_stage_zero_effect_data():
    """Single-stage data with zero treatment effect."""
    return generate_dtr_dgp(n=500, n_stages=1, true_blip=0.0, seed=42)


@pytest.fixture
def two_stage_data():
    """Two-stage DTR data."""
    return generate_dtr_dgp(n=500, n_stages=2, true_blip=2.0, seed=42)


@pytest.fixture
def heterogeneous_data():
    """Data with heterogeneous treatment effects."""
    return generate_heterogeneous_dtr_dgp(n=500, seed=42)


@pytest.fixture
def small_sample_data():
    """Small sample for edge case testing."""
    return generate_dtr_dgp(n=40, n_stages=1, true_blip=2.0, seed=42)


@pytest.fixture
def high_dimensional_data():
    """High-dimensional covariates."""
    return generate_dtr_dgp(n=500, n_stages=1, n_covariates=50, true_blip=2.0, seed=42)
