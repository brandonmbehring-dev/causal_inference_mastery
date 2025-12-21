"""
Test fixtures for MTE module.

DGP generating known MTE patterns for testing.
"""

import pytest
import numpy as np
from typing import Tuple, Optional


@pytest.fixture
def rng():
    """Seeded RNG for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_binary_iv_data(rng) -> dict:
    """
    Simple DGP with binary instrument and constant treatment effect.

    Model:
        Z ~ Bernoulli(0.5)           (instrument)
        U ~ Uniform(0, 1)            (latent resistance)
        D = 1 if U < 0.3 + 0.4*Z     (selection)
        Y = 1 + 2*D + N(0, 0.5)      (outcome)

    True LATE = 2.0 (constant effect)
    Complier share = P(D=1|Z=1) - P(D=1|Z=0) = 0.7 - 0.3 = 0.4
    """
    n = 1000

    Z = rng.binomial(1, 0.5, size=n).astype(float)
    U = rng.uniform(0, 1, size=n)

    # Selection: D = 1 if U < threshold(Z)
    threshold = 0.3 + 0.4 * Z
    D = (U < threshold).astype(float)

    # Outcome with constant treatment effect
    Y = 1 + 2 * D + rng.normal(0, 0.5, size=n)

    return {
        "outcome": Y,
        "treatment": D,
        "instrument": Z,
        "true_late": 2.0,
        "true_complier_share": 0.4,
        "true_always_taker": 0.3,
        "true_never_taker": 0.3,
        "n": n,
    }


@pytest.fixture
def heterogeneous_mte_data(rng) -> dict:
    """
    DGP with heterogeneous treatment effects indexed by U.

    Model:
        Z ~ Normal(0, 1)             (continuous instrument)
        U ~ Uniform(0, 1)            (latent resistance)
        P(Z) = Logistic(Z)           (propensity)
        D = 1 if U < P(Z)            (selection)
        MTE(U) = 3 - 2*U             (linear MTE: 3 at U=0, 1 at U=1)
        Y = 1 + MTE(U)*D + N(0, 0.5) (outcome)

    True:
        MTE(u) = 3 - 2*u (linear, decreasing)
        ATE = ∫ MTE(u) du = 3 - 1 = 2
        LATE ≈ 2 (depends on complier distribution)
    """
    n = 2000

    Z = rng.normal(0, 1, size=n)
    U = rng.uniform(0, 1, size=n)

    # Propensity via logistic function
    propensity = 1 / (1 + np.exp(-Z))

    # Selection
    D = (U < propensity).astype(float)

    # Heterogeneous effect: MTE(U) = 3 - 2*U
    # Those with low U (easy to treat) have high returns
    # Those with high U (hard to treat) have low returns
    individual_effect = 3 - 2 * U

    Y = 1 + individual_effect * D + rng.normal(0, 0.5, size=n)

    return {
        "outcome": Y,
        "treatment": D,
        "instrument": Z,
        "propensity": propensity,
        "U": U,
        "true_mte": lambda u: 3 - 2 * u,  # MTE function
        "true_ate": 2.0,  # ∫(3-2u)du from 0 to 1 = 2
        "n": n,
    }


@pytest.fixture
def constant_mte_data(rng) -> dict:
    """
    DGP with constant MTE (no heterogeneity).

    When MTE is constant, LATE = ATE = ATT = ATU.
    """
    n = 1000

    Z = rng.normal(0, 1, size=n)
    U = rng.uniform(0, 1, size=n)

    propensity = 1 / (1 + np.exp(-Z))
    D = (U < propensity).astype(float)

    # Constant effect: MTE(U) = 1.5 for all U
    Y = 0.5 + 1.5 * D + rng.normal(0, 0.3, size=n)

    return {
        "outcome": Y,
        "treatment": D,
        "instrument": Z,
        "propensity": propensity,
        "true_mte": lambda u: np.full_like(u, 1.5),
        "true_ate": 1.5,
        "true_late": 1.5,
        "n": n,
    }


@pytest.fixture
def quadratic_mte_data(rng) -> dict:
    """
    DGP with quadratic MTE pattern.

    MTE(u) = 4 - 6*u + 3*u^2
    """
    n = 2000

    Z = rng.normal(0, 1, size=n)
    U = rng.uniform(0, 1, size=n)

    propensity = 1 / (1 + np.exp(-Z))
    D = (U < propensity).astype(float)

    # Quadratic MTE
    def mte_func(u):
        return 4 - 6 * u + 3 * u**2

    individual_effect = mte_func(U)
    Y = 1 + individual_effect * D + rng.normal(0, 0.5, size=n)

    # ATE = ∫(4 - 6u + 3u²)du = 4 - 3 + 1 = 2
    return {
        "outcome": Y,
        "treatment": D,
        "instrument": Z,
        "propensity": propensity,
        "true_mte": mte_func,
        "true_ate": 2.0,
        "n": n,
    }


@pytest.fixture
def weak_instrument_data(rng) -> dict:
    """
    DGP with weak first stage.

    Instrument has small effect on treatment probability.
    """
    n = 1000

    Z = rng.binomial(1, 0.5, size=n).astype(float)
    U = rng.uniform(0, 1, size=n)

    # Weak first stage: small effect of Z on D
    threshold = 0.4 + 0.05 * Z  # Only 5% difference
    D = (U < threshold).astype(float)

    Y = 1 + 2 * D + rng.normal(0, 0.5, size=n)

    return {
        "outcome": Y,
        "treatment": D,
        "instrument": Z,
        "true_late": 2.0,
        "true_complier_share": 0.05,  # Very few compliers
        "n": n,
    }


@pytest.fixture
def no_first_stage_data(rng) -> dict:
    """
    DGP where instrument has no effect on treatment.

    This should trigger weak instrument warnings.
    """
    n = 500

    Z = rng.binomial(1, 0.5, size=n).astype(float)
    U = rng.uniform(0, 1, size=n)

    # No first stage: Z doesn't affect D
    D = (U < 0.4).astype(float)

    Y = 1 + 2 * D + rng.normal(0, 0.5, size=n)

    return {
        "outcome": Y,
        "treatment": D,
        "instrument": Z,
        "true_late": np.nan,  # Undefined when no first stage
        "true_complier_share": 0.0,
        "n": n,
    }


@pytest.fixture
def multivariate_instrument_data(rng) -> dict:
    """
    DGP with multiple instruments.
    """
    n = 1500

    Z1 = rng.normal(0, 1, size=n)
    Z2 = rng.normal(0, 1, size=n)
    Z = np.column_stack([Z1, Z2])

    U = rng.uniform(0, 1, size=n)

    # Propensity depends on both instruments
    propensity = 1 / (1 + np.exp(-(0.5 * Z1 + 0.3 * Z2)))
    D = (U < propensity).astype(float)

    individual_effect = 2 - U
    Y = 1 + individual_effect * D + rng.normal(0, 0.5, size=n)

    return {
        "outcome": Y,
        "treatment": D,
        "instrument": Z,
        "propensity": propensity,
        "true_mte": lambda u: 2 - u,
        "true_ate": 1.5,  # ∫(2-u)du = 1.5
        "n": n,
    }


@pytest.fixture
def covariate_data(rng) -> dict:
    """
    DGP with covariates that affect both treatment and outcome.
    """
    n = 1000

    # Covariate
    X = rng.normal(0, 1, size=n)

    # Instrument (independent of X)
    Z = rng.normal(0, 1, size=n)

    U = rng.uniform(0, 1, size=n)

    # Propensity depends on Z and X
    propensity = 1 / (1 + np.exp(-(0.5 * Z + 0.3 * X)))
    D = (U < propensity).astype(float)

    # Outcome depends on X and treatment
    # Treatment effect is 2.0, covariate effect is 0.5
    Y = 1 + 2 * D + 0.5 * X + rng.normal(0, 0.5, size=n)

    return {
        "outcome": Y,
        "treatment": D,
        "instrument": Z,
        "covariates": X.reshape(-1, 1),
        "propensity": propensity,
        "true_late": 2.0,
        "n": n,
    }


@pytest.fixture
def defier_data(rng) -> dict:
    """
    DGP with defiers (monotonicity violation).

    Some units decrease treatment when instrument increases.
    """
    n = 1000

    Z = rng.binomial(1, 0.5, size=n).astype(float)
    U = rng.uniform(0, 1, size=n)

    # Create some defiers
    # Compliers: D increases with Z
    # Defiers: D decreases with Z
    is_defier = U > 0.9  # 10% defiers

    D = np.zeros(n)
    # Non-defiers: threshold increases with Z
    D[~is_defier] = (U[~is_defier] < 0.3 + 0.4 * Z[~is_defier]).astype(float)
    # Defiers: threshold decreases with Z
    D[is_defier] = (U[is_defier] < 0.7 - 0.4 * Z[is_defier]).astype(float)

    Y = 1 + 2 * D + rng.normal(0, 0.5, size=n)

    return {
        "outcome": Y,
        "treatment": D,
        "instrument": Z,
        "has_defiers": True,
        "defier_share": 0.1,
        "n": n,
    }


# --- Helper functions for test DGPs ---


def generate_mte_dgp(
    n: int = 1000,
    mte_func=None,
    propensity_spread: float = 1.0,
    noise_sd: float = 0.5,
    rng=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate MTE DGP with customizable MTE function.

    Parameters
    ----------
    n : int
        Sample size
    mte_func : callable, optional
        Function MTE(u) -> effect. Default is linear decreasing.
    propensity_spread : float
        How spread out propensity scores are (higher = more variation)
    noise_sd : float
        Outcome noise standard deviation
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    Y, D, Z, propensity
    """
    if rng is None:
        rng = np.random.default_rng()

    if mte_func is None:
        mte_func = lambda u: 3 - 2 * u

    Z = rng.normal(0, propensity_spread, size=n)
    U = rng.uniform(0, 1, size=n)

    propensity = 1 / (1 + np.exp(-Z))
    D = (U < propensity).astype(float)

    individual_effect = mte_func(U)
    Y = 1 + individual_effect * D + rng.normal(0, noise_sd, size=n)

    return Y, D, Z, propensity
