"""
Test fixtures for Shift-Share IV tests.

Provides DGP generators for shift-share estimation testing.
"""

from typing import Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray


def generate_shift_share_data(
    n: int = 200,
    n_sectors: int = 10,
    true_beta: float = 2.0,
    first_stage_strength: float = 1.5,
    share_concentration: float = 1.0,
    shock_variance: float = 0.05,
    n_controls: int = 0,
    random_state: int = 42,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    Optional[NDArray[np.float64]],
    float,
]:
    """
    Generate data for shift-share IV testing.

    DGP:
        shares ~ Dirichlet(alpha) where alpha = share_concentration
        shocks ~ N(0, shock_variance)
        Z_bartik = shares @ shocks
        D = first_stage_strength * Z_bartik + nu
        Y = true_beta * D + epsilon

    Parameters
    ----------
    n : int
        Number of observations (regions/units).
    n_sectors : int
        Number of sectors.
    true_beta : float
        True causal effect.
    first_stage_strength : float
        Coefficient on Bartik instrument in first stage.
    share_concentration : float
        Dirichlet concentration (higher = more uniform shares).
    shock_variance : float
        Variance of sector shocks.
    n_controls : int
        Number of exogenous controls.
    random_state : int
        Random seed.

    Returns
    -------
    Y : ndarray, shape (n,)
    D : ndarray, shape (n,)
    shares : ndarray, shape (n, n_sectors)
    shocks : ndarray, shape (n_sectors,)
    X : ndarray or None
    true_beta : float
    """
    rng = np.random.default_rng(random_state)

    # Sector shares (Dirichlet ensures they sum to 1)
    alpha = np.ones(n_sectors) * share_concentration
    shares = rng.dirichlet(alpha, n)

    # Aggregate shocks
    shocks = rng.normal(0, np.sqrt(shock_variance), n_sectors)

    # Bartik instrument
    Z_bartik = shares @ shocks

    # Controls
    if n_controls > 0:
        X = rng.normal(0, 1, (n, n_controls))
        gamma = 0.3  # Effect on D
        delta = 0.5  # Effect on Y
    else:
        X = None
        gamma = 0
        delta = 0

    # First-stage error
    nu = rng.normal(0, 0.5, n)

    # Treatment
    D = first_stage_strength * Z_bartik
    if X is not None:
        D += gamma * X.sum(axis=1)
    D += nu

    # Outcome error
    epsilon = rng.normal(0, 1, n)

    # Outcome
    Y = true_beta * D
    if X is not None:
        Y += delta * X.sum(axis=1)
    Y += epsilon

    return Y, D, shares, shocks, X, true_beta


@pytest.fixture
def ss_basic():
    """Basic shift-share data with strong first stage."""
    return generate_shift_share_data(
        n=200,
        n_sectors=10,
        true_beta=2.0,
        first_stage_strength=2.0,
        random_state=42,
    )


@pytest.fixture
def ss_with_controls():
    """Shift-share with exogenous controls."""
    return generate_shift_share_data(
        n=200,
        n_sectors=10,
        true_beta=1.5,
        first_stage_strength=2.0,
        n_controls=3,
        random_state=123,
    )


@pytest.fixture
def ss_many_sectors():
    """Many sectors (50)."""
    return generate_shift_share_data(
        n=300,
        n_sectors=50,
        true_beta=2.0,
        first_stage_strength=1.5,
        random_state=456,
    )


@pytest.fixture
def ss_concentrated_shares():
    """Concentrated shares (few dominant sectors)."""
    return generate_shift_share_data(
        n=200,
        n_sectors=10,
        true_beta=2.0,
        first_stage_strength=2.0,
        share_concentration=0.2,  # Low = concentrated
        random_state=789,
    )


@pytest.fixture
def ss_uniform_shares():
    """Uniform shares (all sectors equal)."""
    return generate_shift_share_data(
        n=200,
        n_sectors=10,
        true_beta=2.0,
        first_stage_strength=2.0,
        share_concentration=10.0,  # High = uniform
        random_state=111,
    )


@pytest.fixture
def ss_weak_first_stage():
    """Weak first-stage relationship."""
    return generate_shift_share_data(
        n=200,
        n_sectors=10,
        true_beta=2.0,
        first_stage_strength=0.3,  # Weak
        random_state=222,
    )


@pytest.fixture
def ss_large_sample():
    """Large sample for precision."""
    return generate_shift_share_data(
        n=2000,
        n_sectors=15,
        true_beta=1.0,
        first_stage_strength=2.0,
        random_state=333,
    )


@pytest.fixture
def ss_negative_effect():
    """Negative treatment effect."""
    return generate_shift_share_data(
        n=200,
        n_sectors=10,
        true_beta=-1.5,
        first_stage_strength=2.0,
        random_state=444,
    )
