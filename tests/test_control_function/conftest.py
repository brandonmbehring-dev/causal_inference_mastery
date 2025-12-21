"""
Test fixtures for Control Function estimation tests.

Provides known-answer test data for validating CF estimators:
- Just-identified CF (1 instrument, 1 endogenous)
- Over-identified CF (2 instruments, 1 endogenous)
- Strong instrument (F > 20)
- Weak instrument (F < 10) - for adversarial tests
- Endogenous vs exogenous treatment scenarios
- With and without controls

All fixtures return (Y, D, Z, X, true_beta, rho) where:
- Y: Outcome variable
- D: Endogenous treatment variable
- Z: Instrumental variable(s)
- X: Exogenous controls (can be None)
- true_beta: True causal effect
- rho: Endogeneity strength (correlation between nu and epsilon)

The control function approach:
- First stage: D = pi*Z + nu
- Second stage: Y = beta*D + rho*nu_hat + u
"""

from typing import Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray


def generate_cf_data(
    n: int = 1000,
    true_beta: float = 2.0,
    pi: float = 0.5,
    rho: float = 0.7,
    sigma_nu: float = 1.0,
    sigma_epsilon: float = 0.5,
    n_instruments: int = 1,
    n_controls: int = 0,
    random_state: int = 42,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    Optional[NDArray[np.float64]],
    float,
    float,
]:
    """
    Generate data for control function testing.

    DGP:
        Z ~ N(0, 1)  [instruments]
        X ~ N(0, 1)  [controls, if any]
        nu ~ N(0, sigma_nu^2)  [first-stage error]
        D = pi * Z + gamma * X + nu  [treatment]
        epsilon = rho * nu + sqrt(1 - rho^2) * N(0, sigma_epsilon^2)  [structural error]
        Y = beta * D + delta * X + epsilon  [outcome]

    The key endogeneity arises because nu enters both D and epsilon.
    When rho > 0, D is endogenous.
    When rho = 0, D is exogenous and OLS is consistent.

    Parameters
    ----------
    n : int
        Sample size.
    true_beta : float
        True causal effect of D on Y.
    pi : float
        First-stage coefficient on Z.
    rho : float
        Endogeneity parameter (correlation between nu and epsilon).
    sigma_nu : float
        Standard deviation of first-stage error.
    sigma_epsilon : float
        Additional noise in outcome.
    n_instruments : int
        Number of instruments.
    n_controls : int
        Number of exogenous controls.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Y : ndarray, shape (n,)
        Outcome variable.
    D : ndarray, shape (n,)
        Endogenous treatment variable.
    Z : ndarray, shape (n, n_instruments)
        Instrumental variable(s).
    X : ndarray or None
        Exogenous controls (None if n_controls=0).
    true_beta : float
        True causal effect.
    rho : float
        True endogeneity parameter.
    """
    rng = np.random.default_rng(random_state)

    # Instruments
    Z = rng.normal(0, 1, (n, n_instruments))

    # Controls
    if n_controls > 0:
        X = rng.normal(0, 1, (n, n_controls))
        gamma = 0.3  # Effect of X on D
        delta = 0.5  # Effect of X on Y
    else:
        X = None
        gamma = 0
        delta = 0

    # First-stage error
    nu = rng.normal(0, sigma_nu, n)

    # Treatment: D = pi * Z + gamma * X + nu
    D = pi * Z.sum(axis=1)
    if X is not None:
        D += gamma * X.sum(axis=1)
    D += nu

    # Structural error: epsilon = rho * nu + independent noise
    # This creates the endogeneity
    epsilon = rho * nu + np.sqrt(1 - rho**2) * rng.normal(0, sigma_epsilon, n)

    # Outcome: Y = beta * D + delta * X + epsilon
    Y = true_beta * D
    if X is not None:
        Y += delta * X.sum(axis=1)
    Y += epsilon

    return Y, D, Z, X, true_beta, rho


@pytest.fixture
def cf_endogenous():
    """
    Strong endogeneity case: rho = 0.7.

    Data Generating Process:
    - D is correlated with structural error (rho = 0.7)
    - OLS would be biased upward
    - CF should recover true_beta = 2.0

    Sample size: n = 1,000
    First-stage F-statistic: ~250 (strong instrument)
    True effect: beta = 2.0
    Endogeneity: rho = 0.7 (significant)

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_cf_data(
        n=1000,
        true_beta=2.0,
        pi=0.5,
        rho=0.7,
        sigma_nu=1.0,
        sigma_epsilon=0.3,
        random_state=42,
    )


@pytest.fixture
def cf_exogenous():
    """
    No endogeneity case: rho = 0.

    Data Generating Process:
    - D is uncorrelated with structural error (rho = 0)
    - OLS would be consistent
    - CF control coefficient should be insignificant

    Sample size: n = 1,000
    First-stage F-statistic: ~250 (strong instrument)
    True effect: beta = 1.5
    Endogeneity: rho = 0 (none)

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_cf_data(
        n=1000,
        true_beta=1.5,
        pi=0.5,
        rho=0.0,
        sigma_nu=1.0,
        sigma_epsilon=0.5,
        random_state=123,
    )


@pytest.fixture
def cf_with_controls():
    """
    Endogenous treatment with exogenous controls.

    Data Generating Process:
    - 2 exogenous controls included
    - Controls affect both D and Y
    - Endogeneity present (rho = 0.5)

    Sample size: n = 1,000
    True effect: beta = 3.0
    Endogeneity: rho = 0.5

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_cf_data(
        n=1000,
        true_beta=3.0,
        pi=0.6,
        rho=0.5,
        sigma_nu=1.0,
        sigma_epsilon=0.4,
        n_controls=2,
        random_state=456,
    )


@pytest.fixture
def cf_over_identified():
    """
    Over-identified: 2 instruments for 1 endogenous variable.

    Data Generating Process:
    - 2 instruments predict treatment
    - Standard endogeneity (rho = 0.6)

    Sample size: n = 1,000
    True effect: beta = 2.5
    Endogeneity: rho = 0.6

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_cf_data(
        n=1000,
        true_beta=2.5,
        pi=0.4,
        rho=0.6,
        sigma_nu=1.0,
        sigma_epsilon=0.4,
        n_instruments=2,
        random_state=789,
    )


@pytest.fixture
def cf_weak_instrument():
    """
    Weak instrument: F < 10 (Stock-Yogo threshold).

    Data Generating Process:
    - Weak first-stage relationship (pi = 0.1)
    - Should trigger weak IV warning

    Sample size: n = 500
    First-stage F-statistic: ~5 (weak)
    True effect: beta = 2.0
    Endogeneity: rho = 0.7

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_cf_data(
        n=500,
        true_beta=2.0,
        pi=0.1,
        rho=0.7,
        sigma_nu=1.0,
        sigma_epsilon=0.5,
        random_state=999,
    )


@pytest.fixture
def cf_large_sample():
    """
    Large sample for precision testing.

    Sample size: n = 10,000
    True effect: beta = 1.0
    Endogeneity: rho = 0.5

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_cf_data(
        n=10000,
        true_beta=1.0,
        pi=0.5,
        rho=0.5,
        sigma_nu=1.0,
        sigma_epsilon=0.3,
        random_state=111,
    )


@pytest.fixture
def cf_small_sample():
    """
    Small sample for robustness testing.

    Sample size: n = 50
    True effect: beta = 2.0
    Endogeneity: rho = 0.6

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_cf_data(
        n=50,
        true_beta=2.0,
        pi=0.8,
        rho=0.6,
        sigma_nu=1.0,
        sigma_epsilon=0.5,
        random_state=222,
    )


@pytest.fixture
def cf_mild_endogeneity():
    """
    Mild endogeneity: rho = 0.3.

    Borderline case where endogeneity test may not detect.

    Sample size: n = 1,000
    True effect: beta = 2.0
    Endogeneity: rho = 0.3 (mild)

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_cf_data(
        n=1000,
        true_beta=2.0,
        pi=0.5,
        rho=0.3,
        sigma_nu=1.0,
        sigma_epsilon=0.5,
        random_state=333,
    )


@pytest.fixture
def cf_strong_instrument():
    """
    Very strong instrument: F > 100.

    Data Generating Process:
    - Very strong first-stage relationship (pi = 1.0)
    - Large sample for high F

    Sample size: n = 2,000
    First-stage F-statistic: ~1000 (very strong)
    True effect: beta = 2.0
    Endogeneity: rho = 0.7

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_cf_data(
        n=2000,
        true_beta=2.0,
        pi=1.0,
        rho=0.7,
        sigma_nu=1.0,
        sigma_epsilon=0.3,
        random_state=444,
    )


@pytest.fixture
def cf_negative_effect():
    """
    Negative treatment effect case.

    Data Generating Process:
    - True effect is negative (beta = -1.5)
    - Tests handling of negative effects

    Sample size: n = 1,000
    True effect: beta = -1.5
    Endogeneity: rho = 0.5

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_cf_data(
        n=1000,
        true_beta=-1.5,
        pi=0.5,
        rho=0.5,
        sigma_nu=1.0,
        sigma_epsilon=0.4,
        random_state=555,
    )


# =============================================================================
# Nonlinear Control Function Fixtures (Binary Outcomes)
# =============================================================================


def generate_nonlinear_cf_data(
    n: int = 1000,
    true_beta: float = 1.0,
    pi: float = 0.5,
    rho: float = 0.5,
    sigma_nu: float = 1.0,
    model_type: str = "probit",
    n_instruments: int = 1,
    n_controls: int = 0,
    random_state: int = 42,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    Optional[NDArray[np.float64]],
    float,
    float,
]:
    """
    Generate data for nonlinear control function testing with binary outcome.

    DGP:
        Z ~ N(0, 1)  [instruments]
        X ~ N(0, 1)  [controls, if any]
        nu ~ N(0, sigma_nu^2)  [first-stage error]
        D = pi * Z + gamma * X + nu  [treatment]
        Y* = beta * D + rho * nu + delta * X + u  [latent outcome]
        Y = 1{Y* > 0}  [binary outcome]

    where u ~ Logistic(0, 1) for logit or u ~ N(0, 1) for probit.

    Parameters
    ----------
    n : int
        Sample size.
    true_beta : float
        True causal effect on latent scale.
    pi : float
        First-stage coefficient on Z.
    rho : float
        Endogeneity parameter.
    sigma_nu : float
        Standard deviation of first-stage error.
    model_type : {'probit', 'logit'}
        Latent error distribution.
    n_instruments : int
        Number of instruments.
    n_controls : int
        Number of exogenous controls.
    random_state : int
        Random seed.

    Returns
    -------
    Y : ndarray, shape (n,)
        Binary outcome (0 or 1).
    D : ndarray, shape (n,)
        Endogenous treatment variable.
    Z : ndarray, shape (n, n_instruments)
        Instrumental variable(s).
    X : ndarray or None
        Exogenous controls.
    true_beta : float
        True causal effect (latent scale).
    rho : float
        True endogeneity parameter.
    """
    rng = np.random.default_rng(random_state)

    # Instruments
    Z = rng.normal(0, 1, (n, n_instruments))

    # Controls
    if n_controls > 0:
        X = rng.normal(0, 1, (n, n_controls))
        gamma = 0.3
        delta = 0.3
    else:
        X = None
        gamma = 0
        delta = 0

    # First-stage error
    nu = rng.normal(0, sigma_nu, n)

    # Treatment: D = pi * Z + gamma * X + nu
    D = pi * Z.sum(axis=1)
    if X is not None:
        D += gamma * X.sum(axis=1)
    D += nu

    # Latent error (depends on model type)
    if model_type == "logit":
        u = rng.logistic(0, 1, n)
    else:  # probit
        u = rng.normal(0, 1, n)

    # Latent outcome: Y* = beta * D + rho * nu + delta * X + u
    Y_star = true_beta * D + rho * nu
    if X is not None:
        Y_star += delta * X.sum(axis=1)
    Y_star += u

    # Binary outcome
    Y = (Y_star > 0).astype(np.float64)

    return Y, D, Z, X, true_beta, rho


@pytest.fixture
def nonlinear_cf_probit():
    """
    Binary outcome with probit link and endogeneity.

    Sample size: n = 1,500
    True effect (latent): beta = 1.0
    Endogeneity: rho = 0.5

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_nonlinear_cf_data(
        n=1500,
        true_beta=1.0,
        pi=0.5,
        rho=0.5,
        sigma_nu=1.0,
        model_type="probit",
        random_state=42,
    )


@pytest.fixture
def nonlinear_cf_logit():
    """
    Binary outcome with logit link and endogeneity.

    Sample size: n = 1,500
    True effect (latent): beta = 0.8
    Endogeneity: rho = 0.4

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_nonlinear_cf_data(
        n=1500,
        true_beta=0.8,
        pi=0.5,
        rho=0.4,
        sigma_nu=1.0,
        model_type="logit",
        random_state=123,
    )


@pytest.fixture
def nonlinear_cf_no_endogeneity():
    """
    Binary outcome with no endogeneity (rho = 0).

    Control coefficient should be insignificant.

    Sample size: n = 1,500
    True effect (latent): beta = 0.8
    Endogeneity: rho = 0.0

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_nonlinear_cf_data(
        n=1500,
        true_beta=0.8,
        pi=0.5,
        rho=0.0,
        sigma_nu=1.0,
        model_type="probit",
        random_state=456,
    )


@pytest.fixture
def nonlinear_cf_with_controls():
    """
    Binary outcome with exogenous controls.

    Sample size: n = 1,500
    True effect (latent): beta = 0.6
    Endogeneity: rho = 0.5
    Controls: 2

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_nonlinear_cf_data(
        n=1500,
        true_beta=0.6,
        pi=0.5,
        rho=0.5,
        sigma_nu=1.0,
        model_type="probit",
        n_controls=2,
        random_state=789,
    )


@pytest.fixture
def nonlinear_cf_large_sample():
    """
    Large sample binary outcome for precision testing.

    Sample size: n = 5,000
    True effect (latent): beta = 0.5
    Endogeneity: rho = 0.5

    Returns
    -------
    Y, D, Z, X, true_beta, rho
    """
    return generate_nonlinear_cf_data(
        n=5000,
        true_beta=0.5,
        pi=0.5,
        rho=0.5,
        sigma_nu=1.0,
        model_type="probit",
        random_state=111,
    )
