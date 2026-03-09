"""
Shared fixtures for Heckman selection model tests.

Provides DGPs with known parameters for validation:
1. Simple selection with known λ
2. Strong selection (high |ρ|)
3. No selection (ρ = 0)
4. With exclusion restriction
5. Without exclusion restriction (fragile identification)
"""

import numpy as np
import pytest
from scipy import stats


def generate_heckman_dgp(
    n: int = 500,
    rho: float = 0.5,
    beta_x: float = 2.0,
    gamma_z: float = 1.0,
    sigma_u: float = 1.0,
    has_exclusion: bool = True,
    seed: int = 42,
) -> dict:
    """
    Generate data from Heckman selection model DGP.

    Model:
        Selection: S* = γ₀ + γ_z * Z + γ_x * X + v
                   S = 1{S* > 0}
        Outcome:   Y = β₀ + β_x * X + u  (observed only when S = 1)

        where (u, v) ~ Bivariate Normal with Corr(u, v) = ρ

    Parameters
    ----------
    n : int
        Sample size.
    rho : float
        Correlation between selection and outcome errors.
        rho = 0 means no selection bias.
    beta_x : float
        True coefficient on X in outcome equation.
    gamma_z : float
        Coefficient on exclusion restriction Z in selection equation.
    sigma_u : float
        Standard deviation of outcome error.
    has_exclusion : bool
        If True, Z affects selection but not outcome (valid exclusion).
        If False, Z also affects outcome (no exclusion restriction).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with:
        - outcome: Full outcome array (NaN for unselected)
        - outcome_selected: Outcomes for selected only
        - selected: Selection indicator
        - selection_covariates: Z matrix for selection equation
        - outcome_covariates: X matrix for outcome equation
        - true_beta: True coefficient on X
        - true_rho: True selection correlation
        - true_lambda_coef: True λ = ρ * σ
        - n_selected: Number of selected observations
        - n_total: Total observations
    """
    np.random.seed(seed)

    # Covariates
    X = np.random.normal(0, 1, n)  # Affects both selection and outcome
    Z = np.random.normal(0, 1, n)  # Exclusion restriction (affects selection only)

    # Generate correlated errors (u, v)
    # Covariance matrix: [[σ²_u, ρ*σ_u], [ρ*σ_u, 1]]
    cov_matrix = np.array(
        [
            [sigma_u**2, rho * sigma_u],
            [rho * sigma_u, 1.0],
        ]
    )
    errors = np.random.multivariate_normal([0, 0], cov_matrix, n)
    u = errors[:, 0]  # Outcome error
    v = errors[:, 1]  # Selection error

    # Selection equation: S* = 0.5 + γ_z * Z + 0.3 * X + v
    # (coefficients chosen to get reasonable selection rate)
    gamma_0 = 0.5
    gamma_x = 0.3
    s_star = gamma_0 + gamma_z * Z + gamma_x * X + v
    selected = (s_star > 0).astype(float)

    # Outcome equation: Y = 1.0 + β_x * X + u
    beta_0 = 1.0
    if has_exclusion:
        # Z does NOT affect outcome (valid exclusion)
        outcome_latent = beta_0 + beta_x * X + u
    else:
        # Z affects outcome (no exclusion - fragile identification)
        outcome_latent = beta_0 + beta_x * X + 0.5 * Z + u

    # Observed outcome (NaN for unselected)
    outcome = np.where(selected == 1, outcome_latent, np.nan)
    outcome_selected = outcome_latent[selected == 1]

    # Selection covariates (include intercept in model, not here)
    selection_covariates = np.column_stack([X, Z])

    # Outcome covariates
    outcome_covariates = X.reshape(-1, 1)

    # True lambda coefficient
    true_lambda_coef = rho * sigma_u

    return {
        "outcome": outcome,
        "outcome_selected": outcome_selected,
        "selected": selected,
        "selection_covariates": selection_covariates,
        "outcome_covariates": outcome_covariates,
        "true_beta": beta_x,
        "true_rho": rho,
        "true_sigma": sigma_u,
        "true_lambda_coef": true_lambda_coef,
        "n_selected": int(np.sum(selected)),
        "n_total": n,
        "X": X,
        "Z": Z,
    }


@pytest.fixture
def simple_heckman_data():
    """
    Simple Heckman DGP with moderate selection.

    - n = 500
    - ρ = 0.5 (moderate positive selection correlation)
    - β_x = 2.0 (true coefficient on X)
    - Valid exclusion restriction

    Expected:
    - λ = ρ * σ = 0.5 * 1.0 = 0.5
    - Selection rate ~70%
    """
    return generate_heckman_dgp(
        n=500,
        rho=0.5,
        beta_x=2.0,
        gamma_z=1.0,
        sigma_u=1.0,
        has_exclusion=True,
        seed=42,
    )


@pytest.fixture
def strong_selection_data():
    """
    Strong selection bias (high ρ).

    - n = 500
    - ρ = 0.8 (strong positive selection)
    - β_x = 1.5
    - Valid exclusion restriction

    Expected:
    - λ = 0.8 * 1.0 = 0.8
    - Clear selection bias in OLS
    """
    return generate_heckman_dgp(
        n=500,
        rho=0.8,
        beta_x=1.5,
        gamma_z=1.2,
        sigma_u=1.0,
        has_exclusion=True,
        seed=123,
    )


@pytest.fixture
def no_selection_data():
    """
    No selection bias (ρ = 0).

    - n = 500
    - ρ = 0.0 (no correlation)
    - β_x = 2.5
    - OLS should be consistent

    Expected:
    - λ = 0
    - Heckman and OLS should give similar results
    """
    return generate_heckman_dgp(
        n=500,
        rho=0.0,
        beta_x=2.5,
        gamma_z=1.0,
        sigma_u=1.0,
        has_exclusion=True,
        seed=456,
    )


@pytest.fixture
def negative_selection_data():
    """
    Negative selection bias (ρ < 0).

    - n = 500
    - ρ = -0.6 (negative selection)
    - β_x = 1.8

    Expected:
    - λ = -0.6 * 1.0 = -0.6
    - Selection favors low outcome individuals
    """
    return generate_heckman_dgp(
        n=500,
        rho=-0.6,
        beta_x=1.8,
        gamma_z=0.8,
        sigma_u=1.0,
        has_exclusion=True,
        seed=789,
    )


@pytest.fixture
def fragile_identification_data():
    """
    No exclusion restriction (fragile identification).

    - n = 500
    - ρ = 0.5
    - Z affects both selection AND outcome
    - Identification relies only on IMR nonlinearity

    Expected:
    - Estimates may be unstable
    - Wide confidence intervals
    """
    return generate_heckman_dgp(
        n=500,
        rho=0.5,
        beta_x=2.0,
        gamma_z=1.0,
        sigma_u=1.0,
        has_exclusion=False,  # Z affects outcome
        seed=321,
    )


@pytest.fixture
def large_sample_data():
    """
    Large sample for asymptotic properties.

    - n = 2000
    - ρ = 0.5
    - β_x = 2.0

    For Monte Carlo validation of coverage.
    """
    return generate_heckman_dgp(
        n=2000,
        rho=0.5,
        beta_x=2.0,
        gamma_z=1.0,
        sigma_u=1.0,
        has_exclusion=True,
        seed=654,
    )


@pytest.fixture
def small_sample_data():
    """
    Small sample (edge case).

    - n = 100
    - ρ = 0.5
    - β_x = 2.0

    Tests behavior with limited data.
    """
    return generate_heckman_dgp(
        n=100,
        rho=0.5,
        beta_x=2.0,
        gamma_z=1.5,  # Stronger Z to ensure selection variation
        sigma_u=1.0,
        has_exclusion=True,
        seed=111,
    )


@pytest.fixture
def high_selection_rate_data():
    """
    High selection rate (~90% selected).

    - n = 500
    - γ_0 = 1.5 (high intercept in selection)
    - ρ = 0.4

    Tests with many selected observations.
    """
    np.random.seed(222)
    n = 500
    rho = 0.4
    sigma_u = 1.0

    X = np.random.normal(0, 1, n)
    Z = np.random.normal(0, 1, n)

    cov_matrix = np.array([[sigma_u**2, rho * sigma_u], [rho * sigma_u, 1.0]])
    errors = np.random.multivariate_normal([0, 0], cov_matrix, n)
    u, v = errors[:, 0], errors[:, 1]

    # High selection intercept
    s_star = 1.5 + 0.5 * Z + 0.2 * X + v
    selected = (s_star > 0).astype(float)

    outcome_latent = 1.0 + 2.0 * X + u
    outcome = np.where(selected == 1, outcome_latent, np.nan)

    return {
        "outcome": outcome,
        "outcome_selected": outcome_latent[selected == 1],
        "selected": selected,
        "selection_covariates": np.column_stack([X, Z]),
        "outcome_covariates": X.reshape(-1, 1),
        "true_beta": 2.0,
        "true_rho": rho,
        "true_sigma": sigma_u,
        "true_lambda_coef": rho * sigma_u,
        "n_selected": int(np.sum(selected)),
        "n_total": n,
    }


@pytest.fixture
def low_selection_rate_data():
    """
    Low selection rate (~30% selected).

    - n = 500
    - γ_0 = -0.5 (low intercept in selection)
    - ρ = 0.6

    Tests with few selected observations.
    """
    np.random.seed(333)
    n = 500
    rho = 0.6
    sigma_u = 1.0

    X = np.random.normal(0, 1, n)
    Z = np.random.normal(0, 1, n)

    cov_matrix = np.array([[sigma_u**2, rho * sigma_u], [rho * sigma_u, 1.0]])
    errors = np.random.multivariate_normal([0, 0], cov_matrix, n)
    u, v = errors[:, 0], errors[:, 1]

    # Low selection intercept
    s_star = -0.5 + 0.8 * Z + 0.3 * X + v
    selected = (s_star > 0).astype(float)

    outcome_latent = 1.0 + 2.0 * X + u
    outcome = np.where(selected == 1, outcome_latent, np.nan)

    return {
        "outcome": outcome,
        "outcome_selected": outcome_latent[selected == 1],
        "selected": selected,
        "selection_covariates": np.column_stack([X, Z]),
        "outcome_covariates": X.reshape(-1, 1),
        "true_beta": 2.0,
        "true_rho": rho,
        "true_sigma": sigma_u,
        "true_lambda_coef": rho * sigma_u,
        "n_selected": int(np.sum(selected)),
        "n_total": n,
    }
