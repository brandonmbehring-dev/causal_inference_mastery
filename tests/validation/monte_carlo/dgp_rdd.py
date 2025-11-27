"""
Data Generating Processes for RDD Monte Carlo validation.

Provides generators for Sharp RDD and diagnostic validation:
1. Sharp RDD: Linear, quadratic, zero effect, heteroskedastic, high noise
2. Diagnostics: No manipulation, manipulation, balanced covariates, sorting

Key RDD Properties:
- Treatment effect τ = E[Y(1) - Y(0) | X = c] at cutoff c
- Local linear regression: Y = α + τ*D + β*(X - c) + ε for X near c
- Identification: continuity of potential outcomes at cutoff

Key References:
    - Imbens & Lemieux (2008). "Regression Discontinuity Designs: A Guide to Practice"
    - Lee & Lemieux (2010). "Regression Discontinuity Designs in Economics"
    - McCrary (2008). "Manipulation of the Running Variable"
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class RDDData:
    """
    Container for RDD simulation data.

    Attributes
    ----------
    Y : np.ndarray
        Outcome variable
    X : np.ndarray
        Running variable (assignment variable)
    D : np.ndarray
        Treatment indicator (1 if X >= cutoff)
    W : Optional[np.ndarray]
        Covariates (for balance tests)
    cutoff : float
        Cutoff value where treatment changes
    true_tau : float
        True treatment effect at cutoff
    n : int
        Sample size
    dgp_type : str
        Description of the DGP
    slope_left : float
        Slope of E[Y|X] to the left of cutoff
    slope_right : float
        Slope of E[Y|X] to the right of cutoff
    error_sd : float
        Standard deviation of errors
    """
    Y: np.ndarray
    X: np.ndarray
    D: np.ndarray
    W: Optional[np.ndarray]
    cutoff: float
    true_tau: float
    n: int
    dgp_type: str
    slope_left: float
    slope_right: float
    error_sd: float


# =============================================================================
# Sharp RDD DGPs
# =============================================================================

def dgp_rdd_linear(
    n: int = 500,
    true_tau: float = 2.0,
    cutoff: float = 0.0,
    slope: float = 1.0,
    error_sd: float = 1.0,
    random_state: Optional[int] = None,
) -> RDDData:
    """
    Sharp RDD with linear conditional mean.

    DGP:
        X ~ U(-5, 5)
        D = 1{X >= cutoff}
        Y = τ*D + β*(X - c) + ε, ε ~ N(0, σ²)

    This is the simplest case: local linear regression should recover τ
    exactly (up to sampling noise).

    Parameters
    ----------
    n : int
        Sample size
    true_tau : float
        True treatment effect at cutoff
    cutoff : float
        Cutoff value
    slope : float
        Slope of E[Y|X] (same on both sides)
    error_sd : float
        Standard deviation of errors
    random_state : int, optional
        Random seed

    Returns
    -------
    RDDData
        Container with simulation data
    """
    rng = np.random.RandomState(random_state)

    # Running variable: uniform on [-5, 5] centered at cutoff
    X = rng.uniform(cutoff - 5, cutoff + 5, n)

    # Treatment indicator
    D = (X >= cutoff).astype(float)

    # Outcome: linear with jump
    Y = true_tau * D + slope * (X - cutoff) + rng.normal(0, error_sd, n)

    return RDDData(
        Y=Y,
        X=X,
        D=D,
        W=None,
        cutoff=cutoff,
        true_tau=true_tau,
        n=n,
        dgp_type="linear",
        slope_left=slope,
        slope_right=slope,
        error_sd=error_sd,
    )


def dgp_rdd_quadratic(
    n: int = 500,
    true_tau: float = 2.0,
    cutoff: float = 0.0,
    curvature: float = 0.5,
    error_sd: float = 1.0,
    random_state: Optional[int] = None,
) -> RDDData:
    """
    Sharp RDD with quadratic conditional mean.

    DGP:
        X ~ U(-4, 4)
        D = 1{X >= cutoff}
        Y = τ*D + γ*(X - c)² + ε, ε ~ N(0, σ²)

    Tests local linear's ability to handle curvature.
    With sufficient data near cutoff, local linear should still recover τ.

    Parameters
    ----------
    n : int
        Sample size
    true_tau : float
        True treatment effect at cutoff
    cutoff : float
        Cutoff value
    curvature : float
        Coefficient on (X - c)²
    error_sd : float
        Standard deviation of errors
    random_state : int, optional
        Random seed

    Returns
    -------
    RDDData
        Container with simulation data
    """
    rng = np.random.RandomState(random_state)

    # Running variable
    X = rng.uniform(cutoff - 4, cutoff + 4, n)

    # Treatment indicator
    D = (X >= cutoff).astype(float)

    # Outcome: quadratic with jump (derivative = 0 at cutoff)
    Y = true_tau * D + curvature * (X - cutoff)**2 + rng.normal(0, error_sd, n)

    return RDDData(
        Y=Y,
        X=X,
        D=D,
        W=None,
        cutoff=cutoff,
        true_tau=true_tau,
        n=n,
        dgp_type="quadratic",
        slope_left=0.0,  # Slope at cutoff is 0 for quadratic
        slope_right=0.0,
        error_sd=error_sd,
    )


def dgp_rdd_zero_effect(
    n: int = 500,
    cutoff: float = 0.0,
    slope: float = 1.0,
    error_sd: float = 1.0,
    random_state: Optional[int] = None,
) -> RDDData:
    """
    Sharp RDD with zero treatment effect.

    DGP:
        X ~ U(-3, 3)
        D = 1{X >= cutoff}
        Y = β*(X - c) + ε, ε ~ N(0, σ²)  (NO jump)

    Tests that RDD doesn't find phantom effects.
    Estimate should be ≈0, p-value > 0.05.

    Parameters
    ----------
    n : int
        Sample size
    cutoff : float
        Cutoff value
    slope : float
        Slope of E[Y|X]
    error_sd : float
        Standard deviation of errors
    random_state : int, optional
        Random seed

    Returns
    -------
    RDDData
        Container with simulation data
    """
    rng = np.random.RandomState(random_state)

    # Running variable
    X = rng.uniform(cutoff - 3, cutoff + 3, n)

    # Treatment indicator (still defined even with no effect)
    D = (X >= cutoff).astype(float)

    # Outcome: NO discontinuity
    Y = slope * (X - cutoff) + rng.normal(0, error_sd, n)

    return RDDData(
        Y=Y,
        X=X,
        D=D,
        W=None,
        cutoff=cutoff,
        true_tau=0.0,
        n=n,
        dgp_type="zero_effect",
        slope_left=slope,
        slope_right=slope,
        error_sd=error_sd,
    )


def dgp_rdd_heteroskedastic(
    n: int = 500,
    true_tau: float = 2.0,
    cutoff: float = 0.0,
    slope: float = 1.0,
    error_sd_left: float = 0.5,
    error_sd_right: float = 2.0,
    random_state: Optional[int] = None,
) -> RDDData:
    """
    Sharp RDD with heteroskedastic errors.

    DGP:
        X ~ U(-5, 5)
        D = 1{X >= cutoff}
        Y = τ*D + β*(X - c) + ε
        ε ~ N(0, σ_L²) if X < c
        ε ~ N(0, σ_R²) if X >= c

    Tests robust standard errors.
    Standard (homoskedastic) SEs will be incorrect.

    Parameters
    ----------
    n : int
        Sample size
    true_tau : float
        True treatment effect at cutoff
    cutoff : float
        Cutoff value
    slope : float
        Slope of E[Y|X]
    error_sd_left : float
        SD of errors left of cutoff
    error_sd_right : float
        SD of errors right of cutoff
    random_state : int, optional
        Random seed

    Returns
    -------
    RDDData
        Container with simulation data
    """
    rng = np.random.RandomState(random_state)

    # Running variable
    X = rng.uniform(cutoff - 5, cutoff + 5, n)

    # Treatment indicator
    D = (X >= cutoff).astype(float)

    # Heteroskedastic errors
    error_sd = np.where(X < cutoff, error_sd_left, error_sd_right)
    errors = rng.normal(0, 1, n) * error_sd

    # Outcome
    Y = true_tau * D + slope * (X - cutoff) + errors

    return RDDData(
        Y=Y,
        X=X,
        D=D,
        W=None,
        cutoff=cutoff,
        true_tau=true_tau,
        n=n,
        dgp_type="heteroskedastic",
        slope_left=slope,
        slope_right=slope,
        error_sd=(error_sd_left + error_sd_right) / 2,  # Average for reference
    )


def dgp_rdd_high_noise(
    n: int = 500,
    true_tau: float = 2.0,
    cutoff: float = 0.0,
    slope: float = 0.5,
    error_sd: float = 3.0,
    random_state: Optional[int] = None,
) -> RDDData:
    """
    Sharp RDD with high noise (low signal-to-noise ratio).

    DGP:
        X ~ U(-5, 5)
        D = 1{X >= cutoff}
        Y = τ*D + β*(X - c) + ε, ε ~ N(0, σ²) with large σ

    Tests that:
    - Estimates remain unbiased
    - CIs are appropriately wide
    - SEs scale correctly with noise

    Parameters
    ----------
    n : int
        Sample size
    true_tau : float
        True treatment effect at cutoff
    cutoff : float
        Cutoff value
    slope : float
        Slope of E[Y|X]
    error_sd : float
        Standard deviation of errors (large)
    random_state : int, optional
        Random seed

    Returns
    -------
    RDDData
        Container with simulation data
    """
    rng = np.random.RandomState(random_state)

    X = rng.uniform(cutoff - 5, cutoff + 5, n)
    D = (X >= cutoff).astype(float)
    Y = true_tau * D + slope * (X - cutoff) + rng.normal(0, error_sd, n)

    return RDDData(
        Y=Y,
        X=X,
        D=D,
        W=None,
        cutoff=cutoff,
        true_tau=true_tau,
        n=n,
        dgp_type="high_noise",
        slope_left=slope,
        slope_right=slope,
        error_sd=error_sd,
    )


def dgp_rdd_different_slopes(
    n: int = 500,
    true_tau: float = 2.0,
    cutoff: float = 0.0,
    slope_left: float = 1.0,
    slope_right: float = 0.5,
    error_sd: float = 1.0,
    random_state: Optional[int] = None,
) -> RDDData:
    """
    Sharp RDD with different slopes on each side of cutoff.

    DGP:
        X ~ U(-5, 5)
        D = 1{X >= cutoff}
        Y = τ*D + β_L*(X - c)*1{X < c} + β_R*(X - c)*1{X >= c} + ε

    Tests local linear's ability to estimate different slopes.
    Treatment effect is still the jump at cutoff.

    Parameters
    ----------
    n : int
        Sample size
    true_tau : float
        True treatment effect at cutoff
    cutoff : float
        Cutoff value
    slope_left : float
        Slope left of cutoff
    slope_right : float
        Slope right of cutoff
    error_sd : float
        Standard deviation of errors
    random_state : int, optional
        Random seed

    Returns
    -------
    RDDData
        Container with simulation data
    """
    rng = np.random.RandomState(random_state)

    X = rng.uniform(cutoff - 5, cutoff + 5, n)
    D = (X >= cutoff).astype(float)

    # Different slopes on each side
    slope = np.where(X < cutoff, slope_left, slope_right)
    Y = true_tau * D + slope * (X - cutoff) + rng.normal(0, error_sd, n)

    return RDDData(
        Y=Y,
        X=X,
        D=D,
        W=None,
        cutoff=cutoff,
        true_tau=true_tau,
        n=n,
        dgp_type="different_slopes",
        slope_left=slope_left,
        slope_right=slope_right,
        error_sd=error_sd,
    )


# =============================================================================
# Diagnostic Test DGPs
# =============================================================================

def dgp_rdd_no_manipulation(
    n: int = 1000,
    true_tau: float = 2.0,
    cutoff: float = 0.0,
    random_state: Optional[int] = None,
) -> RDDData:
    """
    RDD with NO manipulation of running variable.

    DGP:
        X ~ U(-5, 5)  (uniform density - NO bunching at cutoff)
        D = 1{X >= cutoff}
        Y = τ*D + 0.5*(X - c) + ε

    McCrary test should NOT reject H₀ (no manipulation).
    Expected: p-value > 0.05 in ~95% of simulations.

    Parameters
    ----------
    n : int
        Sample size (larger for density estimation)
    true_tau : float
        True treatment effect
    cutoff : float
        Cutoff value
    random_state : int, optional
        Random seed

    Returns
    -------
    RDDData
        Container with simulation data
    """
    rng = np.random.RandomState(random_state)

    # Uniform running variable - no manipulation
    X = rng.uniform(cutoff - 5, cutoff + 5, n)
    D = (X >= cutoff).astype(float)
    Y = true_tau * D + 0.5 * (X - cutoff) + rng.normal(0, 1, n)

    return RDDData(
        Y=Y,
        X=X,
        D=D,
        W=None,
        cutoff=cutoff,
        true_tau=true_tau,
        n=n,
        dgp_type="no_manipulation",
        slope_left=0.5,
        slope_right=0.5,
        error_sd=1.0,
    )


def dgp_rdd_manipulation(
    n: int = 1000,
    true_tau: float = 2.0,
    cutoff: float = 0.0,
    bunching_fraction: float = 0.15,
    bunching_width: float = 0.3,
    random_state: Optional[int] = None,
) -> RDDData:
    """
    RDD with manipulation (bunching) at cutoff.

    DGP:
        X_base ~ U(-5, 5)
        With probability bunching_fraction, units just below cutoff
        are moved to just above cutoff (manipulation).

    McCrary test SHOULD reject H₀.
    Expected: p-value < 0.05 in most simulations.

    Parameters
    ----------
    n : int
        Sample size
    true_tau : float
        True treatment effect
    cutoff : float
        Cutoff value
    bunching_fraction : float
        Fraction of units manipulated
    bunching_width : float
        Width of manipulation window (units within this distance can manipulate)
    random_state : int, optional
        Random seed

    Returns
    -------
    RDDData
        Container with simulation data
    """
    rng = np.random.RandomState(random_state)

    # Base running variable
    X = rng.uniform(cutoff - 5, cutoff + 5, n)

    # Manipulation: units just below cutoff move to just above
    manipulate_candidates = (X >= cutoff - bunching_width) & (X < cutoff)
    manipulate = manipulate_candidates & (rng.uniform(0, 1, n) < bunching_fraction / (bunching_width / 5))

    # Move manipulated units to just above cutoff
    X[manipulate] = cutoff + rng.uniform(0, bunching_width, manipulate.sum())

    D = (X >= cutoff).astype(float)
    Y = true_tau * D + 0.5 * (X - cutoff) + rng.normal(0, 1, n)

    return RDDData(
        Y=Y,
        X=X,
        D=D,
        W=None,
        cutoff=cutoff,
        true_tau=true_tau,
        n=n,
        dgp_type="manipulation",
        slope_left=0.5,
        slope_right=0.5,
        error_sd=1.0,
    )


def dgp_rdd_balanced_covariates(
    n: int = 500,
    true_tau: float = 2.0,
    cutoff: float = 0.0,
    n_covariates: int = 3,
    random_state: Optional[int] = None,
) -> RDDData:
    """
    RDD with balanced covariates (no sorting).

    DGP:
        X ~ U(-5, 5)
        W_j ~ N(0, 1) independent of X  (no sorting)
        D = 1{X >= cutoff}
        Y = τ*D + 0.5*(X - c) + W'γ + ε

    Covariate balance test should NOT reject H₀.
    No discontinuity in E[W|X] at cutoff.

    Parameters
    ----------
    n : int
        Sample size
    true_tau : float
        True treatment effect
    cutoff : float
        Cutoff value
    n_covariates : int
        Number of covariates
    random_state : int, optional
        Random seed

    Returns
    -------
    RDDData
        Container with simulation data
    """
    rng = np.random.RandomState(random_state)

    X = rng.uniform(cutoff - 5, cutoff + 5, n)
    D = (X >= cutoff).astype(float)

    # Covariates: independent of X (balanced)
    W = rng.normal(0, 1, (n, n_covariates))

    # Outcome includes covariate effects
    coef = rng.normal(0, 0.5, n_covariates)
    Y = true_tau * D + 0.5 * (X - cutoff) + W @ coef + rng.normal(0, 1, n)

    return RDDData(
        Y=Y,
        X=X,
        D=D,
        W=W,
        cutoff=cutoff,
        true_tau=true_tau,
        n=n,
        dgp_type="balanced_covariates",
        slope_left=0.5,
        slope_right=0.5,
        error_sd=1.0,
    )


def dgp_rdd_sorting(
    n: int = 500,
    true_tau: float = 2.0,
    cutoff: float = 0.0,
    sorting_strength: float = 0.5,
    random_state: Optional[int] = None,
) -> RDDData:
    """
    RDD with sorting on a covariate.

    DGP:
        X ~ U(-5, 5)
        W = α + β*1{X >= cutoff} + ε_W  (DISCONTINUITY in W at cutoff)
        D = 1{X >= cutoff}
        Y = τ*D + 0.5*(X - c) + γ*W + ε

    Covariate balance test SHOULD reject H₀.
    E[W|X=c⁺] ≠ E[W|X=c⁻] indicates sorting.

    Parameters
    ----------
    n : int
        Sample size
    true_tau : float
        True treatment effect
    cutoff : float
        Cutoff value
    sorting_strength : float
        Size of jump in E[W|X] at cutoff
    random_state : int, optional
        Random seed

    Returns
    -------
    RDDData
        Container with simulation data
    """
    rng = np.random.RandomState(random_state)

    X = rng.uniform(cutoff - 5, cutoff + 5, n)
    D = (X >= cutoff).astype(float)

    # Covariate with discontinuity (sorting)
    W = sorting_strength * D + rng.normal(0, 1, n)
    W = W.reshape(-1, 1)

    # Outcome
    Y = true_tau * D + 0.5 * (X - cutoff) + 0.5 * W.ravel() + rng.normal(0, 1, n)

    return RDDData(
        Y=Y,
        X=X,
        D=D,
        W=W,
        cutoff=cutoff,
        true_tau=true_tau,
        n=n,
        dgp_type="sorting",
        slope_left=0.5,
        slope_right=0.5,
        error_sd=1.0,
    )


# =============================================================================
# Additional DGPs for specific tests
# =============================================================================

def dgp_rdd_small_sample(
    n: int = 100,
    true_tau: float = 2.0,
    cutoff: float = 0.0,
    random_state: Optional[int] = None,
) -> RDDData:
    """
    Sharp RDD with small sample size.

    Tests finite-sample behavior:
    - Larger SEs
    - Potentially wider CIs
    - May need t-distribution critical values

    Parameters
    ----------
    n : int
        Sample size (small)
    true_tau : float
        True treatment effect
    cutoff : float
        Cutoff value
    random_state : int, optional
        Random seed

    Returns
    -------
    RDDData
        Container with simulation data
    """
    return dgp_rdd_linear(
        n=n,
        true_tau=true_tau,
        cutoff=cutoff,
        slope=1.0,
        error_sd=1.0,
        random_state=random_state,
    )


def dgp_rdd_large_sample(
    n: int = 2000,
    true_tau: float = 2.0,
    cutoff: float = 0.0,
    random_state: Optional[int] = None,
) -> RDDData:
    """
    Sharp RDD with large sample size.

    Tests asymptotic behavior:
    - Smaller SEs (proportional to 1/√n)
    - Normal approximation should work well
    - Estimate should be very close to true τ

    Parameters
    ----------
    n : int
        Sample size (large)
    true_tau : float
        True treatment effect
    cutoff : float
        Cutoff value
    random_state : int, optional
        Random seed

    Returns
    -------
    RDDData
        Container with simulation data
    """
    return dgp_rdd_linear(
        n=n,
        true_tau=true_tau,
        cutoff=cutoff,
        slope=1.0,
        error_sd=1.0,
        random_state=random_state,
    )
