"""
Data Generating Processes for Bunching Monte Carlo Validation.

Generates synthetic data with known bunching parameters for validating
the bunching estimator (Saez 2010).

Key Parameters:
- Counterfactual distribution (shape, location, scale)
- Kink point (threshold where marginal rate changes)
- Bunching response (elasticity, excess mass)
- Measurement noise

References:
- Saez (2010) - Original bunching methodology
- Chetty et al. (2011) - Frictions and optimization errors
- Kleven (2016) - Bunching estimation review
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray


def dgp_bunching_simple(
    n: int = 1000,
    kink_point: float = 50.0,
    true_excess_mass: float = 2.0,
    counterfactual_std: float = 15.0,
    bunching_std: float = 1.0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray[np.float64], float, float, float]:
    """Generate simple bunching data with known excess mass.

    Creates a mixture of:
    1. Background counterfactual (normal distribution)
    2. Bunchers concentrated at kink

    Parameters
    ----------
    n : int
        Total sample size.
    kink_point : float
        Location of the kink (e.g., tax threshold).
    true_excess_mass : float
        Target excess mass b = B/h0 (normalized bunching).
    counterfactual_std : float
        Standard deviation of counterfactual distribution.
    bunching_std : float
        Standard deviation of buncher concentration at kink.
    random_state : Optional[int]
        Random seed.

    Returns
    -------
    data : NDArray[np.float64]
        Observed data with bunching.
    true_excess_mass : float
        True excess mass parameter.
    kink_point : float
        Kink location.
    bunching_width : float
        Recommended bunching region half-width.
    """
    rng = np.random.default_rng(random_state)

    # Counterfactual height at kink (for normal centered at kink)
    # h0 = n * pdf(kink) = n * (1 / (sqrt(2π) * σ))
    h0_per_unit = 1 / (np.sqrt(2 * np.pi) * counterfactual_std)

    # Target excess count B = b * h0
    # But h0 depends on bin width, so we work with fractions
    # Fraction of bunchers = b * h0_per_unit approximately
    buncher_fraction = min(0.3, true_excess_mass * h0_per_unit)
    n_bunchers = int(n * buncher_fraction)
    n_background = n - n_bunchers

    # Generate background (counterfactual)
    background = rng.normal(kink_point, counterfactual_std, size=n_background)

    # Generate bunchers (concentrated at kink)
    bunchers = rng.normal(kink_point, bunching_std, size=n_bunchers)

    # Combine
    data = np.concatenate([background, bunchers])

    # Recommended bunching width
    bunching_width = 3 * bunching_std

    return data, true_excess_mass, kink_point, bunching_width


def dgp_bunching_uniform_counterfactual(
    n: int = 1000,
    kink_point: float = 50.0,
    data_range: Tuple[float, float] = (20.0, 80.0),
    buncher_fraction: float = 0.15,
    bunching_std: float = 1.0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray[np.float64], float, float, float]:
    """Generate bunching data with uniform counterfactual.

    Uniform counterfactual makes excess mass calculation simpler
    (constant h0 across all bins).

    Parameters
    ----------
    n : int
        Total sample size.
    kink_point : float
        Kink location.
    data_range : Tuple[float, float]
        (min, max) of data range.
    buncher_fraction : float
        Fraction of observations that are bunchers.
    bunching_std : float
        Standard deviation of bunching concentration.
    random_state : Optional[int]
        Random seed.

    Returns
    -------
    data : NDArray[np.float64]
        Observed data.
    true_excess_mass : float
        True excess mass (approximately buncher_fraction * range / bunching_width).
    kink_point : float
        Kink location.
    bunching_width : float
        Recommended bunching region half-width.
    """
    rng = np.random.default_rng(random_state)

    n_bunchers = int(n * buncher_fraction)
    n_background = n - n_bunchers

    # Background (uniform)
    background = rng.uniform(data_range[0], data_range[1], size=n_background)

    # Bunchers
    bunchers = rng.normal(kink_point, bunching_std, size=n_bunchers)

    data = np.concatenate([background, bunchers])

    # True excess mass approximation
    # h0 ≈ n_background / n_bins for uniform
    # B ≈ n_bunchers in bunching region
    bunching_width = 3 * bunching_std
    true_excess_mass = (
        buncher_fraction
        / (1 - buncher_fraction)
        * ((data_range[1] - data_range[0]) / (2 * bunching_width))
    )

    return data, true_excess_mass, kink_point, bunching_width


def dgp_bunching_no_effect(
    n: int = 1000,
    kink_point: float = 50.0,
    counterfactual_std: float = 15.0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray[np.float64], float, float, float]:
    """Generate data with no bunching (null effect).

    For testing Type I error - no excess mass at kink.

    Parameters
    ----------
    n : int
        Sample size.
    kink_point : float
        Kink location.
    counterfactual_std : float
        Standard deviation.
    random_state : Optional[int]
        Random seed.

    Returns
    -------
    data : NDArray[np.float64]
        Data without bunching.
    true_excess_mass : float
        0.0 (no bunching).
    kink_point : float
        Kink location.
    bunching_width : float
        Recommended bunching width.
    """
    rng = np.random.default_rng(random_state)

    # Pure counterfactual - no bunching
    data = rng.normal(kink_point, counterfactual_std, size=n)

    bunching_width = 5.0

    return data, 0.0, kink_point, bunching_width


def dgp_bunching_with_elasticity(
    n: int = 1000,
    kink_point: float = 50000.0,
    t1_rate: float = 0.20,
    t2_rate: float = 0.30,
    true_elasticity: float = 0.25,
    counterfactual_std: float = 12000.0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray[np.float64], float, float, float, float, float]:
    """Generate bunching data with known elasticity.

    Uses the bunching-elasticity relationship:
    e = b / ln((1-t1)/(1-t2))

    Parameters
    ----------
    n : int
        Sample size.
    kink_point : float
        Kink location (e.g., $50,000 tax threshold).
    t1_rate : float
        Marginal rate below kink.
    t2_rate : float
        Marginal rate above kink.
    true_elasticity : float
        Target behavioral elasticity.
    counterfactual_std : float
        Standard deviation of counterfactual.
    random_state : Optional[int]
        Random seed.

    Returns
    -------
    data : NDArray[np.float64]
        Observed data.
    true_elasticity : float
        True elasticity.
    true_excess_mass : float
        Implied excess mass.
    kink_point : float
        Kink location.
    t1_rate : float
        Rate below kink.
    t2_rate : float
        Rate above kink.
    """
    rng = np.random.default_rng(random_state)

    # Calculate required excess mass from elasticity
    # e = b / ln((1-t1)/(1-t2))
    # b = e * ln((1-t1)/(1-t2))
    log_rate_change = np.log((1 - t1_rate) / (1 - t2_rate))
    true_excess_mass = true_elasticity * log_rate_change

    # Counterfactual height at kink
    h0_per_unit = 1 / (np.sqrt(2 * np.pi) * counterfactual_std)

    # Buncher fraction
    buncher_fraction = min(0.25, true_excess_mass * h0_per_unit)
    n_bunchers = int(n * buncher_fraction)
    n_background = n - n_bunchers

    # Generate
    bunching_std = counterfactual_std * 0.05  # Tight bunching
    background = rng.normal(kink_point, counterfactual_std, size=n_background)
    bunchers = rng.normal(kink_point, bunching_std, size=n_bunchers)

    data = np.concatenate([background, bunchers])

    return data, true_elasticity, true_excess_mass, kink_point, t1_rate, t2_rate


def dgp_bunching_asymmetric(
    n: int = 1000,
    kink_point: float = 50.0,
    buncher_fraction: float = 0.15,
    bunching_std: float = 1.0,
    bunching_offset: float = -1.0,
    counterfactual_std: float = 15.0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray[np.float64], float, float, float]:
    """Generate bunching data where bunchers are offset from kink.

    Tests robustness when bunching is not perfectly centered at kink
    (e.g., due to rounding, optimization frictions).

    Parameters
    ----------
    n : int
        Sample size.
    kink_point : float
        Kink location.
    buncher_fraction : float
        Fraction of bunchers.
    bunching_std : float
        Standard deviation of bunching.
    bunching_offset : float
        Offset of bunching center from kink (negative = below kink).
    counterfactual_std : float
        Counterfactual standard deviation.
    random_state : Optional[int]
        Random seed.

    Returns
    -------
    data : NDArray[np.float64]
        Observed data.
    true_excess_mass : float
        Approximate excess mass.
    kink_point : float
        Kink location.
    bunching_width : float
        Recommended bunching width.
    """
    rng = np.random.default_rng(random_state)

    n_bunchers = int(n * buncher_fraction)
    n_background = n - n_bunchers

    # Background
    background = rng.normal(kink_point, counterfactual_std, size=n_background)

    # Bunchers offset from kink
    bunching_center = kink_point + bunching_offset
    bunchers = rng.normal(bunching_center, bunching_std, size=n_bunchers)

    data = np.concatenate([background, bunchers])

    # Approximate excess mass
    h0_per_unit = 1 / (np.sqrt(2 * np.pi) * counterfactual_std)
    true_excess_mass = buncher_fraction / h0_per_unit

    bunching_width = 3 * bunching_std + abs(bunching_offset)

    return data, true_excess_mass, kink_point, bunching_width


def dgp_bunching_diffuse(
    n: int = 1000,
    kink_point: float = 50.0,
    buncher_fraction: float = 0.15,
    bunching_std: float = 5.0,
    counterfactual_std: float = 15.0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray[np.float64], float, float, float]:
    """Generate bunching data with diffuse bunching (optimization frictions).

    Per Chetty et al. (2011), adjustment costs create diffuse bunching
    rather than sharp bunching at the kink.

    Parameters
    ----------
    n : int
        Sample size.
    kink_point : float
        Kink location.
    buncher_fraction : float
        Fraction of bunchers.
    bunching_std : float
        Standard deviation of bunching (larger = more diffuse).
    counterfactual_std : float
        Counterfactual standard deviation.
    random_state : Optional[int]
        Random seed.

    Returns
    -------
    data : NDArray[np.float64]
        Observed data.
    true_excess_mass : float
        Approximate excess mass.
    kink_point : float
        Kink location.
    bunching_width : float
        Recommended bunching width.
    """
    rng = np.random.default_rng(random_state)

    n_bunchers = int(n * buncher_fraction)
    n_background = n - n_bunchers

    background = rng.normal(kink_point, counterfactual_std, size=n_background)
    bunchers = rng.normal(kink_point, bunching_std, size=n_bunchers)

    data = np.concatenate([background, bunchers])

    # Excess mass harder to detect with diffuse bunching
    h0_per_unit = 1 / (np.sqrt(2 * np.pi) * counterfactual_std)
    true_excess_mass = buncher_fraction / h0_per_unit

    bunching_width = 2 * bunching_std

    return data, true_excess_mass, kink_point, bunching_width


def dgp_bunching_large_sample(
    n: int = 10000,
    kink_point: float = 50.0,
    true_excess_mass: float = 1.5,
    counterfactual_std: float = 15.0,
    bunching_std: float = 1.0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray[np.float64], float, float, float]:
    """Generate large sample bunching data for precision testing.

    Parameters
    ----------
    n : int
        Large sample size (default 10000).
    kink_point : float
        Kink location.
    true_excess_mass : float
        Target excess mass.
    counterfactual_std : float
        Counterfactual standard deviation.
    bunching_std : float
        Bunching standard deviation.
    random_state : Optional[int]
        Random seed.

    Returns
    -------
    data, true_excess_mass, kink_point, bunching_width
    """
    return dgp_bunching_simple(
        n=n,
        kink_point=kink_point,
        true_excess_mass=true_excess_mass,
        counterfactual_std=counterfactual_std,
        bunching_std=bunching_std,
        random_state=random_state,
    )


def dgp_bunching_small_sample(
    n: int = 200,
    kink_point: float = 50.0,
    true_excess_mass: float = 2.5,
    counterfactual_std: float = 15.0,
    bunching_std: float = 1.5,
    random_state: Optional[int] = None,
) -> Tuple[NDArray[np.float64], float, float, float]:
    """Generate small sample bunching data.

    Tests behavior with limited data where estimates are noisier.

    Parameters
    ----------
    n : int
        Small sample size (default 200).
    kink_point : float
        Kink location.
    true_excess_mass : float
        Target excess mass (larger for detectability).
    counterfactual_std : float
        Counterfactual standard deviation.
    bunching_std : float
        Bunching standard deviation.
    random_state : Optional[int]
        Random seed.

    Returns
    -------
    data, true_excess_mass, kink_point, bunching_width
    """
    return dgp_bunching_simple(
        n=n,
        kink_point=kink_point,
        true_excess_mass=true_excess_mass,
        counterfactual_std=counterfactual_std,
        bunching_std=bunching_std,
        random_state=random_state,
    )
