"""
Excess mass estimation for bunching analysis.

Implements the Saez (2010) bunching estimator with extensions from Chetty et al. (2011)
and Kleven (2016).

Key Concepts:
- Excess mass (b): Normalized bunching mass = B / h0
- B: Raw excess count (actual - counterfactual in bunching region)
- h0: Counterfactual density height at the kink
- Elasticity: Behavioral response derived from bunching

References:
- Saez, E. (2010). "Do Taxpayers Bunch at Kink Points?" AEJ: Economic Policy.
- Chetty, R., et al. (2011). "Adjustment Costs, Firm Responses, and Micro vs.
    Macro Labor Supply Elasticities." Quarterly Journal of Economics.
- Kleven, H. J. (2016). "Bunching." Annual Review of Economics.
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from .types import BunchingResult, CounterfactualResult
from .counterfactual import estimate_counterfactual


def compute_excess_mass(
    counterfactual_result: CounterfactualResult,
) -> Tuple[float, float, float]:
    """Compute excess mass from counterfactual estimation.

    Parameters
    ----------
    counterfactual_result : CounterfactualResult
        Result from counterfactual density estimation.

    Returns
    -------
    excess_mass : float
        Normalized excess mass (b = B / h0).
    excess_mass_count : float
        Raw excess count (B = actual - counterfactual in bunching region).
    h0 : float
        Counterfactual density height at kink.

    Notes
    -----
    The excess mass b is the key quantity for elasticity estimation.
    It measures how many "extra" agents bunch at the kink relative to
    what we'd expect under the counterfactual.

    b = B / h0 = (∑ bunching region [actual - counterfactual]) / h0

    where h0 is the counterfactual height at the kink point.
    """
    bin_centers = counterfactual_result["bin_centers"]
    actual = counterfactual_result["actual_counts"]
    counterfactual = counterfactual_result["counterfactual_counts"]
    bunching_lower, bunching_upper = counterfactual_result["bunching_region"]

    # Identify bins in bunching region
    bunching_mask = (bin_centers >= bunching_lower) & (bin_centers <= bunching_upper)

    # Raw excess count
    excess_count = np.sum(actual[bunching_mask] - counterfactual[bunching_mask])

    # Find counterfactual height at kink (center of bunching region)
    kink_point = (bunching_lower + bunching_upper) / 2
    kink_idx = np.argmin(np.abs(bin_centers - kink_point))
    h0 = counterfactual[kink_idx]

    # Normalized excess mass
    if h0 > 0:
        excess_mass = excess_count / h0
    else:
        # Degenerate case: counterfactual is zero at kink
        excess_mass = np.inf if excess_count > 0 else 0.0

    return excess_mass, excess_count, h0


def compute_elasticity(
    excess_mass: float,
    t1_rate: float,
    t2_rate: float,
) -> float:
    """Compute behavioral elasticity from excess mass.

    Parameters
    ----------
    excess_mass : float
        Normalized excess mass (b = B / h0).
    t1_rate : float
        Marginal rate below the kink (e.g., lower tax rate).
    t2_rate : float
        Marginal rate above the kink (e.g., higher tax rate).

    Returns
    -------
    elasticity : float
        Estimated behavioral elasticity.

    Raises
    ------
    ValueError
        If rates are not in valid range or t2_rate <= t1_rate.

    Notes
    -----
    For a tax kink where the marginal rate increases from t1 to t2:

        e = b / ln((1 - t1) / (1 - t2))

    This assumes:
    1. Isoelastic utility
    2. No optimization frictions
    3. Perfect bunching (all bunchers locate exactly at kink)

    For frictions, see Chetty et al. (2011) for adjustment costs.

    Examples
    --------
    >>> # Tax kink: rate increases from 20% to 30%
    >>> elasticity = compute_elasticity(excess_mass=0.5, t1_rate=0.2, t2_rate=0.3)
    >>> print(f"Elasticity: {elasticity:.3f}")
    """
    # Input validation
    if t1_rate < 0 or t1_rate >= 1:
        raise ValueError(f"t1_rate must be in [0, 1), got {t1_rate}")
    if t2_rate < 0 or t2_rate >= 1:
        raise ValueError(f"t2_rate must be in [0, 1), got {t2_rate}")
    if t2_rate <= t1_rate:
        raise ValueError(f"t2_rate ({t2_rate}) must be greater than t1_rate ({t1_rate}) for a kink")

    # Log change in net-of-tax rate
    log_change = np.log((1 - t1_rate) / (1 - t2_rate))

    if log_change == 0:
        raise ValueError("Rates are too close: log change is zero")

    elasticity = excess_mass / log_change

    return elasticity


def bootstrap_bunching_se(
    data: NDArray[np.float64],
    kink_point: float,
    bunching_width: float,
    n_bins: Optional[int] = None,
    bin_width: Optional[float] = None,
    polynomial_order: int = 7,
    t1_rate: Optional[float] = None,
    t2_rate: Optional[float] = None,
    n_bootstrap: int = 200,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """Bootstrap standard errors for bunching estimates.

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data.
    kink_point : float
        Location of the kink.
    bunching_width : float
        Half-width of bunching region.
    n_bins : Optional[int]
        Number of bins.
    bin_width : Optional[float]
        Bin width (alternative to n_bins).
    polynomial_order : int
        Polynomial order for counterfactual.
    t1_rate : Optional[float]
        Rate below kink (for elasticity SE).
    t2_rate : Optional[float]
        Rate above kink (for elasticity SE).
    n_bootstrap : int, default=200
        Number of bootstrap iterations.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    excess_mass_se : float
        Standard error of excess mass.
    excess_mass_count_se : float
        Standard error of raw excess count.
    elasticity_se : float
        Standard error of elasticity (if rates provided, else 0).
    h0_se : float
        Standard error of counterfactual height.

    Notes
    -----
    Uses nonparametric bootstrap: resample data with replacement,
    re-estimate, compute standard deviation across bootstrap samples.
    """
    rng = np.random.default_rng(random_state)

    excess_mass_samples = []
    excess_count_samples = []
    elasticity_samples = []
    h0_samples = []

    n = len(data)

    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_indices = rng.integers(0, n, size=n)
        bootstrap_data = data[bootstrap_indices]

        try:
            # Estimate counterfactual
            result = estimate_counterfactual(
                data=bootstrap_data,
                kink_point=kink_point,
                bunching_width=bunching_width,
                n_bins=n_bins,
                bin_width=bin_width,
                polynomial_order=polynomial_order,
            )

            # Compute excess mass
            b, B, h0 = compute_excess_mass(result)

            excess_mass_samples.append(b)
            excess_count_samples.append(B)
            h0_samples.append(h0)

            # Compute elasticity if rates provided
            if t1_rate is not None and t2_rate is not None:
                try:
                    e = compute_elasticity(b, t1_rate, t2_rate)
                    elasticity_samples.append(e)
                except (ValueError, ZeroDivisionError):
                    pass

        except (ValueError, np.linalg.LinAlgError):
            # Skip failed bootstrap iterations
            continue

    # Compute standard errors
    if len(excess_mass_samples) < 10:
        raise ValueError(
            f"Too few successful bootstrap iterations ({len(excess_mass_samples)}). "
            "Check data quality and bunching parameters."
        )

    excess_mass_se = np.std(excess_mass_samples, ddof=1)
    excess_count_se = np.std(excess_count_samples, ddof=1)
    h0_se = np.std(h0_samples, ddof=1)

    if len(elasticity_samples) >= 10:
        elasticity_se = np.std(elasticity_samples, ddof=1)
    else:
        elasticity_se = 0.0

    return excess_mass_se, excess_count_se, elasticity_se, h0_se


def bunching_estimator(
    data: NDArray[np.float64],
    kink_point: float,
    bunching_width: float,
    n_bins: Optional[int] = None,
    bin_width: Optional[float] = None,
    polynomial_order: int = 7,
    t1_rate: Optional[float] = None,
    t2_rate: Optional[float] = None,
    n_bootstrap: int = 200,
    random_state: Optional[int] = None,
) -> BunchingResult:
    """Main bunching estimator (Saez 2010).

    Estimates behavioral responses from bunching at a kink in the budget constraint.

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data (e.g., reported income values).
    kink_point : float
        Location of the kink (e.g., tax bracket threshold).
    bunching_width : float
        Half-width of bunching region. Region is
        [kink_point - bunching_width, kink_point + bunching_width].
    n_bins : Optional[int], default=None
        Number of bins for histogram. If None, uses bin_width.
    bin_width : Optional[float], default=None
        Width of each bin. If None, uses n_bins. One must be specified.
    polynomial_order : int, default=7
        Order of polynomial for counterfactual. Standard is 7 (Saez 2010).
    t1_rate : Optional[float], default=None
        Marginal rate below kink (for elasticity). Must be in [0, 1).
    t2_rate : Optional[float], default=None
        Marginal rate above kink (for elasticity). Must be > t1_rate.
    n_bootstrap : int, default=200
        Number of bootstrap iterations for standard errors.
    random_state : Optional[int], default=None
        Random seed for reproducibility.

    Returns
    -------
    BunchingResult
        Dictionary containing:
        - excess_mass: Normalized excess mass (b = B/h0)
        - excess_mass_se: Standard error of excess mass
        - excess_mass_count: Raw excess count (B)
        - elasticity: Behavioral elasticity (if rates provided)
        - elasticity_se: Standard error of elasticity
        - kink_point: Location of kink
        - bunching_region: (lower, upper) bounds
        - counterfactual: Full CounterfactualResult
        - t1_rate, t2_rate: Tax rates (if provided)
        - n_obs: Number of observations
        - n_bootstrap: Bootstrap iterations
        - convergence: Whether estimation succeeded
        - message: Descriptive message

    Raises
    ------
    ValueError
        If inputs are invalid or estimation fails.

    Examples
    --------
    >>> import numpy as np
    >>> # Simulate income data with bunching at $50,000 threshold
    >>> np.random.seed(42)
    >>> # Background distribution
    >>> normal_income = np.random.normal(45000, 12000, 900)
    >>> # Bunchers at kink
    >>> bunchers = np.random.normal(50000, 500, 100)
    >>> data = np.concatenate([normal_income, bunchers])
    >>> data = data[(data > 20000) & (data < 80000)]  # Trim
    >>>
    >>> # Estimate bunching
    >>> result = bunching_estimator(
    ...     data=data,
    ...     kink_point=50000,
    ...     bunching_width=2000,
    ...     n_bins=60,
    ...     t1_rate=0.25,  # 25% marginal rate below
    ...     t2_rate=0.35,  # 35% marginal rate above
    ... )
    >>> print(f"Excess mass: {result['excess_mass']:.2f}")
    >>> print(f"Elasticity: {result['elasticity']:.3f}")

    Notes
    -----
    The bunching estimator:

    1. Bins the data into a histogram
    2. Excludes the bunching region and fits a polynomial counterfactual
    3. Computes excess mass: b = B / h0
       - B = sum of (actual - counterfactual) in bunching region
       - h0 = counterfactual height at kink
    4. Computes elasticity: e = b / ln((1-t1)/(1-t2))
    5. Bootstrap standard errors

    Key assumptions:
    - Agents respond to incentives by bunching at kink
    - Counterfactual density is smooth (can be approximated by polynomial)
    - No other discontinuities in the data generating process

    References
    ----------
    - Saez (2010) - Original methodology
    - Chetty et al. (2011) - Integration constraint, optimization frictions
    - Kleven (2016) - Review and best practices
    """
    # Input validation
    if len(data) == 0:
        raise ValueError("data cannot be empty")

    if not np.isfinite(data).all():
        raise ValueError("data contains non-finite values (NaN or Inf)")

    if bunching_width <= 0:
        raise ValueError(f"bunching_width must be positive (got {bunching_width})")

    # Estimate counterfactual
    counterfactual_result = estimate_counterfactual(
        data=data,
        kink_point=kink_point,
        bunching_width=bunching_width,
        n_bins=n_bins,
        bin_width=bin_width,
        polynomial_order=polynomial_order,
    )

    # Compute excess mass
    excess_mass, excess_count, h0 = compute_excess_mass(counterfactual_result)

    # Compute elasticity if rates provided
    if t1_rate is not None and t2_rate is not None:
        try:
            elasticity = compute_elasticity(excess_mass, t1_rate, t2_rate)
        except ValueError as e:
            elasticity = np.nan
            message = f"Elasticity calculation failed: {e}"
    else:
        elasticity = np.nan
        message = "No tax rates provided; elasticity not computed"

    # Bootstrap standard errors
    try:
        excess_mass_se, _, elasticity_se, _ = bootstrap_bunching_se(
            data=data,
            kink_point=kink_point,
            bunching_width=bunching_width,
            n_bins=n_bins,
            bin_width=bin_width,
            polynomial_order=polynomial_order,
            t1_rate=t1_rate,
            t2_rate=t2_rate,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        convergence = True
        message = "Estimation successful"
    except ValueError as e:
        excess_mass_se = np.nan
        elasticity_se = np.nan
        convergence = False
        message = f"Bootstrap failed: {e}"

    return BunchingResult(
        excess_mass=excess_mass,
        excess_mass_se=excess_mass_se,
        excess_mass_count=excess_count,
        elasticity=elasticity,
        elasticity_se=elasticity_se,
        kink_point=kink_point,
        bunching_region=counterfactual_result["bunching_region"],
        counterfactual=counterfactual_result,
        t1_rate=t1_rate,
        t2_rate=t2_rate,
        n_obs=len(data),
        n_bootstrap=n_bootstrap,
        convergence=convergence,
        message=message,
    )
