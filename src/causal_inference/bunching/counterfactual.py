"""
Counterfactual density estimation for bunching analysis.

Implements polynomial fitting to estimate what the density would look like
in the absence of bunching behavior.

Key Insight:
The observed density near a kink shows "bunching" - excess mass where agents
cluster. The counterfactual density is what we'd observe without the kink,
estimated by fitting a polynomial to bins OUTSIDE the bunching region.

References:
- Saez (2010) - Original bunching methodology
- Chetty et al. (2011) - Refinements and integration constraint
- Kleven (2016) - Review and best practices
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from .types import CounterfactualResult


def polynomial_counterfactual(
    bin_centers: NDArray[np.float64],
    counts: NDArray[np.float64],
    bunching_lower: float,
    bunching_upper: float,
    polynomial_order: int = 7,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Fit polynomial counterfactual excluding bunching region.

    Parameters
    ----------
    bin_centers : NDArray[np.float64]
        Centers of histogram bins.
    counts : NDArray[np.float64]
        Observed counts in each bin.
    bunching_lower : float
        Lower bound of bunching region to exclude.
    bunching_upper : float
        Upper bound of bunching region to exclude.
    polynomial_order : int, default=7
        Order of polynomial to fit. Higher order allows more flexibility
        but risks overfitting. Saez (2010) uses 7.

    Returns
    -------
    counterfactual_counts : NDArray[np.float64]
        Predicted counterfactual counts for all bins.
    coeffs : NDArray[np.float64]
        Polynomial coefficients.
    r_squared : float
        R-squared of fit on non-bunching bins.

    Raises
    ------
    ValueError
        If insufficient bins outside bunching region for polynomial order.

    Notes
    -----
    The polynomial is fit to bins OUTSIDE the bunching region, then used
    to predict what counts WOULD be in the bunching region without bunching.
    """
    if len(bin_centers) != len(counts):
        raise ValueError(
            f"bin_centers and counts must have same length "
            f"(got {len(bin_centers)} and {len(counts)})"
        )

    if polynomial_order < 1:
        raise ValueError(f"polynomial_order must be >= 1 (got {polynomial_order})")

    # Identify bins outside bunching region
    outside_bunching = (bin_centers < bunching_lower) | (bin_centers > bunching_upper)
    n_outside = np.sum(outside_bunching)

    if n_outside < polynomial_order + 1:
        raise ValueError(
            f"Need at least {polynomial_order + 1} bins outside bunching region "
            f"for polynomial order {polynomial_order}, but only have {n_outside}"
        )

    # Fit polynomial to bins outside bunching region
    x_fit = bin_centers[outside_bunching]
    y_fit = counts[outside_bunching]

    # Center x for numerical stability
    x_mean = np.mean(bin_centers)
    x_centered = x_fit - x_mean

    # Fit polynomial
    coeffs = np.polyfit(x_centered, y_fit, polynomial_order)

    # Predict counterfactual for all bins
    all_x_centered = bin_centers - x_mean
    counterfactual = np.polyval(coeffs, all_x_centered)

    # Ensure non-negative (density can't be negative)
    counterfactual = np.maximum(counterfactual, 0)

    # Compute R-squared on fitting region
    y_pred_fit = np.polyval(coeffs, x_centered)
    ss_res = np.sum((y_fit - y_pred_fit) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return counterfactual, coeffs, r_squared


def estimate_counterfactual(
    data: NDArray[np.float64],
    kink_point: float,
    bunching_width: float,
    n_bins: Optional[int] = None,
    bin_width: Optional[float] = None,
    polynomial_order: int = 7,
    data_range: Optional[Tuple[float, float]] = None,
) -> CounterfactualResult:
    """Estimate counterfactual density for bunching analysis.

    This is the main interface for counterfactual estimation. It bins the data,
    identifies the bunching region, and fits a polynomial counterfactual.

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data (e.g., reported income values).
    kink_point : float
        Location of the kink (e.g., tax bracket threshold).
    bunching_width : float
        Half-width of bunching region around kink. Region is
        [kink_point - bunching_width, kink_point + bunching_width].
    n_bins : Optional[int], default=None
        Number of bins for histogram. If None, uses bin_width.
    bin_width : Optional[float], default=None
        Width of each bin. If None, uses n_bins. One must be specified.
    polynomial_order : int, default=7
        Order of polynomial for counterfactual. Higher order = more flexible.
    data_range : Optional[Tuple[float, float]], default=None
        (min, max) range for binning. If None, uses data range.

    Returns
    -------
    CounterfactualResult
        Dictionary containing:
        - bin_centers: Center of each bin
        - actual_counts: Observed counts
        - counterfactual_counts: Estimated counterfactual
        - polynomial_coeffs: Fitted coefficients
        - polynomial_order: Order used
        - bunching_region: (lower, upper) bounds
        - r_squared: Fit quality
        - n_bins: Number of bins
        - bin_width: Width of bins

    Raises
    ------
    ValueError
        If neither n_bins nor bin_width specified, or if data is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> # Simulate data with bunching at kink=50000
    >>> np.random.seed(42)
    >>> normal_data = np.random.normal(45000, 10000, 900)
    >>> bunching_data = np.random.normal(50000, 500, 100)  # Bunching
    >>> data = np.concatenate([normal_data, bunching_data])
    >>> result = estimate_counterfactual(
    ...     data, kink_point=50000, bunching_width=2000, n_bins=50
    ... )
    >>> print(f"R-squared: {result['r_squared']:.3f}")
    """
    # Input validation
    if len(data) == 0:
        raise ValueError("data cannot be empty")

    if not np.isfinite(data).all():
        raise ValueError("data contains non-finite values (NaN or Inf)")

    if n_bins is None and bin_width is None:
        raise ValueError("Must specify either n_bins or bin_width")

    if n_bins is not None and bin_width is not None:
        raise ValueError("Specify only one of n_bins or bin_width, not both")

    if bunching_width <= 0:
        raise ValueError(f"bunching_width must be positive (got {bunching_width})")

    # Determine data range
    if data_range is None:
        data_min, data_max = np.min(data), np.max(data)
    else:
        data_min, data_max = data_range

    # Create bins
    if n_bins is not None:
        if n_bins < 10:
            raise ValueError(f"n_bins must be >= 10 (got {n_bins})")
        bin_edges = np.linspace(data_min, data_max, n_bins + 1)
        actual_bin_width = (data_max - data_min) / n_bins
    else:
        if bin_width <= 0:
            raise ValueError(f"bin_width must be positive (got {bin_width})")
        bin_edges = np.arange(data_min, data_max + bin_width, bin_width)
        n_bins = len(bin_edges) - 1
        actual_bin_width = bin_width

    # Compute histogram
    counts, _ = np.histogram(data, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Define bunching region
    bunching_lower = kink_point - bunching_width
    bunching_upper = kink_point + bunching_width

    # Check that bunching region is within data range
    if bunching_lower < data_min or bunching_upper > data_max:
        # Adjust to data range
        bunching_lower = max(bunching_lower, data_min)
        bunching_upper = min(bunching_upper, data_max)

    # Fit polynomial counterfactual
    counterfactual, coeffs, r_squared = polynomial_counterfactual(
        bin_centers=bin_centers,
        counts=counts.astype(np.float64),
        bunching_lower=bunching_lower,
        bunching_upper=bunching_upper,
        polynomial_order=polynomial_order,
    )

    return CounterfactualResult(
        bin_centers=bin_centers,
        actual_counts=counts.astype(np.float64),
        counterfactual_counts=counterfactual,
        polynomial_coeffs=coeffs,
        polynomial_order=polynomial_order,
        bunching_region=(bunching_lower, bunching_upper),
        r_squared=r_squared,
        n_bins=n_bins,
        bin_width=actual_bin_width,
    )


def iterative_counterfactual(
    data: NDArray[np.float64],
    kink_point: float,
    initial_bunching_width: float,
    n_bins: Optional[int] = None,
    bin_width: Optional[float] = None,
    polynomial_order: int = 7,
    max_iterations: int = 20,
    tolerance: float = 1e-4,
) -> Tuple[CounterfactualResult, float, bool]:
    """Iteratively estimate counterfactual with integration constraint.

    The integration constraint (Chetty et al. 2011) requires that the area
    "missing" above the kink (where bunchers came from) equals the excess
    mass at the kink. This is solved iteratively.

    Parameters
    ----------
    data : NDArray[np.float64]
        Observed data.
    kink_point : float
        Location of the kink.
    initial_bunching_width : float
        Initial guess for bunching region half-width.
    n_bins : Optional[int]
        Number of bins.
    bin_width : Optional[float]
        Bin width (alternative to n_bins).
    polynomial_order : int, default=7
        Polynomial order for counterfactual.
    max_iterations : int, default=20
        Maximum iterations for convergence.
    tolerance : float, default=1e-4
        Convergence tolerance for delta_z (upper bound shift).

    Returns
    -------
    counterfactual_result : CounterfactualResult
        Final counterfactual estimation.
    delta_z : float
        Estimated upper bound of bunching region (shift from kink).
    converged : bool
        Whether iteration converged.

    Notes
    -----
    The integration constraint: bunchers at the kink came from the region
    just above the kink. The counterfactual density above the kink is
    "shifted down" to account for this missing mass.
    """
    # Initial estimate
    bunching_width = initial_bunching_width
    delta_z = bunching_width

    for iteration in range(max_iterations):
        # Estimate counterfactual with current bunching width
        result = estimate_counterfactual(
            data=data,
            kink_point=kink_point,
            bunching_width=bunching_width,
            n_bins=n_bins,
            bin_width=bin_width,
            polynomial_order=polynomial_order,
        )

        # Compute excess mass in bunching region
        bunching_mask = (result["bin_centers"] >= result["bunching_region"][0]) & (
            result["bin_centers"] <= result["bunching_region"][1]
        )
        excess = np.sum(
            result["actual_counts"][bunching_mask] - result["counterfactual_counts"][bunching_mask]
        )

        # Compute counterfactual height at kink
        kink_idx = np.argmin(np.abs(result["bin_centers"] - kink_point))
        h0 = result["counterfactual_counts"][kink_idx]

        if h0 <= 0:
            # Degenerate case
            break

        # New delta_z from integration constraint
        # B = h0 * delta_z  =>  delta_z = B / h0
        new_delta_z = excess / h0 if h0 > 0 else delta_z

        # Check convergence
        if abs(new_delta_z - delta_z) / max(abs(delta_z), 1e-10) < tolerance:
            return result, new_delta_z, True

        delta_z = new_delta_z
        bunching_width = max(initial_bunching_width, delta_z)

    # Did not converge
    return result, delta_z, False
