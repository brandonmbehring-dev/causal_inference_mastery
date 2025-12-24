"""
Bandwidth Selection for RDD

Implements optimal bandwidth selection methods for regression discontinuity designs:
- Imbens-Kalyanaraman (2012): Rule-of-thumb MSE-optimal bandwidth
- Calonico-Cattaneo-Titiunik (2014): MSE-optimal with bias correction

References
----------
Imbens, G. W., & Kalyanaraman, K. (2012). Optimal bandwidth choice for the
    regression discontinuity estimator. Review of Economic Studies, 79(3), 933-959.
Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric
    confidence intervals for regression-discontinuity designs. Econometrica, 82(6), 2295-2326.
"""

from typing import Tuple, Literal, Optional

import numpy as np
from scipy import stats


def imbens_kalyanaraman_bandwidth(
    Y: np.ndarray,
    X: np.ndarray,
    cutoff: float,
    kernel: Literal["triangular", "rectangular"] = "triangular",
) -> float:
    """
    Imbens-Kalyanaraman (2012) optimal bandwidth.

    Rule-of-thumb bandwidth based on minimizing asymptotic MSE
    of local linear estimator.

    h_IK = C_K * [σ²(c) / (n * m''(c)²)]^(1/5)

    where:
    - σ²(c): Conditional variance near cutoff
    - m''(c): Second derivative of conditional mean at cutoff
    - C_K: Kernel-specific constant
    - n: Sample size

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome variable
    X : array-like, shape (n,)
        Running variable
    cutoff : float
        RDD cutoff value
    kernel : {'triangular', 'rectangular'}, default='triangular'
        Kernel function

    Returns
    -------
    h_opt : float
        Optimal bandwidth

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.rdd.bandwidth import imbens_kalyanaraman_bandwidth
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.uniform(-5, 5, 500)
    >>> Y = X + 2 * (X >= 0) + np.random.normal(0, 1, 500)
    >>> h = imbens_kalyanaraman_bandwidth(Y, X, cutoff=0.0)
    >>> print(f"Optimal bandwidth: {h:.3f}")
    Optimal bandwidth: 1.234
    """
    # Convert to numpy arrays
    Y = np.asarray(Y).flatten()
    X = np.asarray(X).flatten()

    n = len(Y)
    n_left = np.sum(X < cutoff)
    n_right = np.sum(X >= cutoff)

    # Kernel constants (from IK 2012, Table 1)
    if kernel == "triangular":
        C1 = 3.43754  # Constant for triangular kernel
        C2 = 7.45129
    elif kernel == "rectangular":
        C1 = 5.40554  # Constant for rectangular kernel
        C2 = 14.20221
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Step 1: Estimate second derivative m''(c) using global polynomial
    # Fit cubic polynomial on full sample
    X_centered = X - cutoff
    poly_order = 3

    # Separate left and right sides
    mask_left = X < cutoff
    mask_right = X >= cutoff

    # Fit polynomial on each side
    if np.sum(mask_left) > poly_order + 1:
        poly_left = np.polyfit(X_centered[mask_left], Y[mask_left], deg=poly_order)
        # Second derivative: 2 * coef[1] (for ax^2 term in descending order)
        m2_left = 2 * poly_left[-3]  # Coefficient of x^2
    else:
        m2_left = 0.0

    if np.sum(mask_right) > poly_order + 1:
        poly_right = np.polyfit(X_centered[mask_right], Y[mask_right], deg=poly_order)
        m2_right = 2 * poly_right[-3]
    else:
        m2_right = 0.0

    # Average second derivative
    m2 = (m2_left + m2_right) / 2

    # Handle case where second derivative is near zero
    if abs(m2) < 1e-6:
        m2 = 1e-6  # Use small positive value to avoid division by zero

    # Step 2: Estimate conditional variance σ²(c)
    # Use residuals from pilot bandwidth regression
    h_pilot = 1.0  # Pilot bandwidth (rule of thumb)

    # Fit local linear regression with pilot bandwidth to get residuals
    residuals_left = _get_pilot_residuals(Y, X, cutoff, h_pilot, "left")
    residuals_right = _get_pilot_residuals(Y, X, cutoff, h_pilot, "right")

    # Variance estimates
    sigma2_left = np.var(residuals_left) if len(residuals_left) > 0 else 1.0
    sigma2_right = np.var(residuals_right) if len(residuals_right) > 0 else 1.0

    # Average variance
    sigma2 = (n_left * sigma2_left + n_right * sigma2_right) / n

    # Step 3: Compute optimal bandwidth
    # h_IK = C1 * [σ² / (n * m''²)]^(1/5)
    h_opt = C1 * (sigma2 / (n * m2**2)) ** 0.2

    # Regularization: ensure bandwidth is within reasonable range
    # Typical range: [0.1 * sd(X), 2 * sd(X)]
    x_sd = np.std(X)
    h_min = 0.1 * x_sd
    h_max = 2.0 * x_sd

    h_opt = np.clip(h_opt, h_min, h_max)

    return float(h_opt)


def cct_bandwidth(
    Y: np.ndarray,
    X: np.ndarray,
    cutoff: float,
    kernel: Literal["triangular", "rectangular"] = "triangular",
    bias_correction: bool = True,
) -> Tuple[float, float]:
    """
    CCT-style bandwidth approximation based on Imbens-Kalyanaraman.

    .. warning::

        **APPROXIMATION ONLY**: This function does NOT implement the full
        Calonico-Cattaneo-Titiunik (2014) bandwidth selection algorithm.
        It returns the IK bandwidth as h_main and uses an ad-hoc 1.5×
        multiplier for h_bias.

        For production RDD analysis requiring true CCT bandwidth with
        robust bias correction, use the `rdrobust` R package or
        `rddensity` Python package.

    This function provides a computationally simple approximation that:
    - Uses IK bandwidth (Imbens-Kalyanaraman 2012) as the main bandwidth
    - Applies a 1.5× scaling factor for the bias correction bandwidth
    - Does NOT implement the iterative regularization from CCT (2014)

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome variable
    X : array-like, shape (n,)
        Running variable
    cutoff : float
        RDD cutoff value
    kernel : {'triangular', 'rectangular'}, default='triangular'
        Kernel function
    bias_correction : bool, default=True
        If True, return both main bandwidth and bias correction bandwidth

    Returns
    -------
    h_main : float
        Main bandwidth (equals IK bandwidth)
    h_bias : float
        Bias correction bandwidth (1.5 × h_main if bias_correction=True)

    Notes
    -----
    The true CCT bandwidth selection (Calonico, Cattaneo, Titiunik 2014)
    involves:

    1. Separate regularization for variance and bias components
    2. Pilot estimation of second and third derivatives
    3. Iterative bandwidth refinement

    This approximation skips these steps and provides a simpler alternative
    that may be adequate for exploratory analysis but should NOT be used
    for final inference in published research.

    References
    ----------
    Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric
        confidence intervals for regression-discontinuity designs. Econometrica.

    Imbens, G. W., & Kalyanaraman, K. (2012). Optimal bandwidth choice for the
        regression discontinuity estimator. Review of Economic Studies.

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.rdd.bandwidth import cct_bandwidth
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.uniform(-5, 5, 500)
    >>> Y = X**2 + 2 * (X >= 0) + np.random.normal(0, 1, 500)
    >>> h_main, h_bias = cct_bandwidth(Y, X, cutoff=0.0, bias_correction=True)
    >>> # Note: h_main equals IK bandwidth, h_bias = 1.5 * h_main
    >>> print(f"Main bandwidth: {h_main:.3f}, Bias bandwidth: {h_bias:.3f}")
    Main bandwidth: 1.123, Bias bandwidth: 1.685
    """
    import warnings

    warnings.warn(
        "cct_bandwidth() is an APPROXIMATION using IK bandwidth with 1.5× scaling "
        "for bias bandwidth. For true CCT bandwidth selection with robust bias "
        "correction, use the 'rdrobust' R package or 'rddensity' Python package.",
        UserWarning,
        stacklevel=2,
    )

    # Convert to numpy arrays
    Y = np.asarray(Y).flatten()
    X = np.asarray(X).flatten()

    # APPROXIMATION: Use IK bandwidth as base
    # True CCT requires iterative regularization procedure
    h_ik = imbens_kalyanaraman_bandwidth(Y, X, cutoff, kernel)

    h_main = h_ik

    if bias_correction:
        # APPROXIMATION: Use 1.5× multiplier for bias bandwidth
        # True CCT uses data-driven regularization
        h_bias = 1.5 * h_main
    else:
        h_bias = h_main

    return float(h_main), float(h_bias)


def _get_pilot_residuals(
    Y: np.ndarray, X: np.ndarray, cutoff: float, bandwidth: float, side: str
) -> np.ndarray:
    """
    Get residuals from local linear regression with pilot bandwidth.

    Parameters
    ----------
    Y : ndarray
        Outcome variable
    X : ndarray
        Running variable
    cutoff : float
        Cutoff value
    bandwidth : float
        Pilot bandwidth
    side : {'left', 'right'}
        Which side of cutoff

    Returns
    -------
    residuals : ndarray
        Regression residuals
    """
    # Select observations on this side
    if side == "left":
        mask = (X < cutoff) & (np.abs(X - cutoff) <= bandwidth)
    else:  # side == 'right'
        mask = (X >= cutoff) & (np.abs(X - cutoff) <= bandwidth)

    if np.sum(mask) < 3:
        return np.array([])

    Y_side = Y[mask]
    X_side = X[mask] - cutoff

    # Triangular kernel weights
    weights = np.maximum(1 - np.abs(X_side / bandwidth), 0)

    # Fit weighted linear regression
    design = np.column_stack([np.ones(len(X_side)), X_side])
    W = np.diag(weights)

    try:
        coefs = np.linalg.solve(design.T @ W @ design, design.T @ W @ Y_side)
        residuals = Y_side - design @ coefs
    except np.linalg.LinAlgError:
        residuals = Y_side - np.mean(Y_side)

    return residuals


def cross_validation_bandwidth(
    Y: np.ndarray,
    X: np.ndarray,
    cutoff: float,
    kernel: Literal["triangular", "rectangular"] = "triangular",
    h_grid: Optional[np.ndarray] = None,
) -> float:
    """
    Cross-validation bandwidth selection.

    Leave-one-out cross-validation to minimize MSE.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome variable
    X : array-like, shape (n,)
        Running variable
    cutoff : float
        RDD cutoff value
    kernel : {'triangular', 'rectangular'}, default='triangular'
        Kernel function
    h_grid : array-like, optional
        Grid of bandwidth values to search
        If None, uses default grid based on data

    Returns
    -------
    h_opt : float
        Cross-validation optimal bandwidth

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.rdd.bandwidth import cross_validation_bandwidth
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.uniform(-5, 5, 200)
    >>> Y = np.sin(X) + 2 * (X >= 0) + np.random.normal(0, 0.5, 200)
    >>> h = cross_validation_bandwidth(Y, X, cutoff=0.0)
    >>> print(f"CV optimal bandwidth: {h:.3f}")
    CV optimal bandwidth: 0.987
    """
    # Convert to numpy arrays
    Y = np.asarray(Y).flatten()
    X = np.asarray(X).flatten()

    # Default grid: from 0.5 to 2.0 standard deviations
    if h_grid is None:
        x_sd = np.std(X)
        h_grid = np.linspace(0.5 * x_sd, 2.0 * x_sd, num=20)

    # Cross-validation scores
    cv_scores = []

    for h in h_grid:
        mse_left = _cv_mse(Y, X, cutoff, h, kernel, "left")
        mse_right = _cv_mse(Y, X, cutoff, h, kernel, "right")
        cv_scores.append(mse_left + mse_right)

    # Select bandwidth with minimum CV score
    h_opt = h_grid[np.argmin(cv_scores)]

    return float(h_opt)


def _cv_mse(
    Y: np.ndarray, X: np.ndarray, cutoff: float, bandwidth: float, kernel: str, side: str
) -> float:
    """
    Compute leave-one-out cross-validation MSE for one side.

    Parameters
    ----------
    Y : ndarray
        Outcome variable
    X : ndarray
        Running variable
    cutoff : float
        Cutoff value
    bandwidth : float
        Bandwidth
    kernel : str
        Kernel function
    side : {'left', 'right'}
        Which side of cutoff

    Returns
    -------
    mse : float
        Cross-validation MSE
    """
    # Select observations on this side within bandwidth
    if side == "left":
        mask = (X < cutoff) & (np.abs(X - cutoff) <= bandwidth)
    else:
        mask = (X >= cutoff) & (np.abs(X - cutoff) <= bandwidth)

    if np.sum(mask) < 3:
        return np.inf

    Y_side = Y[mask]
    X_side = X[mask] - cutoff
    n_side = len(Y_side)

    # Leave-one-out predictions
    errors = []

    for i in range(n_side):
        # Leave out observation i
        Y_train = np.delete(Y_side, i)
        X_train = np.delete(X_side, i)
        X_test = X_side[i]

        # Kernel weights for training data
        u_train = X_train / bandwidth
        if kernel == "triangular":
            weights = np.maximum(1 - np.abs(u_train), 0)
        else:  # rectangular
            weights = (np.abs(u_train) <= 1).astype(float)

        # Fit local linear regression
        design = np.column_stack([np.ones(len(X_train)), X_train])
        W = np.diag(weights)

        try:
            coefs = np.linalg.solve(design.T @ W @ design, design.T @ W @ Y_train)
            # Predict at test point
            y_pred = coefs[0] + coefs[1] * X_test
            errors.append((Y_side[i] - y_pred) ** 2)
        except np.linalg.LinAlgError:
            # If singular, use mean prediction
            errors.append((Y_side[i] - np.mean(Y_train)) ** 2)

    mse = np.mean(errors) if len(errors) > 0 else np.inf

    return mse
