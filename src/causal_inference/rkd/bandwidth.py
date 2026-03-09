"""
Bandwidth Selection for Regression Kink Design (RKD)

Implements optimal bandwidth selectors adapted for kink designs.
RKD requires larger bandwidths than RDD because we estimate slopes
(first derivatives) rather than levels (function values).

Key Insight:
-----------
For local polynomial regression of order p:
- Estimating level at boundary: optimal bandwidth ~ n^(-1/(2p+3))
- Estimating 1st derivative: optimal bandwidth ~ n^(-1/(2p+5))

For RKD with local quadratic (p=2):
- Level (RDD): h ~ n^(-1/7) ≈ n^(-0.143)
- Slope (RKD): h ~ n^(-1/9) ≈ n^(-0.111)

Thus RKD uses a slightly larger bandwidth.

References
----------
Card, D., Lee, D. S., Pei, Z., & Weber, A. (2015). Inference on causal
    effects in a generalized regression kink design. Econometrica, 83(6).
Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric
    confidence intervals for regression-discontinuity designs.
"""

import warnings
from typing import Optional

import numpy as np


def rkd_ik_bandwidth(
    y: np.ndarray,
    x: np.ndarray,
    cutoff: float,
    kernel: str = "triangular",
) -> float:
    """
    Imbens-Kalyanaraman style bandwidth adapted for RKD.

    This adapts the IK bandwidth selector for RDD to account for the
    different convergence rate when estimating derivatives.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome variable
    x : array-like, shape (n,)
        Running variable
    cutoff : float
        Kink point
    kernel : str, default='triangular'
        Kernel function

    Returns
    -------
    float
        Optimal bandwidth for RKD

    Notes
    -----
    The formula adapts IK (2012) for derivative estimation:

    h_RKD = C_K * σ * n^(-1/9) * [f(c) / (f''(c))²]^(1/9)

    where:
    - C_K is the kernel constant
    - σ is the residual standard deviation
    - n is sample size
    - f(c) is density at cutoff
    - f''(c) is curvature of regression function

    For practical implementation, we use a pilot bandwidth approach.
    """
    y = np.asarray(y).flatten()
    x = np.asarray(x).flatten()
    n = len(y)

    if n < 20:
        # Fallback for very small samples
        x_range = np.ptp(x)
        return x_range / 4

    # Kernel constant (for triangular)
    if kernel == "triangular":
        C_K = 3.43  # Adjusted for derivative estimation
    elif kernel == "rectangular":
        C_K = 2.70
    else:
        C_K = 3.43

    # Step 1: Pilot bandwidth using Silverman's rule
    sigma_x = np.std(x)
    iqr_x = np.percentile(x, 75) - np.percentile(x, 25)
    h_pilot = 1.06 * min(sigma_x, iqr_x / 1.34) * n ** (-0.2)

    # Step 2: Estimate density at cutoff
    left_mask = (x >= cutoff - h_pilot) & (x < cutoff)
    right_mask = (x >= cutoff) & (x <= cutoff + h_pilot)

    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)

    if n_left < 5 or n_right < 5:
        # Not enough data near cutoff, use larger bandwidth
        warnings.warn(
            f"Small sample near cutoff (n_left={n_left}, n_right={n_right}). "
            "Using larger default bandwidth."
        )
        return h_pilot * 2

    # Density estimate (using histogram approach)
    f_c = (n_left + n_right) / (2 * h_pilot * n)

    # Step 3: Estimate curvature using global polynomial
    # Fit quadratic on each side to estimate second derivative
    x_centered = x - cutoff

    # Left side quadratic
    left_global = x < cutoff
    if np.sum(left_global) >= 3:
        try:
            poly_left = np.polyfit(x_centered[left_global], y[left_global], 2)
            curv_left = 2 * poly_left[0]  # Second derivative of quadratic
        except (np.linalg.LinAlgError, ValueError):
            curv_left = 0.1
    else:
        curv_left = 0.1

    # Right side quadratic
    right_global = x >= cutoff
    if np.sum(right_global) >= 3:
        try:
            poly_right = np.polyfit(x_centered[right_global], y[right_global], 2)
            curv_right = 2 * poly_right[0]
        except (np.linalg.LinAlgError, ValueError):
            curv_right = 0.1
    else:
        curv_right = 0.1

    # Average curvature
    curv = (abs(curv_left) + abs(curv_right)) / 2
    curv = max(curv, 0.01)  # Avoid division by very small numbers

    # Step 4: Estimate residual variance
    # Use residuals from quadratic fit
    if np.sum(left_global) >= 3:
        y_pred_left = np.polyval(poly_left, x_centered[left_global])
        resid_left = y[left_global] - y_pred_left
        sigma_left = np.std(resid_left)
    else:
        sigma_left = np.std(y)

    if np.sum(right_global) >= 3:
        y_pred_right = np.polyval(poly_right, x_centered[right_global])
        resid_right = y[right_global] - y_pred_right
        sigma_right = np.std(resid_right)
    else:
        sigma_right = np.std(y)

    sigma = (sigma_left + sigma_right) / 2

    # Step 5: Compute optimal bandwidth for RKD
    # h_RKD ~ C_K * sigma * n^(-1/9) * [f(c) / curv²]^(1/9)
    #
    # The n^(-1/9) rate is slower than RDD's n^(-1/5) or n^(-1/7)
    # because we're estimating derivatives

    if f_c > 0 and curv > 0:
        h_opt = C_K * sigma * (n ** (-1 / 9)) * ((f_c / (curv**2)) ** (1 / 9))
    else:
        h_opt = h_pilot * 1.5

    # Bound the bandwidth
    x_range = np.ptp(x)
    h_min = x_range / 20
    h_max = x_range / 2

    h_opt = np.clip(h_opt, h_min, h_max)

    return float(h_opt)


def rkd_bandwidth(
    y: np.ndarray,
    x: np.ndarray,
    cutoff: float,
    method: str = "ik",
    kernel: str = "triangular",
) -> float:
    """
    Compute optimal bandwidth for Regression Kink Design.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome variable
    x : array-like, shape (n,)
        Running variable
    cutoff : float
        Kink point
    method : {'ik', 'rot'}, default='ik'
        Bandwidth selection method:
        - 'ik': Imbens-Kalyanaraman style (adapted for RKD)
        - 'rot': Rule of thumb based on Silverman
    kernel : str, default='triangular'
        Kernel function

    Returns
    -------
    float
        Optimal bandwidth

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.rkd import rkd_bandwidth
    >>>
    >>> np.random.seed(42)
    >>> x = np.random.uniform(-5, 5, 500)
    >>> y = x + 0.5 * x * (x >= 0) + np.random.normal(0, 1, 500)
    >>> h = rkd_bandwidth(y, x, cutoff=0.0)
    >>> print(f"Optimal bandwidth: {h:.3f}")
    """
    y = np.asarray(y).flatten()
    x = np.asarray(x).flatten()

    if method == "ik":
        return rkd_ik_bandwidth(y, x, cutoff, kernel)

    elif method == "rot":
        # Rule of thumb: Silverman with RKD rate adjustment
        n = len(x)
        sigma_x = np.std(x)
        iqr_x = np.percentile(x, 75) - np.percentile(x, 25)

        # Standard Silverman
        h_silverman = 1.06 * min(sigma_x, iqr_x / 1.34) * n ** (-0.2)

        # Adjust for RKD (derivative estimation needs larger bandwidth)
        # Ratio of rates: n^(-1/9) / n^(-1/5) = n^(4/45)
        adjustment = n ** (4 / 45)
        h_rkd = h_silverman * adjustment

        # Bound
        x_range = np.ptp(x)
        h_rkd = np.clip(h_rkd, x_range / 20, x_range / 2)

        return float(h_rkd)

    else:
        raise ValueError(f"Unknown bandwidth method: {method}. Use 'ik' or 'rot'.")


def rkd_cross_validation_bandwidth(
    y: np.ndarray,
    x: np.ndarray,
    cutoff: float,
    h_grid: Optional[np.ndarray] = None,
    polynomial_order: int = 2,
    kernel: str = "triangular",
) -> float:
    """
    Select bandwidth using leave-one-out cross-validation.

    This method selects the bandwidth that minimizes prediction error
    for the local polynomial fit, adapted for RKD.

    Parameters
    ----------
    y : array-like
        Outcome variable
    x : array-like
        Running variable
    cutoff : float
        Kink point
    h_grid : array-like, optional
        Grid of bandwidths to search. If None, uses automatic grid.
    polynomial_order : int, default=2
        Order of local polynomial
    kernel : str, default='triangular'
        Kernel function

    Returns
    -------
    float
        Cross-validated optimal bandwidth

    Notes
    -----
    CV for RKD is computationally expensive and may overfit.
    Prefer IK bandwidth for most applications.
    """
    y = np.asarray(y).flatten()
    x = np.asarray(x).flatten()
    n = len(y)

    # Create bandwidth grid if not provided
    if h_grid is None:
        h_pilot = rkd_ik_bandwidth(y, x, cutoff, kernel)
        h_grid = np.linspace(h_pilot * 0.5, h_pilot * 2.0, 10)

    def _kernel_weight(u: np.ndarray, kernel: str) -> np.ndarray:
        if kernel == "triangular":
            return np.maximum(1 - np.abs(u), 0)
        else:
            return (np.abs(u) <= 1).astype(float)

    def _cv_score(h: float) -> float:
        """Compute CV score for a given bandwidth."""
        cv_errors = []

        for i in range(n):
            # Leave one out
            x_train = np.delete(x, i)
            y_train = np.delete(y, i)
            x_test = x[i]
            y_test = y[i]

            # Determine side
            if x_test < cutoff:
                side_mask = x_train < cutoff
            else:
                side_mask = x_train >= cutoff

            x_side = x_train[side_mask]
            y_side = y_train[side_mask]

            # Weights
            u = (x_side - cutoff) / h
            w = _kernel_weight(u, kernel)

            if np.sum(w > 0) < polynomial_order + 1:
                continue

            # Fit local polynomial
            x_c = x_side - cutoff
            try:
                X_design = np.column_stack([x_c**p for p in range(polynomial_order + 1)])
                W = np.diag(w)
                coef = np.linalg.solve(X_design.T @ W @ X_design, X_design.T @ W @ y_side)

                # Predict
                x_test_c = x_test - cutoff
                y_pred = sum(coef[p] * x_test_c**p for p in range(polynomial_order + 1))
                cv_errors.append((y_test - y_pred) ** 2)
            except (np.linalg.LinAlgError, ValueError):
                continue

        if len(cv_errors) < n // 2:
            return np.inf

        return np.mean(cv_errors)

    # Find best bandwidth
    cv_scores = [_cv_score(h) for h in h_grid]
    best_idx = np.argmin(cv_scores)

    return float(h_grid[best_idx])
