"""
Sharp Regression Kink Design (RKD) Estimator

Implements local polynomial regression at a kink point for causal effect
estimation when treatment intensity changes slope at a threshold.

Key Insight:
-----------
RKD exploits kinks (slope changes) rather than jumps (level changes) in
the relationship between a running variable and treatment assignment.

Model:
    Left (x < c):  Y = α_L + β_L*(x - c) + γ_L*(x - c)² + ε_L
    Right (x ≥ c): Y = α_R + β_R*(x - c) + γ_R*(x - c)² + ε_R

    D = f(x) with kink at c: slope changes from δ_L to δ_R

Treatment effect: τ = (β_R - β_L) / (δ_R - δ_L)
                    = Δslope(Y) / Δslope(D)

References
----------
Card, D., Lee, D. S., Pei, Z., & Weber, A. (2015). Inference on causal
    effects in a generalized regression kink design. Econometrica, 83(6).
"""

from dataclasses import dataclass
from typing import Optional, Literal, Union

import numpy as np
from scipy import stats


@dataclass
class SharpRKDResult:
    """
    Result container for Sharp RKD estimation.

    Attributes
    ----------
    estimate : float
        RKD treatment effect estimate (τ)
    se : float
        Standard error of the estimate
    t_stat : float
        T-statistic
    p_value : float
        Two-sided p-value
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval
    bandwidth : float
        Bandwidth used for estimation
    n_left : int
        Effective sample size on left of cutoff
    n_right : int
        Effective sample size on right of cutoff
    slope_y_left : float
        Estimated slope of Y on left side
    slope_y_right : float
        Estimated slope of Y on right side
    slope_d_left : float
        Slope of D on left side (from kink specification)
    slope_d_right : float
        Slope of D on right side (from kink specification)
    delta_slope_y : float
        Change in Y slope at kink (β_R - β_L)
    delta_slope_d : float
        Change in D slope at kink (δ_R - δ_L)
    alpha : float
        Significance level used
    retcode : str
        Return code ('success', 'warning', 'error')
    message : str
        Descriptive message about estimation
    """

    estimate: float
    se: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    bandwidth: float
    n_left: int
    n_right: int
    slope_y_left: float
    slope_y_right: float
    slope_d_left: float
    slope_d_right: float
    delta_slope_y: float
    delta_slope_d: float
    alpha: float
    retcode: str
    message: str


class SharpRKD:
    """
    Sharp Regression Kink Design estimator.

    Uses local polynomial regression on each side of the cutoff to estimate
    the change in slope, then divides by the known kink in the policy.

    The key identification assumption is that all other determinants of the
    outcome vary smoothly at the kink, so any change in the outcome slope
    must be caused by the change in treatment slope.

    Parameters
    ----------
    cutoff : float
        Threshold value where treatment intensity slope changes
    bandwidth : float or str, default='auto'
        Bandwidth for local polynomial regression
        - float: Use specified bandwidth
        - 'auto': Automatic optimal bandwidth selection
    kernel : {'triangular', 'rectangular', 'epanechnikov'}, default='triangular'
        Kernel function for weighting observations
    polynomial_order : int, default=2
        Order of local polynomial (1=linear, 2=quadratic)
        Note: RKD typically requires at least order 2
    alpha : float, default=0.05
        Significance level for confidence intervals

    Attributes
    ----------
    cutoff : float
        The kink point
    bandwidth_ : float
        Bandwidth used for estimation
    result_ : SharpRKDResult
        Full estimation results

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.rkd import SharpRKD
    >>>
    >>> # Generate kink data
    >>> np.random.seed(42)
    >>> n = 1000
    >>> X = np.random.uniform(-5, 5, n)
    >>> # Policy with kink: slope changes from 0.5 to 1.5 at X=0
    >>> D = np.where(X < 0, 0.5 * X, 1.5 * X)
    >>> # Outcome with treatment effect = 2.0
    >>> Y = 2.0 * D + 0.3 * X + np.random.normal(0, 1, n)
    >>>
    >>> # Fit RKD
    >>> rkd = SharpRKD(cutoff=0.0, bandwidth=2.0)
    >>> result = rkd.fit(Y, X, D)
    >>> print(f"Estimated effect: {result.estimate:.3f}")
    Estimated effect: 2.000

    Notes
    -----
    Sharp RKD requires:
    1. Known kink in policy/treatment variable D at cutoff
    2. Smooth behavior of all other determinants at cutoff
    3. No manipulation of running variable around cutoff

    The estimator uses:
    - Local polynomial regression to estimate slopes on each side
    - Triangular kernel weighting (default)
    - Delta method for standard errors
    """

    def __init__(
        self,
        cutoff: float,
        bandwidth: Union[float, str] = "auto",
        kernel: Literal["triangular", "rectangular", "epanechnikov"] = "triangular",
        polynomial_order: int = 2,
        alpha: float = 0.05,
    ):
        self.cutoff = cutoff
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.polynomial_order = polynomial_order
        self.alpha = alpha

        # Will be set after fitting
        self.bandwidth_: Optional[float] = None
        self.result_: Optional[SharpRKDResult] = None

    def _compute_kernel_weights(self, x: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute kernel weights based on distance from cutoff."""
        u = (x - self.cutoff) / bandwidth

        if self.kernel == "triangular":
            weights = np.maximum(1 - np.abs(u), 0)
        elif self.kernel == "rectangular":
            weights = (np.abs(u) <= 1).astype(float)
        elif self.kernel == "epanechnikov":
            weights = np.maximum(0.75 * (1 - u**2), 0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        return weights

    def _fit_local_polynomial(
        self,
        y: np.ndarray,
        x: np.ndarray,
        weights: np.ndarray,
        order: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Fit weighted local polynomial regression.

        Returns
        -------
        coefficients : np.ndarray
            Polynomial coefficients [intercept, slope, quadratic, ...]
        vcov : np.ndarray
            Variance-covariance matrix of coefficients
        residual_variance : float
            Estimated residual variance
        """
        # Center x at cutoff
        x_centered = x - self.cutoff

        # Build design matrix [1, x, x², ...]
        n = len(x)
        X_design = np.column_stack([x_centered**p for p in range(order + 1)])

        # Weighted least squares
        W = np.diag(weights)
        XtWX = X_design.T @ W @ X_design
        XtWy = X_design.T @ W @ y

        try:
            XtWX_inv = np.linalg.inv(XtWX)
            coef = XtWX_inv @ XtWy
        except np.linalg.LinAlgError:
            # Use pseudo-inverse for singular matrices
            XtWX_inv = np.linalg.pinv(XtWX)
            coef = XtWX_inv @ XtWy

        # Residuals and variance
        residuals = y - X_design @ coef
        weighted_residuals = weights * residuals**2

        # Degrees of freedom: effective observations - parameters
        df = np.sum(weights > 0) - (order + 1)
        if df <= 0:
            df = 1  # Avoid division by zero

        residual_var = np.sum(weighted_residuals) / df

        # Robust (sandwich) variance-covariance
        # V = (X'WX)^{-1} X'W diag(e²) WX (X'WX)^{-1}
        meat = X_design.T @ W @ np.diag(residuals**2) @ W @ X_design
        vcov = XtWX_inv @ meat @ XtWX_inv

        return coef, vcov, residual_var

    def _estimate_kink_slopes(self, d: np.ndarray, x: np.ndarray) -> tuple[float, float]:
        """
        Estimate the slopes of D on each side of the cutoff.

        For Sharp RKD, D is deterministic given X, so we can estimate
        the slopes directly from the data or they can be specified.

        Returns
        -------
        slope_left : float
            Slope of D for X < cutoff
        slope_right : float
            Slope of D for X >= cutoff
        """
        # Left side
        left_mask = x < self.cutoff
        if np.sum(left_mask) >= 2:
            x_left = x[left_mask] - self.cutoff
            d_left = d[left_mask]
            # Simple OLS for slope
            slope_left = np.sum(x_left * d_left) / np.sum(x_left**2)
        else:
            slope_left = 0.0

        # Right side
        right_mask = x >= self.cutoff
        if np.sum(right_mask) >= 2:
            x_right = x[right_mask] - self.cutoff
            d_right = d[right_mask]
            slope_right = np.sum(x_right * d_right) / np.sum(x_right**2)
        else:
            slope_right = 0.0

        return slope_left, slope_right

    def _select_bandwidth(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Select optimal bandwidth for RKD.

        Uses a rule-of-thumb based on IK bandwidth adapted for kink designs.
        For RKD, we need a wider bandwidth than RDD because we're estimating
        slopes rather than levels.
        """
        from .bandwidth import rkd_ik_bandwidth

        return rkd_ik_bandwidth(y, x, self.cutoff)

    def fit(
        self,
        y: np.ndarray,
        x: np.ndarray,
        d: np.ndarray,
        slope_d_left: Optional[float] = None,
        slope_d_right: Optional[float] = None,
    ) -> SharpRKDResult:
        """
        Fit the Sharp RKD estimator.

        Parameters
        ----------
        y : array-like, shape (n,)
            Outcome variable
        x : array-like, shape (n,)
            Running variable (determines treatment via kink)
        d : array-like, shape (n,)
            Treatment intensity variable (with kink at cutoff)
        slope_d_left : float, optional
            Known slope of D for X < cutoff. If None, estimated from data.
        slope_d_right : float, optional
            Known slope of D for X >= cutoff. If None, estimated from data.

        Returns
        -------
        SharpRKDResult
            Estimation results including point estimate, SE, CI, etc.

        Raises
        ------
        ValueError
            If inputs are invalid or kink is too small to identify effect
        """
        # Validate inputs
        y = np.asarray(y).flatten()
        x = np.asarray(x).flatten()
        d = np.asarray(d).flatten()

        if len(y) != len(x) or len(y) != len(d):
            raise ValueError(f"Input length mismatch: y({len(y)}), x({len(x)}), d({len(d)})")

        if len(y) < 10:
            raise ValueError(f"Insufficient observations: {len(y)} < 10")

        if not np.isfinite(y).all():
            raise ValueError("Outcome y contains non-finite values")
        if not np.isfinite(x).all():
            raise ValueError("Running variable x contains non-finite values")
        if not np.isfinite(d).all():
            raise ValueError("Treatment d contains non-finite values")

        # Select bandwidth
        if isinstance(self.bandwidth, str):
            if self.bandwidth == "auto":
                self.bandwidth_ = self._select_bandwidth(y, x)
            else:
                raise ValueError(f"Unknown bandwidth method: {self.bandwidth}")
        else:
            self.bandwidth_ = float(self.bandwidth)

        # Estimate D slopes if not provided
        # BUG-3 FIX: Track whether D slopes are estimated (need variance) or known
        d_slopes_estimated = slope_d_left is None or slope_d_right is None

        if d_slopes_estimated:
            est_slope_left, est_slope_right = self._estimate_kink_slopes(d, x)
            slope_d_left = slope_d_left if slope_d_left is not None else est_slope_left
            slope_d_right = slope_d_right if slope_d_right is not None else est_slope_right

        delta_slope_d = slope_d_right - slope_d_left
        # Store flag for later use in SE computation
        self._d_slopes_estimated = d_slopes_estimated

        # Check kink magnitude
        if abs(delta_slope_d) < 1e-10:
            return SharpRKDResult(
                estimate=np.nan,
                se=np.nan,
                t_stat=np.nan,
                p_value=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                bandwidth=self.bandwidth_,
                n_left=0,
                n_right=0,
                slope_y_left=np.nan,
                slope_y_right=np.nan,
                slope_d_left=slope_d_left,
                slope_d_right=slope_d_right,
                delta_slope_y=np.nan,
                delta_slope_d=delta_slope_d,
                alpha=self.alpha,
                retcode="error",
                message="No kink detected in treatment variable (Δslope ≈ 0)",
            )

        # Compute kernel weights
        weights = self._compute_kernel_weights(x, self.bandwidth_)

        # Split by cutoff
        left_mask = (x < self.cutoff) & (weights > 0)
        right_mask = (x >= self.cutoff) & (weights > 0)

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        min_obs = self.polynomial_order + 2
        if n_left < min_obs or n_right < min_obs:
            return SharpRKDResult(
                estimate=np.nan,
                se=np.nan,
                t_stat=np.nan,
                p_value=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                bandwidth=self.bandwidth_,
                n_left=n_left,
                n_right=n_right,
                slope_y_left=np.nan,
                slope_y_right=np.nan,
                slope_d_left=slope_d_left,
                slope_d_right=slope_d_right,
                delta_slope_y=np.nan,
                delta_slope_d=delta_slope_d,
                alpha=self.alpha,
                retcode="error",
                message=f"Insufficient observations: left={n_left}, right={n_right}, need {min_obs}",
            )

        # Fit local polynomial on each side for Y
        coef_y_left, vcov_y_left, _ = self._fit_local_polynomial(
            y[left_mask], x[left_mask], weights[left_mask], self.polynomial_order
        )
        coef_y_right, vcov_y_right, _ = self._fit_local_polynomial(
            y[right_mask], x[right_mask], weights[right_mask], self.polynomial_order
        )

        # Extract Y slopes (coefficient on linear term, index 1)
        slope_y_left = coef_y_left[1]
        slope_y_right = coef_y_right[1]
        delta_slope_y = slope_y_right - slope_y_left

        # RKD estimate: ratio of slope changes
        estimate = delta_slope_y / delta_slope_d

        # BUG-3 FIX: Standard error via full delta method for ratio
        # For τ = Δβ_Y / Δδ_D, the delta method variance is:
        #   Var(τ) = [Var(Δβ_Y) + τ²·Var(Δδ_D)] / Δδ_D²
        # This accounts for uncertainty in both numerator AND denominator.

        # Y slope variances
        var_slope_y_left = vcov_y_left[1, 1]
        var_slope_y_right = vcov_y_right[1, 1]
        var_delta_slope_y = var_slope_y_left + var_slope_y_right

        # D slope variances (only if D slopes were estimated, not known)
        if self._d_slopes_estimated:
            # Fit local polynomial for D to get slope variances
            coef_d_left, vcov_d_left, _ = self._fit_local_polynomial(
                d[left_mask], x[left_mask], weights[left_mask], self.polynomial_order
            )
            coef_d_right, vcov_d_right, _ = self._fit_local_polynomial(
                d[right_mask], x[right_mask], weights[right_mask], self.polynomial_order
            )
            var_slope_d_left = vcov_d_left[1, 1]
            var_slope_d_right = vcov_d_right[1, 1]
            var_delta_slope_d = var_slope_d_left + var_slope_d_right
        else:
            # D slopes were provided (known), variance = 0
            var_delta_slope_d = 0.0

        # Full delta method variance for ratio
        var_estimate = (var_delta_slope_y + estimate**2 * var_delta_slope_d) / delta_slope_d**2
        se = np.sqrt(var_estimate)

        # Inference
        if se > 0:
            t_stat = estimate / se
            # Use t-distribution with effective df
            df = n_left + n_right - 2 * (self.polynomial_order + 1)
            df = max(df, 1)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            t_crit = stats.t.ppf(1 - self.alpha / 2, df)
            ci_lower = estimate - t_crit * se
            ci_upper = estimate + t_crit * se
        else:
            t_stat = np.inf if estimate != 0 else 0
            p_value = 0.0 if estimate != 0 else 1.0
            ci_lower = estimate
            ci_upper = estimate

        # Determine return code
        retcode = "success"
        message = "Estimation completed successfully"

        if n_left < 30 or n_right < 30:
            retcode = "warning"
            message = f"Small sample warning: left={n_left}, right={n_right}"

        self.result_ = SharpRKDResult(
            estimate=float(estimate),
            se=float(se),
            t_stat=float(t_stat),
            p_value=float(p_value),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            bandwidth=float(self.bandwidth_),
            n_left=int(n_left),
            n_right=int(n_right),
            slope_y_left=float(slope_y_left),
            slope_y_right=float(slope_y_right),
            slope_d_left=float(slope_d_left),
            slope_d_right=float(slope_d_right),
            delta_slope_y=float(delta_slope_y),
            delta_slope_d=float(delta_slope_d),
            alpha=float(self.alpha),
            retcode=retcode,
            message=message,
        )

        return self.result_

    def summary(self) -> str:
        """Return a formatted summary of the estimation results."""
        if self.result_ is None:
            return "Model not yet fitted. Call fit() first."

        r = self.result_
        lines = [
            "=" * 60,
            "Sharp Regression Kink Design Results",
            "=" * 60,
            f"Cutoff:           {self.cutoff:.4f}",
            f"Bandwidth:        {r.bandwidth:.4f}",
            f"Polynomial order: {self.polynomial_order}",
            f"Kernel:           {self.kernel}",
            "-" * 60,
            f"Estimate (τ):     {r.estimate:.4f}",
            f"Std. Error:       {r.se:.4f}",
            f"t-statistic:      {r.t_stat:.4f}",
            f"p-value:          {r.p_value:.4f}",
            f"95% CI:           [{r.ci_lower:.4f}, {r.ci_upper:.4f}]",
            "-" * 60,
            "Slope Estimates:",
            f"  Y slope (left):  {r.slope_y_left:.4f}",
            f"  Y slope (right): {r.slope_y_right:.4f}",
            f"  ΔY slope:        {r.delta_slope_y:.4f}",
            f"  D slope (left):  {r.slope_d_left:.4f}",
            f"  D slope (right): {r.slope_d_right:.4f}",
            f"  ΔD slope:        {r.delta_slope_d:.4f}",
            "-" * 60,
            f"N (left):         {r.n_left}",
            f"N (right):        {r.n_right}",
            f"Status:           {r.retcode}",
            f"Message:          {r.message}",
            "=" * 60,
        ]
        return "\n".join(lines)
