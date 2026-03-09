"""
Fuzzy Regression Kink Design (RKD) Estimator

Implements 2SLS-style estimation for fuzzy kinks where treatment is not
deterministically assigned based on the running variable, but there is
a kink in the probability/intensity of treatment at the threshold.

Key Insight:
-----------
Sharp RKD: D = f(X) with known kink (deterministic)
Fuzzy RKD: E[D|X] has a kink at cutoff (stochastic)

The Fuzzy RKD estimator uses local polynomial 2SLS:
    First stage:  Estimate Δslope(D) at cutoff
    Second stage: Estimate Δslope(Y) at cutoff
    Effect:       τ = Δslope(Y) / Δslope(D)

This identifies a Local Average Treatment Effect (LATE) for compliers -
units whose treatment status changes due to the kink.

References
----------
Card, D., Lee, D. S., Pei, Z., & Weber, A. (2015). Inference on causal
    effects in a generalized regression kink design. Econometrica, 83(6).
Dong, Y. (2018). Regression kink design: Theory and practice.
    Handbook of Regression Discontinuity Designs.
"""

from dataclasses import dataclass
from typing import Optional, Literal, Union

import numpy as np
from scipy import stats


@dataclass
class FuzzyRKDResult:
    """
    Result container for Fuzzy RKD estimation.

    Attributes
    ----------
    estimate : float
        Fuzzy RKD treatment effect estimate (LATE at kink)
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
    first_stage_slope_left : float
        Estimated slope of D on left side (first stage)
    first_stage_slope_right : float
        Estimated slope of D on right side (first stage)
    first_stage_kink : float
        Change in D slope at kink (Δslope_D)
    reduced_form_slope_left : float
        Estimated slope of Y on left side (reduced form)
    reduced_form_slope_right : float
        Estimated slope of Y on right side (reduced form)
    reduced_form_kink : float
        Change in Y slope at kink (Δslope_Y)
    first_stage_f_stat : float
        F-statistic for first stage strength
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
    first_stage_slope_left: float
    first_stage_slope_right: float
    first_stage_kink: float
    reduced_form_slope_left: float
    reduced_form_slope_right: float
    reduced_form_kink: float
    first_stage_f_stat: float
    alpha: float
    retcode: str
    message: str


class FuzzyRKD:
    """
    Fuzzy Regression Kink Design estimator.

    Uses local polynomial 2SLS to estimate the LATE at a kink point where
    treatment intensity changes slope but is not deterministic.

    The key identification assumption is that the change in the slope of
    E[D|X] at the cutoff is exogenous - units cannot precisely manipulate
    their position around the kink.

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
    alpha : float, default=0.05
        Significance level for confidence intervals

    Attributes
    ----------
    cutoff : float
        The kink point
    bandwidth_ : float
        Bandwidth used for estimation
    result_ : FuzzyRKDResult
        Full estimation results

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.rkd import FuzzyRKD
    >>>
    >>> # Generate fuzzy kink data
    >>> np.random.seed(42)
    >>> n = 1000
    >>> X = np.random.uniform(-5, 5, n)
    >>> # Treatment with fuzzy kink: expected slope changes
    >>> D_expected = np.where(X < 0, 0.5 * X, 1.5 * X)
    >>> D = D_expected + np.random.normal(0, 0.5, n)  # Add noise
    >>> Y = 2.0 * D + 0.3 * X + np.random.normal(0, 1, n)
    >>>
    >>> # Fit Fuzzy RKD
    >>> rkd = FuzzyRKD(cutoff=0.0, bandwidth=2.0)
    >>> result = rkd.fit(Y, X, D)
    >>> print(f"Estimated LATE: {result.estimate:.3f}")

    Notes
    -----
    Fuzzy RKD requires:
    1. A kink in E[D|X] at the cutoff (first stage)
    2. Smooth behavior of all other determinants at cutoff
    3. No manipulation of running variable around cutoff
    4. Monotonicity: the kink affects D in the same direction for all units

    The estimator identifies a LATE - the effect for "compliers" whose
    treatment changes due to the kink.
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
        self.result_: Optional[FuzzyRKDResult] = None

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

        Returns coefficients, variance-covariance matrix, and residual variance.
        """
        # Center x at cutoff
        x_centered = x - self.cutoff

        # Build design matrix [1, x, x², ...]
        X_design = np.column_stack([x_centered**p for p in range(order + 1)])

        # Weighted least squares
        W = np.diag(weights)
        XtWX = X_design.T @ W @ X_design
        XtWy = X_design.T @ W @ y

        try:
            XtWX_inv = np.linalg.inv(XtWX)
            coef = XtWX_inv @ XtWy
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)
            coef = XtWX_inv @ XtWy

        # Residuals and variance
        residuals = y - X_design @ coef
        weighted_residuals = weights * residuals**2

        df = np.sum(weights > 0) - (order + 1)
        if df <= 0:
            df = 1

        residual_var = np.sum(weighted_residuals) / df

        # Robust variance-covariance
        meat = X_design.T @ W @ np.diag(residuals**2) @ W @ X_design
        vcov = XtWX_inv @ meat @ XtWX_inv

        return coef, vcov, residual_var

    def _select_bandwidth(self, y: np.ndarray, x: np.ndarray) -> float:
        """Select optimal bandwidth for Fuzzy RKD."""
        from .bandwidth import rkd_ik_bandwidth

        return rkd_ik_bandwidth(y, x, self.cutoff)

    def fit(
        self,
        y: np.ndarray,
        x: np.ndarray,
        d: np.ndarray,
    ) -> FuzzyRKDResult:
        """
        Fit the Fuzzy RKD estimator.

        Parameters
        ----------
        y : array-like, shape (n,)
            Outcome variable
        x : array-like, shape (n,)
            Running variable
        d : array-like, shape (n,)
            Treatment variable (with fuzzy kink at cutoff)

        Returns
        -------
        FuzzyRKDResult
            Estimation results including LATE estimate, SE, CI, etc.

        Raises
        ------
        ValueError
            If inputs are invalid or first stage is too weak
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

        # Compute kernel weights
        weights = self._compute_kernel_weights(x, self.bandwidth_)

        # Split by cutoff
        left_mask = (x < self.cutoff) & (weights > 0)
        right_mask = (x >= self.cutoff) & (weights > 0)

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        min_obs = self.polynomial_order + 2
        if n_left < min_obs or n_right < min_obs:
            return FuzzyRKDResult(
                estimate=np.nan,
                se=np.nan,
                t_stat=np.nan,
                p_value=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                bandwidth=self.bandwidth_,
                n_left=n_left,
                n_right=n_right,
                first_stage_slope_left=np.nan,
                first_stage_slope_right=np.nan,
                first_stage_kink=np.nan,
                reduced_form_slope_left=np.nan,
                reduced_form_slope_right=np.nan,
                reduced_form_kink=np.nan,
                first_stage_f_stat=np.nan,
                alpha=self.alpha,
                retcode="error",
                message=f"Insufficient observations: left={n_left}, right={n_right}",
            )

        # =====================================================================
        # First Stage: Estimate kink in E[D|X]
        # =====================================================================
        coef_d_left, vcov_d_left, _ = self._fit_local_polynomial(
            d[left_mask], x[left_mask], weights[left_mask], self.polynomial_order
        )
        coef_d_right, vcov_d_right, _ = self._fit_local_polynomial(
            d[right_mask], x[right_mask], weights[right_mask], self.polynomial_order
        )

        fs_slope_left = coef_d_left[1]
        fs_slope_right = coef_d_right[1]
        fs_kink = fs_slope_right - fs_slope_left

        # First stage F-statistic (test that kink ≠ 0)
        var_fs_kink = vcov_d_left[1, 1] + vcov_d_right[1, 1]
        if var_fs_kink > 0:
            fs_f_stat = fs_kink**2 / var_fs_kink
        else:
            fs_f_stat = np.inf if fs_kink != 0 else 0

        # Check for weak first stage
        if abs(fs_kink) < 1e-10:
            return FuzzyRKDResult(
                estimate=np.nan,
                se=np.nan,
                t_stat=np.nan,
                p_value=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                bandwidth=self.bandwidth_,
                n_left=n_left,
                n_right=n_right,
                first_stage_slope_left=float(fs_slope_left),
                first_stage_slope_right=float(fs_slope_right),
                first_stage_kink=float(fs_kink),
                reduced_form_slope_left=np.nan,
                reduced_form_slope_right=np.nan,
                reduced_form_kink=np.nan,
                first_stage_f_stat=float(fs_f_stat),
                alpha=self.alpha,
                retcode="error",
                message="No kink detected in first stage (Δslope_D ≈ 0)",
            )

        # =====================================================================
        # Reduced Form: Estimate kink in E[Y|X]
        # =====================================================================
        coef_y_left, vcov_y_left, _ = self._fit_local_polynomial(
            y[left_mask], x[left_mask], weights[left_mask], self.polynomial_order
        )
        coef_y_right, vcov_y_right, _ = self._fit_local_polynomial(
            y[right_mask], x[right_mask], weights[right_mask], self.polynomial_order
        )

        rf_slope_left = coef_y_left[1]
        rf_slope_right = coef_y_right[1]
        rf_kink = rf_slope_right - rf_slope_left

        # =====================================================================
        # 2SLS Estimate: τ = Reduced Form / First Stage
        # =====================================================================
        estimate = rf_kink / fs_kink

        # Standard error via delta method
        # Var(τ) = (1/fs_kink²) * [Var(rf_kink) + τ² * Var(fs_kink)]
        var_rf_kink = vcov_y_left[1, 1] + vcov_y_right[1, 1]

        var_estimate = (1 / fs_kink**2) * (var_rf_kink + estimate**2 * var_fs_kink)
        se = np.sqrt(var_estimate)

        # Inference
        if se > 0 and np.isfinite(se):
            t_stat = estimate / se
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

        if fs_f_stat < 10:
            retcode = "warning"
            message = f"Weak first stage (F={fs_f_stat:.2f} < 10). LATE may be biased."

        if n_left < 30 or n_right < 30:
            retcode = "warning"
            message = f"Small sample warning: left={n_left}, right={n_right}"

        self.result_ = FuzzyRKDResult(
            estimate=float(estimate),
            se=float(se),
            t_stat=float(t_stat),
            p_value=float(p_value),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            bandwidth=float(self.bandwidth_),
            n_left=int(n_left),
            n_right=int(n_right),
            first_stage_slope_left=float(fs_slope_left),
            first_stage_slope_right=float(fs_slope_right),
            first_stage_kink=float(fs_kink),
            reduced_form_slope_left=float(rf_slope_left),
            reduced_form_slope_right=float(rf_slope_right),
            reduced_form_kink=float(rf_kink),
            first_stage_f_stat=float(fs_f_stat),
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
            "=" * 65,
            "Fuzzy Regression Kink Design Results",
            "=" * 65,
            f"Cutoff:           {self.cutoff:.4f}",
            f"Bandwidth:        {r.bandwidth:.4f}",
            f"Polynomial order: {self.polynomial_order}",
            f"Kernel:           {self.kernel}",
            "-" * 65,
            "LATE Estimate (2SLS)",
            "-" * 65,
            f"Estimate (τ):     {r.estimate:.4f}",
            f"Std. Error:       {r.se:.4f}",
            f"t-statistic:      {r.t_stat:.4f}",
            f"p-value:          {r.p_value:.4f}",
            f"95% CI:           [{r.ci_lower:.4f}, {r.ci_upper:.4f}]",
            "-" * 65,
            "First Stage: E[D|X]",
            "-" * 65,
            f"  Slope (left):   {r.first_stage_slope_left:.4f}",
            f"  Slope (right):  {r.first_stage_slope_right:.4f}",
            f"  Kink (ΔD):      {r.first_stage_kink:.4f}",
            f"  F-statistic:    {r.first_stage_f_stat:.2f}",
            "-" * 65,
            "Reduced Form: E[Y|X]",
            "-" * 65,
            f"  Slope (left):   {r.reduced_form_slope_left:.4f}",
            f"  Slope (right):  {r.reduced_form_slope_right:.4f}",
            f"  Kink (ΔY):      {r.reduced_form_kink:.4f}",
            "-" * 65,
            f"N (left):         {r.n_left}",
            f"N (right):        {r.n_right}",
            f"Status:           {r.retcode}",
            f"Message:          {r.message}",
            "=" * 65,
        ]
        return "\n".join(lines)
