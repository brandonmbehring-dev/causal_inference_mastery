"""
Sharp Regression Discontinuity Design (RDD) Estimator

Implements local linear regression at cutoff for causal effect estimation
when treatment is deterministically assigned based on a running variable.

References
----------
Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs:
    A guide to practice. Journal of Econometrics, 142(2), 615-635.
Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric
    confidence intervals for regression-discontinuity designs. Econometrica, 82(6), 2295-2326.
"""

import warnings
from typing import Optional, Tuple, Literal

import numpy as np
from scipy import stats

from src.causal_inference.utils.validation import (
    validate_arrays_same_length,
    validate_finite,
    validate_not_empty,
)


class SharpRDD:
    """
    Sharp Regression Discontinuity Design estimator.

    Uses local linear regression on each side of the cutoff to estimate
    the treatment effect at the discontinuity.

    Model:
        Left (x < c):  Y = α_L + β_L*(x - c) + ε_L
        Right (x ≥ c): Y = α_R + β_R*(x - c) + ε_R

    Treatment effect: τ = α_R - α_L (difference at cutoff)

    Parameters
    ----------
    cutoff : float
        Threshold value where treatment discontinuously changes
    bandwidth : float or str, default='ik'
        Bandwidth for local linear regression
        - float: Use specified bandwidth
        - 'ik': Imbens-Kalyanaraman optimal bandwidth
        - 'cct': Calonico-Cattaneo-Titiunik optimal bandwidth
    kernel : {'triangular', 'rectangular'}, default='triangular'
        Kernel function for weighting observations
    inference : {'standard', 'robust'}, default='robust'
        Standard errors method
        - 'standard': Homoskedastic SEs
        - 'robust': Heteroskedasticity-robust SEs (recommended)
    alpha : float, default=0.05
        Significance level for confidence intervals

    Attributes
    ----------
    coef_ : float
        RDD treatment effect estimate (τ)
    se_ : float
        Standard error
    t_stat_ : float
        T-statistic
    p_value_ : float
        P-value (two-sided)
    ci_ : tuple of float
        (lower, upper) confidence interval
    bandwidth_left_ : float
        Bandwidth used on left side
    bandwidth_right_ : float
        Bandwidth used on right side
    n_left_ : int
        Effective sample size on left
    n_right_ : int
        Effective sample size on right
    alpha_left_ : float
        Intercept estimate on left side
    alpha_right_ : float
        Intercept estimate on right side
    beta_left_ : float
        Slope estimate on left side
    beta_right_ : float
        Slope estimate on right side

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.rdd import SharpRDD
    >>>
    >>> # Generate RDD data with discontinuity at 0
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.uniform(-5, 5, n)
    >>> treatment_effect = 2.0
    >>> Y = X + treatment_effect * (X >= 0) + np.random.normal(0, 1, n)
    >>>
    >>> # Fit Sharp RDD
    >>> rdd = SharpRDD(cutoff=0.0, bandwidth='ik')
    >>> rdd.fit(Y, X)
    >>> print(f"Treatment effect: {rdd.coef_:.3f} (SE: {rdd.se_:.3f})")
    Treatment effect: 2.000 (SE: 0.150)
    """

    def __init__(
        self,
        cutoff: float,
        bandwidth: float | Literal["ik", "cct"] = "ik",
        kernel: Literal["triangular", "rectangular"] = "triangular",
        inference: Literal["standard", "robust"] = "robust",
        alpha: float = 0.05,
    ):
        self.cutoff = cutoff
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.inference = inference
        self.alpha = alpha

        # Validate inputs
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if kernel not in ["triangular", "rectangular"]:
            raise ValueError(f"kernel must be 'triangular' or 'rectangular', got {kernel}")
        if inference not in ["standard", "robust"]:
            raise ValueError(f"inference must be 'standard' or 'robust', got {inference}")

        # Results (set after fitting)
        self.coef_: Optional[float] = None
        self.se_: Optional[float] = None
        self.t_stat_: Optional[float] = None
        self.p_value_: Optional[float] = None
        self.ci_: Optional[Tuple[float, float]] = None
        self.bandwidth_left_: Optional[float] = None
        self.bandwidth_right_: Optional[float] = None
        self.n_left_: Optional[int] = None
        self.n_right_: Optional[int] = None
        self.alpha_left_: Optional[float] = None
        self.alpha_right_: Optional[float] = None
        self.beta_left_: Optional[float] = None
        self.beta_right_: Optional[float] = None

    def fit(self, Y: np.ndarray, X: np.ndarray) -> "SharpRDD":
        """
        Fit Sharp RDD estimator.

        Parameters
        ----------
        Y : array-like, shape (n,)
            Outcome variable
        X : array-like, shape (n,)
            Running variable (assignment variable)

        Returns
        -------
        self : SharpRDD
            Fitted estimator
        """
        # Convert to numpy arrays
        Y = np.asarray(Y).flatten()
        X = np.asarray(X).flatten()

        # Validation (using shared utilities)
        validate_not_empty(Y, "Y")
        validate_finite(Y, "Y")
        validate_finite(X, "X")
        validate_arrays_same_length(Y=Y, X=X)

        # Check observations on both sides of cutoff
        n_left = np.sum(X < self.cutoff)
        n_right = np.sum(X >= self.cutoff)
        if n_left == 0:
            raise ValueError("No observations with X < cutoff. Cannot estimate left side.")
        if n_right == 0:
            raise ValueError("No observations with X >= cutoff. Cannot estimate right side.")

        # Select bandwidth
        if isinstance(self.bandwidth, str):
            if self.bandwidth == "ik":
                from .bandwidth import imbens_kalyanaraman_bandwidth

                h = imbens_kalyanaraman_bandwidth(Y, X, self.cutoff, self.kernel)
            elif self.bandwidth == "cct":
                from .bandwidth import cct_bandwidth

                h, _ = cct_bandwidth(Y, X, self.cutoff, self.kernel)
            else:
                raise ValueError(f"Unknown bandwidth selector: {self.bandwidth}")
        else:
            h = float(self.bandwidth)
            if h <= 0:
                raise ValueError(f"bandwidth must be positive, got {h}")

        self.bandwidth_left_ = h
        self.bandwidth_right_ = h

        # Fit local linear regression on each side
        alpha_left, beta_left, var_left, n_left_eff = self._local_linear_regression(
            Y, X, "left", h, self.kernel
        )
        alpha_right, beta_right, var_right, n_right_eff = self._local_linear_regression(
            Y, X, "right", h, self.kernel
        )

        self.alpha_left_ = alpha_left
        self.alpha_right_ = alpha_right
        self.beta_left_ = beta_left
        self.beta_right_ = beta_right
        self.n_left_ = n_left_eff
        self.n_right_ = n_right_eff

        # Treatment effect: difference at cutoff
        self.coef_ = alpha_right - alpha_left

        # Standard error
        if self.inference == "robust":
            # Heteroskedasticity-robust SE
            se_left = np.sqrt(var_left)
            se_right = np.sqrt(var_right)
            self.se_ = np.sqrt(se_left**2 + se_right**2)
        else:
            # Standard SE (assumes homoskedasticity)
            self.se_ = np.sqrt(var_left + var_right)

        # Inference
        self.t_stat_ = self.coef_ / self.se_
        df = n_left_eff + n_right_eff - 4  # 2 parameters per side
        self.p_value_ = 2 * (1 - stats.t.cdf(abs(self.t_stat_), df=df))

        # Confidence interval
        t_crit = stats.t.ppf(1 - self.alpha / 2, df=df)
        self.ci_ = (self.coef_ - t_crit * self.se_, self.coef_ + t_crit * self.se_)

        # Warn if effective sample sizes are small
        if n_left_eff < 30 or n_right_eff < 30:
            warnings.warn(
                f"Small effective sample size (n_left={n_left_eff}, n_right={n_right_eff}). "
                f"Consider increasing bandwidth or checking data sparsity near cutoff.",
                RuntimeWarning,
            )

        return self

    def _local_linear_regression(
        self, Y: np.ndarray, X: np.ndarray, side: str, bandwidth: float, kernel: str
    ) -> Tuple[float, float, float, int]:
        """
        Fit local linear regression on one side of cutoff.

        Parameters
        ----------
        Y : ndarray
            Outcome variable
        X : ndarray
            Running variable
        side : {'left', 'right'}
            Which side of cutoff to fit
        bandwidth : float
            Bandwidth
        kernel : str
            Kernel function

        Returns
        -------
        alpha : float
            Intercept estimate (value at cutoff)
        beta : float
            Slope estimate
        variance : float
            Variance of alpha estimate
        n_eff : int
            Effective sample size (observations with positive weight)
        """
        # Select observations on this side of cutoff
        if side == "left":
            mask = X < self.cutoff
        else:  # side == 'right'
            mask = X >= self.cutoff

        Y_side = Y[mask]
        X_side = X[mask]

        # Centered running variable
        X_centered = X_side - self.cutoff

        # Kernel weights
        u = X_centered / bandwidth
        weights = self._kernel_weight(u, kernel)

        # Effective sample size
        n_eff = np.sum(weights > 0)

        # Weighted least squares: Y = alpha + beta * X_centered + error
        # Design matrix: [1, X_centered]
        design = np.column_stack([np.ones(len(X_side)), X_centered])

        # Weighted design matrix
        W = np.diag(weights)
        XtWX = design.T @ W @ design
        XtWY = design.T @ W @ Y_side

        # Solve for coefficients
        try:
            coefs = np.linalg.solve(XtWX, XtWY)
        except np.linalg.LinAlgError:
            raise ValueError(
                f"Singular matrix on {side} side. Try increasing bandwidth or checking for collinearity."
            )

        alpha, beta = coefs[0], coefs[1]

        # Variance of alpha (intercept = value at cutoff)
        if self.inference == "robust":
            # Heteroskedasticity-robust variance (sandwich estimator)
            residuals = Y_side - design @ coefs
            meat = design.T @ W @ np.diag(residuals**2) @ W @ design
            try:
                XtWX_inv = np.linalg.inv(XtWX)
            except np.linalg.LinAlgError:
                XtWX_inv = np.linalg.pinv(XtWX)
            var_matrix = XtWX_inv @ meat @ XtWX_inv
            variance = var_matrix[0, 0]  # Variance of intercept
        else:
            # Standard variance (assumes homoskedasticity)
            residuals = Y_side - design @ coefs
            sigma2 = np.sum(weights * residuals**2) / (n_eff - 2)
            try:
                XtWX_inv = np.linalg.inv(XtWX)
            except np.linalg.LinAlgError:
                XtWX_inv = np.linalg.pinv(XtWX)
            variance = sigma2 * XtWX_inv[0, 0]

        return alpha, beta, variance, n_eff

    def _kernel_weight(self, u: np.ndarray, kernel: str) -> np.ndarray:
        """
        Compute kernel weight for normalized distance u.

        Parameters
        ----------
        u : ndarray
            Normalized distance: (X - cutoff) / bandwidth
        kernel : str
            Kernel function: 'triangular' or 'rectangular'

        Returns
        -------
        weights : ndarray
            Kernel weights
        """
        if kernel == "triangular":
            # K(u) = (1 - |u|) if |u| <= 1, else 0
            weights = np.maximum(1 - np.abs(u), 0)
        elif kernel == "rectangular":
            # K(u) = 1 if |u| <= 1, else 0
            weights = (np.abs(u) <= 1).astype(float)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        return weights

    def summary(self) -> str:
        """
        Return formatted results table.

        Returns
        -------
        summary : str
            Formatted summary table
        """
        if self.coef_ is None:
            raise ValueError("Must call .fit() before .summary()")

        lines = [
            "=" * 70,
            "Sharp RDD Results",
            "=" * 70,
            f"Cutoff:           {self.cutoff:.4f}",
            f"Bandwidth (left): {self.bandwidth_left_:.4f}",
            f"Bandwidth (right):{self.bandwidth_right_:.4f}",
            f"Kernel:           {self.kernel}",
            f"Inference:        {self.inference}",
            "-" * 70,
            f"Treatment Effect: {self.coef_:.4f}",
            f"Std. Error:       {self.se_:.4f}",
            f"t-statistic:      {self.t_stat_:.4f}",
            f"p-value:          {self.p_value_:.4f}",
            f"95% CI:           [{self.ci_[0]:.4f}, {self.ci_[1]:.4f}]",
            "-" * 70,
            f"n (left):         {self.n_left_:,}",
            f"n (right):        {self.n_right_:,}",
            f"Total n:          {self.n_left_ + self.n_right_:,}",
            "-" * 70,
            f"alpha (left):     {self.alpha_left_:.4f}",
            f"alpha (right):    {self.alpha_right_:.4f}",
            f"beta (left):      {self.beta_left_:.4f}",
            f"beta (right):     {self.beta_right_:.4f}",
            "=" * 70,
        ]

        return "\n".join(lines)
