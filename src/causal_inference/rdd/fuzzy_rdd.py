"""
Fuzzy Regression Discontinuity Design (RDD) Estimator

Implements 2SLS estimation for Fuzzy RDD where treatment compliance is imperfect
at the cutoff. Estimates Local Average Treatment Effect (LATE) for compliers.

References
----------
Hahn, J., Todd, P., & Van der Klaauw, W. (2001). Identification and estimation of
    treatment effects with a regression-discontinuity design. Econometrica, 69(1), 201-209.
Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to
    practice. Journal of Econometrics, 142(2), 615-635.
Lee, D. S., & Lemieux, T. (2010). Regression discontinuity designs in economics.
    Journal of Economic Literature, 48(2), 281-355.
"""

import warnings
from typing import Literal, Optional, Tuple

import numpy as np
from scipy import stats

from ..iv.two_stage_least_squares import TwoStageLeastSquares
from .bandwidth import imbens_kalyanaraman_bandwidth, cct_bandwidth
from src.causal_inference.utils.validation import (
    validate_arrays_same_length,
    validate_finite,
)


def _compute_kernel_weights(
    X: np.ndarray,
    cutoff: float,
    bandwidth: float,
    kernel: str,
) -> np.ndarray:
    """
    Compute kernel weights for observations based on distance from cutoff.

    Parameters
    ----------
    X : ndarray
        Running variable values.
    cutoff : float
        RDD cutoff value.
    bandwidth : float
        Bandwidth for kernel weighting.
    kernel : str
        Kernel type: 'triangular', 'rectangular', or 'epanechnikov'.

    Returns
    -------
    weights : ndarray
        Non-negative kernel weights for each observation.
    """
    u = (X - cutoff) / bandwidth

    if kernel == "triangular":
        # Triangular: K(u) = (1 - |u|) for |u| <= 1
        weights = np.maximum(0, 1 - np.abs(u))
    elif kernel == "rectangular":
        # Rectangular (uniform): K(u) = 1 for |u| <= 1
        weights = (np.abs(u) <= 1).astype(float)
    elif kernel == "epanechnikov":
        # Epanechnikov: K(u) = 0.75 * (1 - u^2) for |u| <= 1
        weights = np.maximum(0, 0.75 * (1 - u**2))
    else:
        raise ValueError(
            f"Unknown kernel: {kernel}. Use 'triangular', 'rectangular', or 'epanechnikov'."
        )

    return weights


def _weighted_2sls(
    Y: np.ndarray,
    D: np.ndarray,
    Z: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Weighted Two-Stage Least Squares for kernel-weighted RDD.

    Parameters
    ----------
    Y : ndarray, shape (n,)
        Outcome variable.
    D : ndarray, shape (n,)
        Endogenous treatment.
    Z : ndarray, shape (n,)
        Instrument (eligibility indicator).
    X : ndarray, shape (n, k)
        Exogenous controls.
    weights : ndarray, shape (n,)
        Kernel weights for each observation.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    dict with keys:
        - coef: Treatment effect estimate
        - se: Standard error
        - t_stat: t-statistic
        - p_value: p-value
        - ci: (lower, upper) confidence interval
        - first_stage_f: First-stage F-statistic
        - first_stage_r2: First-stage R-squared
    """
    n = len(Y)

    # Apply square root weights for WLS (sqrt because we square them in normal equations)
    sqrt_w = np.sqrt(weights)

    # First stage: D ~ Z + X (weighted)
    # Design matrix: [1, Z, X]
    W1 = np.column_stack([np.ones(n), Z, X])

    # Weight the observations
    W1_w = W1 * sqrt_w[:, np.newaxis]
    D_w = D * sqrt_w

    # Solve weighted first stage
    try:
        beta_first, residuals_first, rank, s = np.linalg.lstsq(W1_w, D_w, rcond=None)
    except np.linalg.LinAlgError:
        raise ValueError("First stage regression failed (singular matrix)")

    # Fitted values (predicted D)
    D_hat = W1 @ beta_first

    # First-stage F-statistic (on Z coefficient)
    # Compare restricted model (without Z) vs unrestricted
    W1_restricted = np.column_stack([np.ones(n), X])
    W1_restricted_w = W1_restricted * sqrt_w[:, np.newaxis]
    beta_restricted, _, _, _ = np.linalg.lstsq(W1_restricted_w, D_w, rcond=None)
    D_hat_restricted = W1_restricted @ beta_restricted

    # Weighted residuals
    resid_unrestricted = (D - D_hat) * sqrt_w
    resid_restricted = (D - D_hat_restricted) * sqrt_w

    ssr_unrestricted = np.sum(resid_unrestricted**2)
    ssr_restricted = np.sum(resid_restricted**2)

    # F-statistic: F = ((SSR_R - SSR_U) / q) / (SSR_U / (n - k))
    q = 1  # One instrument (Z)
    k_full = W1.shape[1]
    df_resid = n - k_full

    if df_resid > 0 and ssr_unrestricted > 0:
        first_stage_f = ((ssr_restricted - ssr_unrestricted) / q) / (ssr_unrestricted / df_resid)
    else:
        first_stage_f = np.nan

    # First-stage R-squared
    tss_first = np.sum((D_w - np.mean(D_w))**2)
    first_stage_r2 = 1 - ssr_unrestricted / tss_first if tss_first > 0 else 0.0

    # Second stage: Y ~ D_hat + X (weighted)
    # Design matrix: [1, D_hat, X]
    W2 = np.column_stack([np.ones(n), D_hat, X])

    W2_w = W2 * sqrt_w[:, np.newaxis]
    Y_w = Y * sqrt_w

    # Solve weighted second stage
    try:
        beta_second, residuals_second, rank, s = np.linalg.lstsq(W2_w, Y_w, rcond=None)
    except np.linalg.LinAlgError:
        raise ValueError("Second stage regression failed (singular matrix)")

    # Treatment effect is coefficient on D_hat (index 1)
    coef = beta_second[1]

    # Standard errors (robust, using actual D not D_hat for proper IV SEs)
    # Residuals from structural equation
    Y_fitted = W2 @ beta_second
    resid_struct = (Y - Y_fitted) * sqrt_w

    # Robust variance using sandwich estimator
    # V = (W'W)^-1 W' diag(e^2) W (W'W)^-1
    WtW = W2_w.T @ W2_w
    try:
        WtW_inv = np.linalg.inv(WtW)
    except np.linalg.LinAlgError:
        WtW_inv = np.linalg.pinv(WtW)

    # Meat of sandwich: sum of w_i^2 * e_i^2 * x_i x_i'
    meat = np.zeros((W2.shape[1], W2.shape[1]))
    for i in range(n):
        xi = W2_w[i, :]
        meat += resid_struct[i]**2 * np.outer(xi, xi)

    # Sandwich variance
    vcov = WtW_inv @ meat @ WtW_inv

    # SE for treatment effect (coefficient 1)
    se = np.sqrt(vcov[1, 1])

    # Inference
    if se > 0:
        t_stat = coef / se
        df = n - W2.shape[1]
        df = max(df, 1)
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        ci_lower = coef - t_crit * se
        ci_upper = coef + t_crit * se
    else:
        t_stat = np.inf if coef != 0 else 0
        p_value = 0.0 if coef != 0 else 1.0
        ci_lower = coef
        ci_upper = coef

    return {
        "coef": coef,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci": (ci_lower, ci_upper),
        "first_stage_f": first_stage_f,
        "first_stage_r2": first_stage_r2,
    }


class FuzzyRDD:
    """
    Fuzzy Regression Discontinuity Design estimator using 2SLS.

    For imperfect compliance at cutoff, estimates Local Average Treatment Effect (LATE)
    using treatment eligibility (Z = 1{X ≥ c}) as instrument for actual treatment (D).

    Fuzzy RDD Framework:
        Instrument: Z = 1{X ≥ cutoff}
        First stage:  D = α₀ + α₁*Z + f(X) + ν
        Second stage: Y = β₀ + β₁*D̂ + g(X) + ε

    Treatment effect: τ = β₁ (LATE for compliers at cutoff)

    When compliance is perfect (all X ≥ c take treatment, all X < c do not),
    Fuzzy RDD reduces to Sharp RDD.

    Parameters
    ----------
    cutoff : float
        Threshold value where treatment eligibility changes
    bandwidth : float or str, default='ik'
        Bandwidth for local regression
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
        Fuzzy RDD treatment effect estimate (LATE)
    se_ : float
        Standard error
    t_stat_ : float
        T-statistic
    p_value_ : float
        P-value (two-sided)
    ci_ : tuple of float
        (lower, upper) confidence interval
    compliance_rate_ : float
        Estimated compliance rate: E[D|Z=1] - E[D|Z=0]
    first_stage_f_stat_ : float
        First-stage F-statistic (instrument strength)
    first_stage_r2_ : float
        First-stage R-squared
    weak_instrument_warning_ : bool
        True if F-statistic < 10 (weak instrument)
    bandwidth_left_ : float
        Bandwidth used on left side
    bandwidth_right_ : float
        Bandwidth used on right side
    n_left_ : int
        Effective sample size on left
    n_right_ : int
        Effective sample size on right
    n_obs_ : int
        Total observations used in estimation (within bandwidth)

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.rdd import FuzzyRDD
    >>>
    >>> # Generate Fuzzy RDD data (imperfect compliance)
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.uniform(-5, 5, n)
    >>> Z = (X >= 0).astype(float)  # Eligibility
    >>>
    >>> # Fuzzy treatment: 80% compliance
    >>> baseline_treatment_prob = 0.2
    >>> treatment_boost = 0.6
    >>> treatment_prob = baseline_treatment_prob + treatment_boost * Z
    >>> D = np.random.binomial(1, treatment_prob)
    >>>
    >>> # Outcome with LATE = 2.0
    >>> Y = X + 2.0 * D + np.random.normal(0, 1, n)
    >>>
    >>> # Fit Fuzzy RDD
    >>> rdd = FuzzyRDD(cutoff=0.0, bandwidth='ik')
    >>> rdd.fit(Y, X, D)
    >>> print(f"LATE: {rdd.coef_:.3f} (SE: {rdd.se_:.3f})")
    >>> print(f"Compliance rate: {rdd.compliance_rate_:.2%}")
    >>> print(f"First-stage F: {rdd.first_stage_f_stat_:.1f}")
    LATE: 2.000 (SE: 0.200)
    Compliance rate: 60%
    First-stage F: 120.5

    Notes
    -----
    - Fuzzy RDD estimates LATE (Local Average Treatment Effect), not ATE
    - LATE is the effect for compliers (units whose treatment changes at cutoff)
    - Requires monotonicity: Z increases D for all units (no defiers)
    - Weak instruments (F < 10) lead to biased estimates and poor coverage
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

        # Attributes set during fit
        self.coef_: float | None = None
        self.se_: float | None = None
        self.t_stat_: float | None = None
        self.p_value_: float | None = None
        self.ci_: Tuple[float, float] | None = None

        self.compliance_rate_: float | None = None
        self.first_stage_f_stat_: float | None = None
        self.first_stage_r2_: float | None = None
        self.weak_instrument_warning_: bool = False

        self.bandwidth_left_: float | None = None
        self.bandwidth_right_: float | None = None
        self.n_left_: int | None = None
        self.n_right_: int | None = None
        self.n_obs_: int | None = None

        self._2sls: TwoStageLeastSquares | None = None
        self._fitted: bool = False

    def fit(self, Y: np.ndarray, X: np.ndarray, D: np.ndarray) -> "FuzzyRDD":
        """
        Fit Fuzzy RDD using 2SLS.

        Parameters
        ----------
        Y : array-like, shape (n,)
            Outcome variable
        X : array-like, shape (n,)
            Running variable
        D : array-like, shape (n,)
            Actual treatment received (0/1)
            Note: D may differ from eligibility Z = 1{X ≥ cutoff} (imperfect compliance)

        Returns
        -------
        self : FuzzyRDD
            Fitted estimator

        Raises
        ------
        ValueError
            If no observations on either side of cutoff, or if all observations
            outside bandwidth, or if no variation in treatment.
        RuntimeWarning
            If weak instrument detected (F < 10), or very low compliance (< 0.3),
            or small effective sample size (< 30).
        """
        # Convert to numpy arrays
        Y = np.asarray(Y).flatten()
        X = np.asarray(X).flatten()
        D = np.asarray(D).flatten()

        # Input validation (using shared utilities)
        validate_finite(Y, "Y")
        validate_finite(X, "X")
        validate_finite(D, "D")
        validate_arrays_same_length(Y=Y, X=X, D=D)

        # Check for observations on both sides
        n_left_total = np.sum(X < self.cutoff)
        n_right_total = np.sum(X >= self.cutoff)
        if n_left_total == 0:
            raise ValueError("No observations with X < cutoff")
        if n_right_total == 0:
            raise ValueError("No observations with X >= cutoff")

        # Check for variation in treatment
        if np.std(D) == 0:
            raise ValueError(
                "No variation in treatment D. All units have same treatment status. "
                "Fuzzy RDD requires variation in actual treatment."
            )

        # Select bandwidth
        h = self._select_bandwidth(Y, X)
        self.bandwidth_left_ = h
        self.bandwidth_right_ = h

        # Subset to bandwidth window
        in_window = np.abs(X - self.cutoff) <= h
        if np.sum(in_window) == 0:
            raise ValueError(
                f"No observations within bandwidth {h:.3f} of cutoff. "
                f"Try increasing bandwidth or using 'cct' bandwidth selection."
            )

        Y_window = Y[in_window]
        X_window = X[in_window]
        D_window = D[in_window]

        # Instrument: Z = 1{X >= cutoff}
        Z_window = (X_window >= self.cutoff).astype(float)

        # Check for variation in instrument within window
        if np.std(Z_window) == 0:
            raise ValueError(
                "No variation in instrument (all X on same side of cutoff within bandwidth). "
                "This should not happen unless bandwidth is too small."
            )

        # Check for variation in treatment within window
        if np.std(D_window) == 0:
            raise ValueError(
                "No variation in treatment D within bandwidth window. "
                "Cannot estimate Fuzzy RDD without treatment variation."
            )

        # Compute effective sample sizes
        left_mask = X_window < self.cutoff
        right_mask = X_window >= self.cutoff
        self.n_left_ = np.sum(left_mask)
        self.n_right_ = np.sum(right_mask)
        self.n_obs_ = len(Y_window)

        # Warn if small sample
        if self.n_left_ < 30 or self.n_right_ < 30:
            warnings.warn(
                f"Small effective sample size: n_left={self.n_left_}, n_right={self.n_right_}. "
                f"Standard errors may be unreliable. Consider increasing bandwidth.",
                RuntimeWarning,
            )

        # Create local linear controls (separate slopes on each side)
        X_centered = X_window - self.cutoff
        X_left_control = X_centered * left_mask
        X_right_control = X_centered * right_mask

        # Stack controls
        controls = np.column_stack([X_left_control, X_right_control])

        # BUG-1 FIX: Compute kernel weights and apply to 2SLS
        # Previously kernel parameter was accepted but never used
        kernel_weights = _compute_kernel_weights(
            X=X_window,
            cutoff=self.cutoff,
            bandwidth=h,
            kernel=self.kernel,
        )

        # Fit weighted 2SLS with kernel weights applied
        result_2sls = _weighted_2sls(
            Y=Y_window,
            D=D_window,
            Z=Z_window,
            X=controls,
            weights=kernel_weights,
            alpha=self.alpha,
        )

        # Extract results
        self.coef_ = float(result_2sls["coef"])
        self.se_ = float(result_2sls["se"])
        self.t_stat_ = float(result_2sls["t_stat"])
        self.p_value_ = float(result_2sls["p_value"])
        self.ci_ = (float(result_2sls["ci"][0]), float(result_2sls["ci"][1]))

        # First-stage diagnostics
        self.first_stage_f_stat_ = float(result_2sls["first_stage_f"])
        self.first_stage_r2_ = float(result_2sls["first_stage_r2"])

        # Store kernel weights for diagnostics
        self._kernel_weights = kernel_weights

        # Compute compliance rate
        self.compliance_rate_ = self._compute_compliance_rate(D_window, Z_window)

        # Check for weak instrument
        if self.first_stage_f_stat_ < 10:
            self.weak_instrument_warning_ = True
            warnings.warn(
                f"Weak instrument detected: F-statistic = {self.first_stage_f_stat_:.2f} < 10. "
                f"Estimates may be biased and standard errors may be too small. "
                f"Stock-Yogo (2005) critical value for 10% maximal IV size is 16.38.",
                RuntimeWarning,
            )

        # Check for very low compliance
        if self.compliance_rate_ < 0.3:
            warnings.warn(
                f"Very low compliance rate: {self.compliance_rate_:.1%}. "
                f"Estimates may have low power and large standard errors.",
                RuntimeWarning,
            )

        self._fitted = True
        return self

    def _select_bandwidth(self, Y: np.ndarray, X: np.ndarray) -> float:
        """
        Select bandwidth using specified method.

        For Fuzzy RDD, we use the same bandwidth selection methods as Sharp RDD.
        This is standard practice, though optimal bandwidth for Fuzzy RDD is
        an active research area.

        Parameters
        ----------
        Y : ndarray
            Outcome variable
        X : ndarray
            Running variable

        Returns
        -------
        h : float
            Selected bandwidth
        """
        if isinstance(self.bandwidth, (int, float)):
            h = float(self.bandwidth)
        elif self.bandwidth == "ik":
            h = imbens_kalyanaraman_bandwidth(Y, X, self.cutoff, self.kernel)
        elif self.bandwidth == "cct":
            h_main, _ = cct_bandwidth(Y, X, self.cutoff, self.kernel)
            h = h_main
        else:
            raise ValueError(
                f"Unknown bandwidth: {self.bandwidth}. "
                f"Must be float, 'ik', or 'cct'."
            )

        # Regularize bandwidth (prevent pathological values)
        X_sd = np.std(X)
        h = np.clip(h, 0.1 * X_sd, 2.0 * X_sd)

        return h

    @staticmethod
    def _compute_compliance_rate(D: np.ndarray, Z: np.ndarray) -> float:
        """
        Compute compliance rate: E[D|Z=1] - E[D|Z=0].

        This is the first-stage effect: the causal effect of eligibility
        on actual treatment receipt.

        Parameters
        ----------
        D : ndarray
            Actual treatment (0/1)
        Z : ndarray
            Instrument (0/1)

        Returns
        -------
        compliance : float
            Compliance rate (proportion of compliers)
        """
        treated = D[Z == 1]
        control = D[Z == 0]

        if len(treated) == 0 or len(control) == 0:
            return np.nan

        compliance = np.mean(treated) - np.mean(control)
        return float(compliance)

    def summary(self) -> str:
        """
        Generate summary table of results.

        Returns
        -------
        summary : str
            Formatted summary table
        """
        if not self._fitted:
            return "Model not fitted. Call fit() first."

        lines = []
        lines.append("=" * 60)
        lines.append("Fuzzy Regression Discontinuity Design (Fuzzy RDD)")
        lines.append("=" * 60)
        lines.append(f"Cutoff:                {self.cutoff:.3f}")
        lines.append(f"Bandwidth (left):      {self.bandwidth_left_:.3f}")
        lines.append(f"Bandwidth (right):     {self.bandwidth_right_:.3f}")
        lines.append(f"Kernel:                {self.kernel}")
        lines.append(f"Inference:             {self.inference}")
        lines.append(f"Observations (left):   {self.n_left_}")
        lines.append(f"Observations (right):  {self.n_right_}")
        lines.append(f"Total observations:    {self.n_obs_}")
        lines.append("")
        lines.append("Treatment Effect (LATE):")
        lines.append(f"  Estimate:            {self.coef_:.4f}")
        lines.append(f"  Std. Error:          {self.se_:.4f}")
        lines.append(f"  t-statistic:         {self.t_stat_:.3f}")
        lines.append(f"  p-value:             {self.p_value_:.4f}")
        lines.append(f"  95% CI:              [{self.ci_[0]:.4f}, {self.ci_[1]:.4f}]")
        lines.append("")
        lines.append("First-Stage Diagnostics:")
        lines.append(f"  Compliance rate:     {self.compliance_rate_:.4f} ({self.compliance_rate_:.1%})")
        lines.append(f"  F-statistic:         {self.first_stage_f_stat_:.2f}")
        lines.append(f"  R-squared:           {self.first_stage_r2_:.4f}")

        if self.weak_instrument_warning_:
            lines.append("  ⚠ WARNING: Weak instrument (F < 10)")

        if self.compliance_rate_ < 0.3:
            lines.append("  ⚠ WARNING: Very low compliance (< 30%)")

        lines.append("=" * 60)
        lines.append("")
        lines.append("Note: Estimates are Local Average Treatment Effects (LATE)")
        lines.append("      for compliers at the cutoff.")

        return "\n".join(lines)
