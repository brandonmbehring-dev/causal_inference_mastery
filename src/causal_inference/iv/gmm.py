"""
Generalized Method of Moments (GMM) estimator for instrumental variables.

GMM provides an efficient estimator that optimally combines moment conditions
when there are more instruments than endogenous variables (overidentification).

Mathematical Details
--------------------
GMM minimizes the quadratic form:

    Q(β) = g(β)' W g(β)

where:
    g(β) = (1/n) Z'(Y - Dβ)  (sample moment conditions)
    W = weighting matrix

One-step GMM uses W = (Z'Z)^(-1)
Two-step GMM:
    1. Estimate β₁ with W₁ = (Z'Z)^(-1)
    2. Compute Ω = (1/n) Z'diag(û²)Z where û = Y - Dβ₁
    3. Estimate β₂ with W₂ = Ω^(-1) (optimal weighting)

Hansen J-test for overidentification:
    J = n * Q(β_GMM) ~ χ²(q - p) under H₀: all moment conditions valid

References
----------
- Hansen, L. P. (1982). Large sample properties of generalized method of
  moments estimators. Econometrica, 50(4), 1029-1054.

- Hayashi, F. (2000). Econometrics, Chapter 3.

- Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel
  Data, Chapter 8.
"""

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class GMM:
    """
    Generalized Method of Moments (GMM) estimator.

    GMM provides efficient estimation when there are more instruments than
    endogenous variables. The two-step GMM uses an optimal weighting matrix
    that accounts for heteroskedasticity.

    Parameters
    ----------
    steps : {'one', 'two'}, default='two'
        Number of GMM steps:
        - 'one': Use W = (Z'Z)^(-1) (equivalent to 2SLS)
        - 'two': Use optimal weighting matrix (efficient GMM)

    inference : {'standard', 'robust'}, default='robust'
        Type of standard errors to compute.

    alpha : float, default=0.05
        Significance level for confidence intervals and J-test.

    Attributes
    ----------
    coef_ : np.ndarray
        Coefficient estimates.
    se_ : np.ndarray
        Standard errors.
    t_stats_ : np.ndarray
        t-statistics.
    p_values_ : np.ndarray
        Two-sided p-values.
    ci_ : np.ndarray
        Confidence intervals (n_params × 2).
    j_statistic_ : float
        Hansen J-test statistic.
    j_pvalue_ : float
        Hansen J-test p-value.
    j_df_ : int
        Degrees of freedom for J-test (q - p).
    n_obs_ : int
        Number of observations.
    n_instruments_ : int
        Number of instruments.
    n_endogenous_ : int
        Number of endogenous variables.

    Examples
    --------
    >>> from causal_inference.iv import GMM
    >>> import numpy as np
    >>>
    >>> # Generate data with overidentification
    >>> np.random.seed(42)
    >>> n = 1000
    >>> Z1 = np.random.normal(0, 1, n)
    >>> Z2 = np.random.normal(0, 1, n)
    >>> Z = np.column_stack([Z1, Z2])  # 2 instruments
    >>> D = 2 * Z1 + 1.5 * Z2 + np.random.normal(0, 1, n)
    >>> Y = 0.5 * D + np.random.normal(0, 1, n)
    >>>
    >>> # Fit two-step GMM
    >>> gmm = GMM(steps='two', inference='robust')
    >>> gmm.fit(Y, D, Z)
    >>> print(f"GMM estimate: {gmm.coef_[0]:.3f}")
    >>> print(f"J-statistic: {gmm.j_statistic_:.3f} (p={gmm.j_pvalue_:.3f})")

    Notes
    -----
    - Two-step GMM is asymptotically more efficient than one-step
    - Hansen J-test checks validity of overidentifying restrictions
    - J-test is exactly zero for just-identified models (q = p)
    - High J-statistic suggests invalid instruments or misspecification
    """

    def __init__(
        self,
        steps: Literal["one", "two"] = "two",
        inference: Literal["standard", "robust"] = "robust",
        alpha: float = 0.05,
    ):
        if steps not in ["one", "two"]:
            raise ValueError(f"steps must be 'one' or 'two', got '{steps}'")
        if inference not in ["standard", "robust"]:
            raise ValueError(f"inference must be 'standard' or 'robust', got '{inference}'")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.steps = steps
        self.inference = inference
        self.alpha = alpha

        # Fitted attributes
        self.coef_: Optional[np.ndarray] = None
        self.se_: Optional[np.ndarray] = None
        self.t_stats_: Optional[np.ndarray] = None
        self.p_values_: Optional[np.ndarray] = None
        self.ci_: Optional[np.ndarray] = None
        self.j_statistic_: Optional[float] = None
        self.j_pvalue_: Optional[float] = None
        self.j_df_: Optional[int] = None
        self.n_obs_: Optional[int] = None
        self.n_instruments_: Optional[int] = None
        self.n_endogenous_: Optional[int] = None

    def fit(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> "GMM":
        """
        Fit GMM estimator.

        Parameters
        ----------
        Y : np.ndarray, shape (n,)
            Outcome variable.
        D : np.ndarray, shape (n,) or (n, p)
            Endogenous treatment variable(s).
        Z : np.ndarray, shape (n,) or (n, q)
            Instrumental variable(s).
        X : np.ndarray, shape (n, k), optional
            Exogenous control variables (excluding intercept).

        Returns
        -------
        self : GMM
            Fitted estimator.

        Raises
        ------
        ValueError
            If inputs are invalid or model is underidentified.
        """
        # Validate inputs
        Y, D, Z, X = self._validate_inputs(Y, D, Z, X)

        # Ensure 2D
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if D.ndim == 1:
            D = D.reshape(-1, 1)

        # Store dimensions
        n = len(Y)
        self.n_obs_ = n
        self.n_instruments_ = Z.shape[1]
        self.n_endogenous_ = D.shape[1]

        # Add intercept
        if X is None:
            X_with_intercept = np.ones((n, 1))
        else:
            X_with_intercept = np.column_stack([np.ones(n), X])

        # Combine instruments and exogenous variables
        ZX = np.column_stack([Z, X_with_intercept])

        # Combine endogenous and exogenous regressors
        DX = np.column_stack([D, X_with_intercept])

        # Estimate based on number of steps
        if self.steps == "one":
            coef, residuals, W = self._one_step_gmm(Y, DX, ZX)
        else:  # two-step
            coef, residuals, W = self._two_step_gmm(Y, DX, ZX)

        # Store results
        self.coef_ = coef

        # Compute standard errors
        se = self._compute_standard_errors(Y, DX, ZX, coef, residuals, W)
        self.se_ = se

        # Inference
        self.t_stats_ = coef / se
        df = n - len(coef)
        self.p_values_ = 2 * (1 - stats.t.cdf(np.abs(self.t_stats_), df=df))

        # Confidence intervals
        t_crit = stats.t.ppf(1 - self.alpha / 2, df=df)
        ci_lower = coef - t_crit * se
        ci_upper = coef + t_crit * se
        self.ci_ = np.column_stack([ci_lower, ci_upper])

        # Hansen J-test
        self.j_statistic_, self.j_pvalue_, self.j_df_ = self._hansen_j_test(residuals, Z, W)

        return self

    def _one_step_gmm(
        self, Y: np.ndarray, DX: np.ndarray, ZX: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One-step GMM with W = (ZX'ZX)^(-1).

        This is equivalent to 2SLS.

        Parameters:
        - Y: outcome (n,)
        - DX: [D, X_with_intercept] (n, p+k) - all regressors
        - ZX: [Z, X_with_intercept] (n, q+k) - all instruments
        """
        n = len(Y)

        # Weighting matrix: W = (ZX'ZX)^(-1)
        W = np.linalg.inv(ZX.T @ ZX / n)

        # GMM estimator: θ = (DX'ZX W ZX'DX)^(-1) (DX'ZX W ZX'Y)
        # where θ = [β, γ]' (effects of D and X)
        DX_ZX = DX.T @ ZX / n
        ZX_Y = ZX.T @ Y / n

        coef = np.linalg.solve(DX_ZX @ W @ DX_ZX.T, DX_ZX @ W @ ZX_Y)

        # Residuals: û = Y - DX @ θ = Y - D'β - X'γ
        residuals = Y - DX @ coef

        return coef, residuals, W

    def _two_step_gmm(
        self, Y: np.ndarray, DX: np.ndarray, ZX: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Two-step efficient GMM with optimal weighting matrix.

        Parameters:
        - Y: outcome (n,)
        - DX: [D, X_with_intercept] (n, p+k) - all regressors
        - ZX: [Z, X_with_intercept] (n, q+k) - all instruments
        """
        n = len(Y)

        # Step 1: One-step GMM to get initial estimates
        coef_1step, residuals_1step, W_1step = self._one_step_gmm(Y, DX, ZX)

        # Step 2: Compute optimal weighting matrix
        # Ω = (1/n) ZX'diag(û²)ZX
        Omega = (ZX.T * (residuals_1step**2)) @ ZX / n

        # Check conditioning
        try:
            W = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            # Fallback to one-step if Omega is singular
            return coef_1step, residuals_1step, W_1step

        # Step 2: Re-estimate with optimal W
        DX_ZX = DX.T @ ZX / n
        ZX_Y = ZX.T @ Y / n

        coef = np.linalg.solve(DX_ZX @ W @ DX_ZX.T, DX_ZX @ W @ ZX_Y)

        # Residuals: û = Y - DX @ θ
        residuals = Y - DX @ coef

        return coef, residuals, W

    def _hansen_j_test(
        self, residuals: np.ndarray, Z: np.ndarray, W: np.ndarray
    ) -> Tuple[float, float, int]:
        """
        Compute Hansen J-test for overidentifying restrictions.

        J = n * g'Wg where g = (1/n)Z'û

        Under H₀ (all moment conditions valid): J ~ χ²(q - p)
        """
        n = len(residuals)
        q = self.n_instruments_
        p = self.n_endogenous_

        # Degrees of freedom
        df = q - p

        # Just-identified: J = 0 exactly
        if df == 0:
            return 0.0, 1.0, 0

        # Moment conditions: g = (1/n)Z'û
        g = Z.T @ residuals / n

        # J-statistic: J = n * g'Wg
        # W includes both Z and X, so we need just Z part
        W_Z = W[: Z.shape[1], : Z.shape[1]]
        j_stat = n * (g.T @ W_Z @ g)

        # P-value from chi-squared distribution
        p_value = 1 - stats.chi2.cdf(j_stat, df=df)

        return float(j_stat), float(p_value), int(df)

    def _compute_standard_errors(
        self,
        Y: np.ndarray,
        DX: np.ndarray,
        ZX: np.ndarray,
        coef: np.ndarray,
        residuals: np.ndarray,
        W: np.ndarray,
    ) -> np.ndarray:
        """
        Compute GMM standard errors.

        Robust SE formula:
        Var(θ) = (1/n)(DX'ZX W ZX'DX)^(-1) (DX'ZX W S W ZX'DX) (DX'ZX W ZX'DX)^(-1)

        where S = (1/n)Σ(û²zz') for robust SEs
        """
        n = len(Y)

        # DX'ZX / n
        DX_ZX = DX.T @ ZX / n

        # (DX'ZX W ZX'DX)^(-1)
        bread = np.linalg.inv(DX_ZX @ W @ DX_ZX.T)

        if self.inference == "standard":
            # Homoskedastic: S = σ²(ZX'ZX)
            sigma2 = np.sum(residuals**2) / (n - len(coef))
            S = (ZX.T @ ZX / n) * sigma2
        else:  # robust
            # Heteroskedastic-robust: S = (1/n)Σ(û²zz')
            S = (ZX.T * (residuals**2)) @ ZX / n

        # Sandwich: (DX'ZX W ZX'DX)^(-1) (DX'ZX W S W ZX'DX) (DX'ZX W ZX'DX)^(-1)
        meat = DX_ZX @ W @ S @ W @ DX_ZX.T
        var_theta = bread @ meat @ bread / n

        se = np.sqrt(np.diag(var_theta))

        return se

    def _validate_inputs(
        self, Y: np.ndarray, D: np.ndarray, Z: np.ndarray, X: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Validate and preprocess inputs."""
        # Convert to numpy arrays
        Y = np.asarray(Y)
        D = np.asarray(D)
        Z = np.asarray(Z)
        if X is not None:
            X = np.asarray(X)

        # Check dimensions
        if Y.ndim != 1:
            raise ValueError(f"Y must be 1D, got shape {Y.shape}")

        n = len(Y)

        if D.ndim == 1:
            D_n = len(D)
            p = 1
        else:
            D_n, p = D.shape

        if D_n != n:
            raise ValueError(f"Y and D must have same length, got {n} and {D_n}")

        if Z.ndim == 1:
            Z_n = len(Z)
            q = 1
        else:
            Z_n, q = Z.shape

        if Z_n != n:
            raise ValueError(f"Y and Z must have same length, got {n} and {Z_n}")

        if X is not None:
            if X.ndim == 1:
                X_n = len(X)
            else:
                X_n = X.shape[0]
            if X_n != n:
                raise ValueError(f"Y and X must have same length, got {n} and {X_n}")

        # Check identification
        if q < p:
            raise ValueError(
                f"Model is underidentified: {p} endogenous variables but only {q} instruments. "
                f"Need at least {p} instruments."
            )

        return Y, D, Z, X

    def summary(self) -> pd.DataFrame:
        """
        Return formatted summary table of results.

        Returns
        -------
        summary : pd.DataFrame
            Table with coefficients, SEs, t-stats, p-values, and CIs.

        Raises
        ------
        ValueError
            If model has not been fitted.
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted. Call .fit() first.")

        # Create variable names
        var_names = [f"D_{i}" for i in range(self.n_endogenous_)]
        var_names += ["Intercept"]
        if self.coef_.shape[0] > self.n_endogenous_ + 1:
            n_controls = self.coef_.shape[0] - self.n_endogenous_ - 1
            var_names += [f"X_{i}" for i in range(n_controls)]

        summary = pd.DataFrame(
            {
                "Variable": var_names,
                "Coefficient": self.coef_,
                "Std. Error": self.se_,
                "t-statistic": self.t_stats_,
                "p-value": self.p_values_,
                f"CI Lower ({(1 - self.alpha) * 100:.0f}%)": self.ci_[:, 0],
                f"CI Upper ({(1 - self.alpha) * 100:.0f}%)": self.ci_[:, 1],
            }
        )

        # Add J-test info at bottom
        j_info = pd.DataFrame(
            {
                "Variable": [""],
                "Coefficient": [""],
                "Std. Error": [""],
                "t-statistic": [""],
                "p-value": [""],
                f"CI Lower ({(1 - self.alpha) * 100:.0f}%)": [""],
                f"CI Upper ({(1 - self.alpha) * 100:.0f}%)": [""],
            }
        )
        j_test = pd.DataFrame(
            {
                "Variable": [f"Hansen J-test (df={self.j_df_})"],
                "Coefficient": [f"{self.j_statistic_:.4f}"],
                "Std. Error": [""],
                "t-statistic": [""],
                "p-value": [f"{self.j_pvalue_:.4f}"],
                f"CI Lower ({(1 - self.alpha) * 100:.0f}%)": [""],
                f"CI Upper ({(1 - self.alpha) * 100:.0f}%)": [""],
            }
        )

        summary = pd.concat([summary, j_info, j_test], ignore_index=True)

        return summary
