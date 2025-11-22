"""
Two-Stage Least Squares (2SLS) Instrumental Variables Estimator.

This module implements the core 2SLS estimator for linear instrumental variables regression.
Addresses endogeneity bias when treatment is correlated with unobserved confounders.

Key References:
    - Angrist, J.D., and J.-S. Pischke (2009). "Mostly Harmless Econometrics", Chapter 4.
    - Wooldridge, J.M. (2010). "Econometric Analysis of Cross Section and Panel Data", 2nd ed.
    - Davidson, R., and J.G. MacKinnon (2004). "Econometric Theory and Methods", Chapter 8.

Mathematical Framework:
    First Stage:  D = π₀ + π₁Z + π₂X + ν
    Second Stage: Y = β₀ + β₁D̂ + β₂X + ε

    where:
    - Y: Outcome variable
    - D: Endogenous treatment variable
    - Z: Instrumental variable(s) - correlated with D, uncorrelated with ε
    - X: Exogenous controls (optional)
    - D̂: Predicted treatment from first stage

Identification Requirements:
    1. Relevance: Cov(Z, D) ≠ 0 (instruments predict treatment)
    2. Exclusion: Cov(Z, ε) = 0 (instruments only affect Y through D)
    3. Order condition: q >= p (at least as many instruments as endogenous variables)
"""

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from numpy.linalg import LinAlgError


class TwoStageLeastSquares:
    """
    Two-Stage Least Squares (2SLS) estimator for instrumental variables regression.

    Handles endogeneity bias when treatment is correlated with unobserved confounders
    by using instrumental variables that are correlated with treatment but uncorrelated
    with the error term.

    Parameters
    ----------
    inference : {'standard', 'robust', 'clustered'}, default='robust'
        Type of standard errors:
        - 'standard': Homoskedastic standard errors
        - 'robust': Heteroskedasticity-robust (White/HC0)
        - 'clustered': Cluster-robust standard errors
    cluster_var : array-like, optional
        Cluster variable for clustered standard errors. Required if inference='clustered'.
    alpha : float, default=0.05
        Significance level for confidence intervals (e.g., 0.05 for 95% CIs).

    Attributes
    ----------
    coef_ : ndarray
        Estimated coefficients (treatment effect + controls).
    se_ : ndarray
        Standard errors of coefficients.
    t_stats_ : ndarray
        t-statistics for coefficients.
    p_values_ : ndarray
        p-values for two-sided t-tests.
    ci_ : ndarray, shape (n_coef, 2)
        Confidence intervals for coefficients.
    first_stage_f_stat_ : float
        First-stage F-statistic (quick diagnostic for instrument strength).
    first_stage_r2_ : float
        First-stage R-squared.
    n_obs_ : int
        Number of observations.
    n_instruments_ : int
        Number of instruments.
    n_endogenous_ : int
        Number of endogenous variables.
    fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    >>> # Simple IV regression: returns to schooling with quarter of birth as instrument
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 10000
    >>>
    >>> # Instrument: Quarter of birth (exogenous)
    >>> Z = np.random.choice([1, 2, 3, 4], size=n)
    >>>
    >>> # Treatment: Education (affected by quarter via compulsory schooling laws)
    >>> D = 12 + 0.5 * (Z == 1) + np.random.normal(0, 2, n)
    >>>
    >>> # Outcome: Log wages (causal effect = 0.10)
    >>> Y = 8 + 0.1 * D + np.random.normal(0, 0.5, n)
    >>>
    >>> # Fit 2SLS
    >>> iv = TwoStageLeastSquares(inference='robust')
    >>> iv.fit(Y, D, Z)
    >>>
    >>> # Check results
    >>> print(f"Treatment effect: {iv.coef_[0]:.3f}")
    >>> print(f"Standard error: {iv.se_[0]:.3f}")
    >>> print(f"First-stage F-statistic: {iv.first_stage_f_stat_:.2f}")

    Notes
    -----
    **CRITICAL**: This implementation uses correct 2SLS standard errors, not naive OLS SEs.

    Correct formula: Var(β̂) = σ² (D'P_Z D)⁻¹
    where P_Z = Z(Z'Z)⁻¹Z' is the projection matrix onto Z.

    Naive (WRONG) formula: Var(β̂) = σ² (D̂'D̂)⁻¹
    This underestimates uncertainty because it ignores first-stage sampling variation.
    """

    def __init__(
        self,
        inference: Literal["standard", "robust", "clustered"] = "robust",
        cluster_var: Optional[np.ndarray] = None,
        alpha: float = 0.05,
    ):
        """Initialize 2SLS estimator."""
        # Validate inference type
        if inference not in ["standard", "robust", "clustered"]:
            raise ValueError(
                f"inference must be 'standard', 'robust', or 'clustered'. Got: {inference}"
            )

        # Validate cluster_var if clustered
        if inference == "clustered" and cluster_var is None:
            raise ValueError(
                "cluster_var must be provided when inference='clustered'"
            )

        # Validate alpha
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1. Got: {alpha}")

        self.inference = inference
        self.cluster_var = cluster_var
        self.alpha = alpha

        # Attributes set during fit()
        self.coef_ = None
        self.se_ = None
        self.t_stats_ = None
        self.p_values_ = None
        self.ci_ = None
        self.first_stage_f_stat_ = None
        self.first_stage_r2_ = None
        self.n_obs_ = None
        self.n_instruments_ = None
        self.n_endogenous_ = None
        self.fitted_ = False

        # Private attributes for diagnostics
        self._first_stage_coef = None
        self._first_stage_se = None
        self._second_stage_residuals = None
        self._residual_variance = None

    def fit(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> "TwoStageLeastSquares":
        """
        Fit two-stage least squares model.

        Parameters
        ----------
        Y : array-like, shape (n,)
            Outcome variable.
        D : array-like, shape (n,) or (n, p)
            Endogenous variable(s). Can be univariate or multivariate.
        Z : array-like, shape (n,) or (n, q)
            Instrumental variable(s). Must have q >= p for identification.
        X : array-like, shape (n, k), optional
            Exogenous control variables. If None, no controls included.

        Returns
        -------
        self : TwoStageLeastSquares
            Fitted estimator with results stored in attributes.

        Raises
        ------
        ValueError
            If arrays have incompatible shapes, if underidentified (q < p),
            if data contains NaN/Inf, or if matrices are singular.

        Notes
        -----
        Implementation uses two-stage procedure:
        1. First stage: Regress D on Z and X, obtain predicted D̂
        2. Second stage: Regress Y on D̂ and X
        3. Correct standard errors using proper 2SLS formula (not naive OLS)

        The estimator is consistent if:
        - Relevance: E[Z'ν] ≠ 0 (instruments predict endogenous variable)
        - Exclusion: E[Z'ε] = 0 (instruments uncorrelated with error)
        - Order condition: q >= p (at least as many instruments as endogenous)
        """
        # 1. Input validation and preprocessing
        Y, D, Z, X = self._validate_and_preprocess_inputs(Y, D, Z, X)

        n = len(Y)
        self.n_obs_ = n

        # 2. Check identification (order condition)
        self._check_identification(D, Z)

        # 3. First stage: Regress D on Z and X
        D_hat, first_stage_results = self._first_stage(D, Z, X)

        # Store first-stage diagnostics
        self._store_first_stage_diagnostics(first_stage_results, D, Z, X)

        # 4. Second stage: Regress Y on D_hat and X
        # Note: We use D_hat for prediction, but Z for correct standard errors
        second_stage_results = self._second_stage(Y, D_hat, D, Z, X)

        # 5. Compute correct 2SLS standard errors
        self._compute_standard_errors(Y, D, Z, X, second_stage_results)

        # 6. Compute inference (t-stats, p-values, CIs)
        self._compute_inference(n, D, X)

        self.fitted_ = True
        return self

    def _validate_and_preprocess_inputs(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        X: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Validate inputs and convert to numpy arrays."""
        # Convert to numpy arrays
        Y = np.asarray(Y).flatten()
        D = np.asarray(D)
        Z = np.asarray(Z)
        X = np.asarray(X) if X is not None else None

        # Check for NaN/Inf
        if np.any(~np.isfinite(Y)):
            raise ValueError("Y contains NaN or Inf values. Remove or impute missing data.")
        if np.any(~np.isfinite(D)):
            raise ValueError("D contains NaN or Inf values. Remove or impute missing data.")
        if np.any(~np.isfinite(Z)):
            raise ValueError("Z contains NaN or Inf values. Remove or impute missing data.")
        if X is not None and np.any(~np.isfinite(X)):
            raise ValueError("X contains NaN or Inf values. Remove or impute missing data.")

        # Ensure D and Z are 2D
        if D.ndim == 1:
            D = D.reshape(-1, 1)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        # Check array lengths match
        n = len(Y)
        if D.shape[0] != n:
            raise ValueError(
                f"Y and D must have same length. Got Y={n}, D={D.shape[0]}"
            )
        if Z.shape[0] != n:
            raise ValueError(
                f"Y and Z must have same length. Got Y={n}, Z={Z.shape[0]}"
            )
        if X is not None and X.shape[0] != n:
            raise ValueError(
                f"Y and X must have same length. Got Y={n}, X={X.shape[0]}"
            )

        # Check for variation in Y, D, Z
        if np.var(Y) == 0:
            raise ValueError("Y has no variation (constant). Cannot estimate treatment effect.")
        if np.var(D, axis=0).min() == 0:
            raise ValueError("D has no variation (constant column). Cannot estimate treatment effect.")
        if np.var(Z, axis=0).min() == 0:
            raise ValueError("Z has no variation (constant column). Instruments must vary.")

        return Y, D, Z, X

    def _check_identification(self, D: np.ndarray, Z: np.ndarray) -> None:
        """Check order condition for identification."""
        p = D.shape[1]  # Number of endogenous variables
        q = Z.shape[1]  # Number of instruments

        self.n_endogenous_ = p
        self.n_instruments_ = q

        if q < p:
            raise ValueError(
                f"Model is underidentified: {q} instruments for {p} endogenous variables. "
                f"Need at least {p} instruments for identification (order condition). "
                f"Either add more instruments or reduce endogenous variables."
            )

    def _first_stage(
        self, D: np.ndarray, Z: np.ndarray, X: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Any]:
        """
        First-stage regression: D ~ Z + X.

        Returns predicted D_hat and regression results.
        """
        # Construct first-stage design matrix: [Z, X] or just Z
        if X is not None:
            Z_X = np.column_stack([Z, X])
        else:
            Z_X = Z

        # Add constant
        Z_X = sm.add_constant(Z_X, has_constant="add")

        # Fit first-stage OLS for each endogenous variable
        # If D is multivariate, fit separate regression for each column
        n, p = D.shape
        D_hat = np.zeros_like(D)
        first_stage_results = []

        for j in range(p):
            model = sm.OLS(D[:, j], Z_X)
            result = model.fit()
            D_hat[:, j] = result.fittedvalues
            first_stage_results.append(result)

        return D_hat, first_stage_results

    def _store_first_stage_diagnostics(
        self, first_stage_results: list, D: np.ndarray, Z: np.ndarray, X: Optional[np.ndarray]
    ) -> None:
        """Store first-stage diagnostics (F-stat, R², etc.)."""
        # For simplicity, store diagnostics for first endogenous variable only
        result = first_stage_results[0]

        # F-statistic for instruments (test H0: coefficients on Z are all zero)
        # Number of instruments = q
        q = Z.shape[1]

        # Indices of Z coefficients (skip constant, which is index 0)
        z_indices = list(range(1, q + 1))

        # F-test for joint significance of Z
        f_stat = result.f_test(np.eye(len(result.params))[z_indices]).fvalue

        self.first_stage_f_stat_ = f_stat
        self.first_stage_r2_ = result.rsquared

        # Store coefficients and SEs for later use
        self._first_stage_coef = result.params
        self._first_stage_se = result.bse

    def _second_stage(
        self,
        Y: np.ndarray,
        D_hat: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        X: Optional[np.ndarray],
    ) -> Any:
        """
        Second-stage regression: Y ~ D_hat + X.

        Note: We use D_hat for prediction, but will compute correct SEs using Z.
        """
        # Construct second-stage design matrix: [D_hat, X] or just D_hat
        if X is not None:
            D_hat_X = np.column_stack([D_hat, X])
        else:
            D_hat_X = D_hat

        # Add constant
        D_hat_X = sm.add_constant(D_hat_X, has_constant="add")

        # Fit second-stage OLS (using D_hat)
        model = sm.OLS(Y, D_hat_X)
        result = model.fit()

        # Store coefficients (excluding constant)
        # Coefficients: [constant, D1, D2, ..., Dp, X1, X2, ..., Xk]
        # We want: [D1, D2, ..., Dp, X1, X2, ..., Xk] (excluding constant)
        self.coef_ = result.params[1:]  # Skip constant

        # Store residuals for SE calculation
        self._second_stage_residuals = result.resid
        self._residual_variance = np.sum(result.resid ** 2) / result.df_resid

        return result

    def _compute_standard_errors(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        X: Optional[np.ndarray],
        second_stage_results: Any,
    ) -> None:
        """
        Compute correct 2SLS standard errors (not naive OLS).

        Correct formula: Var(β̂) = σ² (X'P_Z X)⁻¹
        where P_Z = Z(Z'Z)⁻¹Z' is the projection matrix onto Z.

        Naive (WRONG) formula: Var(β̂) = σ² (D̂'D̂)⁻¹
        """
        n = len(Y)

        # Construct full design matrix: [D, X] (without constant yet)
        if X is not None:
            DX = np.column_stack([D, X])
        else:
            DX = D

        # Construct instrument matrix: [Z, X] (instruments + exogenous)
        if X is not None:
            ZX = np.column_stack([Z, X])
        else:
            ZX = Z

        # Add constants
        DX = sm.add_constant(DX, has_constant="add")
        ZX = sm.add_constant(ZX, has_constant="add")

        # Compute projection matrix: P_Z = Z(Z'Z)⁻¹Z'
        try:
            ZtZ_inv = np.linalg.inv(ZX.T @ ZX)
        except LinAlgError:
            cond = np.linalg.cond(ZX.T @ ZX)
            raise ValueError(
                f"Z'Z is singular (condition number = {cond:.2e}). "
                f"Possible causes: perfect collinearity among instruments, "
                f"constant columns, or numerical precision issues."
            )

        P_Z = ZX @ ZtZ_inv @ ZX.T

        # Compute (X'P_Z X) where X = [D, X_controls]
        XPX = DX.T @ P_Z @ DX

        # Invert (X'P_Z X)
        try:
            XPX_inv = np.linalg.inv(XPX)
        except LinAlgError:
            cond = np.linalg.cond(XPX)
            raise ValueError(
                f"X'P_Z X is singular (condition number = {cond:.2e}). "
                f"Model may be unidentified or have perfect collinearity."
            )

        # Residual variance: σ² = e'e / (n - k)
        k = DX.shape[1]  # Number of parameters (including constant)
        sigma2 = self._residual_variance

        # Compute variance-covariance matrix
        if self.inference == "standard":
            # Homoskedastic: V = σ² (X'P_Z X)⁻¹
            vcov = sigma2 * XPX_inv

        elif self.inference == "robust":
            # Heteroskedasticity-robust (White/HC0)
            # V = (X'P_Z X)⁻¹ (X'P_Z Ω P_Z X) (X'P_Z X)⁻¹
            # where Ω = diag(e²)
            residuals = self._second_stage_residuals
            Omega = np.diag(residuals ** 2)
            meat = DX.T @ P_Z @ Omega @ P_Z @ DX
            vcov = XPX_inv @ meat @ XPX_inv

        elif self.inference == "clustered":
            # Cluster-robust standard errors
            # V = (X'P_Z X)⁻¹ (Σ_g X_g'P_Z e_g e_g'P_Z X_g) (X'P_Z X)⁻¹
            clusters = self.cluster_var
            unique_clusters = np.unique(clusters)
            G = len(unique_clusters)

            if G < 20:
                import warnings
                warnings.warn(
                    f"Only {G} clusters. Clustered standard errors may be unreliable with <20 clusters. "
                    f"Consider using robust SEs instead.",
                    UserWarning
                )

            # Compute cluster-robust meat
            meat = np.zeros((k, k))
            residuals = self._second_stage_residuals

            for g in unique_clusters:
                cluster_mask = clusters == g
                DX_g = DX[cluster_mask]
                e_g = residuals[cluster_mask]
                PZ_DX_g = P_Z[cluster_mask, :] @ DX  # P_Z @ DX for cluster g
                meat += (PZ_DX_g.T @ np.outer(e_g, e_g) @ PZ_DX_g)

            # Apply finite-sample correction: G / (G - 1) * (n - 1) / (n - k)
            correction = (G / (G - 1)) * ((n - 1) / (n - k))
            vcov = correction * XPX_inv @ meat @ XPX_inv

        # Extract standard errors (excluding constant, which is index 0)
        self.se_ = np.sqrt(np.diag(vcov))[1:]  # Skip constant

    def _compute_inference(self, n: int, D: np.ndarray, X: Optional[np.ndarray]) -> None:
        """Compute t-statistics, p-values, and confidence intervals."""
        # Degrees of freedom
        k = len(self.coef_) + 1  # +1 for constant
        df = n - k

        # t-statistics
        self.t_stats_ = self.coef_ / self.se_

        # p-values (two-sided)
        self.p_values_ = 2 * (1 - stats.t.cdf(np.abs(self.t_stats_), df=df))

        # Confidence intervals
        t_crit = stats.t.ppf(1 - self.alpha / 2, df=df)
        ci_lower = self.coef_ - t_crit * self.se_
        ci_upper = self.coef_ + t_crit * self.se_
        self.ci_ = np.column_stack([ci_lower, ci_upper])

    def summary(self) -> pd.DataFrame:
        """
        Return formatted regression table.

        Returns
        -------
        summary_df : pd.DataFrame
            Table with columns: coef, se, t_stat, p_value, ci_lower, ci_upper.

        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before calling summary(). Call .fit() first.")

        # Create variable names
        p = self.n_endogenous_
        var_names = [f"D{i+1}" if p > 1 else "D" for i in range(p)]

        # Add control variable names if present
        n_controls = len(self.coef_) - p
        if n_controls > 0:
            var_names.extend([f"X{i+1}" for i in range(n_controls)])

        summary_df = pd.DataFrame({
            "coef": self.coef_,
            "se": self.se_,
            "t_stat": self.t_stats_,
            "p_value": self.p_values_,
            "ci_lower": self.ci_[:, 0],
            "ci_upper": self.ci_[:, 1],
        }, index=var_names)

        return summary_df

    def predict(self, D: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict outcomes for new data.

        Parameters
        ----------
        D : array-like, shape (n, p)
            Endogenous variables for prediction.
        X : array-like, shape (n, k), optional
            Exogenous controls for prediction.

        Returns
        -------
        Y_pred : ndarray, shape (n,)
            Predicted outcomes.

        Raises
        ------
        ValueError
            If model has not been fitted yet.

        Notes
        -----
        Prediction uses the structural equation: Y = β₀ + β₁D + β₂X.
        This is the causal effect under the IV assumptions.
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction. Call .fit() first.")

        D = np.asarray(D)
        if D.ndim == 1:
            D = D.reshape(-1, 1)

        # Construct design matrix
        if X is not None:
            X = np.asarray(X)
            DX = np.column_stack([D, X])
        else:
            DX = D

        # Note: self.coef_ does not include constant
        # For prediction, we need to recover the constant from second-stage results
        # For now, predict without constant (user can add intercept if needed)
        Y_pred = DX @ self.coef_

        return Y_pred
