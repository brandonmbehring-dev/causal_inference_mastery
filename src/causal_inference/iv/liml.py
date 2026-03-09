"""
Limited Information Maximum Likelihood (LIML) estimator for instrumental variables.

LIML is less biased than 2SLS with weak or many instruments, though it has higher
variance in small samples.

Mathematical Details
--------------------
LIML is a k-class estimator where k = λ (smallest eigenvalue):

    β_LIML = (D'(I - λ*M_Z)D)^(-1) (D'(I - λ*M_Z)Y)

where:
    λ = smallest eigenvalue of (Y,D)'M_X(Y,D) / (Y,D)'M_Z(Y,D)
    M_X = I - X(X'X)^(-1)X' (annihilator matrix for X)
    M_Z = I - Z(Z'Z)^(-1)Z' (annihilator matrix for Z)

When instruments are strong, LIML ≈ 2SLS. With weak instruments, LIML is less biased
but has higher variance.

References
----------
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single
  equation in a complete system of stochastic equations. Annals of Mathematical
  Statistics, 20(1), 46-63.

- Stock, J. H., Wright, J. H., & Yogo, M. (2002). A survey of weak instruments and
  weak identification in generalized method of moments. Journal of Business & Economic
  Statistics, 20(4), 518-529.
"""

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class LIML:
    """
    Limited Information Maximum Likelihood (LIML) estimator.

    LIML is a k-class estimator that uses the smallest eigenvalue of the
    canonical correlation matrix. It is less biased than 2SLS with weak
    instruments, though it has higher variance in small samples.

    Parameters
    ----------
    inference : {'standard', 'robust'}, default='robust'
        Type of standard errors to compute:
        - 'standard': Homoskedastic standard errors
        - 'robust': Heteroskedasticity-robust (HC0)

    alpha : float, default=0.05
        Significance level for confidence intervals (0 < alpha < 1).

    Attributes
    ----------
    coef_ : np.ndarray
        Coefficient estimates (endogenous + controls).
    se_ : np.ndarray
        Standard errors.
    t_stats_ : np.ndarray
        t-statistics for coefficients.
    p_values_ : np.ndarray
        Two-sided p-values.
    ci_ : np.ndarray
        Confidence intervals (n_params × 2).
    kappa_ : float
        LIML kappa parameter (smallest eigenvalue).
    n_obs_ : int
        Number of observations.
    n_instruments_ : int
        Number of instruments.
    n_endogenous_ : int
        Number of endogenous variables.

    Examples
    --------
    >>> from causal_inference.iv import LIML
    >>> import numpy as np
    >>>
    >>> # Generate data with weak instruments
    >>> np.random.seed(42)
    >>> n = 1000
    >>> Z = np.random.normal(0, 1, n)
    >>> D = 0.3 * Z + np.random.normal(0, 1, n)  # Weak first stage
    >>> Y = 0.5 * D + np.random.normal(0, 1, n)
    >>>
    >>> # Fit LIML
    >>> liml = LIML(inference='robust')
    >>> liml.fit(Y, D, Z)
    >>> print(f"LIML estimate: {liml.coef_[0]:.3f}")
    >>> print(f"Kappa: {liml.kappa_:.3f}")

    Notes
    -----
    - LIML is median-unbiased but may have higher variance than 2SLS
    - With very weak instruments (F < 5), LIML can be unstable
    - Fuller k-class estimator addresses LIML's higher variance
    """

    def __init__(
        self,
        inference: Literal["standard", "robust"] = "robust",
        alpha: float = 0.05,
    ):
        if inference not in ["standard", "robust"]:
            raise ValueError(f"inference must be 'standard' or 'robust', got '{inference}'")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.inference = inference
        self.alpha = alpha

        # Fitted attributes
        self.coef_: Optional[np.ndarray] = None
        self.se_: Optional[np.ndarray] = None
        self.t_stats_: Optional[np.ndarray] = None
        self.p_values_: Optional[np.ndarray] = None
        self.ci_: Optional[np.ndarray] = None
        self.kappa_: Optional[float] = None
        self.n_obs_: Optional[int] = None
        self.n_instruments_: Optional[int] = None
        self.n_endogenous_: Optional[int] = None

    def fit(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> "LIML":
        """
        Fit LIML estimator.

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
        self : LIML
            Fitted estimator.

        Raises
        ------
        ValueError
            If inputs are invalid or model is underidentified.
        """
        # Input validation
        Y, D, Z, X = self._validate_inputs(Y, D, Z, X)

        n = len(Y)
        self.n_obs_ = n

        # Ensure 2D arrays for shape calculations
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if D.ndim == 1:
            D = D.reshape(-1, 1)

        self.n_instruments_ = Z.shape[1]
        self.n_endogenous_ = D.shape[1]

        # Check identification
        if self.n_instruments_ < self.n_endogenous_:
            raise ValueError(
                f"Model is underidentified: {self.n_instruments_} instruments for "
                f"{self.n_endogenous_} endogenous variables. Need at least "
                f"{self.n_endogenous_} instruments."
            )

        # Add intercept to controls
        if X is None:
            X = np.ones((n, 1))
        else:
            X = np.column_stack([np.ones(n), X])

        # Compute LIML kappa (smallest eigenvalue)
        kappa = self._compute_kappa(Y, D, Z, X)
        self.kappa_ = kappa

        # Check for numerical issues
        if kappa < 1e-6:
            raise ValueError(
                f"LIML failed: kappa = {kappa:.2e} is too close to zero. "
                "This usually indicates very weak instruments or numerical issues. "
                "Try using 2SLS or Fuller estimator instead."
            )

        # k-class estimation with k = kappa
        coef, residuals = self._k_class_estimation(Y, D, Z, X, kappa)

        # Compute standard errors
        se = self._compute_standard_errors(Y, D, Z, X, coef, residuals, kappa)

        # Inference
        t_stats = coef / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - len(coef)))

        # Confidence intervals
        t_crit = stats.t.ppf(1 - self.alpha / 2, df=n - len(coef))
        ci_lower = coef - t_crit * se
        ci_upper = coef + t_crit * se

        # Store results
        self.coef_ = coef
        self.se_ = se
        self.t_stats_ = t_stats
        self.p_values_ = p_values
        self.ci_ = np.column_stack([ci_lower, ci_upper])

        return self

    def _compute_kappa(self, Y: np.ndarray, D: np.ndarray, Z: np.ndarray, X: np.ndarray) -> float:
        """
        Compute LIML kappa parameter (smallest eigenvalue).

        Parameters
        ----------
        Y, D, Z, X : np.ndarray
            Data arrays (validated).

        Returns
        -------
        kappa : float
            Smallest eigenvalue of (Y,D)'M_X(Y,D) / (Y,D)'M_Z(Y,D).
        """
        n = len(Y)

        # Annihilator matrices
        # M_X = I - X(X'X)^(-1)X'
        XtX_inv = np.linalg.inv(X.T @ X)
        P_X = X @ XtX_inv @ X.T
        M_X = np.eye(n) - P_X

        # M_Z = I - [Z, X]([Z, X]'[Z, X])^(-1)[Z, X]'
        ZX = np.column_stack([Z, X])
        ZXtZX_inv = np.linalg.inv(ZX.T @ ZX)
        P_ZX = ZX @ ZXtZX_inv @ ZX.T
        M_ZX = np.eye(n) - P_ZX

        # Stack (Y, D)
        if D.ndim == 1:
            D = D.reshape(-1, 1)
        YD = np.column_stack([Y, D])

        # Numerator: (Y,D)' M_X (Y,D)
        numerator = YD.T @ M_X @ YD

        # Denominator: (Y,D)' M_ZX (Y,D)
        denominator = YD.T @ M_ZX @ YD

        # Solve generalized eigenvalue problem: numerator @ v = lambda * denominator @ v
        from scipy.linalg import eigh

        try:
            eigvals, _ = eigh(numerator, denominator)
        except np.linalg.LinAlgError:
            # Fallback: Use standard eigenvalue of inv(denominator) @ numerator
            denom_inv = np.linalg.inv(denominator)
            eigvals = np.linalg.eigvalsh(denom_inv @ numerator)

        # LIML kappa = smallest eigenvalue
        kappa = np.min(eigvals)

        return float(kappa)

    def _k_class_estimation(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        X: np.ndarray,
        k: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        k-class estimation: β_k = (D'(I - k*M_Z)D)^(-1) (D'(I - k*M_Z)Y).

        Parameters
        ----------
        Y, D, Z, X : np.ndarray
            Data arrays.
        k : float
            k-class parameter (k=1 for 2SLS, k=kappa for LIML).

        Returns
        -------
        coef : np.ndarray
            Coefficient estimates.
        residuals : np.ndarray
            Residuals from k-class regression.
        """
        n = len(Y)

        # Projection matrix P_Z = [Z, X]([Z, X]'[Z, X])^(-1)[Z, X]'
        ZX = np.column_stack([Z, X])
        ZXtZX_inv = np.linalg.inv(ZX.T @ ZX)
        P_ZX = ZX @ ZXtZX_inv @ ZX.T

        # Annihilator M_Z = I - P_Z
        M_ZX = np.eye(n) - P_ZX

        # k-class transformation: W = I - k*M_Z
        W = np.eye(n) - k * M_ZX

        # Stack [D, X]
        if D.ndim == 1:
            D = D.reshape(-1, 1)
        DX = np.column_stack([D, X])

        # k-class normal equations: (DX' W DX) beta = DX' W Y
        DWD = DX.T @ W @ DX
        DWY = DX.T @ W @ Y

        try:
            coef = np.linalg.solve(DWD, DWY)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Singular matrix in k-class estimation. "
                "Possible causes: perfect collinearity or numerical instability."
            )

        # Residuals
        residuals = Y - DX @ coef

        return coef, residuals

    def _compute_standard_errors(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        X: np.ndarray,
        coef: np.ndarray,
        residuals: np.ndarray,
        k: float,
    ) -> np.ndarray:
        """
        Compute standard errors for k-class estimator.

        Parameters
        ----------
        Y, D, Z, X : np.ndarray
            Data arrays.
        coef : np.ndarray
            Coefficient estimates.
        residuals : np.ndarray
            Residuals.
        k : float
            k-class parameter.

        Returns
        -------
        se : np.ndarray
            Standard errors.
        """
        n = len(Y)

        # Stack [D, X]
        if D.ndim == 1:
            D = D.reshape(-1, 1)
        DX = np.column_stack([D, X])

        # Projection matrix P_ZX
        ZX = np.column_stack([Z, X])
        ZXtZX_inv = np.linalg.inv(ZX.T @ ZX)
        P_ZX = ZX @ ZXtZX_inv @ ZX.T

        # For k-class estimator, covariance is: σ² * (DX' P_ZX DX)^(-1)
        # (Same as 2SLS formula since LIML is a k-class estimator)
        DPD = DX.T @ P_ZX @ DX
        DPD_inv = np.linalg.inv(DPD)

        # Residual variance
        sigma_sq = (residuals.T @ residuals) / (n - len(coef))

        if self.inference == "standard":
            # Homoskedastic SEs
            vcov = sigma_sq * DPD_inv
        else:  # robust
            # Heteroskedasticity-robust (HC0)
            Omega = np.diag(residuals**2)
            meat = DX.T @ P_ZX @ Omega @ P_ZX @ DX
            vcov = DPD_inv @ meat @ DPD_inv

        se = np.sqrt(np.diag(vcov))

        return se

    def _validate_inputs(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        X: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Validate and preprocess inputs."""
        # Convert to numpy arrays
        Y = np.asarray(Y).flatten()
        D = np.asarray(D)
        Z = np.asarray(Z)
        if X is not None:
            X = np.asarray(X)

        # Check shapes
        n = len(Y)
        if D.shape[0] != n:
            raise ValueError(f"Y and D must have same length, got {n} and {D.shape[0]}")
        if Z.shape[0] != n:
            raise ValueError(f"Y and Z must have same length, got {n} and {Z.shape[0]}")
        if X is not None and X.shape[0] != n:
            raise ValueError(f"Y and X must have same length, got {n} and {X.shape[0]}")

        # Check for NaNs
        if np.any(np.isnan(Y)):
            raise ValueError("Y contains NaN values")
        if np.any(np.isnan(D)):
            raise ValueError("D contains NaN values")
        if np.any(np.isnan(Z)):
            raise ValueError("Z contains NaN values")
        if X is not None and np.any(np.isnan(X)):
            raise ValueError("X contains NaN values")

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

        return summary
