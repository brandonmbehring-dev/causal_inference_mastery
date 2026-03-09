"""
Fuller k-class estimator for instrumental variables.

Fuller's modified LIML estimator improves finite-sample properties by applying
a bias correction to the LIML kappa parameter.

Mathematical Details
--------------------
Fuller is a k-class estimator with:

    k_Fuller = k_LIML - α/(n - L)

where:
    k_LIML = smallest eigenvalue from LIML
    α = adjustment parameter (Fuller-1: α=1, Fuller-4: α=4)
    n = sample size
    L = number of instruments + exogenous variables

Fuller-1 (α=1) is often recommended as it balances bias and variance better
than LIML while maintaining robustness to weak instruments.

References
----------
- Fuller, W. A. (1977). Some properties of a modification of the limited
  information estimator. Econometrica, 45(4), 939-953.

- Hahn, J., Hausman, J., & Kuersteiner, G. (2004). Estimation with weak
  instruments: Accuracy of higher-order bias and MSE approximations.
  Econometrics Journal, 7(1), 272-306.
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd

from .liml import LIML


class Fuller:
    """
    Fuller k-class estimator.

    Fuller modifies LIML by applying a finite-sample bias correction:
    k_Fuller = k_LIML - α/(n - L), where α is the adjustment parameter.

    Fuller-1 (α=1) is the most commonly recommended variant, providing
    better finite-sample properties than LIML while maintaining robustness
    to weak instruments.

    Parameters
    ----------
    alpha_param : float, default=1.0
        Fuller adjustment parameter. Common choices:
        - alpha_param=1.0: Fuller-1 (recommended)
        - alpha_param=4.0: Fuller-4 (more conservative)

    inference : {'standard', 'robust'}, default='robust'
        Type of standard errors to compute.

    alpha : float, default=0.05
        Significance level for confidence intervals.

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
    kappa_ : float
        Fuller kappa parameter (LIML kappa - α/(n-L)).
    kappa_liml_ : float
        Unadjusted LIML kappa (before Fuller correction).
    n_obs_ : int
        Number of observations.
    n_instruments_ : int
        Number of instruments.
    n_endogenous_ : int
        Number of endogenous variables.

    Examples
    --------
    >>> from causal_inference.iv import Fuller
    >>> import numpy as np
    >>>
    >>> # Generate data with weak instruments
    >>> np.random.seed(42)
    >>> n = 1000
    >>> Z = np.random.normal(0, 1, n)
    >>> D = 0.3 * Z + np.random.normal(0, 1, n)  # Weak first stage
    >>> Y = 0.5 * D + np.random.normal(0, 1, n)
    >>>
    >>> # Fit Fuller-1
    >>> fuller = Fuller(alpha_param=1.0, inference='robust')
    >>> fuller.fit(Y, D, Z)
    >>> print(f"Fuller-1 estimate: {fuller.coef_[0]:.3f}")
    >>> print(f"Fuller kappa: {fuller.kappa_:.3f}")
    >>> print(f"LIML kappa: {fuller.kappa_liml_:.3f}")

    Notes
    -----
    - Fuller-1 typically has lower MSE than LIML in finite samples
    - Fuller is less sensitive to weak instruments than 2SLS
    - As n → ∞, Fuller → LIML → 2SLS (asymptotic equivalence)
    """

    def __init__(
        self,
        alpha_param: float = 1.0,
        inference: Literal["standard", "robust"] = "robust",
        alpha: float = 0.05,
    ):
        if alpha_param <= 0:
            raise ValueError(f"alpha_param must be positive, got {alpha_param}")
        if inference not in ["standard", "robust"]:
            raise ValueError(f"inference must be 'standard' or 'robust', got '{inference}'")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha_param = alpha_param
        self.inference = inference
        self.alpha = alpha

        # Fitted attributes
        self.coef_: Optional[np.ndarray] = None
        self.se_: Optional[np.ndarray] = None
        self.t_stats_: Optional[np.ndarray] = None
        self.p_values_: Optional[np.ndarray] = None
        self.ci_: Optional[np.ndarray] = None
        self.kappa_: Optional[float] = None
        self.kappa_liml_: Optional[float] = None
        self.n_obs_: Optional[int] = None
        self.n_instruments_: Optional[int] = None
        self.n_endogenous_: Optional[int] = None

        # Internal LIML instance for kappa calculation
        self._liml = LIML(inference=inference, alpha=alpha)

    def fit(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> "Fuller":
        """
        Fit Fuller k-class estimator.

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
        self : Fuller
            Fitted estimator.

        Raises
        ------
        ValueError
            If inputs are invalid or model is underidentified.
        """
        # First, fit LIML to get unadjusted kappa
        self._liml.fit(Y, D, Z, X)

        # Store LIML kappa
        self.kappa_liml_ = self._liml.kappa_

        # Store dimensions
        self.n_obs_ = self._liml.n_obs_
        self.n_instruments_ = self._liml.n_instruments_
        self.n_endogenous_ = self._liml.n_endogenous_

        # Fuller bias correction: k = k_LIML - α/(n - L)
        # where L = number of instruments + exogenous variables (including intercept)
        n = self.n_obs_
        L = self.n_instruments_

        # Add number of exogenous variables (if any)
        if X is not None:
            L += X.shape[1] if X.ndim > 1 else 1

        # Add 1 for intercept (always present)
        L += 1

        # Fuller adjustment
        correction = self.alpha_param / (n - L)
        self.kappa_ = self.kappa_liml_ - correction

        # Check if Fuller kappa is positive
        if self.kappa_ <= 0:
            raise ValueError(
                f"Fuller kappa = {self.kappa_:.4f} is non-positive "
                f"(LIML kappa = {self.kappa_liml_:.4f}, correction = {correction:.4f}). "
                "This indicates extremely weak instruments or small sample size. "
                "Try using a smaller alpha_param or 2SLS estimator."
            )

        # Use k-class estimation with Fuller kappa
        # We'll reuse LIML's k-class estimation method
        from .liml import LIML as LIML_base

        # Validate inputs (same as LIML)
        Y, D, Z, X = self._validate_inputs(Y, D, Z, X)

        # Ensure 2D
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if D.ndim == 1:
            D = D.reshape(-1, 1)

        # Add intercept
        if X is None:
            X_with_intercept = np.ones((n, 1))
        else:
            X_with_intercept = np.column_stack([np.ones(n), X])

        # k-class estimation with Fuller kappa
        coef, residuals = self._liml._k_class_estimation(Y, D, Z, X_with_intercept, self.kappa_)

        # Standard errors (same formula as LIML)
        se = self._liml._compute_standard_errors(
            Y, D, Z, X_with_intercept, coef, residuals, self.kappa_
        )

        # Store results
        self.coef_ = coef
        self.se_ = se

        # Inference
        from scipy import stats as sp_stats

        self.t_stats_ = coef / se
        self.p_values_ = 2 * (1 - sp_stats.t.cdf(np.abs(self.t_stats_), df=n - len(coef)))

        # Confidence intervals
        t_crit = sp_stats.t.ppf(1 - self.alpha / 2, df=n - len(coef))
        ci_lower = coef - t_crit * se
        ci_upper = coef + t_crit * se
        self.ci_ = np.column_stack([ci_lower, ci_upper])

        return self

    def _validate_inputs(self, Y, D, Z, X):
        """Validate and preprocess inputs (same as LIML)."""
        return self._liml._validate_inputs(Y, D, Z, X)

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
