"""
Three Stages of Instrumental Variables Regression.

This module provides separate classes for each stage of 2SLS estimation,
enabling detailed inspection of first-stage strength, reduced-form effects,
and second-stage structural parameters.

Stages:
    1. First Stage: D = π₀ + π₁Z + π₂X + ν (instrument relevance)
    2. Reduced Form: Y = γ₀ + γ₁Z + γ₂X + u (total effect of Z on Y)
    3. Second Stage: Y = β₀ + β₁D̂ + β₂X + ε (structural causal effect)

Key Identity (Wald estimator):
    γ = π × β
    (Reduced form) = (First stage) × (Second stage)

References:
    - Angrist & Pischke (2009). Mostly Harmless Econometrics, Section 4.1.2
    - Wooldridge (2010). Econometric Analysis, Chapter 5
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


class FirstStage:
    """
    First-stage regression: D = π₀ + π₁Z + π₂X + ν.

    Tests instrument relevance (do instruments predict endogenous variable?).
    Key diagnostic: F-statistic for H₀: π₁ = 0.

    Parameters
    ----------
    None (stateless class)

    Attributes
    ----------
    coef_ : ndarray
        Coefficients on Z and X (excluding constant).
    se_ : ndarray
        Standard errors.
    f_statistic_ : float
        F-statistic for joint significance of instruments.
    partial_r2_ : float
        Partial R² (variance in D explained by Z controlling for X).
    r2_ : float
        Overall R² of first-stage regression.
    fitted_values_ : ndarray
        Predicted D̂ (used in second stage).
    residuals_ : ndarray
        First-stage residuals ν̂.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 1000
    >>> Z = np.random.normal(0, 1, n)
    >>> D = 2 * Z + np.random.normal(0, 1, n)
    >>> Y = 0.5 * D + np.random.normal(0, 1, n)
    >>>
    >>> first = FirstStage()
    >>> first.fit(D, Z)
    >>> print(f"F-statistic: {first.f_statistic_:.2f}")
    >>> print(f"Partial R²: {first.partial_r2_:.3f}")
    """

    def __init__(self):
        """Initialize first-stage regression."""
        self.coef_ = None
        self.se_ = None
        self.f_statistic_ = None
        self.partial_r2_ = None
        self.r2_ = None
        self.fitted_values_ = None
        self.residuals_ = None
        self._result = None  # Store statsmodels result for advanced usage

    def fit(self, D: np.ndarray, Z: np.ndarray, X: Optional[np.ndarray] = None) -> "FirstStage":
        """
        Fit first-stage regression D ~ Z + X.

        Parameters
        ----------
        D : array-like, shape (n,) or (n, p)
            Endogenous variable(s).
        Z : array-like, shape (n,) or (n, q)
            Instrumental variable(s).
        X : array-like, shape (n, k), optional
            Exogenous control variables.

        Returns
        -------
        self : FirstStage
            Fitted first-stage regression.

        Notes
        -----
        If D is multivariate (p > 1), fits separate regression for each column
        and stores results for first endogenous variable only.
        """
        # Convert to arrays
        D = np.asarray(D)
        Z = np.asarray(Z)
        X = np.asarray(X) if X is not None else None

        # Ensure 2D
        if D.ndim == 1:
            D = D.reshape(-1, 1)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        # Construct design matrix: [Z, X] or just Z
        if X is not None:
            ZX = np.column_stack([Z, X])
        else:
            ZX = Z

        # Add constant
        ZX = sm.add_constant(ZX, has_constant="add")

        # Fit OLS for first endogenous variable (if multivariate)
        model = sm.OLS(D[:, 0], ZX)
        result = model.fit()

        # Store results
        self.coef_ = result.params[1:]  # Exclude constant
        self.se_ = result.bse[1:]
        self.r2_ = result.rsquared
        self.fitted_values_ = result.fittedvalues
        self.residuals_ = result.resid
        self._result = result

        # Compute F-statistic for instruments
        q = Z.shape[1]
        z_indices = list(range(1, q + 1))
        self.f_statistic_ = result.f_test(np.eye(len(result.params))[z_indices]).fvalue

        # Compute partial R² (variance explained by Z controlling for X)
        if X is not None:
            # Partial R² = R²(Z,X) - R²(X)
            # Fit restricted model with only X
            X_with_const = sm.add_constant(X, has_constant="add")
            restricted_model = sm.OLS(D[:, 0], X_with_const)
            restricted_result = restricted_model.fit()
            r2_restricted = restricted_result.rsquared
            self.partial_r2_ = self.r2_ - r2_restricted
        else:
            # No controls: partial R² = R²
            self.partial_r2_ = self.r2_

        return self

    def predict(self, Z: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict D̂ for new instruments.

        Parameters
        ----------
        Z : array-like, shape (n, q)
            New instrumental variables.
        X : array-like, shape (n, k), optional
            New exogenous controls.

        Returns
        -------
        D_hat : ndarray, shape (n,)
            Predicted endogenous variable.
        """
        if self._result is None:
            raise ValueError("Must call fit() before predict()")

        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        if X is not None:
            X = np.asarray(X)
            ZX = np.column_stack([Z, X])
        else:
            ZX = Z

        ZX = sm.add_constant(ZX, has_constant="add")
        return self._result.predict(ZX)

    def summary(self) -> pd.DataFrame:
        """
        Return formatted first-stage regression table.

        Returns
        -------
        summary_df : pd.DataFrame
            Table with columns: coef, se, t_stat, p_value.
        """
        if self._result is None:
            raise ValueError("Must call fit() before summary()")

        # Create variable names
        q = np.sum(self.coef_ != 0)  # Approximate number of instruments
        var_names = [f"Z{i+1}" for i in range(len(self.coef_))]

        return pd.DataFrame({
            "coef": self.coef_,
            "se": self.se_,
            "t_stat": self._result.tvalues[1:],
            "p_value": self._result.pvalues[1:],
        }, index=var_names)


class ReducedForm:
    """
    Reduced-form regression: Y = γ₀ + γ₁Z + γ₂X + u.

    Shows direct effect of instruments on outcome (without explicitly modeling D).
    Combines first-stage and structural effects: γ = π × β.

    Useful for:
    - Anderson-Rubin confidence intervals (robust to weak instruments)
    - Diagnostic plots (visualize Z → Y relationship)
    - Intent-to-treat (ITT) effects

    Parameters
    ----------
    None (stateless class)

    Attributes
    ----------
    coef_ : ndarray
        Coefficients on Z and X (excluding constant).
    se_ : ndarray
        Standard errors.
    r2_ : float
        R² of reduced-form regression.
    fitted_values_ : ndarray
        Predicted Y from reduced form.
    residuals_ : ndarray
        Reduced-form residuals.

    Examples
    --------
    >>> # Wald estimator identity: γ = π × β
    >>> first = FirstStage().fit(D, Z)
    >>> reduced = ReducedForm().fit(Y, Z)
    >>> iv = TwoStageLeastSquares().fit(Y, D, Z)
    >>>
    >>> # Reduced form coef ≈ First stage coef × Second stage coef
    >>> gamma = reduced.coef_[0]
    >>> pi = first.coef_[0]
    >>> beta = iv.coef_[0]
    >>> assert np.isclose(gamma, pi * beta)
    """

    def __init__(self):
        """Initialize reduced-form regression."""
        self.coef_ = None
        self.se_ = None
        self.r2_ = None
        self.fitted_values_ = None
        self.residuals_ = None
        self._result = None

    def fit(self, Y: np.ndarray, Z: np.ndarray, X: Optional[np.ndarray] = None) -> "ReducedForm":
        """
        Fit reduced-form regression Y ~ Z + X.

        Parameters
        ----------
        Y : array-like, shape (n,)
            Outcome variable.
        Z : array-like, shape (n,) or (n, q)
            Instrumental variable(s).
        X : array-like, shape (n, k), optional
            Exogenous control variables.

        Returns
        -------
        self : ReducedForm
            Fitted reduced-form regression.
        """
        # Convert to arrays
        Y = np.asarray(Y).flatten()
        Z = np.asarray(Z)
        X = np.asarray(X) if X is not None else None

        # Ensure Z is 2D
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        # Construct design matrix: [Z, X] or just Z
        if X is not None:
            ZX = np.column_stack([Z, X])
        else:
            ZX = Z

        # Add constant
        ZX = sm.add_constant(ZX, has_constant="add")

        # Fit OLS
        model = sm.OLS(Y, ZX)
        result = model.fit()

        # Store results
        self.coef_ = result.params[1:]  # Exclude constant
        self.se_ = result.bse[1:]
        self.r2_ = result.rsquared
        self.fitted_values_ = result.fittedvalues
        self.residuals_ = result.resid
        self._result = result

        return self

    def summary(self) -> pd.DataFrame:
        """
        Return formatted reduced-form regression table.

        Returns
        -------
        summary_df : pd.DataFrame
            Table with columns: coef, se, t_stat, p_value.
        """
        if self._result is None:
            raise ValueError("Must call fit() before summary()")

        var_names = [f"Z{i+1}" if i < len(self.coef_) else f"X{i+1}" for i in range(len(self.coef_))]

        return pd.DataFrame({
            "coef": self.coef_,
            "se": self.se_,
            "t_stat": self._result.tvalues[1:],
            "p_value": self._result.pvalues[1:],
        }, index=var_names)


class SecondStage:
    """
    Second-stage regression: Y = β₀ + β₁D̂ + β₂X + ε.

    Structural equation with corrected standard errors.

    WARNING: This class is primarily for educational purposes.
    For production use, call TwoStageLeastSquares.fit() which handles
    correct standard errors automatically.

    Parameters
    ----------
    None (stateless class)

    Attributes
    ----------
    coef_ : ndarray
        Coefficients on D̂ and X (excluding constant).
    se_naive_ : ndarray
        Naive OLS standard errors (INCORRECT for inference).
    r2_ : float
        R² of second-stage regression.
    fitted_values_ : ndarray
        Predicted Y.
    residuals_ : ndarray
        Second-stage residuals.

    Notes
    -----
    Standard errors from this class are NAIVE (treat D̂ as given).
    They are BIASED DOWNWARD and should NOT be used for inference.

    Use TwoStageLeastSquares for correct 2SLS standard errors.

    Examples
    --------
    >>> # Manual 2SLS (educational)
    >>> first = FirstStage().fit(D, Z, X)
    >>> D_hat = first.fitted_values_
    >>>
    >>> second = SecondStage().fit(Y, D_hat, X)
    >>> print(f"Coefficient: {second.coef_[0]:.3f}")
    >>> print("WARNING: Standard errors are incorrect!")
    >>>
    >>> # Correct 2SLS (production)
    >>> iv = TwoStageLeastSquares().fit(Y, D, Z, X)
    >>> print(f"Coefficient: {iv.coef_[0]:.3f}")
    >>> print(f"Correct SE: {iv.se_[0]:.3f}")
    """

    def __init__(self):
        """Initialize second-stage regression."""
        self.coef_ = None
        self.se_naive_ = None  # Explicitly marked as naive
        self.r2_ = None
        self.fitted_values_ = None
        self.residuals_ = None
        self._result = None

    def fit(
        self,
        Y: np.ndarray,
        D_hat: np.ndarray,
        X: Optional[np.ndarray] = None,
        first_stage_residuals: Optional[np.ndarray] = None,
    ) -> "SecondStage":
        """
        Fit second-stage regression Y ~ D̂ + X.

        Parameters
        ----------
        Y : array-like, shape (n,)
            Outcome variable.
        D_hat : array-like, shape (n,) or (n, p)
            Predicted endogenous variable from first stage.
        X : array-like, shape (n, k), optional
            Exogenous control variables.
        first_stage_residuals : array-like, optional
            First-stage residuals (for diagnostic purposes, not used in fit).

        Returns
        -------
        self : SecondStage
            Fitted second-stage regression.

        Notes
        -----
        Standard errors are NAIVE (incorrect). Use TwoStageLeastSquares
        for correct 2SLS standard errors.
        """
        # Convert to arrays
        Y = np.asarray(Y).flatten()
        D_hat = np.asarray(D_hat)
        X = np.asarray(X) if X is not None else None

        # Ensure D_hat is 2D
        if D_hat.ndim == 1:
            D_hat = D_hat.reshape(-1, 1)

        # Construct design matrix: [D_hat, X] or just D_hat
        if X is not None:
            D_hat_X = np.column_stack([D_hat, X])
        else:
            D_hat_X = D_hat

        # Add constant
        D_hat_X = sm.add_constant(D_hat_X, has_constant="add")

        # Fit OLS
        model = sm.OLS(Y, D_hat_X)
        result = model.fit()

        # Store results
        self.coef_ = result.params[1:]  # Exclude constant
        self.se_naive_ = result.bse[1:]  # Mark as naive
        self.r2_ = result.rsquared
        self.fitted_values_ = result.fittedvalues
        self.residuals_ = result.resid
        self._result = result

        return self

    def summary(self) -> pd.DataFrame:
        """
        Return formatted second-stage regression table.

        WARNING: Standard errors are INCORRECT (naive OLS).

        Returns
        -------
        summary_df : pd.DataFrame
            Table with columns: coef, se_naive, t_stat_naive, p_value_naive.
        """
        if self._result is None:
            raise ValueError("Must call fit() before summary()")

        p = np.sum([c != 0 for c in self.coef_ if c is not None])
        var_names = [f"D{i+1}" if i < p else f"X{i-p+1}" for i in range(len(self.coef_))]

        return pd.DataFrame({
            "coef": self.coef_,
            "se_naive": self.se_naive_,
            "t_stat_naive": self._result.tvalues[1:],
            "p_value_naive": self._result.pvalues[1:],
        }, index=var_names)
