"""
Nonlinear Control Function estimators for binary outcomes.

The control function approach is essential for nonlinear models with endogeneity
because 2SLS is INVALID for probit/logit. The problem:

- 2SLS substitutes D_hat for D: Pr(Y=1) = Phi(beta*D_hat + ...)
- This is WRONG because E[Phi(beta*D)] != Phi(beta*E[D])
- Jensen's inequality: the expectation of a nonlinear function != nonlinear function of expectation

Solution: Control Function
- Include first-stage residuals as additional regressor
- This "controls for" the endogeneity directly
- Valid for both continuous and discrete endogenous treatments

References
----------
- Rivers & Vuong (1988). "Limited Information Estimators and Exogeneity Tests
  for Simultaneous Probit Models"
- Wooldridge (2010). "Econometric Analysis of Cross Section and Panel Data",
  Chapter 15.7.3
- Blundell & Powell (2004). "Endogeneity in Semiparametric Binary Response Models"
"""

from typing import Any, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats
import statsmodels.api as sm

from .types import FirstStageResult, NonlinearCFResult


class NonlinearControlFunction:
    """
    Nonlinear Control Function estimator for binary outcomes.

    When the outcome Y is binary (0/1), standard 2SLS is invalid because
    of Jensen's inequality. The control function approach includes first-stage
    residuals as an additional control, which "absorbs" the endogeneity.

    Parameters
    ----------
    model_type : {'probit', 'logit'}, default='probit'
        Nonlinear model for the second stage.
    n_bootstrap : int, default=500
        Number of bootstrap iterations for inference.
    alpha : float, default=0.05
        Significance level.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    result_ : NonlinearCFResult
        Estimation results after fitting.
    first_stage_ : FirstStageResult
        First-stage regression results.
    fitted_ : bool
        Whether model has been fitted.

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.control_function import NonlinearControlFunction
    >>>
    >>> # Binary outcome with endogeneity
    >>> np.random.seed(42)
    >>> n = 1000
    >>> Z = np.random.normal(0, 1, n)
    >>> nu = np.random.normal(0, 1, n)
    >>> D = 0.5 * Z + nu  # Endogenous
    >>> latent = 1.0 * D + 0.5 * nu + np.random.logistic(0, 1, n)
    >>> Y = (latent > 0).astype(float)
    >>>
    >>> cf = NonlinearControlFunction(model_type='probit')
    >>> result = cf.fit(Y, D, Z)
    >>> print(f"AME: {result['estimate']:.3f}")

    Notes
    -----
    The procedure:
    1. First stage: D = pi*Z + pi_X*X + nu (OLS for continuous D)
    2. Compute residuals: nu_hat = D - D_hat
    3. Second stage: Pr(Y=1) = Phi(beta*D + rho*nu_hat + gamma*X)
    4. Compute average marginal effect (AME) of D

    The coefficient on nu_hat tests for endogeneity. If rho = 0, simple
    probit/logit would be consistent.
    """

    def __init__(
        self,
        model_type: Literal["probit", "logit"] = "probit",
        n_bootstrap: int = 500,
        alpha: float = 0.05,
        random_state: Optional[int] = None,
    ) -> None:
        self.model_type = model_type
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state
        self.fitted_ = False
        self.result_: Optional[NonlinearCFResult] = None
        self.first_stage_: Optional[FirstStageResult] = None

    def fit(
        self,
        Y: NDArray[np.floating],
        D: NDArray[np.floating],
        Z: NDArray[np.floating],
        X: Optional[NDArray[np.floating]] = None,
    ) -> NonlinearCFResult:
        """
        Fit nonlinear control function model.

        Parameters
        ----------
        Y : array-like, shape (n,)
            Binary outcome (0 or 1).
        D : array-like, shape (n,)
            Endogenous treatment (continuous or binary).
        Z : array-like, shape (n,) or (n, q)
            Instrumental variables.
        X : array-like, shape (n, k), optional
            Exogenous controls.

        Returns
        -------
        NonlinearCFResult
            Estimation results including AME and endogeneity test.

        Raises
        ------
        ValueError
            If Y is not binary or inputs are invalid.
        """
        # Validate inputs
        Y, D, Z, X = self._validate_inputs(Y, D, Z, X)
        n = len(Y)

        # First stage: D ~ Z + X (OLS)
        first_stage = self._first_stage(D, Z, X)
        self.first_stage_ = first_stage
        nu_hat = first_stage["residuals"]

        # Second stage: probit/logit with control function
        # Bootstrap for inference (analytical is complex)
        result = self._bootstrap_estimation(Y, D, Z, X, nu_hat)

        self.result_ = result
        self.fitted_ = True
        return result

    def _validate_inputs(
        self,
        Y: NDArray[np.floating],
        D: NDArray[np.floating],
        Z: NDArray[np.floating],
        X: Optional[NDArray[np.floating]],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        Optional[NDArray[np.float64]],
    ]:
        """Validate inputs for nonlinear CF."""
        Y = np.asarray(Y, dtype=np.float64).ravel()
        D = np.asarray(D, dtype=np.float64).ravel()
        Z = np.asarray(Z, dtype=np.float64)

        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        n = len(Y)

        # Check Y is binary
        unique_y = np.unique(Y)
        if not np.all(np.isin(unique_y, [0, 1])):
            raise ValueError(
                f"Y must be binary (0/1). Found values: {unique_y[:10]}..."
            )

        if len(D) != n:
            raise ValueError(f"Length mismatch: Y ({n}) != D ({len(D)})")
        if len(Z) != n:
            raise ValueError(f"Length mismatch: Y ({n}) != Z ({len(Z)})")

        if X is not None:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if len(X) != n:
                raise ValueError(f"Length mismatch: Y ({n}) != X ({len(X)})")

        # Check for NaN/Inf
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            raise ValueError("NaN or infinite values in Y")
        if np.any(np.isnan(D)) or np.any(np.isinf(D)):
            raise ValueError("NaN or infinite values in D")
        if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
            raise ValueError("NaN or infinite values in Z")
        if X is not None and (np.any(np.isnan(X)) or np.any(np.isinf(X))):
            raise ValueError("NaN or infinite values in X")

        # Minimum sample size (need enough for convergence)
        if n < 50:
            raise ValueError(f"Insufficient sample size (n={n}). Need at least 50.")

        # Check for variation
        if np.std(D) < 1e-10:
            raise ValueError("No variation in treatment D")
        if np.all(np.std(Z, axis=0) < 1e-10):
            raise ValueError("No variation in instruments Z")
        if Y.mean() < 0.05 or Y.mean() > 0.95:
            # Very imbalanced outcomes may cause convergence issues
            pass  # Just a warning, not an error

        return Y, D, Z, X

    def _first_stage(
        self,
        D: NDArray[np.float64],
        Z: NDArray[np.float64],
        X: Optional[NDArray[np.float64]],
    ) -> FirstStageResult:
        """First-stage OLS regression for continuous endogenous variable."""
        n = len(D)
        n_instruments = Z.shape[1]

        # Build design matrix
        if X is not None:
            design = np.column_stack([np.ones(n), Z, X])
            n_controls = X.shape[1]
        else:
            design = np.column_stack([np.ones(n), Z])
            n_controls = 0

        # Fit OLS
        model = sm.OLS(D, design)
        fit = model.fit(cov_type="HC3")

        coefficients = fit.params
        se = fit.bse
        residuals = fit.resid
        fitted_values = fit.fittedvalues
        r2 = fit.rsquared

        # F-statistic for instruments
        if n_instruments == 1:
            f_statistic = fit.tvalues[1] ** 2
            f_pvalue = fit.pvalues[1]
        else:
            r_matrix = np.zeros((n_instruments, len(coefficients)))
            r_matrix[:, 1 : 1 + n_instruments] = np.eye(n_instruments)
            f_test = fit.f_test(r_matrix)
            f_statistic = float(f_test.fvalue)
            f_pvalue = float(f_test.pvalue)

        # Partial R-squared
        if X is not None:
            X_with_const = np.column_stack([np.ones(n), X])
            Z_resid = Z - X_with_const @ np.linalg.lstsq(X_with_const, Z, rcond=None)[0]
            D_resid = D - X_with_const @ np.linalg.lstsq(X_with_const, D, rcond=None)[0]
            partial_r2 = float(np.corrcoef(Z_resid.ravel(), D_resid)[0, 1] ** 2)
        else:
            partial_r2 = r2

        weak_iv_warning = f_statistic < 10

        return FirstStageResult(
            coefficients=coefficients,
            se=se,
            residuals=residuals,
            fitted_values=fitted_values,
            f_statistic=f_statistic,
            f_pvalue=f_pvalue,
            partial_r2=partial_r2,
            r2=r2,
            n_obs=n,
            n_instruments=n_instruments,
            weak_iv_warning=weak_iv_warning,
        )

    def _fit_second_stage(
        self,
        Y: NDArray[np.float64],
        D: NDArray[np.float64],
        nu_hat: NDArray[np.float64],
        X: Optional[NDArray[np.float64]],
    ) -> Tuple[Any, float]:
        """
        Fit second-stage probit/logit with control function.

        Returns
        -------
        fit : BinaryResultsWrapper
            Fitted model.
        ame : float
            Average marginal effect of D.
        """
        n = len(Y)

        # Design matrix: [1, D, nu_hat, X]
        if X is not None:
            design = np.column_stack([np.ones(n), D, nu_hat, X])
        else:
            design = np.column_stack([np.ones(n), D, nu_hat])

        # Choose model
        if self.model_type == "probit":
            model = sm.Probit(Y, design)
        else:
            model = sm.Logit(Y, design)

        try:
            fit = model.fit(disp=0, maxiter=100)
            converged = fit.mle_retvals["converged"]
        except Exception:
            # Return None to indicate failed fit
            return None, np.nan

        if not converged:
            return None, np.nan

        # Compute average marginal effect
        ame = self._compute_ame(fit, D, nu_hat, X)

        return fit, ame

    def _compute_ame(
        self,
        fit: Any,
        D: NDArray[np.float64],
        nu_hat: NDArray[np.float64],
        X: Optional[NDArray[np.float64]],
    ) -> float:
        """
        Compute average marginal effect of treatment D.

        For probit: AME = beta_D * mean(phi(X*beta))
        For logit: AME = beta_D * mean(Lambda(X*beta) * (1 - Lambda(X*beta)))

        where phi is standard normal PDF and Lambda is logistic CDF.
        """
        n = len(D)
        beta_D = fit.params[1]  # Coefficient on D

        # Build design matrix at data values
        if X is not None:
            design = np.column_stack([np.ones(n), D, nu_hat, X])
        else:
            design = np.column_stack([np.ones(n), D, nu_hat])

        # Linear index
        xb = design @ fit.params

        # Marginal effect at each observation
        if self.model_type == "probit":
            # ME_i = beta_D * phi(x_i * beta)
            me_i = beta_D * stats.norm.pdf(xb)
        else:
            # ME_i = beta_D * Lambda(xb) * (1 - Lambda(xb))
            prob = 1 / (1 + np.exp(-xb))
            me_i = beta_D * prob * (1 - prob)

        # Average marginal effect
        ame = np.mean(me_i)
        return ame

    def _bootstrap_estimation(
        self,
        Y: NDArray[np.float64],
        D: NDArray[np.float64],
        Z: NDArray[np.float64],
        X: Optional[NDArray[np.float64]],
        nu_hat: NDArray[np.float64],
    ) -> NonlinearCFResult:
        """Bootstrap inference for nonlinear CF."""
        n = len(Y)
        first_stage = self.first_stage_

        # Point estimates from full sample
        fit, ame = self._fit_second_stage(Y, D, nu_hat, X)

        if fit is None:
            # Convergence failure
            return NonlinearCFResult(
                estimate=np.nan,
                se=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                p_value=np.nan,
                control_coef=np.nan,
                control_se=np.nan,
                control_p_value=np.nan,
                endogeneity_detected=False,
                first_stage=first_stage,
                model_type=self.model_type,
                n_obs=n,
                n_bootstrap=self.n_bootstrap,
                alpha=self.alpha,
                convergence=False,
                message="Second-stage model failed to converge",
            )

        # Extract point estimates
        beta_D = fit.params[1]
        rho = fit.params[2]  # Control coefficient

        # Bootstrap
        rng = np.random.default_rng(self.random_state)
        ame_boots = []
        rho_boots = []

        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            Y_b = Y[idx]
            D_b = D[idx]
            Z_b = Z[idx]
            X_b = X[idx] if X is not None else None

            try:
                # Re-estimate first stage
                fs_b = self._first_stage(D_b, Z_b, X_b)
                nu_hat_b = fs_b["residuals"]

                # Re-estimate second stage
                fit_b, ame_b = self._fit_second_stage(Y_b, D_b, nu_hat_b, X_b)

                if fit_b is not None:
                    ame_boots.append(ame_b)
                    rho_boots.append(fit_b.params[2])
            except Exception:
                continue

        if len(ame_boots) < 50:
            # Not enough successful bootstrap samples
            return NonlinearCFResult(
                estimate=ame,
                se=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                p_value=np.nan,
                control_coef=rho,
                control_se=np.nan,
                control_p_value=np.nan,
                endogeneity_detected=False,
                first_stage=first_stage,
                model_type=self.model_type,
                n_obs=n,
                n_bootstrap=len(ame_boots),
                alpha=self.alpha,
                convergence=True,
                message=f"Only {len(ame_boots)} bootstrap samples converged (< 50)",
            )

        ame_boots = np.array(ame_boots)
        rho_boots = np.array(rho_boots)

        # Bootstrap inference
        se = np.std(ame_boots, ddof=1)
        ci_lower = np.percentile(ame_boots, 100 * self.alpha / 2)
        ci_upper = np.percentile(ame_boots, 100 * (1 - self.alpha / 2))
        p_value = 2 * (1 - stats.norm.cdf(abs(ame / se))) if se > 0 else np.nan

        # Endogeneity test on rho
        rho_se = np.std(rho_boots, ddof=1)
        rho_t = rho / rho_se if rho_se > 0 else 0
        control_p_value = 2 * (1 - stats.norm.cdf(abs(rho_t)))
        endogeneity_detected = control_p_value < self.alpha

        # Generate message
        message_parts = []
        if first_stage["weak_iv_warning"]:
            message_parts.append(
                f"WARNING: Weak instruments (F={first_stage['f_statistic']:.1f} < 10)"
            )
        if endogeneity_detected:
            message_parts.append("Endogeneity detected")
        else:
            message_parts.append(f"No endogeneity detected (naive {self.model_type} may be OK)")
        message = "; ".join(message_parts)

        return NonlinearCFResult(
            estimate=ame,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            control_coef=rho,
            control_se=rho_se,
            control_p_value=control_p_value,
            endogeneity_detected=endogeneity_detected,
            first_stage=first_stage,
            model_type=self.model_type,
            n_obs=n,
            n_bootstrap=len(ame_boots),
            alpha=self.alpha,
            convergence=True,
            message=message,
        )

    def summary(self) -> str:
        """Return formatted summary."""
        if not self.fitted_:
            return "Model not fitted. Call fit() first."

        r = self.result_
        lines = [
            "=" * 60,
            f"Nonlinear Control Function ({r['model_type'].title()}) Results",
            "=" * 60,
            f"N observations:     {r['n_obs']}",
            f"N bootstrap:        {r['n_bootstrap']}",
            f"Convergence:        {'Yes' if r['convergence'] else 'No'}",
            "",
            "Average Marginal Effect of Treatment:",
            f"  AME:              {r['estimate']:.4f}",
            f"  Std. Error:       {r['se']:.4f}",
            f"  p-value:          {r['p_value']:.4f}",
            f"  95% CI:           [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]",
            "",
            "Endogeneity Test (H0: rho = 0):",
            f"  Control coef:     {r['control_coef']:.4f}",
            f"  Std. Error:       {r['control_se']:.4f}",
            f"  p-value:          {r['control_p_value']:.4f}",
            f"  Endogeneity:      {'Yes' if r['endogeneity_detected'] else 'No'}",
            "",
            "First Stage:",
            f"  F-statistic:      {r['first_stage']['f_statistic']:.2f}",
            f"  Weak IV warning:  {'Yes' if r['first_stage']['weak_iv_warning'] else 'No'}",
            "",
            f"Message: {r['message']}",
            "=" * 60,
        ]
        return "\n".join(lines)


def nonlinear_control_function(
    Y: NDArray[np.floating],
    D: NDArray[np.floating],
    Z: NDArray[np.floating],
    X: Optional[NDArray[np.floating]] = None,
    model_type: Literal["probit", "logit"] = "probit",
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> NonlinearCFResult:
    """
    Convenience function for nonlinear control function estimation.

    Estimates causal effect with binary outcome when treatment is endogenous.
    Returns average marginal effect (AME) with bootstrap inference.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Binary outcome (0 or 1).
    D : array-like, shape (n,)
        Endogenous treatment.
    Z : array-like, shape (n,) or (n, q)
        Instrumental variables.
    X : array-like, shape (n, k), optional
        Exogenous controls.
    model_type : {'probit', 'logit'}, default='probit'
        Nonlinear model for second stage.
    n_bootstrap : int, default=500
        Number of bootstrap iterations.
    alpha : float, default=0.05
        Significance level.
    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    NonlinearCFResult
        Estimation results.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 1000
    >>> Z = np.random.normal(0, 1, n)
    >>> nu = np.random.normal(0, 1, n)
    >>> D = 0.5 * Z + nu
    >>> latent = 1.0 * D + 0.5 * nu + np.random.logistic(0, 1, n)
    >>> Y = (latent > 0).astype(float)
    >>>
    >>> result = nonlinear_control_function(Y, D, Z, model_type='probit')
    >>> print(f"AME: {result['estimate']:.3f}")
    """
    cf = NonlinearControlFunction(
        model_type=model_type,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )
    return cf.fit(Y, D, Z, X)
