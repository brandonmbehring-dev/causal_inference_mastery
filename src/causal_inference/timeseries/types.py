"""
Time Series Causal Inference Types.

Session 135: Data structures for Granger causality and VAR analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class GrangerResult:
    """
    Result from pairwise Granger causality test.

    Tests H0: X does not Granger-cause Y
    (past values of X do not help predict Y given past values of Y)

    Attributes
    ----------
    cause_var : str
        Name of potential cause variable
    effect_var : str
        Name of effect variable
    f_statistic : float
        F-test statistic
    p_value : float
        P-value for null hypothesis (no Granger causality)
    lags : int
        Number of lags used in the test
    granger_causes : bool
        True if null rejected (X Granger-causes Y)
    alpha : float
        Significance level used
    r2_unrestricted : float
        R-squared of unrestricted model (includes X lags)
    r2_restricted : float
        R-squared of restricted model (Y lags only)
    aic_unrestricted : float
        AIC of unrestricted model
    aic_restricted : float
        AIC of restricted model
    df_num : int
        Numerator degrees of freedom (number of restrictions)
    df_denom : int
        Denominator degrees of freedom (residual df)
    rss_unrestricted : float
        Residual sum of squares from unrestricted model
    rss_restricted : float
        Residual sum of squares from restricted model
    """

    cause_var: str
    effect_var: str
    f_statistic: float
    p_value: float
    lags: int
    granger_causes: bool
    alpha: float = 0.05
    r2_unrestricted: float = 0.0
    r2_restricted: float = 0.0
    aic_unrestricted: float = 0.0
    aic_restricted: float = 0.0
    df_num: int = 0
    df_denom: int = 0
    rss_unrestricted: float = 0.0
    rss_restricted: float = 0.0

    def __repr__(self) -> str:
        direction = "→" if self.granger_causes else "↛"
        sig = "*" if self.granger_causes else ""
        return (
            f"GrangerResult({self.cause_var} {direction} {self.effect_var}{sig}, "
            f"F={self.f_statistic:.3f}, p={self.p_value:.4f}, lags={self.lags})"
        )


@dataclass
class MultiGrangerResult:
    """
    Result from multivariate Granger causality analysis.

    Contains pairwise Granger test results for all variable pairs.

    Attributes
    ----------
    n_vars : int
        Number of variables
    var_names : List[str]
        Variable names
    pairwise_results : Dict[Tuple[str, str], GrangerResult]
        Mapping from (cause, effect) to GrangerResult
    causality_matrix : np.ndarray
        (n_vars, n_vars) boolean matrix where [i,j]=True means i Granger-causes j
    lags : int
        Number of lags used
    alpha : float
        Significance level
    """

    n_vars: int
    var_names: List[str]
    pairwise_results: Dict[Tuple[str, str], GrangerResult]
    causality_matrix: np.ndarray
    lags: int
    alpha: float = 0.05

    def get_causes(self, var: str) -> List[str]:
        """Get all variables that Granger-cause the given variable."""
        causes = []
        for (cause, effect), result in self.pairwise_results.items():
            if effect == var and result.granger_causes:
                causes.append(cause)
        return causes

    def get_effects(self, var: str) -> List[str]:
        """Get all variables that the given variable Granger-causes."""
        effects = []
        for (cause, effect), result in self.pairwise_results.items():
            if cause == var and result.granger_causes:
                effects.append(effect)
        return effects

    def __repr__(self) -> str:
        n_causal = int(self.causality_matrix.sum())
        return (
            f"MultiGrangerResult(n_vars={self.n_vars}, "
            f"n_causal_pairs={n_causal}, lags={self.lags})"
        )


@dataclass
class VARResult:
    """
    Result from VAR (Vector Autoregression) estimation.

    Model: Y_t = A_0 + A_1 Y_{t-1} + ... + A_p Y_{t-p} + epsilon_t

    Attributes
    ----------
    coefficients : np.ndarray
        Shape (n_vars, n_vars * lags + 1) coefficient matrix.
        Each row corresponds to an equation.
        Columns: [intercept, var1_lag1, var2_lag1, ..., var1_lag2, ...]
    residuals : np.ndarray
        Shape (n_obs - lags, n_vars) residual matrix
    aic : float
        Akaike Information Criterion
    bic : float
        Bayesian Information Criterion
    hqc : float
        Hannan-Quinn Criterion
    lags : int
        Lag order p
    n_obs : int
        Number of observations (before removing lags)
    n_obs_effective : int
        Effective observations used (n_obs - lags)
    var_names : List[str]
        Variable names
    sigma : np.ndarray
        Shape (n_vars, n_vars) covariance matrix of residuals
    log_likelihood : float
        Log-likelihood of the model
    """

    coefficients: np.ndarray
    residuals: np.ndarray
    aic: float
    bic: float
    hqc: float
    lags: int
    n_obs: int
    n_obs_effective: int
    var_names: List[str]
    sigma: np.ndarray
    log_likelihood: float

    @property
    def n_vars(self) -> int:
        """Number of variables in the VAR system."""
        return len(self.var_names)

    @property
    def n_params_per_eq(self) -> int:
        """Number of parameters per equation (including intercept)."""
        return self.n_vars * self.lags + 1

    def get_lag_matrix(self, lag: int) -> np.ndarray:
        """
        Get coefficient matrix for specific lag.

        Parameters
        ----------
        lag : int
            Lag number (1 to self.lags)

        Returns
        -------
        np.ndarray
            Shape (n_vars, n_vars) coefficient matrix A_lag
        """
        if lag < 1 or lag > self.lags:
            raise ValueError(f"Lag must be between 1 and {self.lags}")

        start_idx = 1 + (lag - 1) * self.n_vars
        end_idx = start_idx + self.n_vars
        return self.coefficients[:, start_idx:end_idx]

    def get_intercepts(self) -> np.ndarray:
        """Get intercept vector."""
        return self.coefficients[:, 0]

    def __repr__(self) -> str:
        return (
            f"VARResult(n_vars={self.n_vars}, lags={self.lags}, "
            f"n_obs={self.n_obs_effective}, AIC={self.aic:.2f})"
        )


@dataclass
class ADFResult:
    """
    Augmented Dickey-Fuller test result.

    Tests H0: Series has unit root (non-stationary)
    vs H1: Series is stationary

    Attributes
    ----------
    statistic : float
        ADF test statistic (more negative = stronger rejection)
    p_value : float
        P-value for the test
    lags : int
        Number of lags used in the test
    n_obs : int
        Number of observations
    critical_values : Dict[str, float]
        Critical values at 1%, 5%, 10% levels
    is_stationary : bool
        True if null rejected (series is stationary)
    regression : str
        Type of regression ("c" = constant, "ct" = constant+trend, "n" = none)
    alpha : float
        Significance level used for is_stationary determination
    """

    statistic: float
    p_value: float
    lags: int
    n_obs: int
    critical_values: Dict[str, float]
    is_stationary: bool
    regression: str = "c"
    alpha: float = 0.05

    def __repr__(self) -> str:
        status = "Stationary" if self.is_stationary else "Non-stationary"
        return (
            f"ADFResult({status}, stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f}, lags={self.lags})"
        )


@dataclass
class LagSelectionResult:
    """
    Result from lag order selection.

    Attributes
    ----------
    optimal_lag : int
        Selected optimal lag order
    criterion : str
        Criterion used ("aic", "bic", "hqc")
    all_values : Dict[int, float]
        Information criterion values for all tested lags
    all_lags : List[int]
        All lags tested
    aic_values : Dict[int, float]
        AIC values for all lags
    bic_values : Dict[int, float]
        BIC values for all lags
    hqc_values : Dict[int, float]
        HQC values for all lags
    """

    optimal_lag: int
    criterion: str
    all_values: Dict[int, float]
    all_lags: List[int]
    aic_values: Dict[int, float] = field(default_factory=dict)
    bic_values: Dict[int, float] = field(default_factory=dict)
    hqc_values: Dict[int, float] = field(default_factory=dict)

    def get_optimal_by_criterion(self, criterion: str) -> int:
        """Get optimal lag for a specific criterion."""
        if criterion == "aic":
            values = self.aic_values
        elif criterion == "bic":
            values = self.bic_values
        elif criterion == "hqc":
            values = self.hqc_values
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        if not values:
            raise ValueError(f"No values computed for {criterion}")

        return min(values, key=values.get)

    def __repr__(self) -> str:
        return (
            f"LagSelectionResult(optimal={self.optimal_lag}, "
            f"criterion={self.criterion}, tested={len(self.all_lags)} lags)"
        )
