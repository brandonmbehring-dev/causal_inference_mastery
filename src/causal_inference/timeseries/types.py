"""
Time Series Causal Inference Types.

Session 135: Data structures for Granger causality and VAR analysis.
Session 145: Added KPSS, Phillips-Perron, and Johansen cointegration types.
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
            f"MultiGrangerResult(n_vars={self.n_vars}, n_causal_pairs={n_causal}, lags={self.lags})"
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


@dataclass
class KPSSResult:
    """
    KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test result.

    Tests H0: Series is trend-stationary (stationary around deterministic trend)
    vs H1: Series has unit root (non-stationary)

    Note: Opposite null hypothesis from ADF test.
    - Low statistic → stationary (fail to reject H0)
    - High statistic → non-stationary (reject H0)

    Attributes
    ----------
    statistic : float
        KPSS test statistic (higher = more evidence against stationarity)
    p_value : float
        Approximate p-value for the test
    lags : int
        Number of lags used for long-run variance estimation
    n_obs : int
        Number of observations
    critical_values : Dict[str, float]
        Critical values at 1%, 2.5%, 5%, 10% levels
    is_stationary : bool
        True if fail to reject H0 (series is stationary)
    regression : str
        Type of regression ("c" = constant, "ct" = constant+trend)
    alpha : float
        Significance level used for is_stationary determination

    References
    ----------
    Kwiatkowski et al. (1992). "Testing the null hypothesis of stationarity
    against the alternative of a unit root." J. Econometrics 54: 159-178.
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
            f"KPSSResult({status}, stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f}, lags={self.lags})"
        )


@dataclass
class PPResult:
    """
    Phillips-Perron test result.

    Tests H0: Series has unit root (non-stationary)
    vs H1: Series is stationary

    Like ADF but uses Newey-West HAC correction instead of augmented lags.
    Robust to heteroskedasticity and autocorrelation of unknown form.

    Attributes
    ----------
    statistic : float
        PP Z_t test statistic (more negative = stronger rejection)
    p_value : float
        P-value for the test
    lags : int
        Number of lags used for Newey-West correction
    n_obs : int
        Number of observations
    critical_values : Dict[str, float]
        Critical values at 1%, 5%, 10% levels
    is_stationary : bool
        True if null rejected (series is stationary)
    regression : str
        Type of regression ("n", "c", "ct")
    alpha : float
        Significance level used for is_stationary determination
    rho_stat : float
        PP Z_rho statistic (alternative form)

    References
    ----------
    Phillips & Perron (1988). "Testing for a unit root in time series
    regression." Biometrika 75(2): 335-346.
    """

    statistic: float
    p_value: float
    lags: int
    n_obs: int
    critical_values: Dict[str, float]
    is_stationary: bool
    regression: str = "c"
    alpha: float = 0.05
    rho_stat: float = 0.0

    def __repr__(self) -> str:
        status = "Stationary" if self.is_stationary else "Non-stationary"
        return (
            f"PPResult({status}, stat={self.statistic:.4f}, p={self.p_value:.4f}, lags={self.lags})"
        )


@dataclass
class JohansenResult:
    """
    Johansen cointegration test result.

    Tests for cointegration rank r in a VAR system of n variables.
    Uses both trace and maximum eigenvalue tests.

    Attributes
    ----------
    rank : int
        Estimated cointegration rank (0 to n_vars-1)
    trace_stats : np.ndarray
        Trace statistics for each null hypothesis r=0,1,...,n-1
    trace_crit : np.ndarray
        Critical values for trace statistics
    trace_pvalues : np.ndarray
        P-values for trace statistics
    max_eigen_stats : np.ndarray
        Max eigenvalue statistics for each null
    max_eigen_crit : np.ndarray
        Critical values for max eigenvalue statistics
    max_eigen_pvalues : np.ndarray
        P-values for max eigenvalue statistics
    eigenvalues : np.ndarray
        Eigenvalues from reduced rank regression (sorted descending)
    eigenvectors : np.ndarray
        Eigenvectors (columns are cointegrating vectors β)
    adjustment : np.ndarray
        Adjustment coefficients α (loading matrix)
    lags : int
        Number of VAR lags used
    n_obs : int
        Number of observations used in estimation
    n_vars : int
        Number of variables
    det_order : int
        Deterministic terms: -1=no const/trend, 0=restricted const,
        1=unrestricted const, 2=restricted trend
    alpha : float
        Significance level used for rank determination

    References
    ----------
    Johansen (1988). "Statistical analysis of cointegration vectors."
    Journal of Economic Dynamics and Control 12: 231-254.
    Johansen (1991). "Estimation and hypothesis testing of cointegration
    vectors in Gaussian vector autoregressive models." Econometrica 59: 1551-1580.
    """

    rank: int
    trace_stats: np.ndarray
    trace_crit: np.ndarray
    trace_pvalues: np.ndarray
    max_eigen_stats: np.ndarray
    max_eigen_crit: np.ndarray
    max_eigen_pvalues: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    adjustment: np.ndarray
    lags: int
    n_obs: int
    n_vars: int
    det_order: int = 0
    alpha: float = 0.05

    @property
    def has_cointegration(self) -> bool:
        """True if at least one cointegrating relationship exists."""
        return self.rank > 0

    @property
    def cointegrating_vectors(self) -> np.ndarray:
        """Return estimated cointegrating vectors (first r columns of β)."""
        if self.rank == 0:
            return np.array([]).reshape(self.n_vars, 0)
        return self.eigenvectors[:, : self.rank]

    @property
    def loading_matrix(self) -> np.ndarray:
        """Return adjustment/loading coefficients (first r columns of α)."""
        if self.rank == 0:
            return np.array([]).reshape(self.n_vars, 0)
        return self.adjustment[:, : self.rank]

    def __repr__(self) -> str:
        coint = "Cointegrated" if self.has_cointegration else "No cointegration"
        return f"JohansenResult({coint}, rank={self.rank}, n_vars={self.n_vars}, lags={self.lags})"


@dataclass
class VECMResult:
    """
    Vector Error Correction Model (VECM) estimation result.

    The VECM representation of a cointegrated VAR(p) is:

        ΔY_t = αβ'Y_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + c + ε_t

    Where:
    - α (k×r): Adjustment/loading coefficients (speed of adjustment to equilibrium)
    - β (k×r): Cointegrating vectors (long-run equilibrium relationships)
    - Π = αβ': Long-run impact matrix (reduced rank = r)
    - Γ_i: Short-run dynamics matrices
    - c: Constant term (if included)
    - ε_t: Error term with covariance Σ

    Attributes
    ----------
    alpha : np.ndarray
        Adjustment coefficients (k × r matrix). Each column represents how
        variables adjust to deviations from one cointegrating relationship.
    beta : np.ndarray
        Cointegrating vectors (k × r matrix). Each column is a long-run
        equilibrium relationship β'Y = 0.
    gamma : np.ndarray
        Short-run dynamics coefficients (k × k*(p-1) matrix).
        Stacked as [Γ₁ | Γ₂ | ... | Γ_{p-1}].
    pi : np.ndarray
        Long-run impact matrix Π = αβ' (k × k matrix).
    const : np.ndarray
        Constant term (k × 1 vector). None if no constant.
    coint_rank : int
        Cointegration rank r (number of cointegrating relationships).
    lags : int
        Number of lags in original VAR (VECM uses p-1 differenced lags).
    residuals : np.ndarray
        Model residuals (T × k matrix).
    sigma : np.ndarray
        Residual covariance matrix (k × k).
    n_obs : int
        Number of observations used in estimation.
    n_vars : int
        Number of variables (k).
    det_order : int
        Deterministic terms: -1=none, 0=restricted const, 1=unrestricted const.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    log_likelihood : float
        Log-likelihood value.

    Properties
    ----------
    error_correction_term : np.ndarray
        The ECT = β'Y_{t-1} for each time period.
    adjustment_half_life : np.ndarray
        Half-life of adjustment for each cointegrating relationship.

    References
    ----------
    Lütkepohl (2005). "New Introduction to Multiple Time Series Analysis"
    Johansen (1995). "Likelihood-Based Inference in Cointegrated VAR Models"
    """

    alpha: np.ndarray  # Adjustment coefficients (k × r)
    beta: np.ndarray  # Cointegrating vectors (k × r)
    gamma: np.ndarray  # Short-run dynamics (k × k*(p-1))
    pi: np.ndarray  # Long-run matrix αβ' (k × k)
    const: Optional[np.ndarray]  # Constant (k × 1) or None
    coint_rank: int  # Cointegration rank r
    lags: int  # VAR lags (VECM has p-1 differenced lags)
    residuals: np.ndarray  # Residuals (T × k)
    sigma: np.ndarray  # Residual covariance (k × k)
    n_obs: int  # Number of observations
    n_vars: int  # Number of variables
    det_order: int = 0  # Deterministic terms
    aic: float = 0.0  # AIC
    bic: float = 0.0  # BIC
    log_likelihood: float = 0.0  # Log-likelihood

    @property
    def error_correction_term(self) -> np.ndarray:
        """
        Compute error correction terms for interpretation.

        Returns β'Y_{t-1} which represents deviations from long-run equilibrium.
        Not stored since requires original data; use compute_ect() for this.
        """
        # This is a placeholder - ECT requires original data
        return self.pi

    @property
    def adjustment_half_life(self) -> np.ndarray:
        """
        Half-life of adjustment back to equilibrium for each variable.

        For diagonal elements of α, half-life ≈ -ln(2) / ln(1 + α_ii)
        Assumes α represents speed of adjustment.
        """
        half_lives = np.zeros(self.coint_rank)
        for i in range(self.coint_rank):
            # Average adjustment speed for each cointegrating relation
            avg_alpha = np.mean(np.abs(self.alpha[:, i]))
            if avg_alpha > 0:
                half_lives[i] = -np.log(2) / np.log(1 - avg_alpha) if avg_alpha < 1 else np.inf
            else:
                half_lives[i] = np.inf
        return half_lives

    def __repr__(self) -> str:
        return (
            f"VECMResult(rank={self.coint_rank}, lags={self.lags}, "
            f"n_vars={self.n_vars}, n_obs={self.n_obs}, "
            f"AIC={self.aic:.2f}, BIC={self.bic:.2f})"
        )
