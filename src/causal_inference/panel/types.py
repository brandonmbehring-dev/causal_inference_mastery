"""Type definitions for Panel DML-CRE.

Defines data structures for panel data and DML-CRE results.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PanelData:
    """Panel data structure for DML-CRE estimation.

    Long-format panel data with unit and time identifiers.

    Attributes
    ----------
    outcomes : np.ndarray
        Outcome variable Y of shape (n_obs,).
    treatment : np.ndarray
        Treatment variable D of shape (n_obs,). Can be binary or continuous.
    covariates : np.ndarray
        Time-varying covariates X of shape (n_obs, p).
    unit_id : np.ndarray
        Unit identifiers of shape (n_obs,). Integer or categorical.
    time : np.ndarray
        Time period identifiers of shape (n_obs,). Integer.

    Properties
    ----------
    n_obs : int
        Total number of observations (unit-time pairs).
    n_units : int
        Number of unique units.
    n_periods : int
        Number of unique time periods.
    is_balanced : bool
        True if all units have same number of observations.
    n_covariates : int
        Number of covariates p.

    Examples
    --------
    >>> import numpy as np
    >>> # Create balanced panel: 10 units, 5 periods each
    >>> n_units, n_periods = 10, 5
    >>> n_obs = n_units * n_periods
    >>> unit_id = np.repeat(np.arange(n_units), n_periods)
    >>> time = np.tile(np.arange(n_periods), n_units)
    >>> Y = np.random.randn(n_obs)
    >>> D = np.random.binomial(1, 0.5, n_obs)
    >>> X = np.random.randn(n_obs, 3)
    >>> panel = PanelData(Y, D, X, unit_id, time)
    >>> print(f"Units: {panel.n_units}, Periods: {panel.n_periods}")
    Units: 10, Periods: 5
    """

    outcomes: np.ndarray
    treatment: np.ndarray
    covariates: np.ndarray
    unit_id: np.ndarray
    time: np.ndarray

    def __post_init__(self):
        """Validate inputs after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate panel data structure."""
        n = len(self.outcomes)

        # Convert to numpy arrays
        self.outcomes = np.asarray(self.outcomes, dtype=np.float64)
        self.treatment = np.asarray(self.treatment, dtype=np.float64)
        self.covariates = np.asarray(self.covariates, dtype=np.float64)
        self.unit_id = np.asarray(self.unit_id)
        self.time = np.asarray(self.time)

        # Length consistency
        if len(self.treatment) != n:
            raise ValueError(
                f"CRITICAL ERROR: Length mismatch.\n"
                f"Function: PanelData._validate\n"
                f"outcomes: {n}, treatment: {len(self.treatment)}"
            )

        if len(self.unit_id) != n:
            raise ValueError(
                f"CRITICAL ERROR: Length mismatch.\n"
                f"Function: PanelData._validate\n"
                f"outcomes: {n}, unit_id: {len(self.unit_id)}"
            )

        if len(self.time) != n:
            raise ValueError(
                f"CRITICAL ERROR: Length mismatch.\n"
                f"Function: PanelData._validate\n"
                f"outcomes: {n}, time: {len(self.time)}"
            )

        # Handle 1D covariates
        if self.covariates.ndim == 1:
            self.covariates = self.covariates.reshape(-1, 1)

        if len(self.covariates) != n:
            raise ValueError(
                f"CRITICAL ERROR: Covariate rows mismatch.\n"
                f"Function: PanelData._validate\n"
                f"outcomes: {n}, covariate rows: {len(self.covariates)}"
            )

        # Check for NaN/Inf
        if np.any(np.isnan(self.outcomes)) or np.any(np.isinf(self.outcomes)):
            raise ValueError(
                f"CRITICAL ERROR: NaN or Inf in outcomes.\nFunction: PanelData._validate"
            )

        if np.any(np.isnan(self.treatment)) or np.any(np.isinf(self.treatment)):
            raise ValueError(
                f"CRITICAL ERROR: NaN or Inf in treatment.\nFunction: PanelData._validate"
            )

        if np.any(np.isnan(self.covariates)) or np.any(np.isinf(self.covariates)):
            raise ValueError(
                f"CRITICAL ERROR: NaN or Inf in covariates.\nFunction: PanelData._validate"
            )

        # Check minimum units
        if self.n_units < 2:
            raise ValueError(
                f"CRITICAL ERROR: Need at least 2 units.\n"
                f"Function: PanelData._validate\n"
                f"n_units: {self.n_units}"
            )

        # Check minimum observations per unit
        obs_per_unit = self._get_obs_per_unit()
        if np.min(obs_per_unit) < 2:
            raise ValueError(
                f"CRITICAL ERROR: Each unit needs at least 2 observations.\n"
                f"Function: PanelData._validate\n"
                f"Min observations per unit: {np.min(obs_per_unit)}"
            )

    def _get_obs_per_unit(self) -> np.ndarray:
        """Get number of observations per unit."""
        unique_units, counts = np.unique(self.unit_id, return_counts=True)
        return counts

    @property
    def n_obs(self) -> int:
        """Total number of observations."""
        return len(self.outcomes)

    @property
    def n_units(self) -> int:
        """Number of unique units."""
        return len(np.unique(self.unit_id))

    @property
    def n_periods(self) -> int:
        """Number of unique time periods."""
        return len(np.unique(self.time))

    @property
    def is_balanced(self) -> bool:
        """Check if panel is balanced (same T for all units)."""
        obs_per_unit = self._get_obs_per_unit()
        return len(np.unique(obs_per_unit)) == 1

    @property
    def n_covariates(self) -> int:
        """Number of covariates."""
        return self.covariates.shape[1]

    def get_unique_units(self) -> np.ndarray:
        """Get array of unique unit identifiers."""
        return np.unique(self.unit_id)

    def get_unit_indices(self, unit: int) -> np.ndarray:
        """Get indices for a specific unit."""
        return np.where(self.unit_id == unit)[0]

    def compute_unit_means(self) -> np.ndarray:
        """Compute time-averaged covariates X̄ᵢ for each observation.

        For each observation (i, t), returns the unit-level mean
        X̄ᵢ = (1/Tᵢ) Σₜ Xᵢₜ

        Returns
        -------
        np.ndarray
            Unit means of shape (n_obs, p), aligned with observations.

        Notes
        -----
        This is the Mundlak (1978) projection: by including X̄ᵢ alongside
        Xᵢₜ, we can model the correlation between covariates and
        unobserved unit effects.
        """
        n = self.n_obs
        p = self.n_covariates
        unit_means = np.zeros((n, p))

        for unit in self.get_unique_units():
            unit_idx = self.get_unit_indices(unit)
            unit_mean = np.mean(self.covariates[unit_idx], axis=0)
            unit_means[unit_idx] = unit_mean

        return unit_means

    def compute_treatment_mean(self) -> np.ndarray:
        """Compute time-averaged treatment D̄ᵢ for each observation.

        Returns
        -------
        np.ndarray
            Unit treatment means of shape (n_obs,).
        """
        n = self.n_obs
        treatment_means = np.zeros(n)

        for unit in self.get_unique_units():
            unit_idx = self.get_unit_indices(unit)
            treatment_means[unit_idx] = np.mean(self.treatment[unit_idx])

        return treatment_means


@dataclass
class DMLCREResult:
    """Result from Panel DML-CRE estimation.

    Attributes
    ----------
    ate : float
        Average treatment effect estimate.
    ate_se : float
        Standard error of ATE (clustered by unit).
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    cate : np.ndarray
        Conditional average treatment effects τ(Xᵢₜ, X̄ᵢ) of shape (n_obs,).
    method : str
        "dml_cre" or "dml_cre_continuous".
    n_units : int
        Number of units.
    n_obs : int
        Number of observations.
    n_folds : int
        Number of cross-fitting folds.
    outcome_r2 : float
        R-squared of outcome model.
    treatment_r2 : float
        R-squared of treatment model (for continuous) or pseudo-R² (for binary).
    unit_effects : np.ndarray
        Estimated unit effects α̂ᵢ of shape (n_units,).
    fold_estimates : np.ndarray
        Per-fold ATE estimates of shape (n_folds,).
    fold_ses : np.ndarray
        Per-fold standard errors of shape (n_folds,).
    """

    ate: float
    ate_se: float
    ci_lower: float
    ci_upper: float
    cate: np.ndarray
    method: str
    n_units: int
    n_obs: int
    n_folds: int
    outcome_r2: float
    treatment_r2: float
    unit_effects: np.ndarray
    fold_estimates: np.ndarray
    fold_ses: np.ndarray


@dataclass
class PanelQTEResult:
    """Result from Panel RIF-QTE estimation.

    Panel quantile treatment effect using Recentered Influence Function
    regression with Mundlak projection and clustered standard errors.

    Attributes
    ----------
    qte : float
        Quantile treatment effect at quantile τ.
    qte_se : float
        Standard error of QTE (clustered by unit).
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    quantile : float
        Quantile τ ∈ (0, 1) at which effect is estimated.
    n_obs : int
        Total number of observations.
    n_units : int
        Number of unique units.
    outcome_quantile : float
        Estimated pooled quantile q̂_τ of outcomes.
    density_at_quantile : float
        Kernel density estimate f̂_Y(q̂_τ) at the quantile.
    bandwidth : float
        Kernel bandwidth used for density estimation.
    method : str
        Estimation method, typically "panel_rif_qte".

    Notes
    -----
    The RIF-QTE approach (Firpo, Fortin, Lemieux 2009) transforms outcomes
    using the Recentered Influence Function:

        RIF(Y; q_τ) = q_τ + (τ - I(Y ≤ q_τ)) / f_Y(q_τ)

    The coefficient on treatment D from OLS regression of RIF on covariates
    estimates the unconditional QTE.

    For panel data, we:
    1. Use Mundlak projection (include X̄ᵢ) to control for unit effects
    2. Cluster standard errors at the unit level

    Examples
    --------
    >>> result = panel_rif_qte(panel, quantile=0.5)
    >>> print(f"Median effect: {result.qte:.3f} ± {result.qte_se:.3f}")
    """

    qte: float
    qte_se: float
    ci_lower: float
    ci_upper: float
    quantile: float
    n_obs: int
    n_units: int
    outcome_quantile: float
    density_at_quantile: float
    bandwidth: float
    method: str


@dataclass
class PanelQTEBandResult:
    """Result from Panel RIF-QTE estimation at multiple quantiles.

    Contains QTE estimates across a range of quantiles, useful for
    understanding distributional treatment effects.

    Attributes
    ----------
    quantiles : np.ndarray
        Array of quantiles τ at which effects were estimated.
    qtes : np.ndarray
        QTE estimates at each quantile.
    qte_ses : np.ndarray
        Standard errors at each quantile (clustered by unit).
    ci_lowers : np.ndarray
        Lower confidence interval bounds at each quantile.
    ci_uppers : np.ndarray
        Upper confidence interval bounds at each quantile.
    n_obs : int
        Total number of observations.
    n_units : int
        Number of unique units.
    method : str
        Estimation method, typically "panel_rif_qte_band".

    Notes
    -----
    A common quantile band is [0.1, 0.25, 0.5, 0.75, 0.9], which
    characterizes treatment effects across the outcome distribution.

    Heterogeneous effects across quantiles indicate the treatment
    affects different parts of the distribution differently.

    Examples
    --------
    >>> result = panel_rif_qte_band(panel, quantiles=[0.1, 0.5, 0.9])
    >>> for q, qte, se in zip(result.quantiles, result.qtes, result.qte_ses):
    ...     print(f"τ={q:.1f}: QTE={qte:.3f} ± {se:.3f}")
    """

    quantiles: np.ndarray
    qtes: np.ndarray
    qte_ses: np.ndarray
    ci_lowers: np.ndarray
    ci_uppers: np.ndarray
    n_obs: int
    n_units: int
    method: str
