"""Type definitions for Dynamic Double Machine Learning.

Defines DynamicDMLResult and TimeSeriesPanelData for dynamic treatment
effect estimation following Lewis & Syrgkanis (2021).

References
----------
Lewis, G., & Syrgkanis, V. (2021). Double/Debiased Machine Learning for
Dynamic Treatment Effects via g-Estimation. arXiv:2002.07285.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


@dataclass
class DynamicDMLResult:
    """Result from Dynamic DML estimation.

    Contains treatment effect estimates at each lag with HAC-robust inference.

    Attributes
    ----------
    theta : np.ndarray
        Shape (max_lag + 1,) for single treatment, (max_lag + 1, n_treatments)
        for multiple treatments. theta[h] = effect of treatment at lag h.
    theta_se : np.ndarray
        HAC-robust standard errors, same shape as theta.
    ci_lower : np.ndarray
        Lower confidence bounds, same shape as theta.
    ci_upper : np.ndarray
        Upper confidence bounds, same shape as theta.
    cumulative_effect : float
        Discounted sum of effects: sum(discount^h * theta[h]).
    cumulative_effect_se : float
        Standard error of cumulative effect.
    influence_function : np.ndarray
        Shape (n_obs, max_lag + 1). Influence scores for each observation
        and lag, used for variance estimation.
    nuisance_r2 : dict
        R-squared values for outcome and propensity nuisance models.
        Keys: "outcome_r2", "propensity_r2" (lists by fold).
    method : str
        Estimation method identifier.
    max_lag : int
        Maximum treatment lag considered.
    n_folds : int
        Number of cross-fitting folds used.
    hac_bandwidth : int
        Bandwidth used for HAC standard errors.
    hac_kernel : str
        Kernel used for HAC estimation ("bartlett" or "qs").
    n_obs : int
        Number of observations used in estimation.
    alpha : float
        Significance level for confidence intervals.
    discount_factor : float
        Discount factor used for cumulative effect.

    Examples
    --------
    >>> result = DynamicDMLResult(
    ...     theta=np.array([2.0, 1.0, 0.5]),
    ...     theta_se=np.array([0.1, 0.15, 0.2]),
    ...     ci_lower=np.array([1.8, 0.7, 0.1]),
    ...     ci_upper=np.array([2.2, 1.3, 0.9]),
    ...     cumulative_effect=3.45,
    ...     cumulative_effect_se=0.25,
    ...     influence_function=np.zeros((100, 3)),
    ...     nuisance_r2={"outcome_r2": [0.8], "propensity_r2": [0.7]},
    ...     method="dynamic_dml",
    ...     max_lag=2,
    ...     n_folds=5,
    ...     hac_bandwidth=4,
    ...     hac_kernel="bartlett",
    ...     n_obs=100,
    ...     alpha=0.05,
    ...     discount_factor=0.99,
    ... )
    >>> print(f"Lag 0 effect: {result.theta[0]:.2f} ± {1.96 * result.theta_se[0]:.2f}")
    Lag 0 effect: 2.00 ± 0.20
    """

    theta: np.ndarray
    theta_se: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    cumulative_effect: float
    cumulative_effect_se: float
    influence_function: np.ndarray
    nuisance_r2: dict
    method: str
    max_lag: int
    n_folds: int
    hac_bandwidth: int
    hac_kernel: str
    n_obs: int
    alpha: float
    discount_factor: float

    def summary(self) -> str:
        """Return formatted summary of dynamic treatment effects.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            f"Dynamic DML Results (n={self.n_obs})",
            "=" * 50,
            f"Method: {self.method}",
            f"Max lag: {self.max_lag}, Folds: {self.n_folds}",
            f"HAC: {self.hac_kernel} (bandwidth={self.hac_bandwidth})",
            "",
            "Treatment Effects by Lag:",
            "-" * 50,
            f"{'Lag':<6} {'Effect':<12} {'SE':<12} {'95% CI':<20}",
            "-" * 50,
        ]

        for h in range(self.max_lag + 1):
            effect = self.theta[h] if self.theta.ndim == 1 else self.theta[h, 0]
            se = self.theta_se[h] if self.theta_se.ndim == 1 else self.theta_se[h, 0]
            ci_lo = self.ci_lower[h] if self.ci_lower.ndim == 1 else self.ci_lower[h, 0]
            ci_hi = self.ci_upper[h] if self.ci_upper.ndim == 1 else self.ci_upper[h, 0]
            lines.append(f"{h:<6} {effect:<12.4f} {se:<12.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

        lines.extend(
            [
                "-" * 50,
                f"Cumulative effect: {self.cumulative_effect:.4f} "
                f"(SE: {self.cumulative_effect_se:.4f})",
                f"Discount factor: {self.discount_factor}",
            ]
        )

        return "\n".join(lines)

    def is_significant(self, lag: int, alpha: Optional[float] = None) -> bool:
        """Check if effect at given lag is statistically significant.

        Parameters
        ----------
        lag : int
            Lag to check (0 to max_lag).
        alpha : float, optional
            Significance level. Default uses self.alpha.

        Returns
        -------
        bool
            True if confidence interval excludes zero.
        """
        if alpha is None:
            alpha = self.alpha

        ci_lo = self.ci_lower[lag] if self.ci_lower.ndim == 1 else self.ci_lower[lag, 0]
        ci_hi = self.ci_upper[lag] if self.ci_upper.ndim == 1 else self.ci_upper[lag, 0]

        return ci_lo > 0 or ci_hi < 0


@dataclass
class TimeSeriesPanelData:
    """Panel or time series data structure for dynamic treatment effects.

    Supports both:
    1. Panel data: Multiple units observed over multiple periods
    2. Single time series: One unit observed over many periods

    Attributes
    ----------
    outcomes : np.ndarray
        Outcome variable Y. Shape (n_obs,) for pooled or (n_units, n_periods).
    treatments : np.ndarray
        Treatment variable(s) T. Shape (n_obs,) or (n_obs, n_treatments) for
        pooled; (n_units, n_periods) or (n_units, n_periods, n_treatments) for panel.
    states : np.ndarray
        Covariate state variables X. Shape (n_obs, n_covariates) for pooled;
        (n_units, n_periods, n_covariates) for panel.
    unit_id : Optional[np.ndarray]
        Unit identifiers for panel data, shape (n_obs,). None for single series.
    time_id : np.ndarray
        Time period identifiers, shape (n_obs,) or (n_periods,).
    data_type : str
        "single_series" or "panel".
    n_units : int
        Number of units (1 for single series).
    n_periods : int
        Number of time periods.
    n_treatments : int
        Number of treatment variables.
    n_covariates : int
        Number of covariate state variables.

    Examples
    --------
    >>> # Single time series
    >>> data = TimeSeriesPanelData.from_arrays(
    ...     outcomes=np.random.randn(100),
    ...     treatments=np.random.binomial(1, 0.5, 100),
    ...     states=np.random.randn(100, 5),
    ... )
    >>> print(f"Data type: {data.data_type}, n_periods: {data.n_periods}")
    Data type: single_series, n_periods: 100

    >>> # Panel data
    >>> data = TimeSeriesPanelData.from_arrays(
    ...     outcomes=np.random.randn(500),
    ...     treatments=np.random.binomial(1, 0.5, 500),
    ...     states=np.random.randn(500, 5),
    ...     unit_id=np.repeat(np.arange(50), 10),  # 50 units, 10 periods each
    ... )
    >>> print(f"Data type: {data.data_type}, n_units: {data.n_units}")
    Data type: panel, n_units: 50
    """

    outcomes: np.ndarray
    treatments: np.ndarray
    states: np.ndarray
    unit_id: Optional[np.ndarray]
    time_id: np.ndarray
    data_type: Literal["single_series", "panel"]
    n_units: int
    n_periods: int
    n_treatments: int
    n_covariates: int

    @classmethod
    def from_arrays(
        cls,
        outcomes: np.ndarray,
        treatments: np.ndarray,
        states: np.ndarray,
        unit_id: Optional[np.ndarray] = None,
        time_id: Optional[np.ndarray] = None,
    ) -> "TimeSeriesPanelData":
        """Create TimeSeriesPanelData from numpy arrays.

        Parameters
        ----------
        outcomes : np.ndarray
            Outcome variable Y, shape (n_obs,).
        treatments : np.ndarray
            Treatment variable(s) T, shape (n_obs,) or (n_obs, n_treatments).
        states : np.ndarray
            Covariate state variables X, shape (n_obs, n_covariates).
        unit_id : np.ndarray, optional
            Unit identifiers. If provided, data is treated as panel.
        time_id : np.ndarray, optional
            Time identifiers. Default: 0 to n_obs-1.

        Returns
        -------
        TimeSeriesPanelData
            Validated data structure.

        Raises
        ------
        ValueError
            If inputs have incompatible shapes.
        """
        outcomes = np.asarray(outcomes, dtype=np.float64)
        treatments = np.asarray(treatments, dtype=np.float64)
        states = np.asarray(states, dtype=np.float64)

        n_obs = len(outcomes)

        # Validate shapes
        if len(treatments) != n_obs:
            raise ValueError(
                f"CRITICAL ERROR: Length mismatch.\n"
                f"outcomes has {n_obs} observations, treatments has {len(treatments)}.\n"
                f"All inputs must have the same number of observations."
            )

        if len(states) != n_obs:
            raise ValueError(
                f"CRITICAL ERROR: Length mismatch.\n"
                f"outcomes has {n_obs} observations, states has {len(states)}.\n"
                f"All inputs must have the same number of observations."
            )

        # Handle 1D treatments (single treatment)
        if treatments.ndim == 1:
            treatments = treatments.reshape(-1, 1)
            n_treatments = 1
        else:
            n_treatments = treatments.shape[1]

        # Handle 1D states (single covariate)
        if states.ndim == 1:
            states = states.reshape(-1, 1)
            n_covariates = 1
        else:
            n_covariates = states.shape[1]

        # Determine data type and structure
        if unit_id is None:
            # Single time series
            data_type = "single_series"
            n_units = 1
            n_periods = n_obs
            time_id = np.arange(n_obs) if time_id is None else np.asarray(time_id)
        else:
            # Panel data
            unit_id = np.asarray(unit_id)
            if len(unit_id) != n_obs:
                raise ValueError(
                    f"CRITICAL ERROR: Length mismatch.\n"
                    f"outcomes has {n_obs} observations, unit_id has {len(unit_id)}.\n"
                )

            data_type = "panel"
            unique_units = np.unique(unit_id)
            n_units = len(unique_units)

            # Infer n_periods (assume balanced panel for simplicity)
            n_periods = n_obs // n_units
            if n_periods * n_units != n_obs:
                # Unbalanced panel - use max periods per unit
                n_periods = max(np.sum(unit_id == u) for u in unique_units)

            if time_id is None:
                # Generate time_id within each unit
                time_id = np.zeros(n_obs, dtype=int)
                for u in unique_units:
                    mask = unit_id == u
                    time_id[mask] = np.arange(np.sum(mask))
            else:
                time_id = np.asarray(time_id)

        return cls(
            outcomes=outcomes,
            treatments=treatments,
            states=states,
            unit_id=unit_id,
            time_id=time_id,
            data_type=data_type,
            n_units=n_units,
            n_periods=n_periods,
            n_treatments=n_treatments,
            n_covariates=n_covariates,
        )

    def get_lagged_data(
        self,
        max_lag: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create lagged treatment matrix for dynamic estimation.

        Parameters
        ----------
        max_lag : int
            Maximum lag to include.

        Returns
        -------
        Y : np.ndarray
            Outcomes trimmed to valid observations, shape (n_valid,).
        T_lagged : np.ndarray
            Lagged treatments, shape (n_valid, max_lag + 1, n_treatments).
            T_lagged[t, h, :] = treatment at time t-h.
        X : np.ndarray
            States trimmed to valid observations, shape (n_valid, n_covariates).
        valid_mask : np.ndarray
            Boolean mask for valid (non-initial) observations.

        Notes
        -----
        First max_lag observations are dropped since lagged treatments
        are not available.
        """
        if self.data_type == "single_series":
            # Simple case: single time series
            n_valid = self.n_periods - max_lag
            valid_mask = np.zeros(self.n_periods, dtype=bool)
            valid_mask[max_lag:] = True

            Y = self.outcomes[valid_mask]
            X = self.states[valid_mask]

            # Build lagged treatment matrix
            T_lagged = np.zeros((n_valid, max_lag + 1, self.n_treatments))
            for h in range(max_lag + 1):
                # Lag h: treatment at t - h
                T_lagged[:, h, :] = self.treatments[max_lag - h : self.n_periods - h]

            return Y, T_lagged, X, valid_mask

        else:
            # Panel data: handle each unit separately
            Y_list = []
            T_list = []
            X_list = []
            valid_indices = []

            unique_units = np.unique(self.unit_id)
            for u in unique_units:
                mask = self.unit_id == u
                unit_outcomes = self.outcomes[mask]
                unit_treatments = self.treatments[mask]
                unit_states = self.states[mask]
                n_periods_u = len(unit_outcomes)

                if n_periods_u <= max_lag:
                    continue  # Skip units with insufficient periods

                n_valid_u = n_periods_u - max_lag
                Y_list.append(unit_outcomes[max_lag:])
                X_list.append(unit_states[max_lag:])

                # Build lagged treatments for this unit
                T_lagged_u = np.zeros((n_valid_u, max_lag + 1, self.n_treatments))
                for h in range(max_lag + 1):
                    T_lagged_u[:, h, :] = unit_treatments[max_lag - h : n_periods_u - h]
                T_list.append(T_lagged_u)

                # Track valid indices
                unit_indices = np.where(mask)[0]
                valid_indices.extend(unit_indices[max_lag:].tolist())

            Y = np.concatenate(Y_list)
            T_lagged = np.concatenate(T_list, axis=0)
            X = np.concatenate(X_list, axis=0)
            valid_mask = np.zeros(len(self.outcomes), dtype=bool)
            valid_mask[valid_indices] = True

            return Y, T_lagged, X, valid_mask


def validate_dynamic_inputs(
    outcomes: np.ndarray,
    treatments: np.ndarray,
    states: np.ndarray,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs for dynamic DML estimation.

    Parameters
    ----------
    outcomes : array-like
        Outcome variable Y, shape (T,).
    treatments : array-like
        Treatment variable(s), shape (T,) or (T, d).
    states : array-like
        Covariate state variables X, shape (T, p).
    max_lag : int
        Maximum treatment lag to consider.

    Returns
    -------
    tuple
        Validated (outcomes, treatments, states) as numpy arrays.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    outcomes = np.asarray(outcomes, dtype=np.float64)
    treatments = np.asarray(treatments, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)

    n = len(outcomes)

    # Validate lengths
    if len(treatments) != n:
        raise ValueError(
            f"CRITICAL ERROR: Length mismatch.\n"
            f"outcomes has {n} observations, treatments has {len(treatments)}.\n"
        )

    if len(states) != n:
        raise ValueError(
            f"CRITICAL ERROR: Length mismatch.\n"
            f"outcomes has {n} observations, states has {len(states)}.\n"
        )

    # Handle 1D inputs
    if treatments.ndim == 1:
        treatments = treatments.reshape(-1, 1)
    if states.ndim == 1:
        states = states.reshape(-1, 1)

    # Validate max_lag
    if max_lag < 0:
        raise ValueError(f"max_lag must be non-negative, got {max_lag}")

    if max_lag >= n:
        raise ValueError(f"max_lag ({max_lag}) must be less than number of observations ({n})")

    # Check sufficient observations after trimming
    n_valid = n - max_lag
    if n_valid < 10:
        raise ValueError(
            f"Insufficient observations after trimming for lags.\n"
            f"n={n}, max_lag={max_lag}, n_valid={n_valid}.\n"
            f"Need at least 10 valid observations."
        )

    return outcomes, treatments, states
