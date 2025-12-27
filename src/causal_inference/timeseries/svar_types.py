"""
Structural VAR Types.

Session 137: Data structures for Structural VAR analysis.

The key distinction from reduced-form VAR:
- Reduced form: u_t are correlated (contemporaneous effects mixed)
- Structural: ε_t are orthogonal (interpretable shocks)

Identification: B₀ Y_t = B₁ Y_{t-1} + ... + Bₚ Y_{t-p} + ε_t
where u_t = B₀⁻¹ ε_t
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union
import numpy as np

from causal_inference.timeseries.types import VARResult


class IdentificationMethod(Enum):
    """SVAR identification method."""

    CHOLESKY = "cholesky"
    """Recursive identification via Cholesky decomposition.
    Assumes lower triangular B₀⁻¹: first variable unaffected contemporaneously.
    """

    SHORT_RUN = "short_run"
    """Short-run restrictions: zero constraints on B₀."""

    LONG_RUN = "long_run"
    """Long-run restrictions (Blanchard-Quah style)."""

    SIGN = "sign"
    """Sign restrictions on impulse responses."""


@dataclass
class SVARResult:
    """
    Result from Structural VAR estimation.

    SVAR decomposes reduced-form errors u_t into structural shocks ε_t:
        u_t = B₀⁻¹ ε_t

    where ε_t ~ N(0, I) are orthogonal structural shocks.

    Attributes
    ----------
    var_result : VARResult
        Underlying reduced-form VAR estimation
    B0_inv : np.ndarray
        Shape (n_vars, n_vars) impact matrix.
        Maps structural shocks to reduced-form errors: u_t = B₀⁻¹ ε_t
    B0 : np.ndarray
        Shape (n_vars, n_vars) structural matrix.
        B₀ = (B₀⁻¹)⁻¹
    structural_shocks : np.ndarray
        Shape (n_obs_effective, n_vars) structural shock series.
        ε_t = B₀ u_t
    identification : IdentificationMethod
        Identification strategy used
    n_restrictions : int
        Number of identifying restrictions imposed
    is_just_identified : bool
        True if exactly identified (n_restrictions = n_vars*(n_vars-1)/2)
    is_over_identified : bool
        True if over-identified (more restrictions than needed)
    log_likelihood : float
        Log-likelihood of structural model
    ordering : Optional[List[str]]
        Variable ordering (for Cholesky identification)
    restrictions : Optional[Dict]
        Details of restrictions imposed (for non-recursive identification)

    Example
    -------
    >>> from causal_inference.timeseries import var_estimate, cholesky_svar
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = np.random.randn(200, 3)
    >>> var_result = var_estimate(data, lags=2)
    >>> svar_result = cholesky_svar(var_result)
    >>> print(f"B0_inv shape: {svar_result.B0_inv.shape}")
    """

    var_result: VARResult
    B0_inv: np.ndarray
    B0: np.ndarray
    structural_shocks: np.ndarray
    identification: IdentificationMethod
    n_restrictions: int
    is_just_identified: bool
    is_over_identified: bool
    log_likelihood: float
    ordering: Optional[List[str]] = None
    restrictions: Optional[Dict] = None

    @property
    def n_vars(self) -> int:
        """Number of variables."""
        return self.var_result.n_vars

    @property
    def lags(self) -> int:
        """VAR lag order."""
        return self.var_result.lags

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self.var_result.n_obs_effective

    @property
    def var_names(self) -> List[str]:
        """Variable names."""
        return self.var_result.var_names

    def get_structural_coefficient(self, shock_var: int, response_var: int) -> float:
        """
        Get contemporaneous impact of structural shock on variable.

        Parameters
        ----------
        shock_var : int
            Index of shock variable
        response_var : int
            Index of response variable

        Returns
        -------
        float
            Impact coefficient: (B₀⁻¹)_{response, shock}
        """
        return self.B0_inv[response_var, shock_var]

    def __repr__(self) -> str:
        id_method = self.identification.value
        return (
            f"SVARResult(n_vars={self.n_vars}, lags={self.lags}, "
            f"identification={id_method}, n_restrictions={self.n_restrictions})"
        )


@dataclass
class IRFResult:
    """
    Impulse Response Function result.

    IRF measures the response of each variable to a one-unit structural shock.

    At horizon h:
        IRF_h = Φ_h · B₀⁻¹

    where Φ_h is the VMA coefficient at horizon h.

    Attributes
    ----------
    irf : np.ndarray
        Shape (n_vars, n_vars, horizons+1) impulse response matrix.
        irf[i, j, h] = response of var i to shock in var j at horizon h.
        h=0 is the contemporaneous impact.
    irf_lower : Optional[np.ndarray]
        Lower confidence band (same shape as irf)
    irf_upper : Optional[np.ndarray]
        Upper confidence band (same shape as irf)
    horizons : int
        Maximum horizon (irf goes from 0 to horizons inclusive)
    cumulative : bool
        If True, irf is cumulative (sum up to horizon h)
    orthogonalized : bool
        If True, shocks are orthogonalized (structural)
    var_names : List[str]
        Variable names
    alpha : float
        Confidence level for bands (e.g., 0.05 for 95% CI)
    n_bootstrap : int
        Number of bootstrap replications (0 if no CI computed)

    Example
    -------
    >>> irf_result = compute_irf(svar_result, horizons=20)
    >>> # Response of var 2 to shock in var 1 at horizon 5
    >>> response = irf_result.irf[2, 1, 5]
    """

    irf: np.ndarray
    irf_lower: Optional[np.ndarray]
    irf_upper: Optional[np.ndarray]
    horizons: int
    cumulative: bool
    orthogonalized: bool
    var_names: List[str]
    alpha: float = 0.05
    n_bootstrap: int = 0

    @property
    def n_vars(self) -> int:
        """Number of variables."""
        return self.irf.shape[0]

    @property
    def has_confidence_bands(self) -> bool:
        """Whether confidence bands are available."""
        return self.irf_lower is not None and self.irf_upper is not None

    def get_response(
        self,
        response_var: Union[int, str],
        shock_var: Union[int, str],
        horizon: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get impulse response for specific shock-response pair.

        Parameters
        ----------
        response_var : int or str
            Response variable (index or name)
        shock_var : int or str
            Shock variable (index or name)
        horizon : int, optional
            Specific horizon. If None, returns all horizons.

        Returns
        -------
        np.ndarray
            Response values (scalar if horizon specified, else 1D array)
        """
        # Convert names to indices
        if isinstance(response_var, str):
            response_idx = self.var_names.index(response_var)
        else:
            response_idx = response_var

        if isinstance(shock_var, str):
            shock_idx = self.var_names.index(shock_var)
        else:
            shock_idx = shock_var

        if horizon is not None:
            return self.irf[response_idx, shock_idx, horizon]
        return self.irf[response_idx, shock_idx, :]

    def get_response_with_ci(
        self,
        response_var: Union[int, str],
        shock_var: Union[int, str],
    ) -> Dict[str, np.ndarray]:
        """
        Get impulse response with confidence bands.

        Returns
        -------
        dict
            Keys: 'irf', 'lower', 'upper', 'horizon'
        """
        if isinstance(response_var, str):
            response_idx = self.var_names.index(response_var)
        else:
            response_idx = response_var

        if isinstance(shock_var, str):
            shock_idx = self.var_names.index(shock_var)
        else:
            shock_idx = shock_var

        result = {
            "irf": self.irf[response_idx, shock_idx, :],
            "horizon": np.arange(self.horizons + 1),
        }

        if self.has_confidence_bands:
            result["lower"] = self.irf_lower[response_idx, shock_idx, :]
            result["upper"] = self.irf_upper[response_idx, shock_idx, :]

        return result

    def __repr__(self) -> str:
        cum_str = ", cumulative" if self.cumulative else ""
        ci_str = f", CI={100*(1-self.alpha):.0f}%" if self.has_confidence_bands else ""
        return (
            f"IRFResult(n_vars={self.n_vars}, horizons={self.horizons}"
            f"{cum_str}{ci_str})"
        )


@dataclass
class FEVDResult:
    """
    Forecast Error Variance Decomposition result.

    FEVD measures the proportion of forecast error variance of each variable
    attributable to each structural shock.

    At horizon h:
        FEVD_{i,j,h} = Σ_{k=0}^{h} (IRF_{i,j,k})² / Σ_{m} Σ_{k=0}^{h} (IRF_{i,m,k})²

    Rows sum to 1 at each horizon.

    Attributes
    ----------
    fevd : np.ndarray
        Shape (n_vars, n_vars, horizons+1) decomposition matrix.
        fevd[i, j, h] = proportion of var i's h-step FEV due to shock j.
        Sum over j equals 1 for each (i, h).
    horizons : int
        Maximum horizon
    var_names : List[str]
        Variable names

    Example
    -------
    >>> fevd_result = compute_fevd(svar_result, horizons=20)
    >>> # Proportion of var 2's 10-step FEV due to shock in var 1
    >>> prop = fevd_result.fevd[2, 1, 10]
    >>> # Check row sums to 1
    >>> np.allclose(fevd_result.fevd[:, :, 10].sum(axis=1), 1.0)
    True
    """

    fevd: np.ndarray
    horizons: int
    var_names: List[str]

    @property
    def n_vars(self) -> int:
        """Number of variables."""
        return self.fevd.shape[0]

    def get_decomposition(
        self,
        response_var: Union[int, str],
        horizon: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get variance decomposition for a specific variable.

        Parameters
        ----------
        response_var : int or str
            Variable whose FEV is being decomposed
        horizon : int, optional
            Specific horizon. If None, returns all horizons.

        Returns
        -------
        np.ndarray
            Proportions due to each shock.
            Shape (n_vars,) if horizon specified, else (n_vars, horizons+1)
        """
        if isinstance(response_var, str):
            response_idx = self.var_names.index(response_var)
        else:
            response_idx = response_var

        if horizon is not None:
            return self.fevd[response_idx, :, horizon]
        return self.fevd[response_idx, :, :]

    def get_contribution(
        self,
        response_var: Union[int, str],
        shock_var: Union[int, str],
        horizon: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get contribution of specific shock to variable's FEV.

        Parameters
        ----------
        response_var : int or str
            Response variable
        shock_var : int or str
            Shock variable
        horizon : int, optional
            Specific horizon

        Returns
        -------
        np.ndarray or float
            Proportion of FEV due to shock
        """
        if isinstance(response_var, str):
            response_idx = self.var_names.index(response_var)
        else:
            response_idx = response_var

        if isinstance(shock_var, str):
            shock_idx = self.var_names.index(shock_var)
        else:
            shock_idx = shock_var

        if horizon is not None:
            return self.fevd[response_idx, shock_idx, horizon]
        return self.fevd[response_idx, shock_idx, :]

    def validate_rows_sum_to_one(self, tol: float = 1e-10) -> bool:
        """Check that FEVD rows sum to 1 at each horizon."""
        for h in range(self.horizons + 1):
            row_sums = self.fevd[:, :, h].sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=tol):
                return False
        return True

    def __repr__(self) -> str:
        return f"FEVDResult(n_vars={self.n_vars}, horizons={self.horizons})"


@dataclass
class HistoricalDecompositionResult:
    """
    Historical Decomposition result.

    Decomposes each variable's time path into contributions from
    each structural shock.

    Attributes
    ----------
    contributions : np.ndarray
        Shape (n_vars, n_vars, n_obs) contribution matrix.
        contributions[i, j, t] = contribution of shock j to var i at time t
    baseline : np.ndarray
        Shape (n_vars, n_obs) baseline (deterministic component)
    actual : np.ndarray
        Shape (n_vars, n_obs) actual observed values
    var_names : List[str]
        Variable names
    """

    contributions: np.ndarray
    baseline: np.ndarray
    actual: np.ndarray
    var_names: List[str]

    @property
    def n_vars(self) -> int:
        """Number of variables."""
        return self.contributions.shape[0]

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self.contributions.shape[2]

    def get_shock_contribution(
        self,
        response_var: Union[int, str],
        shock_var: Union[int, str],
    ) -> np.ndarray:
        """Get time series of shock contribution to variable."""
        if isinstance(response_var, str):
            response_idx = self.var_names.index(response_var)
        else:
            response_idx = response_var

        if isinstance(shock_var, str):
            shock_idx = self.var_names.index(shock_var)
        else:
            shock_idx = shock_var

        return self.contributions[response_idx, shock_idx, :]

    def __repr__(self) -> str:
        return f"HistoricalDecompositionResult(n_vars={self.n_vars}, n_obs={self.n_obs})"
