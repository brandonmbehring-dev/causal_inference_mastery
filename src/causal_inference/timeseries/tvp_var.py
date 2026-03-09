"""
Time-Varying Parameter VAR (TVP-VAR) Estimation.

Session 165: Kalman filter estimation for VAR with time-varying coefficients.

The TVP-VAR model (Primiceri 2005, Cogley & Sargent 2005):

State-Space Representation:
    Measurement:  Y_t = X_t β_t + ε_t,     ε_t ~ N(0, Σ)
    Transition:   β_t = β_{t-1} + w_t,     w_t ~ N(0, Q)

Where:
    Y_t: k×1 observation vector (VAR endogenous variables)
    X_t: k×(k²p+k) regressor matrix (lagged Y, intercepts)
    β_t: (k²p+k)×1 time-varying coefficient vector
    Σ: k×k observation covariance
    Q: (k²p+k)×(k²p+k) state transition covariance

The Kalman filter recursion:
    1. Predict: β_{t|t-1} = β_{t-1|t-1}, P_{t|t-1} = P_{t-1|t-1} + Q
    2. Update: v_t = Y_t - X_t β_{t|t-1} (innovation)
               F_t = X_t P_{t|t-1} X_t' + Σ (innovation covariance)
               K_t = P_{t|t-1} X_t' F_t⁻¹ (Kalman gain)
               β_{t|t} = β_{t|t-1} + K_t v_t
               P_{t|t} = (I - K_t X_t) P_{t|t-1} (Joseph form for stability)
    3. Log-likelihood: ℓ = -0.5 Σ_t [k log(2π) + log|F_t| + v_t' F_t⁻¹ v_t]

References
----------
Primiceri (2005). "Time Varying Structural VARs and Monetary Policy."
    Review of Economic Studies 72(3): 821-852.

Cogley & Sargent (2005). "Drifts and Volatilities: Monetary Policies
    and Outcomes in the Post WWII US." Review of Economic Dynamics 8: 262-302.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
import warnings

import numpy as np
from scipy import linalg

from causal_inference.timeseries.types import VARResult
from causal_inference.timeseries.var import var_estimate, _build_var_matrices


@dataclass
class TVPVARResult:
    """
    Result from Time-Varying Parameter VAR estimation.

    Primiceri (2005) style TVP-VAR with Kalman filter estimation.
    Coefficients follow random walk: β_t = β_{t-1} + w_t, w_t ~ N(0, Q).

    Attributes
    ----------
    coefficients_filtered : np.ndarray
        Shape (T, n_vars, n_vars*lags+1). Filtered coefficient estimates β_{t|t}.
        Index [:, i, :] gives coefficients for equation i at each time.
    coefficients_smoothed : np.ndarray
        Shape (T, n_vars, n_vars*lags+1). Smoothed coefficient estimates β_{t|T}.
        Full-sample optimal estimates using backward RTS smoother.
    covariance_filtered : np.ndarray
        Shape (T, state_dim, state_dim). Filtered state covariance P_{t|t}.
    covariance_smoothed : np.ndarray
        Shape (T, state_dim, state_dim). Smoothed state covariance P_{t|T}.
    innovations : np.ndarray
        Shape (T, n_vars). Innovation sequence v_t = Y_t - X_t β_{t|t-1}.
    innovation_covariance : np.ndarray
        Shape (T, n_vars, n_vars). Innovation covariance F_t.
    kalman_gain : np.ndarray
        Shape (T, state_dim, n_vars). Kalman gain K_t at each time.
    sigma : np.ndarray
        Shape (n_vars, n_vars). Observation covariance Σ.
    Q : np.ndarray
        Shape (state_dim, state_dim). State transition covariance Q.
    log_likelihood : float
        Total log-likelihood from prediction error decomposition.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    lags : int
        Number of VAR lags p.
    n_vars : int
        Number of endogenous variables k.
    n_obs : int
        Number of time observations T.
    n_obs_effective : int
        Effective observations (T - lags).
    state_dim : int
        State dimension (k × (k*p + 1)).
    var_names : list
        Variable names.
    initialization : str
        Initialization method used ("ols", "diffuse", "custom").
    """

    coefficients_filtered: np.ndarray
    coefficients_smoothed: Optional[np.ndarray]
    covariance_filtered: np.ndarray
    covariance_smoothed: Optional[np.ndarray]
    innovations: np.ndarray
    innovation_covariance: np.ndarray
    kalman_gain: np.ndarray
    sigma: np.ndarray
    Q: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    lags: int
    n_vars: int
    n_obs: int
    n_obs_effective: int
    state_dim: int
    var_names: List[str]
    initialization: str = "ols"

    @property
    def n_params_per_eq(self) -> int:
        """Number of parameters per equation (n_vars*lags + 1)."""
        return self.n_vars * self.lags + 1

    @property
    def has_smoothed(self) -> bool:
        """Whether smoothed estimates are available."""
        return self.coefficients_smoothed is not None

    def get_coefficients_at_time(
        self,
        t: int,
        smoothed: bool = True,
    ) -> np.ndarray:
        """
        Get coefficient matrix at specific time.

        Parameters
        ----------
        t : int
            Time index (0 to n_obs_effective-1)
        smoothed : bool
            Use smoothed (True) or filtered (False) estimates

        Returns
        -------
        np.ndarray
            Shape (n_vars, n_vars*lags+1) coefficient matrix at time t
        """
        if t < 0 or t >= self.n_obs_effective:
            raise ValueError(f"Time index {t} out of bounds [0, {self.n_obs_effective - 1}]")

        if smoothed and self.has_smoothed:
            return self.coefficients_smoothed[t]
        return self.coefficients_filtered[t]

    def get_lag_matrix_at_time(
        self,
        t: int,
        lag: int,
        smoothed: bool = True,
    ) -> np.ndarray:
        """
        Get coefficient matrix for specific lag at specific time.

        Parameters
        ----------
        t : int
            Time index
        lag : int
            Lag number (1 to lags)
        smoothed : bool
            Use smoothed or filtered estimates

        Returns
        -------
        np.ndarray
            Shape (n_vars, n_vars) coefficient matrix A_lag at time t
        """
        if lag < 1 or lag > self.lags:
            raise ValueError(f"Lag must be between 1 and {self.lags}, got {lag}")

        coef = self.get_coefficients_at_time(t, smoothed)
        start_idx = 1 + (lag - 1) * self.n_vars
        end_idx = start_idx + self.n_vars
        return coef[:, start_idx:end_idx]

    def get_intercepts_at_time(self, t: int, smoothed: bool = True) -> np.ndarray:
        """Get intercept vector at specific time."""
        coef = self.get_coefficients_at_time(t, smoothed)
        return coef[:, 0]

    def to_var_result_at_time(self, t: int, smoothed: bool = True) -> VARResult:
        """
        Convert to VARResult at specific time point.

        Parameters
        ----------
        t : int
            Time index
        smoothed : bool
            Use smoothed or filtered estimates

        Returns
        -------
        VARResult
            Standard VAR result using coefficients at time t
        """
        coef = self.get_coefficients_at_time(t, smoothed)
        return VARResult(
            coefficients=coef,
            residuals=self.innovations[t : t + 1],
            aic=self.aic,
            bic=self.bic,
            hqc=0.0,
            lags=self.lags,
            n_obs=self.n_obs,
            n_obs_effective=1,
            var_names=list(self.var_names),
            sigma=self.sigma,
            log_likelihood=self.log_likelihood,
        )

    def coefficient_trajectory(
        self,
        equation_idx: int,
        coef_idx: int,
        smoothed: bool = True,
    ) -> np.ndarray:
        """
        Get trajectory of a single coefficient over time.

        Parameters
        ----------
        equation_idx : int
            Which equation (0 to n_vars-1)
        coef_idx : int
            Which coefficient in that equation (0 to n_params_per_eq-1)
        smoothed : bool
            Use smoothed or filtered estimates

        Returns
        -------
        np.ndarray
            Shape (T,) coefficient values over time
        """
        if smoothed and self.has_smoothed:
            return self.coefficients_smoothed[:, equation_idx, coef_idx]
        return self.coefficients_filtered[:, equation_idx, coef_idx]


def tvp_var_estimate(
    data: np.ndarray,
    lags: int = 1,
    Q_init: Optional[np.ndarray] = None,
    Q_scale: float = 0.001,
    sigma_init: Optional[np.ndarray] = None,
    initialization: Literal["ols", "diffuse", "custom"] = "ols",
    beta_init: Optional[np.ndarray] = None,
    P_init: Optional[np.ndarray] = None,
    diffuse_scale: float = 1e6,
    smooth: bool = True,
    var_names: Optional[List[str]] = None,
) -> TVPVARResult:
    """
    Estimate Time-Varying Parameter VAR via Kalman filter.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, n_vars). Time series data.
    lags : int
        Number of VAR lags p (default 1).
    Q_init : np.ndarray, optional
        Shape (state_dim, state_dim). State transition covariance.
        If None, uses Q_scale * I.
    Q_scale : float
        Scale factor for Q when Q_init is None (default 0.001).
        Smaller values → smoother coefficient evolution.
    sigma_init : np.ndarray, optional
        Shape (n_vars, n_vars). Observation covariance.
        If None, estimated from OLS residuals.
    initialization : str
        How to initialize the Kalman filter:
        - "ols": Use OLS estimates on full sample (default)
        - "diffuse": Diffuse prior (large variance, zero mean)
        - "custom": Use provided beta_init and P_init
    beta_init : np.ndarray, optional
        Initial state β_{0|0}. Required if initialization="custom".
    P_init : np.ndarray, optional
        Initial state covariance P_{0|0}. Required if initialization="custom".
    diffuse_scale : float
        Scale for diffuse initialization (default 1e6).
    smooth : bool
        Whether to run RTS smoother (default True).
    var_names : list, optional
        Variable names. Auto-generated if None.

    Returns
    -------
    TVPVARResult
        Estimated TVP-VAR model.

    Raises
    ------
    ValueError
        If data has insufficient observations or invalid parameters.

    Example
    -------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = np.random.randn(200, 2)
    >>> result = tvp_var_estimate(data, lags=1)
    >>> print(f"Log-likelihood: {result.log_likelihood:.2f}")

    References
    ----------
    Primiceri (2005). "Time Varying Structural VARs and Monetary Policy."
    Cogley & Sargent (2005). "Drifts and Volatilities."
    """
    # Input validation
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")

    n_obs, n_vars = data.shape

    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}")

    n_params_per_eq = n_vars * lags + 1  # Include intercept
    state_dim = n_vars * n_params_per_eq
    n_obs_effective = n_obs - lags

    if n_obs_effective < 10:
        raise ValueError(f"Insufficient observations. Need at least {lags + 10}, got {n_obs}")

    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

    # Variable names
    if var_names is None:
        var_names = [f"var_{i + 1}" for i in range(n_vars)]
    elif len(var_names) != n_vars:
        raise ValueError(f"var_names length ({len(var_names)}) must match n_vars ({n_vars})")

    # Build design matrices
    Y, X_full = _build_var_matrices(data, lags, include_constant=True)

    # Convert to TVP-VAR format: X_t is block-diagonal
    # X_tvp[t] has shape (n_vars, state_dim) where each row corresponds to one equation
    X_tvp = _build_tvp_regressor_matrix(Y, X_full, n_vars, n_params_per_eq)

    # Initialize Kalman filter
    if initialization == "ols":
        beta_init_vec, P_init_mat, sigma = _initialize_from_ols(
            data, lags, n_vars, n_params_per_eq, state_dim
        )
    elif initialization == "diffuse":
        beta_init_vec = np.zeros(state_dim)
        P_init_mat = np.eye(state_dim) * diffuse_scale
        # Still need sigma from OLS
        _, _, sigma = _initialize_from_ols(data, lags, n_vars, n_params_per_eq, state_dim)
    elif initialization == "custom":
        if beta_init is None or P_init is None:
            raise ValueError("beta_init and P_init must be provided for custom initialization")
        beta_init_vec = np.asarray(beta_init, dtype=np.float64).ravel()
        P_init_mat = np.asarray(P_init, dtype=np.float64)

        if beta_init_vec.shape[0] != state_dim:
            raise ValueError(
                f"beta_init has wrong shape. Expected ({state_dim},), got {beta_init_vec.shape}"
            )
        if P_init_mat.shape != (state_dim, state_dim):
            raise ValueError(
                f"P_init has wrong shape. Expected ({state_dim}, {state_dim}), "
                f"got {P_init_mat.shape}"
            )

        # Get sigma from OLS or user-provided
        if sigma_init is not None:
            sigma = np.asarray(sigma_init, dtype=np.float64)
        else:
            _, _, sigma = _initialize_from_ols(data, lags, n_vars, n_params_per_eq, state_dim)
    else:
        raise ValueError(
            f"initialization must be 'ols', 'diffuse', or 'custom', got {initialization}"
        )

    # Override sigma if provided
    if sigma_init is not None:
        sigma = np.asarray(sigma_init, dtype=np.float64)
        if sigma.shape != (n_vars, n_vars):
            raise ValueError(
                f"sigma_init has wrong shape. Expected ({n_vars}, {n_vars}), got {sigma.shape}"
            )

    # State transition covariance Q
    if Q_init is not None:
        Q = np.asarray(Q_init, dtype=np.float64)
        if Q.shape != (state_dim, state_dim):
            raise ValueError(
                f"Q_init has wrong shape. Expected ({state_dim}, {state_dim}), got {Q.shape}"
            )
        # Ensure Q is positive semi-definite
        eigvals = np.linalg.eigvalsh(Q)
        if np.any(eigvals < -1e-10):
            raise ValueError("Q_init must be positive semi-definite")
    else:
        Q = np.eye(state_dim) * Q_scale

    # Run Kalman filter
    (
        beta_filt,
        P_filt,
        beta_pred,
        P_pred,
        innovations,
        innovation_cov,
        kalman_gains,
        log_lik,
    ) = _kalman_filter(Y, X_tvp, beta_init_vec, P_init_mat, Q, sigma, n_vars)

    # Reshape filtered coefficients: (T, state_dim) -> (T, n_vars, n_params_per_eq)
    coef_filtered = beta_filt.reshape(n_obs_effective, n_vars, n_params_per_eq)

    # RTS smoother
    if smooth:
        beta_smooth, P_smooth = _rts_smoother(beta_filt, P_filt, beta_pred, P_pred, Q)
        coef_smoothed = beta_smooth.reshape(n_obs_effective, n_vars, n_params_per_eq)
    else:
        coef_smoothed = None
        P_smooth = None

    # Information criteria
    # Effective number of parameters is harder to define for TVP-VAR
    # Using heuristic: state_dim + n_vars*(n_vars+1)/2 for Sigma
    n_params_effective = state_dim + n_vars * (n_vars + 1) // 2
    aic = -2 * log_lik + 2 * n_params_effective
    bic = -2 * log_lik + n_params_effective * np.log(n_obs_effective)

    return TVPVARResult(
        coefficients_filtered=coef_filtered,
        coefficients_smoothed=coef_smoothed,
        covariance_filtered=P_filt,
        covariance_smoothed=P_smooth,
        innovations=innovations,
        innovation_covariance=innovation_cov,
        kalman_gain=kalman_gains,
        sigma=sigma,
        Q=Q,
        log_likelihood=log_lik,
        aic=aic,
        bic=bic,
        lags=lags,
        n_vars=n_vars,
        n_obs=n_obs,
        n_obs_effective=n_obs_effective,
        state_dim=state_dim,
        var_names=list(var_names),
        initialization=initialization,
    )


def tvp_var_smooth(filtered_result: TVPVARResult) -> TVPVARResult:
    """
    Apply Rauch-Tung-Striebel (RTS) smoother to filtered TVP-VAR.

    Parameters
    ----------
    filtered_result : TVPVARResult
        Result from tvp_var_estimate with smooth=False

    Returns
    -------
    TVPVARResult
        Result with smoothed estimates populated
    """
    if filtered_result.has_smoothed:
        warnings.warn("Result already has smoothed estimates")
        return filtered_result

    # Need to reconstruct predicted states/covariances
    # For simplicity, re-run the filter to get predicted values
    # This is inefficient but correct
    raise NotImplementedError(
        "Use tvp_var_estimate with smooth=True instead. "
        "Standalone smoothing requires storing predicted states."
    )


def compute_tvp_irf(
    result: TVPVARResult,
    t: int,
    horizons: int = 20,
    shock_idx: int = 0,
    shock_size: float = 1.0,
    smoothed: bool = True,
    orthogonalize: bool = True,
) -> np.ndarray:
    """
    Compute impulse response function at specific time point.

    Parameters
    ----------
    result : TVPVARResult
        Estimated TVP-VAR
    t : int
        Time index to compute IRF at
    horizons : int
        Number of IRF horizons
    shock_idx : int
        Which structural shock (0-indexed)
    shock_size : float
        Size of shock (default 1.0 = one unit)
    smoothed : bool
        Use smoothed (True) or filtered (False) coefficients
    orthogonalize : bool
        Use Cholesky orthogonalization (default True)

    Returns
    -------
    np.ndarray
        Shape (n_vars, horizons+1). IRF for all variables.

    Example
    -------
    >>> result = tvp_var_estimate(data, lags=2)
    >>> irf = compute_tvp_irf(result, t=100, horizons=20, shock_idx=0)
    >>> print(f"IRF shape: {irf.shape}")  # (n_vars, 21)
    """
    if t < 0 or t >= result.n_obs_effective:
        raise ValueError(f"Time index {t} out of bounds [0, {result.n_obs_effective - 1}]")

    if shock_idx < 0 or shock_idx >= result.n_vars:
        raise ValueError(f"shock_idx {shock_idx} out of bounds [0, {result.n_vars - 1}]")

    if horizons < 0:
        raise ValueError(f"horizons must be >= 0, got {horizons}")

    n_vars = result.n_vars
    lags = result.lags

    # Get coefficient matrix at time t
    coef = result.get_coefficients_at_time(t, smoothed)

    # Compute VMA coefficients Φ_h (structural IRF)
    # Φ_0 = B₀⁻¹ (impact matrix from Cholesky)
    # Φ_h = Σ_{j=1}^{min(h,p)} Φ_{h-j} A_j

    if orthogonalize:
        # Cholesky decomposition of Σ gives impact matrix
        try:
            P = np.linalg.cholesky(result.sigma)  # Lower triangular
        except np.linalg.LinAlgError:
            # Regularize if not positive definite
            eigvals, eigvecs = np.linalg.eigh(result.sigma)
            eigvals = np.maximum(eigvals, 1e-10)
            sigma_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
            P = np.linalg.cholesky(sigma_reg)
    else:
        P = np.eye(n_vars)

    # Extract lag matrices A_1, ..., A_p
    A_matrices = []
    for lag in range(1, lags + 1):
        A_lag = result.get_lag_matrix_at_time(t, lag, smoothed)
        A_matrices.append(A_lag)

    # Compute IRF
    irf = np.zeros((n_vars, horizons + 1))

    # Initial shock: unit shock to variable shock_idx
    shock = np.zeros(n_vars)
    shock[shock_idx] = shock_size

    # h=0: immediate impact
    irf[:, 0] = P @ shock

    # Compute VMA coefficients iteratively
    Phi = [P]  # Φ_0 = P

    for h in range(1, horizons + 1):
        Phi_h = np.zeros((n_vars, n_vars))
        for j in range(min(h, lags)):
            Phi_h += Phi[h - 1 - j] @ A_matrices[j]
        Phi.append(Phi_h)
        irf[:, h] = Phi_h @ shock

    return irf


def compute_tvp_irf_all_times(
    result: TVPVARResult,
    horizons: int = 20,
    shock_idx: int = 0,
    shock_size: float = 1.0,
    smoothed: bool = True,
    orthogonalize: bool = True,
) -> np.ndarray:
    """
    Compute IRF at all time points (time-varying IRF).

    Parameters
    ----------
    result : TVPVARResult
        Estimated TVP-VAR
    horizons : int
        Number of IRF horizons
    shock_idx : int
        Which structural shock (0-indexed)
    shock_size : float
        Size of shock
    smoothed : bool
        Use smoothed or filtered coefficients
    orthogonalize : bool
        Use Cholesky orthogonalization

    Returns
    -------
    np.ndarray
        Shape (T, n_vars, horizons+1). Time-varying IRF.

    Example
    -------
    >>> result = tvp_var_estimate(data, lags=2)
    >>> irf_all = compute_tvp_irf_all_times(result, horizons=20, shock_idx=0)
    >>> print(f"IRF shape: {irf_all.shape}")  # (T, n_vars, 21)
    """
    T = result.n_obs_effective
    n_vars = result.n_vars

    irf_all = np.zeros((T, n_vars, horizons + 1))

    for t in range(T):
        irf_all[t] = compute_tvp_irf(
            result,
            t=t,
            horizons=horizons,
            shock_idx=shock_idx,
            shock_size=shock_size,
            smoothed=smoothed,
            orthogonalize=orthogonalize,
        )

    return irf_all


def check_tvp_stability(
    coefficients: np.ndarray,
    lags: int,
    n_vars: int,
) -> Tuple[bool, np.ndarray]:
    """
    Check VAR stability at a time point.

    A VAR is stable if all eigenvalues of the companion matrix
    are inside the unit circle.

    Parameters
    ----------
    coefficients : np.ndarray
        Shape (n_vars, n_vars*lags+1). Coefficient matrix including intercept.
    lags : int
        Number of lags
    n_vars : int
        Number of variables

    Returns
    -------
    is_stable : bool
        True if all eigenvalues inside unit circle
    eigenvalues : np.ndarray
        Eigenvalues of companion matrix

    Example
    -------
    >>> coef = result.get_coefficients_at_time(100)
    >>> is_stable, eigvals = check_tvp_stability(coef, result.lags, result.n_vars)
    >>> print(f"Stable: {is_stable}, max |eigenvalue|: {np.max(np.abs(eigvals)):.4f}")
    """
    # Build companion matrix
    # [A_1 A_2 ... A_p]
    # [I   0  ... 0  ]
    # [0   I  ... 0  ]
    # [       ...    ]
    # [0   0  ... I 0]

    companion_dim = n_vars * lags
    companion = np.zeros((companion_dim, companion_dim))

    # First n_vars rows: A_1, A_2, ..., A_p
    for lag in range(lags):
        start_col = lag * n_vars
        end_col = (lag + 1) * n_vars
        # Skip intercept at column 0
        coef_start = 1 + lag * n_vars
        coef_end = 1 + (lag + 1) * n_vars
        companion[:n_vars, start_col:end_col] = coefficients[:, coef_start:coef_end]

    # Identity blocks below
    if lags > 1:
        companion[n_vars:, : n_vars * (lags - 1)] = np.eye(n_vars * (lags - 1))

    eigenvalues = np.linalg.eigvals(companion)
    is_stable = np.all(np.abs(eigenvalues) < 1.0)

    return is_stable, eigenvalues


def check_tvp_stability_all_times(
    result: TVPVARResult,
    smoothed: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check VAR stability at all time points.

    Parameters
    ----------
    result : TVPVARResult
        Estimated TVP-VAR
    smoothed : bool
        Use smoothed or filtered coefficients

    Returns
    -------
    is_stable : np.ndarray
        Shape (T,) boolean array
    max_eigenvalue_modulus : np.ndarray
        Shape (T,) maximum eigenvalue modulus at each time
    """
    T = result.n_obs_effective
    is_stable = np.zeros(T, dtype=bool)
    max_mod = np.zeros(T)

    for t in range(T):
        coef = result.get_coefficients_at_time(t, smoothed)
        stable, eigvals = check_tvp_stability(coef, result.lags, result.n_vars)
        is_stable[t] = stable
        max_mod[t] = np.max(np.abs(eigvals))

    return is_stable, max_mod


# =============================================================================
# Helper Functions
# =============================================================================


def _build_tvp_regressor_matrix(
    Y: np.ndarray,
    X_full: np.ndarray,
    n_vars: int,
    n_params_per_eq: int,
) -> np.ndarray:
    """
    Build TVP-VAR regressor matrices.

    For TVP-VAR, we need X_t such that Y_t = X_t β_t where β_t is the
    vectorized coefficient matrix. X_t is block-diagonal.

    Parameters
    ----------
    Y : np.ndarray
        Shape (T, n_vars). Dependent variables.
    X_full : np.ndarray
        Shape (T, n_vars*lags+1). Standard VAR design matrix.
    n_vars : int
        Number of variables
    n_params_per_eq : int
        Parameters per equation

    Returns
    -------
    X_tvp : np.ndarray
        Shape (T, n_vars, state_dim). Block-diagonal regressor matrices.
    """
    T = Y.shape[0]
    state_dim = n_vars * n_params_per_eq

    X_tvp = np.zeros((T, n_vars, state_dim))

    for t in range(T):
        for eq in range(n_vars):
            # Each equation uses the same regressors but different coefficients
            start_col = eq * n_params_per_eq
            end_col = (eq + 1) * n_params_per_eq
            X_tvp[t, eq, start_col:end_col] = X_full[t, :]

    return X_tvp


def _kalman_filter(
    Y: np.ndarray,
    X: np.ndarray,
    beta_init: np.ndarray,
    P_init: np.ndarray,
    Q: np.ndarray,
    sigma: np.ndarray,
    n_vars: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Run Kalman filter for TVP-VAR.

    Parameters
    ----------
    Y : np.ndarray
        Shape (T, n_vars). Observations.
    X : np.ndarray
        Shape (T, n_vars, state_dim). Regressor matrices.
    beta_init : np.ndarray
        Shape (state_dim,). Initial state.
    P_init : np.ndarray
        Shape (state_dim, state_dim). Initial state covariance.
    Q : np.ndarray
        Shape (state_dim, state_dim). State transition covariance.
    sigma : np.ndarray
        Shape (n_vars, n_vars). Observation covariance.
    n_vars : int
        Number of variables.

    Returns
    -------
    beta_filt : np.ndarray
        Shape (T, state_dim). Filtered states.
    P_filt : np.ndarray
        Shape (T, state_dim, state_dim). Filtered covariances.
    beta_pred : np.ndarray
        Shape (T, state_dim). Predicted states.
    P_pred : np.ndarray
        Shape (T, state_dim, state_dim). Predicted covariances.
    innovations : np.ndarray
        Shape (T, n_vars). Innovation sequence.
    innovation_cov : np.ndarray
        Shape (T, n_vars, n_vars). Innovation covariances.
    kalman_gains : np.ndarray
        Shape (T, state_dim, n_vars). Kalman gains.
    log_lik : float
        Log-likelihood.
    """
    T = Y.shape[0]
    state_dim = beta_init.shape[0]

    # Storage
    beta_filt = np.zeros((T, state_dim))
    P_filt = np.zeros((T, state_dim, state_dim))
    beta_pred = np.zeros((T, state_dim))
    P_pred = np.zeros((T, state_dim, state_dim))
    innovations = np.zeros((T, n_vars))
    innovation_cov = np.zeros((T, n_vars, n_vars))
    kalman_gains = np.zeros((T, state_dim, n_vars))

    log_lik = 0.0
    log_2pi = np.log(2 * np.pi)

    # Initialize
    beta_curr = beta_init.copy()
    P_curr = P_init.copy()

    for t in range(T):
        # === Prediction step ===
        # β_{t|t-1} = β_{t-1|t-1} (random walk)
        beta_pred_t = beta_curr
        # P_{t|t-1} = P_{t-1|t-1} + Q
        P_pred_t = P_curr + Q

        beta_pred[t] = beta_pred_t
        P_pred[t] = P_pred_t

        # === Update step ===
        X_t = X[t]  # Shape: (n_vars, state_dim)

        # Innovation: v_t = Y_t - X_t β_{t|t-1}
        y_pred = X_t @ beta_pred_t
        v_t = Y[t] - y_pred
        innovations[t] = v_t

        # Innovation covariance: F_t = X_t P_{t|t-1} X_t' + Σ
        F_t = X_t @ P_pred_t @ X_t.T + sigma
        innovation_cov[t] = F_t

        # Kalman gain: K_t = P_{t|t-1} X_t' F_t⁻¹
        try:
            F_inv = np.linalg.solve(F_t, np.eye(n_vars))
        except np.linalg.LinAlgError:
            # Regularize if singular
            F_t_reg = F_t + np.eye(n_vars) * 1e-6
            F_inv = np.linalg.solve(F_t_reg, np.eye(n_vars))

        K_t = P_pred_t @ X_t.T @ F_inv
        kalman_gains[t] = K_t

        # State update: β_{t|t} = β_{t|t-1} + K_t v_t
        beta_filt_t = beta_pred_t + K_t @ v_t
        beta_filt[t] = beta_filt_t

        # Covariance update (Joseph form for numerical stability)
        P_filt_t = _joseph_form_update(P_pred_t, K_t, X_t, sigma)
        P_filt[t] = P_filt_t

        # Log-likelihood contribution
        # ℓ_t = -0.5 * (k*log(2π) + log|F_t| + v_t' F_t⁻¹ v_t)
        sign, logdet = np.linalg.slogdet(F_t)
        if sign <= 0:
            logdet = np.sum(np.log(np.maximum(np.diag(F_t), 1e-10)))

        quad_form = v_t @ F_inv @ v_t
        log_lik_t = -0.5 * (n_vars * log_2pi + logdet + quad_form)
        log_lik += log_lik_t

        # Update for next iteration
        beta_curr = beta_filt_t
        P_curr = P_filt_t

    return (
        beta_filt,
        P_filt,
        beta_pred,
        P_pred,
        innovations,
        innovation_cov,
        kalman_gains,
        log_lik,
    )


def _rts_smoother(
    beta_filt: np.ndarray,
    P_filt: np.ndarray,
    beta_pred: np.ndarray,
    P_pred: np.ndarray,
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rauch-Tung-Striebel backward smoother.

    Parameters
    ----------
    beta_filt : np.ndarray
        Shape (T, state_dim). Filtered states.
    P_filt : np.ndarray
        Shape (T, state_dim, state_dim). Filtered covariances.
    beta_pred : np.ndarray
        Shape (T, state_dim). Predicted states.
    P_pred : np.ndarray
        Shape (T, state_dim, state_dim). Predicted covariances.
    Q : np.ndarray
        Shape (state_dim, state_dim). State transition covariance.

    Returns
    -------
    beta_smooth : np.ndarray
        Shape (T, state_dim). Smoothed states.
    P_smooth : np.ndarray
        Shape (T, state_dim, state_dim). Smoothed covariances.
    """
    T, state_dim = beta_filt.shape

    beta_smooth = np.zeros_like(beta_filt)
    P_smooth = np.zeros_like(P_filt)

    # Initialize at T-1 (last time point)
    beta_smooth[T - 1] = beta_filt[T - 1]
    P_smooth[T - 1] = P_filt[T - 1]

    # Backward recursion
    for t in range(T - 2, -1, -1):
        # J_t = P_{t|t} P_{t+1|t}⁻¹
        # For random walk: P_{t+1|t} = P_{t|t} + Q
        P_pred_next = P_pred[t + 1]

        try:
            J_t = P_filt[t] @ np.linalg.solve(P_pred_next, np.eye(state_dim)).T
        except np.linalg.LinAlgError:
            # Regularize if singular
            P_pred_reg = P_pred_next + np.eye(state_dim) * 1e-6
            J_t = P_filt[t] @ np.linalg.solve(P_pred_reg, np.eye(state_dim)).T

        # β_{t|T} = β_{t|t} + J_t (β_{t+1|T} - β_{t+1|t})
        beta_smooth[t] = beta_filt[t] + J_t @ (beta_smooth[t + 1] - beta_pred[t + 1])

        # P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t'
        P_smooth[t] = P_filt[t] + J_t @ (P_smooth[t + 1] - P_pred_next) @ J_t.T

    return beta_smooth, P_smooth


def _joseph_form_update(
    P_pred: np.ndarray,
    K: np.ndarray,
    X_t: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Joseph form covariance update for numerical stability.

    P_{t|t} = (I - K_t X_t) P_{t|t-1} (I - K_t X_t)' + K_t Σ K_t'

    This form guarantees positive semi-definiteness even with
    numerical errors, unlike the standard form P = (I - KX)P.

    Parameters
    ----------
    P_pred : np.ndarray
        Shape (state_dim, state_dim). Predicted covariance.
    K : np.ndarray
        Shape (state_dim, n_vars). Kalman gain.
    X_t : np.ndarray
        Shape (n_vars, state_dim). Observation matrix.
    sigma : np.ndarray
        Shape (n_vars, n_vars). Observation covariance.

    Returns
    -------
    P_filt : np.ndarray
        Shape (state_dim, state_dim). Updated covariance.
    """
    state_dim = P_pred.shape[0]
    I = np.eye(state_dim)

    # (I - K X)
    I_KX = I - K @ X_t

    # Joseph form
    P_filt = I_KX @ P_pred @ I_KX.T + K @ sigma @ K.T

    # Ensure symmetry
    P_filt = 0.5 * (P_filt + P_filt.T)

    return P_filt


def _initialize_from_ols(
    data: np.ndarray,
    lags: int,
    n_vars: int,
    n_params_per_eq: int,
    state_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize TVP-VAR from OLS VAR estimates.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, n_vars). Data.
    lags : int
        Number of lags.
    n_vars : int
        Number of variables.
    n_params_per_eq : int
        Parameters per equation.
    state_dim : int
        Total state dimension.

    Returns
    -------
    beta_init : np.ndarray
        Shape (state_dim,). Initial state (vectorized OLS coefficients).
    P_init : np.ndarray
        Shape (state_dim, state_dim). Initial covariance.
    sigma : np.ndarray
        Shape (n_vars, n_vars). Observation covariance.
    """
    # Estimate OLS VAR
    var_result = var_estimate(data, lags=lags, include_constant=True)

    # Vectorize coefficients (row by row)
    beta_init = var_result.coefficients.ravel()

    # Initial covariance: scaled identity based on OLS residual variance
    # This provides a reasonable diffuse prior centered at OLS
    resid_var = np.var(var_result.residuals)
    P_init = np.eye(state_dim) * resid_var * 10  # Scale up for some uncertainty

    # Observation covariance from OLS
    sigma = var_result.sigma

    return beta_init, P_init, sigma


def coefficient_change_test(
    result: TVPVARResult,
    equation_idx: int,
    coef_idx: int,
    smoothed: bool = True,
) -> Tuple[float, float]:
    """
    Test for significant coefficient change over time.

    Uses a simple variance ratio test comparing actual coefficient
    variance to expected variance under constant coefficients.

    Parameters
    ----------
    result : TVPVARResult
        Estimated TVP-VAR
    equation_idx : int
        Which equation
    coef_idx : int
        Which coefficient
    smoothed : bool
        Use smoothed estimates

    Returns
    -------
    variance_ratio : float
        Ratio of observed to expected coefficient variance
    p_value : float
        Approximate p-value (chi-squared based)
    """
    from scipy import stats

    trajectory = result.coefficient_trajectory(equation_idx, coef_idx, smoothed)

    # Observed variance
    obs_var = np.var(trajectory, ddof=1)

    # Expected variance under constant β: comes from filtering uncertainty
    # Use average diagonal element of P for this coefficient
    coef_flat_idx = equation_idx * result.n_params_per_eq + coef_idx

    if smoothed and result.has_smoothed:
        P = result.covariance_smoothed
    else:
        P = result.covariance_filtered

    expected_var = np.mean(P[:, coef_flat_idx, coef_flat_idx])

    if expected_var < 1e-10:
        return 0.0, 1.0

    variance_ratio = obs_var / expected_var

    # Under null of constant coefficients, variance ratio follows chi-squared
    T = result.n_obs_effective
    test_stat = variance_ratio * T
    p_value = 1.0 - stats.chi2.cdf(test_stat, T - 1)

    return variance_ratio, p_value
