"""
Structural VAR Estimation.

Session 137: SVAR identification and estimation.

The key insight: reduced-form VAR errors u_t are correlated because they
mix structural shocks. SVAR identifies the structural shocks ε_t by
imposing restrictions on the impact matrix B₀⁻¹.

Reduced form: Y_t = A₁ Y_{t-1} + ... + Aₚ Y_{t-p} + u_t
Structural:   B₀ Y_t = B₁ Y_{t-1} + ... + Bₚ Y_{t-p} + ε_t

Relationship: u_t = B₀⁻¹ ε_t, where Σ_u = B₀⁻¹ Σ_ε (B₀⁻¹)'

For orthogonal structural shocks (Σ_ε = I):
    Σ_u = B₀⁻¹ (B₀⁻¹)'

Identification requires n(n-1)/2 restrictions to recover B₀⁻¹ from Σ_u.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import linalg

from causal_inference.timeseries.types import VARResult
from causal_inference.timeseries.svar_types import (
    SVARResult,
    IdentificationMethod,
)


def cholesky_svar(
    var_result: VARResult,
    ordering: Optional[List[str]] = None,
) -> SVARResult:
    """
    Structural VAR using Cholesky (recursive) identification.

    The Cholesky decomposition of Σ_u yields a lower triangular B₀⁻¹:
        Σ_u = P P'  =>  B₀⁻¹ = P

    Interpretation: Variables are ordered causally.
    - First variable is exogenous (not affected contemporaneously by others)
    - Second variable affected only by first contemporaneously
    - Last variable affected by all others contemporaneously

    Parameters
    ----------
    var_result : VARResult
        Estimated reduced-form VAR model
    ordering : List[str], optional
        Variable ordering for Cholesky decomposition.
        If None, uses order in var_result.var_names.
        First variable is most exogenous.

    Returns
    -------
    SVARResult
        Structural VAR estimation results

    Raises
    ------
    ValueError
        If ordering contains invalid variable names
    np.linalg.LinAlgError
        If covariance matrix is not positive definite

    Example
    -------
    >>> import numpy as np
    >>> from causal_inference.timeseries import var_estimate, cholesky_svar
    >>> np.random.seed(42)
    >>> data = np.random.randn(200, 3)
    >>> var_result = var_estimate(data, lags=2)
    >>> svar_result = cholesky_svar(var_result)
    >>> print(f"Impact matrix shape: {svar_result.B0_inv.shape}")

    Notes
    -----
    The Cholesky identification is just-identified (exactly n(n-1)/2 zeros).
    Results depend on variable ordering - economic theory should guide this choice.
    """
    n_vars = var_result.n_vars
    sigma = var_result.sigma

    # Handle ordering
    if ordering is not None:
        if len(ordering) != n_vars:
            raise ValueError(
                f"ordering has {len(ordering)} elements, expected {n_vars}"
            )
        for name in ordering:
            if name not in var_result.var_names:
                raise ValueError(f"Variable '{name}' not in VAR model")

        # Permute covariance matrix according to ordering
        perm = [var_result.var_names.index(name) for name in ordering]
        P = np.eye(n_vars)[perm, :]
        sigma_ordered = P @ sigma @ P.T
    else:
        ordering = var_result.var_names
        sigma_ordered = sigma
        P = np.eye(n_vars)

    # Cholesky decomposition: Σ = L L'
    try:
        L = linalg.cholesky(sigma_ordered, lower=True)
    except linalg.LinAlgError as e:
        raise ValueError(
            f"Covariance matrix is not positive definite. "
            f"Check for collinearity or insufficient data. Original error: {e}"
        )

    # B₀⁻¹ = L (in reordered space)
    B0_inv_ordered = L

    # Transform back to original ordering
    if ordering != var_result.var_names:
        P_inv = P.T  # Orthogonal, so inverse = transpose
        B0_inv = P_inv @ B0_inv_ordered @ P_inv.T
    else:
        B0_inv = B0_inv_ordered

    # Compute B₀ = (B₀⁻¹)⁻¹
    B0 = linalg.inv(B0_inv)

    # Compute structural shocks: ε_t = B₀ u_t
    residuals = var_result.residuals
    structural_shocks = (B0 @ residuals.T).T

    # Identification info
    n_restrictions = n_vars * (n_vars - 1) // 2  # Lower triangular zeros

    # Log-likelihood (same as reduced form, identification doesn't change fit)
    log_likelihood = var_result.log_likelihood

    return SVARResult(
        var_result=var_result,
        B0_inv=B0_inv,
        B0=B0,
        structural_shocks=structural_shocks,
        identification=IdentificationMethod.CHOLESKY,
        n_restrictions=n_restrictions,
        is_just_identified=True,
        is_over_identified=False,
        log_likelihood=log_likelihood,
        ordering=list(ordering),
    )


def short_run_svar(
    var_result: VARResult,
    A_restrictions: Optional[np.ndarray] = None,
    B_restrictions: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> SVARResult:
    """
    Structural VAR with short-run restrictions.

    Estimates the AB-model:
        A u_t = B ε_t

    where A and B are (n_vars x n_vars) matrices with restrictions.
    For the B-model (A = I): u_t = B ε_t, so B = B₀⁻¹.

    Parameters
    ----------
    var_result : VARResult
        Estimated reduced-form VAR model
    A_restrictions : np.ndarray, optional
        Shape (n_vars, n_vars) restriction matrix for A.
        np.nan indicates free parameter, numeric value indicates fixed.
        Default: identity matrix (B-model).
    B_restrictions : np.ndarray, optional
        Shape (n_vars, n_vars) restriction matrix for B.
        np.nan indicates free parameter, numeric value indicates fixed.
        Default: lower triangular (Cholesky-like).
    max_iter : int
        Maximum iterations for optimization
    tol : float
        Convergence tolerance

    Returns
    -------
    SVARResult
        Structural VAR estimation results

    Notes
    -----
    For just-identification, need n² - n(n+1)/2 = n(n-1)/2 restrictions
    on the free elements of A and B combined.
    """
    n_vars = var_result.n_vars
    sigma = var_result.sigma

    # Default: B-model with lower triangular B
    if A_restrictions is None:
        A_restrictions = np.eye(n_vars)

    if B_restrictions is None:
        B_restrictions = np.full((n_vars, n_vars), np.nan)
        # Lower triangular free, upper triangular zero
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                B_restrictions[i, j] = 0.0

    # Count restrictions and free parameters
    A_fixed = ~np.isnan(A_restrictions)
    B_fixed = ~np.isnan(B_restrictions)
    n_free_A = np.sum(~A_fixed)
    n_free_B = np.sum(~B_fixed)
    n_free = n_free_A + n_free_B

    # For identification: need n_free <= n*(n+1)/2
    n_needed = n_vars * (n_vars + 1) // 2
    n_restrictions = 2 * n_vars**2 - n_free

    is_over_identified = n_free < n_needed
    is_just_identified = n_free == n_needed

    if n_free > n_needed:
        raise ValueError(
            f"Under-identified: {n_free} free parameters but only {n_needed} "
            f"moment conditions. Need {n_free - n_needed} more restrictions."
        )

    # Initialize with Cholesky for B
    L = linalg.cholesky(sigma, lower=True)
    B_init = L.copy()
    A_init = np.eye(n_vars)

    # Apply fixed restrictions
    A = np.where(A_fixed, A_restrictions, A_init)
    B = np.where(B_fixed, B_restrictions, B_init)

    # Iterative estimation (scoring algorithm)
    # For just-identified B-model with lower triangular B, Cholesky is exact
    if np.allclose(A, np.eye(n_vars)) and is_just_identified:
        # Direct Cholesky solution
        B0_inv = L
    else:
        # Need numerical optimization for general case
        B0_inv = _optimize_svar_ab(
            sigma, A, B, A_fixed, B_fixed, max_iter, tol
        )

    B0 = linalg.inv(B0_inv)

    # Structural shocks
    structural_shocks = (B0 @ var_result.residuals.T).T

    return SVARResult(
        var_result=var_result,
        B0_inv=B0_inv,
        B0=B0,
        structural_shocks=structural_shocks,
        identification=IdentificationMethod.SHORT_RUN,
        n_restrictions=n_restrictions,
        is_just_identified=is_just_identified,
        is_over_identified=is_over_identified,
        log_likelihood=var_result.log_likelihood,
        restrictions={
            "A": A_restrictions.tolist() if A_restrictions is not None else None,
            "B": B_restrictions.tolist() if B_restrictions is not None else None,
        },
    )


def _optimize_svar_ab(
    sigma: np.ndarray,
    A_init: np.ndarray,
    B_init: np.ndarray,
    A_fixed: np.ndarray,
    B_fixed: np.ndarray,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """
    Optimize AB-model using iterative scoring.

    For the AB-model: A Σ_u A' = B B'

    This is a simplified implementation; for production use,
    consider specialized SVAR packages.
    """
    n = sigma.shape[0]
    A = A_init.copy()
    B = B_init.copy()

    for iteration in range(max_iter):
        # Target: A Σ_u A' = B B'
        target = A @ sigma @ A.T

        # Update B to match (Cholesky of target)
        try:
            L = linalg.cholesky(target, lower=True)
        except linalg.LinAlgError:
            # Regularize if needed
            target += 1e-6 * np.eye(n)
            L = linalg.cholesky(target, lower=True)

        B_new = np.where(B_fixed, B, L)

        # Check convergence
        if np.max(np.abs(B_new - B)) < tol:
            break

        B = B_new

    # B₀⁻¹ = A⁻¹ B (for A u = B ε, we have u = A⁻¹ B ε)
    A_inv = linalg.inv(A)
    B0_inv = A_inv @ B

    return B0_inv


def companion_form(var_result: VARResult) -> np.ndarray:
    """
    Build VAR companion matrix for IRF computation.

    For VAR(p): Y_t = A₁ Y_{t-1} + ... + Aₚ Y_{t-p} + u_t

    The companion form is:
    [Y_t    ]   [A₁ A₂ ... A_{p-1} Aₚ] [Y_{t-1}  ]   [u_t]
    [Y_{t-1}] = [I  0  ... 0      0 ] [Y_{t-2}  ] + [0  ]
    [  ...  ]   [0  I  ... 0      0 ] [  ...    ]   [...]
    [Y_{t-p+1}] [0  0  ... I      0 ] [Y_{t-p}  ]   [0  ]

    Parameters
    ----------
    var_result : VARResult
        Estimated VAR model

    Returns
    -------
    np.ndarray
        Shape (n_vars * lags, n_vars * lags) companion matrix F

    Example
    -------
    >>> F = companion_form(var_result)
    >>> # VMA coefficients: Φ_h = F^h [I, 0, ..., 0]'
    """
    n_vars = var_result.n_vars
    lags = var_result.lags

    # Total size of companion matrix
    m = n_vars * lags

    # Initialize companion matrix
    F = np.zeros((m, m))

    # Fill in lag coefficient matrices in first block row
    for lag in range(1, lags + 1):
        A_lag = var_result.get_lag_matrix(lag)
        col_start = (lag - 1) * n_vars
        col_end = lag * n_vars
        F[:n_vars, col_start:col_end] = A_lag

    # Fill in identity blocks on sub-diagonal
    if lags > 1:
        F[n_vars:, : (lags - 1) * n_vars] = np.eye((lags - 1) * n_vars)

    return F


def vma_coefficients(
    var_result: VARResult,
    horizons: int,
) -> np.ndarray:
    """
    Compute VMA (Vector Moving Average) coefficients.

    Y_t = Σ_{h=0}^{∞} Φ_h u_{t-h}

    where Φ_0 = I and Φ_h are the impulse response coefficients
    to reduced-form shocks.

    Parameters
    ----------
    var_result : VARResult
        Estimated VAR model
    horizons : int
        Maximum horizon to compute

    Returns
    -------
    np.ndarray
        Shape (n_vars, n_vars, horizons + 1) VMA coefficient matrices.
        Φ[i, j, h] = response of var i to shock in var j at horizon h.
    """
    n_vars = var_result.n_vars
    lags = var_result.lags

    # Initialize output
    Phi = np.zeros((n_vars, n_vars, horizons + 1))
    Phi[:, :, 0] = np.eye(n_vars)  # Φ_0 = I

    if lags == 0:
        return Phi

    # Use companion form for efficient computation
    F = companion_form(var_result)
    m = F.shape[0]

    # Selector: J = [I_k, 0, ..., 0] picks first k rows
    J = np.zeros((n_vars, m))
    J[:n_vars, :n_vars] = np.eye(n_vars)

    # Φ_h = J F^h J'
    F_power = np.eye(m)
    for h in range(1, horizons + 1):
        F_power = F_power @ F
        Phi[:, :, h] = J @ F_power @ J.T

    return Phi


def structural_vma_coefficients(
    svar_result: SVARResult,
    horizons: int,
) -> np.ndarray:
    """
    Compute structural VMA coefficients (orthogonalized IRF).

    Ψ_h = Φ_h B₀⁻¹

    where Φ_h are reduced-form VMA coefficients.

    Parameters
    ----------
    svar_result : SVARResult
        Structural VAR result
    horizons : int
        Maximum horizon

    Returns
    -------
    np.ndarray
        Shape (n_vars, n_vars, horizons + 1) structural VMA coefficients.
        Ψ[i, j, h] = response of var i to structural shock j at horizon h.
    """
    Phi = vma_coefficients(svar_result.var_result, horizons)
    B0_inv = svar_result.B0_inv

    # Ψ_h = Φ_h B₀⁻¹
    Psi = np.zeros_like(Phi)
    for h in range(horizons + 1):
        Psi[:, :, h] = Phi[:, :, h] @ B0_inv

    return Psi


def check_stability(var_result: VARResult) -> Tuple[bool, np.ndarray]:
    """
    Check VAR stability (stationarity).

    VAR is stable if all eigenvalues of companion matrix are inside unit circle.

    Parameters
    ----------
    var_result : VARResult
        Estimated VAR model

    Returns
    -------
    is_stable : bool
        True if all eigenvalues have modulus < 1
    eigenvalues : np.ndarray
        Eigenvalues of companion matrix
    """
    F = companion_form(var_result)
    eigenvalues = linalg.eigvals(F)
    moduli = np.abs(eigenvalues)
    is_stable = np.all(moduli < 1.0)
    return is_stable, eigenvalues


def long_run_impact_matrix(svar_result: SVARResult) -> np.ndarray:
    """
    Compute long-run impact matrix.

    The long-run impact of structural shocks:
        Ξ = (I - A₁ - ... - Aₚ)⁻¹ B₀⁻¹

    Parameters
    ----------
    svar_result : SVARResult
        Structural VAR result

    Returns
    -------
    np.ndarray
        Shape (n_vars, n_vars) long-run impact matrix
    """
    var_result = svar_result.var_result
    n_vars = var_result.n_vars

    # Sum of lag coefficient matrices
    A_sum = np.zeros((n_vars, n_vars))
    for lag in range(1, var_result.lags + 1):
        A_sum += var_result.get_lag_matrix(lag)

    # (I - A₁ - ... - Aₚ)⁻¹
    try:
        long_run_mult = linalg.inv(np.eye(n_vars) - A_sum)
    except linalg.LinAlgError:
        raise ValueError(
            "Cannot compute long-run impact: I - A_sum is singular. "
            "This typically indicates a unit root."
        )

    return long_run_mult @ svar_result.B0_inv


def verify_identification(
    sigma: np.ndarray,
    B0_inv: np.ndarray,
    tol: float = 1e-8,
) -> Tuple[bool, float]:
    """
    Verify SVAR identification by checking Σ_u = B₀⁻¹ (B₀⁻¹)'.

    Parameters
    ----------
    sigma : np.ndarray
        Reduced-form covariance matrix
    B0_inv : np.ndarray
        Impact matrix
    tol : float
        Tolerance for comparison

    Returns
    -------
    is_valid : bool
        True if identification is valid
    max_error : float
        Maximum element-wise error
    """
    reconstructed = B0_inv @ B0_inv.T
    error = np.abs(reconstructed - sigma)
    max_error = np.max(error)
    is_valid = max_error < tol

    return is_valid, max_error
