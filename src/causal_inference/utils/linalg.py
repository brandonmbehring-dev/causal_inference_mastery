"""
Linear algebra utilities with standardized error handling.

This module provides safe wrappers around numpy.linalg operations that:
1. NEVER fail silently (core principle from CLAUDE.md)
2. Provide diagnostic information when operations fail
3. Check for numerical issues (singularity, conditioning)
4. Offer graceful degradation where appropriate

Replaces bare np.linalg calls across 9+ modules with standardized error handling.

Usage Examples
--------------
**Before (bare numpy.linalg):**
```python
# Risky: Fails silently or with cryptic errors
XtX_inv = np.linalg.inv(X.T @ X)  # What if X'X is singular?
beta = np.linalg.lstsq(X, y, rcond=None)[0]  # No warning if rank-deficient
```

**After (standardized error handling):**
```python
from src.causal_inference.utils.linalg import safe_inv, safe_lstsq

# Safe: Provides diagnostic errors and warnings
XtX_inv = safe_inv(X.T @ X, name="design_matrix")  # Clear error if singular
beta, *_ = safe_lstsq(X, y, name="OLS")  # Warns if rank-deficient
```

Migration Guide
---------------
Current usage across codebase:
- 16 uses of np.linalg.inv (9 files)
- 3 uses of np.linalg.lstsq
- 2 uses of np.linalg.eigvalsh
- 1 use of np.linalg.solve

To migrate existing code:
1. Import: `from src.causal_inference.utils.linalg import safe_inv`
2. Replace: `np.linalg.inv(A)` → `safe_inv(A, name="descriptive_name")`
3. Add meaningful names for better error messages
4. Remove any existing try/except LinAlgError blocks (handled internally)
"""

from typing import Optional, Tuple
import warnings
import numpy as np


def safe_inv(
    matrix: np.ndarray,
    name: str = "matrix",
    check_condition: bool = True,
    condition_threshold: float = 1e10,
) -> np.ndarray:
    """
    Safely invert a matrix with comprehensive error handling.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to invert (n, n).
    name : str, default='matrix'
        Name of matrix for error messages.
    check_condition : bool, default=True
        Whether to check condition number and warn if ill-conditioned.
    condition_threshold : float, default=1e10
        Threshold for ill-conditioning warning.

    Returns
    -------
    matrix_inv : np.ndarray
        Inverted matrix (n, n).

    Raises
    ------
    ValueError
        If matrix is not square.
    np.linalg.LinAlgError
        If matrix is singular or near-singular (with diagnostic info).

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> X_inv = safe_inv(X, name="design_matrix")

    >>> # Singular matrix raises informative error
    >>> singular = np.array([[1, 1], [1, 1]])
    >>> safe_inv(singular, name="XtX")  # Raises LinAlgError with diagnostics

    Notes
    -----
    Matrix inversion can fail due to:
    1. **Singularity**: Determinant = 0 (perfect collinearity)
    2. **Near-singularity**: Very small determinant (numerical instability)
    3. **Ill-conditioning**: Large condition number (amplifies numerical errors)

    This function checks for these issues and provides diagnostic information.
    """
    # Validate square matrix
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"{name} must be square for inversion. "
            f"Got shape {matrix.shape}. "
            f"Check that you're inverting the correct matrix (e.g., X'X not X)."
        )

    n = matrix.shape[0]

    # Check condition number before attempting inversion
    if check_condition:
        try:
            cond = np.linalg.cond(matrix)
            if cond > condition_threshold:
                warnings.warn(
                    f"{name} is ill-conditioned (condition number={cond:.2e}). "
                    f"Inversion may be numerically unstable. "
                    f"Consider: (1) Regularization, (2) Removing collinear variables, "
                    f"(3) Scaling features.",
                    UserWarning,
                )
        except np.linalg.LinAlgError:
            # Condition number computation can fail for singular matrices
            pass

    # Attempt inversion with diagnostic error handling
    try:
        matrix_inv = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as e:
        # Compute diagnostics for informative error
        det = np.linalg.det(matrix)
        eigenvalues = np.linalg.eigvalsh(matrix)
        min_eigenvalue = eigenvalues.min()
        max_eigenvalue = eigenvalues.max()

        raise np.linalg.LinAlgError(
            f"Failed to invert {name} (shape {matrix.shape}). "
            f"Matrix is singular or near-singular. "
            f"Diagnostics: "
            f"det={det:.2e}, "
            f"min_eigenvalue={min_eigenvalue:.2e}, "
            f"max_eigenvalue={max_eigenvalue:.2e}. "
            f"This indicates perfect or near-perfect collinearity in your data. "
            f"Solutions: (1) Remove collinear variables, (2) Add regularization, "
            f"(3) Increase sample size relative to number of variables. "
            f"Original error: {str(e)}"
        ) from e

    return matrix_inv


def safe_solve(
    A: np.ndarray,
    b: np.ndarray,
    name: str = "system",
    check_condition: bool = True,
) -> np.ndarray:
    """
    Safely solve linear system Ax = b with error handling.

    More numerically stable than explicit inversion for solving linear systems.

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix (n, n).
    b : np.ndarray
        Right-hand side vector(s) (n,) or (n, k).
    name : str, default='system'
        Name for error messages.
    check_condition : bool, default=True
        Whether to check condition number.

    Returns
    -------
    x : np.ndarray
        Solution vector(s) (n,) or (n, k).

    Raises
    ------
    ValueError
        If shapes are incompatible.
    np.linalg.LinAlgError
        If system is singular (with diagnostics).

    Examples
    --------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> b = np.array([5, 6])
    >>> x = safe_solve(A, b, name="OLS_system")

    Notes
    -----
    Prefer `safe_solve(A, b)` over `safe_inv(A) @ b` for numerical stability.
    """
    # Validate shapes
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name}: Coefficient matrix A must be square. Got shape {A.shape}.")

    if len(b) != A.shape[0]:
        raise ValueError(
            f"{name}: Incompatible shapes A={A.shape}, b={b.shape}. "
            f"Number of equations ({A.shape[0]}) must match length of b ({len(b)})."
        )

    # Check condition number
    if check_condition:
        try:
            cond = np.linalg.cond(A)
            if cond > 1e10:
                warnings.warn(
                    f"{name}: Coefficient matrix is ill-conditioned (cond={cond:.2e}). "
                    f"Solution may be numerically unstable.",
                    UserWarning,
                )
        except np.linalg.LinAlgError:
            pass

    # Solve with error handling
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        det = np.linalg.det(A)
        raise np.linalg.LinAlgError(
            f"Failed to solve linear system '{name}' (A: {A.shape}, b: {b.shape}). "
            f"Matrix A is singular (det={det:.2e}). "
            f"This indicates collinearity in the coefficient matrix. "
            f"Original error: {str(e)}"
        ) from e

    return x


def safe_lstsq(
    A: np.ndarray,
    b: np.ndarray,
    name: str = "regression",
    rcond: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Safely compute least-squares solution with error handling.

    Solves min ||Ax - b||_2 even when A is not full rank.

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix (m, n).
    b : np.ndarray
        Observation vector (m,) or (m, k).
    name : str, default='regression'
        Name for error messages.
    rcond : float, optional
        Cutoff for small singular values. Default: machine precision * max(m, n).

    Returns
    -------
    x : np.ndarray
        Least-squares solution (n,) or (n, k).
    residuals : np.ndarray
        Sum of squared residuals.
    rank : int
        Effective rank of A.
    s : np.ndarray
        Singular values of A.

    Raises
    ------
    ValueError
        If shapes are incompatible.

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randn(100)
    >>> beta, residuals, rank, s = safe_lstsq(X, y, name="OLS")

    Notes
    -----
    Unlike `safe_solve`, this works for over-determined systems (m > n) and
    rank-deficient matrices. Warns if matrix is rank-deficient.
    """
    # Validate shapes
    if A.ndim != 2:
        raise ValueError(f"{name}: Matrix A must be 2D. Got shape {A.shape}.")

    m, n = A.shape
    if len(b) != m:
        raise ValueError(
            f"{name}: Incompatible shapes A={A.shape}, b={b.shape}. "
            f"Number of observations ({m}) must match length of b ({len(b)})."
        )

    # Compute least squares
    result = np.linalg.lstsq(A, b, rcond=rcond)
    x, residuals, rank, s = result

    # Warn if rank-deficient
    if rank < min(m, n):
        warnings.warn(
            f"{name}: Matrix is rank-deficient (rank={rank} < min(m={m}, n={n})={min(m, n)}). "
            f"This indicates perfect collinearity. "
            f"Smallest singular values: {s[-3:]!r}. "
            f"Solution may not be unique.",
            UserWarning,
        )

    return x, residuals, rank, s


def safe_eigvalsh(
    matrix: np.ndarray,
    name: str = "matrix",
    check_positive_definite: bool = False,
) -> np.ndarray:
    """
    Safely compute eigenvalues of symmetric/Hermitian matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric matrix (n, n).
    name : str, default='matrix'
        Name for error messages.
    check_positive_definite : bool, default=False
        Whether to check and warn if matrix is not positive definite.

    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues in ascending order (n,).

    Raises
    ------
    ValueError
        If matrix is not square.
    np.linalg.LinAlgError
        If eigenvalue computation fails.

    Examples
    --------
    >>> A = np.array([[2, 1], [1, 2]])
    >>> eigvals = safe_eigvalsh(A, name="covariance_matrix")

    Notes
    -----
    Assumes matrix is symmetric. For general matrices, use np.linalg.eig.
    """
    # Validate
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"{name} must be square for eigenvalue computation. Got shape {matrix.shape}."
        )

    # Compute eigenvalues
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"Failed to compute eigenvalues of {name} (shape {matrix.shape}). "
            f"Original error: {str(e)}"
        ) from e

    # Check positive definiteness if requested
    if check_positive_definite:
        min_eigenvalue = eigenvalues.min()
        if min_eigenvalue <= 0:
            warnings.warn(
                f"{name} is not positive definite (min_eigenvalue={min_eigenvalue:.2e} ≤ 0). "
                f"This may indicate numerical issues or a singular covariance matrix.",
                UserWarning,
            )

    return eigenvalues
