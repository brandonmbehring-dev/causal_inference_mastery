"""
Shared validation utilities for causal inference estimators.

This module consolidates duplicate validation logic across 9+ modules,
providing a single source of truth for input validation. Extracted from:
- observational/propensity.py
- psm/matching.py
- iv/two_stage_least_squares.py
- did/did_estimator.py
- did/callaway_santanna.py
- rdd/sharp_rdd.py
- rdd/fuzzy_rdd.py
- iv/diagnostics.py
- did/sun_abraham.py

Core principle: NEVER fail silently. All validation errors provide diagnostic info.
"""

from typing import Optional
import numpy as np


def validate_arrays_same_length(**arrays: np.ndarray) -> None:
    """
    Validate that all arrays have the same length.

    Parameters
    ----------
    **arrays : dict of str -> np.ndarray
        Named arrays to validate (e.g., Y=y_array, D=d_array).

    Raises
    ------
    ValueError
        If arrays have different lengths.

    Examples
    --------
    >>> Y = np.array([1, 2, 3])
    >>> D = np.array([0, 1, 0])
    >>> validate_arrays_same_length(Y=Y, D=D)  # OK

    >>> X = np.array([1, 2])
    >>> validate_arrays_same_length(Y=Y, X=X)  # Raises ValueError
    """
    if not arrays:
        return

    lengths = {name: len(arr) for name, arr in arrays.items()}
    first_name, first_length = next(iter(lengths.items()))

    mismatches = [(name, length) for name, length in lengths.items() if length != first_length]

    if mismatches:
        mismatch_str = ", ".join([f"{name}={length}" for name, length in mismatches])
        raise ValueError(
            f"All inputs must have same length. "
            f"Expected length {first_length} (from {first_name}), "
            f"but got: {mismatch_str}"
        )


def validate_finite(array: np.ndarray, name: str) -> None:
    """
    Validate that array contains only finite values (no NaN or Inf).

    Parameters
    ----------
    array : np.ndarray
        Array to validate.
    name : str
        Name of array for error messages.

    Raises
    ------
    ValueError
        If array contains NaN or Inf values.

    Examples
    --------
    >>> Y = np.array([1.0, 2.0, 3.0])
    >>> validate_finite(Y, "Y")  # OK

    >>> Y_bad = np.array([1.0, np.nan, 3.0])
    >>> validate_finite(Y_bad, "Y")  # Raises ValueError
    """
    if not np.all(np.isfinite(array)):
        n_nan = np.sum(np.isnan(array))
        n_inf = np.sum(np.isinf(array))
        raise ValueError(
            f"{name} contains non-finite values. "
            f"Found {n_nan} NaN and {n_inf} Inf values. "
            f"Remove or impute missing data before estimation."
        )


def validate_binary(array: np.ndarray, name: str) -> None:
    """
    Validate that array is binary (contains only 0 and 1).

    Parameters
    ----------
    array : np.ndarray
        Array to validate.
    name : str
        Name of array for error messages.

    Raises
    ------
    ValueError
        If array contains values other than 0 or 1.

    Examples
    --------
    >>> D = np.array([0, 1, 1, 0])
    >>> validate_binary(D, "D")  # OK

    >>> D_bad = np.array([0, 1, 2])
    >>> validate_binary(D_bad, "D")  # Raises ValueError
    """
    unique_vals = np.unique(array)
    if not np.array_equal(unique_vals, [0, 1]) and not (
        len(unique_vals) == 1 and unique_vals[0] in [0, 1]
    ):
        raise ValueError(
            f"{name} must be binary (0 or 1). "
            f"Got unique values: {unique_vals}"
        )


def validate_not_empty(array: np.ndarray, name: str) -> None:
    """
    Validate that array is not empty.

    Parameters
    ----------
    array : np.ndarray
        Array to validate.
    name : str
        Name of array for error messages.

    Raises
    ------
    ValueError
        If array is empty (length 0).

    Examples
    --------
    >>> Y = np.array([1, 2, 3])
    >>> validate_not_empty(Y, "Y")  # OK

    >>> Y_empty = np.array([])
    >>> validate_not_empty(Y_empty, "Y")  # Raises ValueError
    """
    if len(array) == 0:
        raise ValueError(f"{name} cannot be empty (length 0)")


def validate_has_variation(array: np.ndarray, name: str, axis: Optional[int] = None) -> None:
    """
    Validate that array has variation (not all constant).

    Parameters
    ----------
    array : np.ndarray
        Array to validate.
    name : str
        Name of array for error messages.
    axis : int, optional
        Axis along which to check variance (for 2D arrays).
        If None, checks overall variance.

    Raises
    ------
    ValueError
        If array has no variation (variance = 0).

    Examples
    --------
    >>> Y = np.array([1, 2, 3])
    >>> validate_has_variation(Y, "Y")  # OK

    >>> Y_constant = np.array([1, 1, 1])
    >>> validate_has_variation(Y_constant, "Y")  # Raises ValueError
    """
    if axis is None:
        if np.var(array) == 0:
            raise ValueError(
                f"{name} has no variation (all values are {array.flat[0]}). "
                f"Cannot estimate effects without variation."
            )
    else:
        variances = np.var(array, axis=axis)
        if np.any(variances == 0):
            zero_var_cols = np.where(variances == 0)[0]
            raise ValueError(
                f"{name} has constant columns (no variation): {zero_var_cols}. "
                f"Remove constant columns before estimation."
            )


def validate_in_range(
    array: np.ndarray, name: str, min_val: float, max_val: float, inclusive: bool = True
) -> None:
    """
    Validate that array values are in specified range.

    Parameters
    ----------
    array : np.ndarray
        Array to validate.
    name : str
        Name of array for error messages.
    min_val : float
        Minimum allowed value.
    max_val : float
        Maximum allowed value.
    inclusive : bool, default=True
        If True, uses [min_val, max_val]. If False, uses (min_val, max_val).

    Raises
    ------
    ValueError
        If array contains values outside the specified range.

    Examples
    --------
    >>> propensity = np.array([0.2, 0.5, 0.8])
    >>> validate_in_range(propensity, "propensity", 0.0, 1.0)  # OK

    >>> propensity_bad = np.array([-0.1, 0.5, 1.2])
    >>> validate_in_range(propensity_bad, "propensity", 0.0, 1.0)  # Raises
    """
    if inclusive:
        if np.any((array < min_val) | (array > max_val)):
            raise ValueError(
                f"{name} must be in [{min_val}, {max_val}]. "
                f"Got: min={np.min(array)}, max={np.max(array)}"
            )
    else:
        if np.any((array <= min_val) | (array >= max_val)):
            raise ValueError(
                f"{name} must be in ({min_val}, {max_val}). "
                f"Got: min={np.min(array)}, max={np.max(array)}"
            )


def validate_treatment_outcome(Y: np.ndarray, D: np.ndarray) -> None:
    """
    Validate treatment and outcome arrays for causal inference.

    Performs comprehensive validation:
    - Same length
    - No NaN/Inf
    - Not empty
    - Treatment is binary
    - Both treated and control units present
    - Outcome has variation

    Parameters
    ----------
    Y : np.ndarray
        Outcome variable.
    D : np.ndarray
        Treatment indicator (binary: 0=control, 1=treated).

    Raises
    ------
    ValueError
        If any validation fails.

    Examples
    --------
    >>> Y = np.array([1.0, 2.0, 3.0, 4.0])
    >>> D = np.array([0, 0, 1, 1])
    >>> validate_treatment_outcome(Y, D)  # OK
    """
    # Convert to arrays
    Y = np.asarray(Y)
    D = np.asarray(D)

    # Length validation
    validate_arrays_same_length(Y=Y, D=D)

    # Empty check
    validate_not_empty(Y, "Y")

    # Finite check
    validate_finite(Y, "Y")
    validate_finite(D, "D")

    # Binary treatment
    validate_binary(D, "D")

    # Both groups present
    n_treated = np.sum(D == 1)
    n_control = np.sum(D == 0)
    if n_treated == 0:
        raise ValueError("No treated units found (all D=0). Need both treated and control.")
    if n_control == 0:
        raise ValueError("No control units found (all D=1). Need both treated and control.")

    # Outcome variation
    validate_has_variation(Y, "Y")


def validate_iv_inputs(
    Y: np.ndarray, D: np.ndarray, Z: np.ndarray, X: Optional[np.ndarray] = None
) -> None:
    """
    Validate inputs for instrumental variables estimation.

    Performs comprehensive validation for 2SLS and related IV methods:
    - Same length for all inputs
    - No NaN/Inf
    - Not empty
    - All variables have variation
    - Order condition: # instruments >= # endogenous variables

    Parameters
    ----------
    Y : np.ndarray
        Outcome variable (n,).
    D : np.ndarray
        Treatment/endogenous variable (n,) or (n, p).
    Z : np.ndarray
        Instruments (n,) or (n, q).
    X : np.ndarray, optional
        Exogenous controls (n, k).

    Raises
    ------
    ValueError
        If any validation fails.

    Examples
    --------
    >>> Y = np.array([1.0, 2.0, 3.0, 4.0])
    >>> D = np.array([0.5, 1.0, 1.5, 2.0])
    >>> Z = np.array([0, 0, 1, 1])
    >>> validate_iv_inputs(Y, D, Z)  # OK
    """
    # Convert to arrays
    Y = np.asarray(Y).flatten()
    D = np.asarray(D)
    Z = np.asarray(Z)
    if X is not None:
        X = np.asarray(X)

    # Ensure D and Z are 2D
    if D.ndim == 1:
        D = D.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    # Length validation
    if X is not None:
        validate_arrays_same_length(Y=Y, D=D[:, 0], Z=Z[:, 0], X=X[:, 0])
    else:
        validate_arrays_same_length(Y=Y, D=D[:, 0], Z=Z[:, 0])

    # Empty check
    validate_not_empty(Y, "Y")

    # Finite checks
    validate_finite(Y, "Y")
    validate_finite(D, "D")
    validate_finite(Z, "Z")
    if X is not None:
        validate_finite(X, "X")

    # Variation checks
    validate_has_variation(Y, "Y")
    validate_has_variation(D, "D", axis=0)
    validate_has_variation(Z, "Z", axis=0)

    # Order condition: # instruments >= # endogenous
    p = D.shape[1]  # Number of endogenous variables
    q = Z.shape[1]  # Number of instruments
    if q < p:
        raise ValueError(
            f"Order condition fails: Need at least as many instruments as endogenous variables. "
            f"Got {q} instruments and {p} endogenous variables."
        )


def validate_did_inputs(
    outcomes: np.ndarray, treatment: np.ndarray, post: np.ndarray, unit_id: np.ndarray
) -> None:
    """
    Validate inputs for difference-in-differences estimation.

    Performs comprehensive validation for DiD:
    - Same length for all inputs
    - No NaN/Inf
    - Not empty
    - Treatment and post are binary
    - Both treated/control and pre/post periods present

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable.
    treatment : np.ndarray
        Treatment indicator (binary: 0=control, 1=treated).
    post : np.ndarray
        Post-treatment indicator (binary: 0=pre, 1=post).
    unit_id : np.ndarray
        Unit identifiers.

    Raises
    ------
    ValueError
        If any validation fails.

    Examples
    --------
    >>> outcomes = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> treatment = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    >>> post = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    >>> unit_id = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    >>> validate_did_inputs(outcomes, treatment, post, unit_id)  # OK
    """
    # Convert to arrays
    outcomes = np.asarray(outcomes)
    treatment = np.asarray(treatment)
    post = np.asarray(post)
    unit_id = np.asarray(unit_id)

    # Length validation
    validate_arrays_same_length(outcomes=outcomes, treatment=treatment, post=post, unit_id=unit_id)

    # Empty check
    validate_not_empty(outcomes, "outcomes")

    # Finite checks
    validate_finite(outcomes, "outcomes")
    validate_finite(treatment, "treatment")
    validate_finite(post, "post")

    # Binary checks
    validate_binary(treatment, "treatment")
    validate_binary(post, "post")

    # Check both groups present
    n_treated = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)
    if n_treated == 0:
        raise ValueError("No treated units found (all treatment=0)")
    if n_control == 0:
        raise ValueError("No control units found (all treatment=1)")

    # Check both periods present
    n_pre = np.sum(post == 0)
    n_post = np.sum(post == 1)
    if n_pre == 0:
        raise ValueError("No pre-treatment periods found (all post=1)")
    if n_post == 0:
        raise ValueError("No post-treatment periods found (all post=0)")
