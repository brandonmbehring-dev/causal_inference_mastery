"""
Sign Restrictions for SVAR Identification (Uhlig 2005).

Session 161: Set-identified SVAR using sign restrictions on impulse responses.

Unlike point-identified methods (Cholesky, Long-Run), sign restrictions yield
a **set of valid structural matrices**. The algorithm samples random rotations
of the Cholesky factor and keeps those satisfying the sign constraints.

Algorithm (Uhlig 2005):
1. Start from reduced-form VAR: Σ_u = PP' (Cholesky decomposition)
2. For i = 1 to N_draws:
   a. Generate random orthogonal Q via Givens rotations
   b. Candidate impact matrix: B₀⁻¹ = P·Q
   c. Compute IRF for candidate
   d. If IRF satisfies all sign constraints → Accept
3. Report identified set bounds (percentiles across accepted draws)

References
----------
Uhlig, H. (2005). "What Are the Effects of Monetary Policy on Output?
Results from an Agnostic Identification Procedure." Journal of Monetary
Economics 52(2): 381-419.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Tuple, Union
import warnings
import numpy as np
from scipy import linalg

from causal_inference.timeseries.types import VARResult
from causal_inference.timeseries.svar_types import (
    SVARResult,
    IdentificationMethod,
    IRFResult,
)
from causal_inference.timeseries.svar import cholesky_svar, vma_coefficients


@dataclass
class SignRestrictionConstraint:
    """
    Single sign constraint on an impulse response.

    Specifies that the response of variable `response_idx` to a unit shock
    in variable `shock_idx` at horizon `horizon` must be positive or negative.

    Attributes
    ----------
    shock_idx : int
        Index of the structural shock (0-indexed)
    response_idx : int
        Index of the response variable (0-indexed)
    horizon : int
        Horizon at which constraint applies (0 = impact)
    sign : {1, -1}
        Required sign: +1 for positive, -1 for negative

    Example
    -------
    >>> # Money shock must increase output at impact
    >>> constraint = SignRestrictionConstraint(
    ...     shock_idx=0, response_idx=1, horizon=0, sign=1
    ... )
    """

    shock_idx: int
    response_idx: int
    horizon: int
    sign: Literal[1, -1]

    def __post_init__(self):
        if self.sign not in (1, -1):
            raise ValueError(f"sign must be 1 or -1, got {self.sign}")
        if self.horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {self.horizon}")
        if self.shock_idx < 0:
            raise ValueError(f"shock_idx must be >= 0, got {self.shock_idx}")
        if self.response_idx < 0:
            raise ValueError(f"response_idx must be >= 0, got {self.response_idx}")

    def __repr__(self) -> str:
        sign_str = "+" if self.sign > 0 else "-"
        return (
            f"SignRestrictionConstraint(shock={self.shock_idx}, "
            f"response={self.response_idx}, h={self.horizon}, sign={sign_str})"
        )


@dataclass
class SignRestrictionResult:
    """
    Result from sign-restricted SVAR (set-identified).

    Unlike point-identified SVARs, sign restrictions yield a set of valid
    structural matrices. This result contains the full set and summary statistics.

    Attributes
    ----------
    var_result : VARResult
        Underlying reduced-form VAR
    B0_inv : np.ndarray
        Median impact matrix from accepted draws
    B0 : np.ndarray
        Inverse of B0_inv
    structural_shocks : np.ndarray
        Structural shocks using median B0
    identification : IdentificationMethod
        Always SIGN for this result type
    n_restrictions : int
        Number of sign constraints imposed
    is_just_identified : bool
        Always False for sign restrictions
    is_over_identified : bool
        Always True for sign restrictions
    constraints : List[SignRestrictionConstraint]
        Sign constraints that were imposed
    B0_inv_set : List[np.ndarray]
        All accepted impact matrices
    irf_median : np.ndarray
        Median IRF across accepted draws (n_vars, n_vars, horizons+1)
    irf_lower : np.ndarray
        Lower percentile IRF (16th by default)
    irf_upper : np.ndarray
        Upper percentile IRF (84th by default)
    n_draws : int
        Total number of rotation draws attempted
    n_accepted : int
        Number of draws satisfying all constraints
    acceptance_rate : float
        Fraction of draws accepted
    rotation_method : str
        Method used for random rotation generation
    horizons : int
        Maximum IRF horizon computed

    Example
    -------
    >>> result = sign_restriction_svar(var_result, constraints)
    >>> print(f"Acceptance rate: {result.acceptance_rate:.1%}")
    >>> print(f"IRF bounds shape: {result.irf_lower.shape}")
    """

    var_result: VARResult
    B0_inv: np.ndarray
    B0: np.ndarray
    structural_shocks: np.ndarray
    identification: IdentificationMethod = IdentificationMethod.SIGN
    n_restrictions: int = 0
    is_just_identified: bool = False
    is_over_identified: bool = True

    # Sign restriction specific
    constraints: List[SignRestrictionConstraint] = field(default_factory=list)
    B0_inv_set: List[np.ndarray] = field(default_factory=list)

    # IRF bounds (set identification)
    irf_median: np.ndarray = None
    irf_lower: np.ndarray = None
    irf_upper: np.ndarray = None

    # Algorithm diagnostics
    n_draws: int = 0
    n_accepted: int = 0
    acceptance_rate: float = 0.0
    rotation_method: str = "givens"
    horizons: int = 20

    @property
    def n_vars(self) -> int:
        """Number of variables."""
        return self.var_result.n_vars

    @property
    def lags(self) -> int:
        """VAR lag order."""
        return self.var_result.lags

    @property
    def var_names(self) -> List[str]:
        """Variable names."""
        return self.var_result.var_names

    def get_irf_bounds(
        self,
        response_idx: int,
        shock_idx: int,
    ) -> Dict[str, np.ndarray]:
        """
        Get IRF bounds for a specific response-shock pair.

        Returns
        -------
        dict
            Keys: 'median', 'lower', 'upper', 'horizon'
        """
        return {
            "median": self.irf_median[response_idx, shock_idx, :],
            "lower": self.irf_lower[response_idx, shock_idx, :],
            "upper": self.irf_upper[response_idx, shock_idx, :],
            "horizon": np.arange(self.horizons + 1),
        }

    def to_irf_result(self) -> IRFResult:
        """Convert to standard IRFResult for compatibility."""
        return IRFResult(
            irf=self.irf_median,
            irf_lower=self.irf_lower,
            irf_upper=self.irf_upper,
            horizons=self.horizons,
            cumulative=False,
            orthogonalized=True,
            var_names=self.var_names,
            alpha=0.32,  # 16th/84th percentiles
            n_bootstrap=self.n_accepted,
        )

    def __repr__(self) -> str:
        return (
            f"SignRestrictionResult(n_vars={self.n_vars}, "
            f"n_constraints={len(self.constraints)}, "
            f"acceptance={self.acceptance_rate:.1%}, "
            f"n_accepted={self.n_accepted})"
        )


def sign_restriction_svar(
    var_result: VARResult,
    constraints: List[SignRestrictionConstraint],
    horizons: int = 20,
    n_draws: int = 5000,
    rotation_method: str = "givens",
    percentiles: Tuple[float, float] = (16.0, 84.0),
    seed: Optional[int] = None,
    min_acceptance_rate: float = 0.01,
) -> SignRestrictionResult:
    """
    SVAR identification via sign restrictions (Uhlig 2005).

    Generates random orthogonal rotations of the Cholesky factor and keeps
    those satisfying all sign constraints on the implied impulse responses.

    Parameters
    ----------
    var_result : VARResult
        Estimated reduced-form VAR model
    constraints : List[SignRestrictionConstraint]
        Sign constraints on impulse responses
    horizons : int
        Maximum IRF horizon to compute
    n_draws : int
        Number of random rotation draws to attempt
    rotation_method : str
        Method for generating random orthogonal matrices:
        - "givens": Compose Givens rotations (default, uniform on SO(n))
        - "qr": QR decomposition of random matrix (faster, approximate)
    percentiles : Tuple[float, float]
        Lower and upper percentiles for confidence bounds (default: 16, 84)
    seed : int, optional
        Random seed for reproducibility
    min_acceptance_rate : float
        Minimum acceptable acceptance rate (warning if below)

    Returns
    -------
    SignRestrictionResult
        Sign-restricted SVAR results with identified set bounds

    Raises
    ------
    ValueError
        If no valid rotations found, or constraints are invalid

    Example
    -------
    >>> from causal_inference.timeseries import var_estimate
    >>> var_result = var_estimate(data, lags=4)
    >>> constraints = [
    ...     SignRestrictionConstraint(shock_idx=0, response_idx=1, horizon=0, sign=1),
    ...     SignRestrictionConstraint(shock_idx=0, response_idx=2, horizon=0, sign=-1),
    ... ]
    >>> result = sign_restriction_svar(var_result, constraints, seed=42)
    >>> print(f"Acceptance rate: {result.acceptance_rate:.1%}")
    """
    n_vars = var_result.n_vars

    # Validate constraints
    validate_constraints(constraints, n_vars, horizons)

    if len(constraints) == 0:
        warnings.warn(
            "No constraints provided - all rotations will be accepted",
            UserWarning,
        )

    # Initialize random generator
    rng = np.random.default_rng(seed)

    # Get Cholesky starting point
    svar_chol = cholesky_svar(var_result)
    P = svar_chol.B0_inv  # Cholesky factor

    # Compute VMA coefficients for IRF calculation
    Phi = vma_coefficients(var_result, horizons)

    # Storage for accepted draws
    accepted_B0_inv = []
    accepted_irfs = []

    # Main rotation loop
    for draw_idx in range(n_draws):
        # Generate random orthogonal matrix
        if rotation_method == "givens":
            Q = _random_orthogonal_givens(n_vars, rng)
        elif rotation_method == "qr":
            Q = _random_orthogonal_qr(n_vars, rng)
        else:
            raise ValueError(f"rotation_method must be 'givens' or 'qr', got '{rotation_method}'")

        # Candidate impact matrix: B₀⁻¹ = P @ Q
        B0_inv_candidate = P @ Q

        # Compute IRF for this candidate
        irf = _compute_irf_from_impact(Phi, B0_inv_candidate, horizons)

        # Check sign constraints
        if _check_sign_constraints(irf, constraints):
            accepted_B0_inv.append(B0_inv_candidate)
            accepted_irfs.append(irf)

    n_accepted = len(accepted_B0_inv)
    acceptance_rate = n_accepted / n_draws

    # Check acceptance rate
    if n_accepted == 0:
        raise ValueError(
            f"No rotations satisfied sign constraints (0/{n_draws}). "
            f"Check if constraints are feasible or increase n_draws."
        )

    if acceptance_rate < min_acceptance_rate:
        warnings.warn(
            f"Very low acceptance rate ({acceptance_rate:.2%}). "
            f"Constraints may be too restrictive or conflicting.",
            UserWarning,
        )

    # Stack accepted IRFs for percentile computation
    irfs_array = np.array(accepted_irfs)  # (n_accepted, n_vars, n_vars, horizons+1)

    # Compute percentile bounds
    lower_pct, upper_pct = percentiles
    irf_median = np.median(irfs_array, axis=0)
    irf_lower = np.percentile(irfs_array, lower_pct, axis=0)
    irf_upper = np.percentile(irfs_array, upper_pct, axis=0)

    # Use median B0_inv as point estimate
    B0_inv_median = np.median(np.array(accepted_B0_inv), axis=0)
    B0_median = linalg.inv(B0_inv_median)

    # Compute structural shocks using median impact matrix
    residuals = var_result.residuals
    structural_shocks = (B0_median @ residuals.T).T

    return SignRestrictionResult(
        var_result=var_result,
        B0_inv=B0_inv_median,
        B0=B0_median,
        structural_shocks=structural_shocks,
        identification=IdentificationMethod.SIGN,
        n_restrictions=len(constraints),
        is_just_identified=False,
        is_over_identified=True,
        constraints=constraints,
        B0_inv_set=accepted_B0_inv,
        irf_median=irf_median,
        irf_lower=irf_lower,
        irf_upper=irf_upper,
        n_draws=n_draws,
        n_accepted=n_accepted,
        acceptance_rate=acceptance_rate,
        rotation_method=rotation_method,
        horizons=horizons,
    )


def _givens_rotation_matrix(n: int, i: int, j: int, theta: float) -> np.ndarray:
    """
    Create Givens rotation matrix G(i,j,θ).

    Rotates in the (i,j) plane by angle θ.

    G[i,i] = G[j,j] = cos(θ)
    G[i,j] = -sin(θ)
    G[j,i] = sin(θ)
    All other diagonal = 1, off-diagonal = 0

    Parameters
    ----------
    n : int
        Matrix dimension
    i, j : int
        Plane indices (i < j)
    theta : float
        Rotation angle in radians

    Returns
    -------
    np.ndarray
        n×n Givens rotation matrix
    """
    G = np.eye(n)
    c = np.cos(theta)
    s = np.sin(theta)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = -s
    G[j, i] = s
    return G


def _random_orthogonal_givens(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate random orthogonal Q ∈ SO(n) via Givens rotations.

    Composes n(n-1)/2 Givens rotations with random angles uniformly
    distributed on [0, 2π). This produces matrices uniformly distributed
    on the special orthogonal group SO(n).

    Parameters
    ----------
    n : int
        Matrix dimension
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray
        n×n orthogonal matrix with det = +1
    """
    Q = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            theta = rng.uniform(0, 2 * np.pi)
            G = _givens_rotation_matrix(n, i, j, theta)
            Q = Q @ G
    return Q


def _random_orthogonal_qr(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate random orthogonal matrix via QR decomposition.

    Faster than Givens but produces matrices with det = ±1.
    We correct for det = -1 by flipping a column.

    Parameters
    ----------
    n : int
        Matrix dimension
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray
        n×n orthogonal matrix with det = +1
    """
    # Random matrix with standard normal entries
    A = rng.standard_normal((n, n))

    # QR decomposition
    Q, R = linalg.qr(A)

    # Ensure Q has positive diagonal in R (standard form)
    d = np.diag(R)
    Q = Q @ np.diag(np.sign(d))

    # Ensure det(Q) = +1 (special orthogonal group)
    if linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    return Q


def _compute_irf_from_impact(
    Phi: np.ndarray,
    B0_inv: np.ndarray,
    horizons: int,
) -> np.ndarray:
    """
    Compute IRF from VMA coefficients and impact matrix.

    IRF_h = Φ_h @ B₀⁻¹

    Parameters
    ----------
    Phi : np.ndarray
        VMA coefficients, shape (n_vars, n_vars, horizons+1)
    B0_inv : np.ndarray
        Impact matrix, shape (n_vars, n_vars)
    horizons : int
        Maximum horizon

    Returns
    -------
    np.ndarray
        IRF matrix, shape (n_vars, n_vars, horizons+1)
    """
    n_vars = B0_inv.shape[0]
    irf = np.zeros((n_vars, n_vars, horizons + 1))

    for h in range(horizons + 1):
        irf[:, :, h] = Phi[:, :, h] @ B0_inv

    return irf


def _check_sign_constraints(
    irf: np.ndarray,
    constraints: List[SignRestrictionConstraint],
) -> bool:
    """
    Check if IRF satisfies all sign constraints.

    Parameters
    ----------
    irf : np.ndarray
        Impulse response matrix (n_vars, n_vars, horizons+1)
    constraints : List[SignRestrictionConstraint]
        Sign constraints to check

    Returns
    -------
    bool
        True if ALL constraints are satisfied
    """
    for c in constraints:
        val = irf[c.response_idx, c.shock_idx, c.horizon]

        if c.sign > 0 and val <= 0:
            return False
        if c.sign < 0 and val >= 0:
            return False

    return True


def validate_constraints(
    constraints: List[SignRestrictionConstraint],
    n_vars: int,
    horizons: int,
) -> None:
    """
    Validate constraint specification.

    Checks that all constraints reference valid variable indices and horizons.

    Parameters
    ----------
    constraints : List[SignRestrictionConstraint]
        Constraints to validate
    n_vars : int
        Number of variables in the VAR
    horizons : int
        Maximum horizon for IRF computation

    Raises
    ------
    ValueError
        If any constraint references invalid indices or horizons
    """
    for i, c in enumerate(constraints):
        if c.shock_idx >= n_vars:
            raise ValueError(f"Constraint {i}: shock_idx {c.shock_idx} >= n_vars {n_vars}")
        if c.response_idx >= n_vars:
            raise ValueError(f"Constraint {i}: response_idx {c.response_idx} >= n_vars {n_vars}")
        if c.horizon > horizons:
            raise ValueError(f"Constraint {i}: horizon {c.horizon} > max horizons {horizons}")


def create_monetary_policy_constraints(
    money_shock_idx: int = 0,
    output_idx: int = 1,
    price_idx: int = 2,
    interest_idx: Optional[int] = None,
    max_horizon: int = 4,
) -> List[SignRestrictionConstraint]:
    """
    Create standard monetary policy sign restrictions.

    Follows Uhlig (2005) baseline specification:
    - Contractionary money shock increases interest rate
    - Contractionary money shock decreases output
    - Contractionary money shock decreases prices

    Parameters
    ----------
    money_shock_idx : int
        Index of the monetary policy shock
    output_idx : int
        Index of the output variable
    price_idx : int
        Index of the price level variable
    interest_idx : int, optional
        Index of the interest rate variable
    max_horizon : int
        Maximum horizon for constraints

    Returns
    -------
    List[SignRestrictionConstraint]
        List of sign constraints for monetary policy identification

    Example
    -------
    >>> constraints = create_monetary_policy_constraints()
    >>> result = sign_restriction_svar(var_result, constraints)
    """
    constraints = []

    # Output declines after contractionary shock
    for h in range(max_horizon + 1):
        constraints.append(
            SignRestrictionConstraint(
                shock_idx=money_shock_idx,
                response_idx=output_idx,
                horizon=h,
                sign=-1,
            )
        )

    # Prices decline after contractionary shock
    for h in range(max_horizon + 1):
        constraints.append(
            SignRestrictionConstraint(
                shock_idx=money_shock_idx,
                response_idx=price_idx,
                horizon=h,
                sign=-1,
            )
        )

    # Interest rate rises after contractionary shock
    if interest_idx is not None:
        for h in range(max_horizon + 1):
            constraints.append(
                SignRestrictionConstraint(
                    shock_idx=money_shock_idx,
                    response_idx=interest_idx,
                    horizon=h,
                    sign=1,
                )
            )

    return constraints


def check_cholesky_in_set(
    var_result: VARResult,
    constraints: List[SignRestrictionConstraint],
    horizons: int = 20,
) -> bool:
    """
    Check if Cholesky identification satisfies sign constraints.

    Useful diagnostic: if Cholesky satisfies constraints, it should
    always be in the identified set.

    Parameters
    ----------
    var_result : VARResult
        VAR estimation result
    constraints : List[SignRestrictionConstraint]
        Sign constraints
    horizons : int
        Maximum horizon

    Returns
    -------
    bool
        True if Cholesky identification satisfies all constraints
    """
    svar_chol = cholesky_svar(var_result)
    Phi = vma_coefficients(var_result, horizons)
    irf = _compute_irf_from_impact(Phi, svar_chol.B0_inv, horizons)

    return _check_sign_constraints(irf, constraints)
