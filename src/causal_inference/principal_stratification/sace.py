"""Survivor Average Causal Effect (SACE) estimation.

SACE addresses the problem of "truncation by death" where outcomes are
undefined for units who don't survive (or are otherwise censored).

Key Estimand
------------
SACE = E[Y(1) - Y(0) | S(0)=1, S(1)=1]

The treatment effect for units who would survive under BOTH treatment conditions
("always-survivors" principal stratum).

The Problem
-----------
- We observe Y only if S = 1 (survival)
- Some units have S(1) = 1, S(0) = 0 (protected by treatment)
- Some units have S(1) = 0, S(0) = 1 (harmed by treatment)
- Always-survivor stratum (S(0)=S(1)=1) is not directly identifiable

Methods
-------
sace_bounds : Compute bounds on SACE under various assumptions
sace_sensitivity : Sensitivity analysis varying selection assumptions

References
----------
- Zhang, J. L., & Rubin, D. B. (2003). Estimation of Causal Effects via Principal
  Stratification When Some Outcomes Are Truncated by Death. Journal of Educational
  and Behavioral Statistics, 28(4), 353-368.
- Frangakis, C. E., & Rubin, D. B. (2002). Principal Stratification in Causal
  Inference. Biometrics, 58(1), 21-29.
- Lee, D. S. (2009). Training, Wages, and Sample Selection: Estimating Sharp Bounds
  on Treatment Effects. Review of Economic Studies, 76(3), 1071-1102.
"""

import warnings
from typing import Optional, Tuple, Dict, Literal

import numpy as np
from numpy.typing import ArrayLike

from .types import SACEResult


def _validate_sace_inputs(
    outcome: ArrayLike,
    treatment: ArrayLike,
    survival: ArrayLike,
    instrument: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Validate inputs for SACE estimation.

    Parameters
    ----------
    outcome : array-like
        Outcome variable Y (may have NaN for non-survivors).
    treatment : array-like
        Treatment indicator D (binary).
    survival : array-like
        Survival indicator S (binary: 1 if outcome observed).
    instrument : array-like, optional
        Instrument Z (binary) for randomization.

    Returns
    -------
    Y, D, S, Z : tuple of arrays
        Validated arrays (Z may be None).
    """
    Y = np.asarray(outcome, dtype=float)
    D = np.asarray(treatment, dtype=float)
    S = np.asarray(survival, dtype=float)

    n = len(Y)
    if len(D) != n or len(S) != n:
        raise ValueError(
            f"Length mismatch: outcome ({len(Y)}), treatment ({len(D)}), "
            f"survival ({len(S)}) must have same length."
        )

    # Check binary treatment
    D_vals = np.unique(D[~np.isnan(D)])
    if not np.all(np.isin(D_vals, [0, 1])):
        raise ValueError(
            f"Treatment must be binary (0 or 1), got unique values: {D_vals}"
        )

    # Check binary survival
    S_vals = np.unique(S[~np.isnan(S)])
    if not np.all(np.isin(S_vals, [0, 1])):
        raise ValueError(
            f"Survival must be binary (0 or 1), got unique values: {S_vals}"
        )

    Z = None
    if instrument is not None:
        Z = np.asarray(instrument, dtype=float)
        if len(Z) != n:
            raise ValueError(
                f"Instrument length ({len(Z)}) must match outcome length ({n})."
            )
        Z_vals = np.unique(Z[~np.isnan(Z)])
        if not np.all(np.isin(Z_vals, [0, 1])):
            raise ValueError(
                f"Instrument must be binary (0 or 1), got unique values: {Z_vals}"
            )

    return Y, D, S, Z


def _compute_survival_proportions(
    D: np.ndarray, S: np.ndarray, Z: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute survival proportions by treatment/assignment status.

    Returns
    -------
    dict with keys:
        'p_S1_D1': P(S=1 | D=1) - survival rate under treatment
        'p_S1_D0': P(S=1 | D=0) - survival rate under control
        'p_S1_Z1': P(S=1 | Z=1) - survival rate when assigned to treatment (if Z provided)
        'p_S1_Z0': P(S=1 | Z=0) - survival rate when assigned to control (if Z provided)
    """
    result = {}

    # By observed treatment
    D1_mask = D == 1
    D0_mask = D == 0
    result["p_S1_D1"] = np.mean(S[D1_mask]) if np.sum(D1_mask) > 0 else np.nan
    result["p_S1_D0"] = np.mean(S[D0_mask]) if np.sum(D0_mask) > 0 else np.nan

    # By instrument if available
    if Z is not None:
        Z1_mask = Z == 1
        Z0_mask = Z == 0
        result["p_S1_Z1"] = np.mean(S[Z1_mask]) if np.sum(Z1_mask) > 0 else np.nan
        result["p_S1_Z0"] = np.mean(S[Z0_mask]) if np.sum(Z0_mask) > 0 else np.nan

    return result


def sace_bounds(
    outcome: ArrayLike,
    treatment: ArrayLike,
    survival: ArrayLike,
    instrument: Optional[ArrayLike] = None,
    monotonicity: Literal["none", "treatment", "selection", "both"] = "none",
    outcome_support: Optional[Tuple[float, float]] = None,
) -> SACEResult:
    """Compute bounds on the Survivor Average Causal Effect (SACE).

    SACE = E[Y(1) - Y(0) | S(0)=1, S(1)=1]

    The treatment effect for always-survivors (those who would survive
    under both treatment and control).

    Parameters
    ----------
    outcome : array-like
        Outcome variable Y. Should be NaN or missing for non-survivors (S=0).
    treatment : array-like
        Treatment indicator D (binary: 0 or 1).
    survival : array-like
        Survival indicator S (binary: 1 if outcome observed, 0 otherwise).
    instrument : array-like, optional
        Instrument Z (binary). If provided, enables ITT-style analysis
        and tighter bounds with randomization.
    monotonicity : {'none', 'treatment', 'selection', 'both'}, default='none'
        Monotonicity assumptions to invoke:
        - 'none': No assumptions (widest bounds)
        - 'treatment': D(1) ≥ D(0) for all (standard IV monotonicity)
        - 'selection': S(1) ≥ S(0) for all (treatment never harms survival)
        - 'both': Both treatment and selection monotonicity
    outcome_support : tuple of (float, float), optional
        Known bounds (Y_min, Y_max) on outcome support.
        If None, uses observed min/max among survivors.

    Returns
    -------
    SACEResult
        Result containing:
        - sace: Point estimate (midpoint of bounds if not identified)
        - se: Standard error (based on bounds width)
        - lower_bound, upper_bound: Bounds on SACE
        - proportion_survivors_treat: P(S=1 | D=1)
        - proportion_survivors_control: P(S=1 | D=0)
        - n: Sample size
        - method: Description of assumptions used

    Notes
    -----
    **Principal Strata for Survival**:
    - Always-survivors (AS): S(0)=1, S(1)=1
    - Protected (P): S(0)=0, S(1)=1 (treatment enables survival)
    - Harmed (H): S(0)=1, S(1)=0 (treatment causes death)
    - Never-survivors (NS): S(0)=0, S(1)=0

    Under selection monotonicity (S(1) ≥ S(0)), the Harmed stratum is empty.

    **Lee (2009) Bounds**:
    Without monotonicity, bounds use trimming to account for selection:
    - Trim the top/bottom of outcome distribution to bound SACE

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 1000
    >>> D = np.random.binomial(1, 0.5, n)
    >>> # Survival depends on treatment (selection into sample)
    >>> S = (np.random.rand(n) < (0.8 + 0.1 * D)).astype(int)
    >>> # Outcome for survivors
    >>> Y_latent = 1.0 + 2.0 * D + np.random.normal(0, 1, n)
    >>> Y = np.where(S == 1, Y_latent, np.nan)
    >>> result = sace_bounds(Y, D, S, monotonicity='selection')
    >>> print(f"SACE bounds: [{result['lower_bound']:.2f}, {result['upper_bound']:.2f}]")
    """
    Y, D, S, Z = _validate_sace_inputs(outcome, treatment, survival, instrument)

    n = len(Y)

    # Compute survival proportions
    surv_props = _compute_survival_proportions(D, S, Z)
    p_S1_D1 = surv_props["p_S1_D1"]
    p_S1_D0 = surv_props["p_S1_D0"]

    # Outcome support
    survivors = S == 1
    Y_survivors = Y[survivors]

    if len(Y_survivors) == 0:
        raise ValueError("No survivors in the data (all S=0).")

    if outcome_support is not None:
        Y_min, Y_max = outcome_support
    else:
        Y_min = np.nanmin(Y_survivors)
        Y_max = np.nanmax(Y_survivors)

    # Conditional means among survivors
    D1_S1 = (D == 1) & (S == 1)
    D0_S1 = (D == 0) & (S == 1)

    E_Y_D1_S1 = np.nanmean(Y[D1_S1]) if np.sum(D1_S1) > 0 else np.nan
    E_Y_D0_S1 = np.nanmean(Y[D0_S1]) if np.sum(D0_S1) > 0 else np.nan

    # Compute bounds based on monotonicity assumption
    if monotonicity == "both":
        # Both treatment and selection monotonicity
        # Tightest bounds - can use standard IV-like reasoning
        assumptions = ["treatment_monotonicity", "selection_monotonicity"]
        lower_bound, upper_bound = _compute_bounds_both_monotonicity(
            Y, D, S, Z, p_S1_D1, p_S1_D0, E_Y_D1_S1, E_Y_D0_S1, Y_min, Y_max
        )
    elif monotonicity == "selection":
        # S(1) ≥ S(0): treatment never harms survival
        # Harmed stratum is empty
        assumptions = ["selection_monotonicity"]
        lower_bound, upper_bound = _compute_bounds_selection_monotonicity(
            Y, D, S, p_S1_D1, p_S1_D0, E_Y_D1_S1, E_Y_D0_S1, Y_min, Y_max
        )
    elif monotonicity == "treatment":
        # D(1) ≥ D(0): standard IV monotonicity
        assumptions = ["treatment_monotonicity"]
        lower_bound, upper_bound = _compute_bounds_treatment_monotonicity(
            Y, D, S, Z, p_S1_D1, p_S1_D0, E_Y_D1_S1, E_Y_D0_S1, Y_min, Y_max
        )
    else:  # none
        # No assumptions - widest bounds (Lee bounds)
        assumptions = []
        lower_bound, upper_bound = _compute_bounds_no_assumption(
            Y, D, S, p_S1_D1, p_S1_D0, E_Y_D1_S1, E_Y_D0_S1, Y_min, Y_max
        )

    # Point estimate is midpoint of bounds
    sace_estimate = (lower_bound + upper_bound) / 2
    # Conservative SE based on bounds width
    se_estimate = (upper_bound - lower_bound) / (2 * 1.96)  # Implied by CI width

    method_str = f"sace_bounds_{'_'.join(assumptions) if assumptions else 'no_assumption'}"

    return SACEResult(
        sace=sace_estimate,
        se=se_estimate,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        proportion_survivors_treat=p_S1_D1,
        proportion_survivors_control=p_S1_D0,
        n=n,
        method=method_str,
    )


def _compute_bounds_no_assumption(
    Y: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    p_S1_D1: float,
    p_S1_D0: float,
    E_Y_D1_S1: float,
    E_Y_D0_S1: float,
    Y_min: float,
    Y_max: float,
) -> Tuple[float, float]:
    """Lee (2009) bounds without monotonicity assumptions."""
    # The always-survivor proportion is bounded by:
    # max(0, p_S1_D1 + p_S1_D0 - 1) ≤ p_AS ≤ min(p_S1_D1, p_S1_D0)

    p_AS_lower = max(0, p_S1_D1 + p_S1_D0 - 1)
    p_AS_upper = min(p_S1_D1, p_S1_D0)

    if p_AS_upper <= 0:
        # No always-survivors possible
        return (Y_min - Y_max, Y_max - Y_min)

    # Lee bounds: trim to account for selection
    # Under D=1: some survivors might be "protected" (not always-survivors)
    # Under D=0: some survivors might be "harmed" stratum survivors

    # Trimming proportion for D=1 survivors
    trim_D1 = 1 - min(p_S1_D0 / p_S1_D1, 1) if p_S1_D1 > 0 else 0
    # Trimming proportion for D=0 survivors
    trim_D0 = 1 - min(p_S1_D1 / p_S1_D0, 1) if p_S1_D0 > 0 else 0

    D1_S1 = (D == 1) & (S == 1)
    D0_S1 = (D == 0) & (S == 1)

    Y_D1 = Y[D1_S1]
    Y_D0 = Y[D0_S1]

    if len(Y_D1) == 0 or len(Y_D0) == 0:
        return (Y_min - Y_max, Y_max - Y_min)

    # Trimmed means for lower bound: trim top of D=1, trim bottom of D=0
    n_trim_D1 = int(len(Y_D1) * trim_D1)
    n_trim_D0 = int(len(Y_D0) * trim_D0)

    Y_D1_sorted = np.sort(Y_D1)
    Y_D0_sorted = np.sort(Y_D0)

    # Lower bound: E[Y|D=1, trimmed top] - E[Y|D=0, trimmed bottom]
    if n_trim_D1 > 0:
        E_Y_D1_trim_top = np.mean(Y_D1_sorted[:-n_trim_D1])
    else:
        E_Y_D1_trim_top = np.mean(Y_D1_sorted)

    if n_trim_D0 > 0:
        E_Y_D0_trim_bottom = np.mean(Y_D0_sorted[n_trim_D0:])
    else:
        E_Y_D0_trim_bottom = np.mean(Y_D0_sorted)

    lower_bound = E_Y_D1_trim_top - E_Y_D0_trim_bottom

    # Upper bound: E[Y|D=1, trimmed bottom] - E[Y|D=0, trimmed top]
    if n_trim_D1 > 0:
        E_Y_D1_trim_bottom = np.mean(Y_D1_sorted[n_trim_D1:])
    else:
        E_Y_D1_trim_bottom = np.mean(Y_D1_sorted)

    if n_trim_D0 > 0:
        E_Y_D0_trim_top = np.mean(Y_D0_sorted[:-n_trim_D0])
    else:
        E_Y_D0_trim_top = np.mean(Y_D0_sorted)

    upper_bound = E_Y_D1_trim_bottom - E_Y_D0_trim_top

    return (lower_bound, upper_bound)


def _compute_bounds_selection_monotonicity(
    Y: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    p_S1_D1: float,
    p_S1_D0: float,
    E_Y_D1_S1: float,
    E_Y_D0_S1: float,
    Y_min: float,
    Y_max: float,
) -> Tuple[float, float]:
    """Bounds under selection monotonicity S(1) ≥ S(0)."""
    # Under selection monotonicity:
    # - Harmed stratum is empty
    # - D=0 survivors are ALL always-survivors
    # - D=1 survivors include always-survivors AND protected

    # p(always-survivor) = p_S1_D0
    # p(protected) = p_S1_D1 - p_S1_D0 (if positive)

    p_AS = p_S1_D0

    if p_AS <= 0:
        return (Y_min - Y_max, Y_max - Y_min)

    # E[Y(0) | AS] = E[Y | D=0, S=1] exactly
    E_Y0_AS = E_Y_D0_S1

    # E[Y(1) | AS] is bounded because D=1 survivors mix AS and Protected
    # Need to "subtract out" the protected group

    p_protected = max(0, p_S1_D1 - p_S1_D0)

    if p_protected > 0 and p_S1_D1 > 0:
        # E[Y|D=1,S=1] = (p_AS/p_S1_D1) * E[Y(1)|AS] + (p_protected/p_S1_D1) * E[Y(1)|Protected]
        # => E[Y(1)|AS] = (p_S1_D1 * E[Y|D=1,S=1] - p_protected * E[Y(1)|Protected]) / p_AS

        # Bounds on E[Y(1)|Protected] in [Y_min, Y_max]
        E_Y1_AS_lower = (p_S1_D1 * E_Y_D1_S1 - p_protected * Y_max) / p_AS
        E_Y1_AS_upper = (p_S1_D1 * E_Y_D1_S1 - p_protected * Y_min) / p_AS

        # Clip to support
        E_Y1_AS_lower = max(Y_min, E_Y1_AS_lower)
        E_Y1_AS_upper = min(Y_max, E_Y1_AS_upper)
    else:
        # No protected stratum - D=1 survivors are all AS
        E_Y1_AS_lower = E_Y_D1_S1
        E_Y1_AS_upper = E_Y_D1_S1

    lower_bound = E_Y1_AS_lower - E_Y0_AS
    upper_bound = E_Y1_AS_upper - E_Y0_AS

    return (lower_bound, upper_bound)


def _compute_bounds_treatment_monotonicity(
    Y: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    Z: Optional[np.ndarray],
    p_S1_D1: float,
    p_S1_D0: float,
    E_Y_D1_S1: float,
    E_Y_D0_S1: float,
    Y_min: float,
    Y_max: float,
) -> Tuple[float, float]:
    """Bounds under treatment monotonicity D(1) ≥ D(0)."""
    # Treatment monotonicity alone doesn't directly constrain survival strata
    # But with an instrument, it enables IV-style reasoning

    if Z is None:
        # Without instrument, treatment monotonicity doesn't help much for SACE
        return _compute_bounds_no_assumption(
            Y, D, S, p_S1_D1, p_S1_D0, E_Y_D1_S1, E_Y_D0_S1, Y_min, Y_max
        )

    # With instrument: use IV monotonicity to identify complier effects
    # But this requires additional assumptions about survival

    # For now, use conservative bounds
    Z1_mask = Z == 1
    Z0_mask = Z == 0

    # ITT on survival
    p_S1_Z1 = np.mean(S[Z1_mask])
    p_S1_Z0 = np.mean(S[Z0_mask])

    # ITT on outcome (among survivors)
    Z1_S1 = Z1_mask & (S == 1)
    Z0_S1 = Z0_mask & (S == 1)

    E_Y_Z1_S1 = np.mean(Y[Z1_S1]) if np.sum(Z1_S1) > 0 else np.nan
    E_Y_Z0_S1 = np.mean(Y[Z0_S1]) if np.sum(Z0_S1) > 0 else np.nan

    if np.isnan(E_Y_Z1_S1) or np.isnan(E_Y_Z0_S1):
        return (Y_min - Y_max, Y_max - Y_min)

    # Use naive difference as point estimate, widen by outcome range
    naive_diff = E_Y_Z1_S1 - E_Y_Z0_S1
    width = (Y_max - Y_min) * 0.5  # Conservative multiplier

    return (naive_diff - width, naive_diff + width)


def _compute_bounds_both_monotonicity(
    Y: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    Z: Optional[np.ndarray],
    p_S1_D1: float,
    p_S1_D0: float,
    E_Y_D1_S1: float,
    E_Y_D0_S1: float,
    Y_min: float,
    Y_max: float,
) -> Tuple[float, float]:
    """Bounds under both treatment and selection monotonicity."""
    # With both:
    # - D(1) ≥ D(0): no defiers
    # - S(1) ≥ S(0): no harmed stratum

    # Selection monotonicity gives us E[Y(0)|AS] = E[Y|D=0,S=1] exactly
    # Combined with treatment monotonicity and instrument, we can potentially
    # point-identify SACE for compliers

    # Start with selection monotonicity bounds, then tighten if instrument available
    lower, upper = _compute_bounds_selection_monotonicity(
        Y, D, S, p_S1_D1, p_S1_D0, E_Y_D1_S1, E_Y_D0_S1, Y_min, Y_max
    )

    if Z is not None:
        # Can potentially tighten using IV structure
        # For now, keep selection monotonicity bounds
        pass

    return (lower, upper)


def sace_sensitivity(
    outcome: ArrayLike,
    treatment: ArrayLike,
    survival: ArrayLike,
    instrument: Optional[ArrayLike] = None,
    alpha_range: Tuple[float, float] = (0.0, 1.0),
    n_points: int = 50,
) -> Dict[str, np.ndarray]:
    """Sensitivity analysis for SACE varying selection assumptions.

    Computes SACE bounds as a function of a sensitivity parameter that
    captures assumptions about selection into survival.

    Parameters
    ----------
    outcome : array-like
        Outcome variable Y.
    treatment : array-like
        Treatment indicator D (binary).
    survival : array-like
        Survival indicator S (binary).
    instrument : array-like, optional
        Instrument Z (binary).
    alpha_range : tuple of (float, float), default=(0.0, 1.0)
        Range of sensitivity parameter alpha.
        alpha = 0: Strong selection (protected have extreme outcomes)
        alpha = 1: No selection (random truncation)
    n_points : int, default=50
        Number of grid points for sensitivity analysis.

    Returns
    -------
    dict with keys:
        'alpha': Array of sensitivity parameter values
        'lower_bound': Array of lower bounds at each alpha
        'upper_bound': Array of upper bounds at each alpha
        'sace': Array of point estimates (midpoints) at each alpha

    Notes
    -----
    The sensitivity parameter alpha interpolates between:
    - alpha = 0: Maximum selection bias (widest bounds)
    - alpha = 1: No differential selection (tightest bounds)

    This allows researchers to assess how sensitive their conclusions
    are to assumptions about selection into survival.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> D = np.random.binomial(1, 0.5, n)
    >>> S = (np.random.rand(n) < (0.7 + 0.2 * D)).astype(int)
    >>> Y_latent = 1.0 + 1.5 * D + np.random.normal(0, 1, n)
    >>> Y = np.where(S == 1, Y_latent, np.nan)
    >>> sens = sace_sensitivity(Y, D, S)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> plt.fill_between(sens['alpha'], sens['lower_bound'], sens['upper_bound'])
    """
    Y, D, S, Z = _validate_sace_inputs(outcome, treatment, survival, instrument)

    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    lower_bounds = np.zeros(n_points)
    upper_bounds = np.zeros(n_points)

    # Get base components
    survivors = S == 1
    Y_survivors = Y[survivors]

    if len(Y_survivors) == 0:
        raise ValueError("No survivors in the data.")

    Y_min = np.nanmin(Y_survivors)
    Y_max = np.nanmax(Y_survivors)

    surv_props = _compute_survival_proportions(D, S, Z)
    p_S1_D1 = surv_props["p_S1_D1"]
    p_S1_D0 = surv_props["p_S1_D0"]

    D1_S1 = (D == 1) & (S == 1)
    D0_S1 = (D == 0) & (S == 1)

    E_Y_D1_S1 = np.nanmean(Y[D1_S1]) if np.sum(D1_S1) > 0 else np.nan
    E_Y_D0_S1 = np.nanmean(Y[D0_S1]) if np.sum(D0_S1) > 0 else np.nan

    # Get extreme bounds (no assumption)
    lb_extreme, ub_extreme = _compute_bounds_no_assumption(
        Y, D, S, p_S1_D1, p_S1_D0, E_Y_D1_S1, E_Y_D0_S1, Y_min, Y_max
    )

    # Get tight bounds (selection monotonicity)
    lb_tight, ub_tight = _compute_bounds_selection_monotonicity(
        Y, D, S, p_S1_D1, p_S1_D0, E_Y_D1_S1, E_Y_D0_S1, Y_min, Y_max
    )

    # Interpolate between extremes based on alpha
    for i, alpha in enumerate(alphas):
        # alpha = 0: extreme bounds, alpha = 1: tight bounds
        lower_bounds[i] = (1 - alpha) * lb_extreme + alpha * lb_tight
        upper_bounds[i] = (1 - alpha) * ub_extreme + alpha * ub_tight

    sace_points = (lower_bounds + upper_bounds) / 2

    return {
        "alpha": alphas,
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds,
        "sace": sace_points,
    }
