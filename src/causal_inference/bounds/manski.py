"""
Manski partial identification bounds for treatment effects.

Implements non-parametric bounds under various identifying assumptions:
1. Worst-case bounds (no assumptions)
2. Monotone Treatment Response (MTR)
3. Monotone Treatment Selection (MTS)
4. Combined MTR + MTS
5. Instrumental variable bounds

The key insight of partial identification is that without strong assumptions,
we can only identify a *set* of possible treatment effects, not a point estimate.

Mathematical Framework
----------------------
The fundamental problem is that we observe:
- E[Y|T=1] for treated
- E[Y|T=0] for control

But we want:
- E[Y₁ - Y₀] = E[Y₁] - E[Y₀]

The challenge is that E[Y₁] ≠ E[Y|T=1] in general (selection bias).

References
----------
- Manski, C. F. (1990). Nonparametric Bounds on Treatment Effects.
  American Economic Review, 80(2), 319-323.
- Manski, C. F. (2003). Partial Identification of Probability Distributions.
  Springer.
- Manski, C. F. & Pepper, J. V. (2000). Monotone Instrumental Variables.
  Econometrica, 68(4), 997-1010.
"""

from typing import Optional, Tuple, Literal
import numpy as np
import warnings

from .types import ManskiBoundsResult, ManskiIVBoundsResult


def manski_worst_case(
    outcome: np.ndarray,
    treatment: np.ndarray,
    outcome_support: Optional[Tuple[float, float]] = None,
) -> ManskiBoundsResult:
    """
    Compute worst-case (no assumption) Manski bounds.

    These are the widest possible bounds, assuming only that outcomes
    are bounded. No assumptions about selection or treatment response.

    The bounds are:
        Lower: E[Y|T=1] - Y_max (assuming all T=0 have Y₁ = Y_max)
        Upper: E[Y|T=1] - Y_min (assuming all T=0 have Y₁ = Y_min)

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes (n,).
    treatment : np.ndarray
        Treatment indicator (n,), binary 0/1.
    outcome_support : tuple, optional
        (Y_min, Y_max) bounds on outcome. If None, uses observed min/max.

    Returns
    -------
    ManskiBoundsResult
        Bounds and diagnostic information.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 1000
    >>> treatment = np.random.binomial(1, 0.5, n)
    >>> outcome = 2 * treatment + np.random.randn(n)  # True ATE = 2
    >>> result = manski_worst_case(outcome, treatment)
    >>> print(f"Bounds: [{result['bounds_lower']:.2f}, {result['bounds_upper']:.2f}]")
    """
    # Input validation
    outcome = np.asarray(outcome, dtype=np.float64)
    treatment = np.asarray(treatment)

    if len(outcome) != len(treatment):
        raise ValueError(
            f"Length mismatch: outcome ({len(outcome)}) != treatment ({len(treatment)})"
        )

    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("Treatment must be binary (0 or 1)")

    n_treated = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)

    if n_treated == 0:
        raise ValueError("No treated observations")
    if n_control == 0:
        raise ValueError("No control observations")

    # Outcome support
    if outcome_support is not None:
        y_min, y_max = outcome_support
        if y_min > y_max:
            raise ValueError(f"Invalid support: y_min ({y_min}) > y_max ({y_max})")
    else:
        y_min, y_max = float(outcome.min()), float(outcome.max())

    # Observed conditional means
    e_y1 = float(outcome[treatment == 1].mean())  # E[Y|T=1]
    e_y0 = float(outcome[treatment == 0].mean())  # E[Y|T=0]

    # Naive ATE
    naive_ate = e_y1 - e_y0

    # Worst-case bounds
    # For ATE = E[Y₁] - E[Y₀]:
    # E[Y₁] = P(T=1)E[Y|T=1] + P(T=0)E[Y₁|T=0]
    # E[Y₀] = P(T=1)E[Y₀|T=1] + P(T=0)E[Y|T=0]
    #
    # Lower bound: assume Y₁|T=0 = Y_min and Y₀|T=1 = Y_max
    # Upper bound: assume Y₁|T=0 = Y_max and Y₀|T=1 = Y_min

    p_t1 = n_treated / len(outcome)
    p_t0 = n_control / len(outcome)

    # E[Y₁] bounds
    e_y1_lower = p_t1 * e_y1 + p_t0 * y_min
    e_y1_upper = p_t1 * e_y1 + p_t0 * y_max

    # E[Y₀] bounds
    e_y0_lower = p_t1 * y_min + p_t0 * e_y0
    e_y0_upper = p_t1 * y_max + p_t0 * e_y0

    # ATE bounds: max lower - min upper gives tightest
    bounds_lower = e_y1_lower - e_y0_upper
    bounds_upper = e_y1_upper - e_y0_lower

    bounds_width = bounds_upper - bounds_lower
    point_identified = bounds_width < 1e-10
    ate_in_bounds = bounds_lower <= naive_ate <= bounds_upper

    interpretation = (
        f"Under no assumptions, the ATE is partially identified in "
        f"[{bounds_lower:.3f}, {bounds_upper:.3f}]. "
        f"The naive estimate ({naive_ate:.3f}) "
        + ("lies within" if ate_in_bounds else "lies outside")
        + " these bounds."
    )

    return ManskiBoundsResult(
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        bounds_width=bounds_width,
        point_identified=point_identified,
        assumptions="worst_case",
        mtr_direction=None,
        naive_ate=naive_ate,
        ate_in_bounds=ate_in_bounds,
        n_treated=int(n_treated),
        n_control=int(n_control),
        outcome_support=(y_min, y_max),
        interpretation=interpretation,
    )


def manski_mtr(
    outcome: np.ndarray,
    treatment: np.ndarray,
    direction: Literal["positive", "negative"] = "positive",
    outcome_support: Optional[Tuple[float, float]] = None,
) -> ManskiBoundsResult:
    """
    Compute Manski bounds under Monotone Treatment Response (MTR).

    MTR assumes that treatment has a monotone effect on outcomes:
    - Positive MTR: Y₁ ≥ Y₀ for all units (treatment never hurts)
    - Negative MTR: Y₁ ≤ Y₀ for all units (treatment never helps)

    These bounds are tighter than worst-case because they constrain
    the counterfactual outcomes.

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes (n,).
    treatment : np.ndarray
        Treatment indicator (n,), binary 0/1.
    direction : {"positive", "negative"}, default="positive"
        Direction of monotone response.
        - "positive": Y₁ ≥ Y₀ (treatment helps or is neutral)
        - "negative": Y₁ ≤ Y₀ (treatment hurts or is neutral)
    outcome_support : tuple, optional
        (Y_min, Y_max) bounds on outcome.

    Returns
    -------
    ManskiBoundsResult
        Bounds and diagnostic information.

    Notes
    -----
    Under positive MTR:
    - Y₁|T=0 ≥ Y|T=0 (treated counterfactual at least as good as observed)
    - Y₀|T=1 ≤ Y|T=1 (control counterfactual at most as good as observed)

    Examples
    --------
    >>> import numpy as np
    >>> # Training program that cannot make workers worse off
    >>> result = manski_mtr(wages, training, direction="positive")
    """
    # Input validation
    outcome = np.asarray(outcome, dtype=np.float64)
    treatment = np.asarray(treatment)

    if len(outcome) != len(treatment):
        raise ValueError(
            f"Length mismatch: outcome ({len(outcome)}) != treatment ({len(treatment)})"
        )

    if direction not in ("positive", "negative"):
        raise ValueError(f"direction must be 'positive' or 'negative', got '{direction}'")

    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("Treatment must be binary (0 or 1)")

    n_treated = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)

    if n_treated == 0:
        raise ValueError("No treated observations")
    if n_control == 0:
        raise ValueError("No control observations")

    # Outcome support
    if outcome_support is not None:
        y_min, y_max = outcome_support
    else:
        y_min, y_max = float(outcome.min()), float(outcome.max())

    # Observed conditional means
    e_y1 = float(outcome[treatment == 1].mean())
    e_y0 = float(outcome[treatment == 0].mean())

    p_t1 = n_treated / len(outcome)
    p_t0 = n_control / len(outcome)

    naive_ate = e_y1 - e_y0

    if direction == "positive":
        # Y₁ ≥ Y₀ for all units
        # E[Y₁|T=0] ≥ E[Y|T=0], so E[Y₁|T=0] ∈ [E[Y|T=0], Y_max]
        # E[Y₀|T=1] ≤ E[Y|T=1], so E[Y₀|T=1] ∈ [Y_min, E[Y|T=1]]

        e_y1_lower = p_t1 * e_y1 + p_t0 * e_y0  # Y₁|T=0 = Y|T=0
        e_y1_upper = p_t1 * e_y1 + p_t0 * y_max  # Y₁|T=0 = Y_max

        e_y0_lower = p_t1 * y_min + p_t0 * e_y0  # Y₀|T=1 = Y_min
        e_y0_upper = p_t1 * e_y1 + p_t0 * e_y0  # Y₀|T=1 = Y|T=1

        # ATE bounds
        bounds_lower = max(0, e_y1_lower - e_y0_upper)  # Must be non-negative
        bounds_upper = e_y1_upper - e_y0_lower

    else:  # negative
        # Y₁ ≤ Y₀ for all units
        # E[Y₁|T=0] ≤ E[Y|T=0], so E[Y₁|T=0] ∈ [Y_min, E[Y|T=0]]
        # E[Y₀|T=1] ≥ E[Y|T=1], so E[Y₀|T=1] ∈ [E[Y|T=1], Y_max]

        e_y1_lower = p_t1 * e_y1 + p_t0 * y_min
        e_y1_upper = p_t1 * e_y1 + p_t0 * e_y0

        e_y0_lower = p_t1 * e_y1 + p_t0 * e_y0
        e_y0_upper = p_t1 * y_max + p_t0 * e_y0

        # ATE bounds
        bounds_lower = e_y1_lower - e_y0_upper
        bounds_upper = min(0, e_y1_upper - e_y0_lower)  # Must be non-positive

    bounds_width = bounds_upper - bounds_lower
    point_identified = bounds_width < 1e-10
    ate_in_bounds = bounds_lower <= naive_ate <= bounds_upper

    direction_text = "non-negative" if direction == "positive" else "non-positive"
    interpretation = (
        f"Under Monotone Treatment Response ({direction}), "
        f"the ATE is bounded in [{bounds_lower:.3f}, {bounds_upper:.3f}]. "
        f"MTR implies the effect is {direction_text}."
    )

    return ManskiBoundsResult(
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        bounds_width=bounds_width,
        point_identified=point_identified,
        assumptions="mtr",
        mtr_direction=direction,
        naive_ate=naive_ate,
        ate_in_bounds=ate_in_bounds,
        n_treated=int(n_treated),
        n_control=int(n_control),
        outcome_support=(y_min, y_max),
        interpretation=interpretation,
    )


def manski_mts(
    outcome: np.ndarray,
    treatment: np.ndarray,
    outcome_support: Optional[Tuple[float, float]] = None,
) -> ManskiBoundsResult:
    """
    Compute Manski bounds under Monotone Treatment Selection (MTS).

    MTS assumes positive selection: units with higher potential outcomes
    are more likely to select into treatment.

    Formally: E[Y₁|T=1] ≥ E[Y₁|T=0] and E[Y₀|T=1] ≥ E[Y₀|T=0]

    This is plausible when more capable/advantaged individuals self-select
    into treatment (e.g., college enrollment, job training).

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes (n,).
    treatment : np.ndarray
        Treatment indicator (n,), binary 0/1.
    outcome_support : tuple, optional
        (Y_min, Y_max) bounds on outcome.

    Returns
    -------
    ManskiBoundsResult
        Bounds and diagnostic information.

    Notes
    -----
    Under MTS:
    - E[Y₁|T=0] ≤ E[Y|T=1] (untreated would have lower Y₁ than treated)
    - E[Y₀|T=1] ≥ E[Y|T=0] (treated would have higher Y₀ than untreated)

    This implies: E[Y|T=1] - E[Y|T=0] ≥ ATE (naive estimate is upward biased)

    Examples
    --------
    >>> import numpy as np
    >>> # College wage premium with positive selection
    >>> result = manski_mts(wages, college_degree)
    """
    # Input validation
    outcome = np.asarray(outcome, dtype=np.float64)
    treatment = np.asarray(treatment)

    if len(outcome) != len(treatment):
        raise ValueError(
            f"Length mismatch: outcome ({len(outcome)}) != treatment ({len(treatment)})"
        )

    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("Treatment must be binary (0 or 1)")

    n_treated = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)

    if n_treated == 0:
        raise ValueError("No treated observations")
    if n_control == 0:
        raise ValueError("No control observations")

    # Outcome support
    if outcome_support is not None:
        y_min, y_max = outcome_support
    else:
        y_min, y_max = float(outcome.min()), float(outcome.max())

    # Observed conditional means
    e_y1 = float(outcome[treatment == 1].mean())
    e_y0 = float(outcome[treatment == 0].mean())

    p_t1 = n_treated / len(outcome)
    p_t0 = n_control / len(outcome)

    naive_ate = e_y1 - e_y0

    # Under MTS:
    # E[Y₁|T=0] ∈ [Y_min, E[Y|T=1]]
    # E[Y₀|T=1] ∈ [E[Y|T=0], Y_max]

    # E[Y₁] bounds
    e_y1_lower = p_t1 * e_y1 + p_t0 * y_min
    e_y1_upper = p_t1 * e_y1 + p_t0 * e_y1  # = E[Y|T=1]

    # E[Y₀] bounds
    e_y0_lower = p_t1 * e_y0 + p_t0 * e_y0  # = E[Y|T=0]
    e_y0_upper = p_t1 * y_max + p_t0 * e_y0

    # ATE bounds
    bounds_lower = e_y1_lower - e_y0_upper
    bounds_upper = e_y1_upper - e_y0_lower  # = E[Y|T=1] - E[Y|T=0] = naive

    bounds_width = bounds_upper - bounds_lower
    point_identified = bounds_width < 1e-10
    ate_in_bounds = bounds_lower <= naive_ate <= bounds_upper

    interpretation = (
        f"Under Monotone Treatment Selection, "
        f"the ATE is bounded in [{bounds_lower:.3f}, {bounds_upper:.3f}]. "
        f"MTS implies the naive estimate ({naive_ate:.3f}) is an upper bound on ATE."
    )

    return ManskiBoundsResult(
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        bounds_width=bounds_width,
        point_identified=point_identified,
        assumptions="mts",
        mtr_direction=None,
        naive_ate=naive_ate,
        ate_in_bounds=ate_in_bounds,
        n_treated=int(n_treated),
        n_control=int(n_control),
        outcome_support=(y_min, y_max),
        interpretation=interpretation,
    )


def manski_mtr_mts(
    outcome: np.ndarray,
    treatment: np.ndarray,
    mtr_direction: Literal["positive", "negative"] = "positive",
    outcome_support: Optional[Tuple[float, float]] = None,
) -> ManskiBoundsResult:
    """
    Compute Manski bounds under combined MTR + MTS assumptions.

    Combines:
    - Monotone Treatment Response: Y₁ ≥ Y₀ (or ≤) for all units
    - Monotone Treatment Selection: positive selection into treatment

    This is the tightest bound under these behavioral assumptions.

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes (n,).
    treatment : np.ndarray
        Treatment indicator (n,), binary 0/1.
    mtr_direction : {"positive", "negative"}, default="positive"
        Direction of monotone treatment response.
    outcome_support : tuple, optional
        (Y_min, Y_max) bounds on outcome.

    Returns
    -------
    ManskiBoundsResult
        Bounds and diagnostic information.

    Notes
    -----
    Under positive MTR + MTS:
    - Lower bound: 0 (from MTR)
    - Upper bound: E[Y|T=1] - E[Y|T=0] (from MTS)

    This gives informative bounds when both assumptions are plausible.

    Examples
    --------
    >>> import numpy as np
    >>> # Job training: positive effect, positive selection
    >>> result = manski_mtr_mts(wages, training, mtr_direction="positive")
    """
    # Input validation
    outcome = np.asarray(outcome, dtype=np.float64)
    treatment = np.asarray(treatment)

    if len(outcome) != len(treatment):
        raise ValueError(
            f"Length mismatch: outcome ({len(outcome)}) != treatment ({len(treatment)})"
        )

    if mtr_direction not in ("positive", "negative"):
        raise ValueError(f"mtr_direction must be 'positive' or 'negative', got '{mtr_direction}'")

    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("Treatment must be binary (0 or 1)")

    n_treated = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)

    if n_treated == 0:
        raise ValueError("No treated observations")
    if n_control == 0:
        raise ValueError("No control observations")

    # Outcome support
    if outcome_support is not None:
        y_min, y_max = outcome_support
    else:
        y_min, y_max = float(outcome.min()), float(outcome.max())

    # Observed conditional means
    e_y1 = float(outcome[treatment == 1].mean())
    e_y0 = float(outcome[treatment == 0].mean())

    naive_ate = e_y1 - e_y0

    if mtr_direction == "positive":
        # MTR (positive): ATE ≥ 0
        # MTS: ATE ≤ naive
        bounds_lower = max(0.0, min(naive_ate, 0.0))  # At least 0
        bounds_upper = max(naive_ate, 0.0)  # At most naive (if positive)

        # More precise: intersect MTR and MTS bounds
        bounds_lower = 0.0
        bounds_upper = naive_ate if naive_ate >= 0 else 0.0

    else:  # negative MTR
        # MTR (negative): ATE ≤ 0
        # MTS: ATE ≤ naive
        bounds_lower = min(naive_ate, 0.0)
        bounds_upper = 0.0

    bounds_width = bounds_upper - bounds_lower
    point_identified = bounds_width < 1e-10
    ate_in_bounds = bounds_lower <= naive_ate <= bounds_upper

    if mtr_direction == "positive":
        interpretation = (
            f"Under MTR (positive) + MTS, the ATE is bounded in "
            f"[{bounds_lower:.3f}, {bounds_upper:.3f}]. "
            f"The effect is non-negative and at most the naive estimate."
        )
    else:
        interpretation = (
            f"Under MTR (negative) + MTS, the ATE is bounded in "
            f"[{bounds_lower:.3f}, {bounds_upper:.3f}]. "
            f"The effect is non-positive."
        )

    return ManskiBoundsResult(
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        bounds_width=bounds_width,
        point_identified=point_identified,
        assumptions="mtr_mts",
        mtr_direction=mtr_direction,
        naive_ate=naive_ate,
        ate_in_bounds=ate_in_bounds,
        n_treated=int(n_treated),
        n_control=int(n_control),
        outcome_support=(y_min, y_max),
        interpretation=interpretation,
    )


def manski_iv(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    outcome_support: Optional[Tuple[float, float]] = None,
) -> ManskiIVBoundsResult:
    """
    Compute Manski bounds with an instrumental variable.

    Uses an instrument to tighten bounds without full LATE assumptions.
    The instrument provides exogenous variation that helps identify
    the treatment effect bounds.

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes (n,).
    treatment : np.ndarray
        Treatment indicator (n,), binary 0/1.
    instrument : np.ndarray
        Instrumental variable (n,), binary 0/1.
    outcome_support : tuple, optional
        (Y_min, Y_max) bounds on outcome.

    Returns
    -------
    ManskiIVBoundsResult
        Bounds with IV-specific diagnostics.

    Notes
    -----
    The IV bounds use the insight that:
    - Among Z=1 group: some are treated, some not
    - Among Z=0 group: some are treated, some not

    By comparing outcomes across Z groups, we can bound the effect.

    Without monotonicity (no defiers), these are partial identification bounds.
    With monotonicity, they can sometimes point identify LATE.

    Examples
    --------
    >>> import numpy as np
    >>> # Draft lottery as instrument for military service
    >>> result = manski_iv(wages, served, draft_eligible)
    """
    # Input validation
    outcome = np.asarray(outcome, dtype=np.float64)
    treatment = np.asarray(treatment)
    instrument = np.asarray(instrument)

    if len(outcome) != len(treatment) or len(outcome) != len(instrument):
        raise ValueError("All arrays must have the same length")

    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("Treatment must be binary (0 or 1)")

    if not np.isin(instrument, [0, 1]).all():
        raise ValueError("Instrument must be binary (0 or 1)")

    # Count observations
    n_iv_1 = np.sum(instrument == 1)
    n_iv_0 = np.sum(instrument == 0)
    n_treated = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)

    if n_iv_1 == 0 or n_iv_0 == 0:
        raise ValueError("Instrument must have variation (both 0 and 1 values)")

    # Outcome support
    if outcome_support is not None:
        y_min, y_max = outcome_support
    else:
        y_min, y_max = float(outcome.min()), float(outcome.max())

    # Conditional means by instrument value
    e_y_z1 = float(outcome[instrument == 1].mean())
    e_y_z0 = float(outcome[instrument == 0].mean())

    # Treatment probabilities by instrument
    p_t1_z1 = float(np.mean(treatment[instrument == 1]))  # P(T=1|Z=1)
    p_t1_z0 = float(np.mean(treatment[instrument == 0]))  # P(T=1|Z=0)

    # Complier share (if monotonicity holds)
    complier_share = abs(p_t1_z1 - p_t1_z0)

    # IV strength measure
    iv_strength = complier_share  # Simple measure; could use F-stat

    if complier_share < 0.01:
        warnings.warn(
            "Weak instrument: complier share < 1%. Bounds may be uninformative.",
            UserWarning,
        )

    # Wald estimate (for comparison)
    if complier_share > 1e-10:
        wald_estimate = (e_y_z1 - e_y_z0) / (p_t1_z1 - p_t1_z0)
    else:
        wald_estimate = float("nan")

    # IV bounds (Manski & Pepper approach)
    # These use the reduced form variation to bound the effect

    # Reduced form effect
    rf_effect = e_y_z1 - e_y_z0

    # Bounds depend on whether Z increases or decreases treatment
    if p_t1_z1 > p_t1_z0:
        # Z=1 increases treatment probability
        # Lower bound: assume non-compliers have extreme outcomes
        bounds_lower = (rf_effect - (1 - complier_share) * (y_max - y_min)) / complier_share
        bounds_upper = (rf_effect + (1 - complier_share) * (y_max - y_min)) / complier_share

        # Tighten to reasonable range
        bounds_lower = max(bounds_lower, y_min - y_max)
        bounds_upper = min(bounds_upper, y_max - y_min)
    else:
        # Z=1 decreases treatment probability
        bounds_lower = -(rf_effect + (1 - complier_share) * (y_max - y_min)) / complier_share
        bounds_upper = -(rf_effect - (1 - complier_share) * (y_max - y_min)) / complier_share

        bounds_lower = max(bounds_lower, y_min - y_max)
        bounds_upper = min(bounds_upper, y_max - y_min)

    bounds_width = bounds_upper - bounds_lower
    point_identified = bounds_width < 1e-10

    interpretation = (
        f"Using IV, the ATE is bounded in [{bounds_lower:.3f}, {bounds_upper:.3f}]. "
        f"Complier share: {complier_share:.1%}. "
        f"Wald estimate: {wald_estimate:.3f}."
    )

    return ManskiIVBoundsResult(
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        bounds_width=bounds_width,
        point_identified=point_identified,
        assumptions="iv",
        iv_strength=iv_strength,
        complier_share=complier_share,
        n_treated=int(n_treated),
        n_control=int(n_control),
        n_iv_1=int(n_iv_1),
        n_iv_0=int(n_iv_0),
        outcome_support=(y_min, y_max),
        interpretation=interpretation,
    )


def compare_bounds(
    outcome: np.ndarray,
    treatment: np.ndarray,
    outcome_support: Optional[Tuple[float, float]] = None,
    mtr_direction: Literal["positive", "negative"] = "positive",
) -> dict:
    """
    Compare bounds under different assumptions.

    Useful for understanding how different identifying assumptions
    affect the width of identification regions.

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes.
    treatment : np.ndarray
        Treatment indicator.
    outcome_support : tuple, optional
        (Y_min, Y_max) bounds on outcome.
    mtr_direction : {"positive", "negative"}, default="positive"
        Direction for MTR assumption.

    Returns
    -------
    dict
        Dictionary with results from each bounds method.

    Examples
    --------
    >>> comparison = compare_bounds(outcome, treatment)
    >>> for method, result in comparison.items():
    ...     print(f"{method}: [{result['bounds_lower']:.2f}, {result['bounds_upper']:.2f}]")
    """
    results = {
        "worst_case": manski_worst_case(outcome, treatment, outcome_support),
        "mtr": manski_mtr(outcome, treatment, mtr_direction, outcome_support),
        "mts": manski_mts(outcome, treatment, outcome_support),
        "mtr_mts": manski_mtr_mts(outcome, treatment, mtr_direction, outcome_support),
    }

    # Add summary
    widths = {k: v["bounds_width"] for k, v in results.items()}
    narrowest = min(widths, key=widths.get)
    widest = max(widths, key=widths.get)

    results["_summary"] = {
        "narrowest": narrowest,
        "widest": widest,
        "width_reduction": 1 - widths[narrowest] / widths[widest] if widths[widest] > 0 else 0,
    }

    return results
