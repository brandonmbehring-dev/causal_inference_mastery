"""
Type definitions for partial identification bounds.

Implements TypedDicts for Manski bounds and Lee bounds results.

References
----------
- Manski, C. F. (1990). Nonparametric Bounds on Treatment Effects.
  American Economic Review, 80(2), 319-323.
- Manski, C. F. (2003). Partial Identification of Probability Distributions.
  Springer.
- Lee, D. S. (2009). Training, Wages, and Sample Selection: Estimating
  Sharp Bounds on Treatment Effects. Review of Economic Studies, 76(3), 1071-1102.
"""

from typing import TypedDict, Optional, Literal
import numpy as np


class ManskiBoundsResult(TypedDict):
    """
    Result from Manski partial identification bounds.

    Attributes
    ----------
    bounds_lower : float
        Lower bound of the identification region.
    bounds_upper : float
        Upper bound of the identification region.
    bounds_width : float
        Width of bounds (upper - lower). Narrower is better.
    point_identified : bool
        True if bounds collapse to a point (width ≈ 0).
    assumptions : str
        Which assumptions were used:
        - "worst_case": No assumptions (widest bounds)
        - "mtr": Monotone Treatment Response (Y₁ ≥ Y₀ or Y₁ ≤ Y₀)
        - "mts": Monotone Treatment Selection
        - "mtr_mts": Combined MTR + MTS (narrowest)
        - "iv": Instrumental variable bounds
    mtr_direction : str, optional
        For MTR bounds: "positive" (Y₁ ≥ Y₀) or "negative" (Y₁ ≤ Y₀).
    naive_ate : float
        Naive difference-in-means ATE for comparison.
    ate_in_bounds : bool
        Whether naive ATE falls within the bounds.
    n_treated : int
        Number of treated observations.
    n_control : int
        Number of control observations.
    outcome_support : tuple
        (Y_min, Y_max) used for bounds computation.
    interpretation : str
        Human-readable interpretation of the bounds.
    """

    bounds_lower: float
    bounds_upper: float
    bounds_width: float
    point_identified: bool
    assumptions: str
    mtr_direction: Optional[str]
    naive_ate: float
    ate_in_bounds: bool
    n_treated: int
    n_control: int
    outcome_support: tuple
    interpretation: str


class ManskiIVBoundsResult(TypedDict):
    """
    Result from Manski bounds with instrumental variable.

    Extends ManskiBoundsResult with IV-specific fields.

    Attributes
    ----------
    bounds_lower : float
        Lower bound of the identification region.
    bounds_upper : float
        Upper bound of the identification region.
    bounds_width : float
        Width of bounds.
    point_identified : bool
        True if bounds collapse to a point.
    assumptions : str
        Always "iv" for this result type.
    iv_strength : float
        Measure of instrument strength (first-stage F or correlation).
    complier_share : float
        Estimated share of compliers (for LATE interpretation).
    n_treated : int
        Number of treated observations.
    n_control : int
        Number of control observations.
    n_iv_1 : int
        Number with instrument = 1.
    n_iv_0 : int
        Number with instrument = 0.
    outcome_support : tuple
        (Y_min, Y_max) used for bounds computation.
    interpretation : str
        Human-readable interpretation.
    """

    bounds_lower: float
    bounds_upper: float
    bounds_width: float
    point_identified: bool
    assumptions: str
    iv_strength: float
    complier_share: float
    n_treated: int
    n_control: int
    n_iv_1: int
    n_iv_0: int
    outcome_support: tuple
    interpretation: str


class LeeBoundsResult(TypedDict):
    """
    Result from Lee (2009) bounds for sample selection.

    Attributes
    ----------
    bounds_lower : float
        Lower bound of the identification region.
    bounds_upper : float
        Upper bound of the identification region.
    bounds_width : float
        Width of bounds.
    ci_lower : float
        Lower bound of confidence interval (via bootstrap).
    ci_upper : float
        Upper bound of confidence interval.
    point_identified : bool
        True if no differential attrition.
    trimming_proportion : float
        Proportion of observations trimmed.
    trimmed_group : str
        Which group was trimmed: "treated" or "control".
    attrition_treated : float
        Attrition rate in treatment group.
    attrition_control : float
        Attrition rate in control group.
    n_treated_observed : int
        Number of treated with observed outcomes.
    n_control_observed : int
        Number of control with observed outcomes.
    n_trimmed : int
        Number of observations trimmed.
    monotonicity_assumption : str
        "positive" (treatment increases observation) or "negative".
    interpretation : str
        Human-readable interpretation.
    """

    bounds_lower: float
    bounds_upper: float
    bounds_width: float
    ci_lower: float
    ci_upper: float
    point_identified: bool
    trimming_proportion: float
    trimmed_group: str
    attrition_treated: float
    attrition_control: float
    n_treated_observed: int
    n_control_observed: int
    n_trimmed: int
    monotonicity_assumption: str
    interpretation: str
