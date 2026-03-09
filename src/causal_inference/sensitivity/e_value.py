"""E-value sensitivity analysis for unmeasured confounding.

The E-value (VanderWeele & Ding, 2017) quantifies the minimum strength of
association that an unmeasured confounder would need to have with both the
treatment and outcome to fully explain away an observed causal effect.

Key Concepts
------------
- E-value answers: "How strong would confounding need to be to explain this away?"
- Higher E-values indicate more robust findings
- E-value = 1.0 means no confounding needed (effect is null)
- E-value = 2.0 means confounder needs 2x association with both T and Y

Formula
-------
For a risk ratio RR > 1:
    E = RR + sqrt(RR * (RR - 1))

For RR < 1, first convert to 1/RR, then apply formula.

References
----------
- VanderWeele TJ, Ding P (2017). "Sensitivity Analysis in Observational Research:
  Introducing the E-Value." Annals of Internal Medicine 167(4): 268-274.
- Mathur MB, Ding P, et al. (2018). "Website and R Package for Computing E-values."
  Epidemiology 29(5): e45-e47.
"""

import numpy as np
from typing import Optional, Literal

from .types import EValueResult


def _compute_e_value(rr: float) -> float:
    """Compute E-value from a risk ratio.

    Parameters
    ----------
    rr : float
        Risk ratio (must be >= 1 for direct computation).

    Returns
    -------
    float
        E-value.
    """
    if rr < 1.0:
        raise ValueError("RR must be >= 1 for E-value computation. Use 1/RR if RR < 1.")

    if np.isclose(rr, 1.0):
        return 1.0

    return rr + np.sqrt(rr * (rr - 1))


def _smd_to_rr(d: float) -> float:
    """Convert standardized mean difference to approximate risk ratio.

    Uses the approximation from VanderWeele (2017):
        RR ≈ exp(0.91 * d)

    This assumes normally distributed outcomes.

    Parameters
    ----------
    d : float
        Standardized mean difference (Cohen's d).

    Returns
    -------
    float
        Approximate risk ratio.
    """
    return np.exp(0.91 * d)


def _ate_to_rr(ate: float, baseline_risk: float) -> float:
    """Convert ATE (risk difference) to risk ratio.

    RR = (baseline_risk + ate) / baseline_risk

    Parameters
    ----------
    ate : float
        Average treatment effect on risk difference scale.
    baseline_risk : float
        Baseline (control) risk, must be in (0, 1).

    Returns
    -------
    float
        Risk ratio.

    Raises
    ------
    ValueError
        If baseline_risk is not in valid range or if resulting RR is invalid.
    """
    if not 0 < baseline_risk < 1:
        raise ValueError(f"baseline_risk must be in (0, 1), got {baseline_risk}")

    treated_risk = baseline_risk + ate

    if treated_risk <= 0:
        raise ValueError(
            f"Treated risk (baseline + ATE = {treated_risk:.4f}) must be > 0. "
            f"ATE={ate:.4f} is too negative for baseline_risk={baseline_risk:.4f}."
        )

    if treated_risk > 1:
        raise ValueError(
            f"Treated risk (baseline + ATE = {treated_risk:.4f}) must be <= 1. "
            f"ATE={ate:.4f} is too large for baseline_risk={baseline_risk:.4f}."
        )

    return treated_risk / baseline_risk


def e_value(
    estimate: float,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    effect_type: Literal["rr", "or", "hr", "smd", "ate"] = "rr",
    baseline_risk: Optional[float] = None,
) -> EValueResult:
    """Compute E-value for sensitivity to unmeasured confounding.

    The E-value represents the minimum strength of association on the risk
    ratio scale that an unmeasured confounder would need to have with both
    the treatment and outcome to fully explain away the observed effect.

    Parameters
    ----------
    estimate : float
        Point estimate of the effect.
    ci_lower : float, optional
        Lower bound of confidence interval.
    ci_upper : float, optional
        Upper bound of confidence interval.
    effect_type : {"rr", "or", "hr", "smd", "ate"}, default="rr"
        Type of effect measure:
        - "rr": Risk ratio (direct)
        - "or": Odds ratio (approximated as RR for rare outcomes)
        - "hr": Hazard ratio (approximated as RR)
        - "smd": Standardized mean difference (converted via exp(0.91*d))
        - "ate": Average treatment effect on risk difference scale
                 (requires baseline_risk)
    baseline_risk : float, optional
        Baseline (control group) risk. Required if effect_type="ate".
        Must be in (0, 1).

    Returns
    -------
    EValueResult
        Dictionary containing:
        - e_value: E-value for point estimate
        - e_value_ci: E-value for CI bound closest to null (1.0 if CI includes null)
        - rr_equivalent: Effect converted to RR scale
        - effect_type: Input effect type
        - interpretation: Human-readable interpretation

    Raises
    ------
    ValueError
        If inputs are invalid or conversions fail.

    Examples
    --------
    >>> # Risk ratio of 2.0
    >>> result = e_value(2.0, effect_type="rr")
    >>> print(f"E-value: {result['e_value']:.2f}")
    E-value: 3.41

    >>> # With confidence interval
    >>> result = e_value(2.0, ci_lower=1.5, ci_upper=2.7, effect_type="rr")
    >>> print(f"E-value for CI: {result['e_value_ci']:.2f}")
    E-value for CI: 2.37

    >>> # Standardized mean difference
    >>> result = e_value(0.5, effect_type="smd")
    >>> print(f"E-value: {result['e_value']:.2f}")
    E-value: 2.70

    Notes
    -----
    **Interpretation Guidelines**:

    - E-value ≈ 1.0: No robustness (effect is at or near null)
    - E-value ≈ 1.5: Weak robustness (modest confounding could explain)
    - E-value ≈ 2.0: Moderate robustness
    - E-value ≈ 3.0: Strong robustness
    - E-value > 4.0: Very robust to unmeasured confounding

    **When E-value_CI = 1.0**: The confidence interval includes the null,
    so no confounding is needed to make the effect statistically non-significant.

    **Limitations**:
    - Assumes single unmeasured confounder
    - Based on risk ratio scale (conversions are approximate)
    - Does not account for measurement error

    References
    ----------
    - VanderWeele & Ding (2017). "Sensitivity Analysis in Observational Research:
      Introducing the E-Value."

    See Also
    --------
    rosenbaum_bounds : Sensitivity analysis for matched studies.
    """
    # =========================================================================
    # Convert estimate to risk ratio scale
    # =========================================================================

    if effect_type == "rr":
        rr = estimate
    elif effect_type in ("or", "hr"):
        # OR and HR approximate RR for rare outcomes / proportional hazards
        rr = estimate
    elif effect_type == "smd":
        rr = _smd_to_rr(estimate)
    elif effect_type == "ate":
        if baseline_risk is None:
            raise ValueError(
                "baseline_risk is required for effect_type='ate'. "
                "Provide the control group risk (probability)."
            )
        rr = _ate_to_rr(estimate, baseline_risk)
    else:
        raise ValueError(
            f"Unknown effect_type: {effect_type}. Must be one of: 'rr', 'or', 'hr', 'smd', 'ate'."
        )

    # Validate RR
    if rr <= 0:
        raise ValueError(f"Risk ratio must be positive, got {rr:.4f}")

    # =========================================================================
    # Compute E-value for point estimate
    # =========================================================================

    # E-value formula requires RR >= 1, so flip if needed
    if rr >= 1:
        e_val = _compute_e_value(rr)
    else:
        # Protective effect: use 1/RR
        e_val = _compute_e_value(1 / rr)

    # =========================================================================
    # Compute E-value for confidence interval
    # =========================================================================

    e_val_ci = 1.0  # Default if no CI or CI includes null

    if ci_lower is not None and ci_upper is not None:
        # Convert CI bounds to RR scale
        if effect_type == "smd":
            ci_lower_rr = _smd_to_rr(ci_lower)
            ci_upper_rr = _smd_to_rr(ci_upper)
        elif effect_type == "ate":
            ci_lower_rr = _ate_to_rr(ci_lower, baseline_risk)
            ci_upper_rr = _ate_to_rr(ci_upper, baseline_risk)
        else:
            ci_lower_rr = ci_lower
            ci_upper_rr = ci_upper

        # Check if CI includes null (RR = 1)
        if ci_lower_rr <= 1.0 <= ci_upper_rr:
            e_val_ci = 1.0  # No confounding needed
        elif rr >= 1:
            # Harmful effect: use lower bound (closest to null=1)
            e_val_ci = _compute_e_value(ci_lower_rr) if ci_lower_rr > 1 else 1.0
        else:
            # Protective effect (rr < 1): use upper bound (closest to null=1)
            e_val_ci = _compute_e_value(1 / ci_upper_rr) if ci_upper_rr < 1 else 1.0

    # =========================================================================
    # Generate interpretation
    # =========================================================================

    interpretation = _generate_interpretation(e_val, e_val_ci, rr, effect_type)

    return EValueResult(
        e_value=float(e_val),
        e_value_ci=float(e_val_ci),
        rr_equivalent=float(rr),
        effect_type=effect_type,
        interpretation=interpretation,
    )


def _generate_interpretation(e_val: float, e_val_ci: float, rr: float, effect_type: str) -> str:
    """Generate human-readable interpretation of E-value results."""
    direction = "harmful" if rr >= 1 else "protective"

    # Robustness assessment
    if e_val < 1.25:
        robustness = "not robust"
    elif e_val < 1.75:
        robustness = "weakly robust"
    elif e_val < 2.5:
        robustness = "moderately robust"
    elif e_val < 4.0:
        robustness = "strongly robust"
    else:
        robustness = "very strongly robust"

    lines = [
        f"E-value = {e_val:.2f} for a {direction} effect (RR = {rr:.2f}).",
        f"",
        f"Interpretation: This finding is {robustness} to unmeasured confounding.",
        f"An unmeasured confounder would need to be associated with both",
        f"treatment and outcome by at least {e_val:.2f}-fold (RR scale) to",
        f"fully explain away the observed effect.",
    ]

    if e_val_ci > 1.0:
        lines.append(f"")
        lines.append(
            f"For the CI: Confounding of {e_val_ci:.2f}-fold would be needed "
            f"to move the CI to include the null."
        )
    elif e_val_ci == 1.0 and e_val > 1.0:
        lines.append(f"")
        lines.append(
            "Note: The confidence interval already includes the null (RR=1), "
            "so no confounding is needed for statistical non-significance."
        )

    return "\n".join(lines)
