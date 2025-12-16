"""Result types for sensitivity analysis.

This module defines TypedDict classes for sensitivity analysis results,
following the same pattern as other modules in the causal inference library.
"""

from typing import TypedDict, Optional
import numpy as np
from numpy.typing import NDArray


class EValueResult(TypedDict):
    """Result from E-value sensitivity analysis.

    The E-value represents the minimum strength of confounding (on the risk
    ratio scale) that an unmeasured confounder would need to have with both
    the treatment and outcome to fully explain away the observed association.

    Attributes
    ----------
    e_value : float
        E-value for the point estimate. An unmeasured confounder would need
        associations of at least this magnitude with both treatment and
        outcome to explain away the effect.
    e_value_ci : float
        E-value for the confidence interval bound closest to the null.
        This represents the minimum confounding needed to shift the CI
        to include the null. Returns 1.0 if CI already includes null.
    rr_equivalent : float
        The effect estimate converted to the risk ratio scale.
        Used for interpretation regardless of input effect type.
    effect_type : str
        The type of effect measure provided ("rr", "or", "hr", "smd", "ate").
    interpretation : str
        Human-readable interpretation of the E-value results.
    """

    e_value: float
    e_value_ci: float
    rr_equivalent: float
    effect_type: str
    interpretation: str


class RosenbaumResult(TypedDict):
    """Result from Rosenbaum bounds sensitivity analysis.

    Rosenbaum bounds assess how sensitive matched-pair study conclusions
    are to potential unmeasured confounding. The parameter Gamma (Γ)
    represents how much more likely similar units could be to receive
    treatment due to an unmeasured confounder.

    Attributes
    ----------
    gamma_values : NDArray[np.float64]
        Array of Γ values evaluated.
    p_upper : NDArray[np.float64]
        Upper bound p-values at each Γ (worst case for finding an effect).
    p_lower : NDArray[np.float64]
        Lower bound p-values at each Γ (best case for finding an effect).
    gamma_critical : Optional[float]
        Smallest Γ at which upper bound p-value exceeds alpha.
        None if result is robust to all tested Γ values.
    observed_statistic : float
        The observed test statistic (Wilcoxon signed-rank).
    n_pairs : int
        Number of matched pairs analyzed.
    alpha : float
        Significance level used for determining gamma_critical.
    interpretation : str
        Human-readable interpretation of sensitivity results.
    """

    gamma_values: NDArray[np.float64]
    p_upper: NDArray[np.float64]
    p_lower: NDArray[np.float64]
    gamma_critical: Optional[float]
    observed_statistic: float
    n_pairs: int
    alpha: float
    interpretation: str
