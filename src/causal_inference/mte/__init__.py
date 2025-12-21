"""
Marginal Treatment Effects (MTE) estimation.

Implements the Heckman & Vytlacil (2005) framework for treatment
effect heterogeneity indexed by unobserved resistance to treatment.

Main Functions
--------------
late_estimator
    Estimate LATE via Wald estimator for binary instruments
late_bounds
    Compute bounds on LATE when monotonicity may fail
complier_characteristics
    Characterize the complier subpopulation
local_iv
    Estimate MTE curve via local instrumental variables
polynomial_mte
    Estimate MTE using polynomial approximation
ate_from_mte
    Integrate MTE to get ATE
att_from_mte
    Integrate MTE to get ATT
atu_from_mte
    Integrate MTE to get ATU
prte
    Compute policy-relevant treatment effect
late_from_mte
    Compute LATE for specific instrument shift from MTE

Diagnostics
-----------
common_support_check
    Check propensity score overlap
mte_sensitivity_to_trimming
    Assess sensitivity to trimming choices
monotonicity_test
    Test for monotonicity assumption
propensity_variation_test
    Test whether propensity varies sufficiently
mte_shape_test
    Test hypotheses about MTE shape (constant, linear, monotone)

Types
-----
MTEResult
    Result from local_iv or polynomial_mte
LATEResult
    Result from late_estimator
PolicyResult
    Result from policy parameter functions
ComplierResult
    Result from complier_characteristics
CommonSupportResult
    Result from common_support_check

Examples
--------
>>> # Estimate LATE with binary instrument
>>> from causal_inference.mte import late_estimator
>>> late = late_estimator(wages, college, distance_to_college)
>>> print(f"LATE: {late['late']:.3f} ({late['ci_lower']:.3f}, {late['ci_upper']:.3f})")

>>> # Estimate MTE curve
>>> from causal_inference.mte import local_iv, ate_from_mte
>>> mte = local_iv(wages, college, distance_to_college)
>>> ate = ate_from_mte(mte)
>>> print(f"ATE from MTE: {ate['estimate']:.3f}")

>>> # Check identifying assumptions
>>> from causal_inference.mte import common_support_check, monotonicity_test
>>> support = common_support_check(propensity, treatment)
>>> print(support['recommendation'])

References
----------
- Heckman, J.J. & Vytlacil, E. (1999). Local Instrumental Variables.
- Heckman, J.J. & Vytlacil, E. (2005). Structural Equations, Treatment Effects.
- Carneiro, P., Heckman, J.J. & Vytlacil, E. (2011). Estimating MTE.
- Abadie, A. (2003). Semiparametric instrumental variable estimation.
"""

from .types import (
    MTEResult,
    LATEResult,
    PolicyResult,
    ComplierResult,
    CommonSupportResult,
)

from .late import (
    late_estimator,
    late_bounds,
    complier_characteristics,
)

from .local_iv import (
    local_iv,
    polynomial_mte,
)

from .policy import (
    ate_from_mte,
    att_from_mte,
    atu_from_mte,
    prte,
    late_from_mte,
)

from .diagnostics import (
    common_support_check,
    mte_sensitivity_to_trimming,
    monotonicity_test,
    propensity_variation_test,
    mte_shape_test,
)

__all__ = [
    # Types
    "MTEResult",
    "LATEResult",
    "PolicyResult",
    "ComplierResult",
    "CommonSupportResult",
    # LATE estimation
    "late_estimator",
    "late_bounds",
    "complier_characteristics",
    # MTE estimation
    "local_iv",
    "polynomial_mte",
    # Policy parameters
    "ate_from_mte",
    "att_from_mte",
    "atu_from_mte",
    "prte",
    "late_from_mte",
    # Diagnostics
    "common_support_check",
    "mte_sensitivity_to_trimming",
    "monotonicity_test",
    "propensity_variation_test",
    "mte_shape_test",
]
