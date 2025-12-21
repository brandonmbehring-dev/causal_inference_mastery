"""
Partial identification bounds for causal inference.

This module implements non-parametric bounds for treatment effects when
point identification is not possible. These methods provide identification
*regions* rather than point estimates.

Manski Bounds
-------------
- `manski_worst_case`: No assumptions (widest bounds)
- `manski_mtr`: Monotone Treatment Response
- `manski_mts`: Monotone Treatment Selection
- `manski_mtr_mts`: Combined MTR + MTS (narrowest)
- `manski_iv`: With instrumental variable
- `compare_bounds`: Compare all Manski methods

Lee Bounds (Session 87)
-----------------------
- `lee_bounds`: Sharp bounds under sample selection (TBD)

References
----------
- Manski, C. F. (1990). Nonparametric Bounds on Treatment Effects.
- Manski, C. F. (2003). Partial Identification of Probability Distributions.
- Lee, D. S. (2009). Training, Wages, and Sample Selection.
"""

from .manski import (
    manski_worst_case,
    manski_mtr,
    manski_mts,
    manski_mtr_mts,
    manski_iv,
    compare_bounds,
)

from .lee import (
    lee_bounds,
    lee_bounds_tightened,
    check_monotonicity,
)

from .types import (
    ManskiBoundsResult,
    ManskiIVBoundsResult,
    LeeBoundsResult,
)

__all__ = [
    # Manski bounds
    "manski_worst_case",
    "manski_mtr",
    "manski_mts",
    "manski_mtr_mts",
    "manski_iv",
    "compare_bounds",
    # Lee bounds
    "lee_bounds",
    "lee_bounds_tightened",
    "check_monotonicity",
    # Types
    "ManskiBoundsResult",
    "ManskiIVBoundsResult",
    "LeeBoundsResult",
]
