"""
Mediation Analysis Module.

Decomposes total treatment effect into direct and indirect effects
through a mediator variable.

Main Functions
--------------
mediation_analysis : Unified interface for mediation estimation
baron_kenny : Classic Baron-Kenny (1986) linear path analysis
natural_direct_effect : NDE via simulation
natural_indirect_effect : NIE via simulation
controlled_direct_effect : CDE at fixed mediator value
mediation_sensitivity : Sensitivity to unmeasured confounding

Mathematical Framework
----------------------
Total Effect (TE):
    TE = E[Y(1, M(1)) - Y(0, M(0))]

Natural Direct Effect (NDE):
    NDE = E[Y(1, M(0)) - Y(0, M(0))]
    Effect of treatment holding mediator at control level

Natural Indirect Effect (NIE):
    NIE = E[Y(1, M(1)) - Y(1, M(0))]
    Effect of treatment operating through mediator

Decomposition:
    TE = NDE + NIE

Identification requires Sequential Ignorability (Imai et al. 2010):
1. {Y(t,m), M(t)} ⊥ T | X  (treatment ignorability)
2. Y(t,m) ⊥ M | T, X       (mediator ignorability - stronger!)

References
----------
- Baron & Kenny (1986). The Moderator-Mediator Variable Distinction
- Pearl (2001). Direct and Indirect Effects
- Imai, Keele, Yamamoto (2010). A General Approach to Causal Mediation
- VanderWeele (2015). Explanation in Causal Inference

Examples
--------
>>> import numpy as np
>>> from causal_inference.mediation import mediation_analysis, baron_kenny
>>>
>>> # Generate simple mediation data
>>> np.random.seed(42)
>>> n = 500
>>> T = np.random.binomial(1, 0.5, n)
>>> M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5  # T -> M
>>> Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5  # T,M -> Y
>>>
>>> # Baron-Kenny analysis
>>> result = baron_kenny(Y, T, M)
>>> print(f"Direct effect: {result['direct_effect']:.3f}")
>>> print(f"Indirect effect: {result['indirect_effect']:.3f}")
>>> print(f"Total effect: {result['total_effect']:.3f}")

Session 92 Implementation.
"""

from .estimators import (
    baron_kenny,
    controlled_direct_effect,
    mediation_analysis,
    natural_direct_effect,
    natural_indirect_effect,
)
from .sensitivity import mediation_sensitivity
from .types import (
    BaronKennyResult,
    CDEResult,
    MediationDiagnostics,
    MediationResult,
    SensitivityResult,
)

__all__ = [
    # Main functions
    "mediation_analysis",
    "baron_kenny",
    "natural_direct_effect",
    "natural_indirect_effect",
    "controlled_direct_effect",
    "mediation_sensitivity",
    # Types
    "MediationResult",
    "BaronKennyResult",
    "SensitivityResult",
    "CDEResult",
    "MediationDiagnostics",
]
