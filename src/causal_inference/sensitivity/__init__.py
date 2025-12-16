"""Sensitivity analysis for unmeasured confounding.

This module provides tools to assess how robust causal conclusions are
to potential violations of the unconfoundedness assumption (no unmeasured
confounders).

Two complementary approaches are provided:

1. **E-value** (VanderWeele & Ding, 2017):
   - Universal metric for any observational estimate
   - Answers: "How strong would confounding need to be to explain this away?"
   - Single interpretable number

2. **Rosenbaum Bounds** (1987, 2002):
   - Specialized for matched-pair studies (PSM)
   - Answers: "How much selection bias could overturn our conclusion?"
   - Finds critical Gamma where significance disappears

Examples
--------
>>> from causal_inference.sensitivity import e_value, rosenbaum_bounds
>>> import numpy as np

>>> # E-value for a risk ratio of 2.0
>>> result = e_value(2.0, ci_lower=1.5, ci_upper=2.7, effect_type="rr")
>>> print(f"E-value: {result['e_value']:.2f}")
E-value: 3.41

>>> # Rosenbaum bounds for matched pairs
>>> np.random.seed(42)
>>> treated = np.random.randn(50) + 1.5  # Effect of 1.5
>>> control = np.random.randn(50)
>>> result = rosenbaum_bounds(treated, control)
>>> print(f"Critical Gamma: {result['gamma_critical']:.2f}")

References
----------
- VanderWeele TJ, Ding P (2017). "Sensitivity Analysis in Observational Research:
  Introducing the E-Value." Annals of Internal Medicine.
- Rosenbaum PR (2002). "Observational Studies" (2nd ed.). Springer.
- Rosenbaum PR (1987). "Sensitivity Analysis for Certain Permutation Inferences
  in Matched Observational Studies." Biometrika.
"""

from .types import EValueResult, RosenbaumResult
from .e_value import e_value
from .rosenbaum import rosenbaum_bounds

__all__ = [
    "EValueResult",
    "RosenbaumResult",
    "e_value",
    "rosenbaum_bounds",
]
