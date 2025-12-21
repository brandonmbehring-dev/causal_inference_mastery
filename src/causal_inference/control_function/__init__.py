"""
Control Function estimation for endogenous treatment effects.

The control function approach is an alternative to 2SLS that:
1. Explicitly estimates the correlation between treatment and errors
2. Provides a built-in test for endogeneity
3. Extends naturally to nonlinear models (probit/logit)

Main Components
---------------
ControlFunction : class
    Main estimator class with fit/test_endogeneity/summary methods.
control_function_ate : function
    Convenience function for quick estimation.

Types
-----
ControlFunctionResult : TypedDict
    Full estimation results including treatment effect and endogeneity test.
FirstStageResult : TypedDict
    First-stage regression diagnostics (F-stat, partial R², residuals).
NonlinearCFResult : TypedDict
    Results from probit/logit control function models.

References
----------
- Wooldridge (2015). "Control Function Methods in Applied Econometrics"
- Murphy & Topel (1985). "Estimation and Inference in Two-Step Models"
- Rivers & Vuong (1988). "Limited Information Estimators for Probit Models"

Examples
--------
>>> import numpy as np
>>> from causal_inference.control_function import control_function_ate
>>>
>>> # Generate data with endogeneity
>>> np.random.seed(42)
>>> n = 1000
>>> Z = np.random.normal(0, 1, n)  # Instrument
>>> nu = np.random.normal(0, 1, n)  # Shared error
>>> D = 0.5 * Z + nu  # Endogenous treatment
>>> epsilon = 0.7 * nu + 0.3 * np.random.normal(0, 1, n)
>>> Y = 2.0 * D + epsilon  # Outcome
>>>
>>> # Estimate with control function
>>> result = control_function_ate(Y, D, Z, n_bootstrap=500)
>>> print(f"Treatment effect: {result['estimate']:.3f} (SE: {result['se']:.3f})")
>>> print(f"Endogeneity detected: {result['endogeneity_detected']}")
"""

from .control_function import ControlFunction, control_function_ate
from .nonlinear import NonlinearControlFunction, nonlinear_control_function
from .types import ControlFunctionResult, FirstStageResult, NonlinearCFResult

__all__ = [
    # Linear Control Function
    "ControlFunction",
    "control_function_ate",
    # Nonlinear Control Function (Probit/Logit)
    "NonlinearControlFunction",
    "nonlinear_control_function",
    # Types
    "ControlFunctionResult",
    "FirstStageResult",
    "NonlinearCFResult",
]
