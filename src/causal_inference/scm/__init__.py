"""
Synthetic Control Methods (SCM) Module

Implements the synthetic control method (Abadie et al. 2003, 2010, 2015) for
comparative case studies with few treated units.

Main Functions
--------------
synthetic_control
    Core SCM estimator with placebo/bootstrap inference
augmented_synthetic_control
    Ben-Michael et al. (2021) augmented SCM with bias correction

Inference
---------
placebo_test_in_space
    In-space placebo test for p-values
placebo_test_in_time
    In-time placebo test
bootstrap_se
    Bootstrap standard error estimation

Diagnostics
-----------
check_pre_treatment_fit
    Assess pre-treatment fit quality
check_covariate_balance
    Check covariate balance
check_weight_properties
    Analyze weight concentration
diagnose_scm_quality
    Comprehensive quality diagnostics

Types
-----
SCMResult, ASCMResult
    TypedDicts containing estimation results

Example
-------
>>> import numpy as np
>>> from causal_inference.scm import synthetic_control
>>>
>>> # Panel data: 10 units, 20 periods
>>> outcomes = np.random.randn(10, 20)
>>> treatment = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
>>> treatment_period = 10
>>>
>>> result = synthetic_control(outcomes, treatment, treatment_period)
>>> print(f"ATT: {result['estimate']:.3f} (p={result['p_value']:.3f})")

References
----------
Abadie, A., & Gardeazabal, J. (2003). "The Economic Costs of Conflict:
    A Case Study of the Basque Country". American Economic Review.

Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods
    for Comparative Case Studies". Journal of the American Statistical Association.

Abadie, A., Diamond, A., & Hainmueller, J. (2015). "Comparative Politics and
    the Synthetic Control Method". American Journal of Political Science.

Ben-Michael, E., Feller, A., & Rothstein, J. (2021). "The Augmented Synthetic
    Control Method". Journal of the American Statistical Association.
"""

# Core estimators
from .basic_scm import synthetic_control
from .augmented_scm import augmented_synthetic_control, ASCMResult

# Types
from .types import SCMResult, validate_panel_data

# Weights
from .weights import compute_scm_weights, compute_pre_treatment_fit

# Inference
from .inference import (
    placebo_test_in_space,
    placebo_test_in_time,
    bootstrap_se,
    compute_confidence_interval,
    compute_p_value,
)

# Diagnostics
from .diagnostics import (
    check_pre_treatment_fit,
    check_covariate_balance,
    check_weight_properties,
    diagnose_scm_quality,
    compute_rmspe_ratio,
)

__all__ = [
    # Main estimators
    "synthetic_control",
    "augmented_synthetic_control",
    # Types
    "SCMResult",
    "ASCMResult",
    # Validation
    "validate_panel_data",
    # Weight utilities
    "compute_scm_weights",
    "compute_pre_treatment_fit",
    # Inference
    "placebo_test_in_space",
    "placebo_test_in_time",
    "bootstrap_se",
    "compute_confidence_interval",
    "compute_p_value",
    # Diagnostics
    "check_pre_treatment_fit",
    "check_covariate_balance",
    "check_weight_properties",
    "diagnose_scm_quality",
    "compute_rmspe_ratio",
]
