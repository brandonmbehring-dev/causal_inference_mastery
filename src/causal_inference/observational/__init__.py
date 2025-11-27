"""Observational causal inference methods.

This package implements methods for estimating causal effects from observational
data where treatment assignment is not randomized.

Modules
-------
propensity : Propensity score estimation and diagnostics
ipw : Inverse probability weighting for observational data
"""

from src.causal_inference.observational.propensity import (
    estimate_propensity,
    trim_propensity,
    stabilize_weights,
)
from src.causal_inference.observational.ipw import ipw_ate_observational

__all__ = [
    "estimate_propensity",
    "trim_propensity",
    "stabilize_weights",
    "ipw_ate_observational",
]
