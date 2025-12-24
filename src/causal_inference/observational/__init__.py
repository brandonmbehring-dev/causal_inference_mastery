"""Observational causal inference methods.

This package implements methods for estimating causal effects from observational
data where treatment assignment is not randomized.

Modules
-------
propensity : Propensity score estimation and diagnostics
ipw : Inverse probability weighting for observational data
doubly_robust : Doubly robust (AIPW) estimation
tmle : Targeted Maximum Likelihood Estimation
"""

from src.causal_inference.observational.propensity import (
    estimate_propensity,
    trim_propensity,
    stabilize_weights,
)
from src.causal_inference.observational.ipw import ipw_ate_observational
from src.causal_inference.observational.doubly_robust import dr_ate
from src.causal_inference.observational.tmle import tmle_ate

__all__ = [
    "estimate_propensity",
    "trim_propensity",
    "stabilize_weights",
    "ipw_ate_observational",
    "dr_ate",
    "tmle_ate",
]
