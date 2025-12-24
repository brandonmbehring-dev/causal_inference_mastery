"""
Bayesian Causal Inference Module.

Provides Bayesian estimation of causal effects using conjugate priors
for closed-form posterior computation, as well as MCMC-based methods
for hierarchical models.

Session 101: Initial implementation with Bayesian ATE (conjugate priors).
Session 102: Added Bayesian propensity score estimation.
Session 103: Added Bayesian Doubly Robust estimation.
Session 104: Added Hierarchical Bayesian ATE with MCMC.
"""

from causal_inference.bayesian.types import (
    BayesianATEResult,
    BayesianPropensityResult,
    BayesianDRResult,
    HierarchicalATEResult,
    StratumInfo,
)
from causal_inference.bayesian.conjugate_ate import bayesian_ate
from causal_inference.bayesian.bayesian_propensity import (
    bayesian_propensity,
    bayesian_propensity_stratified,
    bayesian_propensity_logistic,
)


def __getattr__(name: str):
    """Lazy import for modules with optional dependencies."""
    if name == "bayesian_dr_ate":
        from causal_inference.bayesian.bayesian_dr import bayesian_dr_ate
        return bayesian_dr_ate
    if name == "hierarchical_bayesian_ate":
        from causal_inference.bayesian.hierarchical_ate import hierarchical_bayesian_ate
        return hierarchical_bayesian_ate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Session 101: ATE
    "BayesianATEResult",
    "bayesian_ate",
    # Session 102: Propensity
    "BayesianPropensityResult",
    "StratumInfo",
    "bayesian_propensity",
    "bayesian_propensity_stratified",
    "bayesian_propensity_logistic",
    # Session 103: Doubly Robust
    "BayesianDRResult",
    "bayesian_dr_ate",
    # Session 104: Hierarchical ATE (MCMC)
    "HierarchicalATEResult",
    "hierarchical_bayesian_ate",
]
