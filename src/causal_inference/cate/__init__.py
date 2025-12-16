"""CATE (Conditional Average Treatment Effect) estimation module.

This module implements meta-learners for heterogeneous treatment effect estimation:
- S-Learner: Single model approach (simple, biased toward 0)
- T-Learner: Two separate models approach (intuitive, extrapolation issues)
- X-Learner: Cross-learner with propensity weighting (handles imbalanced groups)
- R-Learner: Robinson transformation (doubly robust, orthogonal)

References
----------
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects
  using machine learning." PNAS 116(10): 4156-4165.
- Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects."
  Biometrika 108(2): 299-319.
"""

from .base import CATEResult
from .meta_learners import s_learner, t_learner, x_learner, r_learner

__all__ = [
    "CATEResult",
    "s_learner",
    "t_learner",
    "x_learner",
    "r_learner",
]
