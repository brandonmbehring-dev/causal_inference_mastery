"""CATE (Conditional Average Treatment Effect) estimation module.

This module implements meta-learners for heterogeneous treatment effect estimation:
- S-Learner: Single model approach (simple, biased toward 0)
- T-Learner: Two separate models approach (intuitive, extrapolation issues)
- X-Learner: Cross-learner with propensity weighting (handles imbalanced groups)
- R-Learner: Robinson transformation (doubly robust, orthogonal)
- Double ML: Cross-fitted Robinson transformation (eliminates regularization bias)
- Causal Forest: Honest random forests for nonlinear heterogeneity

References
----------
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects
  using machine learning." PNAS 116(10): 4156-4165.
- Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects."
  Biometrika 108(2): 299-319.
- Chernozhukov et al. (2018). "Double/debiased machine learning for treatment
  and structural parameters." The Econometrics Journal 21(1): C1-C68.
- Wager & Athey (2018). "Estimation and inference of heterogeneous treatment
  effects using random forests." Annals of Statistics 46(3): 1228-1242.
"""

from .base import CATEResult
from .meta_learners import s_learner, t_learner, x_learner, r_learner
from .dml import double_ml
from .causal_forest import causal_forest

__all__ = [
    "CATEResult",
    "s_learner",
    "t_learner",
    "x_learner",
    "r_learner",
    "double_ml",
    "causal_forest",
]
