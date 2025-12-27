"""CATE (Conditional Average Treatment Effect) estimation module.

This module implements meta-learners for heterogeneous treatment effect estimation:
- S-Learner: Single model approach (simple, biased toward 0)
- T-Learner: Two separate models approach (intuitive, extrapolation issues)
- X-Learner: Cross-learner with propensity weighting (handles imbalanced groups)
- R-Learner: Robinson transformation (doubly robust, orthogonal)
- Double ML: Cross-fitted Robinson transformation (eliminates regularization bias)
- Causal Forest: Honest random forests for nonlinear heterogeneity
- DragonNet: Neural network with shared representation (propensity + outcome)
- TEDVAE: Disentangled VAE for treatment effects (instrumental/confounding/risk factors)
- Neural Meta-Learners: Neural network versions of S/T/X/R-learners
- Neural Double ML: Cross-fitted DML with neural networks

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
- Shi et al. (2019). "Adapting Neural Networks for the Estimation of Treatment
  Effects." NeurIPS 32.
- Zhang et al. (2021). "Treatment Effect Estimation with Disentangled Latent
  Factors." AAAI 2021.
"""

from .base import CATEResult, OMLResult
from .meta_learners import s_learner, t_learner, x_learner, r_learner
from .dml import double_ml
from .oml import irm_dml
from .dml_continuous import dml_continuous, DMLContinuousResult
from .causal_forest import causal_forest
from .dragonnet import dragonnet
from .ganite import ganite
from .tedvae import tedvae
from .neural_meta_learners import (
    neural_s_learner,
    neural_t_learner,
    neural_x_learner,
    neural_r_learner,
)
from .neural_dml import neural_double_ml
from .latent_cate import factor_analysis_cate, ppca_cate, gmm_stratified_cate

__all__ = [
    "CATEResult",
    "OMLResult",
    # Linear meta-learners
    "s_learner",
    "t_learner",
    "x_learner",
    "r_learner",
    "double_ml",
    # OML / IRM
    "irm_dml",
    "dml_continuous",
    "DMLContinuousResult",
    # Tree-based
    "causal_forest",
    # Neural methods
    "dragonnet",
    "ganite",
    "tedvae",
    "neural_s_learner",
    "neural_t_learner",
    "neural_x_learner",
    "neural_r_learner",
    "neural_double_ml",
    # Latent methods
    "factor_analysis_cate",
    "ppca_cate",
    "gmm_stratified_cate",
]
