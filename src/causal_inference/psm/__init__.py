"""
Propensity Score Matching estimators.

Implements PSM for estimating average treatment effects in observational studies.

Main Components:
- PropensityScoreEstimator: Estimates P(T=1|X) via logistic regression
- NearestNeighborMatcher: Greedy nearest neighbor matching algorithm
- AbadieImbensVariance: Analytic variance accounting for matching uncertainty
- BalanceDiagnostics: SMD and variance ratio calculations
- psm_ate: High-level PSM estimation function

References:
- Rosenbaum & Rubin (1983): The central role of the propensity score
- Abadie & Imbens (2006): Large sample properties of matching estimators
- Abadie & Imbens (2008): On the failure of the bootstrap for matching estimators
- Austin (2009): Balance diagnostics for PSM
"""

__all__ = [
    "PropensityScoreEstimator",
    "NearestNeighborMatcher",
    "abadie_imbens_variance",
    "psm_ate",
    # "BalanceDiagnostics",      # Session 3
]

from .propensity import PropensityScoreEstimator
from .matching import NearestNeighborMatcher
from .variance import abadie_imbens_variance
from .psm_estimator import psm_ate

__version__ = "0.2.0"
