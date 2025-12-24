"""
Regression Discontinuity Design (RDD) Module

Implements Sharp and Fuzzy RDD estimators with optimal bandwidth selection,
diagnostics, and robust inference.

Classes
-------
SharpRDD : Sharp regression discontinuity estimator

Functions
---------
imbens_kalyanaraman_bandwidth : IK optimal bandwidth selector
cct_bandwidth : CCT-style approximation (uses IK with 1.5× bias bandwidth)
cross_validation_bandwidth : CV bandwidth selector

Examples
--------
>>> from causal_inference.rdd import SharpRDD
>>> import numpy as np
>>>
>>> # Generate RDD data
>>> np.random.seed(42)
>>> X = np.random.uniform(-5, 5, 500)
>>> Y = X + 2 * (X >= 0) + np.random.normal(0, 1, 500)
>>>
>>> # Fit Sharp RDD
>>> rdd = SharpRDD(cutoff=0.0, bandwidth='ik')
>>> rdd.fit(Y, X)
>>> print(f"Treatment effect: {rdd.coef_:.3f}")
Treatment effect: 2.000

See Also
--------
causal_inference.rct : Randomized controlled trial estimators
causal_inference.psm : Propensity score matching
causal_inference.did : Difference-in-differences
causal_inference.iv : Instrumental variables

References
----------
Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to practice.
    Journal of Econometrics, 142(2), 615-635.
Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence
    intervals for regression-discontinuity designs. Econometrica, 82(6), 2295-2326.
Lee, D. S., & Lemieux, T. (2010). Regression discontinuity designs in economics.
    Journal of Economic Literature, 48(2), 281-355.
"""

from .sharp_rdd import SharpRDD
from .fuzzy_rdd import FuzzyRDD
from .bandwidth import (
    imbens_kalyanaraman_bandwidth,
    cct_bandwidth,
    cross_validation_bandwidth,
)
from .rdd_diagnostics import (
    mccrary_density_test,
    covariate_balance_test,
    bandwidth_sensitivity_analysis,
    polynomial_order_sensitivity,
    donut_hole_rdd,
)

__all__ = [
    "SharpRDD",
    "FuzzyRDD",
    "imbens_kalyanaraman_bandwidth",
    "cct_bandwidth",
    "cross_validation_bandwidth",
    "mccrary_density_test",
    "covariate_balance_test",
    "bandwidth_sensitivity_analysis",
    "polynomial_order_sensitivity",
    "donut_hole_rdd",
]

__version__ = "0.3.0"
