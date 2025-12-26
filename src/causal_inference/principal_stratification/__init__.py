"""Principal Stratification methods for causal inference.

This package implements principal stratification methods for addressing
post-treatment confounding by stratifying on joint potential values of
an intermediate variable under all treatment conditions.

Key Estimands
-------------
- **CACE** (Complier Average Causal Effect): E[Y(1) - Y(0) | D(0)=0, D(1)=1]
  The treatment effect for units who comply with their assignment.

- **SACE** (Survivor Average Causal Effect): E[Y(1) - Y(0) | S(0)=1, S(1)=1]
  The treatment effect for units who survive under both conditions.

Principal Strata
----------------
Under binary treatment and instrument, four strata are possible:

| Stratum | D(0) | D(1) | Definition |
|---------|------|------|------------|
| Compliers | 0 | 1 | Take treatment iff assigned |
| Always-takers | 1 | 1 | Always take treatment |
| Never-takers | 0 | 0 | Never take treatment |
| Defiers | 1 | 0 | Do opposite of assignment |

Under monotonicity (D(1) >= D(0)), defiers are ruled out.

Key Result
----------
Under IV assumptions (independence, exclusion, monotonicity, relevance):

    CACE = LATE = Reduced Form / First Stage

This is estimated via 2SLS or the Wald estimator.

Modules
-------
cace : CACE estimation (2SLS, Wald, EM)
sace : SACE bounds and sensitivity analysis
bounds : Partial identification bounds
bayesian : Bayesian principal stratification (PyMC)
diagnostics : Assumption testing

References
----------
- Frangakis, C. E., & Rubin, D. B. (2002). Principal Stratification in Causal Inference.
  Biometrics, 58(1), 21-29.
- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of Causal Effects
  Using Instrumental Variables. JASA, 91(434), 444-455.
- Imbens, G. W., & Rubin, D. B. (2015). Causal Inference for Statistics, Social, and
  Biomedical Sciences: An Introduction. Cambridge University Press.

Examples
--------
>>> import numpy as np
>>> from src.causal_inference.principal_stratification import cace_2sls
>>> # Generate data with 70% compliers
>>> np.random.seed(42)
>>> n = 1000
>>> Z = np.random.binomial(1, 0.5, n)  # Random assignment
>>> strata = np.random.choice([0, 1, 2], n, p=[0.70, 0.15, 0.15])
>>> D = np.where(strata == 0, Z, np.where(strata == 1, 1, 0))
>>> Y = 1.0 + 2.0 * D + np.random.normal(0, 1, n)
>>> # Estimate CACE
>>> result = cace_2sls(Y, D, Z)
>>> print(f"CACE: {result['cace']:.3f} (true: 2.0)")
>>> print(f"Complier proportion: {result['strata_proportions']['compliers']:.2%}")
"""

from .types import (
    CACEResult,
    SACEResult,
    StrataProportions,
    BoundsResult,
    BayesianPSResult,
    MonotonicityTestResult,
)
from .cace import cace_2sls, wald_estimator, cace_em
from .bounds import ps_bounds_monotonicity, ps_bounds_no_assumption, ps_bounds_balke_pearl
from .sace import sace_bounds, sace_sensitivity

# Lazy import for cace_bayesian (requires optional PyMC dependency)
def cace_bayesian(*args, **kwargs):
    """Bayesian CACE estimation (lazy import).

    See bayesian.cace_bayesian for full documentation.
    Requires: pip install 'causal-inference-mastery[bayesian]'
    """
    from .bayesian import cace_bayesian as _cace_bayesian

    return _cace_bayesian(*args, **kwargs)


__all__ = [
    # CACE Estimators
    "cace_2sls",
    "wald_estimator",
    "cace_em",
    "cace_bayesian",
    # Bounds
    "ps_bounds_monotonicity",
    "ps_bounds_no_assumption",
    "ps_bounds_balke_pearl",
    # SACE
    "sace_bounds",
    "sace_sensitivity",
    # Types
    "CACEResult",
    "SACEResult",
    "StrataProportions",
    "BoundsResult",
    "BayesianPSResult",
    "MonotonicityTestResult",
]
