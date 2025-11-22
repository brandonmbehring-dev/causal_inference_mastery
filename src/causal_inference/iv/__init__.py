"""
Instrumental Variables (IV) Module.

This module provides production-ready instrumental variables estimators for handling
endogeneity bias in causal inference. Implements 2SLS, LIML, GMM, and weak instrument
diagnostics.

Key Features:
    - Two-Stage Least Squares (2SLS) with correct standard errors
    - Weak instrument diagnostics (F-statistic, Stock-Yogo, Anderson-Rubin)
    - Multiple estimators (2SLS, LIML, GMM, Fuller)
    - Three inference methods (standard, robust, clustered)
    - Comprehensive input validation

Quick Start:
    >>> from causal_inference.iv import TwoStageLeastSquares
    >>>
    >>> # Fit 2SLS with robust standard errors
    >>> iv = TwoStageLeastSquares(inference='robust')
    >>> iv.fit(Y, D, Z, X)
    >>>
    >>> # Check results
    >>> print(iv.summary())
    >>> print(f"First-stage F-statistic: {iv.first_stage_f_stat_:.2f}")

References:
    - Angrist & Pischke (2009). Mostly Harmless Econometrics, Chapter 4
    - Stock & Yogo (2005). Testing for weak instruments
    - Wooldridge (2010). Econometric Analysis of Cross Section and Panel Data
"""

from .two_stage_least_squares import TwoStageLeastSquares
from .stages import FirstStage, ReducedForm, SecondStage
from .diagnostics import (
    classify_instrument_strength,
    cragg_donald_statistic,
    anderson_rubin_test,
    weak_instrument_summary,
    STOCK_YOGO_CRITICAL_VALUES,
)

__all__ = [
    "TwoStageLeastSquares",
    "FirstStage",
    "ReducedForm",
    "SecondStage",
    "classify_instrument_strength",
    "cragg_donald_statistic",
    "anderson_rubin_test",
    "weak_instrument_summary",
    "STOCK_YOGO_CRITICAL_VALUES",
]

__version__ = "0.1.0"
