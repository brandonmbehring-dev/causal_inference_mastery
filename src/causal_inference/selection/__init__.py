"""Selection bias correction methods.

This package implements methods for correcting sample selection bias,
where outcomes are only observed for a selected subsample.

Modules
-------
heckman : Heckman two-step selection model
types : TypedDicts for return types
diagnostics : Selection tests and diagnostics

References
----------
- Heckman, J. J. (1979). Sample Selection Bias as a Specification Error.
  Econometrica, 47(1), 153-161. doi:10.2307/1912352
- Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data.
  MIT Press, Chapter 19.
"""

from .heckman import heckman_two_step
from .types import HeckmanResult
from .diagnostics import (
    selection_bias_test,
    plot_imr_distribution,
    diagnose_identification,
)

__all__ = [
    "heckman_two_step",
    "HeckmanResult",
    "selection_bias_test",
    "plot_imr_distribution",
    "diagnose_identification",
]
