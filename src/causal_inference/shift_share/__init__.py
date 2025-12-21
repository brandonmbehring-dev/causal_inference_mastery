"""
Shift-Share (Bartik) Instrumental Variables estimation.

The shift-share instrument (Bartik 1991) combines:
1. Local sector shares (exposure)
2. Aggregate sector shocks (shifts)

Instrument: Z_i = sum_s(share_{i,s} * shock_s)

Main Components
---------------
ShiftShareIV : class
    Main estimator with Rotemberg diagnostics.
shift_share_iv : function
    Convenience function for quick estimation.

Types
-----
ShiftShareResult : TypedDict
    Full estimation results.
RotembergDiagnostics : TypedDict
    Sector contribution weights.
FirstStageResult : TypedDict
    First-stage regression diagnostics.

References
----------
- Bartik (1991). Who Benefits from State and Local Economic Development Policies?
- Goldsmith-Pinkham, Sorkin, Swift (2020). Bartik Instruments
- Borusyak, Hull, Jaravel (2022). Quasi-Experimental Shift-Share Designs

Examples
--------
>>> import numpy as np
>>> from causal_inference.shift_share import shift_share_iv
>>>
>>> # Regional employment example
>>> n_regions, n_sectors = 100, 10
>>> shares = np.random.dirichlet(np.ones(n_sectors), n_regions)
>>> shocks = np.random.normal(0.02, 0.05, n_sectors)
>>>
>>> # Endogenous treatment and outcome
>>> Z_bartik = shares @ shocks
>>> D = 2.0 * Z_bartik + np.random.normal(0, 0.5, n_regions)
>>> Y = 1.5 * D + np.random.normal(0, 1, n_regions)
>>>
>>> result = shift_share_iv(Y, D, shares, shocks)
>>> print(f"Effect: {result['estimate']:.3f} (SE: {result['se']:.3f})")
"""

from .shift_share import ShiftShareIV, shift_share_iv
from .types import FirstStageResult, RotembergDiagnostics, ShiftShareResult

__all__ = [
    # Main estimator
    "ShiftShareIV",
    "shift_share_iv",
    # Types
    "ShiftShareResult",
    "RotembergDiagnostics",
    "FirstStageResult",
]
