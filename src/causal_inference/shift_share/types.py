"""
Type definitions for Shift-Share/Bartik IV estimation.

The shift-share instrument (Bartik 1991) constructs an IV from:
- Shares: Local exposure to sectors (share_{i,s})
- Shifts: Aggregate shocks to sectors (growth_{s,t})
- Instrument: Z_i = sum_s(share_{i,s} * growth_{s,t})

References
----------
- Bartik (1991). Who Benefits from State and Local Economic Development Policies?
- Goldsmith-Pinkham, Sorkin, Swift (2020). Bartik Instruments: What, When, Why, and How
- Borusyak, Hull, Jaravel (2022). Quasi-Experimental Shift-Share Research Designs
"""

from typing import Optional, TypedDict

import numpy as np
from numpy.typing import NDArray


class RotembergDiagnostics(TypedDict):
    """
    Rotemberg (1983) weight diagnostics for shift-share instruments.

    The Rotemberg weights decompose the overall IV estimate into
    contributions from each sector/shock. Negative weights indicate
    potential violations of monotonicity.

    Attributes
    ----------
    weights : NDArray[np.float64]
        Rotemberg weight for each sector (sums to 1).
    negative_weight_share : float
        Fraction of total weight that is negative (0-1).
    top_5_sectors : NDArray[np.int64]
        Indices of 5 sectors with largest absolute weights.
    top_5_weights : NDArray[np.float64]
        Weights for the top 5 sectors.
    herfindahl : float
        Herfindahl index of weights (concentration measure).
    """

    weights: NDArray[np.float64]
    negative_weight_share: float
    top_5_sectors: NDArray[np.int64]
    top_5_weights: NDArray[np.float64]
    herfindahl: float


class FirstStageResult(TypedDict):
    """First-stage regression results."""

    f_statistic: float
    f_pvalue: float
    partial_r2: float
    coefficient: float
    se: float
    t_stat: float
    weak_iv_warning: bool


class ShiftShareResult(TypedDict):
    """
    Result from Shift-Share IV estimation.

    Attributes
    ----------
    estimate : float
        Estimated causal effect (2SLS coefficient on D).
    se : float
        Standard error (robust or clustered).
    t_stat : float
        T-statistic for estimate.
    p_value : float
        Two-sided p-value.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    first_stage : FirstStageResult
        First-stage regression diagnostics.
    rotemberg : RotembergDiagnostics
        Rotemberg weight diagnostics.
    n_obs : int
        Number of observations.
    n_sectors : int
        Number of sectors/shocks.
    share_sum_mean : float
        Mean of share row sums (should be ~1 if shares normalized).
    inference : str
        Inference method used ('robust', 'clustered').
    alpha : float
        Significance level used.
    message : str
        Diagnostic message.
    """

    estimate: float
    se: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    first_stage: FirstStageResult
    rotemberg: RotembergDiagnostics
    n_obs: int
    n_sectors: int
    share_sum_mean: float
    inference: str
    alpha: float
    message: str
