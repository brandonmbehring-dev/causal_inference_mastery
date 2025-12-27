"""Benchmarks for Shift-Share IV Methods.

Methods benchmarked:
- shift_share_iv: Bartik/shift-share instrumental variables
"""

from __future__ import annotations

import numpy as np

from benchmarks.utils import BenchmarkResult, benchmark_method


def generate_shift_share_data(
    n_regions: int = 100,
    n_industries: int = 20,
    n_periods: int = 10,
    seed: int = 42,
) -> dict:
    """Generate shift-share data.

    Model:
        National shift: g_jt ~ N(0.02, 0.05)
        Regional share: s_ij ~ Dirichlet
        Bartik: B_it = sum_j(s_ij * g_jt)
        Outcome: Y_it = alpha + beta * X_it + epsilon

    Parameters
    ----------
    n_regions : int
        Number of regions.
    n_industries : int
        Number of industries.
    n_periods : int
        Number of time periods.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Shift-share data.
    """
    rng = np.random.default_rng(seed)

    # Industry shares by region (sum to 1)
    shares = rng.dirichlet(np.ones(n_industries), size=n_regions)

    # National growth shocks by industry-period
    national_shocks = rng.normal(0.02, 0.05, (n_industries, n_periods))

    # Construct Bartik instrument
    bartik = shares @ national_shocks  # (n_regions, n_periods)

    # Endogenous variable (correlated with unobserved)
    U = rng.normal(0, 1, (n_regions, n_periods))
    X = 0.5 * bartik + 0.5 * U + rng.normal(0, 0.3, (n_regions, n_periods))

    # Outcome
    true_effect = 2.0
    Y = 1.0 + true_effect * X + 0.5 * U + rng.normal(0, 1, (n_regions, n_periods))

    return {
        "outcome": Y.flatten(),
        "endogenous": X.flatten(),
        "shares": shares,
        "national_shocks": national_shocks,
        "bartik": bartik.flatten(),
        "region_id": np.repeat(np.arange(n_regions), n_periods),
        "period_id": np.tile(np.arange(n_periods), n_regions),
        "true_effect": true_effect,
    }


def benchmark_shift_share_iv(
    n_regions: int = 100,
    n_industries: int = 20,
    n_periods: int = 10,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark shift-share IV estimator.

    Bartik instrument approach for regional studies.
    Involves computing predicted exposure and 2SLS.

    Parameters
    ----------
    n_regions : int
        Number of regions.
    n_industries : int
        Number of industries.
    n_periods : int
        Number of time periods.
    seed : int
        Random seed.
    n_repetitions : int
        Number of timing repetitions.
    n_warmup : int
        Number of warmup runs.

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.shift_share import shift_share_iv

    data = generate_shift_share_data(
        n_regions=n_regions,
        n_industries=n_industries,
        n_periods=n_periods,
        seed=seed,
    )

    def run_shift_share():
        return shift_share_iv(
            outcome=data["outcome"],
            endogenous=data["endogenous"],
            shares=data["shares"],
            shocks=data["national_shocks"],
            cluster_id=data["region_id"],
        )

    return benchmark_method(
        func=run_shift_share,
        method_name="shift_share_iv",
        family="shift_share",
        sample_size=n_regions * n_periods,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "shift_share_iv": benchmark_shift_share_iv,
}
