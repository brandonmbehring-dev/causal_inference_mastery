"""Benchmarks for Partial Identification / Bounds Methods.

Methods benchmarked:
- manski_worst_case: Manski worst-case bounds
- manski_mtr: Bounds with monotone treatment response
- lee_bounds: Lee (2009) trimming bounds
"""

from __future__ import annotations

from benchmarks.dgp import generate_bounds_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_manski_worst_case(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark Manski worst-case bounds.

    No-assumptions bounds using outcome support.
    O(n) computation.

    Parameters
    ----------
    n : int
        Sample size.
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
    from src.causal_inference.bounds import manski_worst_case

    data = generate_bounds_data(n=n, seed=seed)

    # Get observed outcomes only (filter NaN)
    import numpy as np
    mask = ~np.isnan(data["outcome"])

    return benchmark_method(
        func=manski_worst_case,
        method_name="manski_worst_case",
        family="bounds",
        sample_size=mask.sum(),
        outcome=data["outcome"][mask],
        treatment=data["treatment"][mask],
        y_min=-5.0,
        y_max=10.0,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_manski_mtr(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark Manski MTR bounds.

    Bounds under monotone treatment response assumption.
    Tighter than worst-case.

    Parameters
    ----------
    n : int
        Sample size.
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
    from src.causal_inference.bounds import manski_mtr

    data = generate_bounds_data(n=n, seed=seed)

    import numpy as np
    mask = ~np.isnan(data["outcome"])

    return benchmark_method(
        func=manski_mtr,
        method_name="manski_mtr",
        family="bounds",
        sample_size=mask.sum(),
        outcome=data["outcome"][mask],
        treatment=data["treatment"][mask],
        y_min=-5.0,
        y_max=10.0,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_lee_bounds(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark Lee (2009) trimming bounds.

    Bounds for selection on observables with sample selection.
    Involves quantile estimation and trimming.

    Parameters
    ----------
    n : int
        Sample size.
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
    from src.causal_inference.bounds import lee_bounds

    data = generate_bounds_data(n=n, seed=seed, missing_rate=0.2)

    return benchmark_method(
        func=lee_bounds,
        method_name="lee_bounds",
        family="bounds",
        sample_size=n,
        outcome=data["outcome"],
        treatment=data["treatment"],
        selection=data["observed"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_lee_bounds_tightened(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark Lee bounds with covariate tightening.

    Uses covariates to tighten bounds via stratification.
    More computation but narrower intervals.

    Parameters
    ----------
    n : int
        Sample size.
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
    from src.causal_inference.bounds import lee_bounds_tightened
    import numpy as np

    data = generate_bounds_data(n=n, seed=seed, missing_rate=0.2)

    # Add covariates for tightening
    rng = np.random.default_rng(seed)
    covariates = rng.normal(0, 1, (n, 3))

    return benchmark_method(
        func=lee_bounds_tightened,
        method_name="lee_bounds_tightened",
        family="bounds",
        sample_size=n,
        outcome=data["outcome"],
        treatment=data["treatment"],
        selection=data["observed"],
        covariates=covariates,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "manski_worst_case": benchmark_manski_worst_case,
    "manski_mtr": benchmark_manski_mtr,
    "lee_bounds": benchmark_lee_bounds,
    "lee_bounds_tightened": benchmark_lee_bounds_tightened,
}
