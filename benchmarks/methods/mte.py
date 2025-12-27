"""Benchmarks for Marginal Treatment Effects (MTE).

Methods benchmarked:
- local_iv: Local IV / MTE estimation
"""

from __future__ import annotations

from benchmarks.dgp import generate_mte_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_local_iv(
    n: int = 2000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark Local IV / MTE estimation.

    Estimates marginal treatment effect curve.
    Involves local polynomial regression on propensity score.

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
    from src.causal_inference.mte import local_iv

    data = generate_mte_data(n=n, seed=seed)

    def run_local_iv():
        return local_iv(
            outcome=data["outcome"],
            treatment=data["treatment"],
            instrument=data["instrument"],
            covariates=data["covariates"],
        )

    return benchmark_method(
        func=run_local_iv,
        method_name="local_iv",
        family="mte",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "local_iv": benchmark_local_iv,
}
