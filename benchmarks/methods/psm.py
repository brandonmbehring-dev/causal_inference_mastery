"""Benchmarks for Propensity Score Matching methods.

Methods benchmarked:
- psm_ate: Propensity score matching with various configurations
"""

from __future__ import annotations

from benchmarks.dgp import generate_psm_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_psm_ate(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
    n_neighbors: int = 1,
    with_replacement: bool = False,
) -> BenchmarkResult:
    """Benchmark psm_ate.

    PSM has O(n²) complexity for naive nearest neighbor search,
    making it scale poorly with sample size.

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
    n_neighbors : int
        Number of neighbors to match (default 1).
    with_replacement : bool
        Whether to match with replacement (default False).

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.psm import psm_ate

    data = generate_psm_data(n=n, seed=seed, overlap="good")

    return benchmark_method(
        func=psm_ate,
        method_name="psm_ate",
        family="psm",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
        M=n_neighbors,
        with_replacement=with_replacement,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_psm_ate_k3(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark psm_ate with k=3 neighbors."""
    return benchmark_psm_ate(
        n=n,
        seed=seed,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
        M=3,
        with_replacement=False,
    )


def benchmark_psm_ate_replacement(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark psm_ate with replacement."""
    return benchmark_psm_ate(
        n=n,
        seed=seed,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
        M=1,
        with_replacement=True,
    )


BENCHMARKS = {
    "psm_ate": benchmark_psm_ate,
    "psm_ate_k3": benchmark_psm_ate_k3,
    "psm_ate_replacement": benchmark_psm_ate_replacement,
}
