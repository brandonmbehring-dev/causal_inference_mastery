"""Benchmarks for Regression Discontinuity Design methods.

Methods benchmarked:
- sharp_rdd: Sharp RDD with local polynomial regression
- fuzzy_rdd: Fuzzy RDD (2SLS at cutoff)
- mccrary: McCrary density test for manipulation
"""

from __future__ import annotations

from benchmarks.dgp import generate_rdd_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_sharp_rdd(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark SharpRDD.

    Sharp RDD uses local polynomial regression which scales
    with bandwidth × sample size.

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
    from src.causal_inference.rdd import SharpRDD

    data = generate_rdd_data(n=n, seed=seed, design="sharp")

    def run_sharp_rdd():
        model = SharpRDD(cutoff=data["cutoff"])
        return model.fit(
            Y=data["outcome"],
            X=data["running_variable"],
        )

    return benchmark_method(
        func=run_sharp_rdd,
        method_name="sharp_rdd",
        family="rdd",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_fuzzy_rdd(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark FuzzyRDD.

    Fuzzy RDD adds 2SLS estimation on top of local polynomial,
    making it slightly more expensive than Sharp RDD.

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
    from src.causal_inference.rdd import FuzzyRDD

    data = generate_rdd_data(n=n, seed=seed, design="fuzzy")

    def run_fuzzy_rdd():
        model = FuzzyRDD(cutoff=data["cutoff"])
        return model.fit(
            Y=data["outcome"],
            X=data["running_variable"],
            D=data["treatment"],
        )

    return benchmark_method(
        func=run_fuzzy_rdd,
        method_name="fuzzy_rdd",
        family="rdd",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_mccrary(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark mccrary_density_test.

    McCrary test involves kernel density estimation,
    scaling with n × bandwidth_grid_size.

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
    from src.causal_inference.rdd import mccrary_density_test

    data = generate_rdd_data(n=n, seed=seed, design="sharp")

    return benchmark_method(
        func=mccrary_density_test,
        method_name="mccrary",
        family="rdd",
        sample_size=n,
        X=data["running_variable"],
        cutoff=data["cutoff"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "sharp_rdd": benchmark_sharp_rdd,
    "fuzzy_rdd": benchmark_fuzzy_rdd,
    "mccrary": benchmark_mccrary,
}
