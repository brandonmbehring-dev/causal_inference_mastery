"""Benchmarks for Bunching Estimator Methods.

Methods benchmarked:
- bunching_estimator: Saez (2010) excess mass bunching estimator
"""

from __future__ import annotations

from benchmarks.dgp import generate_bunching_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_bunching_estimator(
    n: int = 5000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
    polynomial_order: int = 7,
) -> BenchmarkResult:
    """Benchmark bunching estimator.

    Estimates behavioral response from bunching at thresholds.
    Involves polynomial fitting and counterfactual estimation.

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
    polynomial_order : int
        Order of polynomial for counterfactual.

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.bunching import bunching_estimator

    data = generate_bunching_data(n=n, seed=seed)

    def run_bunching():
        return bunching_estimator(
            values=data["observed_value"],
            threshold=data["threshold"],
            polynomial_order=polynomial_order,
        )

    return benchmark_method(
        func=run_bunching,
        method_name="bunching_estimator",
        family="bunching",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "bunching_estimator": benchmark_bunching_estimator,
}
