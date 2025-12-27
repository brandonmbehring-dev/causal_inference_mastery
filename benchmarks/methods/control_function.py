"""Benchmarks for Control Function Methods.

Methods benchmarked:
- control_function_ate: Control function approach to endogeneity
"""

from __future__ import annotations

from benchmarks.dgp import generate_iv_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_control_function_ate(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark control function ATE estimator.

    Two-stage approach: estimate residuals, include as control.
    Similar to 2SLS but allows flexible second stage.

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
    from src.causal_inference.control_function import control_function_ate

    # Use IV DGP (control function handles endogeneity)
    data = generate_iv_data(n=n, seed=seed, instrument_strength="strong")

    return benchmark_method(
        func=control_function_ate,
        method_name="control_function_ate",
        family="control_function",
        sample_size=n,
        outcome=data["outcome"],
        endogenous=data["endogenous"],
        instruments=data["instruments"],
        controls=data["controls"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "control_function_ate": benchmark_control_function_ate,
}
