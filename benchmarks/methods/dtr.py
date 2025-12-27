"""Benchmarks for Dynamic Treatment Regimes (DTR).

Methods benchmarked:
- q_learning_single_stage: Q-learning for single-stage DTR
- a_learning_single_stage: A-learning for single-stage DTR
"""

from __future__ import annotations

from benchmarks.dgp import generate_dtr_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_q_learning_single_stage(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark Q-learning (single stage).

    Estimates optimal treatment regime via Q-function.
    Uses regression for value function estimation.

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
    from src.causal_inference.dtr import q_learning_single_stage

    data = generate_dtr_data(n=n, seed=seed)

    return benchmark_method(
        func=q_learning_single_stage,
        method_name="q_learning_single_stage",
        family="dtr",
        sample_size=n,
        outcome=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_a_learning_single_stage(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark A-learning (single stage).

    Advantage learning for optimal regime estimation.
    Directly models contrast (blip) function.

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
    from src.causal_inference.dtr import a_learning_single_stage

    data = generate_dtr_data(n=n, seed=seed)

    return benchmark_method(
        func=a_learning_single_stage,
        method_name="a_learning_single_stage",
        family="dtr",
        sample_size=n,
        outcome=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "q_learning_single_stage": benchmark_q_learning_single_stage,
    "a_learning_single_stage": benchmark_a_learning_single_stage,
}
