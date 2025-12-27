"""Benchmarks for Mediation Analysis Methods.

Methods benchmarked:
- baron_kenny: Classic Baron-Kenny mediation test
- mediation_analysis: Full mediation analysis with bootstrap
"""

from __future__ import annotations

from benchmarks.dgp import generate_mediation_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_baron_kenny(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark Baron-Kenny mediation test.

    Classic 4-step mediation test.
    Fast OLS-based approach.

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
    from src.causal_inference.mediation import baron_kenny

    data = generate_mediation_data(n=n, seed=seed)

    return benchmark_method(
        func=baron_kenny,
        method_name="baron_kenny",
        family="mediation",
        sample_size=n,
        outcome=data["outcome"],
        treatment=data["treatment"],
        mediator=data["mediator"],
        covariates=data["covariates"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_mediation_analysis(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 5,
    n_warmup: int = 1,
    n_bootstrap: int = 100,
) -> BenchmarkResult:
    """Benchmark full mediation analysis.

    Includes ACME, ADE, and bootstrap inference.
    Slower due to bootstrap.

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
    n_bootstrap : int
        Bootstrap iterations (reduced for benchmark).

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.mediation import mediation_analysis

    data = generate_mediation_data(n=n, seed=seed)

    def run_mediation():
        return mediation_analysis(
            outcome=data["outcome"],
            treatment=data["treatment"],
            mediator=data["mediator"],
            covariates=data["covariates"],
            n_bootstrap=n_bootstrap,
        )

    return benchmark_method(
        func=run_mediation,
        method_name="mediation_analysis",
        family="mediation",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "baron_kenny": benchmark_baron_kenny,
    "mediation_analysis": benchmark_mediation_analysis,
}
