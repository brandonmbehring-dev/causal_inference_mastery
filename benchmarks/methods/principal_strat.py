"""Benchmarks for Principal Stratification Methods.

Methods benchmarked:
- cace_2sls: CACE via 2SLS (Imbens-Angrist)
- cace_em: CACE via EM algorithm
- sace_bounds: SACE bounds for survival outcomes

Note: Bayesian CACE excluded from routine benchmarks due to MCMC runtime.
"""

from __future__ import annotations

from benchmarks.dgp import generate_principal_strat_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_cace_2sls(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark CACE via 2SLS.

    Standard IV approach to complier average causal effect.
    Fast closed-form estimator.

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
    from src.causal_inference.principal_stratification import cace_2sls

    data = generate_principal_strat_data(n=n, seed=seed)

    return benchmark_method(
        func=cace_2sls,
        method_name="cace_2sls",
        family="principal_strat",
        sample_size=n,
        outcome=data["outcome"],
        assignment=data["assignment"],
        received=data["received"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_cace_em(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
    max_iter: int = 100,
) -> BenchmarkResult:
    """Benchmark CACE via EM algorithm.

    Expectation-Maximization for principal strata.
    Slower than 2SLS but handles more complex cases.

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
    max_iter : int
        Maximum EM iterations.

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.principal_stratification import cace_em

    data = generate_principal_strat_data(n=n, seed=seed)

    def run_cace_em():
        return cace_em(
            outcome=data["outcome"],
            assignment=data["assignment"],
            received=data["received"],
            max_iter=max_iter,
        )

    return benchmark_method(
        func=run_cace_em,
        method_name="cace_em",
        family="principal_strat",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_sace_bounds(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark SACE bounds (survivor average causal effect).

    Bounds for effects among always-survivors.
    Uses trimming approach.

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
    from src.causal_inference.principal_stratification import sace_bounds
    import numpy as np

    data = generate_principal_strat_data(n=n, seed=seed)

    # Create survival indicator (strata 1 and 2 survive)
    survival = (data["strata"] >= 1).astype(int)

    def run_sace():
        return sace_bounds(
            outcome=data["outcome"],
            treatment=data["received"],
            survival=survival,
        )

    return benchmark_method(
        func=run_sace,
        method_name="sace_bounds",
        family="principal_strat",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "cace_2sls": benchmark_cace_2sls,
    "cace_em": benchmark_cace_em,
    "sace_bounds": benchmark_sace_bounds,
}
