"""Benchmarks for Observational methods.

Methods benchmarked:
- ipw_ate_obs: Inverse probability weighting (observational)
- dr_ate: Doubly robust AIPW
- tmle_ate: Targeted Maximum Likelihood Estimation
"""

from __future__ import annotations

from benchmarks.dgp import generate_observational_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_ipw_ate_obs(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark ipw_ate_observational.

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
    from src.causal_inference.observational import ipw_ate_observational

    data = generate_observational_data(n=n, seed=seed)

    return benchmark_method(
        func=ipw_ate_observational,
        method_name="ipw_ate_obs",
        family="observational",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_dr_ate(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark dr_ate (doubly robust AIPW).

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
    from src.causal_inference.observational import dr_ate

    data = generate_observational_data(n=n, seed=seed)

    return benchmark_method(
        func=dr_ate,
        method_name="dr_ate",
        family="observational",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_tmle_ate(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark tmle_ate.

    TMLE involves iterative targeting steps, making it more
    computationally expensive than DR.

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
    from src.causal_inference.observational import tmle_ate

    data = generate_observational_data(n=n, seed=seed)

    return benchmark_method(
        func=tmle_ate,
        method_name="tmle_ate",
        family="observational",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "ipw_ate_obs": benchmark_ipw_ate_obs,
    "dr_ate": benchmark_dr_ate,
    "tmle_ate": benchmark_tmle_ate,
}
