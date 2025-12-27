"""Benchmarks for Bayesian Causal Inference Methods.

Methods benchmarked:
- bayesian_ate: Conjugate Bayesian ATE (fast, closed-form)
- bayesian_propensity: Bayesian propensity stratification
- bayesian_dr_ate: Bayesian doubly robust estimation

Note: MCMC-based methods (hierarchical Bayesian) excluded from routine
benchmarks due to runtime. Run with: pytest -m mcmc
"""

from __future__ import annotations

import pytest

from benchmarks.dgp import generate_observational_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_bayesian_ate(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark conjugate Bayesian ATE.

    Uses Normal-Inverse-Gamma conjugate prior.
    O(n) with closed-form posterior.

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
    from src.causal_inference.bayesian import bayesian_ate

    data = generate_observational_data(n=n, seed=seed)

    return benchmark_method(
        func=bayesian_ate,
        method_name="bayesian_ate",
        family="bayesian",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_bayesian_propensity(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark Bayesian propensity stratification.

    Bayesian logistic propensity with stratified estimation.
    Uses Laplace approximation for efficiency.

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
    from src.causal_inference.bayesian import bayesian_propensity

    data = generate_observational_data(n=n, seed=seed)

    return benchmark_method(
        func=bayesian_propensity,
        method_name="bayesian_propensity",
        family="bayesian",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_bayesian_dr_ate(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark Bayesian doubly robust ATE.

    Combines propensity and outcome models in Bayesian framework.
    More compute than conjugate but still avoids MCMC.

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
    from src.causal_inference.bayesian import bayesian_dr_ate

    data = generate_observational_data(n=n, seed=seed)

    return benchmark_method(
        func=bayesian_dr_ate,
        method_name="bayesian_dr_ate",
        family="bayesian",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "bayesian_ate": benchmark_bayesian_ate,
    "bayesian_propensity": benchmark_bayesian_propensity,
    "bayesian_dr_ate": benchmark_bayesian_dr_ate,
}
