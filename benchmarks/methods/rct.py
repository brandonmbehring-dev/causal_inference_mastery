"""Benchmarks for RCT (Randomized Controlled Trial) methods.

Methods benchmarked:
- simple_ate: Simple difference-in-means
- stratified_ate: Stratified estimation
- regression_ate: Covariate-adjusted regression
- permutation_test: Permutation-based inference
- ipw_ate: Inverse probability weighting
"""

from __future__ import annotations

import numpy as np

from benchmarks.dgp import generate_rct_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_simple_ate(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark simple_ate (difference-in-means).

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
    from src.causal_inference.rct import simple_ate

    # Generate data
    data = generate_rct_data(n=n, seed=seed)

    return benchmark_method(
        func=simple_ate,
        method_name="simple_ate",
        family="rct",
        sample_size=n,
        # Arguments for simple_ate
        outcomes=data["outcome"],
        treatment=data["treatment"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_stratified_ate(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark stratified_ate.

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
    from src.causal_inference.rct import stratified_ate

    data = generate_rct_data(n=n, seed=seed)

    return benchmark_method(
        func=stratified_ate,
        method_name="stratified_ate",
        family="rct",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        strata=data["strata"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_regression_ate(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark regression_adjusted_ate.

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
    from src.causal_inference.rct import regression_adjusted_ate

    data = generate_rct_data(n=n, seed=seed)

    return benchmark_method(
        func=regression_adjusted_ate,
        method_name="regression_ate",
        family="rct",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_permutation_test(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
    n_permutations: int = 1000,
) -> BenchmarkResult:
    """Benchmark permutation_test.

    Note: This is one of the most expensive RCT methods due to
    O(n * n_permutations) complexity.

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
    n_permutations : int
        Number of permutations (default 1000).

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.rct import permutation_test

    data = generate_rct_data(n=n, seed=seed)

    return benchmark_method(
        func=permutation_test,
        method_name="permutation_test",
        family="rct",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        n_permutations=n_permutations,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_ipw_ate(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark ipw_ate for RCT context.

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
    from src.causal_inference.rct import ipw_ate

    data = generate_rct_data(n=n, seed=seed)

    # Compute propensity scores (RCT uses constant 0.5)
    propensity = np.full(n, 0.5)

    return benchmark_method(
        func=ipw_ate,
        method_name="ipw_ate",
        family="rct",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        propensity=propensity,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


# Registry for this module
BENCHMARKS = {
    "simple_ate": benchmark_simple_ate,
    "stratified_ate": benchmark_stratified_ate,
    "regression_ate": benchmark_regression_ate,
    "permutation_test": benchmark_permutation_test,
    "ipw_ate": benchmark_ipw_ate,
}
