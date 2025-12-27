"""Benchmarks for Instrumental Variables methods.

Methods benchmarked:
- two_stage_ls: Two-Stage Least Squares (2SLS)
- liml: Limited Information Maximum Likelihood
- fuller: Fuller's k-class estimator
- gmm: Generalized Method of Moments
"""

from __future__ import annotations

from benchmarks.dgp import generate_iv_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_two_stage_ls(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark TwoStageLeastSquares.

    2SLS involves two OLS regressions with matrix operations.
    Scales as O(n * k²) where k is number of instruments + controls.

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
    from src.causal_inference.iv import TwoStageLeastSquares

    data = generate_iv_data(n=n, seed=seed, instrument_strength="strong")

    def run_2sls():
        model = TwoStageLeastSquares()
        return model.fit(
            Y=data["outcome"],
            D=data["endogenous"],
            Z=data["instruments"],
            X=data["controls"],
        )

    return benchmark_method(
        func=run_2sls,
        method_name="two_stage_ls",
        family="iv",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_liml(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark LIML.

    LIML is preferred over 2SLS with weak instruments.
    Involves eigenvalue computation, slightly more expensive.

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
    from src.causal_inference.iv import LIML

    data = generate_iv_data(n=n, seed=seed, instrument_strength="strong")

    def run_liml():
        model = LIML()
        return model.fit(
            Y=data["outcome"],
            D=data["endogenous"],
            Z=data["instruments"],
            X=data["controls"],
        )

    return benchmark_method(
        func=run_liml,
        method_name="liml",
        family="iv",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_fuller(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark Fuller estimator.

    Fuller is a bias-corrected version of LIML.
    Computational cost similar to LIML.

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
    from src.causal_inference.iv import Fuller

    data = generate_iv_data(n=n, seed=seed, instrument_strength="strong")

    def run_fuller():
        model = Fuller(alpha_param=1)  # Fuller(1) = standard Fuller
        return model.fit(
            Y=data["outcome"],
            D=data["endogenous"],
            Z=data["instruments"],
            X=data["controls"],
        )

    return benchmark_method(
        func=run_fuller,
        method_name="fuller",
        family="iv",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_gmm(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark GMM.

    GMM with multiple instruments involves efficient weighting
    matrix computation, making it more expensive than 2SLS.

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
    from src.causal_inference.iv import GMM

    # Use multiple instruments for GMM
    data = generate_iv_data(
        n=n,
        seed=seed,
        instrument_strength="strong",
        n_instruments=3,
    )

    def run_gmm():
        model = GMM()
        return model.fit(
            Y=data["outcome"],
            D=data["endogenous"],
            Z=data["instruments"],
            X=data["controls"],
        )

    return benchmark_method(
        func=run_gmm,
        method_name="gmm",
        family="iv",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "two_stage_ls": benchmark_two_stage_ls,
    "liml": benchmark_liml,
    "fuller": benchmark_fuller,
    "gmm": benchmark_gmm,
}
