"""Benchmarks for Regression Kink Design (RKD).

Methods benchmarked:
- SharpRKD: Sharp kink design with polynomial regression
- FuzzyRKD: Fuzzy kink design with 2SLS
"""

from __future__ import annotations

from benchmarks.dgp import generate_rkd_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_sharp_rkd(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
    polynomial_order: int = 2,
) -> BenchmarkResult:
    """Benchmark SharpRKD.

    Sharp regression kink design with local polynomial regression.
    Estimates slope change at kink point.

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
        Order of polynomial fit.

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.rkd import SharpRKD

    data = generate_rkd_data(n=n, seed=seed, design="sharp")

    def run_sharp_rkd():
        rkd = SharpRKD(polynomial_order=polynomial_order)
        return rkd.fit(
            outcome=data["outcome"],
            running_variable=data["running_variable"],
            kink_point=data["kink_point"],
        )

    return benchmark_method(
        func=run_sharp_rkd,
        method_name="sharp_rkd",
        family="rkd",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_sharp_rkd_bandwidth_selection(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 5,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark SharpRKD with automatic bandwidth selection.

    IK or CCT bandwidth selection adds substantial computation.

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
    from src.causal_inference.rkd import SharpRKD

    data = generate_rkd_data(n=n, seed=seed, design="sharp")

    def run_with_bw():
        rkd = SharpRKD(bandwidth="auto")
        return rkd.fit(
            outcome=data["outcome"],
            running_variable=data["running_variable"],
            kink_point=data["kink_point"],
        )

    return benchmark_method(
        func=run_with_bw,
        method_name="sharp_rkd_bw",
        family="rkd",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_fuzzy_rkd(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark FuzzyRKD.

    Fuzzy regression kink design using 2SLS.
    Requires first-stage slope estimation.

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
    from src.causal_inference.rkd import FuzzyRKD

    data = generate_rkd_data(n=n, seed=seed, design="fuzzy")

    def run_fuzzy_rkd():
        rkd = FuzzyRKD()
        return rkd.fit(
            outcome=data["outcome"],
            treatment=data["treatment_intensity"],
            running_variable=data["running_variable"],
            kink_point=data["kink_point"],
        )

    return benchmark_method(
        func=run_fuzzy_rkd,
        method_name="fuzzy_rkd",
        family="rkd",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "sharp_rkd": benchmark_sharp_rkd,
    "sharp_rkd_bw": benchmark_sharp_rkd_bandwidth_selection,
    "fuzzy_rkd": benchmark_fuzzy_rkd,
}
