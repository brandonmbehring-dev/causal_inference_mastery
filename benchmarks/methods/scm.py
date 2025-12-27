"""Benchmarks for Synthetic Control Methods.

Methods benchmarked:
- synthetic_control: Core SCM estimator
- augmented_synthetic_control: Ben-Michael et al. (2021) ASCM with bias correction
"""

from __future__ import annotations

from benchmarks.dgp import generate_scm_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_synthetic_control(
    n_control: int = 20,
    n_periods: int = 20,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark synthetic_control.

    SCM involves quadratic programming for weight optimization,
    scales as O(n_control² * n_periods).

    Parameters
    ----------
    n_control : int
        Number of control units.
    n_periods : int
        Number of time periods.
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
    from src.causal_inference.scm import synthetic_control

    treatment_period = n_periods // 2
    data = generate_scm_data(
        n_control=n_control,
        n_periods=n_periods,
        treatment_period=treatment_period,
        seed=seed,
    )

    # SCM expects panel matrix with treated unit first
    outcomes = data["outcomes"]

    def run_synthetic_control():
        return synthetic_control(
            outcomes=outcomes,
            treatment_period=treatment_period,
            n_placebo=0,  # Skip inference for speed
        )

    return benchmark_method(
        func=run_synthetic_control,
        method_name="synthetic_control",
        family="scm",
        sample_size=(n_control + 1) * n_periods,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_synthetic_control_inference(
    n_control: int = 20,
    n_periods: int = 20,
    seed: int = 42,
    n_repetitions: int = 5,
    n_warmup: int = 1,
    n_placebo: int = 10,
) -> BenchmarkResult:
    """Benchmark synthetic_control with placebo inference.

    Full inference requires fitting SCM for each placebo unit.

    Parameters
    ----------
    n_control : int
        Number of control units.
    n_periods : int
        Number of time periods.
    seed : int
        Random seed.
    n_repetitions : int
        Number of timing repetitions.
    n_warmup : int
        Number of warmup runs.
    n_placebo : int
        Number of placebo iterations.

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.scm import synthetic_control

    treatment_period = n_periods // 2
    data = generate_scm_data(
        n_control=n_control,
        n_periods=n_periods,
        treatment_period=treatment_period,
        seed=seed,
    )

    outcomes = data["outcomes"]

    def run_with_inference():
        return synthetic_control(
            outcomes=outcomes,
            treatment_period=treatment_period,
            n_placebo=n_placebo,
        )

    return benchmark_method(
        func=run_with_inference,
        method_name="synthetic_control_inference",
        family="scm",
        sample_size=(n_control + 1) * n_periods,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_augmented_scm(
    n_control: int = 20,
    n_periods: int = 20,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark augmented_synthetic_control.

    ASCM adds ridge regression bias correction on top of SCM,
    requires additional regression fit.

    Parameters
    ----------
    n_control : int
        Number of control units.
    n_periods : int
        Number of time periods.
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
    from src.causal_inference.scm import augmented_synthetic_control

    treatment_period = n_periods // 2
    data = generate_scm_data(
        n_control=n_control,
        n_periods=n_periods,
        treatment_period=treatment_period,
        seed=seed,
    )

    outcomes = data["outcomes"]

    def run_ascm():
        return augmented_synthetic_control(
            outcomes=outcomes,
            treatment_period=treatment_period,
        )

    return benchmark_method(
        func=run_ascm,
        method_name="augmented_scm",
        family="scm",
        sample_size=(n_control + 1) * n_periods,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "synthetic_control": benchmark_synthetic_control,
    "synthetic_control_inference": benchmark_synthetic_control_inference,
    "augmented_scm": benchmark_augmented_scm,
}
