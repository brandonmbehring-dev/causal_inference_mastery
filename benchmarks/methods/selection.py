"""Benchmarks for Selection Correction Methods.

Methods benchmarked:
- heckman_two_step: Heckman (1979) two-step selection correction
"""

from __future__ import annotations

from benchmarks.dgp import generate_selection_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_heckman_two_step(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark Heckman two-step selection correction.

    Classic selection bias correction using inverse Mills ratio.
    Requires probit first stage + OLS second stage.

    Parameters
    ----------
    n : int
        Sample size (before selection).
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
    from src.causal_inference.selection import heckman_two_step
    import numpy as np

    data = generate_selection_data(n=n, seed=seed)

    # Filter to observed only for outcome model
    mask = data["selection_indicator"] == 1

    # Stack covariates with exclusion restriction for selection equation
    selection_covariates = np.column_stack([
        data["covariates"],
        data["exclusion_restriction"]
    ])

    def run_heckman():
        return heckman_two_step(
            outcome=data["outcome"],
            selection_indicator=data["selection_indicator"],
            outcome_covariates=data["covariates"],
            selection_covariates=selection_covariates,
        )

    return benchmark_method(
        func=run_heckman,
        method_name="heckman_two_step",
        family="selection",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "heckman_two_step": benchmark_heckman_two_step,
}
