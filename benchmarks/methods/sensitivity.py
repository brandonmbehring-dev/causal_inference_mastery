"""Benchmarks for Sensitivity Analysis Methods.

Methods benchmarked:
- e_value: VanderWeele-Ding E-value for unmeasured confounding
- rosenbaum_bounds: Rosenbaum sensitivity bounds
"""

from __future__ import annotations

from benchmarks.dgp import generate_observational_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_e_value(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark E-value calculation.

    E-value computation is O(1) after effect estimation,
    but includes point estimate and confidence interval E-values.

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
    from src.causal_inference.sensitivity import e_value
    from src.causal_inference.observational import dr_ate

    data = generate_observational_data(n=n, seed=seed)

    # Pre-compute effect estimate
    result = dr_ate(
        outcomes=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
    )

    def run_e_value():
        return e_value(
            estimate=result["ate"],
            se=result["se"],
            outcome_type="continuous",
        )

    return benchmark_method(
        func=run_e_value,
        method_name="e_value",
        family="sensitivity",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_rosenbaum_bounds(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
    gamma_max: float = 2.0,
) -> BenchmarkResult:
    """Benchmark Rosenbaum sensitivity bounds.

    Computes upper bounds on p-values under hidden bias.
    Involves sign-score test with binomial calculations.

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
    gamma_max : float
        Maximum sensitivity parameter.

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.sensitivity import rosenbaum_bounds
    from src.causal_inference.psm import psm_ate
    import numpy as np

    data = generate_observational_data(n=n, seed=seed)

    # Get matched pairs from PSM
    psm_result = psm_ate(
        outcomes=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
    )

    # Create matched pair differences
    rng = np.random.default_rng(seed)
    n_pairs = min(100, n // 4)
    pair_diffs = rng.normal(2.0, 1.0, n_pairs)  # Simulated pair differences

    def run_rosenbaum():
        return rosenbaum_bounds(
            paired_differences=pair_diffs,
            gamma_values=np.linspace(1.0, gamma_max, 10),
        )

    return benchmark_method(
        func=run_rosenbaum,
        method_name="rosenbaum_bounds",
        family="sensitivity",
        sample_size=n_pairs,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "e_value": benchmark_e_value,
    "rosenbaum_bounds": benchmark_rosenbaum_bounds,
}
