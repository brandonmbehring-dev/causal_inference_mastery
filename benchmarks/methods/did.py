"""Benchmarks for Difference-in-Differences methods.

Methods benchmarked:
- did_2x2: Classic 2x2 DiD
- event_study: Event study design
- callaway_santanna: Callaway-Sant'Anna estimator for staggered adoption
"""

from __future__ import annotations

import numpy as np

from benchmarks.dgp import generate_did_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_did_2x2(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark did_2x2 (classic 2x2 DiD).

    Parameters
    ----------
    n : int
        Number of units (total obs = n * n_periods).
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
    from src.causal_inference.did import did_2x2

    # Scale n to be number of units
    n_units = max(10, n // 10)
    n_periods = 10
    treatment_period = 5
    data = generate_did_data(
        n_units=n_units,
        n_periods=n_periods,
        treatment_period=treatment_period,
        seed=seed,
    )

    # Create post indicator
    post = (data["time"] >= treatment_period).astype(int)

    # did_2x2 expects unit-level treatment (constant within each unit)
    # The DGP generates observation-level treatment, so we derive unit-level
    # First half of units are treated (unit < n_units * 0.5)
    n_treated = n_units // 2
    unit_treatment = (data["unit"] < n_treated).astype(int)

    return benchmark_method(
        func=did_2x2,
        method_name="did_2x2",
        family="did",
        sample_size=n_units * n_periods,
        outcomes=data["outcome"],
        treatment=unit_treatment,
        post=post,
        unit_id=data["unit"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_event_study(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark event_study.

    Event study includes dynamic treatment effects estimation
    which requires more computation.

    Parameters
    ----------
    n : int
        Number of units.
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
    from src.causal_inference.did import event_study

    n_units = max(10, n // 10)
    n_periods = 10
    treatment_period = 5
    data = generate_did_data(
        n_units=n_units,
        n_periods=n_periods,
        treatment_period=treatment_period,
        seed=seed,
    )

    return benchmark_method(
        func=event_study,
        method_name="event_study",
        family="did",
        sample_size=n_units * n_periods,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        time=data["time"],
        unit_id=data["unit"],
        treatment_time=treatment_period,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_callaway_santanna(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark callaway_santanna_ate.

    CS estimator is more complex due to:
    - Group-time specific ATT estimation
    - Aggregation across cohorts
    - Bootstrap inference

    Parameters
    ----------
    n : int
        Number of units.
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
    from src.causal_inference.did import callaway_santanna_ate
    from src.causal_inference.did.staggered import StaggeredData

    n_units = max(20, n // 10)
    n_periods = 10
    treatment_period = 5
    data = generate_did_data(
        n_units=n_units,
        n_periods=n_periods,
        treatment_period=treatment_period,
        seed=seed,
    )

    # Create treatment_time array for each unit
    # Treated units (first half) get treatment at period 5, others never treated
    n_treated = n_units // 2
    treatment_time = np.concatenate([
        np.full(n_treated, treatment_period),
        np.full(n_units - n_treated, np.inf),
    ])

    # Create StaggeredData object
    staggered_data = StaggeredData(
        outcomes=data["outcome"],
        treatment=data["treatment"],
        time=data["time"],
        unit_id=data["unit"],
        treatment_time=treatment_time,
    )

    def run_callaway_santanna():
        return callaway_santanna_ate(
            data=staggered_data,
            aggregation="simple",
            n_bootstrap=50,  # Reduce for benchmark speed
        )

    return benchmark_method(
        func=run_callaway_santanna,
        method_name="callaway_santanna",
        family="did",
        sample_size=n_units * n_periods,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "did_2x2": benchmark_did_2x2,
    "event_study": benchmark_event_study,
    "callaway_santanna": benchmark_callaway_santanna,
}
