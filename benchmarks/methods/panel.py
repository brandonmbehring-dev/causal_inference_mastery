"""Benchmarks for Panel Data Methods.

Methods benchmarked:
- dml_cre: Double ML with Correlated Random Effects (binary treatment)
- dml_cre_continuous: DML-CRE for continuous treatment
- panel_rif_qte: Panel RIF Quantile Treatment Effects
"""

from __future__ import annotations

from benchmarks.dgp import generate_panel_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_dml_cre(
    n_units: int = 100,
    n_periods: int = 8,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
    n_folds: int = 5,
) -> BenchmarkResult:
    """Benchmark DML-CRE for binary treatment.

    Combines Mundlak projection with double machine learning
    to handle correlated random effects.

    Parameters
    ----------
    n_units : int
        Number of panel units.
    n_periods : int
        Periods per unit.
    seed : int
        Random seed.
    n_repetitions : int
        Number of timing repetitions.
    n_warmup : int
        Number of warmup runs.
    n_folds : int
        Cross-fitting folds.

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.panel import dml_cre

    data = generate_panel_data(
        n_units=n_units,
        n_periods=n_periods,
        seed=seed,
    )

    def run_dml_cre():
        return dml_cre(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            unit_id=data["unit_id"],
            n_folds=n_folds,
        )

    return benchmark_method(
        func=run_dml_cre,
        method_name="dml_cre",
        family="panel",
        sample_size=n_units * n_periods,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_dml_cre_continuous(
    n_units: int = 100,
    n_periods: int = 8,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
    n_folds: int = 5,
) -> BenchmarkResult:
    """Benchmark DML-CRE for continuous treatment.

    Extended DML-CRE handling continuous treatment variable.

    Parameters
    ----------
    n_units : int
        Number of panel units.
    n_periods : int
        Periods per unit.
    seed : int
        Random seed.
    n_repetitions : int
        Number of timing repetitions.
    n_warmup : int
        Number of warmup runs.
    n_folds : int
        Cross-fitting folds.

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.panel import dml_cre_continuous
    import numpy as np

    data = generate_panel_data(
        n_units=n_units,
        n_periods=n_periods,
        seed=seed,
    )

    # Convert binary treatment to continuous
    rng = np.random.default_rng(seed)
    continuous_treatment = data["treatment"] + rng.normal(0, 0.5, len(data["treatment"]))

    def run_dml_cre_cont():
        return dml_cre_continuous(
            outcome=data["outcome"],
            treatment=continuous_treatment,
            covariates=data["covariates"],
            unit_id=data["unit_id"],
            n_folds=n_folds,
        )

    return benchmark_method(
        func=run_dml_cre_cont,
        method_name="dml_cre_continuous",
        family="panel",
        sample_size=n_units * n_periods,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_panel_rif_qte(
    n_units: int = 100,
    n_periods: int = 8,
    seed: int = 42,
    n_repetitions: int = 5,
    n_warmup: int = 1,
    quantiles: tuple = (0.25, 0.5, 0.75),
) -> BenchmarkResult:
    """Benchmark Panel RIF QTE.

    Recentered influence function approach to quantile effects
    in panel data setting.

    Parameters
    ----------
    n_units : int
        Number of panel units.
    n_periods : int
        Periods per unit.
    seed : int
        Random seed.
    n_repetitions : int
        Number of timing repetitions.
    n_warmup : int
        Number of warmup runs.
    quantiles : tuple
        Quantiles to estimate.

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.panel import panel_rif_qte

    data = generate_panel_data(
        n_units=n_units,
        n_periods=n_periods,
        seed=seed,
    )

    def run_panel_qte():
        return panel_rif_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            unit_id=data["unit_id"],
            quantiles=list(quantiles),
        )

    return benchmark_method(
        func=run_panel_qte,
        method_name="panel_rif_qte",
        family="panel",
        sample_size=n_units * n_periods,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "dml_cre": benchmark_dml_cre,
    "dml_cre_continuous": benchmark_dml_cre_continuous,
    "panel_rif_qte": benchmark_panel_rif_qte,
}
