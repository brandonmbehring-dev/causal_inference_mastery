"""Benchmarks for Quantile Treatment Effects (QTE).

Methods benchmarked:
- unconditional_qte: Unconditional QTE via inverse propensity weighting
- conditional_qte: Conditional QTE with covariate adjustment
"""

from __future__ import annotations

from benchmarks.dgp import generate_qte_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_unconditional_qte(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
    quantiles: tuple = (0.25, 0.5, 0.75),
) -> BenchmarkResult:
    """Benchmark unconditional QTE.

    IPW-based quantile treatment effects.
    Requires propensity estimation + weighted quantile regression.

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
    quantiles : tuple
        Quantiles to estimate.

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.qte import unconditional_qte

    data = generate_qte_data(n=n, seed=seed, distribution="heavy_tailed")

    def run_uqte():
        return unconditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            quantiles=list(quantiles),
        )

    return benchmark_method(
        func=run_uqte,
        method_name="unconditional_qte",
        family="qte",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_conditional_qte(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
    quantiles: tuple = (0.25, 0.5, 0.75),
) -> BenchmarkResult:
    """Benchmark conditional QTE.

    QTE with covariate-specific effects.
    More complex than unconditional.

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
    quantiles : tuple
        Quantiles to estimate.

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.qte import conditional_qte

    data = generate_qte_data(n=n, seed=seed, distribution="heavy_tailed")

    def run_cqte():
        return conditional_qte(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            quantiles=list(quantiles),
        )

    return benchmark_method(
        func=run_cqte,
        method_name="conditional_qte",
        family="qte",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_unconditional_qte_band(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 5,
    n_warmup: int = 1,
    n_quantiles: int = 19,
) -> BenchmarkResult:
    """Benchmark unconditional QTE band (full process).

    Estimates QTE across many quantiles with uniform bands.
    More computation for inference.

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
    n_quantiles : int
        Number of quantiles in band.

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.qte import unconditional_qte_band
    import numpy as np

    data = generate_qte_data(n=n, seed=seed, distribution="heavy_tailed")

    quantiles = np.linspace(0.05, 0.95, n_quantiles)

    def run_uqte_band():
        return unconditional_qte_band(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            quantiles=quantiles.tolist(),
        )

    return benchmark_method(
        func=run_uqte_band,
        method_name="unconditional_qte_band",
        family="qte",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "unconditional_qte": benchmark_unconditional_qte,
    "conditional_qte": benchmark_conditional_qte,
    "unconditional_qte_band": benchmark_unconditional_qte_band,
}
