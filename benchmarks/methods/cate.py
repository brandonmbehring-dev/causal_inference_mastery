"""Benchmarks for Conditional Average Treatment Effects (CATE).

Methods benchmarked:
- s_learner: Single learner (one model)
- t_learner: Two learners (separate models per treatment)
- x_learner: Cross-learner with imputed effects
- r_learner: Robinson's residual-on-residual
- double_ml: Double/debiased machine learning
- causal_forest: Generalized random forest for CATE
"""

from __future__ import annotations

from benchmarks.dgp import generate_cate_data
from benchmarks.utils import BenchmarkResult, benchmark_method


def benchmark_s_learner(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark S-learner.

    Fits single model with treatment as feature.
    Fastest meta-learner but may have regularization bias.

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
    from src.causal_inference.cate import s_learner

    data = generate_cate_data(n=n, seed=seed, effect_type="linear")

    return benchmark_method(
        func=s_learner,
        method_name="s_learner",
        family="cate",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_t_learner(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark T-learner.

    Fits separate models for treatment and control.
    ~2x compute of S-learner.

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
    from src.causal_inference.cate import t_learner

    data = generate_cate_data(n=n, seed=seed, effect_type="linear")

    return benchmark_method(
        func=t_learner,
        method_name="t_learner",
        family="cate",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_x_learner(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark X-learner.

    Three-stage learner with imputed treatment effects.
    ~4x compute of S-learner.

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
    from src.causal_inference.cate import x_learner

    data = generate_cate_data(n=n, seed=seed, effect_type="linear")

    return benchmark_method(
        func=x_learner,
        method_name="x_learner",
        family="cate",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_r_learner(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark R-learner (Robinson's transformation).

    Residual-on-residual regression for CATE.
    Requires nuisance estimation + CATE fit.

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
    from src.causal_inference.cate import r_learner

    data = generate_cate_data(n=n, seed=seed, effect_type="linear")

    return benchmark_method(
        func=r_learner,
        method_name="r_learner",
        family="cate",
        sample_size=n,
        outcomes=data["outcome"],
        treatment=data["treatment"],
        covariates=data["covariates"],
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_double_ml(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 10,
    n_warmup: int = 1,
    n_folds: int = 5,
) -> BenchmarkResult:
    """Benchmark Double/Debiased ML.

    Cross-fitted nuisance estimation with orthogonal score.
    Slower due to cross-validation folds.

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
    n_folds : int
        Number of cross-fitting folds.

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.cate import double_ml

    data = generate_cate_data(n=n, seed=seed, effect_type="linear")

    def run_dml():
        return double_ml(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_folds=n_folds,
        )

    return benchmark_method(
        func=run_dml,
        method_name="double_ml",
        family="cate",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


def benchmark_causal_forest(
    n: int = 1000,
    seed: int = 42,
    n_repetitions: int = 5,
    n_warmup: int = 1,
    n_trees: int = 100,
) -> BenchmarkResult:
    """Benchmark Causal Forest.

    GRF-style causal forest for heterogeneous effects.
    Computationally intensive due to tree fitting.

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
    n_trees : int
        Number of trees (reduced for benchmarks).

    Returns
    -------
    BenchmarkResult
        Benchmark result.
    """
    from src.causal_inference.cate import causal_forest

    data = generate_cate_data(n=n, seed=seed, effect_type="nonlinear")

    def run_cf():
        return causal_forest(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            n_trees=n_trees,
        )

    return benchmark_method(
        func=run_cf,
        method_name="causal_forest",
        family="cate",
        sample_size=n,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
    )


BENCHMARKS = {
    "s_learner": benchmark_s_learner,
    "t_learner": benchmark_t_learner,
    "x_learner": benchmark_x_learner,
    "r_learner": benchmark_r_learner,
    "double_ml": benchmark_double_ml,
    "causal_forest": benchmark_causal_forest,
}
