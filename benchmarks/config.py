"""Benchmark configuration and method registry.

Provides BenchmarkConfig dataclass for controlling benchmark execution
and registry mapping method families to their benchmark functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Callable, Any


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Parameters
    ----------
    sample_sizes : List[int]
        Sample sizes to benchmark at. Default: [100, 500, 1000, 5000, 10000].
    n_repetitions : int
        Number of timing repetitions for stability. Default: 10.
    n_warmup : int
        Number of warmup runs to exclude (JIT compilation). Default: 1.
    seed : int
        Random seed for reproducibility. Default: 42.
    timeout_seconds : float
        Maximum time per benchmark before skip. Default: 300.0.
    slow_threshold_ms : float
        Threshold for marking methods as "slow". Default: 5000.0.
    verbose : bool
        Print progress during benchmark runs. Default: True.
    save_results : bool
        Automatically save results to JSON. Default: True.
    output_dir : str
        Directory for saving results. Default: "benchmarks/results".

    Examples
    --------
    >>> config = BenchmarkConfig(sample_sizes=[100, 1000], n_repetitions=5)
    >>> config.sample_sizes
    [100, 1000]
    """

    sample_sizes: List[int] = field(
        default_factory=lambda: [100, 500, 1000, 5000, 10000]
    )
    n_repetitions: int = 10
    n_warmup: int = 1
    seed: int = 42
    timeout_seconds: float = 300.0
    slow_threshold_ms: float = 5000.0
    verbose: bool = True
    save_results: bool = True
    output_dir: str = "benchmarks/results"


# Default configuration instance
DEFAULT_CONFIG = BenchmarkConfig()


# CI configuration (faster, smaller samples)
CI_CONFIG = BenchmarkConfig(
    sample_sizes=[100, 500, 1000],
    n_repetitions=3,
    timeout_seconds=60.0,
    verbose=False,
)


# Full benchmark configuration (comprehensive)
FULL_CONFIG = BenchmarkConfig(
    sample_sizes=[100, 500, 1000, 5000, 10000, 50000],
    n_repetitions=20,
    timeout_seconds=600.0,
)


# Method family registry
# Maps family names to list of (method_name, import_path, benchmark_func_name)
METHOD_FAMILIES: Dict[str, List[tuple]] = {
    "rct": [
        ("simple_ate", "src.causal_inference.rct.estimators_simple", "simple_ate"),
        ("stratified_ate", "src.causal_inference.rct.estimators_stratified", "stratified_ate"),
        ("regression_ate", "src.causal_inference.rct.estimators_regression", "regression_adjusted_ate"),
        ("permutation_test", "src.causal_inference.rct.estimators_permutation", "permutation_test"),
        ("ipw_ate", "src.causal_inference.rct.estimators_ipw", "ipw_ate"),
    ],
    "observational": [
        ("ipw_ate_obs", "src.causal_inference.observational.ipw", "ipw_ate_observational"),
        ("dr_ate", "src.causal_inference.observational.doubly_robust", "dr_ate"),
        ("tmle_ate", "src.causal_inference.observational.tmle", "tmle_ate"),
    ],
    "psm": [
        ("psm_ate", "src.causal_inference.psm.psm_estimator", "psm_ate"),
    ],
    "did": [
        ("did_2x2", "src.causal_inference.did.classic_did", "did_2x2"),
        ("event_study", "src.causal_inference.did.event_study", "event_study"),
        ("callaway_santanna", "src.causal_inference.did.callaway_santanna", "callaway_santanna_ate"),
    ],
    "iv": [
        ("two_stage_ls", "src.causal_inference.iv.two_stage_least_squares", "TwoStageLeastSquares"),
        ("liml", "src.causal_inference.iv.liml", "LIML"),
        ("fuller", "src.causal_inference.iv.fuller", "Fuller"),
        ("gmm", "src.causal_inference.iv.gmm", "GMM"),
    ],
    "rdd": [
        ("sharp_rdd", "src.causal_inference.rdd.sharp_rdd", "SharpRDD"),
        ("fuzzy_rdd", "src.causal_inference.rdd.fuzzy_rdd", "FuzzyRDD"),
        ("mccrary", "src.causal_inference.rdd.mccrary", "mccrary_density_test"),
    ],
    "scm": [
        ("synthetic_control", "src.causal_inference.scm.basic_scm", "synthetic_control"),
        ("augmented_scm", "src.causal_inference.scm.augmented_scm", "augmented_synthetic_control"),
    ],
    "cate": [
        ("s_learner", "src.causal_inference.cate.meta_learners", "s_learner"),
        ("t_learner", "src.causal_inference.cate.meta_learners", "t_learner"),
        ("x_learner", "src.causal_inference.cate.meta_learners", "x_learner"),
        ("r_learner", "src.causal_inference.cate.meta_learners", "r_learner"),
        ("double_ml", "src.causal_inference.cate.dml", "double_ml"),
        ("causal_forest", "src.causal_inference.cate.causal_forest", "causal_forest"),
    ],
    "sensitivity": [
        ("e_value", "src.causal_inference.sensitivity.e_value", "e_value"),
        ("rosenbaum_bounds", "src.causal_inference.sensitivity.rosenbaum", "rosenbaum_bounds"),
    ],
    "rkd": [
        ("sharp_rkd", "src.causal_inference.rkd.sharp_rkd", "SharpRKD"),
        ("fuzzy_rkd", "src.causal_inference.rkd.fuzzy_rkd", "FuzzyRKD"),
    ],
    "bunching": [
        ("bunching", "src.causal_inference.bunching.bunching_estimator", "bunching_estimator"),
    ],
    "selection": [
        ("heckman", "src.causal_inference.selection.heckman", "heckman_selection"),
    ],
    "bounds": [
        ("manski_bounds", "src.causal_inference.bounds.manski", "manski_bounds"),
        ("lee_bounds", "src.causal_inference.bounds.lee", "lee_bounds"),
    ],
    "qte": [
        ("unconditional_qte", "src.causal_inference.qte.unconditional", "unconditional_qte"),
        ("conditional_qte", "src.causal_inference.qte.conditional", "conditional_qte"),
    ],
    "mte": [
        ("mte_local_iv", "src.causal_inference.mte.local_iv", "mte_local_iv"),
    ],
    "mediation": [
        ("mediation", "src.causal_inference.mediation.mediation", "mediation_analysis"),
    ],
    "control_function": [
        ("linear_cf", "src.causal_inference.control_function.linear", "linear_control_function"),
        ("nonlinear_cf", "src.causal_inference.control_function.nonlinear", "nonlinear_control_function"),
    ],
    "shift_share": [
        ("shift_share_iv", "src.causal_inference.shift_share.shift_share", "shift_share_iv"),
    ],
    "bayesian": [
        ("bayesian_ate", "src.causal_inference.bayesian.bayesian_ate", "bayesian_ate"),
        ("bayesian_propensity", "src.causal_inference.bayesian.bayesian_propensity", "bayesian_propensity"),
        ("bayesian_dr", "src.causal_inference.bayesian.bayesian_dr", "bayesian_dr_ate"),
    ],
    "principal_strat": [
        ("cace_2sls", "src.causal_inference.principal_stratification.cace", "cace_2sls"),
        ("cace_em", "src.causal_inference.principal_stratification.cace", "cace_em"),
    ],
    "panel": [
        ("dml_cre", "src.causal_inference.panel.dml_cre", "dml_cre"),
        ("panel_qte", "src.causal_inference.panel.panel_qte", "panel_rif_qte"),
    ],
    "dtr": [
        ("q_learning", "src.causal_inference.dtr.q_learning", "q_learning_single_stage"),
        ("a_learning", "src.causal_inference.dtr.a_learning", "a_learning_single_stage"),
    ],
}


def get_all_families() -> List[str]:
    """Get list of all method family names."""
    return list(METHOD_FAMILIES.keys())


def get_family_methods(family: str) -> List[tuple]:
    """Get methods for a specific family.

    Parameters
    ----------
    family : str
        Family name (e.g., "rct", "did").

    Returns
    -------
    List[tuple]
        List of (method_name, import_path, func_name) tuples.

    Raises
    ------
    ValueError
        If family not found.
    """
    if family not in METHOD_FAMILIES:
        raise ValueError(
            f"Unknown family '{family}'. Available: {get_all_families()}"
        )
    return METHOD_FAMILIES[family]


# Tolerance bands for regression testing (percentage)
# Faster methods have more variance, so wider tolerance
TOLERANCE_BANDS: Dict[str, float] = {
    "fast": 1.00,      # < 10ms: ±100%
    "medium": 0.50,    # 10-100ms: ±50%
    "slow": 0.30,      # 100ms-1s: ±30%
    "very_slow": 0.20, # > 1s: ±20%
}


def get_tolerance(median_ms: float) -> float:
    """Get tolerance band for a given median time.

    Parameters
    ----------
    median_ms : float
        Median execution time in milliseconds.

    Returns
    -------
    float
        Tolerance as fraction (e.g., 0.50 = ±50%).
    """
    if median_ms < 10:
        return TOLERANCE_BANDS["fast"]
    elif median_ms < 100:
        return TOLERANCE_BANDS["medium"]
    elif median_ms < 1000:
        return TOLERANCE_BANDS["slow"]
    else:
        return TOLERANCE_BANDS["very_slow"]
