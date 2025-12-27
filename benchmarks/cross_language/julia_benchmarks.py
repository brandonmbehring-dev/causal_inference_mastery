"""Julia benchmark wrappers for cross-language comparison.

This module wraps the existing julia_interface.py functions with
timing-friendly signatures matching the Python benchmark patterns.

Each wrapper:
1. Takes data dict from Python DGP
2. Converts to Julia-compatible format
3. Calls the Julia function
4. Returns result (for timing purposes)
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Callable
import numpy as np

# Import Julia interface (handles availability check)
try:
    from tests.validation.cross_language.julia_interface import (
        is_julia_available,
        # RCT
        julia_simple_ate,
        julia_stratified_ate,
        julia_regression_ate,
        julia_ipw_ate,
        julia_permutation_test,
        # IV
        julia_tsls,
        julia_liml,
        julia_gmm,
        # RDD
        julia_sharp_rdd,
        julia_fuzzy_rdd,
        # DiD
        julia_classic_did,
        julia_event_study,
        julia_callaway_santanna,
        # Observational
        julia_ipw_ate_observational,
        julia_dr_ate,
        # PSM
        julia_psm_ate,
        # SCM
        julia_synthetic_control,
        # CATE
        julia_s_learner,
        julia_t_learner,
        julia_x_learner,
        julia_r_learner,
        julia_double_ml,
        # Sensitivity
        julia_e_value,
        julia_rosenbaum_bounds,
        # QTE
        julia_unconditional_qte,
        # Bounds
        julia_manski_worst_case,
        julia_lee_bounds,
        # Principal Stratification
        julia_cace_2sls,
    )
    JULIA_AVAILABLE = is_julia_available()
except ImportError:
    JULIA_AVAILABLE = False

    # Stub functions for when Julia is not available
    def _stub(*args, **kwargs):
        raise RuntimeError("Julia not available")

    julia_simple_ate = _stub
    julia_stratified_ate = _stub
    julia_regression_ate = _stub
    julia_ipw_ate = _stub
    julia_permutation_test = _stub
    julia_tsls = _stub
    julia_liml = _stub
    julia_gmm = _stub
    julia_sharp_rdd = _stub
    julia_fuzzy_rdd = _stub
    julia_classic_did = _stub
    julia_event_study = _stub
    julia_callaway_santanna = _stub
    julia_ipw_ate_observational = _stub
    julia_dr_ate = _stub
    julia_psm_ate = _stub
    julia_synthetic_control = _stub
    julia_s_learner = _stub
    julia_t_learner = _stub
    julia_x_learner = _stub
    julia_r_learner = _stub
    julia_double_ml = _stub
    julia_e_value = _stub
    julia_rosenbaum_bounds = _stub
    julia_unconditional_qte = _stub
    julia_manski_worst_case = _stub
    julia_lee_bounds = _stub
    julia_cace_2sls = _stub


# =============================================================================
# RCT Wrappers
# =============================================================================


def jl_benchmark_simple_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for simple_ate benchmark."""
    return julia_simple_ate(outcome, treatment)


def jl_benchmark_stratified_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    strata: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for stratified_ate benchmark."""
    return julia_stratified_ate(outcome, treatment, strata)


def jl_benchmark_regression_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for regression_ate benchmark."""
    return julia_regression_ate(outcome, treatment, covariates)


def jl_benchmark_ipw_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for ipw_ate benchmark."""
    return julia_ipw_ate(outcome, treatment, covariates)


def jl_benchmark_permutation_test(
    outcome: np.ndarray,
    treatment: np.ndarray,
    n_permutations: int = 1000,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for permutation_test benchmark."""
    return julia_permutation_test(outcome, treatment, n_permutations=n_permutations)


# =============================================================================
# IV Wrappers
# =============================================================================


def jl_benchmark_tsls(
    outcome: np.ndarray,
    endogenous: np.ndarray,
    instruments: np.ndarray,
    controls: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for 2SLS benchmark."""
    return julia_tsls(outcome, endogenous, instruments, controls)


def jl_benchmark_liml(
    outcome: np.ndarray,
    endogenous: np.ndarray,
    instruments: np.ndarray,
    controls: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for LIML benchmark."""
    return julia_liml(outcome, endogenous, instruments, controls)


def jl_benchmark_gmm(
    outcome: np.ndarray,
    endogenous: np.ndarray,
    instruments: np.ndarray,
    controls: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for GMM benchmark."""
    return julia_gmm(outcome, endogenous, instruments, controls)


# =============================================================================
# RDD Wrappers
# =============================================================================


def jl_benchmark_sharp_rdd(
    outcome: np.ndarray,
    running_variable: np.ndarray,
    cutoff: float = 0.0,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for sharp RDD benchmark."""
    return julia_sharp_rdd(outcome, running_variable, cutoff)


def jl_benchmark_fuzzy_rdd(
    outcome: np.ndarray,
    treatment: np.ndarray,
    running_variable: np.ndarray,
    cutoff: float = 0.0,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for fuzzy RDD benchmark."""
    return julia_fuzzy_rdd(outcome, treatment, running_variable, cutoff)


# =============================================================================
# DiD Wrappers
# =============================================================================


def jl_benchmark_did_2x2(
    outcome: np.ndarray,
    treatment: np.ndarray,
    post: np.ndarray,
    unit_id: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for 2x2 DiD benchmark."""
    return julia_classic_did(outcome, treatment, post, unit_id)


def jl_benchmark_event_study(
    outcome: np.ndarray,
    treatment: np.ndarray,
    time: np.ndarray,
    unit_id: np.ndarray,
    treatment_time: int,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for event study benchmark."""
    return julia_event_study(outcome, treatment, time, unit_id, treatment_time)


# =============================================================================
# Observational Wrappers
# =============================================================================


def jl_benchmark_ipw_ate_obs(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for observational IPW benchmark."""
    return julia_ipw_ate_observational(outcome, treatment, covariates)


def jl_benchmark_dr_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for doubly robust ATE benchmark."""
    return julia_dr_ate(outcome, treatment, covariates)


# =============================================================================
# PSM Wrappers
# =============================================================================


def jl_benchmark_psm_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    M: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for PSM benchmark."""
    return julia_psm_ate(outcome, treatment, covariates, n_matches=M)


# =============================================================================
# SCM Wrappers
# =============================================================================


def jl_benchmark_synthetic_control(
    outcomes: np.ndarray,
    treatment_period: int,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for synthetic control benchmark."""
    return julia_synthetic_control(outcomes, treatment_period)


# =============================================================================
# CATE Wrappers
# =============================================================================


def jl_benchmark_s_learner(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for S-learner benchmark."""
    return julia_s_learner(outcome, treatment, covariates)


def jl_benchmark_t_learner(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for T-learner benchmark."""
    return julia_t_learner(outcome, treatment, covariates)


def jl_benchmark_x_learner(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for X-learner benchmark."""
    return julia_x_learner(outcome, treatment, covariates)


def jl_benchmark_r_learner(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for R-learner benchmark."""
    return julia_r_learner(outcome, treatment, covariates)


def jl_benchmark_double_ml(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    n_folds: int = 5,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for Double ML benchmark."""
    return julia_double_ml(outcome, treatment, covariates, n_folds=n_folds)


# =============================================================================
# Sensitivity Wrappers
# =============================================================================


def jl_benchmark_e_value(
    estimate: float,
    se: float,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for E-value benchmark."""
    return julia_e_value(estimate, se)


def jl_benchmark_rosenbaum_bounds(
    paired_differences: np.ndarray,
    gamma_values: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for Rosenbaum bounds benchmark."""
    return julia_rosenbaum_bounds(paired_differences, gamma_values)


# =============================================================================
# Bounds Wrappers
# =============================================================================


def jl_benchmark_manski_worst_case(
    outcome: np.ndarray,
    treatment: np.ndarray,
    y_min: float,
    y_max: float,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for Manski worst-case bounds benchmark."""
    return julia_manski_worst_case(outcome, treatment, y_min, y_max)


def jl_benchmark_lee_bounds(
    outcome: np.ndarray,
    treatment: np.ndarray,
    selection: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for Lee bounds benchmark."""
    return julia_lee_bounds(outcome, treatment, selection)


# =============================================================================
# Principal Stratification Wrappers
# =============================================================================


def jl_benchmark_cace_2sls(
    outcome: np.ndarray,
    assignment: np.ndarray,
    received: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Julia wrapper for CACE 2SLS benchmark."""
    return julia_cace_2sls(outcome, assignment, received)


# =============================================================================
# Registry
# =============================================================================

JULIA_BENCHMARK_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "rct": {
        "simple_ate": jl_benchmark_simple_ate,
        "stratified_ate": jl_benchmark_stratified_ate,
        "regression_ate": jl_benchmark_regression_ate,
        "ipw_ate": jl_benchmark_ipw_ate,
        "permutation_test": jl_benchmark_permutation_test,
    },
    "iv": {
        "tsls": jl_benchmark_tsls,
        "liml": jl_benchmark_liml,
        "gmm": jl_benchmark_gmm,
    },
    "rdd": {
        "sharp_rdd": jl_benchmark_sharp_rdd,
        "fuzzy_rdd": jl_benchmark_fuzzy_rdd,
    },
    "did": {
        "did_2x2": jl_benchmark_did_2x2,
        "event_study": jl_benchmark_event_study,
    },
    "observational": {
        "ipw_ate_obs": jl_benchmark_ipw_ate_obs,
        "dr_ate": jl_benchmark_dr_ate,
    },
    "psm": {
        "psm_ate": jl_benchmark_psm_ate,
    },
    "scm": {
        "synthetic_control": jl_benchmark_synthetic_control,
    },
    "cate": {
        "s_learner": jl_benchmark_s_learner,
        "t_learner": jl_benchmark_t_learner,
        "x_learner": jl_benchmark_x_learner,
        "r_learner": jl_benchmark_r_learner,
        "double_ml": jl_benchmark_double_ml,
    },
    "sensitivity": {
        "e_value": jl_benchmark_e_value,
        "rosenbaum_bounds": jl_benchmark_rosenbaum_bounds,
    },
    "bounds": {
        "manski_worst_case": jl_benchmark_manski_worst_case,
        "lee_bounds": jl_benchmark_lee_bounds,
    },
    "principal_strat": {
        "cace_2sls": jl_benchmark_cace_2sls,
    },
}


def get_julia_benchmark(family: str, method: str) -> Optional[Callable]:
    """Get Julia benchmark wrapper for a method.

    Parameters
    ----------
    family : str
        Method family.
    method : str
        Method name.

    Returns
    -------
    Optional[Callable]
        Julia wrapper function or None if not available.
    """
    if not JULIA_AVAILABLE:
        return None

    family_benchmarks = JULIA_BENCHMARK_REGISTRY.get(family, {})
    return family_benchmarks.get(method)


def list_available_julia_benchmarks() -> Dict[str, list]:
    """List all available Julia benchmark wrappers.

    Returns
    -------
    Dict[str, list]
        Family -> list of method names.
    """
    return {
        family: list(methods.keys())
        for family, methods in JULIA_BENCHMARK_REGISTRY.items()
    }
