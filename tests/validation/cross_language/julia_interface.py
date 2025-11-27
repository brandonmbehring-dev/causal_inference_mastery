"""
Julia interface for Python→Julia cross-validation.

Uses juliacall to call Julia CausalEstimators module from Python.
"""

import numpy as np
from typing import Dict, Union, Optional
import os

# Set Julia project environment
JULIA_PROJECT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "julia"
)

# Initialize juliacall with Julia project
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
os.environ["PYTHON_JULIACALL_THREADS"] = "auto"

try:
    from juliacall import Main as jl

    # Activate Julia project
    jl.seval(f'import Pkg; Pkg.activate("{JULIA_PROJECT_PATH}")')

    # Import CausalEstimators
    jl.seval("using CausalEstimators")

    JULIA_AVAILABLE = True
except ImportError:
    JULIA_AVAILABLE = False
    jl = None


def is_julia_available() -> bool:
    """Check if Julia is available for cross-validation."""
    return JULIA_AVAILABLE


def julia_simple_ate(outcomes: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """
    Call Julia simple_ate via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes
    treatment : np.ndarray
        Treatment assignments (0/1)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with estimate, se, ci_lower, ci_upper
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Create RCTProblem
    problem = jl.RCTProblem(outcomes, treatment, None, None, jl.seval(f"(alpha={alpha},)"))

    # Solve with SimpleATE
    solution = jl.solve(problem, jl.SimpleATE())

    # Extract results
    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
    }


def julia_stratified_ate(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    strata: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Union[float, int, list]]:
    """
    Call Julia stratified_ate via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes
    treatment : np.ndarray
        Treatment assignments (0/1)
    strata : np.ndarray
        Stratum indicators
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with estimate, se, ci_lower, ci_upper, n_strata
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Create RCTProblem with strata
    problem = jl.RCTProblem(outcomes, treatment, strata, None, jl.seval(f"(alpha={alpha},)"))

    # Solve with StratifiedATE
    solution = jl.solve(problem, jl.StratifiedATE())

    # Extract results
    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "n_strata": int(solution.n_strata),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
    }


def julia_regression_ate(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Union[float, int]]:
    """
    Call Julia regression_ate via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes
    treatment : np.ndarray
        Treatment assignments (0/1)
    covariates : np.ndarray
        Covariates (1D or 2D)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with estimate, se, ci_lower, ci_upper, r_squared
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Ensure covariates is 2D
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    # Create RCTProblem with covariates
    problem = jl.RCTProblem(outcomes, treatment, None, covariates, jl.seval(f"(alpha={alpha},)"))

    # Solve with RegressionATE
    solution = jl.solve(problem, jl.RegressionATE())

    # Extract results
    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "r_squared": float(solution.r_squared),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
    }


def julia_ipw_ate(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Call Julia ipw_ate via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes
    treatment : np.ndarray
        Treatment assignments (0/1)
    propensity : np.ndarray
        Propensity scores
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with estimate, se, ci_lower, ci_upper
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Create RCTProblem with propensity
    # Note: Julia IPW might be called IPWATE() instead
    problem = jl.RCTProblem(outcomes, treatment, None, None, jl.seval(f"(alpha={alpha}, propensity={list(propensity)})"))

    # Solve with IPWATE
    solution = jl.solve(problem, jl.IPWATE())

    # Extract results
    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
    }


def julia_permutation_test(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    n_permutations: Optional[int] = 1000,
    alternative: str = "two-sided",
    random_seed: Optional[int] = None
) -> Dict[str, Union[float, int, str]]:
    """
    Call Julia permutation_test via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes
    treatment : np.ndarray
        Treatment assignments (0/1)
    n_permutations : int or None, default=1000
        Number of permutations (None for exact test)
    alternative : str, default="two-sided"
        Alternative hypothesis
    random_seed : int or None, default=None
        Random seed

    Returns
    -------
    dict
        Julia result with p_value, observed_statistic, n_permutations
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Create RCTProblem
    params = jl.seval(f"(n_permutations={n_permutations if n_permutations else 'nothing'}, " +
                     f"alternative=Symbol('{alternative.replace('-', '_')}'), " +
                     f"random_seed={random_seed if random_seed else 'nothing'})")

    problem = jl.RCTProblem(outcomes, treatment, None, None, params)

    # Solve with PermutationTest
    solution = jl.solve(problem, jl.PermutationTest())

    # Extract results
    return {
        "p_value": float(solution.p_value),
        "observed_statistic": float(solution.observed_statistic),
        "n_permutations": int(solution.n_permutations),
        "alternative": str(solution.alternative),
    }
