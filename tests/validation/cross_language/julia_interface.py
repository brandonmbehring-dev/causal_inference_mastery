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

    # Convert to Julia-compatible types
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))  # RCTProblem expects Vector{Bool}

    # Create RCTProblem
    problem = jl.RCTProblem(jl_outcomes, jl_treatment, None, None, jl.seval(f"(alpha={alpha},)"))

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


# =============================================================================
# IV (Instrumental Variables) Functions
# =============================================================================


def julia_tsls(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    instruments: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    robust: bool = True,
) -> Dict[str, Union[float, int, bool]]:
    """
    Call Julia TSLS (Two-Stage Least Squares) via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    treatment : np.ndarray
        Endogenous treatment variable D (n,)
    instruments : np.ndarray
        Instrumental variables Z (n,) or (n, K)
    covariates : np.ndarray, optional
        Exogenous covariates X (n, p)
    alpha : float, default=0.05
        Significance level
    robust : bool, default=True
        Use heteroskedasticity-robust standard errors

    Returns
    -------
    dict
        Julia result with estimate, se, ci, p_value, first_stage_fstat, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Ensure instruments is 2D
    if instruments.ndim == 1:
        instruments = instruments.reshape(-1, 1)

    # Ensure covariates is 2D or None
    if covariates is not None and covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    # Convert numpy arrays to Julia arrays (IVProblem requires Vector{T}, not PyArray)
    jl_outcomes = jl.collect(outcomes)
    jl_treatment = jl.collect(treatment)
    jl_instruments = jl.collect(instruments)
    jl_covariates = jl.collect(covariates) if covariates is not None else jl.seval("nothing")

    # Create IVProblem
    problem = jl.IVProblem(
        jl_outcomes, jl_treatment, jl_instruments, jl_covariates,
        jl.seval(f"(alpha={alpha},)")
    )

    # Solve with TSLS
    solution = jl.solve(problem, jl.TSLS(robust=robust))

    # Extract results
    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "p_value": float(solution.p_value),
        "n": int(solution.n),
        "n_instruments": int(solution.n_instruments),
        "first_stage_fstat": float(solution.first_stage_fstat),
        "weak_iv_warning": bool(solution.weak_iv_warning),
        "overid_pvalue": float(solution.overid_pvalue) if solution.overid_pvalue is not None else None,
    }


def julia_liml(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    instruments: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    robust: bool = True,
    fuller: float = 0.0,
) -> Dict[str, Union[float, int, bool]]:
    """
    Call Julia LIML (Limited Information Maximum Likelihood) via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    treatment : np.ndarray
        Endogenous treatment variable D (n,)
    instruments : np.ndarray
        Instrumental variables Z (n,) or (n, K)
    covariates : np.ndarray, optional
        Exogenous covariates X (n, p)
    alpha : float, default=0.05
        Significance level
    robust : bool, default=True
        Use heteroskedasticity-robust standard errors
    fuller : float, default=0.0
        Fuller modification parameter (0 = pure LIML, 1 = Fuller-LIML)

    Returns
    -------
    dict
        Julia result with estimate, se, ci, p_value, kappa, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Ensure instruments is 2D
    if instruments.ndim == 1:
        instruments = instruments.reshape(-1, 1)

    # Ensure covariates is 2D or None
    if covariates is not None and covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    # Convert numpy arrays to Julia arrays (IVProblem requires Vector{T}, not PyArray)
    jl_outcomes = jl.collect(outcomes)
    jl_treatment = jl.collect(treatment)
    jl_instruments = jl.collect(instruments)
    jl_covariates = jl.collect(covariates) if covariates is not None else jl.seval("nothing")

    # Create IVProblem
    problem = jl.IVProblem(
        jl_outcomes, jl_treatment, jl_instruments, jl_covariates,
        jl.seval(f"(alpha={alpha},)")
    )

    # Solve with LIML
    solution = jl.solve(problem, jl.LIML(robust=robust, fuller=fuller))

    # Extract results
    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "p_value": float(solution.p_value),
        "n": int(solution.n),
        "n_instruments": int(solution.n_instruments),
        "first_stage_fstat": float(solution.first_stage_fstat),
        "weak_iv_warning": bool(solution.weak_iv_warning),
        "overid_pvalue": float(solution.overid_pvalue) if solution.overid_pvalue is not None else None,
    }


def julia_gmm(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    instruments: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    weighting: str = "optimal",
) -> Dict[str, Union[float, int, bool]]:
    """
    Call Julia GMM (Generalized Method of Moments) via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    treatment : np.ndarray
        Endogenous treatment variable D (n,)
    instruments : np.ndarray
        Instrumental variables Z (n,) or (n, K)
    covariates : np.ndarray, optional
        Exogenous covariates X (n, p)
    alpha : float, default=0.05
        Significance level
    weighting : str, default="optimal"
        Weighting matrix: "identity", "optimal", or "iterative"

    Returns
    -------
    dict
        Julia result with estimate, se, ci, p_value, j_stat, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Ensure instruments is 2D
    if instruments.ndim == 1:
        instruments = instruments.reshape(-1, 1)

    # Ensure covariates is 2D or None
    if covariates is not None and covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    # Convert numpy arrays to Julia arrays (IVProblem requires Vector{T}, not PyArray)
    jl_outcomes = jl.collect(outcomes)
    jl_treatment = jl.collect(treatment)
    jl_instruments = jl.collect(instruments)
    jl_covariates = jl.collect(covariates) if covariates is not None else jl.seval("nothing")

    # Create IVProblem
    problem = jl.IVProblem(
        jl_outcomes, jl_treatment, jl_instruments, jl_covariates,
        jl.seval(f"(alpha={alpha},)")
    )

    # Solve with GMM
    solution = jl.solve(problem, jl.GMM(weighting=jl.seval(f'Symbol("{weighting}")')))

    # Extract results
    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "p_value": float(solution.p_value),
        "n": int(solution.n),
        "n_instruments": int(solution.n_instruments),
        "first_stage_fstat": float(solution.first_stage_fstat),
        "weak_iv_warning": bool(solution.weak_iv_warning),
        "overid_pvalue": float(solution.overid_pvalue) if solution.overid_pvalue is not None else None,
    }


# =============================================================================
# IV Stage Decomposition Functions (Session 56)
# =============================================================================


def julia_first_stage(
    treatment: np.ndarray,
    instruments: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Dict[str, Union[float, np.ndarray, int, bool]]:
    """
    Call Julia FirstStage via juliacall.

    Parameters
    ----------
    treatment : np.ndarray
        Endogenous treatment variable D (n,)
    instruments : np.ndarray
        Instrumental variables Z (n,) or (n, K)
    covariates : np.ndarray, optional
        Exogenous covariates X (n, p)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with coef, se, r2, partial_r2, f_statistic, fitted_values, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Ensure instruments is 2D
    if instruments.ndim == 1:
        instruments = instruments.reshape(-1, 1)

    # Ensure covariates is 2D or None
    if covariates is not None and covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    # Convert numpy arrays to Julia arrays
    jl_treatment = jl.collect(treatment.astype(np.float64))
    jl_instruments = jl.collect(instruments.astype(np.float64))
    jl_covariates = jl.collect(covariates.astype(np.float64)) if covariates is not None else jl.seval("nothing")

    # Create FirstStageProblem
    problem = jl.FirstStageProblem(
        jl_treatment, jl_instruments, jl_covariates,
        jl.seval(f"(alpha={alpha},)")
    )

    # Solve with OLS
    solution = jl.solve(problem, jl.OLS())

    # Extract results
    return {
        "coef": np.array([float(c) for c in solution.coef]),
        "se": np.array([float(s) for s in solution.se]),
        "r2": float(solution.r2),
        "partial_r2": float(solution.partial_r2),
        "f_statistic": float(solution.f_statistic),
        "f_pvalue": float(solution.f_pvalue),
        "fitted_values": np.array([float(f) for f in solution.fitted_values]),
        "n": int(solution.n),
        "n_instruments": int(solution.n_instruments),
        "weak_iv_warning": bool(solution.weak_iv_warning),
    }


def julia_reduced_form(
    outcomes: np.ndarray,
    instruments: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Dict[str, Union[float, np.ndarray, int]]:
    """
    Call Julia ReducedForm via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    instruments : np.ndarray
        Instrumental variables Z (n,) or (n, K)
    covariates : np.ndarray, optional
        Exogenous covariates X (n, p)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with coef, se, r2, fitted_values, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Ensure instruments is 2D
    if instruments.ndim == 1:
        instruments = instruments.reshape(-1, 1)

    # Ensure covariates is 2D or None
    if covariates is not None and covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    # Convert numpy arrays to Julia arrays
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_instruments = jl.collect(instruments.astype(np.float64))
    jl_covariates = jl.collect(covariates.astype(np.float64)) if covariates is not None else jl.seval("nothing")

    # Create ReducedFormProblem
    problem = jl.ReducedFormProblem(
        jl_outcomes, jl_instruments, jl_covariates,
        jl.seval(f"(alpha={alpha},)")
    )

    # Solve with OLS
    solution = jl.solve(problem, jl.OLS())

    # Extract results
    return {
        "coef": np.array([float(c) for c in solution.coef]),
        "se": np.array([float(s) for s in solution.se]),
        "r2": float(solution.r2),
        "fitted_values": np.array([float(f) for f in solution.fitted_values]),
        "n": int(solution.n),
        "n_instruments": int(solution.n_instruments),
    }


def julia_second_stage(
    outcomes: np.ndarray,
    fitted_treatment: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Dict[str, Union[float, np.ndarray, int]]:
    """
    Call Julia SecondStage via juliacall.

    WARNING: This returns NAIVE standard errors that are INCORRECT for inference.
    Use julia_tsls() for correct 2SLS standard errors.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    fitted_treatment : np.ndarray
        Fitted treatment D̂ from first stage (n,)
    covariates : np.ndarray, optional
        Exogenous covariates X (n, p)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with coef, se_naive (INCORRECT!), r2, fitted_values, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Ensure covariates is 2D or None
    if covariates is not None and covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    # Convert numpy arrays to Julia arrays
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_fitted_treatment = jl.collect(fitted_treatment.astype(np.float64))
    jl_covariates = jl.collect(covariates.astype(np.float64)) if covariates is not None else jl.seval("nothing")

    # Create SecondStageProblem
    problem = jl.SecondStageProblem(
        jl_outcomes, jl_fitted_treatment, jl_covariates,
        jl.seval(f"(alpha={alpha},)")
    )

    # Solve with OLS (will issue warning about naive SEs)
    solution = jl.solve(problem, jl.OLS())

    # Extract results
    return {
        "coef": np.array([float(c) for c in solution.coef]),
        "se_naive": np.array([float(s) for s in solution.se_naive]),  # Explicitly NAIVE
        "r2": float(solution.r2),
        "fitted_values": np.array([float(f) for f in solution.fitted_values]),
        "n": int(solution.n),
    }


# =============================================================================
# RDD (Regression Discontinuity Design) Functions
# =============================================================================


def julia_sharp_rdd(
    outcomes: np.ndarray,
    running_var: np.ndarray,
    cutoff: float,
    bandwidth: Union[float, str] = "cct",
    kernel: str = "triangular",
    alpha: float = 0.05,
    run_density_test: bool = False,
) -> Dict[str, Union[float, int, bool, None]]:
    """
    Call Julia SharpRDD via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    running_var : np.ndarray
        Running variable X (n,)
    cutoff : float
        Cutoff threshold
    bandwidth : float or str, default="cct"
        Bandwidth: "ik", "cct", or numeric value
    kernel : str, default="triangular"
        Kernel function: "triangular", "uniform", or "rectangular" (mapped to uniform)
    alpha : float, default=0.05
        Significance level
    run_density_test : bool, default=False
        Run McCrary density test (disabled for cross-validation to avoid warnings)

    Returns
    -------
    dict
        Julia result with estimate, se, ci, p_value, bandwidth, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert numpy arrays to Julia arrays
    jl_outcomes = jl.collect(outcomes)
    jl_running_var = jl.collect(running_var)

    # Compute treatment from running variable (Sharp RDD)
    treatment_bool = running_var >= cutoff
    jl_treatment = jl.collect(treatment_bool)

    # Create RDDProblem
    problem = jl.RDDProblem(
        jl_outcomes,
        jl_running_var,
        jl_treatment,
        float(cutoff),
        jl.seval("nothing"),  # No covariates
        jl.seval(f"(alpha={alpha},)")
    )

    # Map kernel name: Python "rectangular" -> Julia "uniform"
    kernel_map = {
        "triangular": "TriangularKernel()",
        "uniform": "UniformKernel()",
        "rectangular": "UniformKernel()",  # Python name for uniform
    }
    jl_kernel = jl.seval(kernel_map.get(kernel, "TriangularKernel()"))

    # Map bandwidth method
    if isinstance(bandwidth, str):
        if bandwidth.lower() == "ik":
            jl_bandwidth_method = jl.IKBandwidth()
        elif bandwidth.lower() == "cct":
            jl_bandwidth_method = jl.CCTBandwidth()
        else:
            raise ValueError(f"Unknown bandwidth method: {bandwidth}")

        # Solve with automatic bandwidth
        solution = jl.solve(
            problem,
            jl.SharpRDD(
                bandwidth_method=jl_bandwidth_method,
                kernel=jl_kernel,
                run_density_test=run_density_test
            )
        )
    else:
        # Fixed bandwidth - use CCT for now (bandwidth selection happens internally)
        jl_bandwidth_method = jl.CCTBandwidth()
        solution = jl.solve(
            problem,
            jl.SharpRDD(
                bandwidth_method=jl_bandwidth_method,
                kernel=jl_kernel,
                run_density_test=run_density_test
            )
        )

    # Extract results
    result = {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "p_value": float(solution.p_value),
        "bandwidth": float(solution.bandwidth),
        "bandwidth_bias": float(solution.bandwidth_bias) if solution.bandwidth_bias is not None else None,
        "kernel": str(solution.kernel),
        "n_eff_left": int(solution.n_eff_left),
        "n_eff_right": int(solution.n_eff_right),
        "bias_corrected": bool(solution.bias_corrected),
    }

    return result


def julia_rdd_bandwidth_ik(
    outcomes: np.ndarray,
    running_var: np.ndarray,
    cutoff: float,
    kernel: str = "triangular",
) -> float:
    """
    Compute IK bandwidth using Julia.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    running_var : np.ndarray
        Running variable X (n,)
    cutoff : float
        Cutoff threshold
    kernel : str, default="triangular"
        Kernel function

    Returns
    -------
    float
        IK optimal bandwidth
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert arrays
    jl_outcomes = jl.collect(outcomes)
    jl_running_var = jl.collect(running_var)
    treatment_bool = running_var >= cutoff
    jl_treatment = jl.collect(treatment_bool)

    # Create RDDProblem
    problem = jl.RDDProblem(
        jl_outcomes,
        jl_running_var,
        jl_treatment,
        float(cutoff),
        jl.seval("nothing"),
        jl.seval("(alpha=0.05,)")
    )

    # Call bandwidth selection
    h = jl.select_bandwidth(problem, jl.IKBandwidth())

    return float(h)


def julia_rdd_bandwidth_cct(
    outcomes: np.ndarray,
    running_var: np.ndarray,
    cutoff: float,
    kernel: str = "triangular",
) -> tuple:
    """
    Compute CCT bandwidth using Julia.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    running_var : np.ndarray
        Running variable X (n,)
    cutoff : float
        Cutoff threshold
    kernel : str, default="triangular"
        Kernel function

    Returns
    -------
    tuple
        (h_main, h_bias) - CCT optimal bandwidths
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert arrays
    jl_outcomes = jl.collect(outcomes)
    jl_running_var = jl.collect(running_var)
    treatment_bool = running_var >= cutoff
    jl_treatment = jl.collect(treatment_bool)

    # Create RDDProblem
    problem = jl.RDDProblem(
        jl_outcomes,
        jl_running_var,
        jl_treatment,
        float(cutoff),
        jl.seval("nothing"),
        jl.seval("(alpha=0.05,)")
    )

    # Call bandwidth selection - CCT returns tuple (h_main, h_bias)
    result = jl.select_bandwidth(problem, jl.CCTBandwidth())

    # Result is a tuple in Julia
    h_main = float(result[0])
    h_bias = float(result[1])

    return (h_main, h_bias)


def julia_fuzzy_rdd(
    outcomes: np.ndarray,
    running_var: np.ndarray,
    treatment: np.ndarray,
    cutoff: float,
    bandwidth: Union[float, str] = "cct",
    kernel: str = "triangular",
    alpha: float = 0.05,
    run_density_test: bool = False,
) -> Dict[str, Union[float, int, bool, None]]:
    """
    Call Julia FuzzyRDD via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    running_var : np.ndarray
        Running variable X (n,)
    treatment : np.ndarray
        Actual treatment received D (n,) - may differ from eligibility Z
    cutoff : float
        Cutoff threshold
    bandwidth : float or str, default="cct"
        Bandwidth: "ik", "cct", or numeric value
    kernel : str, default="triangular"
        Kernel function: "triangular", "uniform", or "rectangular" (mapped to uniform)
    alpha : float, default=0.05
        Significance level
    run_density_test : bool, default=False
        Run McCrary density test (disabled for cross-validation to avoid warnings)

    Returns
    -------
    dict
        Julia result with estimate, se, ci, p_value, first_stage_fstat, compliance_rate, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert numpy arrays to Julia arrays
    jl_outcomes = jl.collect(outcomes)
    jl_running_var = jl.collect(running_var)
    jl_treatment = jl.collect(treatment.astype(np.float64))  # Fuzzy RDD needs Float64 treatment

    # Create RDDProblem
    problem = jl.RDDProblem(
        jl_outcomes,
        jl_running_var,
        jl_treatment,
        float(cutoff),
        jl.seval("nothing"),  # No covariates
        jl.seval(f"(alpha={alpha},)")
    )

    # Map kernel name: Python "rectangular" -> Julia "uniform"
    kernel_map = {
        "triangular": "TriangularKernel()",
        "uniform": "UniformKernel()",
        "rectangular": "UniformKernel()",  # Python name for uniform
    }
    jl_kernel = jl.seval(kernel_map.get(kernel, "TriangularKernel()"))

    # Map bandwidth method
    if isinstance(bandwidth, str):
        if bandwidth.lower() == "ik":
            jl_bandwidth_method = jl.IKBandwidth()
        elif bandwidth.lower() == "cct":
            jl_bandwidth_method = jl.CCTBandwidth()
        else:
            raise ValueError(f"Unknown bandwidth method: {bandwidth}")

        # Solve with automatic bandwidth
        solution = jl.solve(
            problem,
            jl.FuzzyRDD(
                bandwidth_method=jl_bandwidth_method,
                kernel=jl_kernel,
                run_density_test=run_density_test
            )
        )
    else:
        # Fixed bandwidth - use IK for now (bandwidth selection happens internally)
        jl_bandwidth_method = jl.IKBandwidth()
        solution = jl.solve(
            problem,
            jl.FuzzyRDD(
                bandwidth_method=jl_bandwidth_method,
                kernel=jl_kernel,
                run_density_test=run_density_test
            )
        )

    # Extract results
    result = {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "p_value": float(solution.p_value),
        "bandwidth": float(solution.bandwidth),
        "kernel": str(solution.kernel),
        "n_eff_left": int(solution.n_eff_left),
        "n_eff_right": int(solution.n_eff_right),
        "first_stage_fstat": float(solution.first_stage_fstat),
        "compliance_rate": float(solution.compliance_rate),
        "weak_instrument_warning": bool(solution.weak_instrument_warning),
        "retcode": str(solution.retcode),
    }

    return result


# =============================================================================
# DiD (Difference-in-Differences) Functions
# =============================================================================


def julia_classic_did(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    post: np.ndarray,
    unit_id: np.ndarray,
    time: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    cluster_se: bool = True,
    test_parallel_trends: bool = False,
) -> Dict[str, Union[float, int, bool, None]]:
    """
    Call Julia ClassicDiD (2×2 Difference-in-Differences) via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    treatment : np.ndarray
        Treatment group indicator - time-invariant (n,) - 0/1 or bool
    post : np.ndarray
        Post-treatment period indicator (n,) - 0/1 or bool
    unit_id : np.ndarray
        Unit identifiers (n,) - int
    time : np.ndarray, optional
        Time period identifiers (n,) - int
    alpha : float, default=0.05
        Significance level
    cluster_se : bool, default=True
        Use cluster-robust standard errors (by unit_id)
    test_parallel_trends : bool, default=False
        Run pre-treatment trends test (requires ≥2 pre-periods)

    Returns
    -------
    dict
        Julia result with estimate, se, ci_lower, ci_upper, p_value, t_stat, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert numpy arrays to Julia arrays
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_post = jl.collect(post.astype(bool))
    jl_unit_id = jl.collect(unit_id.astype(np.int64))
    jl_time = jl.collect(time.astype(np.int64)) if time is not None else jl.seval("nothing")

    # Create DiDProblem
    problem = jl.DiDProblem(
        jl_outcomes,
        jl_treatment,
        jl_post,
        jl_unit_id,
        jl_time,
        jl.seval(f"(alpha={alpha},)")
    )

    # Create ClassicDiD estimator
    estimator = jl.ClassicDiD(
        cluster_se=cluster_se,
        test_parallel_trends=test_parallel_trends
    )

    # Solve
    solution = jl.solve(problem, estimator)

    # Extract results
    result = {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "p_value": float(solution.p_value),
        "t_stat": float(solution.t_stat),
        "df": int(solution.df),
        "n_obs": int(solution.n_obs),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
        "retcode": str(solution.retcode),
    }

    # Add parallel trends test if available
    if solution.parallel_trends_test is not None:
        result["parallel_trends_pvalue"] = float(solution.parallel_trends_test.p_value)
        result["parallel_trends_passes"] = bool(solution.parallel_trends_test.passes)

    return result


def julia_event_study(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    post: np.ndarray,
    unit_id: np.ndarray,
    time: np.ndarray,
    alpha: float = 0.05,
    n_leads: Optional[int] = None,
    n_lags: Optional[int] = None,
    omit_period: int = -1,
    cluster_se: bool = True,
) -> Dict[str, Union[float, int, bool, list, None]]:
    """
    Call Julia EventStudy (dynamic DiD with leads/lags) via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    treatment : np.ndarray
        Treatment group indicator - time-invariant (n,)
    post : np.ndarray
        Post-treatment period indicator (n,)
    unit_id : np.ndarray
        Unit identifiers (n,)
    time : np.ndarray
        Time period identifiers (n,) - REQUIRED for event study
    alpha : float, default=0.05
        Significance level
    n_leads : int, optional
        Number of pre-treatment periods to include (default: auto-detect)
    n_lags : int, optional
        Number of post-treatment periods to include (default: auto-detect)
    omit_period : int, default=-1
        Period to omit for normalization (typically -1)
    cluster_se : bool, default=True
        Use cluster-robust standard errors

    Returns
    -------
    dict
        Julia result with estimate, se, coefficients_pre, coefficients_post, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert numpy arrays to Julia arrays
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_post = jl.collect(post.astype(bool))
    jl_unit_id = jl.collect(unit_id.astype(np.int64))
    jl_time = jl.collect(time.astype(np.int64))

    # Create DiDProblem
    problem = jl.DiDProblem(
        jl_outcomes,
        jl_treatment,
        jl_post,
        jl_unit_id,
        jl_time,
        jl.seval(f"(alpha={alpha},)")
    )

    # Create EventStudy estimator
    n_leads_jl = n_leads if n_leads is not None else jl.seval("nothing")
    n_lags_jl = n_lags if n_lags is not None else jl.seval("nothing")

    estimator = jl.EventStudy(
        n_leads=n_leads_jl,
        n_lags=n_lags_jl,
        omit_period=omit_period,
        cluster_se=cluster_se
    )

    # Solve
    solution = jl.solve(problem, estimator)

    # Extract results - EventStudy has different solution structure
    result = {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "p_value": float(solution.p_value),
        "t_stat": float(solution.t_stat),
        "df": int(solution.df),
        "n_obs": int(solution.n_obs),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
        "retcode": str(solution.retcode),
    }

    return result


def julia_staggered_twfe(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    time: np.ndarray,
    unit_id: np.ndarray,
    treatment_time: np.ndarray,
    alpha: float = 0.05,
    cluster_se: bool = True,
) -> Dict[str, Union[float, int, bool, None]]:
    """
    Call Julia StaggeredTWFE (Two-Way Fixed Effects for staggered adoption) via juliacall.

    WARNING: TWFE is BIASED with heterogeneous treatment effects. Use
    julia_callaway_santanna or julia_sun_abraham instead.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    treatment : np.ndarray
        Binary treatment indicator D_it (n,) - 1 if unit i treated at time t
    time : np.ndarray
        Time period identifiers (n,)
    unit_id : np.ndarray
        Unit identifiers (n,)
    treatment_time : np.ndarray
        Treatment time per unit (n_units,) - np.inf for never-treated
    alpha : float, default=0.05
        Significance level
    cluster_se : bool, default=True
        Use cluster-robust standard errors

    Returns
    -------
    dict
        Julia result with estimate, se, ci_lower, ci_upper, p_value, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert numpy arrays to Julia arrays
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_time = jl.collect(time.astype(np.int64))
    jl_unit_id = jl.collect(unit_id.astype(np.int64))
    jl_treatment_time = jl.collect(treatment_time.astype(np.float64))

    # Create StaggeredDiDProblem
    problem = jl.StaggeredDiDProblem(
        jl_outcomes,
        jl_treatment,
        jl_time,
        jl_unit_id,
        jl_treatment_time,
        jl.seval(f"(alpha={alpha},)")
    )

    # Create StaggeredTWFE estimator
    estimator = jl.StaggeredTWFE(cluster_se=cluster_se)

    # Solve
    solution = jl.solve(problem, estimator)

    # Extract results
    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "p_value": float(solution.p_value),
        "t_stat": float(solution.t_stat),
        "df": int(solution.df),
        "n_obs": int(solution.n_obs),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
        "retcode": str(solution.retcode),
    }


def julia_callaway_santanna(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    time: np.ndarray,
    unit_id: np.ndarray,
    treatment_time: np.ndarray,
    alpha: float = 0.05,
    aggregation: str = "simple",
    control_group: str = "nevertreated",
    n_bootstrap: int = 250,
    random_seed: Optional[int] = None,
) -> Dict[str, Union[float, int, str, list, None]]:
    """
    Call Julia CallawaySantAnna (2021) estimator via juliacall.

    This estimator is unbiased with heterogeneous treatment effects.
    It computes ATT(g,t) for each cohort-time cell and aggregates them.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    treatment : np.ndarray
        Binary treatment indicator D_it (n,)
    time : np.ndarray
        Time period identifiers (n,)
    unit_id : np.ndarray
        Unit identifiers (n,)
    treatment_time : np.ndarray
        Treatment time per unit (n_units,) - np.inf for never-treated
    alpha : float, default=0.05
        Significance level
    aggregation : str, default="simple"
        Aggregation scheme: "simple", "dynamic", or "group"
    control_group : str, default="nevertreated"
        Control group: "nevertreated" or "notyettreated"
    n_bootstrap : int, default=250
        Number of bootstrap samples for inference
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Julia result with att, se, ci_lower, ci_upper, p_value, att_gt, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert numpy arrays to Julia arrays
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_time = jl.collect(time.astype(np.int64))
    jl_unit_id = jl.collect(unit_id.astype(np.int64))
    jl_treatment_time = jl.collect(treatment_time.astype(np.float64))

    # Create StaggeredDiDProblem
    problem = jl.StaggeredDiDProblem(
        jl_outcomes,
        jl_treatment,
        jl_time,
        jl_unit_id,
        jl_treatment_time,
        jl.seval(f"(alpha={alpha},)")
    )

    # Create CallawaySantAnna estimator
    random_seed_jl = random_seed if random_seed is not None else jl.seval("nothing")
    estimator = jl.CallawaySantAnna(
        aggregation=jl.seval(f':{aggregation}'),
        control_group=jl.seval(f':{control_group}'),
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed_jl
    )

    # Solve
    solution = jl.solve(problem, estimator)

    # Extract ATT(g,t) results
    att_gt_list = []
    for item in solution.att_gt:
        att_gt_list.append({
            "cohort": int(item.cohort),
            "time": int(item.time),
            "event_time": int(item.event_time),
            "att": float(item.att),
            "weight": int(item.weight),
            "n_treated": int(item.n_treated),
            "n_control": int(item.n_control),
        })

    # Extract results
    result = {
        "att": float(solution.att),
        "se": float(solution.se),
        "t_stat": float(solution.t_stat),
        "p_value": float(solution.p_value),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "att_gt": att_gt_list,
        "control_group": str(solution.control_group),
        "n_bootstrap": int(solution.n_bootstrap),
        "n_cohorts": int(solution.n_cohorts),
        "n_obs": int(solution.n_obs),
        "retcode": str(solution.retcode),
    }

    # Handle aggregation-specific results
    if aggregation == "simple":
        result["aggregated"] = float(solution.aggregated)
    elif aggregation == "dynamic":
        # Dynamic aggregation returns Dict{Int, Float64}
        agg_dict = {}
        for k in jl.keys(solution.aggregated):
            agg_dict[int(k)] = float(solution.aggregated[k])
        result["aggregated"] = agg_dict
    elif aggregation == "group":
        # Group aggregation returns Dict{Int, Float64}
        agg_dict = {}
        for k in jl.keys(solution.aggregated):
            agg_dict[int(k)] = float(solution.aggregated[k])
        result["aggregated"] = agg_dict

    return result


def julia_sun_abraham(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    time: np.ndarray,
    unit_id: np.ndarray,
    treatment_time: np.ndarray,
    alpha: float = 0.05,
    cluster_se: bool = True,
) -> Dict[str, Union[float, int, bool, list, None]]:
    """
    Call Julia SunAbraham (2021) interaction-weighted estimator via juliacall.

    This estimator is unbiased with heterogeneous treatment effects.
    Uses cohort × event time interactions with proper weighting.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    treatment : np.ndarray
        Binary treatment indicator D_it (n,)
    time : np.ndarray
        Time period identifiers (n,)
    unit_id : np.ndarray
        Unit identifiers (n,)
    treatment_time : np.ndarray
        Treatment time per unit (n_units,) - np.inf for never-treated
    alpha : float, default=0.05
        Significance level
    cluster_se : bool, default=True
        Use cluster-robust standard errors

    Returns
    -------
    dict
        Julia result with att, se, ci_lower, ci_upper, cohort_effects, weights, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert numpy arrays to Julia arrays
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_time = jl.collect(time.astype(np.int64))
    jl_unit_id = jl.collect(unit_id.astype(np.int64))
    jl_treatment_time = jl.collect(treatment_time.astype(np.float64))

    # Create StaggeredDiDProblem
    problem = jl.StaggeredDiDProblem(
        jl_outcomes,
        jl_treatment,
        jl_time,
        jl_unit_id,
        jl_treatment_time,
        jl.seval(f"(alpha={alpha},)")
    )

    # Create SunAbraham estimator
    estimator = jl.SunAbraham(alpha=alpha, cluster_se=cluster_se)

    # Solve
    solution = jl.solve(problem, estimator)

    # Extract cohort effects
    cohort_effects_list = []
    for item in solution.cohort_effects:
        cohort_effects_list.append({
            "cohort": int(item.cohort),
            "event_time": int(item.event_time),
            "coef": float(item.coef),
            "se": float(item.se),
            "t_stat": float(item.t_stat),
            "p_value": float(item.p_value),
            "ci_lower": float(item.ci_lower),
            "ci_upper": float(item.ci_upper),
        })

    # Extract weights
    weights_list = []
    for item in solution.weights:
        weights_list.append({
            "cohort": int(item.cohort),
            "event_time": int(item.event_time),
            "weight": float(item.weight),
            "n_obs": int(item.n_obs),
        })

    # Extract results
    return {
        "att": float(solution.att),
        "se": float(solution.se),
        "t_stat": float(solution.t_stat),
        "p_value": float(solution.p_value),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "cohort_effects": cohort_effects_list,
        "weights": weights_list,
        "n_obs": int(solution.n_obs),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
        "n_cohorts": int(solution.n_cohorts),
        "cluster_se_used": bool(solution.cluster_se_used),
        "retcode": str(solution.retcode),
    }


# =============================================================================
# PSM (Propensity Score Matching) Functions
# =============================================================================


def julia_psm_nearest_neighbor(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    M: int = 1,
    with_replacement: bool = False,
    caliper: float = np.inf,
    alpha: float = 0.05,
    variance_method: str = "abadie_imbens",
) -> Dict[str, Union[float, int, str, list, None]]:
    """
    Call Julia NearestNeighborPSM via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    treatment : np.ndarray
        Binary treatment indicator (n,) - 0/1 or bool
    covariates : np.ndarray
        Covariate matrix X (n, p) for propensity score estimation
    M : int, default=1
        Number of matches per treated unit
    with_replacement : bool, default=False
        Allow reusing control units
    caliper : float, default=np.inf
        Maximum propensity score distance (np.inf = no restriction)
    alpha : float, default=0.05
        Significance level for confidence intervals
    variance_method : str, default="abadie_imbens"
        Variance estimator: "abadie_imbens" or "bootstrap"

    Returns
    -------
    dict
        Julia result with estimate, se, ci_lower, ci_upper, n_treated, n_control,
        n_matched, propensity_scores, matched_indices, balance_metrics, retcode
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert numpy arrays to Julia arrays
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_covariates = jl.seval("Matrix")(covariates.astype(np.float64))

    # Create PSMProblem
    problem = jl.PSMProblem(
        jl_outcomes,
        jl_treatment,
        jl_covariates,
        jl.seval(f"(alpha={alpha},)")
    )

    # Create NearestNeighborPSM estimator
    # Convert variance_method string to Julia Symbol
    variance_symbol = jl.seval(f":{variance_method}")
    caliper_jl = caliper if not np.isinf(caliper) else jl.seval("Inf")

    estimator = jl.NearestNeighborPSM(
        M=M,
        with_replacement=with_replacement,
        caliper=caliper_jl,
        variance_method=variance_symbol
    )

    # Solve
    solution = jl.solve(problem, estimator)

    # Extract propensity scores
    propensity_scores = np.array([float(p) for p in solution.propensity_scores])

    # Extract matched indices (list of tuples)
    matched_indices = []
    for pair in solution.matched_indices:
        matched_indices.append((int(pair[0]), int(pair[1])))  # Python tuple indexing

    # Extract balance metrics (vectors, one value per covariate)
    balance = solution.balance_metrics
    try:
        balance_dict = {
            "smd_before": [float(x) for x in balance.smd_before] if hasattr(balance, 'smd_before') else None,
            "smd_after": [float(x) for x in balance.smd_after] if hasattr(balance, 'smd_after') else None,
            "vr_before": [float(x) for x in balance.vr_before] if hasattr(balance, 'vr_before') else None,
            "vr_after": [float(x) for x in balance.vr_after] if hasattr(balance, 'vr_after') else None,
        }
    except Exception:
        balance_dict = {"smd_before": None, "smd_after": None, "vr_before": None, "vr_after": None}

    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
        "n_matched": int(solution.n_matched),
        "propensity_scores": propensity_scores,
        "matched_indices": matched_indices,
        "balance_metrics": balance_dict,
        "retcode": str(solution.retcode),
    }


# =============================================================================
# Observational IPW/DR Wrappers (Session 34)
# =============================================================================


def julia_observational_ipw(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    propensity: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    trim_threshold: float = 0.01,
    stabilize: bool = False,
) -> Dict[str, Union[float, int, np.ndarray, str]]:
    """
    Call Julia ObservationalIPW via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes (n,)
    treatment : np.ndarray
        Binary treatment (0/1) (n,)
    covariates : np.ndarray
        Covariate matrix (n, p)
    propensity : np.ndarray, optional
        Pre-computed propensity scores. If None, estimated via logistic regression.
    alpha : float, default=0.05
        Significance level
    trim_threshold : float, default=0.01
        Trim propensities outside (ε, 1-ε)
    stabilize : bool, default=False
        Use stabilized IPW weights

    Returns
    -------
    dict
        Julia result with estimate, se, ci, propensity scores, diagnostics
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert numpy arrays to Julia-compatible format (following PSM pattern)
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_covariates = jl.seval("Matrix")(covariates.astype(np.float64))

    # Handle propensity
    if propensity is not None:
        jl_propensity = jl.collect(propensity.astype(np.float64))
    else:
        jl_propensity = jl.seval("nothing")

    # Create ObservationalProblem
    problem = jl.ObservationalProblem(
        jl_outcomes,
        jl_treatment,
        jl_covariates,
        jl_propensity,
        jl.seval(f"(alpha={alpha}, trim_threshold={trim_threshold}, stabilize={str(stabilize).lower()})")
    )

    # Solve with ObservationalIPW
    solution = jl.solve(problem, jl.ObservationalIPW())

    # Extract propensity scores
    propensity_scores = np.array([float(p) for p in solution.propensity_scores])
    weights = np.array([float(w) for w in solution.weights])

    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "p_value": float(solution.p_value),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
        "n_trimmed": int(solution.n_trimmed),
        "propensity_scores": propensity_scores,
        "weights": weights,
        "propensity_auc": float(solution.propensity_auc),
        "propensity_mean_treated": float(solution.propensity_mean_treated),
        "propensity_mean_control": float(solution.propensity_mean_control),
        "stabilized": bool(solution.stabilized),
        "retcode": str(solution.retcode),
    }


def julia_doubly_robust(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    propensity: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    trim_threshold: float = 0.01,
) -> Dict[str, Union[float, int, np.ndarray, str]]:
    """
    Call Julia DoublyRobust (AIPW) via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes (n,)
    treatment : np.ndarray
        Binary treatment (0/1) (n,)
    covariates : np.ndarray
        Covariate matrix (n, p)
    propensity : np.ndarray, optional
        Pre-computed propensity scores. If None, estimated via logistic regression.
    alpha : float, default=0.05
        Significance level
    trim_threshold : float, default=0.01
        Trim propensities outside (ε, 1-ε)

    Returns
    -------
    dict
        Julia result with estimate, se, ci, outcome predictions, diagnostics
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert numpy arrays to Julia-compatible format (following PSM pattern)
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_covariates = jl.seval("Matrix")(covariates.astype(np.float64))

    # Handle propensity
    if propensity is not None:
        jl_propensity = jl.collect(propensity.astype(np.float64))
    else:
        jl_propensity = jl.seval("nothing")

    # Create ObservationalProblem
    problem = jl.ObservationalProblem(
        jl_outcomes,
        jl_treatment,
        jl_covariates,
        jl_propensity,
        jl.seval(f"(alpha={alpha}, trim_threshold={trim_threshold}, stabilize=false)")
    )

    # Solve with DoublyRobust
    solution = jl.solve(problem, jl.DoublyRobust())

    # Extract arrays
    propensity_scores = np.array([float(p) for p in solution.propensity_scores])
    mu0_predictions = np.array([float(m) for m in solution.mu0_predictions])
    mu1_predictions = np.array([float(m) for m in solution.mu1_predictions])

    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "p_value": float(solution.p_value),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
        "n_trimmed": int(solution.n_trimmed),
        "propensity_scores": propensity_scores,
        "mu0_predictions": mu0_predictions,
        "mu1_predictions": mu1_predictions,
        "propensity_auc": float(solution.propensity_auc),
        "mu0_r2": float(solution.mu0_r2),
        "mu1_r2": float(solution.mu1_r2),
        "retcode": str(solution.retcode),
    }


# =============================================================================
# CATE Meta-Learner Wrappers (Session 45)
# =============================================================================


def julia_s_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Call Julia S-Learner via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes (n,)
    treatment : np.ndarray
        Binary treatment (0/1) (n,)
    covariates : np.ndarray
        Covariate matrix (n, p)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with cate, ate, se, ci bounds
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_covariates = jl.seval("Matrix")(covariates.astype(np.float64))
    params = jl.seval(f"(alpha={alpha},)")

    problem = jl.CATEProblem(jl_outcomes, jl_treatment, jl_covariates, params)
    solution = jl.solve(problem, jl.SLearner())

    return {
        "cate": np.array([float(c) for c in solution.cate]),
        "ate": float(solution.ate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "method": str(solution.method),
        "retcode": str(solution.retcode),
    }


def julia_t_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Call Julia T-Learner via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes (n,)
    treatment : np.ndarray
        Binary treatment (0/1) (n,)
    covariates : np.ndarray
        Covariate matrix (n, p)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with cate, ate, se, ci bounds
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_covariates = jl.seval("Matrix")(covariates.astype(np.float64))
    params = jl.seval(f"(alpha={alpha},)")

    problem = jl.CATEProblem(jl_outcomes, jl_treatment, jl_covariates, params)
    solution = jl.solve(problem, jl.TLearner())

    return {
        "cate": np.array([float(c) for c in solution.cate]),
        "ate": float(solution.ate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "method": str(solution.method),
        "retcode": str(solution.retcode),
    }


def julia_x_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Call Julia X-Learner via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes (n,)
    treatment : np.ndarray
        Binary treatment (0/1) (n,)
    covariates : np.ndarray
        Covariate matrix (n, p)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with cate, ate, se, ci bounds
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_covariates = jl.seval("Matrix")(covariates.astype(np.float64))
    params = jl.seval(f"(alpha={alpha},)")

    problem = jl.CATEProblem(jl_outcomes, jl_treatment, jl_covariates, params)
    solution = jl.solve(problem, jl.XLearner())

    return {
        "cate": np.array([float(c) for c in solution.cate]),
        "ate": float(solution.ate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "method": str(solution.method),
        "retcode": str(solution.retcode),
    }


def julia_r_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Call Julia R-Learner via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes (n,)
    treatment : np.ndarray
        Binary treatment (0/1) (n,)
    covariates : np.ndarray
        Covariate matrix (n, p)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with cate, ate, se, ci bounds
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_covariates = jl.seval("Matrix")(covariates.astype(np.float64))
    params = jl.seval(f"(alpha={alpha},)")

    problem = jl.CATEProblem(jl_outcomes, jl_treatment, jl_covariates, params)
    solution = jl.solve(problem, jl.RLearner())

    return {
        "cate": np.array([float(c) for c in solution.cate]),
        "ate": float(solution.ate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "method": str(solution.method),
        "retcode": str(solution.retcode),
    }


def julia_double_ml(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    n_folds: int = 5,
    alpha: float = 0.05,
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Call Julia DoubleMachineLearning via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Observed outcomes (n,)
    treatment : np.ndarray
        Binary treatment (0/1) (n,)
    covariates : np.ndarray
        Covariate matrix (n, p)
    n_folds : int, default=5
        Number of cross-fitting folds
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with cate, ate, se, ci bounds
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_covariates = jl.seval("Matrix")(covariates.astype(np.float64))
    params = jl.seval(f"(alpha={alpha},)")

    problem = jl.CATEProblem(jl_outcomes, jl_treatment, jl_covariates, params)
    solution = jl.solve(problem, jl.DoubleMachineLearning(n_folds=n_folds))

    return {
        "cate": np.array([float(c) for c in solution.cate]),
        "ate": float(solution.ate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "method": str(solution.method),
        "retcode": str(solution.retcode),
    }


# =============================================================================
# SCM (Synthetic Control Methods) Wrappers (Session 47)
# =============================================================================


def julia_synthetic_control(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    treatment_period: int,
    covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    inference: str = "placebo",
    n_placebo: int = 100,
    covariate_weight: float = 1.0,
) -> Dict[str, Union[float, int, np.ndarray, str, None]]:
    """
    Call Julia SyntheticControl via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Panel data matrix (n_units, n_periods)
    treatment : np.ndarray
        Boolean array indicating treated units (n_units,)
    treatment_period : int
        First period of treatment (1-indexed, Julia-style)
    covariates : np.ndarray, optional
        Unit-level covariates (n_units, p)
    alpha : float, default=0.05
        Significance level
    inference : str, default="placebo"
        Inference method: "placebo", "bootstrap", or "none"
    n_placebo : int, default=100
        Number of placebo iterations
    covariate_weight : float, default=1.0
        Weight for covariates in matching

    Returns
    -------
    dict
        Julia result with estimate, se, ci, weights, pre_fit metrics, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert numpy arrays to Julia-compatible format
    jl_outcomes = jl.seval("Matrix")(outcomes.astype(np.float64))
    jl_treatment = jl.collect([bool(t) for t in treatment])

    if covariates is not None:
        jl_covariates = jl.seval("Matrix")(covariates.astype(np.float64))
    else:
        jl_covariates = jl.seval("nothing")

    # Create SCMProblem
    problem = jl.SCMProblem(
        jl_outcomes,
        jl_treatment,
        int(treatment_period),
        jl_covariates,
        jl.seval(f"(alpha={alpha},)")
    )

    # Create SyntheticControl estimator
    estimator = jl.SyntheticControl(
        inference=jl.seval(f":{inference}"),
        n_placebo=n_placebo,
        covariate_weight=covariate_weight
    )

    # Solve
    solution = jl.solve(problem, estimator)

    # Extract arrays
    weights = np.array([float(w) for w in solution.weights])
    synthetic_control = np.array([float(s) for s in solution.synthetic_control])
    treated_series = np.array([float(t) for t in solution.treated_series])
    gap = np.array([float(g) for g in solution.gap])

    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se) if not np.isnan(float(solution.se)) else None,
        "ci_lower": float(solution.ci_lower) if not np.isnan(float(solution.ci_lower)) else None,
        "ci_upper": float(solution.ci_upper) if not np.isnan(float(solution.ci_upper)) else None,
        "p_value": float(solution.p_value) if not np.isnan(float(solution.p_value)) else None,
        "weights": weights,
        "pre_rmse": float(solution.pre_rmse),
        "pre_r_squared": float(solution.pre_r_squared),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
        "n_pre_periods": int(solution.n_pre_periods),
        "n_post_periods": int(solution.n_post_periods),
        "synthetic_control": synthetic_control,
        "treated_series": treated_series,
        "gap": gap,
        "retcode": str(solution.retcode),
    }


def julia_augmented_scm(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    treatment_period: int,
    covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    inference: str = "jackknife",
    lambda_ridge: Optional[float] = None,
) -> Dict[str, Union[float, int, np.ndarray, str, None]]:
    """
    Call Julia AugmentedSC (Ben-Michael et al. 2021) via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Panel data matrix (n_units, n_periods)
    treatment : np.ndarray
        Boolean array indicating treated units (n_units,)
    treatment_period : int
        First period of treatment (1-indexed, Julia-style)
    covariates : np.ndarray, optional
        Unit-level covariates (n_units, p)
    alpha : float, default=0.05
        Significance level
    inference : str, default="jackknife"
        Inference method: "jackknife", "bootstrap", or "none"
    lambda_ridge : float, optional
        Ridge regression penalty. If None, selected via CV.

    Returns
    -------
    dict
        Julia result with estimate, se, ci, weights, pre_fit metrics, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert numpy arrays to Julia-compatible format
    jl_outcomes = jl.seval("Matrix")(outcomes.astype(np.float64))
    jl_treatment = jl.collect([bool(t) for t in treatment])

    if covariates is not None:
        jl_covariates = jl.seval("Matrix")(covariates.astype(np.float64))
    else:
        jl_covariates = jl.seval("nothing")

    # Create SCMProblem
    problem = jl.SCMProblem(
        jl_outcomes,
        jl_treatment,
        int(treatment_period),
        jl_covariates,
        jl.seval(f"(alpha={alpha},)")
    )

    # Create AugmentedSC estimator
    # Note: Julia kwarg is 'lambda' but Python reserves that keyword, so we use seval
    if lambda_ridge is not None:
        estimator = jl.seval(f"AugmentedSC(inference=:{inference}, lambda={lambda_ridge})")
    else:
        estimator = jl.seval(f"AugmentedSC(inference=:{inference})")

    # Solve
    solution = jl.solve(problem, estimator)

    # Extract arrays
    weights = np.array([float(w) for w in solution.weights])
    synthetic_control = np.array([float(s) for s in solution.synthetic_control])
    treated_series = np.array([float(t) for t in solution.treated_series])
    gap = np.array([float(g) for g in solution.gap])

    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se) if not np.isnan(float(solution.se)) else None,
        "ci_lower": float(solution.ci_lower) if not np.isnan(float(solution.ci_lower)) else None,
        "ci_upper": float(solution.ci_upper) if not np.isnan(float(solution.ci_upper)) else None,
        "p_value": float(solution.p_value) if not np.isnan(float(solution.p_value)) else None,
        "weights": weights,
        "pre_rmse": float(solution.pre_rmse),
        "pre_r_squared": float(solution.pre_r_squared),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
        "n_pre_periods": int(solution.n_pre_periods),
        "n_post_periods": int(solution.n_post_periods),
        "synthetic_control": synthetic_control,
        "treated_series": treated_series,
        "gap": gap,
        "retcode": str(solution.retcode),
    }


# =============================================================================
# Sensitivity Analysis Wrappers (Session 51)
# =============================================================================


def julia_e_value(
    estimate: float,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    effect_type: str = "rr",
    baseline_risk: Optional[float] = None,
) -> Dict[str, Union[float, str]]:
    """
    Call Julia E-value sensitivity analysis via juliacall.

    Parameters
    ----------
    estimate : float
        Point estimate of the effect
    ci_lower : float, optional
        Lower bound of confidence interval
    ci_upper : float, optional
        Upper bound of confidence interval
    effect_type : str, default="rr"
        Effect type: "rr", "or", "hr", "smd", or "ate"
    baseline_risk : float, optional
        Baseline risk (required for ATE effect type)

    Returns
    -------
    dict
        Julia result with e_value, e_value_ci, rr_equivalent, interpretation
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Build problem arguments
    kwargs_parts = [f"effect_type=:{effect_type}"]
    if ci_lower is not None:
        kwargs_parts.append(f"ci_lower={ci_lower}")
    if ci_upper is not None:
        kwargs_parts.append(f"ci_upper={ci_upper}")
    if baseline_risk is not None:
        kwargs_parts.append(f"baseline_risk={baseline_risk}")

    kwargs_str = ", ".join(kwargs_parts)
    problem = jl.seval(f"EValueProblem({estimate}; {kwargs_str})")

    # Solve
    solution = jl.solve(problem, jl.EValue())

    return {
        "e_value": float(solution.e_value),
        "e_value_ci": float(solution.e_value_ci),
        "rr_equivalent": float(solution.rr_equivalent),
        "effect_type": str(solution.effect_type),
        "interpretation": str(solution.interpretation),
    }


def julia_rosenbaum_bounds(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    gamma_range: tuple = (1.0, 3.0),
    n_gamma: int = 20,
    alpha: float = 0.05,
) -> Dict[str, Union[float, int, np.ndarray, str, None]]:
    """
    Call Julia Rosenbaum bounds sensitivity analysis via juliacall.

    Parameters
    ----------
    treated_outcomes : np.ndarray
        Outcomes for treated units in matched pairs
    control_outcomes : np.ndarray
        Outcomes for control units in matched pairs
    gamma_range : tuple, default=(1.0, 3.0)
        Range of Gamma values to evaluate
    n_gamma : int, default=20
        Number of Gamma values in grid
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with gamma_values, p_upper, p_lower, gamma_critical, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert numpy arrays to Julia vectors
    jl_treated = jl.collect(treated_outcomes.astype(np.float64))
    jl_control = jl.collect(control_outcomes.astype(np.float64))

    # Create RosenbaumProblem
    problem = jl.RosenbaumProblem(
        jl_treated,
        jl_control,
        gamma_range=jl.seval(f"({gamma_range[0]}, {gamma_range[1]})"),
        n_gamma=n_gamma,
        alpha=alpha
    )

    # Solve
    solution = jl.solve(problem, jl.RosenbaumBounds())

    # Extract arrays
    gamma_values = np.array([float(g) for g in solution.gamma_values])
    p_upper = np.array([float(p) for p in solution.p_upper])
    p_lower = np.array([float(p) for p in solution.p_lower])

    # Handle gamma_critical (may be nothing)
    gamma_critical = None
    if solution.gamma_critical is not None:
        try:
            gamma_critical = float(solution.gamma_critical)
        except (TypeError, ValueError):
            gamma_critical = None

    return {
        "gamma_values": gamma_values,
        "p_upper": p_upper,
        "p_lower": p_lower,
        "gamma_critical": gamma_critical,
        "observed_statistic": float(solution.observed_statistic),
        "n_pairs": int(solution.n_pairs),
        "alpha": float(solution.alpha),
        "interpretation": str(solution.interpretation),
    }


# =============================================================================
# McCrary Density Test (Session 57)
# =============================================================================


def julia_mccrary_test(
    x: np.ndarray,
    cutoff: float,
    bandwidth: Optional[float] = None,
    alpha: float = 0.05,
) -> Dict[str, Union[float, bool, str, int]]:
    """
    Call Julia McCrary density test via juliacall.

    Session 57: Tests for manipulation of running variable at cutoff.

    Parameters
    ----------
    x : np.ndarray
        Running variable values
    cutoff : float
        RDD cutoff value
    bandwidth : float, optional
        Bandwidth for density estimation (None = automatic ROT)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with theta, se, p_value, passes, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert to Julia
    jl_x = jl.collect(x.astype(np.float64))
    jl_cutoff = float(cutoff)
    jl_bandwidth = None if bandwidth is None else float(bandwidth)

    # Create McCraryProblem
    problem = jl.McCraryProblem(
        jl_x,
        jl_cutoff,
        jl_bandwidth,
        jl.seval(f"(alpha={alpha},)")
    )

    # Solve
    solution = jl.solve(problem, jl.McCraryDensityTest())

    return {
        "theta": float(solution.theta),
        "se": float(solution.se),
        "z_stat": float(solution.z_stat),
        "p_value": float(solution.p_value),
        "passes": bool(solution.passes),
        "f_left": float(solution.f_left),
        "f_right": float(solution.f_right),
        "bandwidth": float(solution.bandwidth),
        "n_left": int(solution.n_left),
        "n_right": int(solution.n_right),
        "interpretation": str(solution.interpretation),
    }
