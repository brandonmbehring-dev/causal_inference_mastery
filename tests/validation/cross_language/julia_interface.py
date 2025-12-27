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
    # Note: Must explicitly type as Vector{Bool} to match SCMProblem signature
    jl_treatment = jl.seval("Vector{Bool}")([bool(t) for t in treatment])

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
    # Note: Must explicitly type as Vector{Bool} to match SCMProblem signature
    jl_treatment = jl.seval("Vector{Bool}")([bool(t) for t in treatment])

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


# =============================================================================
# RKD Estimators (Session 74)
# =============================================================================


def julia_sharp_rkd(
    y: np.ndarray,
    x: np.ndarray,
    d: np.ndarray,
    cutoff: float,
    bandwidth: Optional[float] = None,
    kernel: str = "triangular",
    polynomial_order: int = 1,
    alpha: float = 0.05,
) -> Dict[str, Union[float, int, str]]:
    """
    Call Julia Sharp RKD estimator via juliacall.

    Session 74: Regression Kink Design estimation.

    Parameters
    ----------
    y : np.ndarray
        Outcome variable
    x : np.ndarray
        Running variable
    d : np.ndarray
        Treatment variable (with kink at cutoff)
    cutoff : float
        Kink point
    bandwidth : float, optional
        Bandwidth for local polynomial (None = automatic)
    kernel : str, default="triangular"
        Kernel function
    polynomial_order : int, default=1
        Polynomial order
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with estimate, se, ci, slopes, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert to Julia
    jl_y = jl.collect(y.astype(np.float64))
    jl_x = jl.collect(x.astype(np.float64))
    jl_d = jl.collect(d.astype(np.float64))
    jl_cutoff = float(cutoff)

    # Create RKDProblem
    problem = jl.RKDProblem(
        jl_y, jl_x, jl_d, jl_cutoff,
        None,  # No covariates
        jl.seval(f"(alpha={alpha},)")
    )

    # Create estimator
    if bandwidth is None:
        estimator = jl.SharpRKD(
            bandwidth=None,
            kernel=jl.seval(f":{kernel}"),
            polynomial_order=polynomial_order,
            alpha=alpha
        )
    else:
        estimator = jl.SharpRKD(
            bandwidth=float(bandwidth),
            kernel=jl.seval(f":{kernel}"),
            polynomial_order=polynomial_order,
            alpha=alpha
        )

    # Solve
    solution = jl.solve(problem, estimator)

    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "t_stat": float(solution.t_stat),
        "p_value": float(solution.p_value),
        "bandwidth": float(solution.bandwidth),
        "n_left": int(solution.n_eff_left),
        "n_right": int(solution.n_eff_right),
        "outcome_slope_left": float(solution.outcome_slope_left),
        "outcome_slope_right": float(solution.outcome_slope_right),
        "outcome_kink": float(solution.outcome_kink),
        "treatment_slope_left": float(solution.treatment_slope_left),
        "treatment_slope_right": float(solution.treatment_slope_right),
        "treatment_kink": float(solution.treatment_kink),
        "polynomial_order": int(solution.polynomial_order),
        "retcode": str(solution.retcode),
    }


# =============================================================================
# Bunching Estimation (Session 78)
# =============================================================================


def julia_bunching_estimator(
    data: np.ndarray,
    kink_point: float,
    bunching_width: float,
    t1_rate: Optional[float] = None,
    t2_rate: Optional[float] = None,
    n_bins: int = 50,
    polynomial_order: int = 7,
    n_bootstrap: int = 200,
) -> Dict[str, Union[float, int, bool, str, np.ndarray]]:
    """
    Call Julia Saez (2010) bunching estimator via juliacall.

    Session 78: Bunching estimation for detecting behavioral responses at kinks.

    Parameters
    ----------
    data : np.ndarray
        Observed data (e.g., reported income)
    kink_point : float
        Location of the kink
    bunching_width : float
        Half-width of bunching region
    t1_rate : float, optional
        Marginal rate below kink (for elasticity calculation)
    t2_rate : float, optional
        Marginal rate above kink (for elasticity calculation)
    n_bins : int, default=50
        Number of histogram bins
    polynomial_order : int, default=7
        Polynomial order for counterfactual
    n_bootstrap : int, default=200
        Bootstrap iterations for SE

    Returns
    -------
    dict
        Julia result with excess_mass, elasticity, se, counterfactual, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert to Julia
    jl_data = jl.collect(data.astype(np.float64))

    # Create BunchingProblem
    if t1_rate is not None and t2_rate is not None:
        problem = jl.BunchingProblem(
            jl_data,
            float(kink_point),
            float(bunching_width),
            t1_rate=float(t1_rate),
            t2_rate=float(t2_rate),
        )
    else:
        problem = jl.BunchingProblem(
            jl_data,
            float(kink_point),
            float(bunching_width),
        )

    # Create estimator
    estimator = jl.SaezBunching(
        n_bins=n_bins,
        polynomial_order=polynomial_order,
        n_bootstrap=n_bootstrap,
    )

    # Solve
    solution = jl.solve(problem, estimator)

    # Extract counterfactual result
    cf = solution.counterfactual
    bunching_region = solution.bunching_region

    return {
        "excess_mass": float(solution.excess_mass),
        "excess_mass_se": float(solution.excess_mass_se),
        "excess_mass_count": float(solution.excess_mass_count),
        "elasticity": float(solution.elasticity),
        "elasticity_se": float(solution.elasticity_se),
        "kink_point": float(solution.kink_point),
        "bunching_lower": float(bunching_region[0]),
        "bunching_upper": float(bunching_region[1]),
        "t1_rate": float(solution.t1_rate) if solution.t1_rate is not None else None,
        "t2_rate": float(solution.t2_rate) if solution.t2_rate is not None else None,
        "n_obs": int(solution.n_obs),
        "n_bootstrap": int(solution.n_bootstrap),
        "convergence": bool(solution.convergence),
        "r_squared": float(cf.r_squared),
        "polynomial_order": int(cf.polynomial_order),
        "bin_centers": np.array([float(x) for x in cf.bin_centers]),
        "actual_counts": np.array([float(x) for x in cf.actual_counts]),
        "counterfactual_counts": np.array([float(x) for x in cf.counterfactual_counts]),
    }


def julia_polynomial_counterfactual(
    bin_centers: np.ndarray,
    counts: np.ndarray,
    bunching_lower: float,
    bunching_upper: float,
    polynomial_order: int = 7,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Call Julia polynomial_counterfactual directly.

    Parameters
    ----------
    bin_centers : np.ndarray
        Centers of histogram bins
    counts : np.ndarray
        Observed counts in each bin
    bunching_lower : float
        Lower bound of bunching region
    bunching_upper : float
        Upper bound of bunching region
    polynomial_order : int, default=7
        Polynomial order for fit

    Returns
    -------
    dict
        Counterfactual counts, coefficients, R-squared
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl_centers = jl.collect(bin_centers.astype(np.float64))
    jl_counts = jl.collect(counts.astype(np.float64))

    result = jl.polynomial_counterfactual(
        jl_centers,
        jl_counts,
        float(bunching_lower),
        float(bunching_upper),
        polynomial_order=polynomial_order,
    )

    counterfactual, coeffs, r_squared = result

    return {
        "counterfactual": np.array([float(x) for x in counterfactual]),
        "coefficients": np.array([float(x) for x in coeffs]),
        "r_squared": float(r_squared),
    }


def julia_compute_elasticity(
    excess_mass: float,
    t1_rate: float,
    t2_rate: float,
) -> float:
    """
    Call Julia compute_elasticity.

    Parameters
    ----------
    excess_mass : float
        Normalized excess mass (b = B/h0)
    t1_rate : float
        Marginal rate below kink
    t2_rate : float
        Marginal rate above kink

    Returns
    -------
    float
        Behavioral elasticity
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    return float(jl.compute_elasticity(
        float(excess_mass),
        float(t1_rate),
        float(t2_rate),
    ))


# =============================================================================
# Selection Models (Session 85)
# =============================================================================


def julia_heckman_two_step(
    outcomes: np.ndarray,
    selected: np.ndarray,
    selection_covariates: np.ndarray,
    outcome_covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    add_intercept: bool = True,
) -> Dict[str, Union[float, int, np.ndarray]]:
    """
    Call Julia HeckmanTwoStep via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable (n,). Use NaN for unselected.
    selected : np.ndarray
        Selection indicator (n,), boolean or 0/1.
    selection_covariates : np.ndarray
        Covariates for selection equation (n, k_z).
    outcome_covariates : np.ndarray, optional
        Covariates for outcome equation (n, k_x). If None, uses selection_covariates.
    alpha : float, default=0.05
        Significance level.
    add_intercept : bool, default=True
        Whether to add intercept to equations.

    Returns
    -------
    dict
        Julia result with estimate, se, rho, lambda_coef, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert to Julia-compatible types
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_selected = jl.collect(selected.astype(bool))
    jl_sel_cov = jl.collect(selection_covariates.astype(np.float64))

    if outcome_covariates is not None:
        jl_out_cov = jl.collect(outcome_covariates.astype(np.float64))
    else:
        jl_out_cov = None

    # Create HeckmanProblem
    problem = jl.HeckmanProblem(
        jl_outcomes,
        jl_selected,
        jl_sel_cov,
        jl_out_cov,
        jl.seval(f"(alpha={alpha},)")
    )

    # Solve with HeckmanTwoStep
    solution = jl.solve(problem, jl.HeckmanTwoStep(add_intercept=add_intercept))

    # Extract selection probabilities and IMR as numpy arrays
    selection_probs = np.array([float(x) for x in solution.selection_probs])
    imr = np.array([float(x) for x in solution.imr])
    gamma = np.array([float(x) for x in solution.gamma])
    beta = np.array([float(x) for x in solution.beta])

    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "rho": float(solution.rho),
        "sigma": float(solution.sigma),
        "lambda_coef": float(solution.lambda_coef),
        "lambda_se": float(solution.lambda_se),
        "lambda_pvalue": float(solution.lambda_pvalue),
        "n_selected": int(solution.n_selected),
        "n_total": int(solution.n_total),
        "selection_probs": selection_probs,
        "imr": imr,
        "gamma": gamma,
        "beta": beta,
        "probit_converged": bool(solution.probit_converged),
    }


def julia_selection_bias_test(
    solution_dict: Dict[str, Union[float, int]],
    alpha: float = 0.05,
) -> Dict[str, Union[float, bool, str]]:
    """
    Call Julia selection_bias_test using solution data.

    Note: This reconstructs the test from Python solution dict since
    we cannot pass HeckmanSolution directly.

    Parameters
    ----------
    solution_dict : dict
        Dictionary from julia_heckman_two_step or Python heckman_two_step
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Test result with statistic, pvalue, reject_null
    """
    lambda_coef = solution_dict["lambda_coef"]
    lambda_se = solution_dict["lambda_se"]

    if lambda_se <= 0 or np.isnan(lambda_se):
        return {
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "reject_null": False,
            "interpretation": "Cannot compute test: invalid standard error",
        }

    from scipy.stats import norm as scipy_norm

    t_stat = lambda_coef / lambda_se
    pvalue = 2 * (1 - scipy_norm.cdf(abs(t_stat)))
    reject = pvalue < alpha

    return {
        "statistic": t_stat,
        "pvalue": pvalue,
        "reject_null": reject,
        "interpretation": "Selection bias test",
    }


def julia_compute_imr(selection_probs: np.ndarray) -> np.ndarray:
    """
    Call Julia compute_imr.

    Parameters
    ----------
    selection_probs : np.ndarray
        Selection probabilities (n,)

    Returns
    -------
    np.ndarray
        Inverse Mills Ratio values
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl_probs = jl.collect(selection_probs.astype(np.float64))
    jl_imr = jl.compute_imr(jl_probs)

    return np.array([float(x) for x in jl_imr])


# =============================================================================
# QUANTILE TREATMENT EFFECTS (Session 89)
# =============================================================================


def julia_unconditional_qte(
    outcome: np.ndarray,
    treatment: np.ndarray,
    quantile: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Union[float, int]]:
    """
    Call Julia unconditional_qte via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable
    treatment : np.ndarray
        Binary treatment indicator (0/1)
    quantile : float, default=0.5
        Target quantile in (0, 1)
    n_bootstrap : int, default=1000
        Bootstrap replications
    seed : int, default=42
        Random seed

    Returns
    -------
    dict
        Julia result with tau_q, se, ci_lower, ci_upper, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert to Julia types
    jl_outcome = jl.collect(outcome.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(np.float64))

    # Include QTE module
    jl.seval('include(joinpath(@__DIR__, "src/qte/types.jl"))')
    jl.seval('include(joinpath(@__DIR__, "src/qte/unconditional.jl"))')

    # Call unconditional_qte
    jl_code = f"""
    using Random
    rng = MersenneTwister({seed})
    outcome_jl = {list(outcome.astype(float))}
    treatment_jl = {list(treatment.astype(float))}
    result = unconditional_qte(
        Float64.(outcome_jl),
        Float64.(treatment_jl);
        quantile={quantile},
        n_bootstrap={n_bootstrap},
        rng=rng
    )
    result
    """
    solution = jl.seval(jl_code)

    return {
        "tau_q": float(solution.tau_q),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "quantile": float(solution.quantile),
        "method": str(solution.method),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
        "n_total": int(solution.n_total),
    }


def julia_conditional_qte(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    quantile: float = 0.5,
) -> Dict[str, Union[float, int]]:
    """
    Call Julia conditional_qte via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable
    treatment : np.ndarray
        Binary treatment indicator (0/1)
    covariates : np.ndarray
        Covariate matrix (n x p)
    quantile : float, default=0.5
        Target quantile in (0, 1)

    Returns
    -------
    dict
        Julia result with tau_q, se, ci_lower, ci_upper, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Include QTE module
    jl.seval('include(joinpath(@__DIR__, "src/qte/types.jl"))')
    jl.seval('include(joinpath(@__DIR__, "src/qte/conditional.jl"))')

    n, p = covariates.shape
    cov_list = [list(row) for row in covariates]

    # Call conditional_qte
    jl_code = f"""
    using LinearAlgebra
    using Distributions
    outcome_jl = Float64.({list(outcome.astype(float))})
    treatment_jl = Float64.({list(treatment.astype(float))})
    covariates_jl = Float64.(hcat({cov_list}...)')
    result = conditional_qte(outcome_jl, treatment_jl, covariates_jl; quantile={quantile})
    result
    """
    solution = jl.seval(jl_code)

    return {
        "tau_q": float(solution.tau_q),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "quantile": float(solution.quantile),
        "method": str(solution.method),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
        "n_total": int(solution.n_total),
    }


def julia_rif_qte(
    outcome: np.ndarray,
    treatment: np.ndarray,
    quantile: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 42,
    covariates: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, int]]:
    """
    Call Julia rif_qte via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable
    treatment : np.ndarray
        Binary treatment indicator (0/1)
    quantile : float, default=0.5
        Target quantile in (0, 1)
    n_bootstrap : int, default=1000
        Bootstrap replications
    seed : int, default=42
        Random seed
    covariates : np.ndarray, optional
        Covariate matrix (n x p)

    Returns
    -------
    dict
        Julia result with tau_q, se, ci_lower, ci_upper, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Include QTE module
    jl.seval('include(joinpath(@__DIR__, "src/qte/types.jl"))')
    jl.seval('include(joinpath(@__DIR__, "src/qte/rif.jl"))')

    cov_code = "nothing"
    if covariates is not None:
        cov_list = [list(row) for row in covariates]
        cov_code = f"Float64.(hcat({cov_list}...)')"

    jl_code = f"""
    using Random
    using Statistics
    using LinearAlgebra
    rng = MersenneTwister({seed})
    outcome_jl = Float64.({list(outcome.astype(float))})
    treatment_jl = Float64.({list(treatment.astype(float))})
    covariates_jl = {cov_code}
    result = rif_qte(outcome_jl, treatment_jl; quantile={quantile}, covariates=covariates_jl,
                     n_bootstrap={n_bootstrap}, rng=rng)
    result
    """
    solution = jl.seval(jl_code)

    return {
        "tau_q": float(solution.tau_q),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "quantile": float(solution.quantile),
        "method": str(solution.method),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
        "n_total": int(solution.n_total),
    }


# =============================================================================
# MTE Functions (Session 91)
# =============================================================================


def julia_late_estimator(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Union[float, int, str]]:
    """
    Call Julia late_estimator via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y
    treatment : np.ndarray
        Binary treatment D (0/1)
    instrument : np.ndarray
        Binary instrument Z (0/1)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with late, se, ci_lower, ci_upper, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Include MTE module
    jl.seval('include(joinpath(@__DIR__, "src/mte/types.jl"))')
    jl.seval('include(joinpath(@__DIR__, "src/mte/late.jl"))')

    jl_code = f"""
    using Statistics
    using Distributions
    using LinearAlgebra
    outcome_jl = Float64.({list(outcome.astype(float))})
    treatment_jl = Float64.({list(treatment.astype(float))})
    instrument_jl = Float64.({list(instrument.astype(float))})
    result = late_estimator(outcome_jl, treatment_jl, instrument_jl; alpha={alpha})
    result
    """
    solution = jl.seval(jl_code)

    return {
        "late": float(solution.late),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "pvalue": float(solution.pvalue),
        "complier_share": float(solution.complier_share),
        "always_taker_share": float(solution.always_taker_share),
        "never_taker_share": float(solution.never_taker_share),
        "first_stage_coef": float(solution.first_stage_coef),
        "first_stage_f": float(solution.first_stage_f),
        "n_obs": int(solution.n_obs),
        "method": str(solution.method),
    }


def julia_late_bounds(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
) -> Dict[str, float]:
    """
    Call Julia late_bounds via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y
    treatment : np.ndarray
        Binary treatment D (0/1)
    instrument : np.ndarray
        Binary instrument Z (0/1)

    Returns
    -------
    dict
        Julia result with bounds_lower, bounds_upper, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl.seval('include(joinpath(@__DIR__, "src/mte/types.jl"))')
    jl.seval('include(joinpath(@__DIR__, "src/mte/late.jl"))')

    jl_code = f"""
    using Statistics
    using Distributions
    using LinearAlgebra
    outcome_jl = Float64.({list(outcome.astype(float))})
    treatment_jl = Float64.({list(treatment.astype(float))})
    instrument_jl = Float64.({list(instrument.astype(float))})
    result = late_bounds(outcome_jl, treatment_jl, instrument_jl)
    result
    """
    solution = jl.seval(jl_code)

    return {
        "bounds_lower": float(solution.bounds_lower),
        "bounds_upper": float(solution.bounds_upper),
        "late_under_monotonicity": float(solution.late_under_monotonicity),
        "first_stage": float(solution.first_stage),
        "reduced_form": float(solution.reduced_form),
        "bounds_width": float(solution.bounds_width),
    }


def julia_local_iv(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    n_grid: int = 20,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> Dict[str, Union[np.ndarray, float, int, str]]:
    """
    Call Julia local_iv via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y
    treatment : np.ndarray
        Binary treatment D (0/1)
    instrument : np.ndarray
        Continuous or discrete instrument Z
    n_grid : int, default=20
        Number of grid points
    n_bootstrap : int, default=100
        Bootstrap replications
    seed : int, default=42
        Random seed

    Returns
    -------
    dict
        Julia result with mte_grid, u_grid, se_grid, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl.seval('include(joinpath(@__DIR__, "src/mte/types.jl"))')
    jl.seval('include(joinpath(@__DIR__, "src/mte/local_iv.jl"))')

    jl_code = f"""
    using Statistics
    using Distributions
    using LinearAlgebra
    using Random
    outcome_jl = Float64.({list(outcome.astype(float))})
    treatment_jl = Float64.({list(treatment.astype(float))})
    instrument_jl = Float64.({list(instrument.astype(float))})
    result = local_iv(
        outcome_jl, treatment_jl, instrument_jl;
        n_grid={n_grid}, n_bootstrap={n_bootstrap}, random_state={seed}
    )
    result
    """
    solution = jl.seval(jl_code)

    return {
        "mte_grid": np.array([float(x) for x in solution.mte_grid]),
        "u_grid": np.array([float(x) for x in solution.u_grid]),
        "se_grid": np.array([float(x) for x in solution.se_grid]),
        "ci_lower": np.array([float(x) for x in solution.ci_lower]),
        "ci_upper": np.array([float(x) for x in solution.ci_upper]),
        "propensity_support": (
            float(solution.propensity_support[1]),
            float(solution.propensity_support[2]),
        ),
        "n_obs": int(solution.n_obs),
        "n_trimmed": int(solution.n_trimmed),
        "bandwidth": float(solution.bandwidth),
        "method": str(solution.method),
    }


def julia_ate_from_mte(
    mte_grid: np.ndarray,
    u_grid: np.ndarray,
    se_grid: np.ndarray,
    propensity_support: tuple,
    n_obs: int,
) -> Dict[str, Union[float, str, int]]:
    """
    Call Julia ate_from_mte via juliacall.

    Parameters
    ----------
    mte_grid : np.ndarray
        MTE estimates at grid points
    u_grid : np.ndarray
        Grid points
    se_grid : np.ndarray
        Standard errors at grid points
    propensity_support : tuple
        (p_min, p_max)
    n_obs : int
        Sample size

    Returns
    -------
    dict
        ATE result with estimate, se, ci_lower, ci_upper
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl.seval('include(joinpath(@__DIR__, "src/mte/types.jl"))')
    jl.seval('include(joinpath(@__DIR__, "src/mte/policy.jl"))')

    jl_code = f"""
    using Statistics
    using Distributions
    using Random
    mte_grid = Float64.({list(mte_grid.astype(float))})
    u_grid = Float64.({list(u_grid.astype(float))})
    se_grid = Float64.({list(se_grid.astype(float))})
    ci_lower = mte_grid .- 1.96 .* se_grid
    ci_upper = mte_grid .+ 1.96 .* se_grid
    mte_solution = MTESolution(
        mte_grid=mte_grid,
        u_grid=u_grid,
        se_grid=se_grid,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        propensity_support=({propensity_support[0]}, {propensity_support[1]}),
        n_obs={n_obs},
        n_trimmed=0,
        bandwidth=0.1,
        method=:local_iv
    )
    result = ate_from_mte(mte_solution)
    result
    """
    solution = jl.seval(jl_code)

    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "parameter": str(solution.parameter),
        "n_obs": int(solution.n_obs),
    }


# =============================================================================
# CONTROL FUNCTION (Session 95)
# =============================================================================


def julia_control_function_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, int, bool]]:
    """
    Call Julia control_function_ate via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable
    treatment : np.ndarray
        Treatment indicator (0/1)
    instrument : np.ndarray
        Instrumental variable (0/1)
    covariates : np.ndarray, optional
        Covariate matrix (n x p)

    Returns
    -------
    dict
        Julia result with ate, se, ci_lower, ci_upper, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl.seval('include(joinpath(@__DIR__, "src/control_function/types.jl"))')
    jl.seval('include(joinpath(@__DIR__, "src/control_function/linear.jl"))')

    cov_code = "nothing"
    if covariates is not None:
        cov_list = [list(row) for row in covariates]
        cov_code = f"Float64.(hcat({cov_list}...)')"

    jl_code = f"""
    using LinearAlgebra
    using Distributions
    using GLM
    using DataFrames

    outcome_jl = Float64.({list(outcome.astype(float))})
    treatment_jl = Float64.({list(treatment.astype(float))})
    instrument_jl = Float64.({list(instrument.astype(float))})
    covariates_jl = {cov_code}

    result = control_function_ate(outcome_jl, treatment_jl, instrument_jl; covariates=covariates_jl)
    result
    """
    solution = jl.seval(jl_code)

    return {
        "ate": float(solution.ate),
        "se": float(solution.se),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "pvalue": float(solution.pvalue),
        "rho": float(solution.rho),
        "rho_se": float(solution.rho_se),
        "first_stage_fstat": float(solution.first_stage.fstat),
        "first_stage_r2": float(solution.first_stage.r2),
        "n_obs": int(solution.n_obs),
        "endogeneity_detected": bool(solution.endogeneity_detected),
    }


# =============================================================================
# PARTIAL IDENTIFICATION BOUNDS (Session 95)
# =============================================================================


def julia_manski_worst_case(
    outcome: np.ndarray,
    treatment: np.ndarray,
    outcome_support: Optional[tuple] = None,
) -> Dict[str, Union[float, int, bool]]:
    """
    Call Julia manski_worst_case via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable
    treatment : np.ndarray
        Treatment indicator (0/1)
    outcome_support : tuple, optional
        (Y_min, Y_max) outcome bounds

    Returns
    -------
    dict
        Julia result with bounds_lower, bounds_upper, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl.seval('include(joinpath(@__DIR__, "src/bounds/types.jl"))')
    jl.seval('include(joinpath(@__DIR__, "src/bounds/manski.jl"))')

    support_code = "nothing"
    if outcome_support is not None:
        support_code = f"({outcome_support[0]}, {outcome_support[1]})"

    jl_code = f"""
    using Statistics

    outcome_jl = Float64.({list(outcome.astype(float))})
    treatment_jl = Float64.({list(treatment.astype(float))})

    result = manski_worst_case(outcome_jl, treatment_jl; outcome_support={support_code})
    result
    """
    solution = jl.seval(jl_code)

    return {
        "bounds_lower": float(solution.bounds_lower),
        "bounds_upper": float(solution.bounds_upper),
        "bounds_width": float(solution.bounds_width),
        "point_identified": bool(solution.point_identified),
        "assumptions": str(solution.assumptions),
        "naive_ate": float(solution.naive_ate),
        "ate_in_bounds": bool(solution.ate_in_bounds),
        "n_treated": int(solution.n_treated),
        "n_control": int(solution.n_control),
    }


def julia_lee_bounds(
    outcome: np.ndarray,
    treatment: np.ndarray,
    observed: np.ndarray,
    monotonicity: str = "positive",
    n_bootstrap: int = 100,
    seed: int = 42,
) -> Dict[str, Union[float, int, bool, str]]:
    """
    Call Julia lee_bounds via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable
    treatment : np.ndarray
        Treatment indicator (0/1)
    observed : np.ndarray
        Observation indicator (0/1)
    monotonicity : str
        "positive" or "negative"
    n_bootstrap : int
        Bootstrap replications
    seed : int
        Random seed

    Returns
    -------
    dict
        Julia result with bounds_lower, bounds_upper, ci_lower, ci_upper, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl.seval('include(joinpath(@__DIR__, "src/bounds/types.jl"))')
    jl.seval('include(joinpath(@__DIR__, "src/bounds/lee.jl"))')

    jl_code = f"""
    using Statistics
    using Random
    using Distributions

    outcome_jl = Float64.({list(outcome.astype(float))})
    treatment_jl = Float64.({list(treatment.astype(float))})
    observed_jl = Float64.({list(observed.astype(float))})

    rng = MersenneTwister({seed})
    result = lee_bounds(outcome_jl, treatment_jl, observed_jl;
                        monotonicity=:{monotonicity}, n_bootstrap={n_bootstrap}, rng=rng)
    result
    """
    solution = jl.seval(jl_code)

    return {
        "bounds_lower": float(solution.bounds_lower),
        "bounds_upper": float(solution.bounds_upper),
        "bounds_width": float(solution.bounds_width),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "point_identified": bool(solution.point_identified),
        "trimming_proportion": float(solution.trimming_proportion),
        "trimmed_group": str(solution.trimmed_group),
        "attrition_treated": float(solution.attrition_treated),
        "attrition_control": float(solution.attrition_control),
        "n_treated_observed": int(solution.n_treated_observed),
        "n_control_observed": int(solution.n_control_observed),
        "n_trimmed": int(solution.n_trimmed),
        "monotonicity": str(solution.monotonicity),
    }


# =============================================================================
# MEDIATION ANALYSIS (Session 95)
# =============================================================================


def julia_baron_kenny(
    outcome: np.ndarray,
    treatment: np.ndarray,
    mediator: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, int]]:
    """
    Call Julia baron_kenny via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable
    treatment : np.ndarray
        Treatment indicator (0/1)
    mediator : np.ndarray
        Mediator variable
    covariates : np.ndarray, optional
        Covariate matrix (n x p)

    Returns
    -------
    dict
        Julia result with path coefficients and Sobel test
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl.seval('include(joinpath(@__DIR__, "src/mediation/types.jl"))')
    jl.seval('include(joinpath(@__DIR__, "src/mediation/estimators.jl"))')

    cov_code = "nothing"
    if covariates is not None:
        cov_list = [list(row) for row in covariates]
        cov_code = f"Float64.(hcat({cov_list}...)')"

    jl_code = f"""
    using LinearAlgebra
    using Distributions

    outcome_jl = Float64.({list(outcome.astype(float))})
    treatment_jl = Float64.({list(treatment.astype(float))})
    mediator_jl = Float64.({list(mediator.astype(float))})
    covariates_jl = {cov_code}

    result = baron_kenny(outcome_jl, treatment_jl, mediator_jl; covariates=covariates_jl)
    result
    """
    solution = jl.seval(jl_code)

    return {
        "alpha_1": float(solution.alpha_1),
        "alpha_1_se": float(solution.alpha_1_se),
        "alpha_1_pvalue": float(solution.alpha_1_pvalue),
        "beta_1": float(solution.beta_1),
        "beta_1_se": float(solution.beta_1_se),
        "beta_1_pvalue": float(solution.beta_1_pvalue),
        "beta_2": float(solution.beta_2),
        "beta_2_se": float(solution.beta_2_se),
        "beta_2_pvalue": float(solution.beta_2_pvalue),
        "indirect_effect": float(solution.indirect_effect),
        "indirect_se": float(solution.indirect_se),
        "direct_effect": float(solution.direct_effect),
        "total_effect": float(solution.total_effect),
        "sobel_z": float(solution.sobel_z),
        "sobel_pvalue": float(solution.sobel_pvalue),
        "r2_mediator_model": float(solution.r2_mediator_model),
        "r2_outcome_model": float(solution.r2_outcome_model),
        "n_obs": int(solution.n_obs),
    }


def julia_mediation_analysis(
    outcome: np.ndarray,
    treatment: np.ndarray,
    mediator: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
    covariates: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, int, str, tuple]]:
    """
    Call Julia mediation_analysis via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable
    treatment : np.ndarray
        Treatment indicator (0/1)
    mediator : np.ndarray
        Mediator variable
    n_bootstrap : int
        Bootstrap replications
    seed : int
        Random seed
    covariates : np.ndarray, optional
        Covariate matrix (n x p)

    Returns
    -------
    dict
        Julia result with effects, SEs, CIs, and p-values
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    jl.seval('include(joinpath(@__DIR__, "src/mediation/types.jl"))')
    jl.seval('include(joinpath(@__DIR__, "src/mediation/estimators.jl"))')

    cov_code = "nothing"
    if covariates is not None:
        cov_list = [list(row) for row in covariates]
        cov_code = f"Float64.(hcat({cov_list}...)')"

    jl_code = f"""
    using LinearAlgebra
    using Distributions
    using Random

    outcome_jl = Float64.({list(outcome.astype(float))})
    treatment_jl = Float64.({list(treatment.astype(float))})
    mediator_jl = Float64.({list(mediator.astype(float))})
    covariates_jl = {cov_code}

    rng = MersenneTwister({seed})
    result = mediation_analysis(outcome_jl, treatment_jl, mediator_jl;
                                 covariates=covariates_jl, n_bootstrap={n_bootstrap}, rng=rng)
    result
    """
    solution = jl.seval(jl_code)

    return {
        "total_effect": float(solution.total_effect),
        "direct_effect": float(solution.direct_effect),
        "indirect_effect": float(solution.indirect_effect),
        "proportion_mediated": float(solution.proportion_mediated),
        "te_se": float(solution.te_se),
        "de_se": float(solution.de_se),
        "ie_se": float(solution.ie_se),
        "pm_se": float(solution.pm_se),
        "te_ci": (float(solution.te_ci[0]), float(solution.te_ci[1])),
        "de_ci": (float(solution.de_ci[0]), float(solution.de_ci[1])),
        "ie_ci": (float(solution.ie_ci[0]), float(solution.ie_ci[1])),
        "pm_ci": (float(solution.pm_ci[0]), float(solution.pm_ci[1])),
        "te_pvalue": float(solution.te_pvalue),
        "de_pvalue": float(solution.de_pvalue),
        "ie_pvalue": float(solution.ie_pvalue),
        "method": str(solution.method),
        "n_obs": int(solution.n_obs),
        "n_bootstrap": int(solution.n_bootstrap),
    }


def julia_shift_share_iv(
    Y: np.ndarray,
    D: np.ndarray,
    shares: np.ndarray,
    shocks: np.ndarray,
    X: Optional[np.ndarray] = None,
    inference: str = "robust",
    alpha: float = 0.05,
) -> Dict[str, Union[float, int, np.ndarray, dict]]:
    """
    Call Julia shift_share_iv via juliacall.

    Parameters
    ----------
    Y : np.ndarray
        Outcome variable
    D : np.ndarray
        Endogenous treatment
    shares : np.ndarray
        Sector shares (n x S)
    shocks : np.ndarray
        Sector shocks (S,)
    X : np.ndarray, optional
        Control variables
    inference : str
        'robust' or 'clustered'
    alpha : float
        Significance level

    Returns
    -------
    dict
        Julia result with estimate, SE, diagnostics
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert to Julia arrays
    n = len(Y)
    n_sectors = len(shocks)

    Y_list = list(Y.astype(float))
    D_list = list(D.astype(float))
    shocks_list = list(shocks.astype(float))

    # Shares is 2D - convert row by row
    shares_rows = [list(row) for row in shares.astype(float)]

    # Build controls code
    if X is not None:
        X_rows = [list(row) for row in X.astype(float)]
        x_code = f"hcat({X_rows}...)''"
    else:
        x_code = "nothing"

    jl_code = f"""
    using Random
    using Statistics

    Y_jl = Float64.({Y_list})
    D_jl = Float64.({D_list})
    shocks_jl = Float64.({shocks_list})
    shares_jl = Float64.(vcat([reshape({row}, 1, :) for row in {shares_rows}]...))
    X_jl = {x_code}

    result = shift_share_iv(Y_jl, D_jl, shares_jl, shocks_jl;
                            X=X_jl, inference=:{inference}, alpha=Float64({alpha}))
    result
    """
    solution = jl.seval(jl_code)

    # Extract Rotemberg diagnostics
    rotemberg = {
        "weights": np.array([float(w) for w in solution.rotemberg.weights]),
        "negative_weight_share": float(solution.rotemberg.negative_weight_share),
        "top_5_sectors": [int(s) for s in solution.rotemberg.top_5_sectors],
        "top_5_weights": np.array([float(w) for w in solution.rotemberg.top_5_weights]),
        "herfindahl": float(solution.rotemberg.herfindahl),
    }

    # Extract first stage
    first_stage = {
        "f_statistic": float(solution.first_stage.f_statistic),
        "f_pvalue": float(solution.first_stage.f_pvalue),
        "partial_r2": float(solution.first_stage.partial_r2),
        "coefficient": float(solution.first_stage.coefficient),
        "se": float(solution.first_stage.se),
        "t_stat": float(solution.first_stage.t_stat),
        "weak_iv_warning": bool(solution.first_stage.weak_iv_warning),
    }

    return {
        "estimate": float(solution.estimate),
        "se": float(solution.se),
        "t_stat": float(solution.t_stat),
        "p_value": float(solution.p_value),
        "ci_lower": float(solution.ci_lower),
        "ci_upper": float(solution.ci_upper),
        "first_stage": first_stage,
        "rotemberg": rotemberg,
        "n_obs": int(solution.n_obs),
        "n_sectors": int(solution.n_sectors),
        "share_sum_mean": float(solution.share_sum_mean),
        "inference": str(solution.inference),
        "alpha": float(solution.alpha),
    }


# =============================================================================
# PRINCIPAL STRATIFICATION (Session 111)
# =============================================================================


def julia_cace_2sls(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    inference: str = "robust",
) -> Dict[str, Union[float, int, Dict[str, float]]]:
    """
    Call Julia cace_2sls via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable (n,).
    treatment : np.ndarray
        Treatment received (n,), binary.
    instrument : np.ndarray
        Instrument/assignment (n,), binary.
    covariates : np.ndarray, optional
        Covariates (n, k). If None, no covariates.
    alpha : float, default=0.05
        Significance level.
    inference : str, default="robust"
        Inference type: "robust" or "standard".

    Returns
    -------
    dict
        Julia result with cace, se, ci_lower, ci_upper, strata_proportions, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert to Julia-compatible types
    jl_outcome = jl.collect(outcome.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_instrument = jl.collect(instrument.astype(bool))

    if covariates is not None:
        jl_covariates = jl.collect(covariates.astype(np.float64))
    else:
        jl_covariates = None

    # Call convenience function
    result = jl.cace_2sls(
        jl_outcome,
        jl_treatment,
        jl_instrument,
        covariates=jl_covariates,
        alpha=alpha,
        inference=jl.seval(f":{inference}"),
    )

    # Extract strata proportions
    strata_props = {
        "compliers": float(result.strata_proportions.compliers),
        "always_takers": float(result.strata_proportions.always_takers),
        "never_takers": float(result.strata_proportions.never_takers),
        "compliers_se": float(result.strata_proportions.compliers_se),
    }

    return _format_cace_result(result, strata_props)


def _format_cace_result(result, strata_props: Dict[str, float]) -> Dict[str, Union[float, int, Dict]]:
    """Format Julia CACE result as Python dict."""
    return {
        "cace": float(result.cace),
        "se": float(result.se),
        "ci_lower": float(result.ci_lower),
        "ci_upper": float(result.ci_upper),
        "z_stat": float(result.z_stat),
        "pvalue": float(result.pvalue),
        "strata_proportions": strata_props,
        "first_stage_coef": float(result.first_stage_coef),
        "first_stage_se": float(result.first_stage_se),
        "first_stage_f": float(result.first_stage_f),
        "reduced_form": float(result.reduced_form),
        "reduced_form_se": float(result.reduced_form_se),
        "n": int(result.n),
        "method": str(result.method),
    }


def julia_cace_em(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    alpha: float = 0.05,
) -> Dict[str, Union[float, int, Dict[str, float]]]:
    """
    Call Julia cace_em via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable (n,).
    treatment : np.ndarray
        Treatment received (n,), binary.
    instrument : np.ndarray
        Instrument/assignment (n,), binary.
    max_iter : int, default=100
        Maximum EM iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    dict
        Julia result with cace, se, ci_lower, ci_upper, strata_proportions, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert to Julia-compatible types
    jl_outcome = jl.collect(outcome.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_instrument = jl.collect(instrument.astype(bool))

    # Call convenience function
    result = jl.cace_em(
        jl_outcome, jl_treatment, jl_instrument,
        max_iter=max_iter, tol=tol, alpha=alpha
    )

    # Extract strata proportions
    strata_props = {
        "compliers": float(result.strata_proportions.compliers),
        "always_takers": float(result.strata_proportions.always_takers),
        "never_takers": float(result.strata_proportions.never_takers),
        "compliers_se": float(result.strata_proportions.compliers_se),
    }

    return _format_cace_result(result, strata_props)


def julia_wald_estimator(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Union[float, int, Dict[str, float]]]:
    """
    Call Julia wald_estimator via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable (n,).
    treatment : np.ndarray
        Treatment received (n,), binary.
    instrument : np.ndarray
        Instrument/assignment (n,), binary.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    dict
        Julia result with cace, se, ci_lower, ci_upper, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert to Julia-compatible types
    jl_outcome = jl.collect(outcome.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_instrument = jl.collect(instrument.astype(bool))

    # Call convenience function
    result = jl.wald_estimator(jl_outcome, jl_treatment, jl_instrument, alpha=alpha)

    # Extract strata proportions
    strata_props = {
        "compliers": float(result.strata_proportions.compliers),
        "always_takers": float(result.strata_proportions.always_takers),
        "never_takers": float(result.strata_proportions.never_takers),
        "compliers_se": float(result.strata_proportions.compliers_se),
    }

    return _format_cace_result(result, strata_props)


def _format_cace_result(result, strata_props: Dict[str, float]) -> Dict[str, Union[float, int, Dict]]:
    """Format Julia CACE result as Python dict."""
    return {
        "cace": float(result.cace),
        "se": float(result.se),
        "ci_lower": float(result.ci_lower),
        "ci_upper": float(result.ci_upper),
        "z_stat": float(result.z_stat),
        "pvalue": float(result.pvalue),
        "strata_proportions": strata_props,
        "first_stage_coef": float(result.first_stage_coef),
        "first_stage_se": float(result.first_stage_se),
        "first_stage_f": float(result.first_stage_f),
        "reduced_form": float(result.reduced_form),
        "reduced_form_se": float(result.reduced_form_se),
        "n": int(result.n),
        "method": str(result.method),
    }


def julia_cace_em(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    alpha: float = 0.05,
) -> Dict[str, Union[float, int, Dict[str, float]]]:
    """
    Call Julia cace_em via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable (n,).
    treatment : np.ndarray
        Treatment received (n,), binary.
    instrument : np.ndarray
        Instrument/assignment (n,), binary.
    max_iter : int, default=100
        Maximum EM iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    dict
        Julia result with cace, se, ci_lower, ci_upper, strata_proportions, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert to Julia-compatible types
    jl_outcome = jl.collect(outcome.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_instrument = jl.collect(instrument.astype(bool))

    # Call convenience function
    result = jl.cace_em(
        jl_outcome, jl_treatment, jl_instrument,
        max_iter=max_iter, tol=tol, alpha=alpha
    )

    # Extract strata proportions
    strata_props = {
        "compliers": float(result.strata_proportions.compliers),
        "always_takers": float(result.strata_proportions.always_takers),
        "never_takers": float(result.strata_proportions.never_takers),
        "compliers_se": float(result.strata_proportions.compliers_se),
    }

    return _format_cace_result(result, strata_props)


def julia_cace_bayesian(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    prior_alpha: tuple = (1.0, 1.0, 1.0),
    prior_mu_sd: float = 10.0,
    quick: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Union[float, int, Dict[str, float]]]:
    """
    Call Julia cace_bayesian via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable (n,).
    treatment : np.ndarray
        Treatment received (n,), binary.
    instrument : np.ndarray
        Instrument/assignment (n,), binary.
    prior_alpha : tuple
        Dirichlet prior for strata proportions.
    prior_mu_sd : float
        Prior SD for outcome means.
    quick : bool
        If True, use fast settings.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    dict
        Julia result with cace_mean, cace_sd, cace_hdi_lower, cace_hdi_upper, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Include bayesian module
    jl.seval('include("src/principal_stratification/bayesian.jl")')

    # Convert to Julia-compatible types
    jl_outcome = jl.collect(outcome.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(bool))
    jl_instrument = jl.collect(instrument.astype(bool))

    # Call convenience function
    kwargs = {
        "prior_alpha": prior_alpha,
        "prior_mu_sd": prior_mu_sd,
        "quick": quick,
    }
    if seed is not None:
        kwargs["seed"] = seed

    result = jl.cace_bayesian(jl_outcome, jl_treatment, jl_instrument, **kwargs)

    return {
        "cace_mean": float(result.cace_mean),
        "cace_sd": float(result.cace_sd),
        "cace_hdi_lower": float(result.cace_hdi_lower),
        "cace_hdi_upper": float(result.cace_hdi_upper),
        "pi_c_mean": float(result.pi_c_mean),
        "pi_a_mean": float(result.pi_a_mean),
        "pi_n_mean": float(result.pi_n_mean),
        "n_samples": int(result.n_samples),
        "n_chains": int(result.n_chains),
        "model": str(result.model),
    }


def julia_ps_bounds_monotonicity(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    direct_effect_bound: float = 0.0,
) -> Dict[str, Union[float, bool, list, str]]:
    """
    Call Julia ps_bounds_monotonicity via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable (n,).
    treatment : np.ndarray
        Treatment received (n,), binary.
    instrument : np.ndarray
        Instrument/assignment (n,), binary.
    direct_effect_bound : float
        Maximum direct effect of Z on Y.

    Returns
    -------
    dict
        Julia result with lower_bound, upper_bound, bound_width, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Include bounds module
    jl.seval('include("src/principal_stratification/bounds.jl")')

    # Convert to Julia-compatible types
    jl_outcome = jl.collect(outcome.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(np.float64))
    jl_instrument = jl.collect(instrument.astype(np.float64))

    result = jl.ps_bounds_monotonicity(
        jl_outcome, jl_treatment, jl_instrument,
        direct_effect_bound=direct_effect_bound
    )

    return {
        "lower_bound": float(result.lower_bound),
        "upper_bound": float(result.upper_bound),
        "bound_width": float(result.bound_width),
        "identified": bool(result.identified),
        "assumptions": list(result.assumptions),
        "method": str(result.method),
    }


def julia_ps_bounds_no_assumption(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    outcome_support: Optional[tuple] = None,
) -> Dict[str, Union[float, bool, list, str]]:
    """
    Call Julia ps_bounds_no_assumption via juliacall.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Include bounds module
    jl.seval('include("src/principal_stratification/bounds.jl")')

    # Convert to Julia-compatible types
    jl_outcome = jl.collect(outcome.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(np.float64))
    jl_instrument = jl.collect(instrument.astype(np.float64))

    if outcome_support is not None:
        result = jl.ps_bounds_no_assumption(
            jl_outcome, jl_treatment, jl_instrument,
            outcome_support=outcome_support
        )
    else:
        result = jl.ps_bounds_no_assumption(jl_outcome, jl_treatment, jl_instrument)

    return {
        "lower_bound": float(result.lower_bound),
        "upper_bound": float(result.upper_bound),
        "bound_width": float(result.bound_width),
        "identified": bool(result.identified),
        "assumptions": list(result.assumptions),
        "method": str(result.method),
    }


def julia_sace_bounds(
    outcome: np.ndarray,
    treatment: np.ndarray,
    survival: np.ndarray,
    monotonicity: str = "none",
) -> Dict[str, Union[float, int, str]]:
    """
    Call Julia sace_bounds via juliacall.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable (n,), NaN for non-survivors.
    treatment : np.ndarray
        Treatment received (n,), binary.
    survival : np.ndarray
        Survival indicator (n,), binary.
    monotonicity : str
        Monotonicity assumption.

    Returns
    -------
    dict
        Julia result with sace, se, lower_bound, upper_bound, etc.
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Include sace module
    jl.seval('include("src/principal_stratification/sace.jl")')

    # Convert to Julia-compatible types
    jl_outcome = jl.collect(outcome.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(np.float64))
    jl_survival = jl.collect(survival.astype(np.float64))

    result = jl.sace_bounds(
        jl_outcome, jl_treatment, jl_survival,
        monotonicity=monotonicity
    )

    return {
        "sace": float(result.sace),
        "se": float(result.se),
        "lower_bound": float(result.lower_bound),
        "upper_bound": float(result.upper_bound),
        "proportion_survivors_treat": float(result.proportion_survivors_treat),
        "proportion_survivors_control": float(result.proportion_survivors_control),
        "n": int(result.n),
        "method": str(result.method),
    }


# =============================================================================
# DML Continuous Treatment Functions (Session 116)
# =============================================================================


def julia_dml_continuous(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    n_folds: int = 5,
    model: str = "ridge",
    alpha: float = 0.05,
) -> Dict[str, Union[float, np.ndarray, int, str]]:
    """
    Call Julia dml_continuous via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,)
    treatment : np.ndarray
        Continuous treatment D (n,)
    covariates : np.ndarray
        Covariate matrix X (n, p)
    n_folds : int, default=5
        Number of cross-fitting folds
    model : str, default="ridge"
        Model type for nuisance estimation ("ols" or "ridge")
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with ate, ate_se, cate, ci_lower, ci_upper,
        fold_estimates, fold_ses, outcome_r2, treatment_r2, n, n_folds
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Ensure covariates is 2D
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    # Convert to Julia-compatible types
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(np.float64))
    jl_covariates = jl.collect(covariates.astype(np.float64))

    # Convert model string to Julia Symbol
    jl_model = jl.seval(f":{model}")

    # Call Julia dml_continuous
    result = jl.dml_continuous(
        jl_outcomes, jl_treatment, jl_covariates,
        n_folds=n_folds, model=jl_model, alpha=alpha
    )

    return {
        "ate": float(result.ate),
        "ate_se": float(result.ate_se),
        "ci_lower": float(result.ci_lower),
        "ci_upper": float(result.ci_upper),
        "cate": np.array([float(c) for c in result.cate]),
        "method": str(result.method),
        "fold_estimates": np.array([float(f) for f in result.fold_estimates]),
        "fold_ses": np.array([float(s) for s in result.fold_ses]),
        "outcome_r2": float(result.outcome_r2),
        "treatment_r2": float(result.treatment_r2),
        "n": int(result.n),
        "n_folds": int(result.n_folds),
    }


# =============================================================================
# Panel DML-CRE Functions (Session 117)
# =============================================================================


def julia_dml_cre(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    unit_id: np.ndarray,
    time: np.ndarray,
    n_folds: int = 5,
    alpha: float = 0.05,
) -> Dict[str, Union[float, np.ndarray, int, str]]:
    """
    Call Julia dml_cre (binary treatment) via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n_obs,)
    treatment : np.ndarray
        Binary treatment D (n_obs,)
    covariates : np.ndarray
        Covariate matrix X (n_obs, p)
    unit_id : np.ndarray
        Unit identifiers (n_obs,)
    time : np.ndarray
        Time period identifiers (n_obs,)
    n_folds : int, default=5
        Number of cross-fitting folds
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with ate, ate_se, cate, ci_lower, ci_upper,
        fold_estimates, fold_ses, outcome_r2, treatment_r2, n_obs, n_units, n_folds, unit_effects
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Ensure covariates is 2D
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    # Convert to Julia-compatible types
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(np.float64))
    jl_covariates = jl.collect(covariates.astype(np.float64))
    jl_unit_id = jl.collect(unit_id.astype(np.int64))
    jl_time = jl.collect(time.astype(np.int64))

    # Create PanelData in Julia
    panel = jl.PanelData(jl_outcomes, jl_treatment, jl_covariates, jl_unit_id, jl_time)

    # Call Julia dml_cre
    result = jl.dml_cre(panel, n_folds=n_folds, alpha=alpha)

    return {
        "ate": float(result.ate),
        "ate_se": float(result.ate_se),
        "ci_lower": float(result.ci_lower),
        "ci_upper": float(result.ci_upper),
        "cate": np.array([float(c) for c in result.cate]),
        "method": str(result.method),
        "fold_estimates": np.array([float(f) for f in result.fold_estimates]),
        "fold_ses": np.array([float(s) for s in result.fold_ses]),
        "outcome_r2": float(result.outcome_r2),
        "treatment_r2": float(result.treatment_r2),
        "n_obs": int(result.n_obs),
        "n_units": int(result.n_units),
        "n_folds": int(result.n_folds),
        "unit_effects": np.array([float(u) for u in result.unit_effects]),
    }


def julia_dml_cre_continuous(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    unit_id: np.ndarray,
    time: np.ndarray,
    n_folds: int = 5,
    alpha: float = 0.05,
) -> Dict[str, Union[float, np.ndarray, int, str]]:
    """
    Call Julia dml_cre_continuous via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n_obs,)
    treatment : np.ndarray
        Continuous treatment D (n_obs,)
    covariates : np.ndarray
        Covariate matrix X (n_obs, p)
    unit_id : np.ndarray
        Unit identifiers (n_obs,)
    time : np.ndarray
        Time period identifiers (n_obs,)
    n_folds : int, default=5
        Number of cross-fitting folds
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Julia result with ate, ate_se, cate, ci_lower, ci_upper,
        fold_estimates, fold_ses, outcome_r2, treatment_r2, n_obs, n_units, n_folds, unit_effects
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Ensure covariates is 2D
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    # Convert to Julia-compatible types
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(np.float64))
    jl_covariates = jl.collect(covariates.astype(np.float64))
    jl_unit_id = jl.collect(unit_id.astype(np.int64))
    jl_time = jl.collect(time.astype(np.int64))

    # Create PanelData in Julia
    panel = jl.PanelData(jl_outcomes, jl_treatment, jl_covariates, jl_unit_id, jl_time)

    # Call Julia dml_cre_continuous
    result = jl.dml_cre_continuous(panel, n_folds=n_folds, alpha=alpha)

    return {
        "ate": float(result.ate),
        "ate_se": float(result.ate_se),
        "ci_lower": float(result.ci_lower),
        "ci_upper": float(result.ci_upper),
        "cate": np.array([float(c) for c in result.cate]),
        "method": str(result.method),
        "fold_estimates": np.array([float(f) for f in result.fold_estimates]),
        "fold_ses": np.array([float(s) for s in result.fold_ses]),
        "outcome_r2": float(result.outcome_r2),
        "treatment_r2": float(result.treatment_r2),
        "n_obs": int(result.n_obs),
        "n_units": int(result.n_units),
        "n_folds": int(result.n_folds),
        "unit_effects": np.array([float(u) for u in result.unit_effects]),
    }


# =============================================================================
# Panel QTE Functions (Session 118)
# =============================================================================


def julia_panel_rif_qte(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    unit_id: np.ndarray,
    time: np.ndarray,
    tau: float = 0.5,
    alpha: float = 0.05,
    include_covariates: bool = True,
) -> Dict[str, Union[float, int, str]]:
    """
    Call Julia panel_rif_qte via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n_obs,)
    treatment : np.ndarray
        Binary treatment D (n_obs,)
    covariates : np.ndarray
        Covariate matrix X (n_obs, p)
    unit_id : np.ndarray
        Unit identifiers (n_obs,)
    time : np.ndarray
        Time period identifiers (n_obs,)
    tau : float, default=0.5
        Quantile to estimate
    alpha : float, default=0.05
        Significance level
    include_covariates : bool, default=True
        Include covariates in RIF regression

    Returns
    -------
    dict
        Julia result with qte, qte_se, ci_lower, ci_upper, quantile,
        n_obs, n_units, outcome_quantile, density_at_quantile, bandwidth, method
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Ensure covariates is 2D
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    # Convert to Julia-compatible types
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(np.float64))
    jl_covariates = jl.collect(covariates.astype(np.float64))
    jl_unit_id = jl.collect(unit_id.astype(np.int64))
    jl_time = jl.collect(time.astype(np.int64))

    # Create PanelData in Julia
    panel = jl.PanelData(jl_outcomes, jl_treatment, jl_covariates, jl_unit_id, jl_time)

    # Call Julia panel_rif_qte
    result = jl.panel_rif_qte(
        panel, tau=tau, alpha=alpha, include_covariates=include_covariates
    )

    return {
        "qte": float(result.qte),
        "qte_se": float(result.qte_se),
        "ci_lower": float(result.ci_lower),
        "ci_upper": float(result.ci_upper),
        "quantile": float(result.quantile),
        "n_obs": int(result.n_obs),
        "n_units": int(result.n_units),
        "outcome_quantile": float(result.outcome_quantile),
        "density_at_quantile": float(result.density_at_quantile),
        "bandwidth": float(result.bandwidth),
        "method": str(result.method),
    }


def julia_panel_rif_qte_band(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    unit_id: np.ndarray,
    time: np.ndarray,
    quantiles: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    include_covariates: bool = True,
) -> Dict[str, Union[np.ndarray, int, str]]:
    """
    Call Julia panel_rif_qte_band via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n_obs,)
    treatment : np.ndarray
        Binary treatment D (n_obs,)
    covariates : np.ndarray
        Covariate matrix X (n_obs, p)
    unit_id : np.ndarray
        Unit identifiers (n_obs,)
    time : np.ndarray
        Time period identifiers (n_obs,)
    quantiles : np.ndarray, optional
        Quantiles to estimate. Default: [0.1, 0.25, 0.5, 0.75, 0.9]
    alpha : float, default=0.05
        Significance level
    include_covariates : bool, default=True
        Include covariates in RIF regression

    Returns
    -------
    dict
        Julia result with quantiles, qtes, qte_ses, ci_lowers, ci_uppers,
        n_obs, n_units, method
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Ensure covariates is 2D
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    # Convert to Julia-compatible types
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(np.float64))
    jl_covariates = jl.collect(covariates.astype(np.float64))
    jl_unit_id = jl.collect(unit_id.astype(np.int64))
    jl_time = jl.collect(time.astype(np.int64))

    # Create PanelData in Julia
    panel = jl.PanelData(jl_outcomes, jl_treatment, jl_covariates, jl_unit_id, jl_time)

    # Call Julia panel_rif_qte_band
    if quantiles is not None:
        jl_quantiles = jl.collect(quantiles.astype(np.float64))
        result = jl.panel_rif_qte_band(
            panel, quantiles=jl_quantiles, alpha=alpha, include_covariates=include_covariates
        )
    else:
        result = jl.panel_rif_qte_band(
            panel, alpha=alpha, include_covariates=include_covariates
        )

    return {
        "quantiles": np.array([float(q) for q in result.quantiles]),
        "qtes": np.array([float(q) for q in result.qtes]),
        "qte_ses": np.array([float(s) for s in result.qte_ses]),
        "ci_lowers": np.array([float(c) for c in result.ci_lowers]),
        "ci_uppers": np.array([float(c) for c in result.ci_uppers]),
        "n_obs": int(result.n_obs),
        "n_units": int(result.n_units),
        "method": str(result.method),
    }


def julia_panel_unconditional_qte(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    unit_id: np.ndarray,
    time: np.ndarray,
    tau: float = 0.5,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    cluster_bootstrap: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Union[float, int, str]]:
    """
    Call Julia panel_unconditional_qte via juliacall.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n_obs,)
    treatment : np.ndarray
        Binary treatment D (n_obs,)
    covariates : np.ndarray
        Covariate matrix X (n_obs, p)
    unit_id : np.ndarray
        Unit identifiers (n_obs,)
    time : np.ndarray
        Time period identifiers (n_obs,)
    tau : float, default=0.5
        Quantile to estimate
    n_bootstrap : int, default=1000
        Number of bootstrap replications
    alpha : float, default=0.05
        Significance level
    cluster_bootstrap : bool, default=True
        Use cluster bootstrap
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Julia result with qte, qte_se, ci_lower, ci_upper, quantile,
        n_obs, n_units, outcome_quantile, density_at_quantile, bandwidth, method
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Ensure covariates is 2D
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    # Convert to Julia-compatible types
    jl_outcomes = jl.collect(outcomes.astype(np.float64))
    jl_treatment = jl.collect(treatment.astype(np.float64))
    jl_covariates = jl.collect(covariates.astype(np.float64))
    jl_unit_id = jl.collect(unit_id.astype(np.int64))
    jl_time = jl.collect(time.astype(np.int64))

    # Create PanelData in Julia
    panel = jl.PanelData(jl_outcomes, jl_treatment, jl_covariates, jl_unit_id, jl_time)

    # Call Julia panel_unconditional_qte
    if random_state is not None:
        result = jl.panel_unconditional_qte(
            panel,
            tau=tau,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            cluster_bootstrap=cluster_bootstrap,
            random_state=random_state,
        )
    else:
        result = jl.panel_unconditional_qte(
            panel,
            tau=tau,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            cluster_bootstrap=cluster_bootstrap,
        )

    return {
        "qte": float(result.qte),
        "qte_se": float(result.qte_se),
        "ci_lower": float(result.ci_lower),
        "ci_upper": float(result.ci_upper),
        "quantile": float(result.quantile),
        "n_obs": int(result.n_obs),
        "n_units": int(result.n_units),
        "outcome_quantile": float(result.outcome_quantile),
        "density_at_quantile": float(result.density_at_quantile),
        "bandwidth": float(result.bandwidth),
        "method": str(result.method),
    }


# ============================================================================
# CAUSAL DISCOVERY (Session 133)
# ============================================================================


def julia_generate_random_dag(
    n_vars: int,
    edge_prob: float = 0.3,
    seed: Optional[int] = None,
) -> Dict[str, Union[int, np.ndarray]]:
    """
    Call Julia generate_random_dag via juliacall.

    Parameters
    ----------
    n_vars : int
        Number of variables
    edge_prob : float, default=0.3
        Probability of edge between each pair
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Julia result with adjacency_matrix and n_nodes
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    if seed is not None:
        dag = jl.generate_random_dag(n_vars, edge_prob=edge_prob, seed=seed)
    else:
        dag = jl.generate_random_dag(n_vars, edge_prob=edge_prob)

    # Convert adjacency matrix to numpy
    adj = np.array([[int(dag.adjacency[i, j]) for j in range(1, n_vars + 1)]
                    for i in range(1, n_vars + 1)])

    return {
        "n_nodes": int(dag.n_nodes),
        "adjacency_matrix": adj,
    }


def julia_generate_dag_data(
    adjacency_matrix: np.ndarray,
    n_samples: int,
    noise_type: str = "gaussian",
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Call Julia generate_dag_data via juliacall.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        DAG adjacency matrix (n_vars x n_vars)
    n_samples : int
        Number of samples to generate
    noise_type : str, default="gaussian"
        Noise distribution: "gaussian", "laplace", "uniform", "exponential"
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Julia result with data matrix and coefficient matrix B
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    n_vars = adjacency_matrix.shape[0]

    # Create DAG in Julia
    jl_adj = jl.seval("Matrix{Int8}")(adjacency_matrix.astype(np.int8))
    dag = jl.DAG(n_vars, [f"X{i}" for i in range(n_vars)], jl_adj)

    # Generate data
    noise_sym = jl.seval(f":{noise_type}")
    if seed is not None:
        data, B = jl.generate_dag_data(dag, n_samples, noise_type=noise_sym, seed=seed)
    else:
        data, B = jl.generate_dag_data(dag, n_samples, noise_type=noise_sym)

    # Convert to numpy
    data_np = np.array([[float(data[i, j]) for j in range(1, n_vars + 1)]
                        for i in range(1, n_samples + 1)])
    B_np = np.array([[float(B[i, j]) for j in range(1, n_vars + 1)]
                     for i in range(1, n_vars + 1)])

    return {
        "data": data_np,
        "B": B_np,
    }


def julia_pc_algorithm(
    data: np.ndarray,
    alpha: float = 0.01,
) -> Dict[str, Union[int, float, np.ndarray]]:
    """
    Call Julia pc_algorithm via juliacall.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples x n_vars)
    alpha : float, default=0.01
        Significance level for CI tests

    Returns
    -------
    dict
        Julia result with skeleton, cpdag, n_ci_tests
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    n_samples, n_vars = data.shape

    # Convert data to Julia matrix
    jl_data = jl.seval("Matrix{Float64}")(data.astype(np.float64))

    # Run PC algorithm
    result = jl.pc_algorithm(jl_data, alpha=alpha)

    # Extract skeleton adjacency
    skeleton_adj = np.array([[int(result.skeleton.adjacency[i, j])
                              for j in range(1, n_vars + 1)]
                             for i in range(1, n_vars + 1)])

    # Extract CPDAG (directed and undirected)
    cpdag_directed = np.array([[int(result.cpdag.directed[i, j])
                                for j in range(1, n_vars + 1)]
                               for i in range(1, n_vars + 1)])
    cpdag_undirected = np.array([[int(result.cpdag.undirected[i, j])
                                  for j in range(1, n_vars + 1)]
                                 for i in range(1, n_vars + 1)])

    return {
        "skeleton": skeleton_adj,
        "cpdag_directed": cpdag_directed,
        "cpdag_undirected": cpdag_undirected,
        "n_ci_tests": int(result.n_ci_tests),
        "alpha": float(result.alpha),
    }


def julia_direct_lingam(
    data: np.ndarray,
    seed: Optional[int] = None,
) -> Dict[str, Union[np.ndarray, list]]:
    """
    Call Julia direct_lingam via juliacall.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples x n_vars)
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Julia result with dag, causal_order, adjacency_matrix
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    n_samples, n_vars = data.shape

    # Convert data to Julia matrix
    jl_data = jl.seval("Matrix{Float64}")(data.astype(np.float64))

    # Run DirectLiNGAM
    if seed is not None:
        result = jl.direct_lingam(jl_data, seed=seed)
    else:
        result = jl.direct_lingam(jl_data)

    # Extract DAG adjacency
    dag_adj = np.array([[int(result.dag.adjacency[i, j])
                         for j in range(1, n_vars + 1)]
                        for i in range(1, n_vars + 1)])

    # Extract causal order (convert to 0-indexed)
    causal_order = [int(result.causal_order[i]) - 1 for i in range(1, n_vars + 1)]

    # Extract adjacency matrix (weighted)
    adj_matrix = np.array([[float(result.adjacency_matrix[i, j])
                            for j in range(1, n_vars + 1)]
                           for i in range(1, n_vars + 1)])

    return {
        "dag_adjacency": dag_adj,
        "causal_order": causal_order,
        "adjacency_matrix": adj_matrix,
    }


def julia_skeleton_f1(
    estimated_adj: np.ndarray,
    true_dag_adj: np.ndarray,
) -> Dict[str, float]:
    """
    Call Julia skeleton_f1 via juliacall.

    Parameters
    ----------
    estimated_adj : np.ndarray
        Estimated skeleton adjacency matrix
    true_dag_adj : np.ndarray
        True DAG adjacency matrix

    Returns
    -------
    dict
        Precision, recall, F1 score
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    n_vars = estimated_adj.shape[0]

    # Create Graph from estimated adjacency
    jl_est_adj = jl.seval("Matrix{Int8}")(estimated_adj.astype(np.int8))
    skeleton = jl.Graph(n_vars, jl_est_adj)

    # Create DAG from true adjacency
    jl_true_adj = jl.seval("Matrix{Int8}")(true_dag_adj.astype(np.int8))
    true_dag = jl.DAG(n_vars, [f"X{i}" for i in range(n_vars)], jl_true_adj)

    # Compute F1
    precision, recall, f1 = jl.skeleton_f1(skeleton, true_dag)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def julia_compute_shd(
    cpdag_directed: np.ndarray,
    cpdag_undirected: np.ndarray,
    true_dag_adj: np.ndarray,
) -> Dict[str, int]:
    """
    Call Julia compute_shd via juliacall.

    Parameters
    ----------
    cpdag_directed : np.ndarray
        CPDAG directed edges matrix
    cpdag_undirected : np.ndarray
        CPDAG undirected edges matrix
    true_dag_adj : np.ndarray
        True DAG adjacency matrix

    Returns
    -------
    dict
        Structural Hamming Distance
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    n_vars = cpdag_directed.shape[0]

    # Create CPDAG
    jl_directed = jl.seval("Matrix{Int8}")(cpdag_directed.astype(np.int8))
    jl_undirected = jl.seval("Matrix{Int8}")(cpdag_undirected.astype(np.int8))
    cpdag = jl.CPDAG(n_vars, [f"X{i}" for i in range(n_vars)], jl_directed, jl_undirected)

    # Create true DAG
    jl_true_adj = jl.seval("Matrix{Int8}")(true_dag_adj.astype(np.int8))
    true_dag = jl.DAG(n_vars, [f"X{i}" for i in range(n_vars)], jl_true_adj)

    # Compute SHD
    shd = jl.compute_shd(cpdag, true_dag)

    return {
        "shd": int(shd),
    }


def julia_fisher_z_test(
    data: np.ndarray,
    x: int,
    y: int,
    conditioning_set: Optional[list] = None,
    alpha: float = 0.05,
) -> Dict[str, Union[float, bool]]:
    """
    Call Julia fisher_z_test via juliacall.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples x n_vars)
    x : int
        First variable index (0-based)
    y : int
        Second variable index (0-based)
    conditioning_set : list, optional
        List of conditioning variable indices (0-based)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Test result with pvalue, statistic, independent flag
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia not available. Install juliacall.")

    # Convert data to Julia matrix
    jl_data = jl.seval("Matrix{Float64}")(data.astype(np.float64))

    # Convert to 1-indexed
    jl_x = x + 1
    jl_y = y + 1

    if conditioning_set is not None and len(conditioning_set) > 0:
        jl_cond = jl.collect([c + 1 for c in conditioning_set])
        result = jl.fisher_z_test(jl_data, jl_x, jl_y, jl_cond, alpha=alpha)
    else:
        result = jl.fisher_z_test(jl_data, jl_x, jl_y, alpha=alpha)

    return {
        "pvalue": float(result.pvalue),
        "statistic": float(result.statistic),
        "independent": bool(result.independent),
    }
