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
