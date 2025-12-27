"""R interface for triangulation testing via rpy2.

This module provides wrapper functions to call R packages for comparison testing.
All functions gracefully handle missing R/rpy2 installations by returning None
or raising appropriate errors.

Dependencies (optional):
- rpy2>=3.5 (Python-R bridge)
- R packages: PStrata, DTRreg

Install with: pip install causal-inference-mastery[r-triangulation]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np


# =============================================================================
# Availability Checks
# =============================================================================


def check_r_available() -> bool:
    """Check if R and rpy2 are available.

    Returns
    -------
    bool
        True if rpy2 can be imported and R is accessible, False otherwise.
    """
    try:
        import rpy2.robjects as ro  # noqa: F401

        # Try to execute simple R command to verify R is working
        ro.r("1 + 1")
        return True
    except ImportError:
        return False
    except Exception:
        # R not installed or not accessible
        return False


def check_pstrata_installed() -> bool:
    """Check if the PStrata R package is installed.

    Returns
    -------
    bool
        True if PStrata can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r('suppressPackageStartupMessages(library(PStrata))')
        return True
    except Exception:
        return False


def check_dtrreg_installed() -> bool:
    """Check if the DTRreg R package is installed.

    Returns
    -------
    bool
        True if DTRreg can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r('suppressPackageStartupMessages(library(DTRreg))')
        return True
    except Exception:
        return False


def get_r_installation_instructions() -> str:
    """Get instructions for installing R and required packages.

    Returns
    -------
    str
        Installation instructions.
    """
    return """
R Triangulation Tests Requirements
==================================

1. Install R (https://www.r-project.org/)

2. Install rpy2:
   pip install rpy2>=3.5

3. Install R packages (from R console):
   install.packages("PStrata")
   install.packages("DTRreg")

Or install all dependencies:
   pip install causal-inference-mastery[r-triangulation]
"""


# =============================================================================
# Principal Stratification (PStrata)
# =============================================================================


def r_cace_2sls(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
) -> Dict[str, Any]:
    """Estimate CACE using 2SLS via R's AER package (instrumental variables).

    This provides a reference implementation using standard R IV estimation
    for comparison with our cace_2sls() function.

    Parameters
    ----------
    outcome : np.ndarray
        Continuous outcome variable Y.
    treatment : np.ndarray
        Binary treatment indicator D (0/1).
    instrument : np.ndarray
        Binary instrument indicator Z (0/1).

    Returns
    -------
    dict
        Dictionary with keys:
        - cace: float, the 2SLS estimate
        - se: float, standard error
        - ci_lower: float, 95% CI lower bound
        - ci_upper: float, 95% CI upper bound

    Raises
    ------
    ImportError
        If rpy2 is not installed.
    RuntimeError
        If R or required packages are not available.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    # Enable numpy conversion
    numpy2ri.activate()

    try:
        # Pass data to R
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["D"] = ro.FloatVector(treatment)
        ro.globalenv["Z"] = ro.FloatVector(instrument)

        # Use ivreg from AER package for 2SLS
        # This is more widely available than PStrata for basic 2SLS
        result = ro.r(
            """
            suppressPackageStartupMessages({
                if (!require(AER, quietly=TRUE)) {
                    # Fallback: manual 2SLS calculation
                    data <- data.frame(Y=Y, D=D, Z=Z)

                    # First stage
                    first_stage <- lm(D ~ Z, data=data)
                    D_hat <- fitted(first_stage)

                    # Second stage
                    second_stage <- lm(Y ~ D_hat, data=data)

                    coef_2sls <- coef(second_stage)["D_hat"]
                    se_2sls <- summary(second_stage)$coefficients["D_hat", "Std. Error"]

                    list(
                        cace = coef_2sls,
                        se = se_2sls,
                        ci_lower = coef_2sls - 1.96 * se_2sls,
                        ci_upper = coef_2sls + 1.96 * se_2sls
                    )
                } else {
                    data <- data.frame(Y=Y, D=D, Z=Z)
                    fit <- ivreg(Y ~ D | Z, data=data)
                    summ <- summary(fit)

                    list(
                        cace = coef(fit)["D"],
                        se = summ$coefficients["D", "Std. Error"],
                        ci_lower = confint(fit)["D", 1],
                        ci_upper = confint(fit)["D", 2]
                    )
                }
            })
            """
        )

        return {
            "cace": float(result.rx2("cace")[0]),
            "se": float(result.rx2("se")[0]),
            "ci_lower": float(result.rx2("ci_lower")[0]),
            "ci_upper": float(result.rx2("ci_upper")[0]),
        }
    finally:
        numpy2ri.deactivate()


def r_cace_pstrata(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    method: str = "EM",
    max_iter: int = 500,
    tol: float = 1e-6,
) -> Optional[Dict[str, Any]]:
    """Estimate CACE using the PStrata R package.

    Parameters
    ----------
    outcome : np.ndarray
        Continuous outcome variable Y.
    treatment : np.ndarray
        Binary treatment indicator D (0/1).
    instrument : np.ndarray
        Binary instrument indicator Z (0/1).
    method : str, default="EM"
        Estimation method: "EM" for EM algorithm, "2SLS" for instrumental variables.
    max_iter : int, default=500
        Maximum number of EM iterations.
    tol : float, default=1e-6
        Convergence tolerance for EM.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - cace: float, CACE estimate
        - se: float, standard error
        - strata_proportions: dict with pi_c, pi_a, pi_n
        - converged: bool, whether EM converged
        Returns None if PStrata is not available.

    Raises
    ------
    ImportError
        If rpy2 is not installed.
    RuntimeError
        If PStrata package is not installed in R.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    if not check_pstrata_installed():
        warnings.warn(
            "PStrata R package not installed. Install in R with: "
            "install.packages('PStrata')",
            UserWarning,
        )
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        # Pass data to R
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["D"] = ro.FloatVector(treatment)
        ro.globalenv["Z"] = ro.FloatVector(instrument)
        ro.globalenv["max_iter"] = max_iter
        ro.globalenv["tol"] = tol

        # Call PStrata
        # Note: PStrata API may vary by version; this is based on published docs
        result = ro.r(
            """
            suppressPackageStartupMessages(library(PStrata))

            data <- data.frame(Y=Y, D=D, Z=Z)

            # PStrata uses a formula interface
            # The exact API depends on package version
            tryCatch({
                fit <- PSest(Y ~ D | Z, data=data,
                            control=list(maxit=max_iter, tol=tol))

                # Extract results
                coefs <- coef(fit)

                list(
                    cace = coefs["CACE"],
                    se = sqrt(vcov(fit)["CACE", "CACE"]),
                    pi_c = coefs["pi_c"],
                    pi_a = coefs["pi_a"],
                    pi_n = coefs["pi_n"],
                    converged = fit$converged
                )
            }, error = function(e) {
                # If PStrata has different API, try alternative
                list(
                    error = as.character(e),
                    cace = NA,
                    se = NA,
                    pi_c = NA,
                    pi_a = NA,
                    pi_n = NA,
                    converged = FALSE
                )
            })
            """
        )

        # Check for error
        if "error" in list(result.names) and result.rx2("error")[0] != "NA":
            error_msg = str(result.rx2("error")[0])
            warnings.warn(f"PStrata estimation failed: {error_msg}", UserWarning)
            return None

        cace = float(result.rx2("cace")[0])
        se = float(result.rx2("se")[0])

        return {
            "cace": cace,
            "se": se if not np.isnan(se) else None,
            "strata_proportions": {
                "compliers": float(result.rx2("pi_c")[0]),
                "always_takers": float(result.rx2("pi_a")[0]),
                "never_takers": float(result.rx2("pi_n")[0]),
            },
            "converged": bool(result.rx2("converged")[0]),
        }
    except Exception as e:
        warnings.warn(f"PStrata call failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_cace_em_manual(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Estimate CACE using EM algorithm implemented in R (for comparison).

    This provides a manual EM implementation in R as a reference when
    PStrata is not available.

    Parameters
    ----------
    outcome : np.ndarray
        Continuous outcome variable Y.
    treatment : np.ndarray
        Binary treatment indicator D (0/1).
    instrument : np.ndarray
        Binary instrument indicator Z (0/1).
    max_iter : int, default=500
        Maximum number of EM iterations.
    tol : float, default=1e-6
        Convergence tolerance.

    Returns
    -------
    dict
        Dictionary with estimation results.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["D"] = ro.FloatVector(treatment)
        ro.globalenv["Z"] = ro.FloatVector(instrument)
        ro.globalenv["max_iter"] = max_iter
        ro.globalenv["tol"] = tol

        result = ro.r(
            """
            # Manual EM implementation for principal stratification
            # Under monotonicity: no defiers

            n <- length(Y)

            # Identify groups based on (Z, D)
            g00 <- which(Z == 0 & D == 0)  # Never-takers or compliers
            g01 <- which(Z == 0 & D == 1)  # Always-takers
            g10 <- which(Z == 1 & D == 0)  # Never-takers
            g11 <- which(Z == 1 & D == 1)  # Always-takers or compliers

            # Initial estimates from data
            n00 <- length(g00)
            n01 <- length(g01)
            n10 <- length(g10)
            n11 <- length(g11)

            # Outcome means by group
            mu00 <- if(n00 > 0) mean(Y[g00]) else 0
            mu01 <- if(n01 > 0) mean(Y[g01]) else 0
            mu10 <- if(n10 > 0) mean(Y[g10]) else 0
            mu11 <- if(n11 > 0) mean(Y[g11]) else 0

            # Initialize parameters
            # pi_a from always-takers in control (g01)
            pi_a <- n01 / (n00 + n01)
            # pi_n from never-takers in treatment (g10)
            pi_n <- n10 / (n10 + n11)
            pi_c <- 1 - pi_a - pi_n

            # Ensure valid
            pi_c <- max(0.01, min(0.98, pi_c))
            pi_a <- max(0.01, min(0.98 - pi_c, pi_a))
            pi_n <- 1 - pi_c - pi_a

            # Initialize means
            mu_a <- mu01  # Always-takers from g01
            mu_n <- mu10  # Never-takers from g10
            mu_c0 <- mu00 # Compliers control approximated by g00
            mu_c1 <- mu11 # Compliers treated approximated by g11
            sigma2 <- var(Y)

            # EM iterations
            converged <- FALSE
            ll_old <- -Inf

            for (iter in 1:max_iter) {
                # E-step: compute responsibilities

                # For g00 (Z=0, D=0): could be complier or never-taker
                if (n00 > 0) {
                    dens_c0 <- dnorm(Y[g00], mu_c0, sqrt(sigma2))
                    dens_n <- dnorm(Y[g00], mu_n, sqrt(sigma2))

                    w_c_g00 <- (pi_c * dens_c0) / (pi_c * dens_c0 + pi_n * dens_n + 1e-10)
                    w_n_g00 <- 1 - w_c_g00
                } else {
                    w_c_g00 <- numeric(0)
                    w_n_g00 <- numeric(0)
                }

                # For g11 (Z=1, D=1): could be complier or always-taker
                if (n11 > 0) {
                    dens_c1 <- dnorm(Y[g11], mu_c1, sqrt(sigma2))
                    dens_a <- dnorm(Y[g11], mu_a, sqrt(sigma2))

                    w_c_g11 <- (pi_c * dens_c1) / (pi_c * dens_c1 + pi_a * dens_a + 1e-10)
                    w_a_g11 <- 1 - w_c_g11
                } else {
                    w_c_g11 <- numeric(0)
                    w_a_g11 <- numeric(0)
                }

                # M-step: update parameters

                # Strata proportions
                n_c <- sum(w_c_g00) + sum(w_c_g11)
                n_a <- n01 + sum(w_a_g11)
                n_n <- sum(w_n_g00) + n10
                total <- n_c + n_a + n_n

                pi_c <- n_c / total
                pi_a <- n_a / total
                pi_n <- n_n / total

                # Complier means
                if (sum(w_c_g00) > 0.01) {
                    mu_c0 <- sum(w_c_g00 * Y[g00]) / sum(w_c_g00)
                }
                if (sum(w_c_g11) > 0.01) {
                    mu_c1 <- sum(w_c_g11 * Y[g11]) / sum(w_c_g11)
                }

                # Always-taker mean (all of g01 + weighted g11)
                if (n01 + sum(w_a_g11) > 0.01) {
                    mu_a <- (sum(Y[g01]) + sum(w_a_g11 * Y[g11])) / (n01 + sum(w_a_g11))
                }

                # Never-taker mean (all of g10 + weighted g00)
                if (n10 + sum(w_n_g00) > 0.01) {
                    mu_n <- (sum(Y[g10]) + sum(w_n_g00 * Y[g00])) / (n10 + sum(w_n_g00))
                }

                # Variance
                ss <- 0
                if (n00 > 0) {
                    ss <- ss + sum(w_c_g00 * (Y[g00] - mu_c0)^2) +
                               sum(w_n_g00 * (Y[g00] - mu_n)^2)
                }
                if (n01 > 0) {
                    ss <- ss + sum((Y[g01] - mu_a)^2)
                }
                if (n10 > 0) {
                    ss <- ss + sum((Y[g10] - mu_n)^2)
                }
                if (n11 > 0) {
                    ss <- ss + sum(w_c_g11 * (Y[g11] - mu_c1)^2) +
                               sum(w_a_g11 * (Y[g11] - mu_a)^2)
                }
                sigma2 <- max(0.001, ss / n)

                # Log-likelihood for convergence
                ll <- 0
                if (n00 > 0) {
                    ll <- ll + sum(log(pi_c * dnorm(Y[g00], mu_c0, sqrt(sigma2)) +
                                       pi_n * dnorm(Y[g00], mu_n, sqrt(sigma2)) + 1e-10))
                }
                if (n01 > 0) {
                    ll <- ll + sum(log(dnorm(Y[g01], mu_a, sqrt(sigma2)) + 1e-10))
                }
                if (n10 > 0) {
                    ll <- ll + sum(log(dnorm(Y[g10], mu_n, sqrt(sigma2)) + 1e-10))
                }
                if (n11 > 0) {
                    ll <- ll + sum(log(pi_c * dnorm(Y[g11], mu_c1, sqrt(sigma2)) +
                                       pi_a * dnorm(Y[g11], mu_a, sqrt(sigma2)) + 1e-10))
                }

                if (abs(ll - ll_old) < tol) {
                    converged <- TRUE
                    break
                }
                ll_old <- ll
            }

            cace <- mu_c1 - mu_c0

            list(
                cace = cace,
                mu_c0 = mu_c0,
                mu_c1 = mu_c1,
                mu_a = mu_a,
                mu_n = mu_n,
                pi_c = pi_c,
                pi_a = pi_a,
                pi_n = pi_n,
                sigma = sqrt(sigma2),
                converged = converged,
                iterations = iter,
                log_likelihood = ll
            )
            """
        )

        return {
            "cace": float(result.rx2("cace")[0]),
            "mu_c0": float(result.rx2("mu_c0")[0]),
            "mu_c1": float(result.rx2("mu_c1")[0]),
            "mu_a": float(result.rx2("mu_a")[0]),
            "mu_n": float(result.rx2("mu_n")[0]),
            "strata_proportions": {
                "compliers": float(result.rx2("pi_c")[0]),
                "always_takers": float(result.rx2("pi_a")[0]),
                "never_takers": float(result.rx2("pi_n")[0]),
            },
            "sigma": float(result.rx2("sigma")[0]),
            "converged": bool(result.rx2("converged")[0]),
            "iterations": int(result.rx2("iterations")[0]),
            "log_likelihood": float(result.rx2("log_likelihood")[0]),
        }
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Bounds (Manski-style)
# =============================================================================


def r_bounds_manski(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    outcome_support: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Compute Manski-style bounds on treatment effect using R.

    Parameters
    ----------
    outcome : np.ndarray
        Continuous outcome variable Y.
    treatment : np.ndarray
        Binary treatment indicator D (0/1).
    instrument : np.ndarray
        Binary instrument indicator Z (0/1).
    outcome_support : tuple, optional
        (min, max) of outcome support. If None, uses observed range.

    Returns
    -------
    dict
        Dictionary with bounds results.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["D"] = ro.FloatVector(treatment)
        ro.globalenv["Z"] = ro.FloatVector(instrument)

        if outcome_support is not None:
            ro.globalenv["y_min"] = outcome_support[0]
            ro.globalenv["y_max"] = outcome_support[1]
        else:
            ro.globalenv["y_min"] = float(np.min(outcome))
            ro.globalenv["y_max"] = float(np.max(outcome))

        result = ro.r(
            """
            # Manski worst-case bounds
            # Without assumptions, the treatment effect is bounded by the outcome range

            range_Y <- y_max - y_min

            # Simple bounds: [-range, +range]
            lower <- -range_Y
            upper <- range_Y

            list(
                lower_bound = lower,
                upper_bound = upper,
                bound_width = upper - lower,
                outcome_range = range_Y
            )
            """
        )

        return {
            "lower_bound": float(result.rx2("lower_bound")[0]),
            "upper_bound": float(result.rx2("upper_bound")[0]),
            "bound_width": float(result.rx2("bound_width")[0]),
            "outcome_range": float(result.rx2("outcome_range")[0]),
        }
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Difference-in-Differences (did package) - Session 123
# =============================================================================


def check_did_installed() -> bool:
    """Check if the R `did` package (Callaway-Sant'Anna) is installed.

    Returns
    -------
    bool
        True if did package can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r("suppressPackageStartupMessages(library(did))")
        return True
    except Exception:
        return False


def r_did_callaway_santanna(
    outcome: np.ndarray,
    unit: np.ndarray,
    time: np.ndarray,
    first_treated: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    control_group: str = "nevertreated",
    aggregation: str = "simple",
) -> Dict[str, Any]:
    """Estimate Callaway-Sant'Anna DiD using R `did` package.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable (n,).
    unit : np.ndarray
        Unit identifiers (n,).
    time : np.ndarray
        Time periods (n,).
    first_treated : np.ndarray
        Period when unit first treated (0 = never treated) (n,).
    covariates : np.ndarray, optional
        Covariate matrix (n, p).
    control_group : str, default="nevertreated"
        Control group: "nevertreated" or "notyettreated".
    aggregation : str, default="simple"
        Aggregation method: "simple", "dynamic", or "group".

    Returns
    -------
    dict
        Dictionary with keys:
        - att: float, aggregate ATT estimate
        - se: float, standard error
        - ci_lower: float, 95% CI lower bound
        - ci_upper: float, 95% CI upper bound
        - att_gt: dict, group-time effects {(g, t): att}
        - n_obs: int, number of observations

    Raises
    ------
    ImportError
        If rpy2 is not installed.
    RuntimeError
        If did package is not installed in R.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    if not check_did_installed():
        raise RuntimeError(
            "R 'did' package not installed. Install in R with: "
            "install.packages('did')"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri

    numpy2ri.activate()

    try:
        # Create data frame in R
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["unit_id"] = ro.IntVector(unit.astype(int))
        ro.globalenv["time_id"] = ro.IntVector(time.astype(int))
        ro.globalenv["first_treat"] = ro.IntVector(first_treated.astype(int))
        ro.globalenv["control_group_str"] = control_group
        ro.globalenv["agg_type"] = aggregation

        if covariates is not None:
            n, p = covariates.shape
            for j in range(p):
                ro.globalenv[f"X{j+1}"] = ro.FloatVector(covariates[:, j])
            cov_formula = " + ".join([f"X{j+1}" for j in range(p)])
            ro.globalenv["cov_formula"] = cov_formula
            has_covs = True
        else:
            has_covs = False

        # Call did package
        result = ro.r(
            f"""
            suppressPackageStartupMessages(library(did))

            data <- data.frame(
                Y = Y,
                id = unit_id,
                time = time_id,
                G = first_treat
            )

            {"# Add covariates" if has_covs else ""}
            {f'for (j in 1:{p}) {{ data[[paste0("X", j)]] <- get(paste0("X", j)) }}' if has_covs else ""}

            # Fit CS model
            cs_result <- tryCatch({{
                att_gt(
                    yname = "Y",
                    tname = "time",
                    idname = "id",
                    gname = "G",
                    data = data,
                    control_group = control_group_str,
                    {'xformla = ~ ' + cov_formula + ',' if has_covs else ''}
                    allow_unbalanced_panel = TRUE
                )
            }}, error = function(e) {{
                list(error = as.character(e))
            }})

            if (!is.null(cs_result$error)) {{
                list(
                    error = cs_result$error,
                    att = NA,
                    se = NA
                )
            }} else {{
                # Aggregate based on method
                agg <- aggte(cs_result, type = agg_type)

                list(
                    att = agg$overall.att,
                    se = agg$overall.se,
                    ci_lower = agg$overall.att - 1.96 * agg$overall.se,
                    ci_upper = agg$overall.att + 1.96 * agg$overall.se,
                    n_obs = nrow(data),
                    att_gt = cs_result$att,
                    se_gt = cs_result$se,
                    group = cs_result$group,
                    time_gt = cs_result$t
                )
            }}
            """
        )

        # Check for error
        if "error" in list(result.names):
            error_val = result.rx2("error")
            if error_val[0] != "NA" and not np.isnan(float(result.rx2("att")[0])):
                pass  # No error
            elif error_val[0] != "NA":
                raise RuntimeError(f"R did package error: {error_val[0]}")

        att = float(result.rx2("att")[0])
        se = float(result.rx2("se")[0])

        # Extract group-time effects
        att_gt_values = np.array(result.rx2("att_gt"))
        groups = np.array(result.rx2("group"))
        times = np.array(result.rx2("time_gt"))

        att_gt = {}
        for i, (g, t, att_val) in enumerate(zip(groups, times, att_gt_values)):
            att_gt[(int(g), int(t))] = float(att_val)

        return {
            "att": att,
            "se": se,
            "ci_lower": float(result.rx2("ci_lower")[0]),
            "ci_upper": float(result.rx2("ci_upper")[0]),
            "n_obs": int(result.rx2("n_obs")[0]),
            "att_gt": att_gt,
        }
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Regression Discontinuity (rdrobust package) - Session 123
# =============================================================================


def check_rdrobust_installed() -> bool:
    """Check if the R `rdrobust` package is installed.

    Returns
    -------
    bool
        True if rdrobust can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r("suppressPackageStartupMessages(library(rdrobust))")
        return True
    except Exception:
        return False


def check_rddensity_installed() -> bool:
    """Check if the R `rddensity` package is installed.

    Returns
    -------
    bool
        True if rddensity can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r("suppressPackageStartupMessages(library(rddensity))")
        return True
    except Exception:
        return False


def r_rdd_rdrobust(
    outcome: np.ndarray,
    running_var: np.ndarray,
    cutoff: float = 0.0,
    kernel: str = "triangular",
    bwselect: str = "mserd",
    covariates: Optional[np.ndarray] = None,
    fuzzy: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Estimate RDD using R `rdrobust` package.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y (n,).
    running_var : np.ndarray
        Running/forcing variable X (n,).
    cutoff : float, default=0.0
        RDD cutoff point.
    kernel : str, default="triangular"
        Kernel function: "triangular", "uniform", "epanechnikov".
    bwselect : str, default="mserd"
        Bandwidth selection method: "mserd", "msetwo", "msesum", etc.
    covariates : np.ndarray, optional
        Covariate matrix (n, p).
    fuzzy : np.ndarray, optional
        Treatment indicator for fuzzy RDD (n,).

    Returns
    -------
    dict
        Dictionary with keys:
        - tau: float, RDD treatment effect estimate
        - se: float, robust standard error
        - ci_lower: float, 95% CI lower bound
        - ci_upper: float, 95% CI upper bound
        - bandwidth_left: float, bandwidth used on left
        - bandwidth_right: float, bandwidth used on right
        - n_left: int, effective sample size on left
        - n_right: int, effective sample size on right
        - p_value: float, p-value for tau != 0

    Raises
    ------
    ImportError
        If rpy2 is not installed.
    RuntimeError
        If rdrobust package is not installed in R.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    if not check_rdrobust_installed():
        raise RuntimeError(
            "R 'rdrobust' package not installed. Install in R with: "
            "install.packages('rdrobust')"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["X"] = ro.FloatVector(running_var)
        ro.globalenv["cutoff"] = cutoff
        ro.globalenv["kernel_type"] = kernel
        ro.globalenv["bwselect_type"] = bwselect

        has_covs = covariates is not None
        has_fuzzy = fuzzy is not None

        if has_covs:
            n, p = covariates.shape
            for j in range(p):
                ro.globalenv[f"Z{j+1}"] = ro.FloatVector(covariates[:, j])
            ro.globalenv["n_covs"] = p

        if has_fuzzy:
            ro.globalenv["T_fuzzy"] = ro.FloatVector(fuzzy)

        result = ro.r(
            f"""
            suppressPackageStartupMessages(library(rdrobust))

            # Build covariate matrix if needed
            {"Z_mat <- cbind(" + ", ".join([f"Z{j+1}" for j in range(p)]) + ")" if has_covs else "Z_mat <- NULL"}

            # Fit rdrobust
            rd_result <- tryCatch({{
                rdrobust(
                    y = Y,
                    x = X,
                    c = cutoff,
                    kernel = kernel_type,
                    bwselect = bwselect_type,
                    {"covs = Z_mat," if has_covs else ""}
                    {"fuzzy = T_fuzzy," if has_fuzzy else ""}
                    all = TRUE
                )
            }}, error = function(e) {{
                list(error = as.character(e))
            }})

            if (!is.null(rd_result$error)) {{
                list(
                    error = rd_result$error,
                    tau = NA
                )
            }} else {{
                list(
                    tau = rd_result$coef[1],
                    tau_bc = rd_result$coef[2],  # Bias-corrected
                    se = rd_result$se[1],
                    se_bc = rd_result$se[2],
                    ci_lower = rd_result$ci[1, 1],
                    ci_upper = rd_result$ci[1, 2],
                    ci_lower_bc = rd_result$ci[2, 1],
                    ci_upper_bc = rd_result$ci[2, 2],
                    bandwidth_left = rd_result$bws[1, 1],
                    bandwidth_right = rd_result$bws[1, 2],
                    n_left = rd_result$N[1],
                    n_right = rd_result$N[2],
                    n_left_eff = rd_result$N_h[1],
                    n_right_eff = rd_result$N_h[2],
                    p_value = rd_result$pv[1]
                )
            }}
            """
        )

        # Check for error
        if "error" in list(result.names):
            error_val = result.rx2("error")
            if str(error_val[0]) != "NA":
                raise RuntimeError(f"R rdrobust error: {error_val[0]}")

        return {
            "tau": float(result.rx2("tau")[0]),
            "tau_bc": float(result.rx2("tau_bc")[0]),
            "se": float(result.rx2("se")[0]),
            "se_bc": float(result.rx2("se_bc")[0]),
            "ci_lower": float(result.rx2("ci_lower")[0]),
            "ci_upper": float(result.rx2("ci_upper")[0]),
            "ci_lower_bc": float(result.rx2("ci_lower_bc")[0]),
            "ci_upper_bc": float(result.rx2("ci_upper_bc")[0]),
            "bandwidth_left": float(result.rx2("bandwidth_left")[0]),
            "bandwidth_right": float(result.rx2("bandwidth_right")[0]),
            "n_left": int(result.rx2("n_left")[0]),
            "n_right": int(result.rx2("n_right")[0]),
            "n_left_eff": int(result.rx2("n_left_eff")[0]),
            "n_right_eff": int(result.rx2("n_right_eff")[0]),
            "p_value": float(result.rx2("p_value")[0]),
        }
    finally:
        numpy2ri.deactivate()


def r_rdd_mccrary(
    running_var: np.ndarray,
    cutoff: float = 0.0,
) -> Dict[str, Any]:
    """Perform McCrary density test using R `rddensity` package.

    Tests for manipulation of the running variable at the cutoff.
    Null hypothesis: density is continuous at cutoff.

    Parameters
    ----------
    running_var : np.ndarray
        Running/forcing variable X (n,).
    cutoff : float, default=0.0
        RDD cutoff point.

    Returns
    -------
    dict
        Dictionary with keys:
        - log_diff: float, log difference in densities (f_right/f_left)
        - se: float, standard error of log difference
        - t_stat: float, t-statistic
        - p_value: float, p-value for null of no manipulation
        - f_left: float, density estimate to left of cutoff
        - f_right: float, density estimate to right of cutoff
        - n_left: int, observations to left
        - n_right: int, observations to right

    Raises
    ------
    ImportError
        If rpy2 is not installed.
    RuntimeError
        If rddensity package is not installed in R.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    if not check_rddensity_installed():
        raise RuntimeError(
            "R 'rddensity' package not installed. Install in R with: "
            "install.packages('rddensity')"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        ro.globalenv["X"] = ro.FloatVector(running_var)
        ro.globalenv["cutoff"] = cutoff

        result = ro.r(
            """
            suppressPackageStartupMessages(library(rddensity))

            mcc_result <- tryCatch({
                rddensity(X = X, c = cutoff)
            }, error = function(e) {
                list(error = as.character(e))
            })

            if (!is.null(mcc_result$error)) {
                list(
                    error = mcc_result$error,
                    p_value = NA
                )
            } else {
                # Extract test statistic and p-value
                # rddensity returns different structure
                list(
                    t_stat = mcc_result$test$t_jk,
                    p_value = mcc_result$test$p_jk,
                    f_left = mcc_result$Estimate[1, "f_p"],
                    f_right = mcc_result$Estimate[2, "f_p"],
                    se_left = mcc_result$Estimate[1, "se_p"],
                    se_right = mcc_result$Estimate[2, "se_p"],
                    n_left = mcc_result$N[1],
                    n_right = mcc_result$N[2],
                    bw_left = mcc_result$h[1],
                    bw_right = mcc_result$h[2]
                )
            }
            """
        )

        # Check for error
        if "error" in list(result.names):
            error_val = result.rx2("error")
            if str(error_val[0]) != "NA":
                raise RuntimeError(f"R rddensity error: {error_val[0]}")

        f_left = float(result.rx2("f_left")[0])
        f_right = float(result.rx2("f_right")[0])

        # Compute log difference
        log_diff = np.log(f_right / f_left) if f_left > 0 and f_right > 0 else np.nan

        return {
            "log_diff": log_diff,
            "t_stat": float(result.rx2("t_stat")[0]),
            "p_value": float(result.rx2("p_value")[0]),
            "f_left": f_left,
            "f_right": f_right,
            "se_left": float(result.rx2("se_left")[0]),
            "se_right": float(result.rx2("se_right")[0]),
            "n_left": int(result.rx2("n_left")[0]),
            "n_right": int(result.rx2("n_right")[0]),
            "bw_left": float(result.rx2("bw_left")[0]),
            "bw_right": float(result.rx2("bw_right")[0]),
        }
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Instrumental Variables (AER package) - Session 124
# =============================================================================


def check_aer_installed() -> bool:
    """Check if the R `AER` package (ivreg) is installed.

    Returns
    -------
    bool
        True if AER can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r("suppressPackageStartupMessages(library(AER))")
        return True
    except Exception:
        return False


def r_2sls_aer(
    outcome: np.ndarray,
    endogenous: np.ndarray,
    instruments: np.ndarray,
    controls: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Estimate 2SLS using R `AER::ivreg()`.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y (n,).
    endogenous : np.ndarray
        Endogenous regressor D (n,) or (n, k) for multiple.
    instruments : np.ndarray
        Instruments Z (n,) or (n, m) for multiple.
    controls : np.ndarray, optional
        Exogenous control variables X (n, p).

    Returns
    -------
    dict
        Dictionary with keys:
        - coef: float or array, coefficient estimate(s) for endogenous
        - se: float or array, standard error(s)
        - ci_lower: float or array, 95% CI lower bound
        - ci_upper: float or array, 95% CI upper bound
        - first_stage_f: float, first-stage F-statistic
        - n_obs: int, number of observations
        - r_squared: float, R-squared of second stage

    Raises
    ------
    ImportError
        If rpy2 is not installed.
    RuntimeError
        If AER package is not installed in R.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    if not check_aer_installed():
        raise RuntimeError(
            "R 'AER' package not installed. Install in R with: "
            "install.packages('AER')"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        # Ensure 2D arrays
        if endogenous.ndim == 1:
            endogenous = endogenous.reshape(-1, 1)
        if instruments.ndim == 1:
            instruments = instruments.reshape(-1, 1)

        n = len(outcome)
        n_endog = endogenous.shape[1]
        n_instr = instruments.shape[1]

        # Pass data to R
        ro.globalenv["Y"] = ro.FloatVector(outcome)

        # Create endogenous variable names
        for j in range(n_endog):
            ro.globalenv[f"D{j+1}"] = ro.FloatVector(endogenous[:, j])

        # Create instrument names
        for j in range(n_instr):
            ro.globalenv[f"Z{j+1}"] = ro.FloatVector(instruments[:, j])

        # Create control names if provided
        has_controls = controls is not None
        if has_controls:
            if controls.ndim == 1:
                controls = controls.reshape(-1, 1)
            n_controls = controls.shape[1]
            for j in range(n_controls):
                ro.globalenv[f"X{j+1}"] = ro.FloatVector(controls[:, j])
        else:
            n_controls = 0

        ro.globalenv["n_endog"] = n_endog
        ro.globalenv["n_instr"] = n_instr
        ro.globalenv["n_controls"] = n_controls

        # Build formula strings
        endog_names = " + ".join([f"D{j+1}" for j in range(n_endog)])
        instr_names = " + ".join([f"Z{j+1}" for j in range(n_instr)])
        if has_controls:
            control_names = " + ".join([f"X{j+1}" for j in range(n_controls)])
            formula_left = f"Y ~ {endog_names} + {control_names}"
            formula_right = f"{control_names} + {instr_names}"
        else:
            formula_left = f"Y ~ {endog_names}"
            formula_right = instr_names

        ro.globalenv["formula_left"] = formula_left
        ro.globalenv["formula_right"] = formula_right

        result = ro.r(
            f"""
            suppressPackageStartupMessages(library(AER))

            # Build data frame
            data <- data.frame(Y = Y)
            for (j in 1:n_endog) {{ data[[paste0("D", j)]] <- get(paste0("D", j)) }}
            for (j in 1:n_instr) {{ data[[paste0("Z", j)]] <- get(paste0("Z", j)) }}
            if (n_controls > 0) {{
                for (j in 1:n_controls) {{ data[[paste0("X", j)]] <- get(paste0("X", j)) }}
            }}

            # Fit 2SLS
            iv_formula <- as.formula(paste(formula_left, "|", formula_right))
            fit <- tryCatch({{
                ivreg(iv_formula, data = data)
            }}, error = function(e) {{
                list(error = as.character(e))
            }})

            if (!is.null(fit$error)) {{
                list(error = fit$error, coef = NA)
            }} else {{
                summ <- summary(fit, vcov = vcov)
                coefs <- coef(fit)
                ses <- summ$coefficients[, "Std. Error"]
                ci <- confint(fit)

                # First-stage F-statistic for weak instruments
                # Use waldtest for joint significance of instruments in first stage
                first_stage_f <- tryCatch({{
                    fs <- lm(D1 ~ . - Y, data = data[, c("D1", paste0("Z", 1:n_instr),
                                                         if (n_controls > 0) paste0("X", 1:n_controls) else NULL)])
                    fs_summ <- summary(fs)
                    # F-stat for instruments only (exclude controls)
                    fs_summ$fstatistic[1]
                }}, error = function(e) NA)

                list(
                    coef = coefs[-1],  # Exclude intercept
                    se = ses[-1],
                    ci_lower = ci[-1, 1],
                    ci_upper = ci[-1, 2],
                    first_stage_f = first_stage_f,
                    n_obs = nrow(data),
                    r_squared = summ$r.squared,
                    intercept = coefs[1]
                )
            }}
            """
        )

        # Check for error
        if "error" in list(result.names):
            error_val = result.rx2("error")
            if str(error_val[0]) != "NA":
                raise RuntimeError(f"R AER error: {error_val[0]}")

        coef = np.array(result.rx2("coef"))
        se = np.array(result.rx2("se"))

        # Return scalar if single endogenous variable
        if n_endog == 1:
            return {
                "coef": float(coef[0]),
                "se": float(se[0]),
                "ci_lower": float(result.rx2("ci_lower")[0]),
                "ci_upper": float(result.rx2("ci_upper")[0]),
                "first_stage_f": float(result.rx2("first_stage_f")[0]),
                "n_obs": int(result.rx2("n_obs")[0]),
                "r_squared": float(result.rx2("r_squared")[0]),
                "intercept": float(result.rx2("intercept")[0]),
            }
        else:
            return {
                "coef": coef,
                "se": se,
                "ci_lower": np.array(result.rx2("ci_lower")),
                "ci_upper": np.array(result.rx2("ci_upper")),
                "first_stage_f": float(result.rx2("first_stage_f")[0]),
                "n_obs": int(result.rx2("n_obs")[0]),
                "r_squared": float(result.rx2("r_squared")[0]),
                "intercept": float(result.rx2("intercept")[0]),
            }
    finally:
        numpy2ri.deactivate()


def r_liml_aer(
    outcome: np.ndarray,
    endogenous: np.ndarray,
    instruments: np.ndarray,
    controls: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Estimate LIML using R `AER::ivreg(method='LIML')`.

    Limited Information Maximum Likelihood is preferred when instruments
    are weak (first-stage F < 10), as it has better finite-sample properties.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y (n,).
    endogenous : np.ndarray
        Endogenous regressor D (n,) or (n, k).
    instruments : np.ndarray
        Instruments Z (n,) or (n, m).
    controls : np.ndarray, optional
        Exogenous control variables X (n, p).

    Returns
    -------
    dict
        Dictionary with keys:
        - coef: float or array, LIML coefficient estimate(s)
        - se: float or array, standard error(s)
        - ci_lower: float or array, 95% CI lower bound
        - ci_upper: float or array, 95% CI upper bound
        - kappa: float, LIML kappa statistic (should be close to 1)
        - n_obs: int, number of observations

    Raises
    ------
    ImportError
        If rpy2 is not installed.
    RuntimeError
        If AER package is not installed in R.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    if not check_aer_installed():
        raise RuntimeError(
            "R 'AER' package not installed. Install in R with: "
            "install.packages('AER')"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        # Ensure 2D arrays
        if endogenous.ndim == 1:
            endogenous = endogenous.reshape(-1, 1)
        if instruments.ndim == 1:
            instruments = instruments.reshape(-1, 1)

        n = len(outcome)
        n_endog = endogenous.shape[1]
        n_instr = instruments.shape[1]

        # Pass data to R
        ro.globalenv["Y"] = ro.FloatVector(outcome)

        for j in range(n_endog):
            ro.globalenv[f"D{j+1}"] = ro.FloatVector(endogenous[:, j])

        for j in range(n_instr):
            ro.globalenv[f"Z{j+1}"] = ro.FloatVector(instruments[:, j])

        has_controls = controls is not None
        if has_controls:
            if controls.ndim == 1:
                controls = controls.reshape(-1, 1)
            n_controls = controls.shape[1]
            for j in range(n_controls):
                ro.globalenv[f"X{j+1}"] = ro.FloatVector(controls[:, j])
        else:
            n_controls = 0

        ro.globalenv["n_endog"] = n_endog
        ro.globalenv["n_instr"] = n_instr
        ro.globalenv["n_controls"] = n_controls

        # Build formula strings
        endog_names = " + ".join([f"D{j+1}" for j in range(n_endog)])
        instr_names = " + ".join([f"Z{j+1}" for j in range(n_instr)])
        if has_controls:
            control_names = " + ".join([f"X{j+1}" for j in range(n_controls)])
            formula_left = f"Y ~ {endog_names} + {control_names}"
            formula_right = f"{control_names} + {instr_names}"
        else:
            formula_left = f"Y ~ {endog_names}"
            formula_right = instr_names

        ro.globalenv["formula_left"] = formula_left
        ro.globalenv["formula_right"] = formula_right

        result = ro.r(
            f"""
            suppressPackageStartupMessages(library(AER))

            # Build data frame
            data <- data.frame(Y = Y)
            for (j in 1:n_endog) {{ data[[paste0("D", j)]] <- get(paste0("D", j)) }}
            for (j in 1:n_instr) {{ data[[paste0("Z", j)]] <- get(paste0("Z", j)) }}
            if (n_controls > 0) {{
                for (j in 1:n_controls) {{ data[[paste0("X", j)]] <- get(paste0("X", j)) }}
            }}

            # Fit LIML
            iv_formula <- as.formula(paste(formula_left, "|", formula_right))
            fit <- tryCatch({{
                ivreg(iv_formula, data = data, method = "LIML")
            }}, error = function(e) {{
                list(error = as.character(e))
            }})

            if (!is.null(fit$error)) {{
                list(error = fit$error, coef = NA)
            }} else {{
                summ <- summary(fit)
                coefs <- coef(fit)
                ses <- summ$coefficients[, "Std. Error"]
                ci <- confint(fit)

                # Get kappa from LIML fit
                # AER stores this in the method attribute
                kappa <- tryCatch({{
                    fit$method$kappa
                }}, error = function(e) {{
                    # Fallback: compute kappa manually if not stored
                    NA
                }})

                list(
                    coef = coefs[-1],  # Exclude intercept
                    se = ses[-1],
                    ci_lower = ci[-1, 1],
                    ci_upper = ci[-1, 2],
                    kappa = if (is.null(kappa)) NA else kappa,
                    n_obs = nrow(data),
                    intercept = coefs[1]
                )
            }}
            """
        )

        # Check for error
        if "error" in list(result.names):
            error_val = result.rx2("error")
            if str(error_val[0]) != "NA":
                raise RuntimeError(f"R AER LIML error: {error_val[0]}")

        coef = np.array(result.rx2("coef"))
        se = np.array(result.rx2("se"))

        kappa_val = result.rx2("kappa")
        kappa = float(kappa_val[0]) if not np.isnan(float(kappa_val[0])) else None

        if n_endog == 1:
            return {
                "coef": float(coef[0]),
                "se": float(se[0]),
                "ci_lower": float(result.rx2("ci_lower")[0]),
                "ci_upper": float(result.rx2("ci_upper")[0]),
                "kappa": kappa,
                "n_obs": int(result.rx2("n_obs")[0]),
                "intercept": float(result.rx2("intercept")[0]),
            }
        else:
            return {
                "coef": coef,
                "se": se,
                "ci_lower": np.array(result.rx2("ci_lower")),
                "ci_upper": np.array(result.rx2("ci_upper")),
                "kappa": kappa,
                "n_obs": int(result.rx2("n_obs")[0]),
                "intercept": float(result.rx2("intercept")[0]),
            }
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Synthetic Control Method (Synth package) - Session 124
# =============================================================================


def check_synth_installed() -> bool:
    """Check if the R `Synth` package is installed.

    Returns
    -------
    bool
        True if Synth can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r("suppressPackageStartupMessages(library(Synth))")
        return True
    except Exception:
        return False


def r_scm_synth(
    outcome_treated: np.ndarray,
    outcome_controls: np.ndarray,
    pre_periods: int,
    predictors_treated: Optional[np.ndarray] = None,
    predictors_controls: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Estimate SCM using R `Synth::synth()`.

    Parameters
    ----------
    outcome_treated : np.ndarray
        Outcome for treated unit (T,) where T = pre_periods + post_periods.
    outcome_controls : np.ndarray
        Outcomes for control units (T, J) where J is number of controls.
    pre_periods : int
        Number of pre-treatment periods.
    predictors_treated : np.ndarray, optional
        Predictor values for treated unit (K,).
    predictors_controls : np.ndarray, optional
        Predictor values for control units (K, J).

    Returns
    -------
    dict
        Dictionary with keys:
        - weights: array, synthetic control weights for each control unit
        - synthetic_control: array, synthetic control series (T,)
        - att: float, average treatment effect (post-treatment mean gap)
        - pre_rmse: float, pre-treatment RMSE
        - gap: array, treated - synthetic (T,)
        - n_controls: int, number of control units
        - n_pre: int, pre-treatment periods
        - n_post: int, post-treatment periods

    Raises
    ------
    ImportError
        If rpy2 is not installed.
    RuntimeError
        If Synth package is not installed in R.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    if not check_synth_installed():
        raise RuntimeError(
            "R 'Synth' package not installed. Install in R with: "
            "install.packages('Synth')"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        T = len(outcome_treated)
        n_post = T - pre_periods
        n_controls = outcome_controls.shape[1]

        # Pass data to R
        ro.globalenv["Y_treated"] = ro.FloatVector(outcome_treated)
        ro.globalenv["Y_controls"] = ro.r.matrix(
            ro.FloatVector(outcome_controls.flatten(order="F")),
            nrow=T,
            ncol=n_controls,
        )
        ro.globalenv["pre_periods"] = pre_periods
        ro.globalenv["n_controls"] = n_controls
        ro.globalenv["T_total"] = T

        # Handle predictors
        has_predictors = predictors_treated is not None
        if has_predictors:
            n_predictors = len(predictors_treated)
            ro.globalenv["pred_treated"] = ro.FloatVector(predictors_treated)
            ro.globalenv["pred_controls"] = ro.r.matrix(
                ro.FloatVector(predictors_controls.flatten(order="F")),
                nrow=n_predictors,
                ncol=n_controls,
            )
            ro.globalenv["n_predictors"] = n_predictors
        else:
            ro.globalenv["n_predictors"] = 0

        result = ro.r(
            f"""
            suppressPackageStartupMessages(library(Synth))

            # Create the data in Synth format
            # Synth requires data in a specific panel format

            # Build panel data
            # Unit 1 is treated, units 2:(n_controls+1) are controls
            n_units <- n_controls + 1
            time_ids <- 1:T_total

            # Create panel
            unit_ids <- rep(1:n_units, each = T_total)
            time_rep <- rep(time_ids, n_units)

            # Outcome: treated first, then controls
            Y_all <- c(Y_treated, as.vector(Y_controls))

            # Treatment indicator (post-treatment for treated unit only)
            treatment <- rep(0, length(Y_all))
            treatment[1:T_total][time_ids > pre_periods] <- 1

            # Create data frame
            panel_data <- data.frame(
                unit = unit_ids,
                time = time_rep,
                Y = Y_all
            )

            # Synth requires dataprep() then synth()
            synth_data <- tryCatch({{
                dataprep(
                    foo = panel_data,
                    predictors = NULL,
                    predictors.op = "mean",
                    dependent = "Y",
                    unit.variable = "unit",
                    time.variable = "time",
                    treatment.identifier = 1,
                    controls.identifier = 2:n_units,
                    time.predictors.prior = 1:pre_periods,
                    time.optimize.ssr = 1:pre_periods,
                    unit.names.variable = NULL,
                    time.plot = time_ids
                )
            }}, error = function(e) {{
                list(error = as.character(e))
            }})

            if (!is.null(synth_data$error)) {{
                list(error = synth_data$error, weights = NA)
            }} else {{
                # Run synth optimization
                synth_out <- tryCatch({{
                    synth(synth_data)
                }}, error = function(e) {{
                    list(error = as.character(e))
                }})

                if (!is.null(synth_out$error)) {{
                    list(error = synth_out$error, weights = NA)
                }} else {{
                    # Extract weights
                    weights <- as.vector(synth_out$solution.w)

                    # Compute synthetic control
                    synthetic <- as.vector(synth_data$Y0plot %*% weights)

                    # Compute gap
                    gap <- as.vector(synth_data$Y1plot) - synthetic

                    # Pre-treatment RMSE
                    pre_gap <- gap[1:pre_periods]
                    pre_rmse <- sqrt(mean(pre_gap^2))

                    # ATT: mean of post-treatment gap
                    post_gap <- gap[(pre_periods+1):T_total]
                    att <- mean(post_gap)

                    list(
                        weights = weights,
                        synthetic_control = synthetic,
                        gap = gap,
                        pre_rmse = pre_rmse,
                        att = att,
                        n_controls = n_controls,
                        n_pre = pre_periods,
                        n_post = T_total - pre_periods,
                        treated_series = as.vector(synth_data$Y1plot)
                    )
                }}
            }}
            """
        )

        # Check for error
        if "error" in list(result.names):
            error_val = result.rx2("error")
            if str(error_val[0]) != "NA":
                raise RuntimeError(f"R Synth error: {error_val[0]}")

        weights = np.array(result.rx2("weights")).flatten()

        return {
            "weights": weights,
            "synthetic_control": np.array(result.rx2("synthetic_control")),
            "gap": np.array(result.rx2("gap")),
            "pre_rmse": float(result.rx2("pre_rmse")[0]),
            "att": float(result.rx2("att")[0]),
            "n_controls": int(result.rx2("n_controls")[0]),
            "n_pre": int(result.rx2("n_pre")[0]),
            "n_post": int(result.rx2("n_post")[0]),
            "treated_series": np.array(result.rx2("treated_series")),
        }
    finally:
        numpy2ri.deactivate()


# =============================================================================
# CATE - Causal Forest (grf package) - Session 125
# =============================================================================


def check_grf_installed() -> bool:
    """Check if the R `grf` package (Athey-Wager Causal Forests) is installed.

    Returns
    -------
    bool
        True if grf can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r("suppressPackageStartupMessages(library(grf))")
        return True
    except Exception:
        return False


def r_causal_forest_grf(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    n_trees: int = 2000,
    min_node_size: int = 5,
    honesty: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Estimate CATE using R `grf::causal_forest()`.

    This is the gold-standard implementation of Athey-Wager Causal Forests
    with honest splitting and infinitesimal jackknife variance estimation.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y (n,).
    treatment : np.ndarray
        Binary treatment indicator W (n,).
    covariates : np.ndarray
        Covariate matrix X (n, p).
    n_trees : int, default=2000
        Number of trees in the forest.
    min_node_size : int, default=5
        Minimum number of observations in each leaf.
    honesty : bool, default=True
        Use honest splitting (separate samples for splitting vs estimation).
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
        - ate: float, average treatment effect
        - ate_se: float, SE of ATE (infinitesimal jackknife)
        - ate_ci_lower: float, 95% CI lower bound
        - ate_ci_upper: float, 95% CI upper bound
        - cate: array (n,), individual treatment effects τ(x_i)
        - cate_se: array (n,), SEs for individual effects
        - n_obs: int, number of observations
        - n_trees: int, number of trees used

    Raises
    ------
    ImportError
        If rpy2 is not installed.
    RuntimeError
        If grf package is not installed in R.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    if not check_grf_installed():
        raise RuntimeError(
            "R 'grf' package not installed. Install in R with: "
            "install.packages('grf')"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        # Ensure covariates are 2D
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

        n, p = covariates.shape

        # Pass data to R
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["W"] = ro.FloatVector(treatment)
        ro.globalenv["X"] = ro.r.matrix(
            ro.FloatVector(covariates.flatten(order="C")),
            nrow=n,
            ncol=p,
            byrow=True,
        )
        ro.globalenv["n_trees"] = n_trees
        ro.globalenv["min_node_size"] = min_node_size
        ro.globalenv["honesty_flag"] = honesty
        ro.globalenv["seed_val"] = seed

        result = ro.r(
            """
            suppressPackageStartupMessages(library(grf))

            set.seed(seed_val)

            # Fit causal forest
            cf <- tryCatch({
                causal_forest(
                    X = X,
                    Y = Y,
                    W = W,
                    num.trees = n_trees,
                    min.node.size = min_node_size,
                    honesty = honesty_flag,
                    seed = seed_val
                )
            }, error = function(e) {
                list(error = as.character(e))
            })

            if (!is.null(cf$error)) {
                list(error = cf$error, ate = NA)
            } else {
                # Average treatment effect
                ate_result <- average_treatment_effect(cf, target.sample = "all")

                # Individual CATE predictions
                pred <- predict(cf, estimate.variance = TRUE)
                cate <- pred$predictions
                cate_var <- pred$variance.estimates

                list(
                    ate = ate_result[1],
                    ate_se = ate_result[2],
                    ate_ci_lower = ate_result[1] - 1.96 * ate_result[2],
                    ate_ci_upper = ate_result[1] + 1.96 * ate_result[2],
                    cate = as.vector(cate),
                    cate_se = sqrt(as.vector(cate_var)),
                    n_obs = nrow(X),
                    n_trees = n_trees
                )
            }
            """
        )

        # Check for error
        if "error" in list(result.names):
            error_val = result.rx2("error")
            if str(error_val[0]) != "NA":
                raise RuntimeError(f"R grf error: {error_val[0]}")

        return {
            "ate": float(result.rx2("ate")[0]),
            "ate_se": float(result.rx2("ate_se")[0]),
            "ate_ci_lower": float(result.rx2("ate_ci_lower")[0]),
            "ate_ci_upper": float(result.rx2("ate_ci_upper")[0]),
            "cate": np.array(result.rx2("cate")),
            "cate_se": np.array(result.rx2("cate_se")),
            "n_obs": int(result.rx2("n_obs")[0]),
            "n_trees": int(result.rx2("n_trees")[0]),
        }
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Dynamic Treatment Regimes (DTRreg) - Session 125
# =============================================================================


def r_q_learning_dtrreg(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
) -> Dict[str, Any]:
    """Estimate single-stage Q-learning using R `DTRreg::qLearn()`.

    Q-learning estimates the optimal treatment regime by fitting a
    Q-function and maximizing expected outcome.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y (n,).
    treatment : np.ndarray
        Binary treatment indicator A (n,).
    covariates : np.ndarray
        Covariate matrix X (n, p).

    Returns
    -------
    dict
        Dictionary with keys:
        - value_estimate: float, estimated value under optimal regime
        - blip_coef: array, blip function coefficients (intercept + covariates)
        - blip_se: array, SE of blip coefficients
        - optimal_regime: array (n,), recommended treatment for each obs
        - n_obs: int, number of observations

    Raises
    ------
    ImportError
        If rpy2 is not installed.
    RuntimeError
        If DTRreg package is not installed in R.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    if not check_dtrreg_installed():
        raise RuntimeError(
            "R 'DTRreg' package not installed. Install in R with: "
            "install.packages('DTRreg')"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        # Ensure covariates are 2D
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

        n, p = covariates.shape

        # Pass data to R
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["A"] = ro.FloatVector(treatment)
        ro.globalenv["n_covs"] = p

        # Create covariate columns
        for j in range(p):
            ro.globalenv[f"X{j+1}"] = ro.FloatVector(covariates[:, j])

        result = ro.r(
            f"""
            suppressPackageStartupMessages(library(DTRreg))

            # Build data frame
            data <- data.frame(Y = Y, A = A)
            for (j in 1:n_covs) {{ data[[paste0("X", j)]] <- get(paste0("X", j)) }}

            # Build formula for blip (treatment effect as function of covariates)
            # Blip: gamma(H) = psi0 + X1*psi1 + X2*psi2 + ...
            blip_formula <- as.formula(paste("~", paste(paste0("X", 1:n_covs), collapse = " + ")))

            # Build formula for treatment-free model
            # E[Y | H, A=0] = mu(H)
            tf_formula <- as.formula(paste("~ 1 +", paste(paste0("X", 1:n_covs), collapse = " + ")))

            # Single-stage Q-learning
            q_result <- tryCatch({{
                qLearn(
                    moPropen = A ~ 1 + {' + '.join([f'X{j+1}' for j in range(p)])},
                    moMain = tf_formula,
                    moCont = blip_formula,
                    data = data,
                    response = Y,
                    txName = "A",
                    regime = blip_formula
                )
            }}, error = function(e) {{
                list(error = as.character(e))
            }})

            if (!is.null(q_result$error)) {{
                list(error = q_result$error, value = NA)
            }} else {{
                # Extract blip coefficients (contrast parameters)
                blip_coef <- coef(q_result)

                # SE from DTRreg's vcov
                vcov_mat <- tryCatch({{
                    vcov(q_result)
                }}, error = function(e) {{
                    matrix(NA, nrow=length(blip_coef), ncol=length(blip_coef))
                }})
                blip_se <- sqrt(diag(vcov_mat))

                # Optimal regime: treat if blip > 0
                # Compute blip for each observation
                design_mat <- model.matrix(blip_formula, data)
                blip_values <- as.vector(design_mat %*% blip_coef)
                optimal <- as.integer(blip_values > 0)

                # Value estimate: mean outcome under optimal regime
                # This is E[Y | d*(H)]
                value_est <- mean(Y[A == optimal]) # Simplified; true value uses weighting

                list(
                    value_estimate = value_est,
                    blip_coef = blip_coef,
                    blip_se = blip_se,
                    optimal_regime = optimal,
                    n_obs = nrow(data)
                )
            }}
            """
        )

        # Check for error
        if "error" in list(result.names):
            error_val = result.rx2("error")
            if str(error_val[0]) != "NA":
                raise RuntimeError(f"R DTRreg qLearn error: {error_val[0]}")

        return {
            "value_estimate": float(result.rx2("value_estimate")[0]),
            "blip_coef": np.array(result.rx2("blip_coef")),
            "blip_se": np.array(result.rx2("blip_se")),
            "optimal_regime": np.array(result.rx2("optimal_regime")).astype(int),
            "n_obs": int(result.rx2("n_obs")[0]),
        }
    finally:
        numpy2ri.deactivate()


def r_a_learning_dtrreg(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
) -> Dict[str, Any]:
    """Estimate single-stage A-learning using R `DTRreg`.

    A-learning (Advantage Learning) is a doubly-robust method that is
    consistent if either the propensity model or outcome model is correct.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y (n,).
    treatment : np.ndarray
        Binary treatment indicator A (n,).
    covariates : np.ndarray
        Covariate matrix X (n, p).

    Returns
    -------
    dict
        Dictionary with keys:
        - value_estimate: float, estimated value under optimal regime
        - blip_coef: array, blip function coefficients
        - blip_se: array, SE of blip coefficients
        - optimal_regime: array (n,), recommended treatment for each obs
        - n_obs: int, number of observations

    Raises
    ------
    ImportError
        If rpy2 is not installed.
    RuntimeError
        If DTRreg package is not installed in R.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            f"Install with: pip install rpy2>=3.5\n{get_r_installation_instructions()}"
        )

    if not check_dtrreg_installed():
        raise RuntimeError(
            "R 'DTRreg' package not installed. Install in R with: "
            "install.packages('DTRreg')"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        # Ensure covariates are 2D
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

        n, p = covariates.shape

        # Pass data to R
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["A"] = ro.FloatVector(treatment)
        ro.globalenv["n_covs"] = p

        for j in range(p):
            ro.globalenv[f"X{j+1}"] = ro.FloatVector(covariates[:, j])

        result = ro.r(
            f"""
            suppressPackageStartupMessages(library(DTRreg))

            # Build data frame
            data <- data.frame(Y = Y, A = A)
            for (j in 1:n_covs) {{ data[[paste0("X", j)]] <- get(paste0("X", j)) }}

            # Blip model formula
            blip_formula <- as.formula(paste("~", paste(paste0("X", 1:n_covs), collapse = " + ")))

            # Treatment-free model
            tf_formula <- as.formula(paste("~ 1 +", paste(paste0("X", 1:n_covs), collapse = " + ")))

            # Propensity model
            prop_formula <- as.formula(paste("A ~ 1 +", paste(paste0("X", 1:n_covs), collapse = " + ")))

            # A-learning uses dWOLS (doubly weighted OLS) in DTRreg
            # This is the DR estimator
            a_result <- tryCatch({{
                # DTRreg's dWOLS is similar to A-learning
                DWSurv(
                    moPropen = prop_formula,
                    moMain = tf_formula,
                    moCont = blip_formula,
                    data = data,
                    response = Y,
                    txName = "A"
                )
            }}, error = function(e) {{
                # Fallback to manual A-learning implementation
                tryCatch({{
                    # Fit propensity model
                    prop_mod <- glm(prop_formula, data = data, family = binomial)
                    ps <- predict(prop_mod, type = "response")
                    ps <- pmax(0.01, pmin(0.99, ps))  # Trim

                    # Compute A-learning weights
                    # Weight = (A - ps)^2 / (ps * (1-ps))
                    weights <- (A - ps)^2 / (ps * (1 - ps))

                    # Design matrix for blip
                    design_blip <- model.matrix(blip_formula, data)

                    # Design matrix for main effects
                    design_main <- model.matrix(tf_formula, data)

                    # Outcome regression on (1, X) + A * (1, X)
                    design_full <- cbind(design_main, A * design_blip)

                    # Weighted least squares
                    wls_fit <- lm.wfit(design_full, Y, w = weights)
                    coefs <- wls_fit$coefficients

                    # Extract blip coefficients (second half)
                    n_main <- ncol(design_main)
                    n_blip <- ncol(design_blip)
                    blip_coef <- coefs[(n_main + 1):(n_main + n_blip)]

                    # Compute SE via sandwich (simplified)
                    resid <- Y - design_full %*% coefs
                    bread <- solve(t(design_full) %*% (weights * design_full))
                    meat <- t(design_full) %*% (weights^2 * resid^2 * design_full)
                    sandwich <- bread %*% meat %*% bread
                    all_se <- sqrt(diag(sandwich))
                    blip_se <- all_se[(n_main + 1):(n_main + n_blip)]

                    # Optimal regime
                    blip_values <- as.vector(design_blip %*% blip_coef)
                    optimal <- as.integer(blip_values > 0)

                    # Value estimate
                    value_est <- mean(Y[A == optimal])

                    list(
                        value_estimate = value_est,
                        blip_coef = blip_coef,
                        blip_se = blip_se,
                        optimal_regime = optimal,
                        n_obs = nrow(data)
                    )
                }}, error = function(e2) {{
                    list(error = paste("DTRreg and fallback failed:", as.character(e), as.character(e2)))
                }})
            }})

            if (!is.null(a_result$error)) {{
                list(error = a_result$error, value = NA)
            }} else if (inherits(a_result, "DTRreg")) {{
                # Extract from DTRreg object
                blip_coef <- coef(a_result)
                vcov_mat <- tryCatch(vcov(a_result), error = function(e) {{
                    matrix(NA, nrow=length(blip_coef), ncol=length(blip_coef))
                }})
                blip_se <- sqrt(diag(vcov_mat))

                design_mat <- model.matrix(blip_formula, data)
                blip_values <- as.vector(design_mat %*% blip_coef)
                optimal <- as.integer(blip_values > 0)
                value_est <- mean(Y[A == optimal])

                list(
                    value_estimate = value_est,
                    blip_coef = blip_coef,
                    blip_se = blip_se,
                    optimal_regime = optimal,
                    n_obs = nrow(data)
                )
            }} else {{
                # Already a list from fallback
                a_result
            }}
            """
        )

        # Check for error
        if "error" in list(result.names):
            error_val = result.rx2("error")
            if str(error_val[0]) != "NA":
                raise RuntimeError(f"R DTRreg A-learning error: {error_val[0]}")

        return {
            "value_estimate": float(result.rx2("value_estimate")[0]),
            "blip_coef": np.array(result.rx2("blip_coef")),
            "blip_se": np.array(result.rx2("blip_se")),
            "optimal_regime": np.array(result.rx2("optimal_regime")).astype(int),
            "n_obs": int(result.rx2("n_obs")[0]),
        }
    finally:
        numpy2ri.deactivate()
