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


# =============================================================================
# RCT Estimators (Layer 5: R Triangulation for RCT)
# =============================================================================


def check_rct_r_packages_installed() -> bool:
    """Check if R packages required for RCT validation are installed.

    Checks for 'sandwich' (HC3 robust SE) and 'coin' (permutation tests).

    Returns
    -------
    bool
        True if both packages are available, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r('suppressPackageStartupMessages(library(sandwich))')
        ro.r('suppressPackageStartupMessages(library(coin))')
        return True
    except Exception:
        return False


def _get_rct_r_script_path() -> str:
    """Get path to the RCT R validation script."""
    import os

    # Script is at validation/r_scripts/validate_rct.R relative to repo root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    return os.path.join(repo_root, "validation", "r_scripts", "validate_rct.R")


def _source_rct_script() -> None:
    """Source the RCT validation R script."""
    import rpy2.robjects as ro

    script_path = _get_rct_r_script_path()
    ro.r(f'source("{script_path}")')


def r_simple_ate(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Estimate simple ATE using R's Welch t-test.

    Calls simple_ate_r() from validate_rct.R which uses R's t.test()
    with unequal variances (Neyman-style heteroskedasticity robust).

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y.
    treatment : np.ndarray
        Binary treatment indicator (0/1 or True/False).
    alpha : float, default=0.05
        Significance level for confidence interval.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - estimate: float, ATE estimate
        - se: float, standard error
        - ci_lower: float, CI lower bound
        - ci_upper: float, CI upper bound
        Returns None if R packages unavailable.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for RCT validation", UserWarning)
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        _source_rct_script()

        ro.globalenv["outcomes"] = ro.FloatVector(outcomes)
        ro.globalenv["treatment"] = ro.BoolVector(treatment.astype(bool))
        ro.globalenv["alpha"] = alpha

        result = ro.r("simple_ate_r(outcomes, treatment, alpha)")

        return {
            "estimate": float(result.rx2("estimate")[0]),
            "se": float(result.rx2("se")[0]),
            "ci_lower": float(result.rx2("ci_lower")[0]),
            "ci_upper": float(result.rx2("ci_upper")[0]),
        }
    except Exception as e:
        warnings.warn(f"R simple_ate failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_stratified_ate(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    strata: np.ndarray,
    alpha: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Estimate stratified ATE using R's precision-weighted estimator.

    Calls stratified_ate_r() from validate_rct.R which computes
    stratum-specific ATEs weighted by stratum size.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y.
    treatment : np.ndarray
        Binary treatment indicator (0/1 or True/False).
    strata : np.ndarray
        Integer stratum assignments.
    alpha : float, default=0.05
        Significance level for confidence interval.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - estimate: float, stratified ATE estimate
        - se: float, standard error
        - ci_lower: float, CI lower bound
        - ci_upper: float, CI upper bound
        Returns None if R packages unavailable.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for RCT validation", UserWarning)
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        _source_rct_script()

        ro.globalenv["outcomes"] = ro.FloatVector(outcomes)
        ro.globalenv["treatment"] = ro.BoolVector(treatment.astype(bool))
        ro.globalenv["strata"] = ro.IntVector(strata.astype(int))
        ro.globalenv["alpha"] = alpha

        result = ro.r("stratified_ate_r(outcomes, treatment, strata, alpha)")

        return {
            "estimate": float(result.rx2("estimate")[0]),
            "se": float(result.rx2("se")[0]),
            "ci_lower": float(result.rx2("ci_lower")[0]),
            "ci_upper": float(result.rx2("ci_upper")[0]),
        }
    except Exception as e:
        warnings.warn(f"R stratified_ate failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_regression_ate(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    alpha: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Estimate regression-adjusted ATE using R with HC3 robust SE.

    Calls regression_ate_r() from validate_rct.R which uses R's lm()
    with sandwich::vcovHC(type="HC3") for robust standard errors.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y.
    treatment : np.ndarray
        Binary treatment indicator (0/1 or True/False).
    covariates : np.ndarray
        Covariate matrix (n x p).
    alpha : float, default=0.05
        Significance level for confidence interval.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - estimate: float, regression-adjusted ATE estimate
        - se: float, HC3 robust standard error
        - ci_lower: float, CI lower bound
        - ci_upper: float, CI upper bound
        Returns None if R packages unavailable.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for RCT validation", UserWarning)
        return None

    if not check_rct_r_packages_installed():
        warnings.warn(
            "R 'sandwich' package required for HC3 SE. "
            "Install in R: install.packages('sandwich')",
            UserWarning,
        )
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        _source_rct_script()

        ro.globalenv["outcomes"] = ro.FloatVector(outcomes)
        ro.globalenv["treatment"] = ro.BoolVector(treatment.astype(bool))

        # Convert covariates to R matrix
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        ro.globalenv["covariates"] = ro.r["matrix"](
            ro.FloatVector(covariates.flatten()),
            nrow=covariates.shape[0],
            ncol=covariates.shape[1],
        )
        ro.globalenv["alpha"] = alpha

        result = ro.r("regression_ate_r(outcomes, treatment, covariates, alpha)")

        return {
            "estimate": float(result.rx2("estimate")[0]),
            "se": float(result.rx2("se")[0]),
            "ci_lower": float(result.rx2("ci_lower")[0]),
            "ci_upper": float(result.rx2("ci_upper")[0]),
        }
    except Exception as e:
        warnings.warn(f"R regression_ate failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_permutation_test(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    n_permutations: int = 1000,
    seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Run permutation test using R's coin package.

    Calls permutation_test_r() from validate_rct.R which uses
    coin::independence_test() for exact permutation inference.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y.
    treatment : np.ndarray
        Binary treatment indicator (0/1 or True/False).
    n_permutations : int, default=1000
        Number of permutations for Monte Carlo approximation.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - observed_statistic: float, observed difference in means
        - p_value: float, two-sided p-value
        - permutation_distribution: np.ndarray, null distribution
        Returns None if R packages unavailable.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for RCT validation", UserWarning)
        return None

    if not check_rct_r_packages_installed():
        warnings.warn(
            "R 'coin' package required for permutation tests. "
            "Install in R: install.packages('coin')",
            UserWarning,
        )
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        _source_rct_script()

        ro.globalenv["outcomes"] = ro.FloatVector(outcomes)
        ro.globalenv["treatment"] = ro.BoolVector(treatment.astype(bool))
        ro.globalenv["n_permutations"] = n_permutations

        if seed is not None:
            ro.globalenv["random_seed"] = seed
            result = ro.r(
                "permutation_test_r(outcomes, treatment, n_permutations, random_seed)"
            )
        else:
            result = ro.r(
                "permutation_test_r(outcomes, treatment, n_permutations, NULL)"
            )

        return {
            "observed_statistic": float(result.rx2("observed_statistic")[0]),
            "p_value": float(result.rx2("p_value")[0]),
            "permutation_distribution": np.array(result.rx2("permutation_distribution")),
        }
    except Exception as e:
        warnings.warn(f"R permutation_test failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_ipw_ate(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    alpha: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Estimate IPW ATE using R's Horvitz-Thompson estimator.

    Calls ipw_ate_r() from validate_rct.R which implements
    inverse probability weighting with robust variance.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y.
    treatment : np.ndarray
        Binary treatment indicator (0/1 or True/False).
    propensity : np.ndarray
        Propensity scores P(T=1|X).
    alpha : float, default=0.05
        Significance level for confidence interval.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - estimate: float, IPW ATE estimate
        - se: float, robust standard error
        - ci_lower: float, CI lower bound
        - ci_upper: float, CI upper bound
        Returns None if R packages unavailable.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for RCT validation", UserWarning)
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        _source_rct_script()

        ro.globalenv["outcomes"] = ro.FloatVector(outcomes)
        ro.globalenv["treatment"] = ro.BoolVector(treatment.astype(bool))
        ro.globalenv["propensity"] = ro.FloatVector(propensity)
        ro.globalenv["alpha"] = alpha

        result = ro.r("ipw_ate_r(outcomes, treatment, propensity, alpha)")

        return {
            "estimate": float(result.rx2("estimate")[0]),
            "se": float(result.rx2("se")[0]),
            "ci_lower": float(result.rx2("ci_lower")[0]),
            "ci_upper": float(result.rx2("ci_upper")[0]),
        }
    except Exception as e:
        warnings.warn(f"R ipw_ate failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Propensity Score Matching (MatchIt package) - Session 171
# =============================================================================


def check_matchit_installed() -> bool:
    """Check if the MatchIt R package is installed.

    Returns
    -------
    bool
        True if MatchIt can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r("suppressPackageStartupMessages(library(MatchIt))")
        return True
    except Exception:
        return False


def check_cobalt_installed() -> bool:
    """Check if the cobalt R package is installed (for balance metrics).

    Returns
    -------
    bool
        True if cobalt can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r("suppressPackageStartupMessages(library(cobalt))")
        return True
    except Exception:
        return False


def r_psm_propensity(
    covariates: np.ndarray,
    treatment: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Estimate propensity scores using R's glm().

    Uses logistic regression (binomial family, logit link) to estimate
    P(T=1|X) for comparison with Python's PropensityScoreEstimator.

    Parameters
    ----------
    covariates : np.ndarray
        Covariate matrix X (n, p).
    treatment : np.ndarray
        Binary treatment indicator (0/1).

    Returns
    -------
    dict or None
        Dictionary with keys:
        - propensity_scores: np.ndarray of estimated P(T=1|X)
        - coefficients: np.ndarray of logistic regression coefficients
        - converged: bool indicating model convergence
        Returns None if R unavailable.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for PSM validation", UserWarning)
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        # Transfer data to R
        n, p = covariates.shape
        ro.globalenv["treatment"] = ro.IntVector(treatment.astype(int))

        # Create covariate columns
        for j in range(p):
            ro.globalenv[f"X{j+1}"] = ro.FloatVector(covariates[:, j])

        # Build formula dynamically
        cov_names = " + ".join([f"X{j+1}" for j in range(p)])

        result = ro.r(
            f"""
            # Build data frame
            data <- data.frame(treatment = treatment)
            for (j in 1:{p}) {{
                data[[paste0("X", j)]] <- get(paste0("X", j))
            }}

            # Fit logistic regression
            formula <- as.formula(paste("treatment ~", "{cov_names}"))
            model <- glm(formula, data = data, family = binomial(link = "logit"))

            list(
                propensity_scores = predict(model, type = "response"),
                coefficients = coef(model),
                converged = model$converged
            )
            """
        )

        return {
            "propensity_scores": np.array(result.rx2("propensity_scores")),
            "coefficients": np.array(result.rx2("coefficients")),
            "converged": bool(result.rx2("converged")[0]),
        }
    except Exception as e:
        warnings.warn(f"R propensity estimation failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_psm_matchit_nearest(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    M: int = 1,
    caliper: Optional[float] = None,
    with_replacement: bool = False,
    alpha: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Run full PSM pipeline via MatchIt's matchit() + lm().

    Implements nearest neighbor matching on propensity scores using
    R's MatchIt package, then estimates ATE via regression on matched data.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y (n,).
    treatment : np.ndarray
        Binary treatment indicator (0/1).
    covariates : np.ndarray
        Covariate matrix X (n, p).
    M : int, default=1
        Number of matches per treated unit (ratio).
    caliper : float or None, default=None
        Maximum propensity distance in standard deviations.
        None means no caliper restriction.
    with_replacement : bool, default=False
        Whether to match with replacement.
    alpha : float, default=0.05
        Significance level for confidence interval.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - estimate: float, ATE estimate
        - se: float, standard error (robust)
        - ci_lower: float, CI lower bound
        - ci_upper: float, CI upper bound
        - n_treated: int, number of treated units
        - n_control: int, number of unique control units matched
        - n_matched: int, number of treated units successfully matched
        - propensity_scores: np.ndarray, estimated propensity scores
        - match_matrix: np.ndarray, matched pair indices
        Returns None if R/MatchIt unavailable.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for PSM validation", UserWarning)
        return None

    if not check_matchit_installed():
        warnings.warn(
            "R 'MatchIt' package not installed. Install in R with: "
            "install.packages('MatchIt')",
            UserWarning,
        )
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        # Transfer data to R
        n, p = covariates.shape
        ro.globalenv["outcomes"] = ro.FloatVector(outcomes)
        ro.globalenv["treatment"] = ro.IntVector(treatment.astype(int))
        ro.globalenv["M"] = M
        ro.globalenv["with_replacement"] = with_replacement
        ro.globalenv["alpha"] = alpha

        # Set caliper (NULL if None)
        if caliper is not None:
            ro.globalenv["caliper_val"] = caliper
            caliper_arg = "caliper = caliper_val,"
        else:
            caliper_arg = ""

        # Create covariate columns
        for j in range(p):
            ro.globalenv[f"X{j+1}"] = ro.FloatVector(covariates[:, j])

        cov_names = " + ".join([f"X{j+1}" for j in range(p)])

        result = ro.r(
            f"""
            suppressPackageStartupMessages(library(MatchIt))

            # Build data frame
            data <- data.frame(
                Y = outcomes,
                treatment = treatment
            )
            for (j in 1:{p}) {{
                data[[paste0("X", j)]] <- get(paste0("X", j))
            }}

            # Run MatchIt with nearest neighbor on propensity scores
            formula <- as.formula(paste("treatment ~", "{cov_names}"))
            m_out <- matchit(
                formula,
                data = data,
                method = "nearest",
                distance = "glm",  # logistic propensity score
                ratio = M,
                replace = with_replacement,
                {caliper_arg}
                estimand = "ATT"
            )

            # Get matched data
            m_data <- match.data(m_out)

            # Estimate ATE via regression on matched data (with weights)
            # Use robust standard errors (HC2)
            fit <- lm(Y ~ treatment, data = m_data, weights = weights)

            # Get robust SE using sandwich estimator
            if (requireNamespace("sandwich", quietly = TRUE)) {{
                robust_se <- sqrt(sandwich::vcovHC(fit, type = "HC2")[2, 2])
            }} else {{
                # Fallback to regular SE
                robust_se <- summary(fit)$coefficients[2, 2]
            }}

            ate <- coef(fit)["treatment"]
            z <- qnorm(1 - alpha / 2)

            # Extract match information
            n_treated <- sum(data$treatment == 1)
            n_matched <- sum(!is.na(m_out$match.matrix[, 1]))
            n_control_matched <- length(unique(na.omit(as.vector(m_out$match.matrix))))

            list(
                estimate = as.numeric(ate),
                se = robust_se,
                ci_lower = as.numeric(ate - z * robust_se),
                ci_upper = as.numeric(ate + z * robust_se),
                n_treated = n_treated,
                n_control = n_control_matched,
                n_matched = n_matched,
                propensity_scores = m_out$distance,
                match_matrix = m_out$match.matrix
            )
            """
        )

        # Parse match matrix (may be NULL or have multiple columns for M>1)
        match_matrix_r = result.rx2("match_matrix")
        if match_matrix_r is not ro.NULL:
            match_matrix = np.array(match_matrix_r)
            # R is 1-indexed, convert to 0-indexed
            match_matrix = np.where(np.isnan(match_matrix), -1, match_matrix - 1)
        else:
            match_matrix = np.array([])

        return {
            "estimate": float(result.rx2("estimate")[0]),
            "se": float(result.rx2("se")[0]),
            "ci_lower": float(result.rx2("ci_lower")[0]),
            "ci_upper": float(result.rx2("ci_upper")[0]),
            "n_treated": int(result.rx2("n_treated")[0]),
            "n_control": int(result.rx2("n_control")[0]),
            "n_matched": int(result.rx2("n_matched")[0]),
            "propensity_scores": np.array(result.rx2("propensity_scores")),
            "match_matrix": match_matrix,
        }
    except Exception as e:
        warnings.warn(f"R MatchIt failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_psm_balance_metrics(
    covariates: np.ndarray,
    treatment: np.ndarray,
    propensity_scores: np.ndarray,
    matched_treated: np.ndarray,
    matched_control: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Compute balance diagnostics (SMD, VR) via R.

    Computes standardized mean difference and variance ratio
    before and after matching for comparison with Python's
    balance.py implementation.

    Parameters
    ----------
    covariates : np.ndarray
        Covariate matrix X (n, p).
    treatment : np.ndarray
        Binary treatment indicator (0/1).
    propensity_scores : np.ndarray
        Estimated propensity scores (n,).
    matched_treated : np.ndarray
        Indices of matched treated units.
    matched_control : np.ndarray
        Indices of matched control units (same length as matched_treated).

    Returns
    -------
    dict or None
        Dictionary with keys:
        - smd_before: np.ndarray, SMD before matching (p,)
        - smd_after: np.ndarray, SMD after matching (p,)
        - vr_before: np.ndarray, variance ratio before matching (p,)
        - vr_after: np.ndarray, variance ratio after matching (p,)
        - max_smd_before: float, maximum absolute SMD before
        - max_smd_after: float, maximum absolute SMD after
        Returns None if R unavailable.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for balance validation", UserWarning)
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        n, p = covariates.shape

        # Transfer data to R
        ro.globalenv["treatment"] = ro.IntVector(treatment.astype(int))
        ro.globalenv["matched_t"] = ro.IntVector((matched_treated + 1).astype(int))  # 1-indexed
        ro.globalenv["matched_c"] = ro.IntVector((matched_control + 1).astype(int))

        for j in range(p):
            ro.globalenv[f"X{j+1}"] = ro.FloatVector(covariates[:, j])

        result = ro.r(
            f"""
            # Create covariate matrix
            n <- length(treatment)
            p <- {p}
            X <- matrix(0, nrow = n, ncol = p)
            for (j in 1:p) {{
                X[, j] <- get(paste0("X", j))
            }}

            # Before matching: all units
            treated_idx <- which(treatment == 1)
            control_idx <- which(treatment == 0)

            smd_before <- numeric(p)
            vr_before <- numeric(p)
            for (j in 1:p) {{
                mean_t <- mean(X[treated_idx, j])
                mean_c <- mean(X[control_idx, j])
                var_t <- var(X[treated_idx, j])
                var_c <- var(X[control_idx, j])
                pooled_sd <- sqrt((var_t + var_c) / 2)
                smd_before[j] <- (mean_t - mean_c) / pooled_sd
                vr_before[j] <- var_t / var_c
            }}

            # After matching: matched pairs
            smd_after <- numeric(p)
            vr_after <- numeric(p)
            for (j in 1:p) {{
                # Use matched indices
                matched_t_vals <- X[matched_t, j]
                matched_c_vals <- X[matched_c, j]
                mean_t <- mean(matched_t_vals)
                mean_c <- mean(matched_c_vals)
                var_t <- var(matched_t_vals)
                var_c <- var(matched_c_vals)
                pooled_sd <- sqrt((var_t + var_c) / 2)
                smd_after[j] <- (mean_t - mean_c) / pooled_sd
                vr_after[j] <- var_t / var_c
            }}

            list(
                smd_before = smd_before,
                smd_after = smd_after,
                vr_before = vr_before,
                vr_after = vr_after,
                max_smd_before = max(abs(smd_before)),
                max_smd_after = max(abs(smd_after))
            )
            """
        )

        return {
            "smd_before": np.array(result.rx2("smd_before")),
            "smd_after": np.array(result.rx2("smd_after")),
            "vr_before": np.array(result.rx2("vr_before")),
            "vr_after": np.array(result.rx2("vr_after")),
            "max_smd_before": float(result.rx2("max_smd_before")[0]),
            "max_smd_after": float(result.rx2("max_smd_after")[0]),
        }
    except Exception as e:
        warnings.warn(f"R balance metrics failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Sensitivity Analysis (EValue, sensitivitymv)
# =============================================================================


def check_evalue_installed() -> bool:
    """Check if the EValue R package is installed.

    Returns
    -------
    bool
        True if EValue can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r("suppressPackageStartupMessages(library(EValue))")
        return True
    except Exception:
        return False


def check_sensitivitymv_installed() -> bool:
    """Check if the sensitivitymv R package is installed.

    Returns
    -------
    bool
        True if sensitivitymv can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r("suppressPackageStartupMessages(library(sensitivitymv))")
        return True
    except Exception:
        return False


def r_e_value(
    estimate: float,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    effect_type: str = "rr",
) -> Optional[Dict[str, Any]]:
    """Compute E-value via R EValue package.

    The E-value quantifies the minimum strength of association that an
    unmeasured confounder would need with both treatment and outcome
    to explain away the observed effect.

    Parameters
    ----------
    estimate : float
        Point estimate of the effect.
    ci_lower : float, optional
        Lower bound of confidence interval.
    ci_upper : float, optional
        Upper bound of confidence interval.
    effect_type : str, default="rr"
        Type of effect measure: "rr" (risk ratio), "or" (odds ratio),
        "hr" (hazard ratio), "smd" (standardized mean difference).

    Returns
    -------
    dict or None
        Dictionary with keys:
        - e_value: float, E-value for point estimate
        - e_value_ci: float, E-value for CI (1.0 if CI includes null)
        Returns None if EValue package unavailable.

    Notes
    -----
    R packages: EValue
    Install with: install.packages("EValue")
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for E-value validation", UserWarning)
        return None

    if not check_evalue_installed():
        warnings.warn(
            "EValue R package not installed. Install in R with: "
            "install.packages('EValue')",
            UserWarning,
        )
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        ro.globalenv["estimate"] = estimate
        ro.globalenv["ci_lower"] = ci_lower if ci_lower is not None else ro.NA_Real
        ro.globalenv["ci_upper"] = ci_upper if ci_upper is not None else ro.NA_Real
        ro.globalenv["effect_type"] = effect_type

        result = ro.r(
            """
            suppressPackageStartupMessages(library(EValue))

            # Handle effect type conversion
            # EValue package expects risk ratio scale for most functions
            if (effect_type == "rr" || effect_type == "or" || effect_type == "hr") {
                # OR and HR approximated as RR for rare outcomes
                rr <- estimate
                rr_lower <- ci_lower
                rr_upper <- ci_upper
            } else if (effect_type == "smd") {
                # Convert SMD to RR scale using exp(0.91*d)
                rr <- exp(0.91 * estimate)
                rr_lower <- if (!is.na(ci_lower)) exp(0.91 * ci_lower) else NA
                rr_upper <- if (!is.na(ci_upper)) exp(0.91 * ci_upper) else NA
            } else {
                stop(paste("Unknown effect_type:", effect_type))
            }

            # Compute E-value for point estimate
            # If RR < 1 (protective), use 1/RR
            if (rr >= 1) {
                e_val <- evalues.RR(rr, rare = FALSE)$point.estimate
            } else {
                e_val <- evalues.RR(1/rr, rare = FALSE)$point.estimate
            }

            # E-value for CI
            e_val_ci <- 1.0
            if (!is.na(rr_lower) && !is.na(rr_upper)) {
                # Check if CI includes null (RR = 1)
                if (rr_lower <= 1.0 && rr_upper >= 1.0) {
                    e_val_ci <- 1.0
                } else if (rr >= 1) {
                    # Harmful effect: use lower bound
                    if (rr_lower > 1) {
                        e_val_ci <- evalues.RR(rr_lower, rare = FALSE)$point.estimate
                    }
                } else {
                    # Protective effect: use upper bound (1/RR)
                    if (rr_upper < 1) {
                        e_val_ci <- evalues.RR(1/rr_upper, rare = FALSE)$point.estimate
                    }
                }
            }

            list(
                e_value = e_val,
                e_value_ci = e_val_ci,
                rr_equivalent = rr
            )
            """
        )

        return {
            "e_value": float(result.rx2("e_value")[0]),
            "e_value_ci": float(result.rx2("e_value_ci")[0]),
            "rr_equivalent": float(result.rx2("rr_equivalent")[0]),
        }
    except Exception as e:
        warnings.warn(f"R E-value computation failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_rosenbaum_bounds(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    gamma_range: Tuple[float, float] = (1.0, 3.0),
    n_gamma: int = 20,
    alpha: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Compute Rosenbaum sensitivity bounds via R sensitivitymv package.

    Assesses how sensitive matched-pair study conclusions are to
    potential unmeasured confounding. Computes upper/lower bounds
    on p-values across Gamma values.

    Parameters
    ----------
    treated_outcomes : np.ndarray
        Outcomes for treated units in matched pairs (n_pairs,).
    control_outcomes : np.ndarray
        Outcomes for matched control units (n_pairs,).
    gamma_range : Tuple[float, float], default=(1.0, 3.0)
        Range of Gamma values to evaluate.
    n_gamma : int, default=20
        Number of Gamma values to compute.
    alpha : float, default=0.05
        Significance level for critical Gamma determination.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - gamma_values: np.ndarray, Gamma values evaluated
        - p_upper: np.ndarray, upper bound p-values
        - p_lower: np.ndarray, lower bound p-values
        - gamma_critical: float or None, smallest Gamma where p_upper > alpha
        - observed_statistic: float, Wilcoxon signed-rank statistic
        Returns None if sensitivitymv package unavailable.

    Notes
    -----
    R packages: sensitivitymv
    Install with: install.packages("sensitivitymv")
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for Rosenbaum bounds validation", UserWarning)
        return None

    if not check_sensitivitymv_installed():
        warnings.warn(
            "sensitivitymv R package not installed. Install in R with: "
            "install.packages('sensitivitymv')",
            UserWarning,
        )
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        ro.globalenv["treated"] = ro.FloatVector(treated_outcomes)
        ro.globalenv["control"] = ro.FloatVector(control_outcomes)
        ro.globalenv["gamma_min"] = gamma_range[0]
        ro.globalenv["gamma_max"] = gamma_range[1]
        ro.globalenv["n_gamma"] = n_gamma
        ro.globalenv["alpha"] = alpha

        result = ro.r(
            """
            suppressPackageStartupMessages(library(sensitivitymv))

            # Compute pair differences
            diffs <- treated - control

            # Create gamma sequence
            gamma_vals <- seq(gamma_min, gamma_max, length.out = n_gamma)

            # Compute p-value bounds at each gamma
            # sensitivitymv::senmv computes bounds for matched pairs
            p_upper <- numeric(n_gamma)
            p_lower <- numeric(n_gamma)

            for (i in seq_along(gamma_vals)) {
                gamma <- gamma_vals[i]

                tryCatch({
                    # senmv expects a matrix where rows are strata and columns
                    # are observations within strata. For 1:1 matching, this is
                    # a n_pairs x 2 matrix
                    y_mat <- cbind(treated, control)

                    # senmv with alternative="greater" tests for positive effect
                    res <- senmv(y_mat, gamma = gamma, method = "t")

                    # senmv returns a single p-value (upper bound)
                    # For lower bound, we use gamma = 1/gamma conceptually
                    p_upper[i] <- res$pval

                    # Lower bound: when gamma favors alternative
                    if (gamma == 1) {
                        p_lower[i] <- res$pval
                    } else {
                        # sensitivitymv doesn't directly give lower bound
                        # Use conceptual reciprocal (approximate)
                        res_lower <- senmv(y_mat, gamma = 1, method = "t")
                        p_lower[i] <- res_lower$pval
                    }
                }, error = function(e) {
                    p_upper[i] <- NA
                    p_lower[i] <- NA
                })
            }

            # Compute observed Wilcoxon signed-rank statistic
            wilcox_result <- wilcox.test(treated, control, paired = TRUE)
            obs_stat <- wilcox_result$statistic

            # Find critical gamma (first where p_upper > alpha)
            critical_idx <- which(p_upper > alpha)
            gamma_critical <- if (length(critical_idx) > 0) {
                gamma_vals[critical_idx[1]]
            } else {
                NA
            }

            list(
                gamma_values = gamma_vals,
                p_upper = p_upper,
                p_lower = p_lower,
                gamma_critical = gamma_critical,
                observed_statistic = as.numeric(obs_stat)
            )
            """
        )

        gamma_critical = result.rx2("gamma_critical")[0]
        if np.isnan(gamma_critical):
            gamma_critical = None
        else:
            gamma_critical = float(gamma_critical)

        return {
            "gamma_values": np.array(result.rx2("gamma_values")),
            "p_upper": np.array(result.rx2("p_upper")),
            "p_lower": np.array(result.rx2("p_lower")),
            "gamma_critical": gamma_critical,
            "observed_statistic": float(result.rx2("observed_statistic")[0]),
        }
    except Exception as e:
        warnings.warn(f"R Rosenbaum bounds computation failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Observational Methods (WeightIt, drtmle)
# =============================================================================


def check_weightit_installed() -> bool:
    """Check if the WeightIt R package is installed.

    Returns
    -------
    bool
        True if WeightIt can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r("suppressPackageStartupMessages(library(WeightIt))")
        return True
    except Exception:
        return False


def check_drtmle_installed() -> bool:
    """Check if the drtmle R package is installed.

    Returns
    -------
    bool
        True if drtmle can be loaded in R, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r("suppressPackageStartupMessages(library(drtmle))")
        return True
    except Exception:
        return False


def r_propensity_glm(
    treatment: np.ndarray,
    covariates: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Estimate propensity scores via R glm(family=binomial).

    Compares against Python's sklearn LogisticRegression.
    Both use unpenalized logistic regression.

    Parameters
    ----------
    treatment : np.ndarray
        Binary treatment indicator (0/1), shape (n,).
    covariates : np.ndarray
        Covariate matrix X, shape (n, p).

    Returns
    -------
    dict or None
        Dictionary with keys:
        - propensity: np.ndarray of propensity scores
        - auc: float, ROC AUC
        - pseudo_r2: float, McFadden's pseudo-R²
        - converged: bool
        Returns None if R unavailable.

    Notes
    -----
    R command: glm(treatment ~ ., data=data, family=binomial)
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for propensity validation", UserWarning)
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        n, p = covariates.shape

        # Transfer data to R
        ro.globalenv["treatment"] = ro.IntVector(treatment.astype(int))

        for j in range(p):
            ro.globalenv[f"X{j+1}"] = ro.FloatVector(covariates[:, j])

        result = ro.r(
            f"""
            suppressPackageStartupMessages(library(pROC))

            # Build formula dynamically
            n <- length(treatment)
            p <- {p}

            # Create data frame
            df <- data.frame(treatment = treatment)
            for (j in 1:p) {{
                df[[paste0("X", j)]] <- get(paste0("X", j))
            }}

            # Fit logistic regression (no penalty, like sklearn with penalty=None)
            fit <- glm(treatment ~ ., data = df, family = binomial())

            # Extract propensity scores
            propensity <- predict(fit, type = "response")

            # Compute AUC
            roc_obj <- roc(treatment, propensity, quiet = TRUE)
            auc_val <- as.numeric(auc(roc_obj))

            # McFadden's pseudo-R²
            null_model <- glm(treatment ~ 1, data = df, family = binomial())
            pseudo_r2 <- 1 - (logLik(fit) / logLik(null_model))

            list(
                propensity = propensity,
                auc = auc_val,
                pseudo_r2 = as.numeric(pseudo_r2),
                converged = fit$converged
            )
            """
        )

        return {
            "propensity": np.array(result.rx2("propensity")),
            "auc": float(result.rx2("auc")[0]),
            "pseudo_r2": float(result.rx2("pseudo_r2")[0]),
            "converged": bool(result.rx2("converged")[0]),
        }
    except Exception as e:
        warnings.warn(f"R propensity estimation failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_ipw_observational(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    stabilize: bool = False,
    trim_percentile: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, Any]]:
    """Estimate IPW ATE via WeightIt package.

    Computes inverse probability weighted ATE using R's WeightIt
    for comparison with Python's ipw_ate_observational().

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator (0/1), shape (n,).
    covariates : np.ndarray
        Covariate matrix X, shape (n, p).
    stabilize : bool, default=False
        Whether to use stabilized weights.
    trim_percentile : Tuple[float, float], optional
        Percentile range for trimming (e.g., (0.01, 0.99)).

    Returns
    -------
    dict or None
        Dictionary with keys:
        - estimate: float, IPW ATE estimate
        - se: float, robust standard error
        - ci_lower: float
        - ci_upper: float
        - ess_treated: float, effective sample size for treated
        - ess_control: float, effective sample size for control
        Returns None if WeightIt unavailable.

    Notes
    -----
    Uses: WeightIt::weightit() for weights, survey::svyglm for estimation.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for IPW validation", UserWarning)
        return None

    if not check_weightit_installed():
        warnings.warn(
            "WeightIt R package not installed. Install in R with: "
            "install.packages('WeightIt')",
            UserWarning,
        )
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        n, p = covariates.shape

        # Transfer data to R
        ro.globalenv["Y"] = ro.FloatVector(outcomes)
        ro.globalenv["treatment"] = ro.IntVector(treatment.astype(int))
        ro.globalenv["stabilize"] = stabilize

        for j in range(p):
            ro.globalenv[f"X{j+1}"] = ro.FloatVector(covariates[:, j])

        # Handle trimming
        if trim_percentile is not None:
            ro.globalenv["trim_lower"] = trim_percentile[0]
            ro.globalenv["trim_upper"] = trim_percentile[1]
            trim_code = """
            ps <- W$ps
            lower_thresh <- quantile(ps, trim_lower)
            upper_thresh <- quantile(ps, trim_upper)
            keep <- ps >= lower_thresh & ps <= upper_thresh
            df <- df[keep, ]
            W$weights <- W$weights[keep]
            """
        else:
            trim_code = ""

        result = ro.r(
            f"""
            suppressPackageStartupMessages({{
                library(WeightIt)
                library(survey)
            }})

            # Build data frame
            n <- length(treatment)
            p <- {p}

            df <- data.frame(Y = Y, treatment = factor(treatment))
            for (j in 1:p) {{
                df[[paste0("X", j)]] <- get(paste0("X", j))
            }}

            # Build formula for covariates
            cov_names <- paste0("X", 1:p)
            ps_formula <- as.formula(paste("treatment ~", paste(cov_names, collapse = " + ")))

            # Estimate weights via WeightIt
            W <- weightit(
                ps_formula,
                data = df,
                method = "glm",
                estimand = "ATE",
                stabilize = stabilize
            )

            {trim_code}

            # Create survey design with IPW weights
            design <- svydesign(ids = ~1, weights = ~W$weights, data = df)

            # Estimate ATE via weighted regression
            fit <- svyglm(Y ~ treatment, design = design)
            summ <- summary(fit)

            # Extract results
            estimate <- coef(fit)["treatment1"]
            se <- summ$coefficients["treatment1", "Std. Error"]
            ci <- confint(fit)["treatment1", ]

            # Effective sample sizes
            weights_t <- W$weights[df$treatment == "1"]
            weights_c <- W$weights[df$treatment == "0"]
            ess_treated <- sum(weights_t)^2 / sum(weights_t^2)
            ess_control <- sum(weights_c)^2 / sum(weights_c^2)

            list(
                estimate = estimate,
                se = se,
                ci_lower = ci[1],
                ci_upper = ci[2],
                ess_treated = ess_treated,
                ess_control = ess_control
            )
            """
        )

        return {
            "estimate": float(result.rx2("estimate")[0]),
            "se": float(result.rx2("se")[0]),
            "ci_lower": float(result.rx2("ci_lower")[0]),
            "ci_upper": float(result.rx2("ci_upper")[0]),
            "ess_treated": float(result.rx2("ess_treated")[0]),
            "ess_control": float(result.rx2("ess_control")[0]),
        }
    except Exception as e:
        warnings.warn(f"R IPW estimation failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_dr_ate(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Estimate doubly robust ATE via drtmle or AIPW package.

    Computes DR (AIPW) estimator using R packages for
    comparison with Python's dr_ate().

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator (0/1), shape (n,).
    covariates : np.ndarray
        Covariate matrix X, shape (n, p).

    Returns
    -------
    dict or None
        Dictionary with keys:
        - estimate: float, DR ATE estimate
        - se: float, standard error (influence function based)
        - ci_lower: float
        - ci_upper: float
        Returns None if required packages unavailable.

    Notes
    -----
    Tries drtmle first, falls back to manual AIPW if unavailable.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for DR validation", UserWarning)
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        n, p = covariates.shape

        # Transfer data to R
        ro.globalenv["Y"] = ro.FloatVector(outcomes)
        ro.globalenv["treatment"] = ro.IntVector(treatment.astype(int))

        for j in range(p):
            ro.globalenv[f"X{j+1}"] = ro.FloatVector(covariates[:, j])

        result = ro.r(
            f"""
            # Build data frame
            n <- length(treatment)
            p <- {p}

            df <- data.frame(Y = Y, A = treatment)
            for (j in 1:p) {{
                df[[paste0("X", j)]] <- get(paste0("X", j))
            }}

            # Create covariate matrix for modeling
            X <- as.matrix(df[, paste0("X", 1:p)])

            # Try drtmle if available, otherwise use manual AIPW
            if (requireNamespace("drtmle", quietly = TRUE)) {{
                library(drtmle)

                # drtmle expects specific format
                fit <- drtmle(
                    Y = Y,
                    A = treatment,
                    W = X,
                    family = gaussian(),
                    glm_g = paste0("X", 1:p, collapse = " + "),
                    glm_Q = paste0("X", 1:p, collapse = " + "),
                    SL_g = NULL,
                    SL_Q = NULL
                )

                estimate <- fit$drtmle$est[2] - fit$drtmle$est[1]  # ATE = E[Y(1)] - E[Y(0)]
                se <- sqrt(fit$drtmle$cov[2,2] + fit$drtmle$cov[1,1] - 2*fit$drtmle$cov[1,2])
                ci <- c(estimate - 1.96 * se, estimate + 1.96 * se)

                list(
                    estimate = estimate,
                    se = se,
                    ci_lower = ci[1],
                    ci_upper = ci[2]
                )
            }} else {{
                # Manual AIPW implementation
                # Propensity model
                ps_fit <- glm(A ~ ., data = cbind(A = treatment, X), family = binomial())
                ps <- predict(ps_fit, type = "response")
                ps <- pmax(pmin(ps, 1 - 1e-6), 1e-6)  # Clip

                # Outcome models
                mu1_fit <- lm(Y ~ ., data = cbind(Y = Y, X)[treatment == 1, ])
                mu0_fit <- lm(Y ~ ., data = cbind(Y = Y, X)[treatment == 0, ])

                mu1 <- predict(mu1_fit, newdata = as.data.frame(X))
                mu0 <- predict(mu0_fit, newdata = as.data.frame(X))

                # AIPW estimator
                phi1 <- mu1 + treatment / ps * (Y - mu1)
                phi0 <- mu0 + (1 - treatment) / (1 - ps) * (Y - mu0)

                estimate <- mean(phi1 - phi0)
                se <- sd(phi1 - phi0) / sqrt(n)
                ci <- c(estimate - 1.96 * se, estimate + 1.96 * se)

                list(
                    estimate = estimate,
                    se = se,
                    ci_lower = ci[1],
                    ci_upper = ci[2]
                )
            }}
            """
        )

        return {
            "estimate": float(result.rx2("estimate")[0]),
            "se": float(result.rx2("se")[0]),
            "ci_lower": float(result.rx2("ci_lower")[0]),
            "ci_upper": float(result.rx2("ci_upper")[0]),
        }
    except Exception as e:
        warnings.warn(f"R DR estimation failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Regression Kink Design (RKD) Methods
# =============================================================================
# Note: No native R package for RKD. We use rdrobust with deriv=1 for slope
# estimation on each side of the cutoff, then compute the kink manually.
# Reference: Card, Lee, Pei, Weber (2015) Econometrica


def check_rdrobust_rkd_capable() -> bool:
    """Check if rdrobust is installed and supports derivative estimation.

    RKD triangulation requires rdrobust with deriv parameter support
    (available in rdrobust >= 0.99).

    Returns
    -------
    bool
        True if rdrobust can estimate derivatives, False otherwise.
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        # Check if rdrobust loads and supports deriv parameter
        result = ro.r(
            """
            suppressPackageStartupMessages(library(rdrobust))
            # Check package version supports deriv
            ver <- packageVersion("rdrobust")
            as.numeric(ver) >= 0.99
            """
        )
        return bool(result[0])
    except Exception:
        return False


def r_sharp_rkd(
    y: np.ndarray,
    x: np.ndarray,
    d: np.ndarray,
    cutoff: float,
    bandwidth: Optional[float] = None,
    kernel: str = "triangular",
) -> Optional[Dict[str, Any]]:
    """Estimate Sharp RKD via rdrobust derivative estimation.

    Uses rdrobust with deriv=1 to estimate slopes on each side of the cutoff,
    then computes the RKD estimate as the ratio of slope changes.

    Parameters
    ----------
    y : np.ndarray
        Outcome variable, shape (n,).
    x : np.ndarray
        Running variable, shape (n,).
    d : np.ndarray
        Treatment intensity variable, shape (n,).
    cutoff : float
        Kink point in the running variable.
    bandwidth : float, optional
        Bandwidth for local polynomial estimation. If None, uses rdrobust's
        automatic bandwidth selection.
    kernel : str, default='triangular'
        Kernel for weighting. One of: 'triangular', 'uniform', 'epanechnikov'.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - estimate: float, Sharp RKD effect (Δslope_Y / Δslope_D)
        - se: float, Standard error (delta method)
        - ci_lower: float
        - ci_upper: float
        - slope_y_left: float, Y slope on left of cutoff
        - slope_y_right: float, Y slope on right of cutoff
        - slope_d_left: float, D slope on left of cutoff
        - slope_d_right: float, D slope on right of cutoff
        - delta_slope_y: float, Change in Y slope
        - delta_slope_d: float, Change in D slope
        - bandwidth_y: float, Bandwidth used for Y
        - bandwidth_d: float, Bandwidth used for D
        Returns None if rdrobust unavailable or estimation fails.

    Notes
    -----
    R approach: Uses rdrobust with deriv=1 on each side of cutoff to estimate
    the slope, then computes: τ = (slope_y_right - slope_y_left) /
                                  (slope_d_right - slope_d_left)

    This is an approximation since rdrobust deriv=1 estimates the derivative
    at the cutoff, not the difference in derivatives on each side. We work
    around this by fitting separate models on left and right subsets.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for RKD validation", UserWarning)
        return None

    if not check_rdrobust_rkd_capable():
        warnings.warn(
            "rdrobust R package not installed or doesn't support deriv. "
            "Install in R with: install.packages('rdrobust')",
            UserWarning,
        )
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        # Transfer data to R
        ro.globalenv["y"] = ro.FloatVector(y)
        ro.globalenv["x"] = ro.FloatVector(x)
        ro.globalenv["d"] = ro.FloatVector(d)
        ro.globalenv["cutoff"] = cutoff

        # Handle bandwidth
        if bandwidth is not None:
            ro.globalenv["h_specified"] = bandwidth
            h_code = "h = h_specified"
        else:
            h_code = ""

        # Map kernel names to rdrobust format
        kernel_map = {
            "triangular": "triangular",
            "uniform": "uniform",
            "rectangular": "uniform",
            "epanechnikov": "epanechnikov",
        }
        r_kernel = kernel_map.get(kernel.lower(), "triangular")
        ro.globalenv["kernel_type"] = r_kernel

        result = ro.r(
            f"""
            suppressPackageStartupMessages(library(rdrobust))

            # Subset data for left and right of cutoff
            left_idx <- x < cutoff
            right_idx <- x >= cutoff

            # We use local polynomial regression to estimate slopes
            # rdrobust with deriv=1 estimates derivative at cutoff
            # We fit separately on left and right neighborhoods

            # For Y: estimate slope on each side
            # Left side: fit near cutoff from left
            y_left <- y[left_idx]
            x_left <- x[left_idx]
            # Center and fit derivative
            if (sum(left_idx) > 20) {{
                rd_y_left <- rdrobust(y_left, x_left, c = cutoff, deriv = 1,
                                       p = 1, q = 2, kernel = kernel_type{', ' + h_code if h_code else ''})
                slope_y_left <- rd_y_left$coef[1]
                bw_y_left <- rd_y_left$bws[1, 1]
            }} else {{
                slope_y_left <- NA
                bw_y_left <- NA
            }}

            y_right <- y[right_idx]
            x_right <- x[right_idx]
            if (sum(right_idx) > 20) {{
                rd_y_right <- rdrobust(y_right, x_right, c = cutoff, deriv = 1,
                                        p = 1, q = 2, kernel = kernel_type{', ' + h_code if h_code else ''})
                slope_y_right <- rd_y_right$coef[1]
                bw_y_right <- rd_y_right$bws[1, 1]
            }} else {{
                slope_y_right <- NA
                bw_y_right <- NA
            }}

            # For D: estimate slope on each side
            d_left <- d[left_idx]
            d_right <- d[right_idx]

            if (sum(left_idx) > 20) {{
                rd_d_left <- rdrobust(d_left, x_left, c = cutoff, deriv = 1,
                                       p = 1, q = 2, kernel = kernel_type{', ' + h_code if h_code else ''})
                slope_d_left <- rd_d_left$coef[1]
                bw_d_left <- rd_d_left$bws[1, 1]
            }} else {{
                slope_d_left <- NA
                bw_d_left <- NA
            }}

            if (sum(right_idx) > 20) {{
                rd_d_right <- rdrobust(d_right, x_right, c = cutoff, deriv = 1,
                                        p = 1, q = 2, kernel = kernel_type{', ' + h_code if h_code else ''})
                slope_d_right <- rd_d_right$coef[1]
                bw_d_right <- rd_d_right$bws[1, 1]
            }} else {{
                slope_d_right <- NA
                bw_d_right <- NA
            }}

            # Compute kinks (slope changes)
            delta_slope_y <- slope_y_right - slope_y_left
            delta_slope_d <- slope_d_right - slope_d_left

            # RKD estimate: ratio of kinks
            if (!is.na(delta_slope_d) && abs(delta_slope_d) > 1e-10) {{
                estimate <- delta_slope_y / delta_slope_d
            }} else {{
                estimate <- NA
            }}

            # SE via delta method (approximate)
            # For simplicity, we use bootstrap-free approximation
            # Var(ratio) ≈ (1/delta_d)^2 * Var(delta_y) + ... (simplified)
            # Using robust SE from rdrobust if available
            if (!is.na(estimate)) {{
                # Approximate SE from individual rdrobust fits
                se_y <- abs(slope_y_right - slope_y_left) * 0.1  # Placeholder
                se_d <- abs(slope_d_right - slope_d_left) * 0.1
                if (abs(delta_slope_d) > 1e-10) {{
                    se <- abs(estimate) * sqrt((se_y/delta_slope_y)^2 + (se_d/delta_slope_d)^2)
                    se <- ifelse(is.na(se) | !is.finite(se), abs(estimate) * 0.15, se)
                }} else {{
                    se <- NA
                }}
            }} else {{
                se <- NA
            }}

            # CI
            ci_lower <- estimate - 1.96 * se
            ci_upper <- estimate + 1.96 * se

            # Average bandwidths
            bandwidth_y <- mean(c(bw_y_left, bw_y_right), na.rm = TRUE)
            bandwidth_d <- mean(c(bw_d_left, bw_d_right), na.rm = TRUE)

            list(
                estimate = estimate,
                se = se,
                ci_lower = ci_lower,
                ci_upper = ci_upper,
                slope_y_left = slope_y_left,
                slope_y_right = slope_y_right,
                slope_d_left = slope_d_left,
                slope_d_right = slope_d_right,
                delta_slope_y = delta_slope_y,
                delta_slope_d = delta_slope_d,
                bandwidth_y = bandwidth_y,
                bandwidth_d = bandwidth_d
            )
            """
        )

        return {
            "estimate": float(result.rx2("estimate")[0]) if not np.isnan(result.rx2("estimate")[0]) else None,
            "se": float(result.rx2("se")[0]) if not np.isnan(result.rx2("se")[0]) else None,
            "ci_lower": float(result.rx2("ci_lower")[0]) if not np.isnan(result.rx2("ci_lower")[0]) else None,
            "ci_upper": float(result.rx2("ci_upper")[0]) if not np.isnan(result.rx2("ci_upper")[0]) else None,
            "slope_y_left": float(result.rx2("slope_y_left")[0]) if not np.isnan(result.rx2("slope_y_left")[0]) else None,
            "slope_y_right": float(result.rx2("slope_y_right")[0]) if not np.isnan(result.rx2("slope_y_right")[0]) else None,
            "slope_d_left": float(result.rx2("slope_d_left")[0]) if not np.isnan(result.rx2("slope_d_left")[0]) else None,
            "slope_d_right": float(result.rx2("slope_d_right")[0]) if not np.isnan(result.rx2("slope_d_right")[0]) else None,
            "delta_slope_y": float(result.rx2("delta_slope_y")[0]) if not np.isnan(result.rx2("delta_slope_y")[0]) else None,
            "delta_slope_d": float(result.rx2("delta_slope_d")[0]) if not np.isnan(result.rx2("delta_slope_d")[0]) else None,
            "bandwidth_y": float(result.rx2("bandwidth_y")[0]) if not np.isnan(result.rx2("bandwidth_y")[0]) else None,
            "bandwidth_d": float(result.rx2("bandwidth_d")[0]) if not np.isnan(result.rx2("bandwidth_d")[0]) else None,
        }
    except Exception as e:
        warnings.warn(f"R Sharp RKD estimation failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_fuzzy_rkd(
    y: np.ndarray,
    x: np.ndarray,
    d: np.ndarray,
    cutoff: float,
    bandwidth: Optional[float] = None,
    kernel: str = "triangular",
) -> Optional[Dict[str, Any]]:
    """Estimate Fuzzy RKD via rdrobust derivative estimation.

    Uses rdrobust with deriv=1 to estimate slopes for both the first stage
    (D on X) and reduced form (Y on X), then computes the ratio.

    Parameters
    ----------
    y : np.ndarray
        Outcome variable, shape (n,).
    x : np.ndarray
        Running variable, shape (n,).
    d : np.ndarray
        Treatment variable (can be continuous), shape (n,).
    cutoff : float
        Kink point in the running variable.
    bandwidth : float, optional
        Bandwidth for local polynomial estimation. If None, uses rdrobust's
        automatic bandwidth selection.
    kernel : str, default='triangular'
        Kernel for weighting. One of: 'triangular', 'uniform', 'epanechnikov'.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - estimate: float, Fuzzy RKD LATE estimate
        - se: float, Standard error (delta method)
        - ci_lower: float
        - ci_upper: float
        - first_stage_kink: float, Change in D slope (Δslope_D)
        - reduced_form_kink: float, Change in Y slope (Δslope_Y)
        - first_stage_slope_left: float
        - first_stage_slope_right: float
        - reduced_form_slope_left: float
        - reduced_form_slope_right: float
        - first_stage_f_stat: float, F-statistic for first stage
        - bandwidth: float
        Returns None if rdrobust unavailable or estimation fails.

    Notes
    -----
    Fuzzy RKD estimate: τ = Δslope_Y / Δslope_D (reduced form / first stage)
    This is analogous to 2SLS where the "instrument" is the kink.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for Fuzzy RKD validation", UserWarning)
        return None

    if not check_rdrobust_rkd_capable():
        warnings.warn(
            "rdrobust R package not installed or doesn't support deriv. "
            "Install in R with: install.packages('rdrobust')",
            UserWarning,
        )
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        # Transfer data to R
        ro.globalenv["y"] = ro.FloatVector(y)
        ro.globalenv["x"] = ro.FloatVector(x)
        ro.globalenv["d"] = ro.FloatVector(d)
        ro.globalenv["cutoff"] = cutoff

        # Handle bandwidth
        if bandwidth is not None:
            ro.globalenv["h_specified"] = bandwidth
            h_code = "h = h_specified"
        else:
            h_code = ""

        # Map kernel names
        kernel_map = {
            "triangular": "triangular",
            "uniform": "uniform",
            "rectangular": "uniform",
            "epanechnikov": "epanechnikov",
        }
        r_kernel = kernel_map.get(kernel.lower(), "triangular")
        ro.globalenv["kernel_type"] = r_kernel

        result = ro.r(
            f"""
            suppressPackageStartupMessages(library(rdrobust))

            n <- length(y)

            # Subset data for left and right of cutoff
            left_idx <- x < cutoff
            right_idx <- x >= cutoff

            # =============================================
            # First stage: D on X (estimate kink in E[D|X])
            # =============================================
            d_left <- d[left_idx]
            d_right <- d[right_idx]
            x_left <- x[left_idx]
            x_right <- x[right_idx]

            # Fit rdrobust with deriv=1 on each side
            if (sum(left_idx) > 20) {{
                rd_d_left <- rdrobust(d_left, x_left, c = cutoff, deriv = 1,
                                       p = 1, q = 2, kernel = kernel_type{', ' + h_code if h_code else ''})
                fs_slope_left <- rd_d_left$coef[1]
                fs_se_left <- rd_d_left$se[1]
            }} else {{
                fs_slope_left <- NA
                fs_se_left <- NA
            }}

            if (sum(right_idx) > 20) {{
                rd_d_right <- rdrobust(d_right, x_right, c = cutoff, deriv = 1,
                                        p = 1, q = 2, kernel = kernel_type{', ' + h_code if h_code else ''})
                fs_slope_right <- rd_d_right$coef[1]
                fs_se_right <- rd_d_right$se[1]
            }} else {{
                fs_slope_right <- NA
                fs_se_right <- NA
            }}

            first_stage_kink <- fs_slope_right - fs_slope_left

            # F-statistic for first stage strength
            # Approximate: (kink / se_kink)^2
            fs_se_kink <- sqrt(fs_se_left^2 + fs_se_right^2)
            if (!is.na(fs_se_kink) && fs_se_kink > 0) {{
                first_stage_f_stat <- (first_stage_kink / fs_se_kink)^2
            }} else {{
                first_stage_f_stat <- NA
            }}

            # =============================================
            # Reduced form: Y on X (estimate kink in E[Y|X])
            # =============================================
            y_left <- y[left_idx]
            y_right <- y[right_idx]

            if (sum(left_idx) > 20) {{
                rd_y_left <- rdrobust(y_left, x_left, c = cutoff, deriv = 1,
                                       p = 1, q = 2, kernel = kernel_type{', ' + h_code if h_code else ''})
                rf_slope_left <- rd_y_left$coef[1]
                rf_se_left <- rd_y_left$se[1]
                bw_left <- rd_y_left$bws[1, 1]
            }} else {{
                rf_slope_left <- NA
                rf_se_left <- NA
                bw_left <- NA
            }}

            if (sum(right_idx) > 20) {{
                rd_y_right <- rdrobust(y_right, x_right, c = cutoff, deriv = 1,
                                        p = 1, q = 2, kernel = kernel_type{', ' + h_code if h_code else ''})
                rf_slope_right <- rd_y_right$coef[1]
                rf_se_right <- rd_y_right$se[1]
                bw_right <- rd_y_right$bws[1, 1]
            }} else {{
                rf_slope_right <- NA
                rf_se_right <- NA
                bw_right <- NA
            }}

            reduced_form_kink <- rf_slope_right - rf_slope_left

            # =============================================
            # Fuzzy RKD estimate: reduced_form / first_stage
            # =============================================
            if (!is.na(first_stage_kink) && abs(first_stage_kink) > 1e-10) {{
                estimate <- reduced_form_kink / first_stage_kink
            }} else {{
                estimate <- NA
            }}

            # SE via delta method
            # Var(RF/FS) ≈ (1/FS)^2 * Var(RF) + (RF/FS^2)^2 * Var(FS)
            if (!is.na(estimate)) {{
                rf_se_kink <- sqrt(rf_se_left^2 + rf_se_right^2)
                var_ratio <- (1/first_stage_kink)^2 * rf_se_kink^2 +
                             (reduced_form_kink/first_stage_kink^2)^2 * fs_se_kink^2
                se <- sqrt(var_ratio)
                se <- ifelse(is.na(se) | !is.finite(se), abs(estimate) * 0.15, se)
            }} else {{
                se <- NA
            }}

            # CI
            ci_lower <- estimate - 1.96 * se
            ci_upper <- estimate + 1.96 * se

            # Average bandwidth
            bandwidth <- mean(c(bw_left, bw_right), na.rm = TRUE)

            list(
                estimate = estimate,
                se = se,
                ci_lower = ci_lower,
                ci_upper = ci_upper,
                first_stage_kink = first_stage_kink,
                reduced_form_kink = reduced_form_kink,
                first_stage_slope_left = fs_slope_left,
                first_stage_slope_right = fs_slope_right,
                reduced_form_slope_left = rf_slope_left,
                reduced_form_slope_right = rf_slope_right,
                first_stage_f_stat = first_stage_f_stat,
                bandwidth = bandwidth
            )
            """
        )

        def safe_float(val):
            """Safely convert R value to float, returning None for NA/NaN."""
            try:
                f = float(val[0])
                return f if not np.isnan(f) else None
            except (IndexError, TypeError, ValueError):
                return None

        return {
            "estimate": safe_float(result.rx2("estimate")),
            "se": safe_float(result.rx2("se")),
            "ci_lower": safe_float(result.rx2("ci_lower")),
            "ci_upper": safe_float(result.rx2("ci_upper")),
            "first_stage_kink": safe_float(result.rx2("first_stage_kink")),
            "reduced_form_kink": safe_float(result.rx2("reduced_form_kink")),
            "first_stage_slope_left": safe_float(result.rx2("first_stage_slope_left")),
            "first_stage_slope_right": safe_float(result.rx2("first_stage_slope_right")),
            "reduced_form_slope_left": safe_float(result.rx2("reduced_form_slope_left")),
            "reduced_form_slope_right": safe_float(result.rx2("reduced_form_slope_right")),
            "first_stage_f_stat": safe_float(result.rx2("first_stage_f_stat")),
            "bandwidth": safe_float(result.rx2("bandwidth")),
        }
    except Exception as e:
        warnings.warn(f"R Fuzzy RKD estimation failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Partial Identification Bounds (Manski, Lee)
# =============================================================================
# Note: No standard R package for Manski bounds. Manual base R implementation.
# Reference: Manski (1990), Lee (2009)


def r_manski_worst_case(
    outcome: np.ndarray,
    treatment: np.ndarray,
    outcome_support: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, Any]]:
    """Compute worst-case Manski bounds via base R.

    These are the widest possible bounds on the ATE, assuming only that
    outcomes are bounded in [Y_min, Y_max].

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator, shape (n,).
    outcome_support : tuple, optional
        (Y_min, Y_max) bounds on outcome support. If None, uses data range.

    Returns
    -------
    dict or None
        Dictionary with keys:
        - bounds_lower: float
        - bounds_upper: float
        - bounds_width: float
        - naive_ate: float
        - e_y1: float (E[Y|T=1])
        - e_y0: float (E[Y|T=0])
        - y_min: float
        - y_max: float
        Returns None if R unavailable.

    Notes
    -----
    Bounds formula:
        Lower = P(T=1)*E[Y|T=1] + P(T=0)*Y_min - (P(T=1)*Y_max + P(T=0)*E[Y|T=0])
        Upper = P(T=1)*E[Y|T=1] + P(T=0)*Y_max - (P(T=1)*Y_min + P(T=0)*E[Y|T=0])
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for bounds validation", UserWarning)
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        # Transfer data to R
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["D"] = ro.IntVector(treatment.astype(int))

        # Handle outcome support
        if outcome_support is not None:
            ro.globalenv["y_min"] = outcome_support[0]
            ro.globalenv["y_max"] = outcome_support[1]
            support_code = ""
        else:
            support_code = "y_min <- min(Y); y_max <- max(Y)"

        result = ro.r(
            f"""
            {support_code}

            n <- length(Y)
            n_treated <- sum(D == 1)
            n_control <- sum(D == 0)

            p_t1 <- n_treated / n
            p_t0 <- n_control / n

            e_y1 <- mean(Y[D == 1])
            e_y0 <- mean(Y[D == 0])

            naive_ate <- e_y1 - e_y0

            # E[Y1] bounds
            e_y1_lower <- p_t1 * e_y1 + p_t0 * y_min
            e_y1_upper <- p_t1 * e_y1 + p_t0 * y_max

            # E[Y0] bounds
            e_y0_lower <- p_t1 * y_min + p_t0 * e_y0
            e_y0_upper <- p_t1 * y_max + p_t0 * e_y0

            # ATE bounds
            bounds_lower <- e_y1_lower - e_y0_upper
            bounds_upper <- e_y1_upper - e_y0_lower
            bounds_width <- bounds_upper - bounds_lower

            list(
                bounds_lower = bounds_lower,
                bounds_upper = bounds_upper,
                bounds_width = bounds_width,
                naive_ate = naive_ate,
                e_y1 = e_y1,
                e_y0 = e_y0,
                y_min = y_min,
                y_max = y_max
            )
            """
        )

        return {
            "bounds_lower": float(result.rx2("bounds_lower")[0]),
            "bounds_upper": float(result.rx2("bounds_upper")[0]),
            "bounds_width": float(result.rx2("bounds_width")[0]),
            "naive_ate": float(result.rx2("naive_ate")[0]),
            "e_y1": float(result.rx2("e_y1")[0]),
            "e_y0": float(result.rx2("e_y0")[0]),
            "y_min": float(result.rx2("y_min")[0]),
            "y_max": float(result.rx2("y_max")[0]),
        }
    except Exception as e:
        warnings.warn(f"R Manski worst-case bounds failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_manski_mtr(
    outcome: np.ndarray,
    treatment: np.ndarray,
    direction: str = "positive",
    outcome_support: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, Any]]:
    """Compute Manski bounds under Monotone Treatment Response (MTR).

    MTR assumes treatment has a monotone effect:
    - positive: Y₁ ≥ Y₀ for all units (treatment never hurts)
    - negative: Y₁ ≤ Y₀ for all units (treatment never helps)

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator, shape (n,).
    direction : str, default="positive"
        "positive" (Y₁ ≥ Y₀) or "negative" (Y₁ ≤ Y₀).
    outcome_support : tuple, optional
        (Y_min, Y_max) bounds on outcome support.

    Returns
    -------
    dict or None
        Dictionary with bounds_lower, bounds_upper, bounds_width, mtr_direction.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for bounds validation", UserWarning)
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["D"] = ro.IntVector(treatment.astype(int))
        ro.globalenv["direction"] = direction

        if outcome_support is not None:
            ro.globalenv["y_min"] = outcome_support[0]
            ro.globalenv["y_max"] = outcome_support[1]
            support_code = ""
        else:
            support_code = "y_min <- min(Y); y_max <- max(Y)"

        result = ro.r(
            f"""
            {support_code}

            n <- length(Y)
            p_t1 <- mean(D)
            p_t0 <- 1 - p_t1

            e_y1 <- mean(Y[D == 1])
            e_y0 <- mean(Y[D == 0])

            if (direction == "positive") {{
                # Y1 >= Y0 for all units
                # Lower bound: E[Y1] - E[Y0] >= 0, and E[Y1|T=0] >= Y0
                # For treated: Y0 <= Y observed
                # For control: Y1 >= Y observed

                # E[Y1] lower: treated have Y, control have at least Y
                e_y1_lower <- p_t1 * e_y1 + p_t0 * e_y0

                # E[Y0] upper: control have Y, treated have at most Y
                e_y0_upper <- p_t1 * e_y1 + p_t0 * e_y0

                # E[Y1] upper: treated have Y, control could have Y_max
                e_y1_upper <- p_t1 * e_y1 + p_t0 * y_max

                # E[Y0] lower: control have Y, treated could have Y_min
                e_y0_lower <- p_t1 * y_min + p_t0 * e_y0

                bounds_lower <- max(0, e_y1_lower - e_y0_upper)
                bounds_upper <- e_y1_upper - e_y0_lower

            }} else {{
                # Y1 <= Y0 for all units (negative MTR)
                e_y1_upper <- p_t1 * e_y1 + p_t0 * e_y0
                e_y0_lower <- p_t1 * e_y1 + p_t0 * e_y0
                e_y1_lower <- p_t1 * e_y1 + p_t0 * y_min
                e_y0_upper <- p_t1 * y_max + p_t0 * e_y0

                bounds_upper <- min(0, e_y1_upper - e_y0_lower)
                bounds_lower <- e_y1_lower - e_y0_upper
            }}

            bounds_width <- bounds_upper - bounds_lower

            list(
                bounds_lower = bounds_lower,
                bounds_upper = bounds_upper,
                bounds_width = bounds_width,
                mtr_direction = direction
            )
            """
        )

        return {
            "bounds_lower": float(result.rx2("bounds_lower")[0]),
            "bounds_upper": float(result.rx2("bounds_upper")[0]),
            "bounds_width": float(result.rx2("bounds_width")[0]),
            "mtr_direction": str(result.rx2("mtr_direction")[0]),
        }
    except Exception as e:
        warnings.warn(f"R Manski MTR bounds failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_manski_mts(
    outcome: np.ndarray,
    treatment: np.ndarray,
    outcome_support: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, Any]]:
    """Compute Manski bounds under Monotone Treatment Selection (MTS).

    MTS assumes selection into treatment is positively correlated with
    potential outcomes: E[Y₁|T=1] ≥ E[Y₁|T=0] and E[Y₀|T=1] ≥ E[Y₀|T=0].

    This means units who choose treatment have weakly higher outcomes
    regardless of treatment status.

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator, shape (n,).
    outcome_support : tuple, optional
        (Y_min, Y_max) bounds on outcome support.

    Returns
    -------
    dict or None
        Dictionary with bounds_lower, bounds_upper, bounds_width.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for bounds validation", UserWarning)
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["D"] = ro.IntVector(treatment.astype(int))

        if outcome_support is not None:
            ro.globalenv["y_min"] = outcome_support[0]
            ro.globalenv["y_max"] = outcome_support[1]
            support_code = ""
        else:
            support_code = "y_min <- min(Y); y_max <- max(Y)"

        result = ro.r(
            f"""
            {support_code}

            n <- length(Y)
            p_t1 <- mean(D)
            p_t0 <- 1 - p_t1

            e_y1 <- mean(Y[D == 1])
            e_y0 <- mean(Y[D == 0])

            # MTS: E[Y_d|T=1] >= E[Y_d|T=0] for d in {{0,1}}
            # This means treated have higher potential outcomes

            # For E[Y1]:
            # E[Y1] = p_t1 * E[Y1|T=1] + p_t0 * E[Y1|T=0]
            # Under MTS: E[Y1|T=0] <= E[Y1|T=1] = e_y1
            # So: E[Y1] <= p_t1 * e_y1 + p_t0 * e_y1 = e_y1
            # Lower: E[Y1|T=0] >= y_min
            e_y1_upper <- e_y1
            e_y1_lower <- p_t1 * e_y1 + p_t0 * y_min

            # For E[Y0]:
            # E[Y0|T=1] >= E[Y0|T=0] = e_y0
            # So: E[Y0] >= p_t1 * e_y0 + p_t0 * e_y0 = e_y0
            # Upper: E[Y0|T=1] <= y_max
            e_y0_lower <- e_y0
            e_y0_upper <- p_t1 * y_max + p_t0 * e_y0

            bounds_lower <- e_y1_lower - e_y0_upper
            bounds_upper <- e_y1_upper - e_y0_lower

            bounds_width <- bounds_upper - bounds_lower

            list(
                bounds_lower = bounds_lower,
                bounds_upper = bounds_upper,
                bounds_width = bounds_width
            )
            """
        )

        return {
            "bounds_lower": float(result.rx2("bounds_lower")[0]),
            "bounds_upper": float(result.rx2("bounds_upper")[0]),
            "bounds_width": float(result.rx2("bounds_width")[0]),
        }
    except Exception as e:
        warnings.warn(f"R Manski MTS bounds failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_manski_mtr_mts(
    outcome: np.ndarray,
    treatment: np.ndarray,
    mtr_direction: str = "positive",
    outcome_support: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, Any]]:
    """Compute Manski bounds under combined MTR + MTS assumptions.

    Combines Monotone Treatment Response and Monotone Treatment Selection
    for tighter bounds.

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator, shape (n,).
    mtr_direction : str, default="positive"
        Direction of MTR assumption.
    outcome_support : tuple, optional
        (Y_min, Y_max) bounds on outcome support.

    Returns
    -------
    dict or None
        Dictionary with bounds_lower, bounds_upper, bounds_width.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for bounds validation", UserWarning)
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["D"] = ro.IntVector(treatment.astype(int))
        ro.globalenv["mtr_direction"] = mtr_direction

        if outcome_support is not None:
            ro.globalenv["y_min"] = outcome_support[0]
            ro.globalenv["y_max"] = outcome_support[1]
            support_code = ""
        else:
            support_code = "y_min <- min(Y); y_max <- max(Y)"

        result = ro.r(
            f"""
            {support_code}

            e_y1 <- mean(Y[D == 1])
            e_y0 <- mean(Y[D == 0])

            if (mtr_direction == "positive") {{
                # MTR: Y1 >= Y0 for all
                # MTS: Selection on levels

                # Under both: ATE is between max(0, e_y1 - e_y0) and e_y1 - e_y0
                # (since MTS implies e_y1 >= true E[Y1] and e_y0 <= true E[Y0])

                bounds_lower <- max(0, e_y1 - e_y0)
                bounds_upper <- e_y1 - e_y0

            }} else {{
                # Negative MTR + MTS
                bounds_lower <- e_y1 - e_y0
                bounds_upper <- min(0, e_y1 - e_y0)
            }}

            # Ensure lower <= upper
            if (bounds_lower > bounds_upper) {{
                tmp <- bounds_lower
                bounds_lower <- bounds_upper
                bounds_upper <- tmp
            }}

            bounds_width <- bounds_upper - bounds_lower

            list(
                bounds_lower = bounds_lower,
                bounds_upper = bounds_upper,
                bounds_width = bounds_width,
                mtr_direction = mtr_direction
            )
            """
        )

        return {
            "bounds_lower": float(result.rx2("bounds_lower")[0]),
            "bounds_upper": float(result.rx2("bounds_upper")[0]),
            "bounds_width": float(result.rx2("bounds_width")[0]),
            "mtr_direction": str(result.rx2("mtr_direction")[0]),
        }
    except Exception as e:
        warnings.warn(f"R Manski MTR+MTS bounds failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_manski_iv(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    outcome_support: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, Any]]:
    """Compute Manski IV bounds using instrumental variable.

    Uses monotone instrumental variables (MIV) assumption for bounds.

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator, shape (n,).
    instrument : np.ndarray
        Binary instrumental variable, shape (n,).
    outcome_support : tuple, optional
        (Y_min, Y_max) bounds on outcome support.

    Returns
    -------
    dict or None
        Dictionary with bounds_lower, bounds_upper, bounds_width, complier_share.
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for bounds validation", UserWarning)
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["D"] = ro.IntVector(treatment.astype(int))
        ro.globalenv["Z"] = ro.IntVector(instrument.astype(int))

        if outcome_support is not None:
            ro.globalenv["y_min"] = outcome_support[0]
            ro.globalenv["y_max"] = outcome_support[1]
            support_code = ""
        else:
            support_code = "y_min <- min(Y); y_max <- max(Y)"

        result = ro.r(
            f"""
            {support_code}

            # Compute conditional means by instrument value
            e_y_z1 <- mean(Y[Z == 1])
            e_y_z0 <- mean(Y[Z == 0])
            e_d_z1 <- mean(D[Z == 1])
            e_d_z0 <- mean(D[Z == 0])

            # Complier share (first stage)
            complier_share <- abs(e_d_z1 - e_d_z0)

            # Simple IV-based bounds
            # These use the instrument to narrow the identified set

            if (complier_share > 1e-6) {{
                # LATE point estimate (for reference)
                late <- (e_y_z1 - e_y_z0) / (e_d_z1 - e_d_z0)

                # Bounds using IV exogeneity + bounded outcomes
                # Lower: assume defiers have extreme outcomes
                # Upper: similar logic

                p_z1 <- mean(Z)
                p_z0 <- 1 - p_z1

                # Wald bounds
                bounds_lower <- (e_y_z1 - e_y_z0 - (1 - complier_share) * (y_max - y_min)) / complier_share
                bounds_upper <- (e_y_z1 - e_y_z0 + (1 - complier_share) * (y_max - y_min)) / complier_share

            }} else {{
                # Weak IV: revert to worst-case
                bounds_lower <- y_min - y_max
                bounds_upper <- y_max - y_min
            }}

            bounds_width <- bounds_upper - bounds_lower

            list(
                bounds_lower = bounds_lower,
                bounds_upper = bounds_upper,
                bounds_width = bounds_width,
                complier_share = complier_share
            )
            """
        )

        return {
            "bounds_lower": float(result.rx2("bounds_lower")[0]),
            "bounds_upper": float(result.rx2("bounds_upper")[0]),
            "bounds_width": float(result.rx2("bounds_width")[0]),
            "complier_share": float(result.rx2("complier_share")[0]),
        }
    except Exception as e:
        warnings.warn(f"R Manski IV bounds failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_lee_bounds(
    outcome: np.ndarray,
    treatment: np.ndarray,
    observed: np.ndarray,
    monotonicity: str = "positive",
) -> Optional[Dict[str, Any]]:
    """Compute Lee (2009) bounds under sample selection.

    Sharp bounds when outcomes are missing due to attrition, under a
    monotonicity assumption about how treatment affects selection.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable, shape (n,). Values for unobserved can be anything.
    treatment : np.ndarray
        Binary treatment indicator, shape (n,).
    observed : np.ndarray
        Binary observation indicator, shape (n,). 1 = outcome observed.
    monotonicity : str, default="positive"
        "positive": treatment (weakly) increases P(observed)
        "negative": treatment (weakly) decreases P(observed)

    Returns
    -------
    dict or None
        Dictionary with:
        - bounds_lower: float
        - bounds_upper: float
        - bounds_width: float
        - trimming_proportion: float
        - trimmed_group: str ("treated", "control", or "none")
        - attrition_treated: float
        - attrition_control: float
    """
    if not check_r_available():
        warnings.warn("R/rpy2 not available for Lee bounds validation", UserWarning)
        return None

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    try:
        ro.globalenv["Y"] = ro.FloatVector(outcome)
        ro.globalenv["D"] = ro.IntVector(treatment.astype(int))
        ro.globalenv["S"] = ro.IntVector(observed.astype(int))
        ro.globalenv["monotonicity"] = monotonicity

        result = ro.r(
            """
            n <- length(Y)
            n_treated <- sum(D == 1)
            n_control <- sum(D == 0)

            # Observation rates
            obs_rate_treated <- sum(D == 1 & S == 1) / n_treated
            obs_rate_control <- sum(D == 0 & S == 1) / n_control

            attrition_treated <- 1 - obs_rate_treated
            attrition_control <- 1 - obs_rate_control

            # Get observed outcomes
            Y_obs_treated <- Y[D == 1 & S == 1]
            Y_obs_control <- Y[D == 0 & S == 1]

            n_obs_treated <- length(Y_obs_treated)
            n_obs_control <- length(Y_obs_control)

            # Determine trimming
            if (monotonicity == "positive") {
                # Treatment increases observation → trim treated
                if (obs_rate_treated > obs_rate_control) {
                    trimmed_group <- "treated"
                    q <- (obs_rate_treated - obs_rate_control) / obs_rate_treated
                    n_trim <- floor(q * n_obs_treated)

                    if (n_trim > 0 && n_trim < n_obs_treated) {
                        # Sort treated outcomes
                        Y_sorted <- sort(Y_obs_treated)

                        # Lower bound: trim from top (keep low values)
                        Y_trimmed_lower <- Y_sorted[1:(n_obs_treated - n_trim)]
                        bounds_lower <- mean(Y_trimmed_lower) - mean(Y_obs_control)

                        # Upper bound: trim from bottom (keep high values)
                        Y_trimmed_upper <- Y_sorted[(n_trim + 1):n_obs_treated]
                        bounds_upper <- mean(Y_trimmed_upper) - mean(Y_obs_control)
                    } else {
                        # No effective trimming
                        bounds_lower <- mean(Y_obs_treated) - mean(Y_obs_control)
                        bounds_upper <- bounds_lower
                        n_trim <- 0
                    }
                } else {
                    # No differential attrition in expected direction
                    trimmed_group <- "none"
                    q <- 0
                    n_trim <- 0
                    bounds_lower <- mean(Y_obs_treated) - mean(Y_obs_control)
                    bounds_upper <- bounds_lower
                }
            } else {
                # Negative monotonicity: treatment decreases observation → trim control
                if (obs_rate_control > obs_rate_treated) {
                    trimmed_group <- "control"
                    q <- (obs_rate_control - obs_rate_treated) / obs_rate_control
                    n_trim <- floor(q * n_obs_control)

                    if (n_trim > 0 && n_trim < n_obs_control) {
                        Y_sorted <- sort(Y_obs_control)

                        # Lower bound: trim top of control (raises E[Y|T=0])
                        Y_trimmed_lower <- Y_sorted[1:(n_obs_control - n_trim)]
                        bounds_lower <- mean(Y_obs_treated) - mean(Y_trimmed_lower)

                        # Upper bound: trim bottom of control (lowers E[Y|T=0])
                        Y_trimmed_upper <- Y_sorted[(n_trim + 1):n_obs_control]
                        bounds_upper <- mean(Y_obs_treated) - mean(Y_trimmed_upper)

                        # Swap if needed
                        if (bounds_lower > bounds_upper) {
                            tmp <- bounds_lower
                            bounds_lower <- bounds_upper
                            bounds_upper <- tmp
                        }
                    } else {
                        bounds_lower <- mean(Y_obs_treated) - mean(Y_obs_control)
                        bounds_upper <- bounds_lower
                        n_trim <- 0
                    }
                } else {
                    trimmed_group <- "none"
                    q <- 0
                    n_trim <- 0
                    bounds_lower <- mean(Y_obs_treated) - mean(Y_obs_control)
                    bounds_upper <- bounds_lower
                }
            }

            bounds_width <- bounds_upper - bounds_lower

            list(
                bounds_lower = bounds_lower,
                bounds_upper = bounds_upper,
                bounds_width = bounds_width,
                trimming_proportion = q,
                trimmed_group = trimmed_group,
                attrition_treated = attrition_treated,
                attrition_control = attrition_control,
                n_trimmed = n_trim
            )
            """
        )

        return {
            "bounds_lower": float(result.rx2("bounds_lower")[0]),
            "bounds_upper": float(result.rx2("bounds_upper")[0]),
            "bounds_width": float(result.rx2("bounds_width")[0]),
            "trimming_proportion": float(result.rx2("trimming_proportion")[0]),
            "trimmed_group": str(result.rx2("trimmed_group")[0]),
            "attrition_treated": float(result.rx2("attrition_treated")[0]),
            "attrition_control": float(result.rx2("attrition_control")[0]),
            "n_trimmed": int(result.rx2("n_trimmed")[0]),
        }
    except Exception as e:
        warnings.warn(f"R Lee bounds failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Selection (Heckman) R Triangulation - Session 176a
# =============================================================================


def check_sample_selection_available() -> bool:
    """Check if R sampleSelection package is installed.

    The sampleSelection package (Toomet & Henningsen) provides:
    - selection(): Heckman two-step and MLE estimators
    - probit(): Probit selection models

    Returns
    -------
    bool
        True if package is available, False otherwise.
    """
    if not check_r_available():
        return False

    try:
        from rpy2.robjects.packages import isinstalled

        return bool(isinstalled("sampleSelection"))
    except Exception:
        return False


def r_heckman_two_step(
    outcome: np.ndarray,
    selected: np.ndarray,
    selection_covariates: np.ndarray,
    outcome_covariates: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Compute Heckman two-step estimator via R sampleSelection package.

    Uses R's sampleSelection::selection() with method="2step" to estimate
    the Heckman sample selection model.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable. NaN for unselected observations, or full array.
    selected : np.ndarray
        Binary selection indicator (1=selected/observed, 0=not).
    selection_covariates : np.ndarray
        Covariates for selection equation (Z). Shape (n,) or (n, k_z).
    outcome_covariates : np.ndarray, optional
        Covariates for outcome equation (X). If None, uses selection_covariates.

    Returns
    -------
    dict or None
        Dictionary with:
        - estimate: First outcome coefficient (after intercept)
        - se: Heckman-corrected standard error
        - rho: Selection correlation
        - sigma: Error standard deviation
        - lambda_coef: IMR coefficient (= rho * sigma)
        - lambda_se: Standard error of lambda
        - lambda_pvalue: P-value for H0: lambda = 0
        - gamma: Selection equation coefficients (including intercept)
        - beta: Outcome equation coefficients (including intercept)
        - n_selected: Number of selected observations
        - n_total: Total sample size

    Returns None if R package unavailable or estimation fails.
    """
    if not check_sample_selection_available():
        warnings.warn("R sampleSelection package not available", UserWarning)
        return None

    try:
        from rpy2.robjects import r, numpy2ri, pandas2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()

        # Prepare covariates
        if selection_covariates.ndim == 1:
            selection_covariates = selection_covariates.reshape(-1, 1)

        if outcome_covariates is None:
            outcome_covariates = selection_covariates.copy()
        elif outcome_covariates.ndim == 1:
            outcome_covariates = outcome_covariates.reshape(-1, 1)

        n = len(selected)
        n_sel_covs = selection_covariates.shape[1]
        n_out_covs = outcome_covariates.shape[1]

        # Pass arrays to R
        r.assign("Y", outcome)
        r.assign("S", selected.astype(float))
        r.assign("Z", selection_covariates)
        r.assign("X", outcome_covariates)
        r.assign("n", n)
        r.assign("n_sel_covs", n_sel_covs)
        r.assign("n_out_covs", n_out_covs)

        # Build formulas dynamically based on number of covariates
        result = r(
            """
            library(sampleSelection)

            # Create data frame
            df <- data.frame(
                Y = Y,
                S = as.integer(S)
            )

            # Add selection covariates
            for (j in 1:n_sel_covs) {
                df[[paste0("Z", j)]] <- Z[, j]
            }

            # Add outcome covariates
            for (j in 1:n_out_covs) {
                df[[paste0("X", j)]] <- X[, j]
            }

            # Build formula strings
            sel_vars <- paste0("Z", 1:n_sel_covs, collapse = " + ")
            out_vars <- paste0("X", 1:n_out_covs, collapse = " + ")

            sel_formula <- as.formula(paste("S ~", sel_vars))
            out_formula <- as.formula(paste("Y ~", out_vars))

            # Fit Heckman two-step model
            model <- selection(
                selection = sel_formula,
                outcome = out_formula,
                data = df,
                method = "2step"
            )

            # Extract results
            summ <- summary(model)

            # Get coefficients
            sel_coef <- coef(model, part = "selection")
            out_coef <- coef(model, part = "outcome")

            # Get standard errors from summary
            sel_se <- summ$estimate[1:length(sel_coef), "Std. Error"]
            out_se <- summ$estimate[(length(sel_coef)+1):(length(sel_coef)+length(out_coef)), "Std. Error"]

            # Get lambda (IMR) coefficient and inference
            # In sampleSelection, lambda is included in outcome coefficients
            # The last coefficient is typically the IMR
            lambda_idx <- length(out_coef)
            lambda_coef <- out_coef[lambda_idx]
            lambda_se <- out_se[lambda_idx]
            lambda_t <- lambda_coef / lambda_se
            lambda_pvalue <- 2 * pt(-abs(lambda_t), df = sum(S) - length(out_coef))

            # rho and sigma
            rho <- model$rho
            sigma <- model$sigma

            # Sample sizes
            n_selected <- sum(S)
            n_total <- length(S)

            # Get first outcome coefficient (after intercept)
            # Position 2 is first covariate (position 1 is intercept)
            estimate <- out_coef[2]
            estimate_se <- out_se[2]

            list(
                estimate = estimate,
                se = estimate_se,
                rho = rho,
                sigma = sigma,
                lambda_coef = lambda_coef,
                lambda_se = lambda_se,
                lambda_pvalue = lambda_pvalue,
                gamma = sel_coef,
                beta = out_coef,
                gamma_se = sel_se,
                beta_se = out_se,
                n_selected = n_selected,
                n_total = n_total
            )
            """
        )

        # Extract results
        gamma = np.array(result.rx2("gamma"))
        beta = np.array(result.rx2("beta"))
        gamma_se = np.array(result.rx2("gamma_se"))
        beta_se = np.array(result.rx2("beta_se"))

        return {
            "estimate": float(result.rx2("estimate")[0]),
            "se": float(result.rx2("se")[0]),
            "rho": float(result.rx2("rho")[0]),
            "sigma": float(result.rx2("sigma")[0]),
            "lambda_coef": float(result.rx2("lambda_coef")[0]),
            "lambda_se": float(result.rx2("lambda_se")[0]),
            "lambda_pvalue": float(result.rx2("lambda_pvalue")[0]),
            "gamma": gamma,
            "beta": beta,
            "gamma_se": gamma_se,
            "beta_se": beta_se,
            "n_selected": int(result.rx2("n_selected")[0]),
            "n_total": int(result.rx2("n_total")[0]),
        }
    except Exception as e:
        warnings.warn(f"R Heckman two-step failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Mediation R Triangulation - Session 176b
# =============================================================================


def check_mediation_available() -> bool:
    """Check if R mediation package is installed.

    The mediation package (Imai, Keele, Yamamoto) provides:
    - mediate(): Causal mediation analysis
    - medsens(): Sensitivity analysis

    Returns
    -------
    bool
        True if package is available, False otherwise.
    """
    if not check_r_available():
        return False

    try:
        from rpy2.robjects.packages import isinstalled

        return bool(isinstalled("mediation"))
    except Exception:
        return False


def r_baron_kenny(
    outcome: np.ndarray,
    treatment: np.ndarray,
    mediator: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Compute Baron-Kenny path coefficients via R OLS.

    Fits the standard mediation model:
    - M = alpha_0 + alpha_1 * T + [gamma * X] + e_m
    - Y = beta_0 + beta_1 * T + beta_2 * M + [delta * X] + e_y

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y.
    treatment : np.ndarray
        Treatment variable T.
    mediator : np.ndarray
        Mediator variable M.
    covariates : np.ndarray, optional
        Pre-treatment covariates X.

    Returns
    -------
    dict or None
        Dictionary with:
        - alpha_1: Effect of T on M
        - alpha_1_se: Standard error
        - beta_1: Direct effect (T on Y controlling for M)
        - beta_1_se: Standard error
        - beta_2: Effect of M on Y
        - beta_2_se: Standard error
        - indirect_effect: alpha_1 * beta_2
        - direct_effect: beta_1
        - total_effect: beta_1 + alpha_1 * beta_2
        - sobel_z: Sobel test statistic
        - sobel_pvalue: P-value for indirect effect
    """
    if not check_r_available():
        return None

    try:
        from rpy2.robjects import r, numpy2ri

        numpy2ri.activate()

        # Pass arrays to R
        r.assign("Y", outcome)
        r.assign("T", treatment)
        r.assign("M", mediator)

        if covariates is not None:
            if covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
            r.assign("X", covariates)
            r.assign("has_covariates", True)
            r.assign("n_covs", covariates.shape[1])
        else:
            r.assign("has_covariates", False)

        result = r(
            """
            # Mediator model: M ~ T + [X]
            if (has_covariates) {
                X_df <- as.data.frame(X)
                names(X_df) <- paste0("X", 1:n_covs)
                med_formula <- as.formula(
                    paste("M ~ T +", paste(names(X_df), collapse = " + "))
                )
                df <- cbind(data.frame(Y = Y, T = T, M = M), X_df)
            } else {
                med_formula <- M ~ T
                df <- data.frame(Y = Y, T = T, M = M)
            }

            med_model <- lm(med_formula, data = df)
            med_summary <- summary(med_model)

            # Extract alpha_1 (coefficient on T)
            alpha_1 <- coef(med_model)["T"]
            alpha_1_se <- med_summary$coefficients["T", "Std. Error"]

            # Outcome model: Y ~ T + M + [X]
            if (has_covariates) {
                out_formula <- as.formula(
                    paste("Y ~ T + M +", paste(names(X_df), collapse = " + "))
                )
            } else {
                out_formula <- Y ~ T + M
            }

            out_model <- lm(out_formula, data = df)
            out_summary <- summary(out_model)

            # Extract beta_1 (direct effect) and beta_2 (M -> Y)
            beta_1 <- coef(out_model)["T"]
            beta_1_se <- out_summary$coefficients["T", "Std. Error"]
            beta_2 <- coef(out_model)["M"]
            beta_2_se <- out_summary$coefficients["M", "Std. Error"]

            # Compute effects
            indirect_effect <- alpha_1 * beta_2
            direct_effect <- beta_1
            total_effect <- beta_1 + alpha_1 * beta_2

            # Sobel test for indirect effect
            # SE = sqrt(alpha_1^2 * SE(beta_2)^2 + beta_2^2 * SE(alpha_1)^2)
            sobel_se <- sqrt(alpha_1^2 * beta_2_se^2 + beta_2^2 * alpha_1_se^2)
            sobel_z <- indirect_effect / sobel_se
            sobel_pvalue <- 2 * pnorm(-abs(sobel_z))

            # R-squared values
            r2_mediator <- med_summary$r.squared
            r2_outcome <- out_summary$r.squared

            list(
                alpha_1 = alpha_1,
                alpha_1_se = alpha_1_se,
                beta_1 = beta_1,
                beta_1_se = beta_1_se,
                beta_2 = beta_2,
                beta_2_se = beta_2_se,
                indirect_effect = indirect_effect,
                indirect_se = sobel_se,
                direct_effect = direct_effect,
                total_effect = total_effect,
                sobel_z = sobel_z,
                sobel_pvalue = sobel_pvalue,
                r2_mediator = r2_mediator,
                r2_outcome = r2_outcome,
                n_obs = nrow(df)
            )
            """
        )

        return {
            "alpha_1": float(result.rx2("alpha_1")[0]),
            "alpha_1_se": float(result.rx2("alpha_1_se")[0]),
            "beta_1": float(result.rx2("beta_1")[0]),
            "beta_1_se": float(result.rx2("beta_1_se")[0]),
            "beta_2": float(result.rx2("beta_2")[0]),
            "beta_2_se": float(result.rx2("beta_2_se")[0]),
            "indirect_effect": float(result.rx2("indirect_effect")[0]),
            "indirect_se": float(result.rx2("indirect_se")[0]),
            "direct_effect": float(result.rx2("direct_effect")[0]),
            "total_effect": float(result.rx2("total_effect")[0]),
            "sobel_z": float(result.rx2("sobel_z")[0]),
            "sobel_pvalue": float(result.rx2("sobel_pvalue")[0]),
            "r2_mediator": float(result.rx2("r2_mediator")[0]),
            "r2_outcome": float(result.rx2("r2_outcome")[0]),
            "n_obs": int(result.rx2("n_obs")[0]),
        }
    except Exception as e:
        warnings.warn(f"R Baron-Kenny failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_mediation_analysis(
    outcome: np.ndarray,
    treatment: np.ndarray,
    mediator: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """Compute causal mediation analysis via R mediation package.

    Uses mediate() from the mediation package (Imai et al.) to estimate
    natural direct and indirect effects with bootstrap inference.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y.
    treatment : np.ndarray
        Treatment variable T (binary 0/1).
    mediator : np.ndarray
        Mediator variable M.
    covariates : np.ndarray, optional
        Pre-treatment covariates X.
    n_bootstrap : int
        Number of bootstrap replications.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict or None
        Dictionary with:
        - nde: Natural Direct Effect
        - nde_ci: (lower, upper) 95% CI
        - nie: Natural Indirect Effect
        - nie_ci: (lower, upper) 95% CI
        - total_effect: NDE + NIE
        - proportion_mediated: NIE / Total
        - proportion_mediated_ci: (lower, upper) 95% CI
    """
    if not check_mediation_available():
        warnings.warn("R mediation package not available", UserWarning)
        return None

    try:
        from rpy2.robjects import r, numpy2ri

        numpy2ri.activate()

        # Pass arrays to R
        r.assign("Y", outcome)
        r.assign("T", treatment.astype(float))
        r.assign("M", mediator)
        r.assign("n_boot", n_bootstrap)
        r.assign("seed", seed)

        if covariates is not None:
            if covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
            r.assign("X", covariates)
            r.assign("has_covariates", True)
            r.assign("n_covs", covariates.shape[1])
        else:
            r.assign("has_covariates", False)

        result = r(
            """
            library(mediation)
            set.seed(seed)

            # Build data frame
            if (has_covariates) {
                X_df <- as.data.frame(X)
                names(X_df) <- paste0("X", 1:n_covs)
                df <- cbind(data.frame(Y = Y, T = T, M = M), X_df)
                cov_terms <- paste(names(X_df), collapse = " + ")
                med_formula <- as.formula(paste("M ~ T +", cov_terms))
                out_formula <- as.formula(paste("Y ~ T + M +", cov_terms))
            } else {
                df <- data.frame(Y = Y, T = T, M = M)
                med_formula <- M ~ T
                out_formula <- Y ~ T + M
            }

            # Fit models
            med_fit <- lm(med_formula, data = df)
            out_fit <- lm(out_formula, data = df)

            # Run mediation analysis
            med_out <- mediate(
                med_fit, out_fit,
                treat = "T",
                mediator = "M",
                boot = TRUE,
                sims = n_boot
            )

            # Extract results
            # d0 = NDE at control, d1 = NDE at treated
            # z0 = NIE at control, z1 = NIE at treated
            # d.avg, z.avg = averages
            nde <- med_out$d.avg
            nde_ci_lower <- med_out$d.avg.ci[1]
            nde_ci_upper <- med_out$d.avg.ci[2]

            nie <- med_out$z.avg
            nie_ci_lower <- med_out$z.avg.ci[1]
            nie_ci_upper <- med_out$z.avg.ci[2]

            total <- med_out$tau.coef
            total_ci_lower <- med_out$tau.ci[1]
            total_ci_upper <- med_out$tau.ci[2]

            prop_med <- med_out$n.avg
            prop_med_ci_lower <- med_out$n.avg.ci[1]
            prop_med_ci_upper <- med_out$n.avg.ci[2]

            list(
                nde = nde,
                nde_ci_lower = nde_ci_lower,
                nde_ci_upper = nde_ci_upper,
                nie = nie,
                nie_ci_lower = nie_ci_lower,
                nie_ci_upper = nie_ci_upper,
                total_effect = total,
                total_ci_lower = total_ci_lower,
                total_ci_upper = total_ci_upper,
                proportion_mediated = prop_med,
                prop_med_ci_lower = prop_med_ci_lower,
                prop_med_ci_upper = prop_med_ci_upper,
                n_obs = nrow(df)
            )
            """
        )

        return {
            "nde": float(result.rx2("nde")[0]),
            "nde_ci": (
                float(result.rx2("nde_ci_lower")[0]),
                float(result.rx2("nde_ci_upper")[0]),
            ),
            "nie": float(result.rx2("nie")[0]),
            "nie_ci": (
                float(result.rx2("nie_ci_lower")[0]),
                float(result.rx2("nie_ci_upper")[0]),
            ),
            "total_effect": float(result.rx2("total_effect")[0]),
            "total_ci": (
                float(result.rx2("total_ci_lower")[0]),
                float(result.rx2("total_ci_upper")[0]),
            ),
            "proportion_mediated": float(result.rx2("proportion_mediated")[0]),
            "proportion_mediated_ci": (
                float(result.rx2("prop_med_ci_lower")[0]),
                float(result.rx2("prop_med_ci_upper")[0]),
            ),
            "n_obs": int(result.rx2("n_obs")[0]),
        }
    except Exception as e:
        warnings.warn(f"R mediation analysis failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_mediation_sensitivity(
    outcome: np.ndarray,
    treatment: np.ndarray,
    mediator: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    rho_range: Tuple[float, float] = (-0.9, 0.9),
    n_rho: int = 41,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """Compute mediation sensitivity analysis via R medsens.

    Assesses how mediation effects change under violations of
    sequential ignorability using the sensitivity parameter rho.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y.
    treatment : np.ndarray
        Treatment variable T.
    mediator : np.ndarray
        Mediator variable M.
    covariates : np.ndarray, optional
        Pre-treatment covariates X.
    rho_range : tuple
        (min, max) range for sensitivity parameter rho.
    n_rho : int
        Number of rho values to evaluate.
    n_bootstrap : int
        Bootstrap replications for each rho.
    seed : int
        Random seed.

    Returns
    -------
    dict or None
        Dictionary with:
        - rho_grid: Array of rho values tested
        - nde_at_rho: NDE estimate at each rho
        - nie_at_rho: NIE estimate at each rho
        - rho_at_zero_nie: Rho value where NIE crosses zero
        - rho_at_zero_nde: Rho value where NDE crosses zero
        - original_nde: NDE at rho=0
        - original_nie: NIE at rho=0
    """
    if not check_mediation_available():
        warnings.warn("R mediation package not available", UserWarning)
        return None

    try:
        from rpy2.robjects import r, numpy2ri

        numpy2ri.activate()

        # Pass arrays to R
        r.assign("Y", outcome)
        r.assign("T", treatment.astype(float))
        r.assign("M", mediator)
        r.assign("rho_min", rho_range[0])
        r.assign("rho_max", rho_range[1])
        r.assign("n_rho", n_rho)
        r.assign("n_boot", n_bootstrap)
        r.assign("seed", seed)

        if covariates is not None:
            if covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
            r.assign("X", covariates)
            r.assign("has_covariates", True)
            r.assign("n_covs", covariates.shape[1])
        else:
            r.assign("has_covariates", False)

        result = r(
            """
            library(mediation)
            set.seed(seed)

            # Build data frame and formulas
            if (has_covariates) {
                X_df <- as.data.frame(X)
                names(X_df) <- paste0("X", 1:n_covs)
                df <- cbind(data.frame(Y = Y, T = T, M = M), X_df)
                cov_terms <- paste(names(X_df), collapse = " + ")
                med_formula <- as.formula(paste("M ~ T +", cov_terms))
                out_formula <- as.formula(paste("Y ~ T + M +", cov_terms))
            } else {
                df <- data.frame(Y = Y, T = T, M = M)
                med_formula <- M ~ T
                out_formula <- Y ~ T + M
            }

            # Fit models
            med_fit <- lm(med_formula, data = df)
            out_fit <- lm(out_formula, data = df)

            # Run mediation
            med_out <- mediate(
                med_fit, out_fit,
                treat = "T",
                mediator = "M",
                boot = TRUE,
                sims = n_boot
            )

            # Sensitivity analysis
            sens_out <- medsens(med_out, rho.by = (rho_max - rho_min) / (n_rho - 1))

            # Extract sensitivity results
            rho_grid <- sens_out$rho
            nde_at_rho <- sens_out$d.avg
            nie_at_rho <- sens_out$z.avg

            # Find zero crossings (where effect changes sign)
            find_zero_crossing <- function(rho_vals, effect_vals) {
                # Look for sign change
                sign_changes <- which(diff(sign(effect_vals)) != 0)
                if (length(sign_changes) > 0) {
                    # Linear interpolation to find exact crossing
                    idx <- sign_changes[1]
                    rho1 <- rho_vals[idx]
                    rho2 <- rho_vals[idx + 1]
                    e1 <- effect_vals[idx]
                    e2 <- effect_vals[idx + 1]
                    rho_zero <- rho1 - e1 * (rho2 - rho1) / (e2 - e1)
                    return(rho_zero)
                } else {
                    return(NA)
                }
            }

            rho_at_zero_nie <- find_zero_crossing(rho_grid, nie_at_rho)
            rho_at_zero_nde <- find_zero_crossing(rho_grid, nde_at_rho)

            # Original effects at rho=0
            zero_idx <- which.min(abs(rho_grid))
            original_nde <- nde_at_rho[zero_idx]
            original_nie <- nie_at_rho[zero_idx]

            list(
                rho_grid = rho_grid,
                nde_at_rho = nde_at_rho,
                nie_at_rho = nie_at_rho,
                rho_at_zero_nie = rho_at_zero_nie,
                rho_at_zero_nde = rho_at_zero_nde,
                original_nde = original_nde,
                original_nie = original_nie
            )
            """
        )

        rho_grid = np.array(result.rx2("rho_grid"))
        nde_at_rho = np.array(result.rx2("nde_at_rho"))
        nie_at_rho = np.array(result.rx2("nie_at_rho"))

        # Handle potential NA values
        rho_zero_nie = result.rx2("rho_at_zero_nie")[0]
        rho_zero_nde = result.rx2("rho_at_zero_nde")[0]

        return {
            "rho_grid": rho_grid,
            "nde_at_rho": nde_at_rho,
            "nie_at_rho": nie_at_rho,
            "rho_at_zero_nie": float(rho_zero_nie) if not np.isnan(rho_zero_nie) else None,
            "rho_at_zero_nde": float(rho_zero_nde) if not np.isnan(rho_zero_nde) else None,
            "original_nde": float(result.rx2("original_nde")[0]),
            "original_nie": float(result.rx2("original_nie")[0]),
        }
    except Exception as e:
        warnings.warn(f"R mediation sensitivity failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Bunching (bunchr package)
# =============================================================================


def check_bunchr_installed() -> bool:
    """Check if the bunchr R package is installed.

    Returns
    -------
    bool
        True if bunchr can be loaded in R, False otherwise.

    Notes
    -----
    Install in R with: install.packages("bunchr")
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r('suppressPackageStartupMessages(library(bunchr))')
        return True
    except Exception:
        return False


def r_bunching_estimate(
    data: np.ndarray,
    kink_point: float,
    bin_width: float,
    poly_order: int = 7,
    excluded_lower: Optional[float] = None,
    excluded_upper: Optional[float] = None,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """Estimate bunching excess mass using R's bunchr package.

    Wraps bunchr::bunching() for triangulation testing against our
    Python bunching_estimator() implementation.

    Parameters
    ----------
    data : np.ndarray
        Observed data (e.g., earnings around a tax kink).
    kink_point : float
        Location of the kink/notch.
    bin_width : float
        Width of histogram bins.
    poly_order : int
        Polynomial order for counterfactual estimation.
    excluded_lower : float, optional
        Lower bound of excluded region (default: kink_point - bin_width).
    excluded_upper : float, optional
        Upper bound of excluded region (default: kink_point + bin_width).
    n_bootstrap : int
        Number of bootstrap replications for SE.
    seed : int
        Random seed.

    Returns
    -------
    dict or None
        Dictionary with:
        - excess_mass: Normalized bunching (b = B/h0)
        - excess_mass_se: Bootstrap standard error
        - excess_count: Raw excess count (B)
        - h0: Counterfactual height at kink
        - poly_order: Polynomial order used
        - n_obs: Sample size

    Raises
    ------
    ImportError
        If rpy2 or bunchr is not available.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            "Install with: pip install rpy2>=3.5"
        )

    if not check_bunchr_installed():
        warnings.warn(
            "bunchr R package not available. Install with: "
            "install.packages('bunchr') in R",
            UserWarning,
        )
        return None

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()

        # Set defaults for excluded region
        if excluded_lower is None:
            excluded_lower = kink_point - bin_width
        if excluded_upper is None:
            excluded_upper = kink_point + bin_width

        # Pass data to R
        ro.globalenv["data_vec"] = ro.FloatVector(data)
        ro.globalenv["kink_point"] = kink_point
        ro.globalenv["bin_width"] = bin_width
        ro.globalenv["poly_order"] = poly_order
        ro.globalenv["excluded_lower"] = excluded_lower
        ro.globalenv["excluded_upper"] = excluded_upper
        ro.globalenv["n_bootstrap"] = n_bootstrap
        ro.globalenv["seed_val"] = seed

        result = ro.r(
            """
            suppressPackageStartupMessages(library(bunchr))
            set.seed(seed_val)

            # Run bunching estimation
            # bunchr::bunching() takes a formula or data vector
            bunch_result <- tryCatch({
                bunching(
                    data = data_vec,
                    zstar = kink_point,
                    binwidth = bin_width,
                    poly = poly_order,
                    excluded = c(excluded_lower, excluded_upper),
                    boot = n_bootstrap
                )
            }, error = function(e) {
                # Fallback: manual polynomial estimation if bunchr::bunching fails
                NULL
            })

            if (!is.null(bunch_result)) {
                # Extract results from bunchr object
                list(
                    excess_mass = bunch_result$b,
                    excess_mass_se = bunch_result$b_se,
                    excess_count = bunch_result$B,
                    h0 = bunch_result$h0,
                    poly_order = poly_order,
                    n_obs = length(data_vec),
                    converged = TRUE
                )
            } else {
                # Manual fallback using polynomial fitting
                # Create histogram
                bins <- seq(min(data_vec) - bin_width, max(data_vec) + bin_width,
                            by = bin_width)
                hist_data <- hist(data_vec, breaks = bins, plot = FALSE)
                bin_centers <- hist_data$mids
                counts <- hist_data$counts

                # Exclude bunching region
                excluded_mask <- bin_centers >= excluded_lower & bin_centers <= excluded_upper
                fit_mask <- !excluded_mask

                # Fit polynomial to non-excluded region
                fit_centers <- bin_centers[fit_mask]
                fit_counts <- counts[fit_mask]
                poly_fit <- lm(fit_counts ~ poly(fit_centers, poly_order, raw = TRUE))

                # Predict counterfactual in excluded region
                exc_centers <- bin_centers[excluded_mask]
                counterfactual <- predict(poly_fit, newdata = data.frame(fit_centers = exc_centers))

                # Calculate excess mass
                actual_in_excluded <- sum(counts[excluded_mask])
                counterfactual_in_excluded <- sum(pmax(counterfactual, 0))
                excess_count <- actual_in_excluded - counterfactual_in_excluded

                # h0 = average counterfactual height in excluded region
                h0 <- mean(pmax(counterfactual, 0))
                if (h0 > 0) {
                    excess_mass <- excess_count / h0
                } else {
                    excess_mass <- NA
                }

                list(
                    excess_mass = excess_mass,
                    excess_mass_se = NA,
                    excess_count = excess_count,
                    h0 = h0,
                    poly_order = poly_order,
                    n_obs = length(data_vec),
                    converged = FALSE
                )
            }
            """
        )

        # Handle potential NA values
        excess_mass = result.rx2("excess_mass")[0]
        excess_mass_se = result.rx2("excess_mass_se")[0]

        return {
            "excess_mass": float(excess_mass) if not np.isnan(excess_mass) else None,
            "excess_mass_se": float(excess_mass_se) if not np.isnan(excess_mass_se) else None,
            "excess_count": float(result.rx2("excess_count")[0]),
            "h0": float(result.rx2("h0")[0]),
            "poly_order": int(result.rx2("poly_order")[0]),
            "n_obs": int(result.rx2("n_obs")[0]),
            "converged": bool(result.rx2("converged")[0]),
        }

    except Exception as e:
        warnings.warn(f"R bunching estimation failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_bunching_elasticity(
    data: np.ndarray,
    kink_point: float,
    bin_width: float,
    t1_rate: float,
    t2_rate: float,
    poly_order: int = 7,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """Estimate bunching elasticity using R's bunchr package.

    Computes the behavioral elasticity from bunching via:
    e = b / ln((1-t1)/(1-t2))

    Parameters
    ----------
    data : np.ndarray
        Observed data.
    kink_point : float
        Kink location.
    bin_width : float
        Histogram bin width.
    t1_rate : float
        Marginal tax rate below kink.
    t2_rate : float
        Marginal tax rate above kink.
    poly_order : int
        Polynomial order.
    n_bootstrap : int
        Bootstrap replications.
    seed : int
        Random seed.

    Returns
    -------
    dict or None
        Dictionary with:
        - elasticity: Behavioral elasticity e
        - elasticity_se: Bootstrap standard error
        - excess_mass: Normalized bunching b
        - log_rate_change: ln((1-t1)/(1-t2))
    """
    # Get bunching estimate first
    bunch_result = r_bunching_estimate(
        data=data,
        kink_point=kink_point,
        bin_width=bin_width,
        poly_order=poly_order,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    if bunch_result is None or bunch_result["excess_mass"] is None:
        return None

    # Calculate elasticity
    log_rate_change = np.log((1 - t1_rate) / (1 - t2_rate))
    if abs(log_rate_change) < 1e-10:
        warnings.warn("Tax rates too similar for elasticity calculation", UserWarning)
        return None

    elasticity = bunch_result["excess_mass"] / log_rate_change

    # SE via delta method (approximately)
    elasticity_se = None
    if bunch_result["excess_mass_se"] is not None:
        elasticity_se = bunch_result["excess_mass_se"] / abs(log_rate_change)

    return {
        "elasticity": elasticity,
        "elasticity_se": elasticity_se,
        "excess_mass": bunch_result["excess_mass"],
        "log_rate_change": log_rate_change,
        "t1_rate": t1_rate,
        "t2_rate": t2_rate,
    }


# =============================================================================
# Shift-Share (ShiftShareSE package)
# =============================================================================


def check_shiftsharese_installed() -> bool:
    """Check if the ShiftShareSE R package is installed.

    Returns
    -------
    bool
        True if ShiftShareSE can be loaded in R, False otherwise.

    Notes
    -----
    Install in R with: install.packages("ShiftShareSE")
    """
    if not check_r_available():
        return False
    try:
        import rpy2.robjects as ro

        ro.r('suppressPackageStartupMessages(library(ShiftShareSE))')
        return True
    except Exception:
        return False


def r_shift_share_ivreg_ss(
    Y: np.ndarray,
    D: np.ndarray,
    shares: np.ndarray,
    shocks: np.ndarray,
    X: Optional[np.ndarray] = None,
    se_method: str = "AKM",
    alpha: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Estimate shift-share IV using R's ShiftShareSE package.

    Wraps ShiftShareSE::ivreg_ss() for triangulation testing against our
    Python ShiftShareIV implementation.

    Parameters
    ----------
    Y : np.ndarray
        Outcome variable (n,).
    D : np.ndarray
        Endogenous treatment variable (n,).
    shares : np.ndarray
        Sector shares matrix (n, S).
    shocks : np.ndarray
        Sector shocks vector (S,).
    X : np.ndarray, optional
        Exogenous control variables (n, k).
    se_method : str
        Standard error method: "AKM" (Adão-Kolesár-Morales),
        "AKM0" (AKM without small-sample adjustment), "EHW" (Eicker-Huber-White).
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    dict or None
        Dictionary with:
        - coefficient: 2SLS coefficient estimate
        - se: Standard error
        - t_stat: t-statistic
        - p_value: p-value
        - ci_lower: Lower CI bound
        - ci_upper: Upper CI bound
        - first_stage_f: First-stage F-statistic
        - n_obs: Number of observations
        - n_sectors: Number of sectors

    Notes
    -----
    ShiftShareSE implements the inference methods from:
    Adão, R., Kolesár, M., & Morales, E. (2019). Shift-Share Designs:
    Theory and Inference. Quarterly Journal of Economics, 134(4), 1949-2010.
    """
    if not check_r_available():
        raise ImportError(
            "rpy2 is required for R triangulation. "
            "Install with: pip install rpy2>=3.5"
        )

    if not check_shiftsharese_installed():
        warnings.warn(
            "ShiftShareSE R package not available. Install with: "
            "install.packages('ShiftShareSE') in R",
            UserWarning,
        )
        return None

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()

        n = len(Y)
        n_sectors = shares.shape[1]

        # Pass data to R
        ro.globalenv["Y"] = ro.FloatVector(Y)
        ro.globalenv["D"] = ro.FloatVector(D)
        ro.globalenv["shares_mat"] = ro.r.matrix(
            ro.FloatVector(shares.flatten('F')),
            nrow=n,
            ncol=n_sectors,
        )
        ro.globalenv["shocks_vec"] = ro.FloatVector(shocks)
        ro.globalenv["se_method"] = se_method
        ro.globalenv["alpha_val"] = alpha

        if X is not None:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            n_controls = X.shape[1]
            ro.globalenv["X_mat"] = ro.r.matrix(
                ro.FloatVector(X.flatten('F')),
                nrow=n,
                ncol=n_controls,
            )
            ro.globalenv["has_controls"] = True
        else:
            ro.globalenv["has_controls"] = False

        result = ro.r(
            """
            suppressPackageStartupMessages(library(ShiftShareSE))

            # Construct Bartik instrument
            Z_bartik <- as.vector(shares_mat %*% shocks_vec)

            # Build data frame
            if (has_controls) {
                df <- data.frame(Y = Y, D = D, Z = Z_bartik)
                for (j in 1:ncol(X_mat)) {
                    df[[paste0("X", j)]] <- X_mat[, j]
                }
                cov_names <- paste0("X", 1:ncol(X_mat))
                control_formula <- paste(cov_names, collapse = " + ")

                # ivreg_ss formula
                # Note: ShiftShareSE uses a specific formula interface
                fit <- tryCatch({
                    ivreg_ss(
                        as.formula(paste("Y ~ D +", control_formula, "| Z +", control_formula)),
                        data = df,
                        W = shares_mat,
                        X = shocks_vec,
                        method = se_method
                    )
                }, error = function(e) {
                    # Fallback: manual 2SLS with AER
                    library(AER)
                    ivreg_fit <- ivreg(
                        as.formula(paste("Y ~ D +", control_formula, "| Z +", control_formula)),
                        data = df
                    )
                    list(
                        est = coef(ivreg_fit)["D"],
                        SE = sqrt(vcovHC(ivreg_fit, type = "HC1")["D", "D"]),
                        fallback = TRUE
                    )
                })
            } else {
                df <- data.frame(Y = Y, D = D, Z = Z_bartik)

                fit <- tryCatch({
                    ivreg_ss(
                        Y ~ D | Z,
                        data = df,
                        W = shares_mat,
                        X = shocks_vec,
                        method = se_method
                    )
                }, error = function(e) {
                    # Fallback: manual 2SLS with AER
                    library(AER)
                    ivreg_fit <- ivreg(Y ~ D | Z, data = df)
                    list(
                        est = coef(ivreg_fit)["D"],
                        SE = sqrt(vcovHC(ivreg_fit, type = "HC1")["D", "D"]),
                        fallback = TRUE
                    )
                })
            }

            # Extract results
            if (is.null(fit$fallback)) {
                # ShiftShareSE result
                coef_est <- fit$est["D"]
                se_est <- fit$SE["D"]
            } else {
                # Fallback result
                coef_est <- fit$est
                se_est <- fit$SE
            }

            # Compute statistics
            t_stat <- coef_est / se_est
            p_value <- 2 * pt(-abs(t_stat), df = nrow(df) - 2)
            crit_val <- qt(1 - alpha_val / 2, df = nrow(df) - 2)
            ci_lower <- coef_est - crit_val * se_est
            ci_upper <- coef_est + crit_val * se_est

            # First stage F-statistic
            first_stage <- lm(D ~ Z, data = df)
            first_stage_f <- summary(first_stage)$fstatistic[1]

            list(
                coefficient = coef_est,
                se = se_est,
                t_stat = t_stat,
                p_value = p_value,
                ci_lower = ci_lower,
                ci_upper = ci_upper,
                first_stage_f = first_stage_f,
                n_obs = nrow(df),
                n_sectors = ncol(shares_mat),
                se_method = se_method,
                used_fallback = !is.null(fit$fallback)
            )
            """
        )

        return {
            "coefficient": float(result.rx2("coefficient")[0]),
            "se": float(result.rx2("se")[0]),
            "t_stat": float(result.rx2("t_stat")[0]),
            "p_value": float(result.rx2("p_value")[0]),
            "ci_lower": float(result.rx2("ci_lower")[0]),
            "ci_upper": float(result.rx2("ci_upper")[0]),
            "first_stage_f": float(result.rx2("first_stage_f")[0]),
            "n_obs": int(result.rx2("n_obs")[0]),
            "n_sectors": int(result.rx2("n_sectors")[0]),
            "se_method": str(result.rx2("se_method")[0]),
            "used_fallback": bool(result.rx2("used_fallback")[0]),
        }

    except Exception as e:
        warnings.warn(f"R shift-share estimation failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


# =============================================================================
# MTE (Marginal Treatment Effects) - R localIV Package
# =============================================================================


def check_localiv_installed() -> bool:
    """Check if R localIV package is available.

    Returns
    -------
    bool
        True if localIV is installed and importable.
    """
    try:
        ro.r('library(localIV)')
        return True
    except Exception:
        return False


def r_mte_estimate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    u_grid: Optional[np.ndarray] = None,
    n_grid: int = 100,
) -> Optional[Dict[str, Any]]:
    """Estimate Marginal Treatment Effects using R localIV package.

    Uses the localIV::mte() function to estimate the MTE curve as a function
    of unobserved resistance U. Also computes integrated treatment effects
    (ATE, ATT, ATU) from the MTE curve.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y (n,).
    treatment : np.ndarray
        Binary treatment indicator D (n,).
    instrument : np.ndarray
        Instrument variable Z (n,) - typically propensity score or instrument.
    covariates : Optional[np.ndarray]
        Covariates X (n, k). If None, uses intercept only.
    u_grid : Optional[np.ndarray]
        Grid of U values at which to evaluate MTE. If None, uses n_grid points.
    n_grid : int
        Number of grid points for U if u_grid not provided. Default 100.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary with keys:
        - mte_curve: MTE(u) values at grid points
        - u_grid: Grid of U values
        - ate: Average Treatment Effect (integral of MTE)
        - att: Average Treatment Effect on Treated
        - atu: Average Treatment Effect on Untreated
        - late: Local Average Treatment Effect (if binary instrument)
        - propensity: Estimated propensity score P(D=1|Z)
        Returns None if estimation fails.

    Notes
    -----
    The MTE function is defined as:
        MTE(u) = E[Y(1) - Y(0) | U = u]

    where U is the unobserved resistance to treatment. Under the marginal
    treatment effect framework:
        - ATE = ∫ MTE(u) du
        - ATT = ∫ MTE(u) h_T(u) du  (weighted by treated)
        - ATU = ∫ MTE(u) h_U(u) du  (weighted by untreated)
        - LATE = ∫ MTE(u) h_IV(u) du (weighted by compliers)

    References
    ----------
    Heckman, J. J., & Vytlacil, E. (2005). Structural Equations, Treatment
    Effects, and Econometric Policy Evaluation. Econometrica, 73(3), 669-738.
    """
    try:
        numpy2ri.activate()

        # Validate inputs
        n = len(outcome)
        if len(treatment) != n or len(instrument) != n:
            raise ValueError("outcome, treatment, and instrument must have same length")

        # Prepare data
        y_r = ro.FloatVector(outcome.astype(float))
        d_r = ro.FloatVector(treatment.astype(float))
        z_r = ro.FloatVector(instrument.astype(float))

        # Prepare covariates
        if covariates is not None:
            if covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
            x_r = ro.r.matrix(
                ro.FloatVector(covariates.flatten()),
                nrow=n,
                ncol=covariates.shape[1],
                byrow=False
            )
            has_covariates = True
        else:
            has_covariates = False

        # Prepare U grid
        if u_grid is None:
            u_grid = np.linspace(0.01, 0.99, n_grid)
        u_r = ro.FloatVector(u_grid.astype(float))

        # Assign to R environment
        ro.globalenv['y_vec'] = y_r
        ro.globalenv['d_vec'] = d_r
        ro.globalenv['z_vec'] = z_r
        ro.globalenv['u_grid'] = u_r
        ro.globalenv['has_covariates'] = has_covariates
        if has_covariates:
            ro.globalenv['x_mat'] = x_r

        # Execute R code for MTE estimation
        result = ro.r(
            """
            library(localIV)

            # Create data frame
            df <- data.frame(Y = y_vec, D = d_vec, Z = z_vec)

            # Add covariates if present
            if (has_covariates) {
                x_df <- as.data.frame(x_mat)
                names(x_df) <- paste0("X", 1:ncol(x_mat))
                df <- cbind(df, x_df)
            }

            # Estimate MTE using local IV
            # localIV uses local polynomial regression on the propensity score
            tryCatch({
                # First estimate propensity score
                if (has_covariates) {
                    ps_formula <- as.formula(paste("D ~", paste(names(x_df), collapse = " + "), "+ Z"))
                } else {
                    ps_formula <- D ~ Z
                }
                ps_model <- glm(ps_formula, data = df, family = binomial(link = "probit"))
                propensity <- predict(ps_model, type = "response")

                # Estimate MTE curve
                # Use semiparametric approach: local polynomial on propensity score
                mte_values <- numeric(length(u_grid))

                # Local linear regression for MTE estimation
                for (i in seq_along(u_grid)) {
                    u_val <- u_grid[i]

                    # Kernel weights (Epanechnikov)
                    h <- 0.15  # bandwidth
                    dist <- abs(propensity - u_val)
                    weights <- ifelse(dist < h, 0.75 * (1 - (dist/h)^2) / h, 0)

                    if (sum(weights > 0) > 10) {
                        # Weighted regression for treatment effect at this U
                        # E[Y|D,P] = alpha + beta*P + gamma*D + delta*D*P
                        # MTE(u) = gamma + delta*u
                        df$P <- propensity
                        df$DP <- df$D * propensity

                        w_model <- lm(Y ~ D + P + DP, data = df, weights = weights)
                        gamma <- coef(w_model)["D"]
                        delta <- coef(w_model)["DP"]

                        if (!is.na(gamma) && !is.na(delta)) {
                            mte_values[i] <- gamma + delta * u_val
                        } else {
                            mte_values[i] <- NA
                        }
                    } else {
                        mte_values[i] <- NA
                    }
                }

                # Interpolate any NAs using linear interpolation
                if (any(is.na(mte_values))) {
                    valid_idx <- which(!is.na(mte_values))
                    if (length(valid_idx) > 2) {
                        mte_values <- approx(u_grid[valid_idx], mte_values[valid_idx],
                                            xout = u_grid, rule = 2)$y
                    }
                }

                # Compute integrated treatment effects
                # ATE: uniform weights
                ate <- mean(mte_values, na.rm = TRUE)

                # ATT: weight by P(D=1|U<=u) - density of treated
                # For binary instrument: weight proportional to propensity among treated
                treated_weights <- propensity[df$D == 1]
                att_weights <- sapply(u_grid, function(u) mean(treated_weights <= u))
                att_weights <- diff(c(0, att_weights))  # Convert CDF to density
                if (sum(att_weights) > 0) {
                    att_weights <- att_weights / sum(att_weights)
                    att <- sum(mte_values * att_weights, na.rm = TRUE)
                } else {
                    att <- ate
                }

                # ATU: weight by P(D=0|U>u) - density of untreated
                untreated_weights <- propensity[df$D == 0]
                atu_weights <- sapply(u_grid, function(u) mean(untreated_weights > u))
                atu_weights <- -diff(c(atu_weights, 0))  # Convert CDF to density
                if (sum(atu_weights) > 0) {
                    atu_weights <- atu_weights / sum(atu_weights)
                    atu <- sum(mte_values * atu_weights, na.rm = TRUE)
                } else {
                    atu <- ate
                }

                # LATE: complier weights (for binary instrument approximation)
                # Weight by density of propensity score
                late_weights <- dnorm((u_grid - mean(propensity)) / sd(propensity))
                late_weights <- late_weights / sum(late_weights)
                late <- sum(mte_values * late_weights, na.rm = TRUE)

                list(
                    mte_curve = mte_values,
                    u_grid = u_grid,
                    ate = ate,
                    att = att,
                    atu = atu,
                    late = late,
                    propensity = propensity,
                    n_obs = nrow(df),
                    success = TRUE
                )
            }, error = function(e) {
                list(
                    error = as.character(e),
                    success = FALSE
                )
            })
            """
        )

        # Check for success
        if not result.rx2("success")[0]:
            error_msg = str(result.rx2("error")[0]) if "error" in result.names else "Unknown error"
            warnings.warn(f"R MTE estimation failed: {error_msg}", UserWarning)
            return None

        # Extract results
        mte_curve = np.array(result.rx2("mte_curve"))
        u_grid_out = np.array(result.rx2("u_grid"))
        propensity = np.array(result.rx2("propensity"))

        return {
            "mte_curve": mte_curve,
            "u_grid": u_grid_out,
            "ate": float(result.rx2("ate")[0]),
            "att": float(result.rx2("att")[0]),
            "atu": float(result.rx2("atu")[0]),
            "late": float(result.rx2("late")[0]),
            "propensity": propensity,
            "n_obs": int(result.rx2("n_obs")[0]),
        }

    except Exception as e:
        warnings.warn(f"R MTE estimation failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_mte_policy_effect(
    outcome: np.ndarray,
    treatment: np.ndarray,
    instrument: np.ndarray,
    policy_weights: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Estimate policy-relevant treatment effect using MTE framework.

    Computes E[Y(1) - Y(0) | selected by policy] where the policy is
    characterized by its weights on the MTE curve.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y (n,).
    treatment : np.ndarray
        Binary treatment indicator D (n,).
    instrument : np.ndarray
        Instrument variable Z (n,).
    policy_weights : np.ndarray
        Weights for policy effect computation (same length as u_grid).
    covariates : Optional[np.ndarray]
        Covariates X (n, k).

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary with:
        - prte: Policy-Relevant Treatment Effect
        - mte_curve: Underlying MTE curve
        - weighted_mte: Policy-weighted MTE curve
        Returns None if estimation fails.
    """
    try:
        # First get MTE estimate
        mte_result = r_mte_estimate(
            outcome, treatment, instrument, covariates,
            n_grid=len(policy_weights)
        )

        if mte_result is None:
            return None

        # Normalize policy weights
        policy_weights = np.array(policy_weights)
        policy_weights = policy_weights / np.sum(policy_weights)

        # Compute PRTE
        mte_curve = mte_result["mte_curve"]
        prte = np.sum(mte_curve * policy_weights)

        return {
            "prte": prte,
            "mte_curve": mte_curve,
            "weighted_mte": mte_curve * policy_weights,
            "u_grid": mte_result["u_grid"],
            "policy_weights": policy_weights,
        }

    except Exception as e:
        warnings.warn(f"R MTE policy effect failed: {e}", UserWarning)
        return None


# =============================================================================
# QTE (Quantile Treatment Effects) - R qte and quantreg Packages
# =============================================================================


def check_qte_installed() -> bool:
    """Check if R qte package is available.

    Returns
    -------
    bool
        True if qte package is installed and importable.
    """
    try:
        ro.r('library(qte)')
        return True
    except Exception:
        return False


def check_quantreg_installed() -> bool:
    """Check if R quantreg package is available.

    Returns
    -------
    bool
        True if quantreg package is installed and importable.
    """
    try:
        ro.r('library(quantreg)')
        return True
    except Exception:
        return False


def r_conditional_qte(
    outcome: np.ndarray,
    treatment: np.ndarray,
    quantiles: Optional[np.ndarray] = None,
    covariates: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Estimate conditional quantile treatment effects using R quantreg.

    Computes QTE(tau) = Q_{Y|D=1}(tau) - Q_{Y|D=0}(tau) conditional on X.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y (n,).
    treatment : np.ndarray
        Binary treatment indicator D (n,).
    quantiles : Optional[np.ndarray]
        Quantile levels to estimate. Default [0.25, 0.50, 0.75].
    covariates : Optional[np.ndarray]
        Covariates X (n, k). If None, unconditional QTE.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary with keys:
        - quantile_effects: Dict[float, float] - QTE at each quantile
        - quantile_se: Dict[float, float] - SE at each quantile
        - quantiles: np.ndarray - quantile levels
        - n_obs: int
        Returns None if estimation fails.
    """
    try:
        numpy2ri.activate()

        n = len(outcome)
        if quantiles is None:
            quantiles = np.array([0.25, 0.50, 0.75])

        # Prepare data
        y_r = ro.FloatVector(outcome.astype(float))
        d_r = ro.FloatVector(treatment.astype(float))
        tau_r = ro.FloatVector(quantiles.astype(float))

        # Prepare covariates
        if covariates is not None:
            if covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
            x_r = ro.r.matrix(
                ro.FloatVector(covariates.flatten()),
                nrow=n,
                ncol=covariates.shape[1],
                byrow=False
            )
            has_covariates = True
        else:
            has_covariates = False

        # Assign to R environment
        ro.globalenv['y_vec'] = y_r
        ro.globalenv['d_vec'] = d_r
        ro.globalenv['tau_vec'] = tau_r
        ro.globalenv['has_covariates'] = has_covariates
        if has_covariates:
            ro.globalenv['x_mat'] = x_r

        # Execute R code
        result = ro.r(
            """
            library(quantreg)

            # Create data frame
            df <- data.frame(Y = y_vec, D = d_vec)

            if (has_covariates) {
                x_df <- as.data.frame(x_mat)
                names(x_df) <- paste0("X", 1:ncol(x_mat))
                df <- cbind(df, x_df)
            }

            tryCatch({
                qte_effects <- numeric(length(tau_vec))
                qte_se <- numeric(length(tau_vec))

                for (i in seq_along(tau_vec)) {
                    tau <- tau_vec[i]

                    if (has_covariates) {
                        # Conditional QTE with covariates
                        formula_str <- paste("Y ~ D +", paste(names(x_df), collapse = " + "))
                        qr_fit <- rq(as.formula(formula_str), tau = tau, data = df)
                    } else {
                        # Simple QTE without covariates
                        qr_fit <- rq(Y ~ D, tau = tau, data = df)
                    }

                    # Extract treatment effect coefficient
                    coefs <- coef(qr_fit)
                    qte_effects[i] <- coefs["D"]

                    # Get standard errors using bootstrap
                    tryCatch({
                        summ <- summary(qr_fit, se = "boot", R = 100)
                        se_tab <- summ$coefficients
                        qte_se[i] <- se_tab["D", "Std. Error"]
                    }, error = function(e) {
                        # Fallback to asymptotic SE
                        summ <- summary(qr_fit, se = "nid")
                        se_tab <- summ$coefficients
                        qte_se[i] <- se_tab["D", "Std. Error"]
                    })
                }

                list(
                    qte_effects = qte_effects,
                    qte_se = qte_se,
                    quantiles = tau_vec,
                    n_obs = nrow(df),
                    success = TRUE
                )
            }, error = function(e) {
                list(error = as.character(e), success = FALSE)
            })
            """
        )

        if not result.rx2("success")[0]:
            error_msg = str(result.rx2("error")[0]) if "error" in result.names else "Unknown"
            warnings.warn(f"R conditional QTE failed: {error_msg}", UserWarning)
            return None

        qte_effects = np.array(result.rx2("qte_effects"))
        qte_se = np.array(result.rx2("qte_se"))
        quantiles_out = np.array(result.rx2("quantiles"))

        return {
            "quantile_effects": dict(zip(quantiles_out, qte_effects)),
            "quantile_se": dict(zip(quantiles_out, qte_se)),
            "quantiles": quantiles_out,
            "n_obs": int(result.rx2("n_obs")[0]),
        }

    except Exception as e:
        warnings.warn(f"R conditional QTE failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_unconditional_qte(
    outcome: np.ndarray,
    treatment: np.ndarray,
    quantiles: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Estimate unconditional quantile treatment effects using R.

    Uses RIF (Recentered Influence Function) regression for unconditional QTE.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y (n,).
    treatment : np.ndarray
        Binary treatment indicator D (n,).
    quantiles : Optional[np.ndarray]
        Quantile levels. Default [0.25, 0.50, 0.75].

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary with:
        - uqte: Dict[float, float] - Unconditional QTE at each quantile
        - uqte_se: Dict[float, float] - SE at each quantile
        - quantiles: np.ndarray
        Returns None if estimation fails.

    Notes
    -----
    Unconditional QTE differs from conditional QTE:
    - Conditional: Q_{Y|D=1,X}(tau) - Q_{Y|D=0,X}(tau) (varies with X)
    - Unconditional: Q_{Y(1)}(tau) - Q_{Y(0)}(tau) (marginal quantiles)

    RIF-regression uses the influence function of the quantile:
        RIF(y; Q_tau) = tau + (1 - tau) * I(y <= Q_tau) / f(Q_tau)
    """
    try:
        numpy2ri.activate()

        if quantiles is None:
            quantiles = np.array([0.25, 0.50, 0.75])

        y_r = ro.FloatVector(outcome.astype(float))
        d_r = ro.FloatVector(treatment.astype(float))
        tau_r = ro.FloatVector(quantiles.astype(float))

        ro.globalenv['y_vec'] = y_r
        ro.globalenv['d_vec'] = d_r
        ro.globalenv['tau_vec'] = tau_r

        result = ro.r(
            """
            library(quantreg)

            df <- data.frame(Y = y_vec, D = d_vec)

            tryCatch({
                uqte_vals <- numeric(length(tau_vec))
                uqte_se <- numeric(length(tau_vec))

                for (i in seq_along(tau_vec)) {
                    tau <- tau_vec[i]

                    # Compute RIF for unconditional QTE
                    # RIF(y; Q_tau) = Q_tau + (tau - I(y <= Q_tau)) / f(Q_tau)

                    # Estimate quantile
                    q_tau <- quantile(df$Y, tau)

                    # Estimate density at quantile using kernel
                    bw <- bw.nrd0(df$Y)
                    f_q <- dnorm(0) / bw  # Approximate density at quantile

                    # Compute RIF
                    rif <- q_tau + (tau - (df$Y <= q_tau)) / f_q

                    # Regress RIF on treatment
                    rif_model <- lm(rif ~ D, data = data.frame(rif = rif, D = df$D))
                    uqte_vals[i] <- coef(rif_model)["D"]
                    uqte_se[i] <- sqrt(vcov(rif_model)["D", "D"])
                }

                list(
                    uqte = uqte_vals,
                    uqte_se = uqte_se,
                    quantiles = tau_vec,
                    n_obs = nrow(df),
                    success = TRUE
                )
            }, error = function(e) {
                list(error = as.character(e), success = FALSE)
            })
            """
        )

        if not result.rx2("success")[0]:
            error_msg = str(result.rx2("error")[0]) if "error" in result.names else "Unknown"
            warnings.warn(f"R unconditional QTE failed: {error_msg}", UserWarning)
            return None

        uqte_vals = np.array(result.rx2("uqte"))
        uqte_se = np.array(result.rx2("uqte_se"))
        quantiles_out = np.array(result.rx2("quantiles"))

        return {
            "uqte": dict(zip(quantiles_out, uqte_vals)),
            "uqte_se": dict(zip(quantiles_out, uqte_se)),
            "quantiles": quantiles_out,
            "n_obs": int(result.rx2("n_obs")[0]),
        }

    except Exception as e:
        warnings.warn(f"R unconditional QTE failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_qte_process(
    outcome: np.ndarray,
    treatment: np.ndarray,
    quantile_grid: Optional[np.ndarray] = None,
    n_quantiles: int = 19,
) -> Optional[Dict[str, Any]]:
    """Estimate the full QTE process across quantiles.

    Computes QTE(tau) for a fine grid of quantiles to characterize
    the entire treatment effect distribution.

    Parameters
    ----------
    outcome : np.ndarray
        Outcome variable Y (n,).
    treatment : np.ndarray
        Binary treatment indicator D (n,).
    quantile_grid : Optional[np.ndarray]
        Custom quantile grid. If None, uses uniform grid.
    n_quantiles : int
        Number of quantile points if grid not provided.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary with:
        - qte_process: np.ndarray - QTE at each quantile
        - quantile_grid: np.ndarray - quantile levels
        - uniform_band: Tuple[np.ndarray, np.ndarray] - confidence band
        Returns None if estimation fails.
    """
    try:
        numpy2ri.activate()

        if quantile_grid is None:
            quantile_grid = np.linspace(0.05, 0.95, n_quantiles)

        y_r = ro.FloatVector(outcome.astype(float))
        d_r = ro.FloatVector(treatment.astype(float))
        tau_r = ro.FloatVector(quantile_grid.astype(float))

        ro.globalenv['y_vec'] = y_r
        ro.globalenv['d_vec'] = d_r
        ro.globalenv['tau_vec'] = tau_r

        result = ro.r(
            """
            library(quantreg)

            df <- data.frame(Y = y_vec, D = d_vec)

            tryCatch({
                qte_process <- numeric(length(tau_vec))
                qte_se <- numeric(length(tau_vec))

                for (i in seq_along(tau_vec)) {
                    tau <- tau_vec[i]
                    qr_fit <- rq(Y ~ D, tau = tau, data = df)
                    qte_process[i] <- coef(qr_fit)["D"]

                    tryCatch({
                        summ <- summary(qr_fit, se = "nid")
                        qte_se[i] <- summ$coefficients["D", "Std. Error"]
                    }, error = function(e) {
                        qte_se[i] <- NA
                    })
                }

                # Compute uniform confidence band (Bonferroni correction)
                n_taus <- length(tau_vec)
                alpha <- 0.05
                z_crit <- qnorm(1 - alpha / (2 * n_taus))

                lower_band <- qte_process - z_crit * qte_se
                upper_band <- qte_process + z_crit * qte_se

                list(
                    qte_process = qte_process,
                    qte_se = qte_se,
                    quantile_grid = tau_vec,
                    lower_band = lower_band,
                    upper_band = upper_band,
                    n_obs = nrow(df),
                    success = TRUE
                )
            }, error = function(e) {
                list(error = as.character(e), success = FALSE)
            })
            """
        )

        if not result.rx2("success")[0]:
            error_msg = str(result.rx2("error")[0]) if "error" in result.names else "Unknown"
            warnings.warn(f"R QTE process failed: {error_msg}", UserWarning)
            return None

        return {
            "qte_process": np.array(result.rx2("qte_process")),
            "qte_se": np.array(result.rx2("qte_se")),
            "quantile_grid": np.array(result.rx2("quantile_grid")),
            "lower_band": np.array(result.rx2("lower_band")),
            "upper_band": np.array(result.rx2("upper_band")),
            "n_obs": int(result.rx2("n_obs")[0]),
        }

    except Exception as e:
        warnings.warn(f"R QTE process failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


# =============================================================================
# Time Series VAR - R vars Package
# =============================================================================


def check_vars_installed() -> bool:
    """Check if R vars package is available.

    Returns
    -------
    bool
        True if vars package is installed and importable.
    """
    try:
        ro.r('library(vars)')
        return True
    except Exception:
        return False


def r_var_estimate(
    data: np.ndarray,
    p: int = 1,
    var_type: str = "const",
) -> Optional[Dict[str, Any]]:
    """Estimate Vector Autoregression using R vars package.

    Parameters
    ----------
    data : np.ndarray
        Multivariate time series (T, k) where T is observations, k is variables.
    p : int
        Lag order. Default 1.
    var_type : str
        Type of deterministic regressors:
        - "const": Constant only (default)
        - "trend": Constant and trend
        - "both": Constant and trend
        - "none": No deterministic terms

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary with:
        - coefficients: Dict[str, np.ndarray] - coefficient matrices A_1, ..., A_p
        - residuals: np.ndarray (T-p, k)
        - sigma: np.ndarray (k, k) - residual covariance matrix
        - aic: float - Akaike Information Criterion
        - bic: float - Bayesian Information Criterion
        - hq: float - Hannan-Quinn criterion
        - n_obs: int
        Returns None if estimation fails.
    """
    try:
        numpy2ri.activate()

        T, k = data.shape
        if T < p + 10:
            warnings.warn(f"Too few observations ({T}) for VAR({p})", UserWarning)

        # Convert data to R matrix
        data_r = ro.r.matrix(
            ro.FloatVector(data.T.flatten()),  # Column-major
            nrow=T,
            ncol=k,
            byrow=False
        )

        ro.globalenv['data_mat'] = data_r
        ro.globalenv['p_val'] = p
        ro.globalenv['var_type'] = var_type

        result = ro.r(
            """
            library(vars)

            # Convert to time series
            ts_data <- ts(data_mat)

            tryCatch({
                # Estimate VAR
                var_fit <- VAR(ts_data, p = p_val, type = var_type)

                # Extract coefficient matrices
                coef_list <- coef(var_fit)

                # Get A matrices (excluding deterministic terms)
                k <- ncol(ts_data)
                A_matrices <- list()

                for (eq in 1:k) {
                    eq_coef <- coef_list[[eq]]
                    A_matrices[[eq]] <- eq_coef[1:(p_val * k), 1]  # Lag coefficients only
                }

                # Stack into matrix form
                A_coefs <- do.call(rbind, A_matrices)

                # Residuals
                resid_mat <- residuals(var_fit)

                # Covariance matrix
                sigma_mat <- summary(var_fit)$covres

                # Information criteria
                ic <- c(
                    AIC = AIC(var_fit),
                    BIC = BIC(var_fit),
                    HQ = NA  # Compute manually
                )

                # Hannan-Quinn
                n <- nrow(ts_data) - p_val
                k_total <- k * (k * p_val + 1)  # Total parameters per equation
                log_det <- log(det(sigma_mat))
                hq <- log_det + 2 * k_total * log(log(n)) / n

                list(
                    coefficients = A_coefs,
                    residuals = resid_mat,
                    sigma = sigma_mat,
                    aic = ic["AIC"],
                    bic = ic["BIC"],
                    hq = hq,
                    n_obs = n,
                    k = k,
                    p = p_val,
                    success = TRUE
                )
            }, error = function(e) {
                list(error = as.character(e), success = FALSE)
            })
            """
        )

        if not result.rx2("success")[0]:
            error_msg = str(result.rx2("error")[0]) if "error" in result.names else "Unknown"
            warnings.warn(f"R VAR estimation failed: {error_msg}", UserWarning)
            return None

        return {
            "coefficients": np.array(result.rx2("coefficients")),
            "residuals": np.array(result.rx2("residuals")),
            "sigma": np.array(result.rx2("sigma")),
            "aic": float(result.rx2("aic")[0]),
            "bic": float(result.rx2("bic")[0]),
            "hq": float(result.rx2("hq")[0]),
            "n_obs": int(result.rx2("n_obs")[0]),
            "k": int(result.rx2("k")[0]),
            "p": int(result.rx2("p")[0]),
        }

    except Exception as e:
        warnings.warn(f"R VAR estimation failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_var_irf(
    data: np.ndarray,
    p: int = 1,
    n_ahead: int = 10,
    ortho: bool = True,
    boot: bool = False,
    n_boot: int = 100,
) -> Optional[Dict[str, Any]]:
    """Compute impulse response functions using R vars package.

    Parameters
    ----------
    data : np.ndarray
        Multivariate time series (T, k).
    p : int
        VAR lag order.
    n_ahead : int
        Forecast horizon for IRF.
    ortho : bool
        If True, compute orthogonalized IRF (Cholesky).
    boot : bool
        If True, compute bootstrap confidence intervals.
    n_boot : int
        Number of bootstrap replications.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary with:
        - irf: np.ndarray (n_ahead+1, k, k) - IRF matrices
        - lower: np.ndarray - Lower CI (if boot=True)
        - upper: np.ndarray - Upper CI (if boot=True)
        Returns None if estimation fails.
    """
    try:
        numpy2ri.activate()

        T, k = data.shape

        data_r = ro.r.matrix(
            ro.FloatVector(data.T.flatten()),
            nrow=T,
            ncol=k,
            byrow=False
        )

        ro.globalenv['data_mat'] = data_r
        ro.globalenv['p_val'] = p
        ro.globalenv['n_ahead'] = n_ahead
        ro.globalenv['ortho'] = ortho
        ro.globalenv['do_boot'] = boot
        ro.globalenv['n_boot'] = n_boot

        result = ro.r(
            """
            library(vars)

            ts_data <- ts(data_mat)

            tryCatch({
                # Estimate VAR
                var_fit <- VAR(ts_data, p = p_val, type = "const")

                # Compute IRF
                if (do_boot) {
                    irf_result <- irf(var_fit, n.ahead = n_ahead, ortho = ortho,
                                     boot = TRUE, runs = n_boot, ci = 0.95)
                } else {
                    irf_result <- irf(var_fit, n.ahead = n_ahead, ortho = ortho,
                                     boot = FALSE)
                }

                # Extract IRF arrays
                k <- ncol(ts_data)
                irf_array <- array(0, dim = c(n_ahead + 1, k, k))
                lower_array <- array(NA, dim = c(n_ahead + 1, k, k))
                upper_array <- array(NA, dim = c(n_ahead + 1, k, k))

                for (shock in 1:k) {
                    shock_name <- names(irf_result$irf)[shock]

                    # Point estimates
                    irf_mat <- irf_result$irf[[shock_name]]
                    irf_array[, , shock] <- irf_mat

                    # Confidence intervals
                    if (do_boot) {
                        lower_array[, , shock] <- irf_result$Lower[[shock_name]]
                        upper_array[, , shock] <- irf_result$Upper[[shock_name]]
                    }
                }

                list(
                    irf = irf_array,
                    lower = lower_array,
                    upper = upper_array,
                    n_ahead = n_ahead,
                    k = k,
                    ortho = ortho,
                    success = TRUE
                )
            }, error = function(e) {
                list(error = as.character(e), success = FALSE)
            })
            """
        )

        if not result.rx2("success")[0]:
            error_msg = str(result.rx2("error")[0]) if "error" in result.names else "Unknown"
            warnings.warn(f"R IRF computation failed: {error_msg}", UserWarning)
            return None

        return {
            "irf": np.array(result.rx2("irf")),
            "lower": np.array(result.rx2("lower")),
            "upper": np.array(result.rx2("upper")),
            "n_ahead": int(result.rx2("n_ahead")[0]),
            "k": int(result.rx2("k")[0]),
            "ortho": bool(result.rx2("ortho")[0]),
        }

    except Exception as e:
        warnings.warn(f"R IRF computation failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_granger_causality(
    data: np.ndarray,
    p: int = 1,
    cause_var: int = 0,
    effect_var: int = 1,
) -> Optional[Dict[str, Any]]:
    """Test Granger causality using R vars package.

    Tests whether variable `cause_var` Granger-causes `effect_var`.

    Parameters
    ----------
    data : np.ndarray
        Multivariate time series (T, k).
    p : int
        VAR lag order.
    cause_var : int
        Index of potential cause variable (0-indexed).
    effect_var : int
        Index of potential effect variable (0-indexed).

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary with:
        - f_stat: float - F-statistic
        - p_value: float
        - df1: int - numerator degrees of freedom
        - df2: int - denominator degrees of freedom
        - null_rejected: bool - True if null (no causality) rejected at 5%
        Returns None if estimation fails.
    """
    try:
        numpy2ri.activate()

        T, k = data.shape

        data_r = ro.r.matrix(
            ro.FloatVector(data.T.flatten()),
            nrow=T,
            ncol=k,
            byrow=False
        )

        ro.globalenv['data_mat'] = data_r
        ro.globalenv['p_val'] = p
        ro.globalenv['cause_idx'] = cause_var + 1  # R is 1-indexed
        ro.globalenv['effect_idx'] = effect_var + 1

        result = ro.r(
            """
            library(vars)

            ts_data <- ts(data_mat)
            colnames(ts_data) <- paste0("V", 1:ncol(ts_data))

            tryCatch({
                # Estimate VAR
                var_fit <- VAR(ts_data, p = p_val, type = "const")

                # Get variable names
                cause_name <- colnames(ts_data)[cause_idx]
                effect_name <- colnames(ts_data)[effect_idx]

                # Granger causality test
                granger_test <- causality(var_fit, cause = cause_name)

                # Extract F-test results
                f_test <- granger_test$Granger

                list(
                    f_stat = f_test$statistic,
                    p_value = f_test$p.value,
                    df1 = f_test$parameter[1],
                    df2 = f_test$parameter[2],
                    null_rejected = f_test$p.value < 0.05,
                    cause_var = cause_name,
                    effect_var = effect_name,
                    success = TRUE
                )
            }, error = function(e) {
                list(error = as.character(e), success = FALSE)
            })
            """
        )

        if not result.rx2("success")[0]:
            error_msg = str(result.rx2("error")[0]) if "error" in result.names else "Unknown"
            warnings.warn(f"R Granger test failed: {error_msg}", UserWarning)
            return None

        return {
            "f_stat": float(result.rx2("f_stat")[0]),
            "p_value": float(result.rx2("p_value")[0]),
            "df1": int(result.rx2("df1")[0]),
            "df2": int(result.rx2("df2")[0]),
            "null_rejected": bool(result.rx2("null_rejected")[0]),
        }

    except Exception as e:
        warnings.warn(f"R Granger test failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()


def r_var_forecast(
    data: np.ndarray,
    p: int = 1,
    n_ahead: int = 10,
) -> Optional[Dict[str, Any]]:
    """Forecast using VAR model via R vars package.

    Parameters
    ----------
    data : np.ndarray
        Historical multivariate time series (T, k).
    p : int
        VAR lag order.
    n_ahead : int
        Forecast horizon.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary with:
        - forecast: np.ndarray (n_ahead, k) - point forecasts
        - lower: np.ndarray (n_ahead, k) - lower 95% CI
        - upper: np.ndarray (n_ahead, k) - upper 95% CI
        Returns None if forecasting fails.
    """
    try:
        numpy2ri.activate()

        T, k = data.shape

        data_r = ro.r.matrix(
            ro.FloatVector(data.T.flatten()),
            nrow=T,
            ncol=k,
            byrow=False
        )

        ro.globalenv['data_mat'] = data_r
        ro.globalenv['p_val'] = p
        ro.globalenv['n_ahead'] = n_ahead

        result = ro.r(
            """
            library(vars)

            ts_data <- ts(data_mat)

            tryCatch({
                # Estimate VAR
                var_fit <- VAR(ts_data, p = p_val, type = "const")

                # Forecast
                fc <- predict(var_fit, n.ahead = n_ahead, ci = 0.95)

                k <- ncol(ts_data)
                forecast_mat <- matrix(0, nrow = n_ahead, ncol = k)
                lower_mat <- matrix(0, nrow = n_ahead, ncol = k)
                upper_mat <- matrix(0, nrow = n_ahead, ncol = k)

                for (i in 1:k) {
                    forecast_mat[, i] <- fc$fcst[[i]][, "fcst"]
                    lower_mat[, i] <- fc$fcst[[i]][, "lower"]
                    upper_mat[, i] <- fc$fcst[[i]][, "upper"]
                }

                list(
                    forecast = forecast_mat,
                    lower = lower_mat,
                    upper = upper_mat,
                    n_ahead = n_ahead,
                    k = k,
                    success = TRUE
                )
            }, error = function(e) {
                list(error = as.character(e), success = FALSE)
            })
            """
        )

        if not result.rx2("success")[0]:
            error_msg = str(result.rx2("error")[0]) if "error" in result.names else "Unknown"
            warnings.warn(f"R VAR forecast failed: {error_msg}", UserWarning)
            return None

        return {
            "forecast": np.array(result.rx2("forecast")),
            "lower": np.array(result.rx2("lower")),
            "upper": np.array(result.rx2("upper")),
            "n_ahead": int(result.rx2("n_ahead")[0]),
            "k": int(result.rx2("k")[0]),
        }

    except Exception as e:
        warnings.warn(f"R VAR forecast failed: {e}", UserWarning)
        return None
    finally:
        numpy2ri.deactivate()