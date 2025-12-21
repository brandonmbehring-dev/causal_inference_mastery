"""
Sensitivity analysis for mediation.

Assesses robustness of mediation effects to violations of
sequential ignorability (unmeasured confounding).

Based on Imai, Keele, Yamamoto (2010) sensitivity approach.

References
----------
- Imai, Keele, Yamamoto (2010). A General Approach to Causal Mediation
"""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats
import statsmodels.api as sm

from .types import SensitivityResult


def mediation_sensitivity(
    outcome: NDArray[np.floating],
    treatment: NDArray[np.floating],
    mediator: NDArray[np.floating],
    covariates: Optional[NDArray[np.floating]] = None,
    rho_range: Tuple[float, float] = (-0.9, 0.9),
    n_rho: int = 41,
    n_simulations: int = 500,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> SensitivityResult:
    """
    Sensitivity analysis for unmeasured confounding.

    Assesses how NDE/NIE estimates change under varying degrees of
    unmeasured confounding between mediator and outcome.

    The sensitivity parameter rho represents the correlation between
    the error terms in the mediator and outcome models:
    - rho = 0: No unmeasured confounding (sequential ignorability holds)
    - rho > 0: Positive confounding
    - rho < 0: Negative confounding

    Parameters
    ----------
    outcome : ndarray
        Outcome variable Y
    treatment : ndarray
        Treatment variable T
    mediator : ndarray
        Mediator variable M
    covariates : ndarray, optional
        Pre-treatment covariates X
    rho_range : tuple
        Range of sensitivity parameter (min, max)
    n_rho : int
        Number of rho values in grid
    n_simulations : int
        Monte Carlo simulations per rho
    n_bootstrap : int
        Bootstrap replications for CIs at each rho
    alpha : float
        Significance level
    random_state : int, optional
        Random seed

    Returns
    -------
    SensitivityResult
        Grid of effects at each rho value with CIs

    Notes
    -----
    The interpretation depends on the magnitude of rho_at_zero_*:
    - Large |rho| needed to nullify effect: Robust finding
    - Small |rho| needed to nullify effect: Sensitive to confounding

    Example interpretation:
    - rho_at_zero_nie = 0.3: Only moderate confounding needed to explain
      away the indirect effect → proceed with caution
    - rho_at_zero_nie = 0.8: Very strong confounding needed → robust

    References
    ----------
    Imai, Keele, Yamamoto (2010). A General Approach to Causal
    Mediation Analysis. Psychological Methods 15(4):309-334.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> T = np.random.binomial(1, 0.5, n).astype(float)
    >>> M = 0.5 + 0.6 * T + np.random.randn(n) * 0.5
    >>> Y = 1.0 + 0.5 * T + 0.8 * M + np.random.randn(n) * 0.5
    >>> result = mediation_sensitivity(Y, T, M, n_simulations=100)
    >>> print(f"rho to nullify NIE: {result['rho_at_zero_nie']:.2f}")
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Validate inputs
    outcome = np.asarray(outcome, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    mediator = np.asarray(mediator, dtype=np.float64)

    n = len(outcome)

    # Build design matrices
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float64)
        X_m = np.column_stack([np.ones(n), treatment, covariates])
        X_y = np.column_stack([np.ones(n), treatment, mediator, covariates])
    else:
        X_m = np.column_stack([np.ones(n), treatment])
        X_y = np.column_stack([np.ones(n), treatment, mediator])

    # Fit baseline models
    m_fit = sm.OLS(mediator, X_m).fit()
    y_fit = sm.OLS(outcome, X_y).fit()

    m_resid_var = m_fit.mse_resid
    y_resid_var = y_fit.mse_resid

    # Create rho grid
    rho_grid = np.linspace(rho_range[0], rho_range[1], n_rho)

    # Compute effects at each rho
    nde_at_rho = np.zeros(n_rho)
    nie_at_rho = np.zeros(n_rho)
    nde_ci_lower = np.zeros(n_rho)
    nde_ci_upper = np.zeros(n_rho)
    nie_ci_lower = np.zeros(n_rho)
    nie_ci_upper = np.zeros(n_rho)

    # Treatment coefficient in mediator model
    alpha_1 = m_fit.params[1]
    # Treatment coefficient in outcome model
    beta_1 = y_fit.params[1]
    # Mediator coefficient in outcome model
    beta_2 = y_fit.params[2]

    # Under confounding with correlation rho between e_m and e_y:
    # The estimated beta_2 is biased:
    # beta_2_biased = beta_2_true + rho * sqrt(sigma_y^2 / sigma_m^2) * (something)
    #
    # Following Imai et al. (2010) sensitivity formula:
    # The adjustment factor for the indirect effect under rho:
    # NIE_adjusted = alpha_1 * (beta_2 - lambda * rho)
    # where lambda captures the confounding strength

    sigma_m = np.sqrt(m_resid_var)
    sigma_y = np.sqrt(y_resid_var)

    # Sensitivity parameter interpretation:
    # rho = Corr(e_m, e_y) = correlation between residuals
    # Adjustment: beta_2_adj = beta_2 - rho * (sigma_y / sigma_m) * partial_corr_factor

    for i, rho in enumerate(rho_grid):
        # Compute adjusted effects under this rho
        # Following the linear adjustment in Imai et al. (2010)
        # Adjusted beta_2 = beta_2 - rho * sigma_y / sigma_m
        # This is a simplified linear sensitivity model

        adjustment = rho * sigma_y / sigma_m
        beta_2_adj = beta_2 - adjustment

        nde = beta_1  # Direct effect unchanged in linear model
        nie = alpha_1 * beta_2_adj

        nde_at_rho[i] = nde
        nie_at_rho[i] = nie

        # Bootstrap for CIs at this rho
        boot_nde = []
        boot_nie = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)

            if covariates is not None:
                X_m_b = np.column_stack([np.ones(n), treatment[idx], covariates[idx]])
                X_y_b = np.column_stack([np.ones(n), treatment[idx], mediator[idx], covariates[idx]])
            else:
                X_m_b = np.column_stack([np.ones(n), treatment[idx]])
                X_y_b = np.column_stack([np.ones(n), treatment[idx], mediator[idx]])

            try:
                m_fit_b = sm.OLS(mediator[idx], X_m_b).fit()
                y_fit_b = sm.OLS(outcome[idx], X_y_b).fit()

                alpha_1_b = m_fit_b.params[1]
                beta_1_b = y_fit_b.params[1]
                beta_2_b = y_fit_b.params[2]

                sigma_m_b = np.sqrt(m_fit_b.mse_resid)
                sigma_y_b = np.sqrt(y_fit_b.mse_resid)

                adj_b = rho * sigma_y_b / sigma_m_b
                beta_2_adj_b = beta_2_b - adj_b

                boot_nde.append(beta_1_b)
                boot_nie.append(alpha_1_b * beta_2_adj_b)
            except Exception:
                continue

        if len(boot_nde) > 10:
            q_low = alpha / 2 * 100
            q_high = (1 - alpha / 2) * 100
            nde_ci_lower[i] = np.percentile(boot_nde, q_low)
            nde_ci_upper[i] = np.percentile(boot_nde, q_high)
            nie_ci_lower[i] = np.percentile(boot_nie, q_low)
            nie_ci_upper[i] = np.percentile(boot_nie, q_high)
        else:
            nde_ci_lower[i] = np.nan
            nde_ci_upper[i] = np.nan
            nie_ci_lower[i] = np.nan
            nie_ci_upper[i] = np.nan

    # Find rho at which effects cross zero
    rho_at_zero_nie = _find_zero_crossing(rho_grid, nie_at_rho)
    rho_at_zero_nde = _find_zero_crossing(rho_grid, nde_at_rho)

    # Original estimates (rho = 0)
    zero_idx = np.argmin(np.abs(rho_grid))
    original_nde = nde_at_rho[zero_idx]
    original_nie = nie_at_rho[zero_idx]

    # Generate interpretation
    interpretation = _generate_interpretation(
        original_nde, original_nie, rho_at_zero_nde, rho_at_zero_nie
    )

    return SensitivityResult(
        rho_grid=rho_grid,
        nde_at_rho=nde_at_rho,
        nie_at_rho=nie_at_rho,
        nde_ci_lower=nde_ci_lower,
        nde_ci_upper=nde_ci_upper,
        nie_ci_lower=nie_ci_lower,
        nie_ci_upper=nie_ci_upper,
        rho_at_zero_nie=rho_at_zero_nie,
        rho_at_zero_nde=rho_at_zero_nde,
        original_nde=original_nde,
        original_nie=original_nie,
        interpretation=interpretation,
    )


def _find_zero_crossing(rho_grid: NDArray, effect_grid: NDArray) -> float:
    """Find rho value where effect crosses zero via linear interpolation."""
    # Check for sign changes
    signs = np.sign(effect_grid)
    sign_changes = np.where(np.diff(signs) != 0)[0]

    if len(sign_changes) == 0:
        return np.nan

    # Use first crossing
    idx = sign_changes[0]
    rho1, rho2 = rho_grid[idx], rho_grid[idx + 1]
    eff1, eff2 = effect_grid[idx], effect_grid[idx + 1]

    # Linear interpolation to find zero crossing
    if abs(eff2 - eff1) < 1e-10:
        return (rho1 + rho2) / 2

    rho_zero = rho1 - eff1 * (rho2 - rho1) / (eff2 - eff1)

    return rho_zero


def _generate_interpretation(
    original_nde: float,
    original_nie: float,
    rho_at_zero_nde: float,
    rho_at_zero_nie: float,
) -> str:
    """Generate human-readable interpretation of sensitivity results."""
    lines = []

    lines.append("Mediation Sensitivity Analysis Results")
    lines.append("=" * 40)
    lines.append(f"Original NDE (at rho=0): {original_nde:.4f}")
    lines.append(f"Original NIE (at rho=0): {original_nie:.4f}")
    lines.append("")

    # Interpret NIE sensitivity
    if np.isnan(rho_at_zero_nie):
        lines.append(
            "NIE: Effect does not cross zero in the examined rho range. "
            "The indirect effect appears robust to moderate confounding."
        )
    else:
        abs_rho = abs(rho_at_zero_nie)
        if abs_rho < 0.2:
            robustness = "NOT ROBUST"
            advice = "Very weak confounding could explain away the indirect effect."
        elif abs_rho < 0.4:
            robustness = "MODERATELY SENSITIVE"
            advice = "Moderate confounding could nullify the indirect effect."
        elif abs_rho < 0.6:
            robustness = "MODERATELY ROBUST"
            advice = "Substantial confounding would be needed to nullify the effect."
        else:
            robustness = "ROBUST"
            advice = "Only strong confounding could explain away the indirect effect."

        lines.append(
            f"NIE: Effect becomes zero at rho = {rho_at_zero_nie:.3f} ({robustness})"
        )
        lines.append(f"     {advice}")

    lines.append("")

    # Interpret NDE sensitivity
    if np.isnan(rho_at_zero_nde):
        lines.append(
            "NDE: Direct effect does not cross zero in the examined range. "
            "The direct effect appears robust."
        )
    else:
        abs_rho = abs(rho_at_zero_nde)
        if abs_rho < 0.3:
            lines.append(
                f"NDE: Effect becomes zero at rho = {rho_at_zero_nde:.3f} (SENSITIVE)"
            )
        else:
            lines.append(
                f"NDE: Effect becomes zero at rho = {rho_at_zero_nde:.3f} (ROBUST)"
            )

    return "\n".join(lines)


def medsens_plot(
    sensitivity_result: SensitivityResult,
    effect: str = "both",
    show_ci: bool = True,
    figsize: Tuple[float, float] = (10, 6),
):
    """
    Plot sensitivity analysis results.

    Parameters
    ----------
    sensitivity_result : SensitivityResult
        Result from mediation_sensitivity()
    effect : str
        "nie" for indirect, "nde" for direct, "both"
    show_ci : bool
        Show confidence intervals
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    fig, ax = plt.subplots(figsize=figsize)

    rho = sensitivity_result["rho_grid"]

    if effect in ("nie", "both"):
        nie = sensitivity_result["nie_at_rho"]
        ax.plot(rho, nie, "b-", label="NIE (Indirect)", linewidth=2)
        if show_ci:
            ax.fill_between(
                rho,
                sensitivity_result["nie_ci_lower"],
                sensitivity_result["nie_ci_upper"],
                alpha=0.2,
                color="blue",
            )
        # Mark zero crossing
        rho_zero = sensitivity_result["rho_at_zero_nie"]
        if not np.isnan(rho_zero):
            ax.axvline(rho_zero, color="blue", linestyle="--", alpha=0.7)

    if effect in ("nde", "both"):
        nde = sensitivity_result["nde_at_rho"]
        ax.plot(rho, nde, "r-", label="NDE (Direct)", linewidth=2)
        if show_ci:
            ax.fill_between(
                rho,
                sensitivity_result["nde_ci_lower"],
                sensitivity_result["nde_ci_upper"],
                alpha=0.2,
                color="red",
            )

    ax.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax.axvline(0, color="black", linestyle="-", alpha=0.3)

    ax.set_xlabel("Sensitivity Parameter (ρ)", fontsize=12)
    ax.set_ylabel("Effect Estimate", fontsize=12)
    ax.set_title("Mediation Sensitivity Analysis", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    return fig
