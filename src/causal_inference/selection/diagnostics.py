"""Diagnostics for Heckman selection model.

Provides tests and visualizations for assessing selection bias
and model fit.
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy import stats

from src.causal_inference.selection.types import SelectionTestResult


def selection_bias_test(
    lambda_coef: float,
    lambda_se: float,
    alpha: float = 0.05,
) -> SelectionTestResult:
    """
    Test for selection bias using the IMR coefficient.

    Tests H₀: λ = 0 (no selection bias) vs H₁: λ ≠ 0.

    Parameters
    ----------
    lambda_coef : float
        Estimated coefficient on Inverse Mills Ratio.
    lambda_se : float
        Standard error of lambda coefficient.
    alpha : float, default=0.05
        Significance level for the test.

    Returns
    -------
    SelectionTestResult
        Dictionary with:
        - statistic: t-statistic
        - pvalue: Two-sided p-value
        - reject_null: Whether to reject H₀
        - interpretation: Human-readable interpretation

    Notes
    -----
    - Rejection indicates statistically significant selection bias
    - Failure to reject does NOT prove no selection bias (low power possible)
    - Significant selection bias suggests OLS would be inconsistent
    """
    if lambda_se <= 0 or np.isnan(lambda_se):
        return SelectionTestResult(
            statistic=np.nan,
            pvalue=np.nan,
            reject_null=False,
            interpretation="Cannot compute test: invalid standard error",
        )

    t_stat = lambda_coef / lambda_se
    pvalue = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    reject = pvalue < alpha

    if reject:
        direction = "positive" if lambda_coef > 0 else "negative"
        interpretation = (
            f"Significant selection bias detected (p = {pvalue:.4f}). "
            f"The {direction} λ indicates that unobserved factors affecting "
            f"selection are correlated with the outcome. OLS on selected "
            f"sample would be inconsistent."
        )
    else:
        interpretation = (
            f"No significant selection bias at α = {alpha} (p = {pvalue:.4f}). "
            f"However, this does not prove absence of selection bias - "
            f"the test may have low power with small samples."
        )

    return SelectionTestResult(
        statistic=float(t_stat),
        pvalue=float(pvalue),
        reject_null=reject,
        interpretation=interpretation,
    )


def compute_selection_hazard(
    selection_probs: np.ndarray,
) -> np.ndarray:
    """
    Compute selection hazard rate.

    The hazard rate is the IMR for unselected observations:
    h(p) = -φ(Φ⁻¹(p)) / (1 - p)

    This measures the "risk" of not being selected conditional on
    not yet being selected.

    Parameters
    ----------
    selection_probs : np.ndarray
        Predicted P(S=1|Z) from probit model.

    Returns
    -------
    np.ndarray
        Selection hazard rates.
    """
    p = np.clip(selection_probs, 1e-6, 1 - 1e-6)
    z = stats.norm.ppf(p)
    phi_z = stats.norm.pdf(z)

    # Hazard for non-selection
    hazard = phi_z / (1 - p)

    return hazard


def diagnose_identification(
    selection_covariates: np.ndarray,
    outcome_covariates: np.ndarray,
    threshold: float = 0.99,
) -> dict:
    """
    Check identification conditions for Heckman model.

    Strong identification requires an exclusion restriction: at least
    one variable in selection equation not in outcome equation.

    Parameters
    ----------
    selection_covariates : np.ndarray
        Covariates for selection equation (Z).
    outcome_covariates : np.ndarray
        Covariates for outcome equation (X).
    threshold : float
        Correlation threshold for "essentially collinear".

    Returns
    -------
    dict
        Dictionary with:
        - has_exclusion: Whether any Z variable not in X
        - excluded_indices: Indices of variables in Z but not X
        - collinearity_warning: Whether high collinearity detected
        - identification_strength: 'strong', 'weak', or 'fragile'
    """
    if selection_covariates.ndim == 1:
        selection_covariates = selection_covariates.reshape(-1, 1)
    if outcome_covariates.ndim == 1:
        outcome_covariates = outcome_covariates.reshape(-1, 1)

    k_z = selection_covariates.shape[1]
    k_x = outcome_covariates.shape[1]

    # Check for exclusion restriction by correlation
    excluded_indices = []
    max_corr_for_each_z = []

    for i in range(k_z):
        z_col = selection_covariates[:, i]
        max_corr = 0.0
        for j in range(k_x):
            x_col = outcome_covariates[:, j]
            corr = abs(np.corrcoef(z_col, x_col)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            max_corr = max(max_corr, corr)

        max_corr_for_each_z.append(max_corr)
        if max_corr < threshold:
            excluded_indices.append(i)

    has_exclusion = len(excluded_indices) > 0
    collinearity_warning = any(c > 0.9 for c in max_corr_for_each_z)

    if has_exclusion and not collinearity_warning:
        strength = "strong"
    elif has_exclusion:
        strength = "weak"
    else:
        strength = "fragile"

    return {
        "has_exclusion": has_exclusion,
        "excluded_indices": excluded_indices,
        "collinearity_warning": collinearity_warning,
        "identification_strength": strength,
        "max_correlations": max_corr_for_each_z,
    }


def plot_imr_distribution(
    imr: np.ndarray,
    selected: Optional[np.ndarray] = None,
    ax=None,
    **kwargs,
) -> None:
    """
    Plot distribution of Inverse Mills Ratio.

    Parameters
    ----------
    imr : np.ndarray
        Inverse Mills Ratio values.
    selected : np.ndarray, optional
        Selection indicator to show distribution by selection status.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    **kwargs
        Additional arguments passed to histogram.

    Returns
    -------
    None
        Modifies axes in place.

    Notes
    -----
    High IMR values indicate low selection probability.
    A spike at high IMR values suggests potential identification problems.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        ax = plt.gca()

    if selected is not None:
        imr_selected = imr[selected == 1]
        ax.hist(imr_selected, bins=30, alpha=0.7, label="Selected", **kwargs)
    else:
        ax.hist(imr, bins=30, alpha=0.7, **kwargs)

    ax.set_xlabel("Inverse Mills Ratio (λ)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Inverse Mills Ratio")

    if selected is not None:
        ax.legend()


def plot_selection_probability(
    selection_probs: np.ndarray,
    selected: np.ndarray,
    ax=None,
) -> None:
    """
    Plot selection probability by actual selection status.

    Parameters
    ----------
    selection_probs : np.ndarray
        Predicted P(S=1|Z) from probit model.
    selected : np.ndarray
        Actual selection indicator.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    None
        Modifies axes in place.

    Notes
    -----
    Good model fit shows separation between selected and unselected groups.
    Overlapping distributions indicate weak selection equation.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        ax = plt.gca()

    probs_selected = selection_probs[selected == 1]
    probs_unselected = selection_probs[selected == 0]

    ax.hist(
        probs_selected,
        bins=30,
        alpha=0.5,
        label=f"Selected (n={len(probs_selected)})",
        color="blue",
    )
    ax.hist(
        probs_unselected,
        bins=30,
        alpha=0.5,
        label=f"Not selected (n={len(probs_unselected)})",
        color="red",
    )

    ax.set_xlabel("Predicted P(Selected | Z)")
    ax.set_ylabel("Frequency")
    ax.set_title("Selection Model Fit")
    ax.legend()
