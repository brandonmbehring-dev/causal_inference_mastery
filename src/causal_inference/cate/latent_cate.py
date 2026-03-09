"""Latent CATE methods using dimensionality reduction.

This module implements CATE estimation with latent confounder adjustment,
inspired by CEVAE but using simpler sklearn methods instead of variational inference.

Methods
-------
- factor_analysis_cate: CATE with Factor Analysis latent factor augmentation
- ppca_cate: CATE with Probabilistic PCA component augmentation
- gmm_stratified_cate: CATE with GMM-based stratification

References
----------
- Louizos et al. (2017). "Causal Effect Inference with Deep Latent-Variable Models."
  NeurIPS.
"""

from typing import Literal

import numpy as np
from scipy import stats
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.mixture import GaussianMixture

from .base import CATEResult, validate_cate_inputs
from .meta_learners import _get_model, r_learner, t_learner


def _apply_base_learner(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    base_learner: Literal["t_learner", "r_learner"],
    model: str,
    alpha: float,
) -> CATEResult:
    """Apply base meta-learner and return result.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y.
    treatment : np.ndarray
        Binary treatment T.
    covariates : np.ndarray
        Covariate matrix X (potentially augmented).
    base_learner : {"t_learner", "r_learner"}
        Base meta-learner to use.
    model : str
        Model type for base learner.
    alpha : float
        Significance level.

    Returns
    -------
    CATEResult
        Result from base learner.
    """
    if base_learner == "t_learner":
        return t_learner(outcomes, treatment, covariates, model=model, alpha=alpha)
    elif base_learner == "r_learner":
        return r_learner(outcomes, treatment, covariates, model=model, alpha=alpha)
    else:
        raise ValueError(
            f"Invalid base_learner: {base_learner}. Must be 't_learner' or 'r_learner'."
        )


def factor_analysis_cate(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    n_latent: int = 5,
    base_learner: Literal["t_learner", "r_learner"] = "t_learner",
    model: Literal["linear", "ridge", "random_forest"] = "linear",
    alpha: float = 0.05,
) -> CATEResult:
    """CATE estimation with Factor Analysis latent factor augmentation.

    Extracts latent factors from covariates using Factor Analysis,
    augments the feature space, and applies a meta-learner.

    This approximates the insight from CEVAE (Louizos et al. 2017):
    latent factors may capture unobserved confounders.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator T, shape (n,).
    covariates : np.ndarray
        Covariate matrix X, shape (n, p) or (n,).
    n_latent : int, default=5
        Number of latent factors to extract.
    base_learner : {"t_learner", "r_learner"}, default="t_learner"
        Meta-learner to apply to augmented features.
    model : {"linear", "ridge", "random_forest"}, default="linear"
        Model type for the base learner.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with cate, ate, ate_se, ci_lower, ci_upper, method.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 5)
    >>> T = np.random.binomial(1, 0.5, n)
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)
    >>> result = factor_analysis_cate(Y, T, X, n_latent=3)
    >>> abs(result["ate"] - 2.0) < 0.5
    True

    Notes
    -----
    Factor Analysis assumes X = L @ F + noise, where L is the loading matrix
    and F are latent factors. The extracted factors F are appended to X
    before applying the base meta-learner.
    """
    # Validate inputs
    validate_cate_inputs(outcomes, treatment, covariates)

    # Ensure 2D covariates
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n, p = covariates.shape

    # Validate n_latent
    if n_latent < 1:
        raise ValueError(f"n_latent must be >= 1, got {n_latent}")
    if n_latent >= p:
        # Cap at p-1 to avoid degenerate case
        n_latent = max(1, p - 1)

    # Extract latent factors
    fa = FactorAnalysis(n_components=n_latent, random_state=42, max_iter=1000)
    try:
        latent = fa.fit_transform(covariates)
    except Exception as e:
        raise ValueError(
            f"Factor Analysis fitting failed: {str(e)}. "
            f"Try reducing n_latent or checking for constant columns."
        )

    # Augment covariates
    X_augmented = np.column_stack([covariates, latent])

    # Apply base learner
    result = _apply_base_learner(outcomes, treatment, X_augmented, base_learner, model, alpha)

    # Return with updated method name
    return CATEResult(
        cate=result["cate"],
        ate=result["ate"],
        ate_se=result["ate_se"],
        ci_lower=result["ci_lower"],
        ci_upper=result["ci_upper"],
        method="factor_analysis_cate",
    )


def ppca_cate(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    n_components: int = 5,
    base_learner: Literal["t_learner", "r_learner"] = "t_learner",
    model: Literal["linear", "ridge", "random_forest"] = "linear",
    alpha: float = 0.05,
) -> CATEResult:
    """CATE estimation with Probabilistic PCA augmentation.

    Extracts principal components from covariates using PCA,
    augments the feature space, and applies a meta-learner.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator T, shape (n,).
    covariates : np.ndarray
        Covariate matrix X, shape (n, p) or (n,).
    n_components : int, default=5
        Number of principal components to extract.
    base_learner : {"t_learner", "r_learner"}, default="t_learner"
        Meta-learner to apply to augmented features.
    model : {"linear", "ridge", "random_forest"}, default="linear"
        Model type for the base learner.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with cate, ate, ate_se, ci_lower, ci_upper, method.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 5)
    >>> T = np.random.binomial(1, 0.5, n)
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)
    >>> result = ppca_cate(Y, T, X, n_components=3)
    >>> abs(result["ate"] - 2.0) < 0.5
    True

    Notes
    -----
    This uses sklearn's PCA which can be interpreted as probabilistic PCA
    in the limit. The extracted components are appended to X before
    applying the base meta-learner.
    """
    # Validate inputs
    validate_cate_inputs(outcomes, treatment, covariates)

    # Ensure 2D covariates
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n, p = covariates.shape

    # Validate n_components
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if n_components >= p:
        n_components = max(1, p - 1)

    # Extract principal components
    pca = PCA(n_components=n_components, random_state=42)
    try:
        latent = pca.fit_transform(covariates)
    except Exception as e:
        raise ValueError(
            f"PCA fitting failed: {str(e)}. "
            f"Try reducing n_components or checking for constant columns."
        )

    # Augment covariates
    X_augmented = np.column_stack([covariates, latent])

    # Apply base learner
    result = _apply_base_learner(outcomes, treatment, X_augmented, base_learner, model, alpha)

    # Return with updated method name
    return CATEResult(
        cate=result["cate"],
        ate=result["ate"],
        ate_se=result["ate_se"],
        ci_lower=result["ci_lower"],
        ci_upper=result["ci_upper"],
        method="ppca_cate",
    )


def gmm_stratified_cate(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    n_strata: int = 3,
    base_learner: Literal["t_learner", "r_learner"] = "t_learner",
    model: Literal["linear", "ridge", "random_forest"] = "linear",
    alpha: float = 0.05,
) -> CATEResult:
    """CATE estimation with GMM-based stratification.

    Identifies latent subgroups using Gaussian Mixture Model,
    then estimates CATE within each stratum.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator T, shape (n,).
    covariates : np.ndarray
        Covariate matrix X, shape (n, p) or (n,).
    n_strata : int, default=3
        Number of GMM components (strata).
    base_learner : {"t_learner", "r_learner"}, default="t_learner"
        Meta-learner to apply within each stratum.
    model : {"linear", "ridge", "random_forest"}, default="linear"
        Model type for the base learner.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with cate, ate, ate_se, ci_lower, ci_upper, method.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 5)
    >>> T = np.random.binomial(1, 0.5, n)
    >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)
    >>> result = gmm_stratified_cate(Y, T, X, n_strata=3)
    >>> abs(result["ate"] - 2.0) < 0.5
    True

    Notes
    -----
    This approach identifies latent subgroups in the covariate space,
    then estimates stratum-specific treatment effects. The final CATE
    is computed by assigning each unit to its stratum and using the
    stratum-specific estimate.
    """
    # Validate inputs
    validate_cate_inputs(outcomes, treatment, covariates)

    # Ensure 2D covariates
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n, p = covariates.shape

    # Validate n_strata
    if n_strata < 2:
        raise ValueError(f"n_strata must be >= 2, got {n_strata}")

    # Fit GMM to identify strata
    gmm = GaussianMixture(n_components=n_strata, random_state=42, n_init=3)
    try:
        strata = gmm.fit_predict(covariates)
    except Exception as e:
        raise ValueError(f"GMM fitting failed: {str(e)}")

    # Estimate CATE within each stratum
    cate = np.zeros(n)
    stratum_ates = []
    stratum_weights = []

    for s in range(n_strata):
        stratum_mask = strata == s
        n_stratum = np.sum(stratum_mask)

        if n_stratum < 4:  # Need at least some samples
            continue

        treated_mask = (treatment == 1) & stratum_mask
        control_mask = (treatment == 0) & stratum_mask

        n_treated = np.sum(treated_mask)
        n_control = np.sum(control_mask)

        if n_treated < 2 or n_control < 2:
            continue  # Skip sparse strata

        # Get data for this stratum
        X_stratum = covariates[stratum_mask]
        Y_stratum = outcomes[stratum_mask]
        T_stratum = treatment[stratum_mask]

        # Apply base learner within stratum
        try:
            result_s = _apply_base_learner(
                Y_stratum, T_stratum, X_stratum, base_learner, model, alpha
            )
            cate[stratum_mask] = result_s["cate"]
            stratum_ates.append(result_s["ate"])
            stratum_weights.append(n_stratum)
        except Exception:
            # If stratum fails, use simple difference of means
            y_treated = outcomes[treated_mask].mean()
            y_control = outcomes[control_mask].mean()
            cate[stratum_mask] = y_treated - y_control
            stratum_ates.append(y_treated - y_control)
            stratum_weights.append(n_stratum)

    # Compute weighted ATE
    if stratum_ates:
        weights = np.array(stratum_weights) / np.sum(stratum_weights)
        ate = float(np.sum(np.array(stratum_ates) * weights))
    else:
        # Fallback: simple difference of means
        ate = float(outcomes[treatment == 1].mean() - outcomes[treatment == 0].mean())

    # Compute SE from CATE variance
    cate_var = np.var(cate, ddof=1) if len(cate) > 1 else 0.0
    ate_se = float(np.sqrt(cate_var / n))

    # Ensure SE is positive
    if ate_se < 1e-10:
        # Fallback SE estimation
        n_treated = int(np.sum(treatment))
        n_control = n - n_treated
        var_treated = np.var(outcomes[treatment == 1], ddof=1) if n_treated > 1 else 1.0
        var_control = np.var(outcomes[treatment == 0], ddof=1) if n_control > 1 else 1.0
        ate_se = float(np.sqrt(var_treated / max(n_treated, 1) + var_control / max(n_control, 1)))

    # Confidence interval
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z * ate_se
    ci_upper = ate + z * ate_se

    return CATEResult(
        cate=cate,
        ate=ate,
        ate_se=ate_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        method="gmm_stratified_cate",
    )
