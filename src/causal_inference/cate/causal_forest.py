"""Causal Forest for heterogeneous treatment effect estimation.

Wraps econml.dml.CausalForestDML to provide causal forests with:
- Honest splitting (separate samples for tree building vs estimation)
- Cross-fitting for valid inference
- Asymptotically valid confidence intervals

Key Features
------------
- Honest forests prevent overfitting and ensure valid CIs (CONCERN-28)
- Built-in cross-fitting from econml
- Excels at capturing nonlinear heterogeneity

Algorithm Overview
------------------
Causal forests (Athey & Wager 2018) adapt random forests for causal inference:

1. **Honest splitting**: Split each tree's training data into:
   - Structure sample: Used to determine tree splits
   - Estimation sample: Used to estimate leaf-level treatment effects

2. **Treatment effect estimation**: In each leaf:
   τ̂(x) = Ȳ₁(leaf) - Ȳ₀(leaf)

3. **Forest aggregation**: Average estimates across trees

4. **Variance estimation**: Uses infinitesimal jackknife

References
----------
- Wager & Athey (2018). "Estimation and inference of heterogeneous treatment
  effects using random forests." Annals of Statistics 46(3): 1228-1242.
- Athey, Tibshirani, Wager (2019). "Generalized random forests."
  Annals of Statistics 47(2): 1148-1178.
"""

import numpy as np
from typing import Optional
from scipy import stats

from .base import CATEResult, validate_cate_inputs


def causal_forest(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    n_estimators: int = 100,
    min_samples_leaf: int = 5,
    max_depth: Optional[int] = None,
    honest: bool = True,
    cv: int = 5,
    subforest_size: int = 4,
    alpha: float = 0.05,
) -> CATEResult:
    """Estimate CATE using Causal Forest with honest splitting.

    Wraps econml.dml.CausalForestDML to provide causal forests with valid
    inference. Honest forests use separate samples for tree structure and
    leaf estimation, preventing overfitting and enabling valid CIs.

    Parameters
    ----------
    outcomes : np.ndarray
        Outcome variable Y of shape (n,).
    treatment : np.ndarray
        Binary treatment indicator of shape (n,). Values must be 0 or 1.
    covariates : np.ndarray
        Covariate matrix X of shape (n, p). Can be (n,) for single covariate.
    n_estimators : int, default=100
        Number of trees in the forest. More trees = more stable estimates
        but slower computation.
    min_samples_leaf : int, default=5
        Minimum samples per leaf. Larger values = more regularization.
    max_depth : int or None, default=None
        Maximum tree depth. None means unlimited (split until min_samples_leaf).
    honest : bool, default=True
        Whether to use honest splitting. **CRITICAL for CONCERN-28**.
        Must be True for valid confidence intervals.
    cv : int, default=5
        Number of cross-fitting folds for nuisance estimation.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    CATEResult
        Dictionary with keys:
        - cate: Individual treatment effects τ(xᵢ) of shape (n,)
        - ate: Average treatment effect (mean of CATE)
        - ate_se: Standard error of ATE
        - ci_lower: Lower bound of (1-α)% CI
        - ci_upper: Upper bound of (1-α)% CI
        - method: "causal_forest"

    Raises
    ------
    ValueError
        If inputs are invalid.
    ImportError
        If econml is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 3)
    >>> T = np.random.binomial(1, 0.5, n)
    >>> # Heterogeneous effect: τ(x) = 2 + X[:,0]
    >>> tau_true = 2 + X[:, 0]
    >>> Y = 1 + X[:, 0] + tau_true * T + np.random.randn(n)
    >>> result = causal_forest(Y, T, X)
    >>> print(f"ATE: {result['ate']:.2f}")
    ATE: 2.01

    Notes
    -----
    **Why Honest Splitting Matters** (CONCERN-28):

    Standard random forests use the same data for:
    1. Determining split points (tree structure)
    2. Estimating leaf values (treatment effects)

    This leads to overfitting and invalid CIs (undercoverage).

    Honest forests fix this by using separate samples:
    - Structure sample → determines splits
    - Estimation sample → computes treatment effects in leaves

    Result: Valid confidence intervals with ~95% coverage.

    **When to Use Causal Forests**:
    - Nonlinear heterogeneity in treatment effects
    - Many covariates (forest handles feature selection)
    - Want robust confidence intervals
    - Don't have strong prior on functional form

    **When NOT to Use**:
    - Small samples (need enough for honest splitting)
    - Interpretability is critical (forests are black boxes)
    - Linear heterogeneity (simpler methods work well)

    References
    ----------
    - Wager & Athey (2018). "Estimation and inference of heterogeneous
      treatment effects using random forests."

    See Also
    --------
    double_ml : Double ML with cross-fitting (parametric approach).
    x_learner : Cross-learner for imbalanced groups.
    """
    # Validate inputs
    outcomes, treatment, covariates = validate_cate_inputs(
        outcomes, treatment, covariates
    )

    n = len(outcomes)

    # Import econml (defer import to handle missing dependency gracefully)
    try:
        from econml.dml import CausalForestDML
        from sklearn.linear_model import LogisticRegression, Ridge
    except ImportError:
        raise ImportError(
            "econml is required for causal_forest(). "
            "Install with: pip install econml"
        )

    # Warn if not using honest splitting
    if not honest:
        import warnings
        warnings.warn(
            "CONCERN-28 WARNING: honest=False disables honest splitting. "
            "This may lead to invalid confidence intervals (undercoverage). "
            "Set honest=True for valid inference.",
            UserWarning,
        )

    # =========================================================================
    # Configure and fit CausalForestDML
    # =========================================================================

    # Ensure n_estimators is divisible by subforest_size (econml requirement)
    if n_estimators % subforest_size != 0:
        adjusted = ((n_estimators // subforest_size) + 1) * subforest_size
        import warnings
        warnings.warn(
            f"n_estimators={n_estimators} not divisible by subforest_size={subforest_size}. "
            f"Adjusting to {adjusted}.",
            UserWarning,
        )
        n_estimators = adjusted

    # Create the causal forest model
    # econml CausalForestDML has built-in cross-fitting and optional honesty
    # For binary treatment, use discrete_treatment=True
    model = CausalForestDML(
        model_y=Ridge(alpha=1.0),  # Outcome model
        model_t=LogisticRegression(max_iter=1000),  # Propensity model
        discrete_treatment=True,  # Important for binary treatment
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        subforest_size=subforest_size,
        honest=honest,
        cv=cv,
        random_state=42,
    )

    # Fit the model
    # econml uses: fit(Y, T, X=covariates, W=controls)
    # For our API, covariates are effect modifiers (X)
    model.fit(
        Y=outcomes,
        T=treatment,  # 1D array for discrete treatment
        X=covariates,
    )

    # =========================================================================
    # Extract CATE estimates
    # =========================================================================

    # Predict CATE for all units
    cate = model.effect(covariates).flatten()

    # Compute ATE
    ate = np.mean(cate)

    # =========================================================================
    # Standard errors and confidence intervals
    # =========================================================================

    # econml provides inference for CATE(x)
    # For ATE SE, we use econml's built-in ate_inference() if available

    try:
        # Use econml's ate_inference for proper inference
        ate_inf = model.ate_inference(X=covariates)
        ate_se = float(ate_inf.stderr_mean)
        # Use the confidence interval from econml
        ci_lower, ci_upper = ate_inf.conf_int_mean(alpha=alpha)
        ci_lower = float(ci_lower)
        ci_upper = float(ci_upper)
    except Exception:
        # Fallback: Use individual CATE SEs if ate_inference fails
        try:
            cate_inference = model.effect_inference(covariates)
            cate_se = cate_inference.std_effect().flatten()
            # SE of ATE ≈ mean of individual SEs (conservative)
            ate_se = float(np.mean(cate_se))
        except Exception:
            # Final fallback: bootstrap-style SE
            ate_se = np.std(cate, ddof=1) / np.sqrt(n)

        # Confidence interval using normal approximation
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = ate - z_crit * ate_se
        ci_upper = ate + z_crit * ate_se

    return CATEResult(
        cate=cate,
        ate=float(ate),
        ate_se=float(ate_se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method="causal_forest",
    )
