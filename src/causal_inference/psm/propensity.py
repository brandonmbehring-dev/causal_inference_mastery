"""
Propensity score estimation via logistic regression.

Implements P(T=1|X) estimation following Julia reference (propensity.jl).

Design:
- Uses sklearn LogisticRegression (replaces Julia's GLM.jl)
- Clamps scores to [1e-10, 1-1e-10] for numerical stability (lines 128 in propensity.jl)
- Checks common support (overlap in propensity distributions)
- Warns on perfect separation

Author: Brandon Behring
Date: 2025-11-21
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass
import warnings


@dataclass
class PropensityResult:
    """
    Result from propensity score estimation.

    Attributes:
        propensity_scores: Estimated P(T=1|X) for all units (length n)
        model: Fitted sklearn LogisticRegression model
        has_common_support: True if overlap exists between treated/control
        support_region: (min_overlap, max_overlap) propensity range with both groups
        n_outside_support: Number of units outside common support
        converged: True if logistic regression converged
    """

    propensity_scores: np.ndarray
    model: LogisticRegression
    has_common_support: bool
    support_region: Tuple[float, float]
    n_outside_support: int
    converged: bool


class PropensityScoreEstimator:
    """
    Estimate propensity scores P(T=1|X) via logistic regression.

    Uses sklearn LogisticRegression with:
    - L2 penalty (C=1e4, very weak regularization)
    - LBFGS solver (handles moderate p well)
    - max_iter=1000 (ensure convergence)

    Methods:
        fit(treatment, covariates): Estimate propensity scores
        predict(covariates): Predict propensity for new units
        check_common_support(propensity, treatment): Validate overlap

    Example:
        >>> estimator = PropensityScoreEstimator()
        >>> result = estimator.fit(treatment, covariates)
        >>> if result.has_common_support:
        ...     print(f"Propensity range: {result.support_region}")
    """

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1e4,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        eps: float = 1e-10,
    ):
        """
        Initialize propensity score estimator.

        Args:
            penalty: Regularization type ('l2', 'l1', 'none')
            C: Inverse regularization strength (larger = weaker penalty)
            solver: Optimization algorithm ('lbfgs', 'liblinear', 'newton-cg')
            max_iter: Maximum iterations for convergence
            eps: Clamping threshold to avoid numerical issues (propensity ∈ [eps, 1-eps])

        Notes:
            - Default C=1e4 provides very weak regularization (nearly unregularized MLE)
            - eps=1e-10 matches Julia implementation (propensity.jl:128)
        """
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.eps = eps
        self.model_: Optional[LogisticRegression] = None

    def fit(
        self, treatment: np.ndarray, covariates: np.ndarray
    ) -> PropensityResult:
        """
        Estimate propensity scores via logistic regression.

        Implements algorithm from Julia reference (propensity.jl:45-131):
        1. Fit logistic regression: logit(e(X)) = β₀ + β'X
        2. Predict propensity scores: e(X) = P(T=1|X)
        3. Clamp to [eps, 1-eps] for numerical stability
        4. Check common support (overlap in distributions)
        5. Warn on perfect separation

        Args:
            treatment: Binary treatment indicator (n,) with values {0, 1} or {False, True}
            covariates: Covariate matrix (n, p)

        Returns:
            PropensityResult with propensity scores, model, and diagnostics

        Raises:
            ValueError: If inputs invalid (wrong shapes, no variation, NaN/Inf)

        Example:
            >>> treatment = np.array([1, 1, 0, 0])
            >>> covariates = np.array([[1.0, 2.0], [1.5, 2.5], [0.5, 1.0], [0.8, 1.2]])
            >>> estimator = PropensityScoreEstimator()
            >>> result = estimator.fit(treatment, covariates)
            >>> result.propensity_scores  # Array of P(T=1|X)
        """
        # ====================================================================
        # Input Validation
        # ====================================================================

        treatment = np.asarray(treatment)
        covariates = np.asarray(covariates)

        n = len(treatment)

        if len(covariates) != n:
            raise ValueError(
                f"CRITICAL ERROR: Mismatched lengths.\n"
                f"Function: PropensityScoreEstimator.fit\n"
                f"treatment has length {n}, covariates has {len(covariates)} rows\n"
                f"All inputs must have same length."
            )

        if covariates.ndim != 2:
            raise ValueError(
                f"CRITICAL ERROR: Invalid covariate shape.\n"
                f"Function: PropensityScoreEstimator.fit\n"
                f"covariates must be 2D array (n, p), got shape {covariates.shape}\n"
                f"Use covariates.reshape(-1, 1) for single covariate."
            )

        if n == 0:
            raise ValueError(
                f"CRITICAL ERROR: Empty inputs.\n"
                f"Function: PropensityScoreEstimator.fit\n"
                f"Cannot estimate propensity with zero observations."
            )

        if np.any(np.isnan(treatment)) or np.any(np.isinf(treatment)):
            raise ValueError(
                f"CRITICAL ERROR: NaN or Inf in treatment.\n"
                f"Function: PropensityScoreEstimator.fit\n"
                f"Treatment contains {np.sum(np.isnan(treatment))} NaN "
                f"and {np.sum(np.isinf(treatment))} Inf values."
            )

        if np.any(np.isnan(covariates)) or np.any(np.isinf(covariates)):
            raise ValueError(
                f"CRITICAL ERROR: NaN or Inf in covariates.\n"
                f"Function: PropensityScoreEstimator.fit\n"
                f"Covariates contain {np.sum(np.isnan(covariates))} NaN "
                f"and {np.sum(np.isinf(covariates))} Inf values."
            )

        # Check treatment is binary
        unique_vals = np.unique(treatment)
        if len(unique_vals) != 2:
            raise ValueError(
                f"CRITICAL ERROR: Treatment not binary.\n"
                f"Function: PropensityScoreEstimator.fit\n"
                f"Treatment has {len(unique_vals)} unique values: {unique_vals}\n"
                f"Must be binary (0/1 or False/True)."
            )

        # Convert to 0/1 if needed
        if not np.all(np.isin(treatment, [0, 1])):
            treatment = (treatment == unique_vals[1]).astype(int)

        n_treated = np.sum(treatment)
        n_control = n - n_treated

        if n_treated == 0 or n_control == 0:
            raise ValueError(
                f"CRITICAL ERROR: No treatment variation.\n"
                f"Function: PropensityScoreEstimator.fit\n"
                f"n_treated={n_treated}, n_control={n_control}\n"
                f"Need both treated and control units."
            )

        # ====================================================================
        # Fit Logistic Regression
        # ====================================================================

        # Suppress sklearn warnings about convergence (we check manually)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            self.model_ = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=None,  # Deterministic
            )

            self.model_.fit(covariates, treatment)

        # Check convergence
        converged = self.model_.n_iter_ < self.max_iter

        if not converged:
            warnings.warn(
                f"Logistic regression did not converge after {self.max_iter} iterations.\n"
                f"Propensity scores may be inaccurate. Consider:\n"
                f"  - Increasing max_iter\n"
                f"  - Standardizing covariates\n"
                f"  - Reducing number of covariates (p={covariates.shape[1]})",
                category=RuntimeWarning,
            )

        # ====================================================================
        # Predict Propensity Scores
        # ====================================================================

        # Get P(T=1|X) from fitted model
        propensity_raw = self.model_.predict_proba(covariates)[:, 1]

        # Clamp to [eps, 1-eps] to avoid numerical issues (matches Julia:128)
        propensity = np.clip(propensity_raw, self.eps, 1 - self.eps)

        # ====================================================================
        # Check Common Support
        # ====================================================================

        has_support, support_region, n_outside = self.check_common_support(
            propensity, treatment
        )

        # ====================================================================
        # Warn on Perfect Separation
        # ====================================================================

        # Perfect separation: propensity near 0 or 1 for many units
        n_extreme = np.sum((propensity < 0.01) | (propensity > 0.99))
        if n_extreme > 0.1 * n:
            warnings.warn(
                f"Possible perfect separation detected.\n"
                f"{n_extreme}/{n} ({100*n_extreme/n:.1f}%) units have extreme propensity (<0.01 or >0.99).\n"
                f"This suggests treatment strongly predicted by covariates.\n"
                f"Consider:\n"
                f"  - Trimming extreme propensities\n"
                f"  - Using caliper matching to enforce common support\n"
                f"  - Checking for covariate/treatment perfect correlation",
                category=RuntimeWarning,
            )

        # ====================================================================
        # Return Result
        # ====================================================================

        return PropensityResult(
            propensity_scores=propensity,
            model=self.model_,
            has_common_support=has_support,
            support_region=support_region,
            n_outside_support=n_outside,
            converged=converged,
        )

    def predict(self, covariates: np.ndarray) -> np.ndarray:
        """
        Predict propensity scores for new units.

        Args:
            covariates: Covariate matrix (n_new, p)

        Returns:
            propensity_scores: Estimated P(T=1|X) for new units (n_new,)

        Raises:
            RuntimeError: If called before fit()
            ValueError: If covariate dimension mismatch

        Example:
            >>> estimator.fit(treatment_train, X_train)
            >>> propensity_test = estimator.predict(X_test)
        """
        if self.model_ is None:
            raise RuntimeError(
                "CRITICAL ERROR: Model not fitted.\n"
                "Function: PropensityScoreEstimator.predict\n"
                "Must call fit() before predict()."
            )

        covariates = np.asarray(covariates)

        if covariates.ndim != 2:
            raise ValueError(
                f"CRITICAL ERROR: Invalid covariate shape.\n"
                f"Function: PropensityScoreEstimator.predict\n"
                f"covariates must be 2D array, got shape {covariates.shape}"
            )

        expected_p = self.model_.coef_.shape[1]
        if covariates.shape[1] != expected_p:
            raise ValueError(
                f"CRITICAL ERROR: Covariate dimension mismatch.\n"
                f"Function: PropensityScoreEstimator.predict\n"
                f"Model trained on p={expected_p} covariates, got {covariates.shape[1]}"
            )

        # Predict and clamp
        propensity_raw = self.model_.predict_proba(covariates)[:, 1]
        propensity = np.clip(propensity_raw, self.eps, 1 - self.eps)

        return propensity

    @staticmethod
    def check_common_support(
        propensity: np.ndarray, treatment: np.ndarray, min_overlap: float = 0.001
    ) -> Tuple[bool, Tuple[float, float], int]:
        """
        Check for common support (overlap in propensity distributions).

        Implements check from Julia reference (propensity.jl:168-203).

        Common support exists if:
        - max(min(e_treated), min(e_control)) < min(max(e_treated), max(e_control))

        Args:
            propensity: Propensity scores (n,)
            treatment: Binary treatment indicator (n,)
            min_overlap: Minimum overlap width required (default: 0.1)

        Returns:
            Tuple of:
            - has_common_support: True if sufficient overlap exists
            - support_region: (lower, upper) bounds of overlap region
            - n_outside_support: Number of units outside support

        Example:
            >>> has_support, region, n_out = PropensityScoreEstimator.check_common_support(
            ...     propensity, treatment
            ... )
            >>> if not has_support:
            ...     print(f"{n_out} units outside support region {region}")
        """
        treatment = np.asarray(treatment).astype(bool)

        e_treated = propensity[treatment]
        e_control = propensity[~treatment]

        if len(e_treated) == 0 or len(e_control) == 0:
            # No overlap possible (no units in one group)
            return False, (np.nan, np.nan), len(propensity)

        # Overlap region: [max(min_t, min_c), min(max_t, max_c)]
        lower = max(np.min(e_treated), np.min(e_control))
        upper = min(np.max(e_treated), np.max(e_control))

        # Check if overlap region is non-empty and sufficient width
        # SPECIAL CASE: If upper == lower, all propensities identical (perfect overlap!)
        if abs(upper - lower) < 1e-10:
            # All propensities identical → perfect common support
            has_support = True
            n_outside = 0
        elif upper > lower:
            # Normal case: check if width sufficient
            has_support = (upper - lower >= min_overlap)
            # Count units outside overlap
            if has_support:
                outside = (propensity < lower) | (propensity > upper)
                n_outside = np.sum(outside)
            else:
                n_outside = len(propensity)  # All outside if no support
        else:
            # upper < lower: No overlap
            has_support = False
            n_outside = len(propensity)

        support_region = (lower, upper)

        return has_support, support_region, n_outside
