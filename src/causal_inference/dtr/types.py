"""Type definitions for Dynamic Treatment Regimes (DTR).

Defines data structures for multi-stage treatment data and estimation results
for Q-learning and A-learning methods.

References
----------
Murphy, S. A. (2003). Optimal dynamic treatment regimes. JRSS-B.
Schulte, P. J. et al. (2014). Q- and A-learning methods for estimating
    optimal dynamic treatment regimes. Statistical Science.
Robins, J. M. (2004). Optimal structural nested models for optimal sequential
    decisions. In Proceedings of the Second Seattle Symposium on Biostatistics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class DTRData:
    """Multi-stage treatment data for Dynamic Treatment Regimes.

    Stores sequential decision data where each stage has outcomes, treatments,
    and covariates/history. Supports K >= 1 decision stages.

    Attributes
    ----------
    outcomes : list[np.ndarray]
        Outcomes at each stage [Y_1, ..., Y_K]. Each array has shape (n,).
        Y_k is the outcome observed after treatment A_k.
    treatments : list[np.ndarray]
        Binary treatments at each stage [A_1, ..., A_K]. Each array has shape (n,).
        A_k in {0, 1} is the treatment decision at stage k.
    covariates : list[np.ndarray]
        Covariates/history at each stage [X_1, ..., X_K].
        X_k has shape (n, p_k) and represents information available at stage k
        before treatment decision A_k. May include baseline covariates,
        prior outcomes, and prior treatments.

    Properties
    ----------
    n_stages : int
        Number of decision stages K.
    n_obs : int
        Number of observations n.
    n_covariates : list[int]
        Number of covariates at each stage [p_1, ..., p_K].

    Examples
    --------
    >>> import numpy as np
    >>> # Single-stage DTR (like standard CATE)
    >>> n = 500
    >>> X = np.random.randn(n, 3)
    >>> A = np.random.binomial(1, 0.5, n)
    >>> Y = X[:, 0] + 2.0 * A + np.random.randn(n)
    >>> data = DTRData(outcomes=[Y], treatments=[A], covariates=[X])
    >>> print(f"Stages: {data.n_stages}, Obs: {data.n_obs}")
    Stages: 1, Obs: 500

    >>> # Two-stage DTR
    >>> X1 = np.random.randn(n, 3)
    >>> A1 = np.random.binomial(1, 0.5, n)
    >>> Y1 = X1[:, 0] + A1 + np.random.randn(n)
    >>> X2 = np.column_stack([X1, A1, Y1])  # history includes prior A, Y
    >>> A2 = np.random.binomial(1, 0.5, n)
    >>> Y2 = Y1 + X2[:, 0] + 2.0 * A2 + np.random.randn(n)
    >>> data = DTRData(outcomes=[Y1, Y2], treatments=[A1, A2], covariates=[X1, X2])
    >>> print(f"Stages: {data.n_stages}")
    Stages: 2
    """

    outcomes: list[np.ndarray]
    treatments: list[np.ndarray]
    covariates: list[np.ndarray]

    def __post_init__(self):
        """Validate inputs after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate DTR data structure."""
        # Check non-empty
        if len(self.outcomes) == 0:
            raise ValueError(
                "CRITICAL ERROR: Empty data.\n"
                "Function: DTRData._validate\n"
                "outcomes list is empty. Must have at least one stage."
            )

        # Check consistent number of stages
        n_stages = len(self.outcomes)
        if len(self.treatments) != n_stages:
            raise ValueError(
                f"CRITICAL ERROR: Stage count mismatch.\n"
                f"Function: DTRData._validate\n"
                f"outcomes has {n_stages} stages, treatments has {len(self.treatments)}"
            )
        if len(self.covariates) != n_stages:
            raise ValueError(
                f"CRITICAL ERROR: Stage count mismatch.\n"
                f"Function: DTRData._validate\n"
                f"outcomes has {n_stages} stages, covariates has {len(self.covariates)}"
            )

        # Convert to numpy and validate each stage
        n_obs = None
        for k in range(n_stages):
            # Convert to numpy
            self.outcomes[k] = np.asarray(self.outcomes[k], dtype=np.float64)
            self.treatments[k] = np.asarray(self.treatments[k], dtype=np.float64)
            self.covariates[k] = np.asarray(self.covariates[k], dtype=np.float64)

            # Ensure 2D covariates
            if self.covariates[k].ndim == 1:
                self.covariates[k] = self.covariates[k].reshape(-1, 1)

            # Check consistent n_obs across stages
            n_k = len(self.outcomes[k])
            if n_obs is None:
                n_obs = n_k
            elif n_k != n_obs:
                raise ValueError(
                    f"CRITICAL ERROR: Observation count mismatch.\n"
                    f"Function: DTRData._validate\n"
                    f"Stage 1 has {n_obs} obs, stage {k + 1} has {n_k} obs"
                )

            # Check treatment and covariate lengths
            if len(self.treatments[k]) != n_obs:
                raise ValueError(
                    f"CRITICAL ERROR: Length mismatch at stage {k + 1}.\n"
                    f"Function: DTRData._validate\n"
                    f"outcomes: {n_obs}, treatments: {len(self.treatments[k])}"
                )
            if len(self.covariates[k]) != n_obs:
                raise ValueError(
                    f"CRITICAL ERROR: Length mismatch at stage {k + 1}.\n"
                    f"Function: DTRData._validate\n"
                    f"outcomes: {n_obs}, covariates: {len(self.covariates[k])}"
                )

            # Check binary treatment
            unique_treatments = np.unique(self.treatments[k])
            if not np.all(np.isin(unique_treatments, [0, 1])):
                raise ValueError(
                    f"CRITICAL ERROR: Non-binary treatment at stage {k + 1}.\n"
                    f"Function: DTRData._validate\n"
                    f"Treatment must be binary (0 or 1), found: {unique_treatments}"
                )

    @property
    def n_stages(self) -> int:
        """Number of decision stages K."""
        return len(self.outcomes)

    @property
    def n_obs(self) -> int:
        """Number of observations n."""
        return len(self.outcomes[0])

    @property
    def n_covariates(self) -> list[int]:
        """Number of covariates at each stage [p_1, ..., p_K]."""
        return [X.shape[1] for X in self.covariates]

    def get_history(self, stage: int) -> np.ndarray:
        """Get full history available at a given stage.

        Parameters
        ----------
        stage : int
            Stage index (1-indexed, so stage=1 is first stage).

        Returns
        -------
        np.ndarray
            History H_k = (X_1, A_1, Y_1, ..., X_{k-1}, A_{k-1}, Y_{k-1}, X_k)
            at stage k.
        """
        if stage < 1 or stage > self.n_stages:
            raise ValueError(
                f"CRITICAL ERROR: Invalid stage.\n"
                f"Function: DTRData.get_history\n"
                f"stage must be in [1, {self.n_stages}], got {stage}"
            )

        k = stage - 1  # Convert to 0-indexed

        # Start with current stage covariates
        history_parts = [self.covariates[k]]

        # Add prior stages
        for j in range(k):
            history_parts.insert(-1, self.outcomes[j].reshape(-1, 1))
            history_parts.insert(-1, self.treatments[j].reshape(-1, 1))
            history_parts.insert(-1, self.covariates[j])

        # Note: the above inserts are in reverse order, so we actually
        # want to build it forward
        history_parts = []
        for j in range(k):
            history_parts.append(self.covariates[j])
            history_parts.append(self.treatments[j].reshape(-1, 1))
            history_parts.append(self.outcomes[j].reshape(-1, 1))
        history_parts.append(self.covariates[k])

        return np.hstack(history_parts) if len(history_parts) > 1 else self.covariates[k]


@dataclass
class QLearningResult:
    """Result from Q-learning estimation.

    Contains optimal value function, blip function coefficients,
    and the estimated optimal treatment regime.

    Attributes
    ----------
    value_estimate : float
        Estimated expected outcome under optimal regime E[Y^{d*}].
    value_se : float
        Standard error of value estimate.
    value_ci_lower : float
        Lower bound of (1-alpha) confidence interval for value.
    value_ci_upper : float
        Upper bound of (1-alpha) confidence interval for value.
    blip_coefficients : list[np.ndarray]
        Blip function coefficients at each stage [psi_1, ..., psi_K].
        Blip gamma_k(H_k) = H_k' @ psi_k represents treatment effect given history.
    blip_se : list[np.ndarray]
        Standard errors for blip coefficients at each stage.
    stage_q_functions : list[Callable]
        Q-functions for each stage. Q_k(H, a) returns expected outcome.
    optimal_regime : Callable
        Optimal treatment rule. d*(H, stage) -> {0, 1}.
    n_stages : int
        Number of decision stages.
    se_method : str
        Method used for standard error estimation ("sandwich" or "bootstrap").

    Examples
    --------
    >>> # After fitting Q-learning
    >>> result = q_learning(data)
    >>> print(f"Optimal value: {result.value_estimate:.3f}")
    >>> print(f"Stage 1 blip: {result.blip_coefficients[0]}")
    >>> # Get optimal treatment for new patient
    >>> H_new = np.array([1.0, 0.5, -0.3])
    >>> optimal_A = result.optimal_regime(H_new, stage=1)
    """

    value_estimate: float
    value_se: float
    value_ci_lower: float
    value_ci_upper: float
    blip_coefficients: list[np.ndarray]
    blip_se: list[np.ndarray]
    stage_q_functions: list[Callable]
    optimal_regime: Callable
    n_stages: int
    se_method: str = "sandwich"

    def summary(self) -> str:
        """Generate summary string of Q-learning results.

        Returns
        -------
        str
            Formatted summary of estimation results.
        """
        lines = [
            "Q-Learning Results",
            "=" * 50,
            f"Number of stages: {self.n_stages}",
            f"SE method: {self.se_method}",
            "",
            "Value Function:",
            f"  Optimal value: {self.value_estimate:.4f} (SE: {self.value_se:.4f})",
            f"  95% CI: [{self.value_ci_lower:.4f}, {self.value_ci_upper:.4f}]",
            "",
            "Blip Coefficients by Stage:",
        ]

        for k in range(self.n_stages):
            lines.append(f"  Stage {k + 1}:")
            for j, (coef, se) in enumerate(
                zip(self.blip_coefficients[k], self.blip_se[k])
            ):
                lines.append(f"    psi[{j}]: {coef:.4f} (SE: {se:.4f})")

        return "\n".join(lines)

    def predict_optimal_treatment(
        self, history: np.ndarray, stage: int = 1
    ) -> np.ndarray:
        """Predict optimal treatment for given history.

        Parameters
        ----------
        history : np.ndarray
            History/covariates of shape (n, p) or (p,) for single obs.
        stage : int
            Decision stage (1-indexed).

        Returns
        -------
        np.ndarray
            Optimal treatment {0, 1} for each observation.
        """
        history = np.atleast_2d(history)
        return np.array([self.optimal_regime(h, stage) for h in history])


@dataclass
class ALearningResult:
    """Result from A-learning estimation.

    A-learning (Advantage Learning) is a doubly robust method for estimating
    optimal treatment regimes. It is consistent if either the propensity score
    model or the baseline outcome model is correctly specified.

    Attributes
    ----------
    value_estimate : float
        Estimated expected outcome under optimal regime E[Y^{d*}].
    value_se : float
        Standard error of value estimate.
    value_ci_lower : float
        Lower bound of (1-alpha) confidence interval for value.
    value_ci_upper : float
        Upper bound of (1-alpha) confidence interval for value.
    blip_coefficients : list[np.ndarray]
        Blip function coefficients at each stage [psi_1, ..., psi_K].
        Blip gamma_k(H_k) = H_k' @ psi_k represents treatment contrast.
    blip_se : list[np.ndarray]
        Standard errors for blip coefficients (influence function based).
    optimal_regime : Callable
        Optimal treatment rule. d*(H, stage) -> {0, 1}.
    n_stages : int
        Number of decision stages.
    propensity_model : str
        Propensity model used ("logit" or "probit").
    outcome_model : str
        Baseline outcome model used ("ols" or "ridge").
    doubly_robust : bool
        Whether doubly robust estimation was used.
    se_method : str
        Method used for standard error estimation ("sandwich" or "bootstrap").

    Notes
    -----
    A-learning is doubly robust: it is consistent if EITHER:
    1. The propensity model P(A=1|H) is correctly specified, OR
    2. The baseline outcome model E[Y|H, A=0] is correctly specified

    This makes it more robust to model misspecification than Q-learning.

    Examples
    --------
    >>> # After fitting A-learning
    >>> result = a_learning(data)
    >>> print(f"Optimal value: {result.value_estimate:.3f}")
    >>> print(f"Doubly robust: {result.doubly_robust}")
    >>> # Get optimal treatment for new patient
    >>> H_new = np.array([1.0, 0.5, -0.3])
    >>> optimal_A = result.optimal_regime(H_new, stage=1)
    """

    value_estimate: float
    value_se: float
    value_ci_lower: float
    value_ci_upper: float
    blip_coefficients: list[np.ndarray]
    blip_se: list[np.ndarray]
    optimal_regime: Callable
    n_stages: int
    propensity_model: str = "logit"
    outcome_model: str = "ols"
    doubly_robust: bool = True
    se_method: str = "sandwich"

    def summary(self) -> str:
        """Generate summary string of A-learning results.

        Returns
        -------
        str
            Formatted summary of estimation results.
        """
        lines = [
            "A-Learning Results",
            "=" * 50,
            f"Number of stages: {self.n_stages}",
            f"Propensity model: {self.propensity_model}",
            f"Outcome model: {self.outcome_model}",
            f"Doubly robust: {self.doubly_robust}",
            f"SE method: {self.se_method}",
            "",
            "Value Function:",
            f"  Optimal value: {self.value_estimate:.4f} (SE: {self.value_se:.4f})",
            f"  95% CI: [{self.value_ci_lower:.4f}, {self.value_ci_upper:.4f}]",
            "",
            "Blip Coefficients by Stage:",
        ]

        for k in range(self.n_stages):
            lines.append(f"  Stage {k + 1}:")
            for j, (coef, se) in enumerate(
                zip(self.blip_coefficients[k], self.blip_se[k])
            ):
                lines.append(f"    psi[{j}]: {coef:.4f} (SE: {se:.4f})")

        return "\n".join(lines)

    def predict_optimal_treatment(
        self, history: np.ndarray, stage: int = 1
    ) -> np.ndarray:
        """Predict optimal treatment for given history.

        Parameters
        ----------
        history : np.ndarray
            History/covariates of shape (n, p) or (p,) for single obs.
        stage : int
            Decision stage (1-indexed).

        Returns
        -------
        np.ndarray
            Optimal treatment {0, 1} for each observation.
        """
        history = np.atleast_2d(history)
        return np.array([self.optimal_regime(h, stage) for h in history])
