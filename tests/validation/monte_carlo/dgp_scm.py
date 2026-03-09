"""
Data Generating Processes (DGPs) for Synthetic Control Methods Monte Carlo validation.

This module provides DGPs for validating SCM estimators:
- Basic SCM with different pre-treatment fit quality
- Augmented SCM comparison scenarios
- Various donor pool sizes and pre-treatment periods

All DGPs have known true ATT for validation purposes.

References:
    - Abadie, Diamond, Hainmueller (2010). "Synthetic Control Methods"
    - Ben-Michael, Feller, Rothstein (2021). "Augmented Synthetic Control"
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class SCMData:
    """Container for SCM simulation data with known ground truth."""

    outcomes: np.ndarray  # (n_units, n_periods) panel data
    treatment: np.ndarray  # (n_units,) binary, typically 1 treated unit
    treatment_period: int  # Period when treatment starts (0-indexed)
    true_att: float  # Known ground truth ATT
    n_units: int
    n_periods: int
    n_pre_periods: int
    n_post_periods: int
    dgp_type: str  # Description of DGP
    expected_fit: str  # "perfect", "good", "moderate", "poor"
    true_weights: Optional[np.ndarray] = None  # For known-weight DGPs


# =============================================================================
# Core DGP Functions
# =============================================================================


def dgp_scm_perfect_match(
    n_control: int = 10,
    n_pre: int = 10,
    n_post: int = 5,
    true_att: float = 2.0,
    sigma: float = 0.5,
    random_state: Optional[int] = None,
) -> SCMData:
    """
    SCM DGP with perfect pre-treatment match.

    One control unit is constructed to exactly match the treated unit's
    counterfactual trajectory, allowing weights to concentrate on it.

    DGP:
        - Treated unit: Y_t = α + β·t + true_att·Post_t + ε_t
        - Control 0: Y_t = α + β·t + ε_t (same trajectory without treatment)
        - Controls 1+: Y_t = α_i + β_i·t + ε_t (different intercepts/slopes)

    Expected:
        - Weight on control 0 ≈ 1.0 (or close)
        - pre_rmse ≈ 0 (near perfect fit)
        - Bias < 0.05

    Parameters
    ----------
    n_control : int, default=10
        Number of control units
    n_pre : int, default=10
        Number of pre-treatment periods
    n_post : int, default=5
        Number of post-treatment periods
    true_att : float, default=2.0
        True average treatment effect on treated
    sigma : float, default=0.5
        Standard deviation of errors
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    SCMData
        Container with outcomes, treatment, and ground truth
    """
    rng = np.random.RandomState(random_state)

    n_units = n_control + 1  # 1 treated + n_control
    n_periods = n_pre + n_post
    treatment_period = n_pre  # 0-indexed: treatment starts at period n_pre

    # Common trajectory parameters for treated and control 0
    alpha_treated = 10.0
    beta_treated = 0.3

    # Time vector
    t = np.arange(n_periods)

    # Initialize outcomes
    outcomes = np.zeros((n_units, n_periods))

    # Treated unit (row 0): trajectory + treatment effect post
    treated_trajectory = alpha_treated + beta_treated * t
    treatment_effect = np.zeros(n_periods)
    treatment_effect[treatment_period:] = true_att
    outcomes[0, :] = treated_trajectory + treatment_effect + rng.normal(0, sigma, n_periods)

    # Control 0 (row 1): Same trajectory, no treatment
    outcomes[1, :] = treated_trajectory + rng.normal(0, sigma, n_periods)

    # Other controls (rows 2+): Different trajectories
    for i in range(2, n_units):
        alpha_i = alpha_treated + rng.uniform(-3, 3)
        beta_i = beta_treated + rng.uniform(-0.2, 0.2)
        outcomes[i, :] = alpha_i + beta_i * t + rng.normal(0, sigma, n_periods)

    # Treatment indicator: first unit treated
    treatment = np.array([1] + [0] * n_control)

    # True weights: all weight on control 0
    true_weights = np.zeros(n_control)
    true_weights[0] = 1.0

    return SCMData(
        outcomes=outcomes,
        treatment=treatment,
        treatment_period=treatment_period,
        true_att=true_att,
        n_units=n_units,
        n_periods=n_periods,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        dgp_type="perfect_match",
        expected_fit="perfect",
        true_weights=true_weights,
    )


def dgp_scm_good_fit(
    n_control: int = 20,
    n_pre: int = 10,
    n_post: int = 5,
    true_att: float = 2.0,
    sigma: float = 1.0,
    random_state: Optional[int] = None,
) -> SCMData:
    """
    SCM DGP with good but imperfect pre-treatment match.

    Treated unit's counterfactual is a weighted average of controls.
    SCM should find weights close to the true weights.

    DGP:
        - Controls: Y_it = α_i + β_i·t + γ_i·sin(2πt/n_pre) + ε_it
        - Treated counterfactual: Y_t = Σ w_i^* Y_it (convex combination)
        - Treated observed: Y_t = counterfactual + true_att·Post_t + ε_t

    Expected:
        - pre_rmse < 0.5 (good fit)
        - pre_r_squared > 0.9
        - Bias < 0.10
        - Coverage 93-97%

    Parameters
    ----------
    n_control : int, default=20
        Number of control units
    n_pre : int, default=10
        Number of pre-treatment periods
    n_post : int, default=5
        Number of post-treatment periods
    true_att : float, default=2.0
        True average treatment effect on treated
    sigma : float, default=1.0
        Standard deviation of errors
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    SCMData
        Container with outcomes, treatment, and ground truth
    """
    rng = np.random.RandomState(random_state)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre

    t = np.arange(n_periods)

    # Generate control unit outcomes
    control_outcomes = np.zeros((n_control, n_periods))
    for i in range(n_control):
        alpha_i = 10.0 + rng.uniform(-2, 2)
        beta_i = 0.3 + rng.uniform(-0.1, 0.1)
        gamma_i = 0.5 + rng.uniform(-0.3, 0.3)
        control_outcomes[i, :] = (
            alpha_i
            + beta_i * t
            + gamma_i * np.sin(2 * np.pi * t / n_pre)
            + rng.normal(0, sigma, n_periods)
        )

    # Generate true weights (sparse Dirichlet)
    true_weights = rng.dirichlet(np.ones(n_control) * 0.5)  # Sparse weights

    # Treated counterfactual = weighted average of controls
    treated_counterfactual = control_outcomes.T @ true_weights  # (n_periods,)

    # Treated observed = counterfactual + treatment effect post
    treatment_effect = np.zeros(n_periods)
    treatment_effect[treatment_period:] = true_att
    treated_outcome = (
        treated_counterfactual + treatment_effect + rng.normal(0, sigma * 0.5, n_periods)
    )

    # Stack: treated first, then controls
    outcomes = np.vstack([treated_outcome.reshape(1, -1), control_outcomes])

    treatment = np.array([1] + [0] * n_control)

    return SCMData(
        outcomes=outcomes,
        treatment=treatment,
        treatment_period=treatment_period,
        true_att=true_att,
        n_units=n_units,
        n_periods=n_periods,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        dgp_type="good_fit",
        expected_fit="good",
        true_weights=true_weights,
    )


def dgp_scm_moderate_fit(
    n_control: int = 15,
    n_pre: int = 10,
    n_post: int = 5,
    true_att: float = 2.0,
    sigma: float = 1.0,
    random_state: Optional[int] = None,
) -> SCMData:
    """
    SCM DGP with moderate pre-treatment fit.

    Treated unit's counterfactual is mostly in the convex hull but with some noise,
    leading to moderate pre_rmse but manageable bias.

    DGP:
        - Controls spread across a range
        - Treated is noisy combination of controls (not exact)
        - SCM will have moderate pre_rmse

    Expected:
        - pre_rmse: 0.3-1.0 (moderate fit)
        - Bias < 0.50 for SCM (higher than good_fit)

    Parameters
    ----------
    n_control : int, default=15
        Number of control units
    n_pre : int, default=10
        Number of pre-treatment periods
    n_post : int, default=5
        Number of post-treatment periods
    true_att : float, default=2.0
        True average treatment effect on treated
    sigma : float, default=1.0
        Standard deviation of errors
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    SCMData
        Container with outcomes, treatment, and ground truth
    """
    rng = np.random.RandomState(random_state)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre

    t = np.arange(n_periods)

    # Controls: Spread across a range
    control_outcomes = np.zeros((n_control, n_periods))
    for i in range(n_control):
        alpha_i = 10.0 + rng.uniform(-3, 3)
        beta_i = 0.2 + rng.uniform(-0.1, 0.1)
        control_outcomes[i, :] = alpha_i + beta_i * t + rng.normal(0, sigma, n_periods)

    # Treated: Noisy combination of controls + extra noise
    # This creates moderate fit - achievable but not perfect
    true_weights = rng.dirichlet(np.ones(n_control) * 0.5)
    treated_counterfactual = control_outcomes.T @ true_weights
    # Add extra noise to create moderate mismatch
    treated_counterfactual += rng.normal(0, sigma * 0.8, n_periods)

    # Add treatment effect post
    treatment_effect = np.zeros(n_periods)
    treatment_effect[treatment_period:] = true_att
    treated_outcome = treated_counterfactual + treatment_effect

    outcomes = np.vstack([treated_outcome.reshape(1, -1), control_outcomes])
    treatment = np.array([1] + [0] * n_control)

    return SCMData(
        outcomes=outcomes,
        treatment=treatment,
        treatment_period=treatment_period,
        true_att=true_att,
        n_units=n_units,
        n_periods=n_periods,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        dgp_type="moderate_fit",
        expected_fit="moderate",
        true_weights=None,  # No exact weights due to added noise
    )


def dgp_scm_poor_fit(
    n_control: int = 10,
    n_pre: int = 8,
    n_post: int = 5,
    true_att: float = 2.0,
    sigma: float = 1.0,
    random_state: Optional[int] = None,
) -> SCMData:
    """
    SCM DGP with poor pre-treatment fit (ASCM outperformance expected).

    Treated unit's trajectory is outside the convex hull of controls but
    close enough that ASCM's ridge regression can extrapolate.

    DGP:
        - Controls: Varied intercepts and slopes in a cluster
        - Treated: Higher level than all controls (extrapolation needed)
        - SCM will have moderate pre_rmse; ASCM ridge should reduce bias

    Expected:
        - pre_rmse: 1.0-2.0 (poor but not extreme fit)
        - SCM bias: 0.2-0.5
        - ASCM bias: 0.1-0.25 (meaningful improvement)

    Parameters
    ----------
    n_control : int, default=10
        Number of control units
    n_pre : int, default=8
        Number of pre-treatment periods
    n_post : int, default=5
        Number of post-treatment periods
    true_att : float, default=2.0
        True average treatment effect on treated
    sigma : float, default=1.0
        Standard deviation of errors
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    SCMData
        Container with outcomes, treatment, and ground truth
    """
    rng = np.random.RandomState(random_state)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre

    t = np.arange(n_periods)

    # Controls: Cluster of trajectories with varied but similar characteristics
    control_outcomes = np.zeros((n_control, n_periods))
    base_alpha = 10.0
    base_beta = 0.2
    for i in range(n_control):
        alpha_i = base_alpha + rng.uniform(-2, 2)
        beta_i = base_beta + rng.uniform(-0.08, 0.08)
        control_outcomes[i, :] = alpha_i + beta_i * t + rng.normal(0, sigma, n_periods)

    # Treated: Higher intercept than controls (outside convex hull)
    # But same general trend structure (ridge can extrapolate)
    treated_alpha = base_alpha + 3.0  # Above all controls
    treated_beta = base_beta + 0.05  # Slightly steeper
    treated_counterfactual = (
        treated_alpha + treated_beta * t + rng.normal(0, sigma * 0.5, n_periods)
    )

    # Add treatment effect
    treatment_effect = np.zeros(n_periods)
    treatment_effect[treatment_period:] = true_att
    treated_outcome = treated_counterfactual + treatment_effect

    outcomes = np.vstack([treated_outcome.reshape(1, -1), control_outcomes])
    treatment = np.array([1] + [0] * n_control)

    return SCMData(
        outcomes=outcomes,
        treatment=treatment,
        treatment_period=treatment_period,
        true_att=true_att,
        n_units=n_units,
        n_periods=n_periods,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        dgp_type="poor_fit",
        expected_fit="poor",
        true_weights=None,
    )


def dgp_scm_few_controls(
    n_control: int = 5,
    n_pre: int = 10,
    n_post: int = 5,
    true_att: float = 2.0,
    sigma: float = 0.8,
    random_state: Optional[int] = None,
) -> SCMData:
    """
    SCM DGP with few control units.

    Realistic scenario: Many comparative case studies have limited donors.
    Placebo inference is limited by n_control.

    Expected:
        - Placebo inference limited (only n_control placebos)
        - SE may be less accurate
        - Coverage may be slightly off

    Parameters
    ----------
    n_control : int, default=5
        Number of control units (small)
    n_pre : int, default=10
        Number of pre-treatment periods
    n_post : int, default=5
        Number of post-treatment periods
    true_att : float, default=2.0
        True average treatment effect on treated
    sigma : float, default=0.8
        Standard deviation of errors
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    SCMData
        Container with outcomes, treatment, and ground truth
    """
    rng = np.random.RandomState(random_state)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre

    t = np.arange(n_periods)

    # Controls: Varied trajectories
    control_outcomes = np.zeros((n_control, n_periods))
    for i in range(n_control):
        alpha_i = 10.0 + i * 1.5  # Spread out
        beta_i = 0.2 + rng.uniform(-0.05, 0.05)
        control_outcomes[i, :] = alpha_i + beta_i * t + rng.normal(0, sigma, n_periods)

    # Treated: Interpolates between controls
    true_weights = rng.dirichlet(np.ones(n_control))
    treated_counterfactual = control_outcomes.T @ true_weights

    treatment_effect = np.zeros(n_periods)
    treatment_effect[treatment_period:] = true_att
    treated_outcome = (
        treated_counterfactual + treatment_effect + rng.normal(0, sigma * 0.3, n_periods)
    )

    outcomes = np.vstack([treated_outcome.reshape(1, -1), control_outcomes])
    treatment = np.array([1] + [0] * n_control)

    return SCMData(
        outcomes=outcomes,
        treatment=treatment,
        treatment_period=treatment_period,
        true_att=true_att,
        n_units=n_units,
        n_periods=n_periods,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        dgp_type="few_controls",
        expected_fit="good",
        true_weights=true_weights,
    )


def dgp_scm_many_controls(
    n_control: int = 50,
    n_pre: int = 10,
    n_post: int = 5,
    true_att: float = 2.0,
    sigma: float = 1.0,
    random_state: Optional[int] = None,
) -> SCMData:
    """
    SCM DGP with many control units.

    Asymptotic behavior: More donors improve placebo inference precision.

    Expected:
        - Better placebo inference (more placebos)
        - More precise weights
        - Better SE accuracy

    Parameters
    ----------
    n_control : int, default=50
        Number of control units (large)
    n_pre : int, default=10
        Number of pre-treatment periods
    n_post : int, default=5
        Number of post-treatment periods
    true_att : float, default=2.0
        True average treatment effect on treated
    sigma : float, default=1.0
        Standard deviation of errors
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    SCMData
        Container with outcomes, treatment, and ground truth
    """
    rng = np.random.RandomState(random_state)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre

    t = np.arange(n_periods)

    # Controls: Diverse trajectories
    control_outcomes = np.zeros((n_control, n_periods))
    for i in range(n_control):
        alpha_i = 10.0 + rng.uniform(-4, 4)
        beta_i = 0.3 + rng.uniform(-0.15, 0.15)
        gamma_i = rng.uniform(-0.5, 0.5)
        control_outcomes[i, :] = (
            alpha_i
            + beta_i * t
            + gamma_i * np.sin(2 * np.pi * t / n_pre)
            + rng.normal(0, sigma, n_periods)
        )

    # Treated: Sparse combination of controls
    true_weights = rng.dirichlet(np.ones(n_control) * 0.3)  # Very sparse
    treated_counterfactual = control_outcomes.T @ true_weights

    treatment_effect = np.zeros(n_periods)
    treatment_effect[treatment_period:] = true_att
    treated_outcome = (
        treated_counterfactual + treatment_effect + rng.normal(0, sigma * 0.3, n_periods)
    )

    outcomes = np.vstack([treated_outcome.reshape(1, -1), control_outcomes])
    treatment = np.array([1] + [0] * n_control)

    return SCMData(
        outcomes=outcomes,
        treatment=treatment,
        treatment_period=treatment_period,
        true_att=true_att,
        n_units=n_units,
        n_periods=n_periods,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        dgp_type="many_controls",
        expected_fit="good",
        true_weights=true_weights,
    )


def dgp_scm_short_pre_period(
    n_control: int = 15,
    n_pre: int = 3,
    n_post: int = 5,
    true_att: float = 2.0,
    sigma: float = 1.0,
    random_state: Optional[int] = None,
) -> SCMData:
    """
    SCM DGP with short pre-treatment period.

    Tests behavior with limited pre-treatment data.
    This triggers the warning: "Only 3 pre-treatment periods."

    Expected:
        - Higher variance in weights
        - Less reliable fit assessment
        - ASCM particularly valuable here

    Parameters
    ----------
    n_control : int, default=15
        Number of control units
    n_pre : int, default=3
        Number of pre-treatment periods (very short)
    n_post : int, default=5
        Number of post-treatment periods
    true_att : float, default=2.0
        True average treatment effect on treated
    sigma : float, default=1.0
        Standard deviation of errors
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    SCMData
        Container with outcomes, treatment, and ground truth
    """
    rng = np.random.RandomState(random_state)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre

    t = np.arange(n_periods)

    # Controls
    control_outcomes = np.zeros((n_control, n_periods))
    for i in range(n_control):
        alpha_i = 10.0 + rng.uniform(-2, 2)
        beta_i = 0.3 + rng.uniform(-0.1, 0.1)
        control_outcomes[i, :] = alpha_i + beta_i * t + rng.normal(0, sigma, n_periods)

    # Treated: Combination of controls
    true_weights = rng.dirichlet(np.ones(n_control) * 0.8)
    treated_counterfactual = control_outcomes.T @ true_weights

    treatment_effect = np.zeros(n_periods)
    treatment_effect[treatment_period:] = true_att
    treated_outcome = (
        treated_counterfactual + treatment_effect + rng.normal(0, sigma * 0.5, n_periods)
    )

    outcomes = np.vstack([treated_outcome.reshape(1, -1), control_outcomes])
    treatment = np.array([1] + [0] * n_control)

    return SCMData(
        outcomes=outcomes,
        treatment=treatment,
        treatment_period=treatment_period,
        true_att=true_att,
        n_units=n_units,
        n_periods=n_periods,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        dgp_type="short_pre_period",
        expected_fit="moderate",
        true_weights=true_weights,
    )


def dgp_scm_null_effect(
    n_control: int = 15,
    n_pre: int = 10,
    n_post: int = 5,
    sigma: float = 1.0,
    random_state: Optional[int] = None,
) -> SCMData:
    """
    SCM DGP with zero treatment effect.

    Used to test Type I error calibration and p-value distribution under null.

    Expected:
        - estimate ≈ 0
        - p_value > 0.05 in ~95% of simulations
        - Type I error rate ≤ 5%

    Parameters
    ----------
    n_control : int, default=15
        Number of control units
    n_pre : int, default=10
        Number of pre-treatment periods
    n_post : int, default=5
        Number of post-treatment periods
    sigma : float, default=1.0
        Standard deviation of errors
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    SCMData
        Container with outcomes, treatment, and ground truth (true_att=0)
    """
    rng = np.random.RandomState(random_state)

    n_units = n_control + 1
    n_periods = n_pre + n_post
    treatment_period = n_pre
    true_att = 0.0  # NULL EFFECT

    t = np.arange(n_periods)

    # Controls
    control_outcomes = np.zeros((n_control, n_periods))
    for i in range(n_control):
        alpha_i = 10.0 + rng.uniform(-2, 2)
        beta_i = 0.3 + rng.uniform(-0.1, 0.1)
        control_outcomes[i, :] = alpha_i + beta_i * t + rng.normal(0, sigma, n_periods)

    # Treated: Combination of controls, NO TREATMENT EFFECT
    true_weights = rng.dirichlet(np.ones(n_control) * 0.5)
    treated_counterfactual = control_outcomes.T @ true_weights
    treated_outcome = treated_counterfactual + rng.normal(
        0, sigma * 0.5, n_periods
    )  # No treatment effect added

    outcomes = np.vstack([treated_outcome.reshape(1, -1), control_outcomes])
    treatment = np.array([1] + [0] * n_control)

    return SCMData(
        outcomes=outcomes,
        treatment=treatment,
        treatment_period=treatment_period,
        true_att=true_att,
        n_units=n_units,
        n_periods=n_periods,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        dgp_type="null_effect",
        expected_fit="good",
        true_weights=true_weights,
    )
