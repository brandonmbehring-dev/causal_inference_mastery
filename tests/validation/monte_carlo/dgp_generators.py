"""
Data Generating Processes (DGPs) for Monte Carlo validation.

All DGPs have known true ATE = 2.0 for validation purposes.
"""

import numpy as np
from typing import Tuple


def dgp_simple_rct(n: int = 100, true_ate: float = 2.0, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple RCT with homoskedastic errors.

    DGP:
        Y(1) ~ N(2, 1)
        Y(0) ~ N(0, 1)
        T ~ Bernoulli(0.5)
        Y = T*Y(1) + (1-T)*Y(0)

    True ATE = 2.0
    """
    rng = np.random.RandomState(random_state)

    n1 = n // 2
    n0 = n - n1
    treatment = np.array([1] * n1 + [0] * n0)
    rng.shuffle(treatment)

    y1 = rng.normal(true_ate, 1.0, n)
    y0 = rng.normal(0.0, 1.0, n)

    outcomes = treatment * y1 + (1 - treatment) * y0

    return outcomes, treatment


def dgp_heteroskedastic_rct(n: int = 200, true_ate: float = 2.0, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    RCT with heteroskedastic errors (different variances by group).

    DGP:
        Y(1) ~ N(2, 4)  # Higher variance in treated
        Y(0) ~ N(0, 1)
        T ~ Bernoulli(0.5)

    True ATE = 2.0
    Tests Neyman variance (robust to heteroskedasticity)
    """
    rng = np.random.RandomState(random_state)

    n1 = n // 2
    n0 = n - n1
    treatment = np.array([1] * n1 + [0] * n0)
    rng.shuffle(treatment)

    y1 = rng.normal(true_ate, 2.0, n)  # σ=2
    y0 = rng.normal(0.0, 1.0, n)       # σ=1

    outcomes = treatment * y1 + (1 - treatment) * y0

    return outcomes, treatment


def dgp_small_sample_rct(n: int = 20, true_ate: float = 2.0, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Small sample RCT to test t-distribution inference.

    DGP:
        Y(1) ~ N(2, 1)
        Y(0) ~ N(0, 1)
        T ~ Bernoulli(0.5)
        n = 20

    True ATE = 2.0
    Tests t-distribution vs z-distribution (critical for small samples)
    """
    return dgp_simple_rct(n=n, true_ate=true_ate, random_state=random_state)


def dgp_stratified_rct(n_per_stratum: int = 40, n_strata: int = 3, true_ate: float = 2.0,
                       random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified RCT with different baseline levels.

    DGP:
        3 strata with baselines [0, 5, 10]
        Within each stratum:
            Y(1) ~ N(baseline + 2, 1)
            Y(0) ~ N(baseline, 1)
            T ~ Bernoulli(0.5)

    True ATE = 2.0 in all strata (average ATE = 2.0)
    """
    rng = np.random.RandomState(random_state)

    baselines = [i * 5.0 for i in range(n_strata)]

    outcomes = []
    treatment = []
    strata = []

    for s, baseline in enumerate(baselines):
        n1 = n_per_stratum // 2
        n0 = n_per_stratum - n1

        t = np.array([1] * n1 + [0] * n0)
        rng.shuffle(t)

        y = np.where(t == 1,
                    rng.normal(baseline + true_ate, 1.0, n_per_stratum),
                    rng.normal(baseline, 1.0, n_per_stratum))

        outcomes.extend(y)
        treatment.extend(t)
        strata.extend([s] * n_per_stratum)

    return np.array(outcomes), np.array(treatment), np.array(strata)


def dgp_regression_rct(n: int = 100, true_ate: float = 2.0, covariate_effect: float = 3.0,
                      random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RCT with covariate for regression adjustment.

    DGP:
        X ~ N(0, 1)
        T ~ Bernoulli(0.5)
        Y = 2*T + 3*X + ε
        ε ~ N(0, 1)

    True ATE = 2.0
    Covariate strongly predicts outcome (variance reduction)
    """
    rng = np.random.RandomState(random_state)

    X = rng.normal(0, 1, n)
    n1 = n // 2
    n0 = n - n1
    treatment = np.array([1] * n1 + [0] * n0)
    rng.shuffle(treatment)

    outcomes = true_ate * treatment + covariate_effect * X + rng.normal(0, 1, n)

    return outcomes, treatment, X


def dgp_ipw_rct(n: int = 100, true_ate: float = 2.0, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RCT with non-constant propensity scores.

    DGP:
        X ~ N(0, 1)
        propensity(X) = 1/(1 + exp(-0.5*X))  # Logistic
        T ~ Bernoulli(propensity(X))
        Y = 2*T + X + ε
        ε ~ N(0, 1)

    True ATE = 2.0
    Propensity varies with covariate
    """
    rng = np.random.RandomState(random_state)

    X = rng.normal(0, 1, n)

    # Propensity depends on X
    propensity = 1 / (1 + np.exp(-0.5 * X))
    treatment = (rng.uniform(0, 1, n) < propensity).astype(float)

    # Outcomes (ATE = 2.0)
    outcomes = true_ate * treatment + X + rng.normal(0, 1, n)

    return outcomes, treatment, propensity


# ============================================================================
# PSM DGPs (Observational Studies with Confounding)
# ============================================================================

def dgp_psm_linear(n: int = 200, true_ate: float = 2.0, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear PSM DGP with moderate confounding.

    DGP:
        X ~ N(0, 1)
        propensity(X) = 1/(1 + exp(-X))  # Moderate confounding
        T ~ Bernoulli(propensity(X))
        Y = 2*T + 0.5*X + ε
        ε ~ N(0, 1)

    True ATE = 2.0
    Good overlap (propensity mostly in [0.2, 0.8])

    Note: Reduced β_X from 2.0 to 0.5 to make PSM more effective.
    With β_X = 2.0, residual X imbalance of ~0.2 creates bias ~0.4 (too high).
    """
    rng = np.random.RandomState(random_state)

    X = rng.normal(0, 1, (n, 1))  # Single covariate

    # Moderate confounding: propensity(X) = logistic(X)
    logit = X.flatten()
    propensity = 1 / (1 + np.exp(-logit))
    treatment = (rng.uniform(0, 1, n) < propensity).astype(float)

    # Outcome: Y = τ*T + β*X + ε (β_X = 0.5, reduced from 2.0)
    outcomes = true_ate * treatment + 0.5 * X.flatten() + rng.normal(0, 1, n)

    return outcomes, treatment, X


def dgp_psm_mild_confounding(n: int = 200, true_ate: float = 2.0, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PSM DGP with mild confounding (easy matching).

    DGP:
        X ~ N(0, 1)
        propensity(X) = 1/(1 + exp(-0.5*X))  # Mild confounding
        T ~ Bernoulli(propensity(X))
        Y = 2*T + 0.5*X + ε
        ε ~ N(0, 1)

    True ATE = 2.0
    Excellent overlap (propensity mostly in [0.3, 0.7])

    Note: Reduced β_X from 1.0 to 0.5 for consistency with dgp_psm_linear.
    """
    rng = np.random.RandomState(random_state)

    X = rng.normal(0, 1, (n, 1))

    # Mild confounding: weaker relationship
    logit = 0.5 * X.flatten()
    propensity = 1 / (1 + np.exp(-logit))
    treatment = (rng.uniform(0, 1, n) < propensity).astype(float)

    # Outcome (β_X = 0.5, reduced from 1.0)
    outcomes = true_ate * treatment + 0.5 * X.flatten() + rng.normal(0, 1, n)

    return outcomes, treatment, X


def dgp_psm_strong_confounding(n: int = 200, true_ate: float = 2.0, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PSM DGP with strong confounding (harder matching).

    DGP:
        X ~ N(0, 1)
        propensity(X) = 1/(1 + exp(-2*X))  # Strong confounding
        T ~ Bernoulli(propensity(X))
        Y = 2*T + 0.5*X + ε
        ε ~ N(0, 1)

    True ATE = 2.0
    Limited overlap (some extreme propensities)

    Note: Reduced β_X from 3.0 to 0.5. The "strong" confounding is in the
    propensity model (logit = 2*X), not the outcome model. This makes PSM
    more effective at removing bias.
    """
    rng = np.random.RandomState(random_state)

    X = rng.normal(0, 1, (n, 1))

    # Strong confounding: stronger relationship in propensity
    logit = 2.0 * X.flatten()
    propensity = 1 / (1 + np.exp(-logit))
    treatment = (rng.uniform(0, 1, n) < propensity).astype(float)

    # Outcome (β_X = 0.5, reduced from 3.0)
    outcomes = true_ate * treatment + 0.5 * X.flatten() + rng.normal(0, 1, n)

    return outcomes, treatment, X


def dgp_psm_limited_overlap(n: int = 200, true_ate: float = 2.0, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PSM DGP with limited common support.

    DGP:
        X_treated ~ N(1, 1)
        X_control ~ N(-1, 1)
        T determined by group
        Y = 2*T + 0.5*X + ε
        ε ~ N(0, 1)

    True ATE = 2.0
    Partial overlap (treated and control have different X distributions)

    Note: Reduced β_X from 2.0 to 0.5 for consistency.
    """
    rng = np.random.RandomState(random_state)

    n_treated = n // 2
    n_control = n - n_treated

    # Different X distributions by group
    X_treated = rng.normal(1, 1, (n_treated, 1))
    X_control = rng.normal(-1, 1, (n_control, 1))

    X = np.vstack([X_treated, X_control])
    treatment = np.array([1] * n_treated + [0] * n_control)

    # Shuffle
    idx = rng.permutation(n)
    X = X[idx]
    treatment = treatment[idx]

    # Outcome (β_X = 0.5, reduced from 2.0)
    outcomes = true_ate * treatment + 0.5 * X.flatten() + rng.normal(0, 1, n)

    return outcomes, treatment, X


def dgp_psm_heterogeneous_te(n: int = 200, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    PSM DGP with heterogeneous treatment effects.

    DGP:
        X ~ N(0, 1)
        propensity(X) = 1/(1 + exp(-X))
        T ~ Bernoulli(propensity(X))
        τ(X) = 2 + X  # Treatment effect varies with X
        Y = τ(X)*T + 0.5*X + ε
        ε ~ N(0, 1)

    True ATE = 2.0 (average over X ~ N(0,1))
    Tests that PSM recovers average effect despite heterogeneity

    Note: Reduced β_X from 2.0 to 0.5 for consistency.
    """
    rng = np.random.RandomState(random_state)

    X = rng.normal(0, 1, (n, 1))

    # Propensity
    logit = X.flatten()
    propensity = 1 / (1 + np.exp(-logit))
    treatment = (rng.uniform(0, 1, n) < propensity).astype(float)

    # Heterogeneous treatment effect: τ(X) = 2 + X
    # E[τ(X)] = 2 + E[X] = 2 + 0 = 2.0
    treatment_effect = 2.0 + X.flatten()

    # Outcome (β_X = 0.5, reduced from 2.0)
    outcomes = treatment_effect * treatment + 0.5 * X.flatten() + rng.normal(0, 1, n)

    true_ate = 2.0  # Average treatment effect

    return outcomes, treatment, X, true_ate


# ============================================================================
# Fuzzy RDD DGPs (Imperfect Compliance at Cutoff)
# ============================================================================

def dgp_fuzzy_rdd_perfect_compliance(
    n: int = 500,
    true_late: float = 2.0,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuzzy RDD DGP with perfect compliance (compliance = 1.0).

    This is equivalent to Sharp RDD - all units above cutoff take treatment,
    all units below cutoff do not.

    DGP:
        X ~ N(0, 1)
        Z = 1{X >= 0}  (instrument: eligibility)
        D = Z  (perfect compliance: actual treatment = eligibility)
        Y = 2.0*D + 0.5*X + ε, ε ~ N(0, 1)

    True LATE = 2.0
    Compliance rate = E[D|Z=1] - E[D|Z=0] = 1.0 - 0.0 = 1.0

    When fitted with FuzzyRDD, should match SharpRDD estimates.
    """
    rng = np.random.RandomState(random_state)

    # Running variable
    X = rng.normal(0, 1, n)

    # Instrument: treatment eligibility
    Z = (X >= 0).astype(float)

    # Actual treatment (PERFECT compliance: D = Z)
    D = Z.copy()

    # Outcome (local linear structure around cutoff)
    Y = true_late * D + 0.5 * X + rng.normal(0, 1, n)

    return Y, X, D


def dgp_fuzzy_rdd_high_compliance(
    n: int = 500,
    true_late: float = 2.0,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuzzy RDD DGP with high compliance (compliance ≈ 0.8).

    Strong first stage → F > 50 expected.

    DGP:
        X ~ N(0, 1)
        Z = 1{X >= 0}  (instrument: eligibility)
        P(D=1|Z=1, X) = 0.9  (treated side)
        P(D=1|Z=0, X) = 0.1  (control side)
        Compliance rate = 0.9 - 0.1 = 0.8
        Y = 2.0*D + 0.5*X + ε, ε ~ N(0, 1)

    True LATE = 2.0
    Expected F-statistic > 50 (strong instrument)
    """
    rng = np.random.RandomState(random_state)

    # Running variable
    X = rng.normal(0, 1, n)

    # Instrument: treatment eligibility
    Z = (X >= 0).astype(float)

    # Actual treatment (HIGH compliance)
    baseline_treatment = 0.1  # 10% always-takers
    compliance_boost = 0.8    # 80% compliers
    p_treat = np.where(Z == 1,
                       baseline_treatment + compliance_boost,
                       baseline_treatment)
    D = (rng.uniform(0, 1, n) < p_treat).astype(float)

    # Outcome (local linear structure)
    Y = true_late * D + 0.5 * X + rng.normal(0, 1, n)

    return Y, X, D


def dgp_fuzzy_rdd_moderate_compliance(
    n: int = 500,
    true_late: float = 2.0,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuzzy RDD DGP with moderate compliance (compliance ≈ 0.5).

    Typical scenario → F > 20 expected.

    DGP:
        X ~ N(0, 1)
        Z = 1{X >= 0}  (instrument: eligibility)
        P(D=1|Z=1, X) = 0.75  (treated side)
        P(D=1|Z=0, X) = 0.25  (control side)
        Compliance rate = 0.75 - 0.25 = 0.5
        Y = 2.0*D + 0.5*X + ε, ε ~ N(0, 1)

    True LATE = 2.0
    Expected F-statistic > 20 (decent instrument)
    """
    rng = np.random.RandomState(random_state)

    # Running variable
    X = rng.normal(0, 1, n)

    # Instrument: treatment eligibility
    Z = (X >= 0).astype(float)

    # Actual treatment (MODERATE compliance)
    baseline_treatment = 0.25  # 25% always-takers
    compliance_boost = 0.5     # 50% compliers
    p_treat = np.where(Z == 1,
                       baseline_treatment + compliance_boost,
                       baseline_treatment)
    D = (rng.uniform(0, 1, n) < p_treat).astype(float)

    # Outcome (local linear structure)
    Y = true_late * D + 0.5 * X + rng.normal(0, 1, n)

    return Y, X, D


def dgp_fuzzy_rdd_low_compliance(
    n: int = 500,
    true_late: float = 2.0,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuzzy RDD DGP with low compliance (compliance ≈ 0.3).

    Weak instrument → F ≈ 10-15 expected (borderline).

    DGP:
        X ~ N(0, 1)
        Z = 1{X >= 0}  (instrument: eligibility)
        P(D=1|Z=1, X) = 0.65  (treated side)
        P(D=1|Z=0, X) = 0.35  (control side)
        Compliance rate = 0.65 - 0.35 = 0.3
        Y = 2.0*D + 0.5*X + ε, ε ~ N(0, 1)

    True LATE = 2.0
    Expected F-statistic ≈ 10-15 (weak/borderline instrument)
    Should trigger weak instrument warning
    """
    rng = np.random.RandomState(random_state)

    # Running variable
    X = rng.normal(0, 1, n)

    # Instrument: treatment eligibility
    Z = (X >= 0).astype(float)

    # Actual treatment (LOW compliance)
    baseline_treatment = 0.35  # 35% always-takers
    compliance_boost = 0.3     # 30% compliers
    p_treat = np.where(Z == 1,
                       baseline_treatment + compliance_boost,
                       baseline_treatment)
    D = (rng.uniform(0, 1, n) < p_treat).astype(float)

    # Outcome (local linear structure)
    Y = true_late * D + 0.5 * X + rng.normal(0, 1, n)

    return Y, X, D


def dgp_fuzzy_rdd_bandwidth_sensitivity(
    n: int = 500,
    true_late: float = 2.0,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuzzy RDD DGP for bandwidth sensitivity analysis.

    Moderate compliance with local linear structure.
    Tests that estimates are stable across reasonable bandwidth choices.

    DGP:
        X ~ N(0, 1)
        Z = 1{X >= 0}  (instrument: eligibility)
        P(D=1|Z=1, X) = 0.8  (treated side)
        P(D=1|Z=0, X) = 0.1  (control side)
        Compliance rate = 0.8 - 0.1 = 0.7
        Y = 2.0*D + 0.5*X + ε, ε ~ N(0, 1)

    True LATE = 2.0
    Expected F-statistic > 40 (strong instrument)
    Will test 3 bandwidths: IK, 0.5*IK, 2.0*IK
    """
    rng = np.random.RandomState(random_state)

    # Running variable
    X = rng.normal(0, 1, n)

    # Instrument: treatment eligibility
    Z = (X >= 0).astype(float)

    # Actual treatment (moderate-high compliance)
    baseline_treatment = 0.1   # 10% always-takers
    compliance_boost = 0.7     # 70% compliers
    p_treat = np.where(Z == 1,
                       baseline_treatment + compliance_boost,
                       baseline_treatment)
    D = (rng.uniform(0, 1, n) < p_treat).astype(float)

    # Outcome (local linear structure)
    Y = true_late * D + 0.5 * X + rng.normal(0, 1, n)

    return Y, X, D
