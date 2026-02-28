"""Capture golden results from Python library implementations.

These golden results will be used to validate Julia from-scratch implementations.
We run all 5 estimators on carefully designed test cases and save the outputs.

Following Brandon's principle: NEVER FAIL SILENTLY. All errors explicit.
"""

import numpy as np
import json
from pathlib import Path

from src.causal_inference.rct.estimators import simple_ate
from src.causal_inference.rct.estimators_stratified import stratified_ate
from src.causal_inference.rct.estimators_regression import regression_adjusted_ate
from src.causal_inference.rct.estimators_permutation import permutation_test
from src.causal_inference.rct.estimators_ipw import ipw_ate

# Additional family imports for Session 180 golden reference expansion
from src.causal_inference.psm import psm_ate
from src.causal_inference.iv import TwoStageLeastSquares, LIML
from src.causal_inference.did import did_2x2, event_study
from src.causal_inference.rdd import SharpRDD, FuzzyRDD

# Session 185: Full golden reference expansion imports
# Observational
from src.causal_inference.observational import ipw_ate_observational, dr_ate

# Sensitivity
from src.causal_inference.sensitivity import e_value, rosenbaum_bounds

# CATE (meta-learners)
from src.causal_inference.cate.meta_learners import (
    s_learner,
    t_learner,
    x_learner,
)

# RKD
from src.causal_inference.rkd import SharpRKD

# SCM
from src.causal_inference.scm import synthetic_control

# Bunching
from src.causal_inference.bunching import bunching_estimator

# Shift-Share
from src.causal_inference.shift_share import shift_share_iv

# Time Series
from src.causal_inference.timeseries import (
    var_estimate,
    granger_causality,
)
from src.causal_inference.timeseries.vecm import vecm_estimate

# Selection
from src.causal_inference.selection import heckman_two_step

# Bounds
from src.causal_inference.bounds import manski_worst_case, lee_bounds

# Mediation
from src.causal_inference.mediation import mediation_analysis

# DTR
from src.causal_inference.dtr import q_learning, a_learning, DTRData

# MTE
from src.causal_inference.mte import late_estimator

# QTE
from src.causal_inference.qte import unconditional_qte

# Control Function
from src.causal_inference.control_function import control_function_ate

# Dynamic DML
from src.causal_inference.dynamic import dynamic_dml

# Bayesian
from src.causal_inference.bayesian import bayesian_ate

# Discovery
from src.causal_inference.discovery import pc_algorithm


def capture_golden_results():
    """
    Capture golden results from all 5 estimators on reference datasets.

    Returns
    -------
    dict
        Dictionary mapping test case names to estimator results.
    """
    golden_results = {}

    # ============================================================================
    # Test Case 1: Balanced RCT with moderate effect
    # ============================================================================

    np.random.seed(42)
    n = 100
    treatment_balanced = np.array([1, 0] * (n // 2))
    outcomes_balanced = 5 * treatment_balanced + np.random.normal(10, 2, n)

    golden_results["balanced_rct"] = {
        "description": "Balanced RCT, n=100, true ATE=5, noise sd=2",
        "data": {
            "treatment": treatment_balanced.tolist(),
            "outcomes": outcomes_balanced.tolist(),
        },
        "simple_ate": simple_ate(outcomes_balanced, treatment_balanced),
    }

    # ============================================================================
    # Test Case 2: Stratified RCT (2 strata, different baselines)
    # ============================================================================

    strata = np.array([1]*50 + [2]*50)
    treatment_stratified = np.array([1, 0] * 25 + [1, 0] * 25)

    # Stratum 1: High baseline (Y ~ 100)
    # Stratum 2: Low baseline (Y ~ 10)
    # Both have ATE = 5
    outcomes_stratified = np.zeros(100)
    outcomes_stratified[:50] = 100 + 5 * treatment_stratified[:50] + np.random.normal(0, 2, 50)
    outcomes_stratified[50:] = 10 + 5 * treatment_stratified[50:] + np.random.normal(0, 2, 50)

    golden_results["stratified_rct"] = {
        "description": "Stratified RCT, 2 strata with different baselines, true ATE=5",
        "data": {
            "treatment": treatment_stratified.tolist(),
            "outcomes": outcomes_stratified.tolist(),
            "strata": strata.tolist(),
        },
        "simple_ate": simple_ate(outcomes_stratified, treatment_stratified),
        "stratified_ate": stratified_ate(outcomes_stratified, treatment_stratified, strata),
    }

    # ============================================================================
    # Test Case 3: RCT with covariate (for regression adjustment)
    # ============================================================================

    X_single = np.random.normal(5, 2, 100)
    treatment_reg = np.array([1, 0] * 50)
    # Y = 3*T + 2*X + noise (ATE = 3)
    outcomes_reg = 3 * treatment_reg + 2 * X_single + np.random.normal(0, 1, 100)

    golden_results["regression_rct"] = {
        "description": "RCT with covariate, Y = 3*T + 2*X + noise, true ATE=3",
        "data": {
            "treatment": treatment_reg.tolist(),
            "outcomes": outcomes_reg.tolist(),
            "covariate": X_single.tolist(),
        },
        "simple_ate": simple_ate(outcomes_reg, treatment_reg),
        "regression_adjusted_ate": regression_adjusted_ate(outcomes_reg, treatment_reg, X_single),
    }

    # ============================================================================
    # Test Case 4: Small sample for permutation test (exact)
    # ============================================================================

    treatment_small = np.array([1, 1, 1, 0, 0, 0])
    outcomes_small = np.array([10.0, 12.0, 11.0, 4.0, 5.0, 3.0])

    golden_results["permutation_small"] = {
        "description": "Small sample (n=6) for exact permutation test, strong effect",
        "data": {
            "treatment": treatment_small.tolist(),
            "outcomes": outcomes_small.tolist(),
        },
        "simple_ate": simple_ate(outcomes_small, treatment_small),
        "permutation_test_exact": permutation_test(
            outcomes_small, treatment_small, n_permutations=None, random_seed=42
        ),
        "permutation_test_monte_carlo": permutation_test(
            outcomes_small, treatment_small, n_permutations=1000, random_seed=42
        ),
    }

    # ============================================================================
    # Test Case 5: IPW with varying propensity
    # ============================================================================

    treatment_ipw = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    outcomes_ipw = np.array([10.0, 12.0, 9.0, 11.0, 4.0, 5.0, 3.0, 6.0])
    propensity_ipw = np.array([0.7, 0.6, 0.8, 0.5, 0.6, 0.7, 0.5, 0.8])

    golden_results["ipw_varying"] = {
        "description": "IPW with varying propensity scores, n=8",
        "data": {
            "treatment": treatment_ipw.tolist(),
            "outcomes": outcomes_ipw.tolist(),
            "propensity": propensity_ipw.tolist(),
        },
        "simple_ate": simple_ate(outcomes_ipw, treatment_ipw),
        "ipw_ate": ipw_ate(outcomes_ipw, treatment_ipw, propensity_ipw),
    }

    # ============================================================================
    # Test Case 6: Large sample for Monte Carlo validation
    # ============================================================================

    np.random.seed(123)
    n_large = 500
    treatment_large = np.array([1, 0] * (n_large // 2))
    X_large = np.random.normal(0, 3, n_large)
    outcomes_large = 4 * treatment_large + 1.5 * X_large + np.random.normal(0, 2, n_large)
    propensity_large = np.full(n_large, 0.5)

    golden_results["large_sample"] = {
        "description": "Large sample (n=500) with covariate, true ATE=4",
        "data": {
            "treatment": treatment_large.tolist(),
            "outcomes": outcomes_large.tolist(),
            "covariate": X_large.tolist(),
            "propensity": propensity_large.tolist(),
        },
        "simple_ate": simple_ate(outcomes_large, treatment_large),
        "regression_adjusted_ate": regression_adjusted_ate(outcomes_large, treatment_large, X_large),
        "ipw_ate": ipw_ate(outcomes_large, treatment_large, propensity_large),
        "permutation_test_monte_carlo": permutation_test(
            outcomes_large, treatment_large, n_permutations=1000, random_seed=123
        ),
    }

    # ============================================================================
    # PSM Test Cases (Session 180)
    # ============================================================================
    golden_results.update(capture_psm_golden())

    # ============================================================================
    # IV Test Cases (Session 180)
    # ============================================================================
    golden_results.update(capture_iv_golden())

    # ============================================================================
    # DiD Test Cases (Session 180)
    # ============================================================================
    golden_results.update(capture_did_golden())

    # ============================================================================
    # RDD Test Cases (Session 180)
    # ============================================================================
    golden_results.update(capture_rdd_golden())

    # ============================================================================
    # Session 185: Full Golden Reference Expansion
    # ============================================================================

    # Priority 1: Core Methods
    golden_results.update(capture_observational_golden())
    golden_results.update(capture_sensitivity_golden())
    golden_results.update(capture_cate_golden())

    # Priority 2: Natural Experiments
    golden_results.update(capture_rkd_golden())
    golden_results.update(capture_scm_golden())
    golden_results.update(capture_bunching_golden())
    golden_results.update(capture_shift_share_golden())

    # Priority 3: Time Series
    golden_results.update(capture_timeseries_golden())
    golden_results.update(capture_vecm_golden())

    # Priority 4: Advanced Methods
    golden_results.update(capture_selection_golden())
    golden_results.update(capture_bounds_golden())
    golden_results.update(capture_mediation_golden())
    golden_results.update(capture_dtr_golden())
    golden_results.update(capture_mte_golden())
    golden_results.update(capture_qte_golden())
    golden_results.update(capture_control_function_golden())
    golden_results.update(capture_dynamic_dml_golden())
    golden_results.update(capture_bayesian_golden())
    golden_results.update(capture_discovery_golden())

    return golden_results


def capture_psm_golden() -> dict:
    """
    Capture PSM golden reference results.

    Test cases:
    - Simple matching with strong treatment effect
    - Matching with balance check
    """
    results = {}
    np.random.seed(42)

    # Test Case: PSM with observable confounding
    n = 200
    X = np.random.randn(n, 2)  # 2 covariates
    propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    treatment = (np.random.rand(n) < propensity).astype(int)

    # Outcome depends on treatment and covariates
    # True ATE = 3.0
    outcome = 3.0 * treatment + 2.0 * X[:, 0] + 1.5 * X[:, 1] + np.random.randn(n) * 0.5

    # Run PSM
    psm_result = psm_ate(outcome, treatment, X)

    results["psm_observable_confounding"] = {
        "description": "PSM with observable confounding, true ATE=3.0, n=200",
        "data": {
            "treatment": treatment.tolist(),
            "outcomes": outcome.tolist(),
            "covariates": X.tolist(),
        },
        "psm_ate": psm_result,
    }

    return results


def capture_iv_golden() -> dict:
    """
    Capture IV golden reference results.

    Test cases:
    - 2SLS with single instrument
    - LIML comparison
    """
    results = {}
    np.random.seed(123)

    # Test Case: IV with single instrument
    n = 300
    z = np.random.randn(n)  # Instrument
    u = np.random.randn(n) * 0.5  # Confounder

    # Endogenous treatment: correlated with u
    d = 0.8 * z + 0.5 * u + np.random.randn(n) * 0.3

    # Outcome: true effect = 2.0
    y = 2.0 * d + u + np.random.randn(n) * 0.5

    # Reshape for estimators (D = treatment, Z = instrument)
    D = d.reshape(-1, 1)
    Z = z.reshape(-1, 1)

    # Run 2SLS
    tsls = TwoStageLeastSquares(inference="robust")
    tsls.fit(y, D, Z)

    # Run LIML
    liml = LIML(inference="robust")
    liml.fit(y, D, Z)

    results["iv_single_instrument"] = {
        "description": "IV with single instrument, true effect=2.0, n=300",
        "data": {
            "y": y.tolist(),
            "d": d.tolist(),
            "z": z.tolist(),
        },
        "tsls": {
            "coefficient": float(tsls.coef_[0]),
            "se": float(tsls.se_[0]) if hasattr(tsls, "se_") else None,
        },
        "liml": {
            "coefficient": float(liml.coef_[0]),
            "se": float(liml.se_[0]) if hasattr(liml, "se_") else None,
        },
    }

    return results


def capture_did_golden() -> dict:
    """
    Capture DiD golden reference results.

    Test cases:
    - Classic 2x2 DiD
    """
    results = {}
    np.random.seed(456)

    # Test Case: Classic 2x2 DiD
    n_units = 50  # 25 treated, 25 control
    n_periods = 2  # pre and post

    # Panel structure
    n = n_units * n_periods
    unit_id = np.repeat(np.arange(n_units), n_periods)
    treatment = np.repeat(np.array([1] * 25 + [0] * 25), n_periods)
    post = np.tile([0, 1], n_units)

    # Parallel trends with treatment effect = 5.0
    baseline = 10 + 2 * treatment + np.random.randn(n) * 0.5
    outcome = baseline + 3 * post + 5.0 * (treatment * post) + np.random.randn(n) * 0.5

    did_result = did_2x2(
        outcomes=outcome,
        treatment=treatment,
        post=post,
        unit_id=unit_id,
    )

    results["did_classic_2x2"] = {
        "description": "Classic 2x2 DiD, true ATT=5.0, n_units=50",
        "data": {
            "outcome": outcome.tolist(),
            "treatment": treatment.tolist(),
            "post": post.tolist(),
            "unit_id": unit_id.tolist(),
        },
        "did_2x2": did_result,
    }

    return results


def capture_rdd_golden() -> dict:
    """
    Capture RDD golden reference results.

    Test cases:
    - Sharp RDD at cutoff 0
    - Fuzzy RDD
    """
    results = {}
    np.random.seed(789)

    # Test Case: Sharp RDD
    n = 200
    running = np.random.uniform(-1, 1, n)

    # Continuous outcome with jump at cutoff
    # True treatment effect = 4.0
    treatment = (running >= 0).astype(int)
    outcome = 2.0 * running + 4.0 * treatment + np.random.randn(n) * 0.5

    # Fit Sharp RDD
    rdd = SharpRDD(cutoff=0.0, bandwidth=0.5, inference="robust")
    rdd.fit(outcome, running)

    results["rdd_sharp"] = {
        "description": "Sharp RDD at cutoff=0, true effect=4.0, n=200",
        "data": {
            "outcome": outcome.tolist(),
            "running": running.tolist(),
            "treatment": treatment.tolist(),
        },
        "sharp_rdd": {
            "estimate": float(rdd.coef_),
            "se": float(rdd.se_) if hasattr(rdd, "se_") else None,
            "bandwidth": 0.5,
        },
    }

    # Test Case: Fuzzy RDD
    np.random.seed(321)
    n = 200
    running = np.random.uniform(-1, 1, n)

    # Fuzzy treatment: probability jumps at cutoff
    prob = 0.2 + 0.6 * (running >= 0).astype(float)
    treatment = (np.random.rand(n) < prob).astype(int)

    # True LATE = 3.5
    outcome = 1.5 * running + 3.5 * treatment + np.random.randn(n) * 0.5

    fuzzy = FuzzyRDD(cutoff=0.0, bandwidth=0.5, inference="robust")
    fuzzy.fit(outcome, running, treatment)

    results["rdd_fuzzy"] = {
        "description": "Fuzzy RDD at cutoff=0, true LATE=3.5, n=200",
        "data": {
            "outcome": outcome.tolist(),
            "running": running.tolist(),
            "treatment": treatment.tolist(),
        },
        "fuzzy_rdd": {
            "estimate": float(fuzzy.coef_),
            "se": float(fuzzy.se_) if hasattr(fuzzy, "se_") else None,
            "bandwidth": 0.5,
        },
    }

    return results


# ============================================================================
# Session 185: Golden Reference Expansion Functions
# ============================================================================


def capture_observational_golden() -> dict:
    """
    Capture Observational golden reference results.

    Test cases:
    - IPW with known propensity
    - Doubly robust estimation
    """
    results = {}
    np.random.seed(1001)

    n = 200
    X = np.random.randn(n, 2)

    # True propensity model
    propensity = 1 / (1 + np.exp(-(0.5 * X[:, 0] + 0.3 * X[:, 1])))
    treatment = (np.random.rand(n) < propensity).astype(int)

    # Outcome: true ATE = 2.5
    outcome = 2.5 * treatment + 1.0 * X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.5

    # Run estimators
    ipw_result = ipw_ate_observational(outcome, treatment, X)
    dr_result = dr_ate(outcome, treatment, X)

    results["observational_confounding"] = {
        "description": "Observational study with confounding, true ATE=2.5, n=200",
        "data": {
            "outcome": outcome.tolist(),
            "treatment": treatment.tolist(),
            "covariates": X.tolist(),
        },
        "ipw_ate": ipw_result,
        "dr_ate": dr_result,
    }

    return results


def capture_sensitivity_golden() -> dict:
    """
    Capture Sensitivity analysis golden reference results.

    Test cases:
    - E-value calculation
    - Rosenbaum bounds
    """
    results = {}

    # E-value test case with known estimate and CI
    # Strong effect: RR = 2.5, CI lower = 1.8
    e_val_result = e_value(estimate=2.5, ci_lower=1.8, effect_type="rr")

    results["sensitivity_evalue"] = {
        "description": "E-value for RR=2.5, CI_lower=1.8",
        "data": {
            "estimate": 2.5,
            "ci_lower": 1.8,
            "effect_type": "rr",
        },
        "e_value": e_val_result,
    }

    # Rosenbaum bounds test case
    np.random.seed(1002)
    n = 50
    # Generate matched pairs data
    treated_outcomes = np.random.randn(n) + 1.5  # Higher outcomes
    control_outcomes = np.random.randn(n) + 0.5  # Lower outcomes

    rosenbaum_result = rosenbaum_bounds(
        treated_outcomes,
        control_outcomes,
        gamma_range=(1.0, 2.0),
        n_gamma=10,
    )

    results["sensitivity_rosenbaum"] = {
        "description": "Rosenbaum bounds, gamma_range=(1.0, 2.0), n=50 pairs",
        "data": {
            "treated_outcomes": treated_outcomes.tolist(),
            "control_outcomes": control_outcomes.tolist(),
            "gamma_range": [1.0, 2.0],
        },
        "rosenbaum_bounds": rosenbaum_result,
    }

    return results


def capture_cate_golden() -> dict:
    """
    Capture CATE (heterogeneous treatment effects) golden reference results.

    Test cases:
    - S-learner
    - T-learner
    - X-learner
    """
    results = {}
    np.random.seed(1003)

    n = 300
    X = np.random.randn(n, 3)
    treatment = np.random.binomial(1, 0.5, n)

    # Heterogeneous treatment effect: CATE = 1 + 2*X[0]
    cate_true = 1.0 + 2.0 * X[:, 0]
    outcome = cate_true * treatment + 0.5 * X[:, 1] + np.random.randn(n) * 0.5

    # Run meta-learners
    s_result = s_learner(outcome, treatment, X)
    t_result = t_learner(outcome, treatment, X)
    x_result = x_learner(outcome, treatment, X)

    results["cate_heterogeneous"] = {
        "description": "CATE with heterogeneous effects, CATE=1+2*X[0], n=300",
        "data": {
            "outcome": outcome.tolist(),
            "treatment": treatment.tolist(),
            "covariates": X.tolist(),
        },
        "s_learner": s_result,
        "t_learner": t_result,
        "x_learner": x_result,
    }

    return results


def capture_rkd_golden() -> dict:
    """
    Capture RKD (regression kink design) golden reference results.

    Test cases:
    - Sharp RKD at kink point
    """
    results = {}
    np.random.seed(1004)

    n = 300
    running = np.random.uniform(-2, 2, n)

    # Treatment intensity with kink at 0: d = max(x, 0)
    treatment_intensity = np.maximum(running, 0)

    # Outcome: Y = 1*X + 2*D + noise (kink effect = 2.0)
    outcome = 1.0 * running + 2.0 * treatment_intensity + np.random.randn(n) * 0.3

    rkd = SharpRKD(cutoff=0.0, bandwidth=1.0)
    rkd_result = rkd.fit(outcome, running, treatment_intensity)

    results["rkd_sharp"] = {
        "description": "Sharp RKD at kink=0, true kink effect=2.0, n=300",
        "data": {
            "outcome": outcome.tolist(),
            "running": running.tolist(),
            "treatment_intensity": treatment_intensity.tolist(),
        },
        "sharp_rkd": rkd_result,
    }

    return results


def capture_scm_golden() -> dict:
    """
    Capture SCM (synthetic control) golden reference results.

    Test cases:
    - Basic synthetic control
    """
    results = {}
    np.random.seed(1005)

    # Panel data: 10 units, 20 periods, treatment at period 10
    n_units = 10
    n_periods = 20
    treatment_period = 10

    # Generate panel with common trend (unit 0 is treated)
    time_effects = np.cumsum(np.random.randn(n_periods) * 0.5)
    unit_effects = np.random.randn(n_units) * 2

    # Outcomes: (n_units, n_periods) 2D array
    outcomes = np.zeros((n_units, n_periods))
    for i in range(n_units):
        outcomes[i, :] = unit_effects[i] + time_effects + np.random.randn(n_periods) * 0.2
        # Add treatment effect = 3.0 to unit 0 after treatment period
        if i == 0:
            outcomes[i, treatment_period:] += 3.0

    # Treatment indicator: unit 0 is treated
    treatment = np.array([1] + [0] * (n_units - 1))

    # Run synthetic control
    scm_result = synthetic_control(
        outcomes=outcomes,
        treatment=treatment,
        treatment_period=treatment_period,
    )

    results["scm_basic"] = {
        "description": "Synthetic control, true ATT=3.0, 10 units, 20 periods",
        "data": {
            "outcomes": outcomes.tolist(),
            "treatment": treatment.tolist(),
            "treatment_period": treatment_period,
        },
        "synthetic_control": scm_result,
    }

    return results


def capture_bunching_golden() -> dict:
    """
    Capture Bunching estimator golden reference results.

    Test cases:
    - Bunching at tax kink
    """
    results = {}
    np.random.seed(1006)

    # Generate income distribution with bunching at threshold
    n = 1000
    kink_point = 50000.0
    bunching_width = 5000.0

    # Counterfactual: smooth distribution
    income_cf = np.random.exponential(40000, n) + 20000

    # Actual: bunching at kink (some people reduce income to kink)
    bunchers = (income_cf > kink_point) & (income_cf < kink_point * 1.2)
    income = income_cf.copy()
    income[bunchers] = kink_point + np.random.uniform(-500, 500, bunchers.sum())

    bunching_result = bunching_estimator(
        data=income,
        kink_point=kink_point,
        bunching_width=bunching_width,
        n_bins=50,
        random_state=42,
    )

    results["bunching_kink"] = {
        "description": "Bunching at tax kink=50000, n=1000",
        "data": {
            "income": income.tolist(),
            "kink_point": kink_point,
            "bunching_width": bunching_width,
        },
        "bunching": bunching_result,
    }

    return results


def capture_shift_share_golden() -> dict:
    """
    Capture Shift-Share IV golden reference results.

    Test cases:
    - Basic Bartik instrument
    """
    results = {}
    np.random.seed(1007)

    # 50 regions, 10 industries
    n_regions = 50
    n_industries = 10

    # Initial industry shares (sum to 1 per region)
    shares = np.random.dirichlet(np.ones(n_industries), n_regions)

    # National industry shocks
    shocks = np.random.randn(n_industries)

    # Bartik instrument: sum of share-weighted shocks
    bartik = shares @ shocks

    # Endogenous variable correlated with Bartik
    u = np.random.randn(n_regions) * 0.5
    D = 0.8 * bartik + u + np.random.randn(n_regions) * 0.3

    # Outcome: true effect = 1.5
    Y = 1.5 * D + u + np.random.randn(n_regions) * 0.5

    ssiv_result = shift_share_iv(Y, D, shares, shocks)

    results["shift_share_bartik"] = {
        "description": "Shift-share IV, true effect=1.5, 50 regions, 10 industries",
        "data": {
            "Y": Y.tolist(),
            "D": D.tolist(),
            "shares": shares.tolist(),
            "shocks": shocks.tolist(),
        },
        "shift_share_iv": ssiv_result,
    }

    return results


def capture_timeseries_golden() -> dict:
    """
    Capture Time Series golden reference results.

    Test cases:
    - VAR estimation
    - Granger causality
    """
    results = {}
    np.random.seed(1008)

    # Generate bivariate VAR(1) process
    n = 200
    A = np.array([[0.5, 0.2], [0.1, 0.6]])  # VAR coefficients

    Y = np.zeros((n, 2))
    Y[0] = np.random.randn(2)
    for t in range(1, n):
        Y[t] = A @ Y[t - 1] + np.random.randn(2) * 0.5

    # VAR estimation (lags=1, not max_lag)
    var_result = var_estimate(Y, lags=1)

    # Granger causality: does variable 1 cause variable 0?
    granger_result = granger_causality(Y, lags=1, cause_idx=1, effect_idx=0)

    results["timeseries_var"] = {
        "description": "VAR(1) with A=[[0.5,0.2],[0.1,0.6]], n=200",
        "data": {
            "Y": Y.tolist(),
        },
        "var_estimate": var_result,
        "granger_causality": granger_result,
    }

    return results


def capture_vecm_golden() -> dict:
    """
    Capture VECM golden reference results.

    Test cases:
    - Cointegrated series
    """
    results = {}
    np.random.seed(1009)

    # Generate cointegrated I(1) series
    n = 200
    e1 = np.random.randn(n) * 0.5
    e2 = np.random.randn(n) * 0.5

    # Common trend
    trend = np.cumsum(np.random.randn(n) * 0.3)

    # Cointegrated series: y1 = trend + e1, y2 = 2*trend + e2
    y1 = trend + e1
    y2 = 2 * trend + e2

    Y = np.column_stack([y1, y2])

    # lags=1, not max_lag
    vecm_result = vecm_estimate(Y, coint_rank=1, lags=1)

    results["vecm_cointegrated"] = {
        "description": "VECM with 2 cointegrated series, n=200",
        "data": {
            "Y": Y.tolist(),
        },
        "vecm_estimate": vecm_result,
    }

    return results


def capture_selection_golden() -> dict:
    """
    Capture Selection model golden reference results.

    Test cases:
    - Heckman two-step
    """
    results = {}
    np.random.seed(1010)

    n = 500
    X = np.random.randn(n, 2)
    Z = np.random.randn(n)  # Exclusion restriction

    # Selection equation
    selection_latent = 0.5 * X[:, 0] + 0.3 * Z + np.random.randn(n)
    selected = (selection_latent > 0).astype(int)

    # Outcome equation (only observed for selected)
    # True coefficients: beta = [1.5, 2.0]
    outcome_latent = 1.5 * X[:, 0] + 2.0 * X[:, 1] + 0.5 * (selection_latent > 0) + np.random.randn(n) * 0.5
    outcome = np.where(selected, outcome_latent, np.nan)

    # Selection covariates include exclusion restriction Z
    selection_covariates = np.column_stack([X, Z])

    # Outcome can have NaN for unselected, covariates must be full sample
    heckman_result = heckman_two_step(
        outcome=outcome,  # Full array with NaN for unselected
        selected=selected,
        selection_covariates=selection_covariates,
        outcome_covariates=X,  # Full sample covariates
    )

    results["selection_heckman"] = {
        "description": "Heckman two-step selection, n=500",
        "data": {
            "outcome": outcome.tolist(),
            "X": X.tolist(),
            "Z": Z.tolist(),
            "selected": selected.tolist(),
        },
        "heckman_two_step": heckman_result,
    }

    return results


def capture_bounds_golden() -> dict:
    """
    Capture Bounds (partial identification) golden reference results.

    Test cases:
    - Manski worst-case bounds (complete data)
    - Lee bounds (with selection/missing data)
    """
    results = {}
    np.random.seed(1011)

    n = 300
    treatment = np.random.binomial(1, 0.5, n)
    outcome = 2.0 * treatment + np.random.randn(n)

    # Manski bounds on COMPLETE data (no NaN) - uses outcome support for bounds
    manski_result = manski_worst_case(outcome, treatment, outcome_support=(-5.0, 5.0))

    # For Lee bounds: add selection with missing outcomes
    # Selection depends on treatment (monotonicity assumption)
    observed = np.ones(n, dtype=int)
    selection_prob = 0.9 * treatment + 0.7 * (1 - treatment)  # Treated more likely observed
    observed = (np.random.rand(n) < selection_prob).astype(int)

    lee_result = lee_bounds(
        outcome=outcome,  # Full outcome array
        treatment=treatment,
        observed=observed,
        monotonicity="positive",
        random_state=42,
    )

    # Create outcome_observed for test data (with None for unobserved)
    outcome_observed = np.where(observed == 1, outcome, np.nan)

    results["bounds_partial_id"] = {
        "description": "Partial identification bounds, n=300",
        "data": {
            "outcome": outcome.tolist(),  # Complete outcome for Manski
            "outcome_observed": [float(x) if not np.isnan(x) else None for x in outcome_observed],
            "treatment": treatment.tolist(),
            "observed": observed.tolist(),
        },
        "manski_bounds": manski_result,
        "lee_bounds": lee_result,
    }

    return results


def capture_mediation_golden() -> dict:
    """
    Capture Mediation analysis golden reference results.

    Test cases:
    - Baron-Kenny mediation
    """
    results = {}
    np.random.seed(1012)

    n = 300
    treatment = np.random.binomial(1, 0.5, n)

    # Mediator: M = 0.8*T + noise
    mediator = 0.8 * treatment + np.random.randn(n) * 0.5

    # Outcome: Y = 1.0*T + 0.6*M + noise
    # Total effect = 1.0 + 0.8*0.6 = 1.48
    # Direct effect = 1.0
    # Indirect effect = 0.8*0.6 = 0.48
    outcome = 1.0 * treatment + 0.6 * mediator + np.random.randn(n) * 0.5

    mediation_result = mediation_analysis(outcome, treatment, mediator)

    results["mediation_basic"] = {
        "description": "Mediation: direct=1.0, indirect=0.48, n=300",
        "data": {
            "outcome": outcome.tolist(),
            "treatment": treatment.tolist(),
            "mediator": mediator.tolist(),
        },
        "mediation_analysis": mediation_result,
    }

    return results


def capture_dtr_golden() -> dict:
    """
    Capture DTR (dynamic treatment regime) golden reference results.

    Test cases:
    - Q-learning
    - A-learning
    """
    results = {}
    np.random.seed(1013)

    n = 200

    # Generate DTR data
    X1 = np.random.randn(n, 2)  # Stage 1 covariates
    A1 = np.random.binomial(1, 0.5, n)  # Stage 1 treatment
    X2 = X1 + np.random.randn(n, 2) * 0.3  # Stage 2 covariates
    A2 = np.random.binomial(1, 0.5, n)  # Stage 2 treatment

    # Outcome depends on both stages (one outcome per stage)
    Y1 = 1.0 * A1 * (X1[:, 0] > 0) + np.random.randn(n) * 0.3
    Y2 = 1.5 * A2 * (X2[:, 0] > 0) + np.random.randn(n) * 0.3

    # Create DTRData structure: outcomes, treatments, covariates (all lists)
    dtr_data = DTRData(
        outcomes=[Y1, Y2],
        treatments=[A1, A2],
        covariates=[X1, X2],
    )

    q_result = q_learning(dtr_data)
    a_result = a_learning(dtr_data)

    results["dtr_two_stage"] = {
        "description": "Two-stage DTR, n=200",
        "data": {
            "X1": X1.tolist(),
            "X2": X2.tolist(),
            "A1": A1.tolist(),
            "A2": A2.tolist(),
            "Y1": Y1.tolist(),
            "Y2": Y2.tolist(),
        },
        "q_learning": q_result,
        "a_learning": a_result,
    }

    return results


def capture_mte_golden() -> dict:
    """
    Capture MTE (marginal treatment effect) golden reference results.

    Test cases:
    - LATE estimation with binary instrument
    """
    results = {}
    np.random.seed(1014)

    n = 400
    # Binary instrument
    Z = np.random.binomial(1, 0.5, n)
    U = np.random.randn(n)  # Unobserved heterogeneity

    # Selection: D = 1 if 0.5*Z + U > 0 (D correlated with Z)
    D = (0.5 * Z + 0.3 * U > 0.2).astype(int)

    # Heterogeneous treatment effect
    # True LATE around 2.0
    outcome = 2.0 * D + 0.5 * U + np.random.randn(n) * 0.5

    late_result = late_estimator(outcome, D, Z)

    results["mte_late"] = {
        "description": "LATE estimation with binary IV, n=400",
        "data": {
            "outcome": outcome.tolist(),
            "treatment": D.tolist(),
            "instrument": Z.tolist(),
        },
        "late_estimate": late_result,
    }

    return results


def capture_qte_golden() -> dict:
    """
    Capture QTE (quantile treatment effects) golden reference results.

    Test cases:
    - Unconditional QTE at multiple quantiles
    """
    results = {}
    np.random.seed(1015)

    n = 300
    treatment = np.random.binomial(1, 0.5, n)

    # Location-scale shift: treatment increases mean by 1.0 and variance
    outcome = treatment * (1.0 + 0.5 * np.random.randn(n)) + (1 - treatment) * np.random.randn(n)

    # Run QTE at 3 quantiles (function takes single quantile)
    qte_25 = unconditional_qte(outcome, treatment, quantile=0.25, random_state=42)
    qte_50 = unconditional_qte(outcome, treatment, quantile=0.50, random_state=42)
    qte_75 = unconditional_qte(outcome, treatment, quantile=0.75, random_state=42)

    results["qte_unconditional"] = {
        "description": "Unconditional QTE at 25th, 50th, 75th percentiles, n=300",
        "data": {
            "outcome": outcome.tolist(),
            "treatment": treatment.tolist(),
        },
        "qte_25": qte_25,
        "qte_50": qte_50,
        "qte_75": qte_75,
    }

    return results


def capture_control_function_golden() -> dict:
    """
    Capture Control Function golden reference results.

    Test cases:
    - Two-step control function
    """
    results = {}
    np.random.seed(1016)

    n = 300
    X = np.random.randn(n, 2)
    Z = np.random.randn(n).reshape(-1, 1)  # Excluded instrument (2D)

    # First stage
    u = np.random.randn(n)
    D = 0.5 * Z.ravel() + 0.3 * X[:, 0] + u + np.random.randn(n) * 0.3

    # Outcome: true effect = 2.0, with endogeneity
    outcome = 2.0 * D + 0.5 * X[:, 1] + 0.4 * u + np.random.randn(n) * 0.5

    cf_result = control_function_ate(outcome, D, Z, X)

    results["control_function_2step"] = {
        "description": "Control function with endogeneity, true effect=2.0, n=300",
        "data": {
            "outcome": outcome.tolist(),
            "treatment": D.tolist(),
            "covariates": X.tolist(),
            "instrument": Z.tolist(),
        },
        "control_function": cf_result,
    }

    return results


def capture_dynamic_dml_golden() -> dict:
    """
    Capture Dynamic DML golden reference results.

    Test cases:
    - Dynamic treatment effects
    """
    results = {}
    np.random.seed(1017)

    # Time series: 200 observations
    n = 200

    # States (covariates for dynamic model)
    states = np.random.randn(n, 3)

    # Treatment with dynamic effects
    treatment = np.random.binomial(1, 0.5, n).astype(float)

    # Dynamic effect: depends on lagged treatment
    outcome = 1.5 * treatment + 0.5 * states[:, 0] + np.random.randn(n) * 0.5

    dml_result = dynamic_dml(
        outcomes=outcome,
        treatments=treatment,
        states=states,
        max_lag=3,
    )

    results["dynamic_dml_panel"] = {
        "description": "Dynamic DML, n=200, max_lag=3",
        "data": {
            "outcome": outcome.tolist(),
            "treatment": treatment.tolist(),
            "states": states.tolist(),
        },
        "dynamic_dml": dml_result,
    }

    return results


def capture_bayesian_golden() -> dict:
    """
    Capture Bayesian ATE golden reference results.

    Test cases:
    - Bayesian ATE with posterior sampling
    """
    results = {}
    np.random.seed(1018)

    n = 200
    treatment = np.random.binomial(1, 0.5, n).astype(float)
    outcome = 2.0 * treatment + np.random.randn(n)

    # Bayesian ATE with fixed posterior samples for reproducibility
    bayes_result = bayesian_ate(
        outcomes=outcome,
        treatment=treatment,
        n_posterior_samples=1000,
    )

    results["bayesian_ate"] = {
        "description": "Bayesian ATE, true effect=2.0, n=200",
        "data": {
            "outcome": outcome.tolist(),
            "treatment": treatment.tolist(),
        },
        "bayesian_ate": bayes_result,
    }

    return results


def capture_discovery_golden() -> dict:
    """
    Capture Discovery (causal structure learning) golden reference results.

    Test cases:
    - PC algorithm skeleton
    """
    results = {}
    np.random.seed(1019)

    # Generate data from known DAG: X -> Y -> Z
    n = 300
    X = np.random.randn(n)
    Y = 0.8 * X + np.random.randn(n) * 0.5
    Z = 0.6 * Y + np.random.randn(n) * 0.5

    data = np.column_stack([X, Y, Z])

    pc_result = pc_algorithm(data, alpha=0.05)

    results["discovery_pc"] = {
        "description": "PC algorithm on X->Y->Z, n=300",
        "data": {
            "X": X.tolist(),
            "Y": Y.tolist(),
            "Z": Z.tolist(),
        },
        "pc_algorithm": pc_result,
    }

    return results


def convert_numpy_to_python(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Parameters
    ----------
    obj : any
        Object to convert (dict, list, numpy array, numpy scalar, dataclass, etc.)

    Returns
    -------
    any
        Converted object with native Python types
    """
    if obj is None:
        return None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        # Convert tuple keys to strings for JSON compatibility
        return {
            str(key) if isinstance(key, tuple) else key: convert_numpy_to_python(value)
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, (set, frozenset)):
        return [convert_numpy_to_python(item) for item in sorted(obj, key=str)]
    elif hasattr(obj, '__dataclass_fields__'):
        # Handle dataclasses
        from dataclasses import asdict
        return convert_numpy_to_python(asdict(obj))
    elif hasattr(obj, '_asdict'):
        # Handle named tuples
        return convert_numpy_to_python(obj._asdict())
    elif hasattr(obj, '__dict__'):
        # Handle other objects with __dict__
        return convert_numpy_to_python(obj.__dict__)
    else:
        return obj


def save_golden_results(results: dict, output_path: Path):
    """
    Save golden results to JSON file.

    Parameters
    ----------
    results : dict
        Golden results from capture_golden_results()
    output_path : Path
        Path to save JSON file

    Raises
    ------
    IOError
        If file cannot be written (following NEVER FAIL SILENTLY principle)
    """
    try:
        # Convert numpy types to Python types for JSON serialization
        results_serializable = convert_numpy_to_python(results)

        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"✅ Golden results saved to: {output_path}")
        print(f"   File size: {output_path.stat().st_size:,} bytes")
    except Exception as e:
        raise IOError(
            f"CRITICAL ERROR: Failed to save golden results.\n"
            f"Function: save_golden_results\n"
            f"Path: {output_path}\n"
            f"Error: {str(e)}"
        )


def main():
    """Main entry point."""
    print("=" * 80)
    print("CAPTURING GOLDEN RESULTS FROM PYTHON LIBRARY IMPLEMENTATIONS")
    print("=" * 80)

    # Capture results
    print("\n📊 Running all estimators on reference datasets...")
    golden_results = capture_golden_results()

    # Print summary
    print(f"\n✅ Captured {len(golden_results)} test cases:")
    for test_name, test_data in golden_results.items():
        print(f"   - {test_name}: {test_data['description']}")

    # Save to file
    output_dir = Path("tests/golden_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "python_golden_results.json"

    save_golden_results(golden_results, output_path)

    print("\n" + "=" * 80)
    print("GOLDEN RESULTS CAPTURE COMPLETE")
    print("=" * 80)
    print(f"\nNext step: Implement Julia estimators and validate against these results.")
    print(f"Validation tolerance: rtol < 1e-10 (near machine precision)")


if __name__ == "__main__":
    main()
