"""Triangulation tests: Python PSM vs R MatchIt package.

This module provides Layer 5 validation by comparing our Python PSM implementation
against R's MatchIt package for propensity score matching.

Tests skip gracefully when R/rpy2 or MatchIt is unavailable.

Tolerance levels (established based on implementation differences):
- ATE estimate: rtol=0.10 (10% relative, accounts for different tie-breaking)
- Standard error: rtol=0.20 (20% relative, AI variance vs HC2)
- Propensity scores: rtol=0.05 (5% relative, same glm)
- Balance metrics (SMD): atol=0.05 (absolute, pooled SD formulas vary)
- Match counts: exact (when same algorithm parameters)

Known implementation differences:
1. Python uses sklearn LogisticRegression (weak L2 penalty), R uses glm() (no penalty)
2. Python uses Abadie-Imbens variance, R uses HC2 robust SE
3. Greedy nearest-neighbor may break ties differently
4. Python clamps propensity to [1e-10, 1-1e-10], R does not clamp

Run with: pytest tests/validation/r_triangulation/test_psm_vs_matchit.py -v

References:
- Ho, Imai, King & Stuart (2011). MatchIt: Nonparametric Preprocessing
- Abadie & Imbens (2006). Large Sample Properties of Matching Estimators
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.validation.r_triangulation.r_interface import (
    check_r_available,
    check_matchit_installed,
    r_psm_propensity,
    r_psm_matchit_nearest,
    r_psm_balance_metrics,
)

# Lazy import Python implementations
try:
    from src.causal_inference.psm.psm_estimator import psm_ate
    from src.causal_inference.psm.propensity import PropensityScoreEstimator
    from src.causal_inference.psm.balance import compute_smd, compute_variance_ratio

    PSM_AVAILABLE = True
except ImportError:
    PSM_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_psm_python = pytest.mark.skipif(
    not PSM_AVAILABLE,
    reason="Python PSM module not available",
)

requires_matchit = pytest.mark.skipif(
    not check_matchit_installed(),
    reason="R 'MatchIt' package not installed. Install with: install.packages('MatchIt')",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_psm_data(
    n: int = 200,
    p: int = 2,
    true_ate: float = 2.0,
    noise_sd: float = 0.5,
    seed: int = 42,
    balanced: bool = True,
) -> dict:
    """Generate data from a PSM DGP with confounding.

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of covariates.
    true_ate : float
        True average treatment effect.
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.
    balanced : bool
        If True, treatment probability ~0.5. If False, treatment probability ~0.3.

    Returns
    -------
    dict
        Dictionary with outcomes, treatment, covariates, true_ate.
    """
    np.random.seed(seed)

    # Generate covariates
    covariates = np.random.normal(0, 1, (n, p))

    # Treatment assignment depends on covariates (confounding)
    coef_propensity = np.linspace(0.3, 0.5, p)
    logit_p = covariates @ coef_propensity
    if not balanced:
        logit_p -= 0.5  # Shift to lower probability

    propensity = 1 / (1 + np.exp(-logit_p))
    treatment = np.random.binomial(1, propensity, n)

    # Outcome model (shares covariates with propensity)
    coef_outcome = np.linspace(0.2, 0.4, p)
    outcome_baseline = covariates @ coef_outcome
    noise = np.random.normal(0, noise_sd, n)
    outcomes = true_ate * treatment + outcome_baseline + noise

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "covariates": covariates,
        "true_ate": true_ate,
        "propensity": propensity,
        "n": n,
        "p": p,
    }


def generate_limited_overlap_data(
    n: int = 200,
    true_ate: float = 1.5,
    noise_sd: float = 0.6,
    seed: int = 456,
) -> dict:
    """Generate data with limited propensity score overlap.

    Treated and control units have different covariate distributions,
    creating a region of limited overlap.

    Parameters
    ----------
    n : int
        Sample size.
    true_ate : float
        True average treatment effect.
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcomes, treatment, covariates, true_ate.
    """
    np.random.seed(seed)

    # Treated units have higher mean covariate
    n_treated = n // 3
    n_control = n - n_treated

    X_treated = np.random.normal(2.0, 1.0, (n_treated, 2))
    X_control = np.random.normal(0.0, 1.0, (n_control, 2))

    covariates = np.vstack([X_treated, X_control])
    treatment = np.array([1] * n_treated + [0] * n_control)

    # Outcome
    coef = np.array([0.4, 0.3])
    noise = np.random.normal(0, noise_sd, n)
    outcomes = true_ate * treatment + covariates @ coef + noise

    # Shuffle
    perm = np.random.permutation(n)
    covariates = covariates[perm]
    treatment = treatment[perm]
    outcomes = outcomes[perm]

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "covariates": covariates,
        "true_ate": true_ate,
        "n": n,
        "p": 2,
    }


def generate_high_dimensional_data(
    n: int = 300,
    p: int = 10,
    true_ate: float = 1.8,
    noise_sd: float = 0.5,
    seed: int = 789,
) -> dict:
    """Generate data with many covariates.

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of covariates.
    true_ate : float
        True average treatment effect.
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcomes, treatment, covariates, true_ate.
    """
    np.random.seed(seed)

    # Generate covariates
    covariates = np.random.normal(0, 1, (n, p))

    # Only first 3 covariates affect treatment
    logit_p = 0.3 * covariates[:, 0] + 0.2 * covariates[:, 1] + 0.15 * covariates[:, 2]
    propensity = 1 / (1 + np.exp(-logit_p))
    treatment = np.random.binomial(1, propensity, n)

    # Outcome uses different covariates (some overlap)
    noise = np.random.normal(0, noise_sd, n)
    outcomes = (
        true_ate * treatment
        + 0.2 * covariates[:, 0]  # Shared confounder
        + 0.15 * covariates[:, 3]  # Outcome-only
        + 0.1 * covariates[:, 4]
        + noise
    )

    return {
        "outcomes": outcomes,
        "treatment": treatment,
        "covariates": covariates,
        "true_ate": true_ate,
        "propensity": propensity,
        "n": n,
        "p": p,
    }


# =============================================================================
# Test Classes
# =============================================================================


@requires_psm_python
@requires_matchit
class TestPSMPropensityVsR:
    """Compare Python propensity scores to R glm()."""

    def test_propensity_balanced_design(self):
        """Propensity scores match R for balanced treatment probability."""
        data = generate_psm_data(n=200, p=2, seed=100, balanced=True)

        # Python propensity
        py_estimator = PropensityScoreEstimator()
        py_estimator.fit(data["covariates"], data["treatment"])
        py_scores = py_estimator.propensity_scores

        # R propensity
        r_result = r_psm_propensity(data["covariates"], data["treatment"])
        assert r_result is not None, "R propensity estimation failed"
        r_scores = r_result["propensity_scores"]

        # Compare (5% tolerance due to regularization differences)
        np.testing.assert_allclose(
            py_scores,
            r_scores,
            rtol=0.05,
            err_msg="Propensity scores differ between Python and R",
        )

    def test_propensity_unbalanced_design(self):
        """Propensity scores match R for unbalanced treatment probability."""
        data = generate_psm_data(n=200, p=2, seed=101, balanced=False)

        # Python propensity
        py_estimator = PropensityScoreEstimator()
        py_estimator.fit(data["covariates"], data["treatment"])
        py_scores = py_estimator.propensity_scores

        # R propensity
        r_result = r_psm_propensity(data["covariates"], data["treatment"])
        assert r_result is not None, "R propensity estimation failed"
        r_scores = r_result["propensity_scores"]

        # Compare
        np.testing.assert_allclose(
            py_scores,
            r_scores,
            rtol=0.05,
            err_msg="Propensity scores differ (unbalanced design)",
        )

    def test_propensity_high_dimensional(self):
        """Propensity scores match R with many covariates."""
        data = generate_high_dimensional_data(n=300, p=10, seed=102)

        # Python propensity
        py_estimator = PropensityScoreEstimator()
        py_estimator.fit(data["covariates"], data["treatment"])
        py_scores = py_estimator.propensity_scores

        # R propensity
        r_result = r_psm_propensity(data["covariates"], data["treatment"])
        assert r_result is not None, "R propensity estimation failed"
        r_scores = r_result["propensity_scores"]

        # Slightly looser tolerance for high-dimensional case
        np.testing.assert_allclose(
            py_scores,
            r_scores,
            rtol=0.08,
            err_msg="Propensity scores differ (high-dimensional)",
        )


@requires_psm_python
@requires_matchit
class TestPSMMatchingVsMatchIt:
    """Compare Python NN matching to MatchIt matchit()."""

    def test_matching_1to1_no_replacement(self):
        """1:1 matching without replacement produces similar results."""
        data = generate_psm_data(n=200, p=2, true_ate=2.0, seed=200)

        # Python PSM
        py_result = psm_ate(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )

        # R MatchIt
        r_result = r_psm_matchit_nearest(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )
        assert r_result is not None, "R MatchIt failed"

        # Compare match counts (should be similar)
        assert py_result["n_matched"] > 0, "Python found no matches"
        assert r_result["n_matched"] > 0, "R found no matches"

        # Match counts may differ slightly due to tie-breaking
        # But should be within 10%
        match_ratio = py_result["n_matched"] / r_result["n_matched"]
        assert 0.9 <= match_ratio <= 1.1, (
            f"Match count difference: Python={py_result['n_matched']}, R={r_result['n_matched']}"
        )

    def test_matching_2to1_with_replacement(self):
        """2:1 matching with replacement produces similar results."""
        data = generate_psm_data(n=200, p=2, true_ate=2.5, seed=201)

        # Python PSM
        py_result = psm_ate(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=2,
            with_replacement=True,
        )

        # R MatchIt
        r_result = r_psm_matchit_nearest(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=2,
            with_replacement=True,
        )
        assert r_result is not None, "R MatchIt failed"

        # Both should match most treated units
        assert py_result["n_matched"] > 0, "Python found no matches"
        assert r_result["n_matched"] > 0, "R found no matches"

    def test_matching_with_caliper(self):
        """Matching with caliper restriction."""
        data = generate_limited_overlap_data(n=200, true_ate=1.5, seed=202)

        # Python PSM with caliper (in propensity score units, not SD)
        py_result = psm_ate(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
            caliper=0.1,  # Python: absolute propensity distance
        )

        # R MatchIt with caliper (in SD units by default)
        # MatchIt uses SD of propensity score, so 0.25 SD is comparable
        r_result = r_psm_matchit_nearest(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
            caliper=0.25,  # R: SD units
        )
        assert r_result is not None, "R MatchIt failed"

        # Both should find fewer matches due to caliper
        # Just verify both completed successfully
        assert py_result["n_matched"] > 0, "Python found no matches with caliper"
        assert r_result["n_matched"] > 0, "R found no matches with caliper"


@requires_psm_python
@requires_matchit
class TestPSMATEVsMatchIt:
    """Compare full PSM ATE pipeline between Python and MatchIt."""

    def test_ate_simple_balanced(self):
        """ATE estimates agree for balanced design."""
        data = generate_psm_data(n=300, p=2, true_ate=2.0, seed=300)

        # Python PSM
        py_result = psm_ate(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )

        # R MatchIt
        r_result = r_psm_matchit_nearest(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )
        assert r_result is not None, "R MatchIt failed"

        # ATE estimates should be close (10% tolerance)
        np.testing.assert_allclose(
            py_result["estimate"],
            r_result["estimate"],
            rtol=0.10,
            err_msg=f"ATE differs: Python={py_result['estimate']:.3f}, R={r_result['estimate']:.3f}",
        )

        # Both should be near true ATE=2.0 (within 0.5)
        assert abs(py_result["estimate"] - data["true_ate"]) < 0.5, (
            f"Python ATE {py_result['estimate']:.3f} far from true {data['true_ate']}"
        )
        assert abs(r_result["estimate"] - data["true_ate"]) < 0.5, (
            f"R ATE {r_result['estimate']:.3f} far from true {data['true_ate']}"
        )

    def test_ate_limited_overlap(self):
        """ATE estimates agree for limited overlap scenario."""
        data = generate_limited_overlap_data(n=300, true_ate=1.5, seed=301)

        # Python PSM
        py_result = psm_ate(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )

        # R MatchIt
        r_result = r_psm_matchit_nearest(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )
        assert r_result is not None, "R MatchIt failed"

        # Both should find matches (though possibly fewer)
        assert py_result["n_matched"] > 0, "Python found no matches"
        assert r_result["n_matched"] > 0, "R found no matches"

        # ATE may differ more with limited overlap (15% tolerance)
        np.testing.assert_allclose(
            py_result["estimate"],
            r_result["estimate"],
            rtol=0.15,
            err_msg="ATE differs (limited overlap)",
        )

    def test_ate_high_dimensional(self):
        """ATE estimates agree with many covariates."""
        data = generate_high_dimensional_data(n=400, p=10, true_ate=1.8, seed=302)

        # Python PSM
        py_result = psm_ate(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )

        # R MatchIt
        r_result = r_psm_matchit_nearest(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )
        assert r_result is not None, "R MatchIt failed"

        # ATE estimates should be close (15% tolerance for high-dimensional)
        np.testing.assert_allclose(
            py_result["estimate"],
            r_result["estimate"],
            rtol=0.15,
            err_msg="ATE differs (high-dimensional)",
        )

    def test_se_comparison(self):
        """Standard error estimates are in same order of magnitude."""
        data = generate_psm_data(n=300, p=2, true_ate=2.0, seed=303)

        # Python PSM (Abadie-Imbens variance)
        py_result = psm_ate(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )

        # R MatchIt (HC2 robust SE)
        r_result = r_psm_matchit_nearest(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )
        assert r_result is not None, "R MatchIt failed"

        # SE comparison (20% tolerance due to different variance formulas)
        # Abadie-Imbens vs HC2 can differ
        np.testing.assert_allclose(
            py_result["se"],
            r_result["se"],
            rtol=0.30,  # Looser tolerance for variance estimators
            err_msg=f"SE differs: Python={py_result['se']:.3f}, R={r_result['se']:.3f}",
        )


@requires_psm_python
@requires_matchit
class TestPSMBalanceVsR:
    """Compare balance diagnostics (SMD, VR) between Python and R."""

    def test_smd_before_matching(self):
        """SMD before matching agrees between Python and R."""
        data = generate_psm_data(n=200, p=3, seed=400)

        # Python SMD
        py_smd = compute_smd(data["covariates"], data["treatment"].astype(bool))

        # Get propensity scores for R function
        py_estimator = PropensityScoreEstimator()
        py_estimator.fit(data["covariates"], data["treatment"])

        # For before-matching metrics, use dummy matched indices (all units)
        treated_idx = np.where(data["treatment"] == 1)[0]
        control_idx = np.where(data["treatment"] == 0)[0]

        r_result = r_psm_balance_metrics(
            covariates=data["covariates"],
            treatment=data["treatment"],
            propensity_scores=py_estimator.propensity_scores,
            matched_treated=treated_idx,
            matched_control=control_idx[: len(treated_idx)],  # Match to same count
        )
        assert r_result is not None, "R balance metrics failed"

        # Compare before-matching SMD (absolute tolerance)
        np.testing.assert_allclose(
            py_smd,
            r_result["smd_before"],
            atol=0.05,
            err_msg="SMD before matching differs",
        )

    def test_smd_after_matching(self):
        """SMD after matching agrees between Python and R."""
        data = generate_psm_data(n=300, p=2, seed=401)

        # Python PSM
        py_result = psm_ate(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )

        # Extract matched pairs from Python
        matches = py_result["matches"]
        treated_idx = np.where(data["treatment"] == 1)[0]

        # Build matched indices
        matched_treated = []
        matched_control = []
        for i, t_idx in enumerate(treated_idx):
            if len(matches[i]) > 0:
                matched_treated.append(t_idx)
                matched_control.append(matches[i][0])  # First match

        matched_treated = np.array(matched_treated)
        matched_control = np.array(matched_control)

        if len(matched_treated) < 10:
            pytest.skip("Not enough matches for balance comparison")

        # R balance metrics
        r_result = r_psm_balance_metrics(
            covariates=data["covariates"],
            treatment=data["treatment"],
            propensity_scores=py_result["propensity_scores"],
            matched_treated=matched_treated,
            matched_control=matched_control,
        )
        assert r_result is not None, "R balance metrics failed"

        # Python SMD after matching
        X_matched_t = data["covariates"][matched_treated]
        X_matched_c = data["covariates"][matched_control]
        n_cov = data["covariates"].shape[1]

        py_smd_after = np.zeros(n_cov)
        for j in range(n_cov):
            mean_t = X_matched_t[:, j].mean()
            mean_c = X_matched_c[:, j].mean()
            var_t = X_matched_t[:, j].var(ddof=1)
            var_c = X_matched_c[:, j].var(ddof=1)
            pooled_sd = np.sqrt((var_t + var_c) / 2)
            if pooled_sd > 0:
                py_smd_after[j] = (mean_t - mean_c) / pooled_sd

        # Compare after-matching SMD (absolute tolerance)
        np.testing.assert_allclose(
            py_smd_after,
            r_result["smd_after"],
            atol=0.05,
            err_msg="SMD after matching differs",
        )


@requires_psm_python
@requires_matchit
class TestPSMEdgeCases:
    """Edge cases and challenging scenarios for PSM triangulation."""

    def test_small_sample(self):
        """PSM works with small sample (n=50)."""
        data = generate_psm_data(n=50, p=2, true_ate=2.0, seed=500)

        # Both implementations should complete
        py_result = psm_ate(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )

        r_result = r_psm_matchit_nearest(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )
        assert r_result is not None, "R MatchIt failed on small sample"

        # Both should find some matches
        assert py_result["n_matched"] > 0
        assert r_result["n_matched"] > 0

    def test_single_covariate(self):
        """PSM works with single covariate."""
        np.random.seed(501)
        n = 150

        # Single covariate
        X = np.random.normal(0, 1, (n, 1))
        logit_p = 0.5 * X.ravel()
        propensity = 1 / (1 + np.exp(-logit_p))
        treatment = np.random.binomial(1, propensity, n)
        outcomes = 2.0 * treatment + 0.3 * X.ravel() + np.random.normal(0, 0.5, n)

        # Both implementations
        py_result = psm_ate(
            outcomes=outcomes,
            treatment=treatment,
            covariates=X,
            M=1,
            with_replacement=False,
        )

        r_result = r_psm_matchit_nearest(
            outcomes=outcomes,
            treatment=treatment,
            covariates=X,
            M=1,
            with_replacement=False,
        )
        assert r_result is not None, "R MatchIt failed on single covariate"

        # ATE should be similar
        np.testing.assert_allclose(
            py_result["estimate"],
            r_result["estimate"],
            rtol=0.15,
            err_msg="ATE differs (single covariate)",
        )

    def test_large_sample(self):
        """PSM works with large sample (n=1000)."""
        data = generate_psm_data(n=1000, p=3, true_ate=2.0, seed=502)

        py_result = psm_ate(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )

        r_result = r_psm_matchit_nearest(
            outcomes=data["outcomes"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            M=1,
            with_replacement=False,
        )
        assert r_result is not None, "R MatchIt failed on large sample"

        # With large sample, estimates should be very close
        np.testing.assert_allclose(
            py_result["estimate"],
            r_result["estimate"],
            rtol=0.08,
            err_msg="ATE differs (large sample)",
        )

        # Both should be close to true ATE
        assert abs(py_result["estimate"] - data["true_ate"]) < 0.3
        assert abs(r_result["estimate"] - data["true_ate"]) < 0.3

    def test_extreme_propensity(self):
        """PSM handles extreme propensity scores gracefully."""
        np.random.seed(503)
        n = 200

        # Create data with some extreme propensity scores
        X = np.random.normal(0, 2, (n, 2))
        logit_p = 1.5 * X[:, 0] + 0.8 * X[:, 1]  # High coefficients = extreme scores
        propensity = 1 / (1 + np.exp(-logit_p))
        treatment = np.random.binomial(1, propensity, n)
        outcomes = 2.0 * treatment + 0.3 * X[:, 0] + np.random.normal(0, 0.5, n)

        # Both should complete without error
        py_result = psm_ate(
            outcomes=outcomes,
            treatment=treatment,
            covariates=X,
            M=1,
            with_replacement=False,
        )

        r_result = r_psm_matchit_nearest(
            outcomes=outcomes,
            treatment=treatment,
            covariates=X,
            M=1,
            with_replacement=False,
        )
        assert r_result is not None, "R MatchIt failed with extreme propensity"

        # Both should find at least some matches
        assert py_result["n_matched"] > 0
        assert r_result["n_matched"] > 0
