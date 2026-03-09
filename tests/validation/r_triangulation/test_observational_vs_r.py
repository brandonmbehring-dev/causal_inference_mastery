"""Triangulation tests: Python Observational methods vs R reference implementations.

This module provides Layer 5 validation by comparing our Python observational
causal inference implementations against R implementations using WeightIt,
drtmle, and base R packages.

Tests skip gracefully when R/rpy2 is unavailable.

Tolerance levels (established based on implementation differences):
- Propensity scores: rtol=0.01 (1% relative, same logistic formula)
- IPW estimate: rtol=0.05 (5% relative, minor variance differences)
- DR estimate: rtol=0.10 (10% relative, influence function vs sandwich)
- Standard errors: rtol=0.15 (15% relative, robust SE implementations vary)
- AUC/pseudo-R²: rtol=0.02 (2% relative, diagnostic agreement)

Run with: pytest tests/validation/r_triangulation/test_observational_vs_r.py -v

References:
- Lunceford & Davidian (2004). Stratification and Weighting via PS. Statistics in Medicine.
- Bang & Robins (2005). Doubly Robust Estimation in Missing Data and Causal Inference Models.
- Rosenbaum & Rubin (1983). The Central Role of the Propensity Score.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.validation.r_triangulation.r_interface import (
    check_r_available,
    check_weightit_installed,
    check_drtmle_installed,
    r_propensity_glm,
    r_ipw_observational,
    r_dr_ate,
)

# Lazy import Python implementations
try:
    from src.causal_inference.observational.propensity import estimate_propensity
    from src.causal_inference.observational.ipw import ipw_ate_observational
    from src.causal_inference.observational.doubly_robust import dr_ate

    OBSERVATIONAL_AVAILABLE = True
except ImportError:
    OBSERVATIONAL_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_observational_python = pytest.mark.skipif(
    not OBSERVATIONAL_AVAILABLE,
    reason="Python observational module not available",
)

requires_weightit_r = pytest.mark.skipif(
    not check_weightit_installed(),
    reason="R WeightIt package not installed. Install with: install.packages('WeightIt')",
)

requires_drtmle_r = pytest.mark.skipif(
    not check_drtmle_installed(),
    reason="R drtmle package not installed. Install with: install.packages('drtmle')",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_observational_data(
    n: int = 500,
    confounding_strength: float = 0.5,
    true_ate: float = 2.0,
    seed: int = 42,
) -> dict:
    """Generate confounded observational data with known ATE.

    DGP:
    - X ~ N(0, I_3) (3 covariates)
    - logit(P(T=1|X)) = -0.5 + confounding_strength * X₁
    - Y = true_ate * T + X₁ + 0.5*X₂ + noise

    Parameters
    ----------
    n : int
        Sample size.
    confounding_strength : float
        Strength of confounding (coefficient on X₁ in propensity model).
    true_ate : float
        True average treatment effect.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcome, treatment, covariates, true_ate, propensity.
    """
    np.random.seed(seed)

    # Covariates: 3 standard normal variables
    covariates = np.random.randn(n, 3)

    # Propensity model: logit(P(T=1|X)) = -0.5 + confounding_strength * X₁
    logit_p = -0.5 + confounding_strength * covariates[:, 0]
    true_propensity = 1 / (1 + np.exp(-logit_p))

    # Treatment assignment
    treatment = (np.random.rand(n) < true_propensity).astype(int)

    # Outcome: Y = true_ate * T + X₁ + 0.5*X₂ + noise
    outcome = true_ate * treatment + covariates[:, 0] + 0.5 * covariates[:, 1] + np.random.randn(n)

    return {
        "outcome": outcome,
        "treatment": treatment,
        "covariates": covariates,
        "true_ate": true_ate,
        "true_propensity": true_propensity,
    }


def generate_strong_confounding_data(
    n: int = 500,
    seed: int = 42,
) -> dict:
    """Generate data with strong confounding (harder estimation).

    Parameters
    ----------
    n : int
        Sample size.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcome, treatment, covariates, true_ate.
    """
    return generate_observational_data(
        n=n,
        confounding_strength=1.5,  # Strong confounding
        true_ate=3.0,
        seed=seed,
    )


def generate_limited_overlap_data(
    n: int = 500,
    seed: int = 42,
) -> dict:
    """Generate data with limited propensity score overlap.

    Creates challenging estimation scenario with propensities near 0 and 1.

    Parameters
    ----------
    n : int
        Sample size.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcome, treatment, covariates, true_ate.
    """
    np.random.seed(seed)

    # Covariates
    covariates = np.random.randn(n, 3)

    # Very strong selection on X₁ → limited overlap
    logit_p = -1.0 + 2.0 * covariates[:, 0]
    true_propensity = 1 / (1 + np.exp(-logit_p))

    # Treatment assignment
    treatment = (np.random.rand(n) < true_propensity).astype(int)

    # Outcome
    true_ate = 2.0
    outcome = (
        true_ate * treatment + 1.5 * covariates[:, 0] + 0.5 * covariates[:, 1] + np.random.randn(n)
    )

    return {
        "outcome": outcome,
        "treatment": treatment,
        "covariates": covariates,
        "true_ate": true_ate,
        "true_propensity": true_propensity,
    }


def generate_high_dimensional_data(
    n: int = 500,
    p: int = 10,
    seed: int = 42,
) -> dict:
    """Generate data with many covariates.

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of covariates.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with outcome, treatment, covariates, true_ate.
    """
    np.random.seed(seed)

    # Covariates
    covariates = np.random.randn(n, p)

    # Propensity depends on first 3 covariates
    logit_p = -0.5 + 0.3 * covariates[:, 0] + 0.3 * covariates[:, 1] + 0.3 * covariates[:, 2]
    true_propensity = 1 / (1 + np.exp(-logit_p))

    # Treatment assignment
    treatment = (np.random.rand(n) < true_propensity).astype(int)

    # Outcome depends on first 5 covariates
    true_ate = 2.0
    outcome = (
        true_ate * treatment
        + 0.5 * covariates[:, 0]
        + 0.5 * covariates[:, 1]
        + 0.3 * covariates[:, 2]
        + 0.2 * covariates[:, 3]
        + 0.1 * covariates[:, 4]
        + np.random.randn(n)
    )

    return {
        "outcome": outcome,
        "treatment": treatment,
        "covariates": covariates,
        "true_ate": true_ate,
        "true_propensity": true_propensity,
    }


# =============================================================================
# Test Class: Propensity Score Estimation
# =============================================================================


class TestPropensityVsR:
    """Compare Python estimate_propensity() to R glm(family=binomial)."""

    @requires_observational_python
    def test_propensity_basic(self):
        """Basic propensity estimation comparison."""
        data = generate_observational_data(n=500, seed=42)

        # Python propensity estimation
        py_result = estimate_propensity(
            treatment=data["treatment"],
            covariates=data["covariates"],
            method="logistic",
        )

        # R propensity estimation
        r_result = r_propensity_glm(
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        if r_result is None:
            pytest.skip("R glm() call failed")

        # Compare propensity scores
        np.testing.assert_allclose(
            py_result.propensity,
            r_result["propensity"],
            rtol=0.01,
            err_msg="Propensity scores differ between Python and R",
        )

    @requires_observational_python
    def test_propensity_strong_confounding(self):
        """Propensity estimation with strong confounding."""
        data = generate_strong_confounding_data(n=500, seed=123)

        py_result = estimate_propensity(
            treatment=data["treatment"],
            covariates=data["covariates"],
            method="logistic",
        )

        r_result = r_propensity_glm(
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        if r_result is None:
            pytest.skip("R glm() call failed")

        # Propensity scores should still match
        np.testing.assert_allclose(
            py_result.propensity,
            r_result["propensity"],
            rtol=0.02,  # Slightly looser for strong confounding
            err_msg="Propensity scores differ under strong confounding",
        )

    @requires_observational_python
    def test_propensity_high_dimensional(self):
        """Propensity estimation with many covariates."""
        data = generate_high_dimensional_data(n=500, p=10, seed=456)

        py_result = estimate_propensity(
            treatment=data["treatment"],
            covariates=data["covariates"],
            method="logistic",
        )

        r_result = r_propensity_glm(
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        if r_result is None:
            pytest.skip("R glm() call failed")

        # Compare propensity scores
        np.testing.assert_allclose(
            py_result.propensity,
            r_result["propensity"],
            rtol=0.02,
            err_msg="Propensity scores differ with many covariates",
        )


# =============================================================================
# Test Class: IPW Estimation
# =============================================================================


class TestIPWVsWeightIt:
    """Compare Python ipw_ate_observational() to R WeightIt package."""

    @requires_observational_python
    @requires_weightit_r
    def test_ipw_basic(self):
        """Basic IPW estimation comparison."""
        data = generate_observational_data(n=500, seed=42)

        # Python IPW
        py_result = ipw_ate_observational(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            stabilize=False,
        )

        # R IPW via WeightIt
        r_result = r_ipw_observational(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            stabilize=False,
        )

        if r_result is None:
            pytest.skip("R WeightIt call failed")

        # Compare ATE estimates
        np.testing.assert_allclose(
            py_result.estimate,
            r_result["estimate"],
            rtol=0.05,
            err_msg=f"IPW estimates differ: Python={py_result.estimate:.4f}, R={r_result['estimate']:.4f}",
        )

    @requires_observational_python
    @requires_weightit_r
    def test_ipw_stabilized(self):
        """IPW with stabilized weights."""
        data = generate_observational_data(n=500, seed=789)

        py_result = ipw_ate_observational(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            stabilize=True,
        )

        r_result = r_ipw_observational(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            stabilize=True,
        )

        if r_result is None:
            pytest.skip("R WeightIt call failed")

        np.testing.assert_allclose(
            py_result.estimate,
            r_result["estimate"],
            rtol=0.05,
            err_msg="Stabilized IPW estimates differ",
        )

    @requires_observational_python
    @requires_weightit_r
    def test_ipw_trimmed(self):
        """IPW with propensity trimming."""
        data = generate_limited_overlap_data(n=500, seed=101)

        # Trim at 5th and 95th percentiles
        py_result = ipw_ate_observational(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            trim_at=(0.05, 0.95),
        )

        r_result = r_ipw_observational(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
            trim_percentile=(0.05, 0.95),
        )

        if r_result is None:
            pytest.skip("R WeightIt call failed")

        # Trimmed estimates may differ more due to implementation details
        np.testing.assert_allclose(
            py_result.estimate,
            r_result["estimate"],
            rtol=0.10,
            err_msg="Trimmed IPW estimates differ",
        )

    @requires_observational_python
    @requires_weightit_r
    def test_ipw_strong_confounding(self):
        """IPW under strong confounding."""
        data = generate_strong_confounding_data(n=500, seed=202)

        py_result = ipw_ate_observational(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        r_result = r_ipw_observational(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        if r_result is None:
            pytest.skip("R WeightIt call failed")

        np.testing.assert_allclose(
            py_result.estimate,
            r_result["estimate"],
            rtol=0.10,  # Looser tolerance for challenging data
            err_msg="IPW estimates differ under strong confounding",
        )


# =============================================================================
# Test Class: Doubly Robust Estimation
# =============================================================================


class TestDRVsDRTMLE:
    """Compare Python dr_ate() to R drtmle package (or manual AIPW)."""

    @requires_observational_python
    @requires_drtmle_r
    def test_dr_basic(self):
        """Basic DR estimation comparison."""
        data = generate_observational_data(n=500, seed=42)

        # Python DR
        py_result = dr_ate(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        # R DR
        r_result = r_dr_ate(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        if r_result is None:
            pytest.skip("R DR estimation failed")

        np.testing.assert_allclose(
            py_result.estimate,
            r_result["estimate"],
            rtol=0.10,
            err_msg=f"DR estimates differ: Python={py_result.estimate:.4f}, R={r_result['estimate']:.4f}",
        )

    @requires_observational_python
    @requires_drtmle_r
    def test_dr_both_models_correct(self):
        """DR when both propensity and outcome models are correct."""
        # Use basic DGP where linear models are approximately correct
        data = generate_observational_data(n=1000, confounding_strength=0.3, seed=303)

        py_result = dr_ate(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        r_result = r_dr_ate(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        if r_result is None:
            pytest.skip("R DR estimation failed")

        # With correct models, estimates should be close to true ATE
        np.testing.assert_allclose(
            py_result.estimate,
            data["true_ate"],
            rtol=0.15,
            err_msg="DR estimate far from true ATE with correct models",
        )

        np.testing.assert_allclose(
            py_result.estimate,
            r_result["estimate"],
            rtol=0.10,
            err_msg="DR estimates differ when both models correct",
        )

    @requires_observational_python
    @requires_drtmle_r
    def test_dr_strong_confounding(self):
        """DR under strong confounding."""
        data = generate_strong_confounding_data(n=500, seed=404)

        py_result = dr_ate(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        r_result = r_dr_ate(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        if r_result is None:
            pytest.skip("R DR estimation failed")

        np.testing.assert_allclose(
            py_result.estimate,
            r_result["estimate"],
            rtol=0.15,  # Looser for challenging data
            err_msg="DR estimates differ under strong confounding",
        )

    @requires_observational_python
    @requires_drtmle_r
    def test_dr_high_dimensional(self):
        """DR with many covariates."""
        data = generate_high_dimensional_data(n=500, p=10, seed=505)

        py_result = dr_ate(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        r_result = r_dr_ate(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        if r_result is None:
            pytest.skip("R DR estimation failed")

        np.testing.assert_allclose(
            py_result.estimate,
            r_result["estimate"],
            rtol=0.15,
            err_msg="DR estimates differ with many covariates",
        )


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestObservationalEdgeCases:
    """Edge cases for observational methods."""

    @requires_observational_python
    def test_small_sample(self):
        """Test with small sample size."""
        data = generate_observational_data(n=100, seed=606)

        # Both should run without error
        py_result = ipw_ate_observational(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        r_result = r_ipw_observational(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        # Results should be finite
        assert np.isfinite(py_result.estimate), "Python IPW returned non-finite with small n"

        if r_result is not None:
            assert np.isfinite(r_result["estimate"]), "R IPW returned non-finite with small n"

    @requires_observational_python
    def test_extreme_propensity_values(self):
        """Test handling of extreme propensity values."""
        data = generate_limited_overlap_data(n=500, seed=707)

        # Should handle extreme propensities without failure
        py_result = ipw_ate_observational(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        assert np.isfinite(py_result.estimate), "IPW failed with extreme propensities"

    @requires_observational_python
    @requires_weightit_r
    def test_balanced_treatment(self):
        """Test with nearly balanced treatment groups."""
        np.random.seed(808)
        n = 500
        covariates = np.random.randn(n, 3)

        # Balanced assignment (propensity ~ 0.5 for all)
        logit_p = 0.1 * covariates[:, 0]  # Very weak selection
        propensity = 1 / (1 + np.exp(-logit_p))
        treatment = (np.random.rand(n) < propensity).astype(int)

        true_ate = 2.0
        outcome = true_ate * treatment + covariates[:, 0] + np.random.randn(n)

        py_result = ipw_ate_observational(
            outcomes=outcome,
            treatment=treatment,
            covariates=covariates,
        )

        r_result = r_ipw_observational(
            outcomes=outcome,
            treatment=treatment,
            covariates=covariates,
        )

        if r_result is None:
            pytest.skip("R WeightIt call failed")

        # With balanced treatment, IPW should be close to simple difference in means
        np.testing.assert_allclose(
            py_result.estimate,
            r_result["estimate"],
            rtol=0.05,
            err_msg="IPW estimates differ with balanced treatment",
        )

    @requires_observational_python
    @requires_drtmle_r
    def test_single_covariate(self):
        """Test with single covariate."""
        np.random.seed(909)
        n = 500
        covariates = np.random.randn(n, 1)

        logit_p = -0.5 + 0.5 * covariates[:, 0]
        propensity = 1 / (1 + np.exp(-logit_p))
        treatment = (np.random.rand(n) < propensity).astype(int)

        true_ate = 2.0
        outcome = true_ate * treatment + covariates[:, 0] + np.random.randn(n)

        py_result = dr_ate(
            outcomes=outcome,
            treatment=treatment,
            covariates=covariates,
        )

        r_result = r_dr_ate(
            outcomes=outcome,
            treatment=treatment,
            covariates=covariates,
        )

        if r_result is None:
            pytest.skip("R DR estimation failed")

        np.testing.assert_allclose(
            py_result.estimate,
            r_result["estimate"],
            rtol=0.10,
            err_msg="DR estimates differ with single covariate",
        )


# =============================================================================
# Test Class: Cross-Method Consistency
# =============================================================================


class TestObservationalConsistency:
    """Cross-method consistency checks."""

    @requires_observational_python
    @requires_weightit_r
    @requires_drtmle_r
    def test_ipw_dr_agreement_correct_models(self):
        """IPW and DR should agree when both models are correct."""
        data = generate_observational_data(n=1000, confounding_strength=0.3, seed=1010)

        py_ipw = ipw_ate_observational(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        py_dr = dr_ate(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        # With correct models, IPW and DR should give similar answers
        np.testing.assert_allclose(
            py_ipw.estimate,
            py_dr.estimate,
            rtol=0.15,
            err_msg="Python IPW and DR estimates differ substantially",
        )

        r_ipw = r_ipw_observational(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        r_dr = r_dr_ate(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        if r_ipw is not None and r_dr is not None:
            np.testing.assert_allclose(
                r_ipw["estimate"],
                r_dr["estimate"],
                rtol=0.15,
                err_msg="R IPW and DR estimates differ substantially",
            )

    @requires_observational_python
    @requires_weightit_r
    @requires_drtmle_r
    def test_all_estimates_near_true_ate(self):
        """All methods should estimate near true ATE with correct specification."""
        # Large sample with moderate confounding
        data = generate_observational_data(
            n=2000,
            confounding_strength=0.5,
            true_ate=2.0,
            seed=1111,
        )

        py_ipw = ipw_ate_observational(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        py_dr = dr_ate(
            outcomes=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        # Both should be reasonably close to true ATE
        for name, est in [("IPW", py_ipw.estimate), ("DR", py_dr.estimate)]:
            np.testing.assert_allclose(
                est,
                data["true_ate"],
                rtol=0.20,  # 20% tolerance given finite sample
                err_msg=f"Python {name} estimate {est:.4f} far from true ATE {data['true_ate']}",
            )
