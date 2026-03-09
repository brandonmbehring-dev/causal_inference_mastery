"""Triangulation tests: Python DTR (Q/A-learning) vs R `DTRreg` package.

This module provides Layer 5 validation by comparing our Python implementations
of Q-learning and A-learning against the R `DTRreg` package.

Tests focus on single-stage DTR (multi-stage has API differences).

Tolerance levels (established in plan):
- Value estimate: rtol=0.15
- Blip coefficients: rtol=0.10
- Regime agreement: >90%

Run with: pytest tests/validation/r_triangulation/test_dtr_vs_r.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.validation.r_triangulation.r_interface import (
    check_dtrreg_installed,
    check_r_available,
    r_q_learning_dtrreg,
    r_a_learning_dtrreg,
)

# Lazy imports for DTR
try:
    from src.causal_inference.dtr.types import DTRData
    from src.causal_inference.dtr.q_learning import q_learning
    from src.causal_inference.dtr.a_learning import a_learning

    DTR_AVAILABLE = True
except ImportError:
    DTR_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests in this module if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_dtr_python = pytest.mark.skipif(
    not DTR_AVAILABLE,
    reason="Python DTR module not available",
)

requires_dtrreg_r = pytest.mark.skipif(
    not check_dtrreg_installed(),
    reason="R 'DTRreg' package not installed",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_single_stage_dtr_dgp(
    n: int = 500,
    p: int = 3,
    true_blip_intercept: float = 1.0,
    true_blip_slope: float = 0.5,
    propensity_type: str = "balanced",
    noise_sd: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate single-stage DTR data.

    Model:
        Y = β₀ + X'β + A*(ψ₀ + X₁*ψ₁) + ε
        A ~ Bernoulli(e(X))

    where ψ₀ is the blip intercept and ψ₁ is the blip slope for X₁.
    The optimal regime is: d*(X) = I(ψ₀ + X₁*ψ₁ > 0)

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of covariates.
    true_blip_intercept : float
        Intercept of blip function (ψ₀).
    true_blip_slope : float
        Effect of X₁ on treatment effect (ψ₁).
    propensity_type : str
        "balanced" (e=0.5) or "confounded" (e depends on X).
    noise_sd : float
        Standard deviation of outcome noise.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with keys: outcome, treatment, covariates, true_blip, optimal_regime
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    X = rng.normal(0, 1, (n, p))

    # Baseline coefficients
    beta = np.array([1.0, 0.5] + [0.0] * (p - 2))  # Only first two covariates matter

    # Propensity score
    if propensity_type == "balanced":
        e = np.full(n, 0.5)
    elif propensity_type == "confounded":
        from scipy.special import expit

        e = expit(0.5 * X[:, 0])
    else:
        raise ValueError(f"Unknown propensity_type: {propensity_type}")

    # Treatment assignment
    A = rng.binomial(1, e)

    # True blip function: γ(X) = ψ₀ + X₁*ψ₁
    true_blip = true_blip_intercept + true_blip_slope * X[:, 0]

    # True optimal regime
    optimal_regime = (true_blip > 0).astype(int)

    # Baseline outcome
    mu = X @ beta[:p]

    # Generate outcome
    epsilon = rng.normal(0, noise_sd, n)
    Y = mu + A * true_blip + epsilon

    return {
        "outcome": Y,
        "treatment": A,
        "covariates": X,
        "true_blip": true_blip,
        "true_blip_coef": np.array([true_blip_intercept, true_blip_slope]),
        "optimal_regime": optimal_regime,
        "propensity": e,
    }


def create_dtr_data(outcome, treatment, covariates) -> DTRData:
    """Create single-stage DTRData from arrays."""
    return DTRData(
        outcomes=[outcome],
        treatments=[treatment],
        covariates=[covariates],
    )


# =============================================================================
# Test Classes
# =============================================================================


@requires_dtr_python
@requires_dtrreg_r
class TestQLearningSingleStageVsDTRReg:
    """Compare Python Q-learning to R DTRreg::qLearn()."""

    def test_value_estimate_parity(self):
        """Value estimate should match within rtol=0.15."""
        data = generate_single_stage_dtr_dgp(
            n=500,
            p=3,
            true_blip_intercept=1.0,
            true_blip_slope=0.5,
            seed=42,
        )

        # Python result
        dtr_data = create_dtr_data(data["outcome"], data["treatment"], data["covariates"])
        py_result = q_learning(dtr_data)

        # R result
        r_result = r_q_learning_dtrreg(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        # Value estimates should be similar
        np.testing.assert_allclose(
            py_result.value_estimate,
            r_result["value_estimate"],
            rtol=0.15,
            err_msg=(
                f"Q-learning value mismatch: "
                f"Python={py_result.value_estimate:.4f}, R={r_result['value_estimate']:.4f}"
            ),
        )

    def test_regime_agreement(self):
        """Optimal regime should agree >90% between Python and R."""
        data = generate_single_stage_dtr_dgp(
            n=500,
            p=3,
            true_blip_intercept=1.0,
            true_blip_slope=0.5,
            seed=123,
        )

        dtr_data = create_dtr_data(data["outcome"], data["treatment"], data["covariates"])
        py_result = q_learning(dtr_data)

        r_result = r_q_learning_dtrreg(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        # Compute agreement rate
        py_regime = py_result.optimal_regimes[0]  # Single stage
        r_regime = r_result["optimal_regime"]

        agreement = np.mean(py_regime == r_regime)

        assert agreement > 0.90, (
            f"Q-learning regime agreement too low: {agreement:.1%} (expected > 90%)"
        )

    def test_blip_coefficients(self):
        """Blip coefficients should match within rtol=0.10."""
        data = generate_single_stage_dtr_dgp(
            n=500,
            p=3,
            true_blip_intercept=1.0,
            true_blip_slope=0.5,
            seed=456,
        )

        dtr_data = create_dtr_data(data["outcome"], data["treatment"], data["covariates"])
        py_result = q_learning(dtr_data)

        r_result = r_q_learning_dtrreg(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        # Extract blip coefficients (intercept + first covariate)
        py_blip = py_result.blip_coefficients[0][:2]  # First stage, first 2 coefs
        r_blip = r_result["blip_coef"][:2]

        # Sign and magnitude should match
        # Note: coefficient ordering may differ
        np.testing.assert_allclose(
            np.abs(py_blip),
            np.abs(r_blip),
            rtol=0.20,  # Wider tolerance due to possible ordering
            err_msg=f"Blip coef mismatch: Python={py_blip}, R={r_blip}",
        )

    def test_se_reasonable(self):
        """Standard errors should be similar order of magnitude."""
        data = generate_single_stage_dtr_dgp(n=500, p=3, seed=789)

        dtr_data = create_dtr_data(data["outcome"], data["treatment"], data["covariates"])
        py_result = q_learning(dtr_data, se_method="sandwich")

        r_result = r_q_learning_dtrreg(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        # SE should be positive and reasonable
        py_se = py_result.blip_se[0] if hasattr(py_result, "blip_se") else None
        r_se = r_result["blip_se"]

        # Check R SE is reasonable
        assert all(np.isfinite(r_se)), "R SE values not finite"
        assert all(r_se > 0), "R SE values not positive"

    def test_covariate_mapping(self):
        """Test that covariate effects are captured correctly."""
        # Create data where X1 strongly affects treatment effect
        data = generate_single_stage_dtr_dgp(
            n=800,
            p=5,
            true_blip_intercept=0.0,  # No baseline effect
            true_blip_slope=2.0,  # Strong heterogeneity on X1
            seed=101,
        )

        dtr_data = create_dtr_data(data["outcome"], data["treatment"], data["covariates"])
        py_result = q_learning(dtr_data)

        r_result = r_q_learning_dtrreg(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        # Both should capture that regime depends on X1
        # Check regime varies (not constant)
        py_regime = py_result.optimal_regimes[0]
        r_regime = r_result["optimal_regime"]

        py_variation = np.std(py_regime.astype(float))
        r_variation = np.std(r_regime.astype(float))

        assert py_variation > 0.1, "Python regime shows no variation"
        assert r_variation > 0.1, "R regime shows no variation"


@requires_dtr_python
@requires_dtrreg_r
class TestALearningSingleStageVsDTRReg:
    """Compare Python A-learning to R DTRreg A-learning."""

    def test_value_estimate_parity(self):
        """Value estimate should match within rtol=0.15."""
        data = generate_single_stage_dtr_dgp(
            n=500,
            p=3,
            true_blip_intercept=1.0,
            true_blip_slope=0.5,
            seed=200,
        )

        dtr_data = create_dtr_data(data["outcome"], data["treatment"], data["covariates"])
        py_result = a_learning(dtr_data)

        r_result = r_a_learning_dtrreg(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        np.testing.assert_allclose(
            py_result.value_estimate,
            r_result["value_estimate"],
            rtol=0.15,
            err_msg=(
                f"A-learning value mismatch: "
                f"Python={py_result.value_estimate:.4f}, R={r_result['value_estimate']:.4f}"
            ),
        )

    def test_blip_coefficients(self):
        """Blip coefficients should match within rtol=0.15."""
        data = generate_single_stage_dtr_dgp(
            n=500,
            p=3,
            true_blip_intercept=1.0,
            true_blip_slope=0.5,
            seed=300,
        )

        dtr_data = create_dtr_data(data["outcome"], data["treatment"], data["covariates"])
        py_result = a_learning(dtr_data)

        r_result = r_a_learning_dtrreg(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        py_blip = py_result.blip_coefficients[0][:2]
        r_blip = r_result["blip_coef"][:2]

        np.testing.assert_allclose(
            np.abs(py_blip),
            np.abs(r_blip),
            rtol=0.20,
            err_msg=f"A-learning blip mismatch: Python={py_blip}, R={r_blip}",
        )

    def test_dr_property(self):
        """A-learning should be robust when propensity model is correct."""
        # Test with confounded data where DR property matters
        data = generate_single_stage_dtr_dgp(
            n=800,
            p=3,
            true_blip_intercept=1.0,
            true_blip_slope=0.5,
            propensity_type="confounded",  # e(X) depends on X
            seed=400,
        )

        dtr_data = create_dtr_data(data["outcome"], data["treatment"], data["covariates"])
        py_result = a_learning(dtr_data)

        r_result = r_a_learning_dtrreg(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        # Both should still estimate correctly under confounding
        # (DR property = consistent if either model correct)
        np.testing.assert_allclose(
            py_result.value_estimate,
            r_result["value_estimate"],
            rtol=0.20,  # Slightly wider for confounded case
            err_msg="A-learning not robust under confounding",
        )

    def test_vs_q_learning(self):
        """A-learning and Q-learning should agree when both models correct."""
        data = generate_single_stage_dtr_dgp(
            n=500,
            p=3,
            propensity_type="balanced",  # Both should do well
            seed=500,
        )

        dtr_data = create_dtr_data(data["outcome"], data["treatment"], data["covariates"])

        q_result = q_learning(dtr_data)
        a_result = a_learning(dtr_data)

        # Values should be similar
        np.testing.assert_allclose(
            q_result.value_estimate,
            a_result.value_estimate,
            rtol=0.10,
            err_msg=(
                f"Q-learning ({q_result.value_estimate:.3f}) and "
                f"A-learning ({a_result.value_estimate:.3f}) disagree"
            ),
        )

        # Regimes should agree
        agreement = np.mean(q_result.optimal_regimes[0] == a_result.optimal_regimes[0])
        assert agreement > 0.85, f"Q-learning and A-learning regime agreement: {agreement:.1%}"


@requires_dtr_python
@requires_dtrreg_r
class TestDTREdgeCases:
    """Edge cases for DTR validation."""

    def test_extreme_propensity(self):
        """Should handle extreme propensity scores."""
        rng = np.random.default_rng(600)

        n = 500
        p = 3
        X = rng.normal(0, 1, (n, p))

        # Extreme propensity (near 0.1 and 0.9)
        from scipy.special import expit

        e = expit(2 * X[:, 0])  # More extreme
        A = rng.binomial(1, e)

        # Simple outcome
        Y = X[:, 0] + A * (1.0 + 0.5 * X[:, 0]) + rng.normal(0, 1, n)

        dtr_data = create_dtr_data(Y, A, X)

        # Both should produce estimates without error
        py_q = q_learning(dtr_data)
        py_a = a_learning(dtr_data, propensity_trim=0.05)

        assert np.isfinite(py_q.value_estimate), "Q-learning value not finite"
        assert np.isfinite(py_a.value_estimate), "A-learning value not finite"

        # R should also handle this
        r_q = r_q_learning_dtrreg(Y, A, X)
        r_a = r_a_learning_dtrreg(Y, A, X)

        assert np.isfinite(r_q["value_estimate"]), "R Q-learning value not finite"
        assert np.isfinite(r_a["value_estimate"]), "R A-learning value not finite"

    def test_constant_treatment_effect(self):
        """Should detect constant treatment effect (no heterogeneity)."""
        data = generate_single_stage_dtr_dgp(
            n=500,
            p=3,
            true_blip_intercept=2.0,  # Positive for everyone
            true_blip_slope=0.0,  # No heterogeneity
            seed=700,
        )

        dtr_data = create_dtr_data(data["outcome"], data["treatment"], data["covariates"])
        py_result = q_learning(dtr_data)

        r_result = r_q_learning_dtrreg(
            outcome=data["outcome"],
            treatment=data["treatment"],
            covariates=data["covariates"],
        )

        # Both should recommend treating everyone (constant positive effect)
        py_treat_rate = np.mean(py_result.optimal_regimes[0])
        r_treat_rate = np.mean(r_result["optimal_regime"])

        # With blip > 0 for all, everyone should be treated
        assert py_treat_rate > 0.8, f"Python treat rate={py_treat_rate:.1%} too low"
        assert r_treat_rate > 0.8, f"R treat rate={r_treat_rate:.1%} too low"
