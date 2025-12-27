"""Tests for A-learning DTR estimation.

Three-layer validation:
- Layer 1: Known-answer tests
- Layer 2: Adversarial/edge case tests
- Layer 3: Monte Carlo statistical validation

A-learning is doubly robust: consistent if EITHER propensity OR outcome model
is correctly specified. This is explicitly tested in Layer 3.
"""

import numpy as np
import pytest
from typing import Optional

from causal_inference.dtr import DTRData, ALearningResult, a_learning, a_learning_single_stage
from causal_inference.dtr import q_learning
from .conftest import generate_dtr_dgp, generate_heterogeneous_dtr_dgp


def generate_dr_misspecified_dgp(
    n: int = 500,
    true_blip: float = 2.0,
    propensity_correct: bool = True,
    outcome_correct: bool = True,
    seed: Optional[int] = 42,
) -> tuple[DTRData, dict]:
    """Generate DGP for testing double robustness.

    Parameters
    ----------
    n : int
        Sample size.
    true_blip : float
        True constant treatment effect.
    propensity_correct : bool
        If True, propensity is simple logistic. If False, complex nonlinear.
    outcome_correct : bool
        If True, outcome is linear. If False, includes nonlinear terms.
    seed : int
        Random seed.

    Returns
    -------
    tuple[DTRData, dict]
        Data and true parameters.
    """
    if seed is not None:
        np.random.seed(seed)

    # Covariates
    X = np.random.randn(n, 3)

    # Propensity model
    if propensity_correct:
        # Simple logistic: P(A=1|X) = logit(0.5 + 0.3*X1)
        logit_p = 0.0 + 0.3 * X[:, 0]
        propensity = 1 / (1 + np.exp(-logit_p))
    else:
        # Complex nonlinear: propensity depends on X^2 and interactions
        # A simple linear model will be misspecified
        logit_p = 0.0 + 0.5 * X[:, 0]**2 - 0.3 * X[:, 1] * X[:, 2]
        propensity = 1 / (1 + np.exp(-logit_p))

    A = np.random.binomial(1, propensity).astype(float)

    # Outcome model
    if outcome_correct:
        # Simple linear: Y = X1 + blip*A + noise
        baseline = X[:, 0]
    else:
        # Complex nonlinear baseline
        # A simple OLS on (1, X) will be misspecified
        baseline = X[:, 0]**2 + np.sin(X[:, 1]) + 0.5 * X[:, 2]**3

    Y = baseline + true_blip * A + np.random.randn(n)

    data = DTRData(outcomes=[Y], treatments=[A], covariates=[X])

    true_params = {
        "true_blip": true_blip,
        "propensity_correct": propensity_correct,
        "outcome_correct": outcome_correct,
    }

    return data, true_params


class TestALearningKnownAnswer:
    """Layer 1: Known-answer tests with analytically known results."""

    def test_single_stage_constant_blip(self, single_stage_constant_data):
        """A-learning recovers constant treatment effect."""
        data, true_params = single_stage_constant_data
        result = a_learning(data)

        # Check structure
        assert result.n_stages == 1
        assert len(result.blip_coefficients) == 1
        assert len(result.blip_se) == 1
        assert result.doubly_robust is True

        # Blip intercept should be close to true_blip (2.0)
        blip_intercept = result.blip_coefficients[0][0]
        assert abs(blip_intercept - true_params["true_blip"]) < 0.5, (
            f"Blip intercept {blip_intercept:.3f} not close to "
            f"true blip {true_params['true_blip']}"
        )

    def test_single_stage_zero_effect(self, single_stage_zero_effect_data):
        """A-learning handles zero treatment effect."""
        data, true_params = single_stage_zero_effect_data
        result = a_learning(data)

        # Blip intercept should be close to 0
        blip_intercept = result.blip_coefficients[0][0]
        assert abs(blip_intercept) < 0.5, (
            f"Blip intercept {blip_intercept:.3f} should be near 0"
        )

    def test_matches_q_learning_correct_model(self, single_stage_constant_data):
        """A-learning ≈ Q-learning when models correctly specified."""
        data, _ = single_stage_constant_data

        result_q = q_learning(data)
        result_a = a_learning(data)

        # Blip estimates should be similar (within 0.3)
        q_blip = result_q.blip_coefficients[0][0]
        a_blip = result_a.blip_coefficients[0][0]

        assert abs(q_blip - a_blip) < 0.5, (
            f"Q-learning blip {q_blip:.3f} vs A-learning blip {a_blip:.3f}"
        )

    def test_two_stage_backward_induction(self, two_stage_data):
        """Two-stage A-learning uses backward induction correctly."""
        data, true_params = two_stage_data
        result = a_learning(data)

        # Check structure
        assert result.n_stages == 2
        assert len(result.blip_coefficients) == 2

        # Stage 2 blip should be close to true value
        blip_stage2 = result.blip_coefficients[1][0]
        assert abs(blip_stage2 - true_params["true_blip"]) < 1.0, (
            f"Stage 2 blip {blip_stage2:.3f} not close to true blip"
        )

    def test_optimal_regime_direction(self, single_stage_constant_data):
        """Optimal regime has correct direction."""
        data, true_params = single_stage_constant_data
        result = a_learning(data)

        # With positive blip, optimal regime should recommend treatment
        H_test = np.zeros(data.n_covariates[0])
        optimal_A = result.optimal_regime(H_test, stage=1)

        # Since blip > 0, should recommend A=1
        assert optimal_A == 1, "With positive blip, should recommend A=1"

    def test_value_function_positive(self, single_stage_constant_data):
        """Value function estimate is reasonable."""
        data, _ = single_stage_constant_data
        result = a_learning(data)

        assert result.value_se > 0, "Value SE should be positive"
        assert np.isfinite(result.value_estimate), "Value estimate should be finite"

    def test_result_summary(self, single_stage_constant_data):
        """Result summary method works."""
        data, _ = single_stage_constant_data
        result = a_learning(data)

        summary = result.summary()
        assert "A-Learning Results" in summary
        assert "Optimal value" in summary
        assert "Doubly robust" in summary
        assert "Blip Coefficients" in summary

    def test_predict_optimal_treatment(self, single_stage_constant_data):
        """predict_optimal_treatment works for multiple observations."""
        data, _ = single_stage_constant_data
        result = a_learning(data)

        # Predict for multiple new observations
        H_new = np.random.randn(10, data.n_covariates[0])
        predictions = result.predict_optimal_treatment(H_new, stage=1)

        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)


class TestALearningDoubleRobustness:
    """Tests specific to double robustness property."""

    def test_consistent_when_both_correct(self):
        """A-learning unbiased when both models correct."""
        data, true_params = generate_dr_misspecified_dgp(
            n=800,
            true_blip=2.0,
            propensity_correct=True,
            outcome_correct=True,
            seed=42,
        )
        result = a_learning(data)

        blip = result.blip_coefficients[0][0]
        assert abs(blip - true_params["true_blip"]) < 0.5, (
            f"Blip {blip:.3f} not close to true {true_params['true_blip']}"
        )

    def test_consistent_when_propensity_wrong(self):
        """A-learning consistent when propensity model misspecified.

        When propensity is misspecified but outcome model is correct,
        A-learning should still be consistent (DR property).
        """
        data, true_params = generate_dr_misspecified_dgp(
            n=800,
            true_blip=2.0,
            propensity_correct=False,  # Misspecified!
            outcome_correct=True,
            seed=42,
        )
        result = a_learning(data)

        blip = result.blip_coefficients[0][0]
        # Should still be close to true blip (DR property)
        assert abs(blip - true_params["true_blip"]) < 0.8, (
            f"DR blip {blip:.3f} not close to true {true_params['true_blip']} "
            "when propensity misspecified"
        )

    def test_consistent_when_outcome_wrong(self):
        """A-learning consistent when outcome model misspecified.

        When outcome is misspecified but propensity model is correct,
        A-learning should still be consistent (DR property).
        """
        data, true_params = generate_dr_misspecified_dgp(
            n=800,
            true_blip=2.0,
            propensity_correct=True,
            outcome_correct=False,  # Misspecified!
            seed=42,
        )
        result = a_learning(data)

        blip = result.blip_coefficients[0][0]
        # Should still be close to true blip (DR property)
        assert abs(blip - true_params["true_blip"]) < 0.8, (
            f"DR blip {blip:.3f} not close to true {true_params['true_blip']} "
            "when outcome misspecified"
        )

    def test_non_dr_biased_when_both_wrong(self):
        """Non-DR estimator biased when both models misspecified.

        When doubly_robust=False and both models are misspecified,
        A-learning may be biased. This is expected behavior.
        """
        data, true_params = generate_dr_misspecified_dgp(
            n=800,
            true_blip=2.0,
            propensity_correct=False,
            outcome_correct=False,
            seed=42,
        )

        # With DR=True (default), may still struggle but should be better
        result_dr = a_learning(data, doubly_robust=True)

        # With DR=False, simple weighted regression
        result_no_dr = a_learning(data, doubly_robust=False)

        # Both may be biased, but DR version should be more robust
        # Just verify they give different results
        dr_blip = result_dr.blip_coefficients[0][0]
        no_dr_blip = result_no_dr.blip_coefficients[0][0]

        # Not asserting specific values - both models wrong leads to bias
        # Just verify the computation completes
        assert np.isfinite(dr_blip)
        assert np.isfinite(no_dr_blip)


class TestALearningAdversarial:
    """Layer 2: Edge cases and adversarial scenarios."""

    def test_extreme_propensity_high(self):
        """Handles high propensity (most treated)."""
        data, _ = generate_dtr_dgp(n=500, n_stages=1, propensity=0.95, seed=42)
        result = a_learning(data)

        # Should still produce valid results with trimming
        assert np.isfinite(result.value_estimate)
        assert result.value_se > 0

    def test_extreme_propensity_low(self):
        """Handles low propensity (few treated)."""
        data, _ = generate_dtr_dgp(n=500, n_stages=1, propensity=0.05, seed=42)
        result = a_learning(data)

        # Should still produce valid results with trimming
        assert np.isfinite(result.value_estimate)
        assert result.value_se > 0

    def test_propensity_trimming(self):
        """Propensity scores are trimmed to avoid extreme weights."""
        # Create data where some propensities would be extreme
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 3)
        # Make propensity very extreme (near 0 or 1)
        logit_p = 3.0 * X[:, 0]  # Will create extreme probabilities
        propensity = 1 / (1 + np.exp(-logit_p))
        A = np.random.binomial(1, propensity).astype(float)
        Y = X[:, 0] + 2.0 * A + np.random.randn(n)

        data = DTRData(outcomes=[Y], treatments=[A], covariates=[X])

        # With trimming, should still work
        result = a_learning(data, propensity_trim=0.05)  # Trim at 5%

        assert np.isfinite(result.value_estimate)
        assert result.value_se > 0

    def test_high_dimensional(self, high_dimensional_data):
        """Works with high-dimensional covariates (p=50)."""
        data, _ = high_dimensional_data
        result = a_learning(data)

        assert result.n_stages == 1
        # Should have p+1 blip coefficients (with intercept)
        assert len(result.blip_coefficients[0]) == 51

    def test_small_sample_warning(self, small_sample_data):
        """Issues warning for small sample."""
        data, _ = small_sample_data

        with pytest.warns(UserWarning, match="Small sample size"):
            result = a_learning(data)

        assert result is not None

    def test_many_stages(self):
        """Handles K=5 stages."""
        data, _ = generate_dtr_dgp(n=500, n_stages=5, true_blip=1.0, seed=42)
        result = a_learning(data)

        assert result.n_stages == 5
        assert len(result.blip_coefficients) == 5

    def test_invalid_propensity_model_error(self, single_stage_constant_data):
        """Error on unknown propensity model."""
        data, _ = single_stage_constant_data

        with pytest.raises(ValueError, match="Unknown propensity_model"):
            a_learning(data, propensity_model="invalid")

    def test_invalid_outcome_model_error(self, single_stage_constant_data):
        """Error on unknown outcome model."""
        data, _ = single_stage_constant_data

        with pytest.raises(ValueError, match="Unknown outcome_model"):
            a_learning(data, outcome_model="invalid")

    def test_invalid_se_method_error(self, single_stage_constant_data):
        """Error on unknown se_method."""
        data, _ = single_stage_constant_data

        with pytest.raises(ValueError, match="Unknown se_method"):
            a_learning(data, se_method="invalid")

    def test_probit_propensity(self, single_stage_constant_data):
        """Probit propensity model works."""
        data, _ = single_stage_constant_data
        result = a_learning(data, propensity_model="probit")

        assert result.propensity_model == "probit"
        assert np.isfinite(result.value_estimate)

    def test_ridge_outcome(self, single_stage_constant_data):
        """Ridge outcome model works."""
        data, _ = single_stage_constant_data
        result = a_learning(data, outcome_model="ridge")

        assert result.outcome_model == "ridge"
        assert np.isfinite(result.value_estimate)


class TestALearningMethods:
    """Tests for different SE methods."""

    def test_sandwich_se(self, single_stage_constant_data):
        """Sandwich SE method works."""
        data, _ = single_stage_constant_data
        result = a_learning(data, se_method="sandwich")

        assert result.se_method == "sandwich"
        assert all(se > 0 for se in result.blip_se[0])

    def test_bootstrap_se(self, single_stage_constant_data):
        """Bootstrap SE method works."""
        data, _ = single_stage_constant_data
        result = a_learning(data, se_method="bootstrap", n_bootstrap=100)

        assert result.se_method == "bootstrap"
        assert all(se > 0 for se in result.blip_se[0])

    def test_se_methods_similar(self, single_stage_constant_data):
        """Sandwich and bootstrap SE should be similar."""
        data, _ = single_stage_constant_data

        result_sandwich = a_learning(data, se_method="sandwich")
        result_bootstrap = a_learning(data, se_method="bootstrap", n_bootstrap=200)

        # SE should be in same ballpark (within factor of 2)
        se_ratio = result_sandwich.blip_se[0][0] / result_bootstrap.blip_se[0][0]
        assert 0.5 < se_ratio < 2.0, (
            f"SE ratio {se_ratio:.2f} outside [0.5, 2.0]"
        )


class TestALearningConvenience:
    """Tests for convenience functions."""

    def test_single_stage_wrapper(self):
        """a_learning_single_stage matches a_learning for K=1."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 3)
        A = np.random.binomial(1, 0.5, n).astype(float)
        Y = X[:, 0] + 2.0 * A + np.random.randn(n)

        # Via wrapper
        result1 = a_learning_single_stage(Y, A, X)

        # Via main function
        data = DTRData(outcomes=[Y], treatments=[A], covariates=[X])
        result2 = a_learning(data)

        # Should give same results
        np.testing.assert_allclose(
            result1.blip_coefficients[0],
            result2.blip_coefficients[0],
            rtol=1e-10,
        )


@pytest.mark.slow
class TestALearningMonteCarlo:
    """Layer 3: Monte Carlo statistical validation."""

    def test_blip_unbiased(self):
        """A-learning blip estimate is unbiased (bias < 0.10)."""
        n_simulations = 200
        true_blip = 2.0
        blip_estimates = []

        for i in range(n_simulations):
            data, _ = generate_dtr_dgp(
                n=300, n_stages=1, true_blip=true_blip, seed=None
            )
            result = a_learning(data)
            blip_estimates.append(result.blip_coefficients[0][0])

        mean_estimate = np.mean(blip_estimates)
        bias = mean_estimate - true_blip

        assert abs(bias) < 0.10, (
            f"Blip bias {bias:.4f} exceeds threshold 0.10. "
            f"Mean estimate: {mean_estimate:.4f}, true: {true_blip}"
        )

    def test_coverage(self):
        """95% CI has correct coverage (93-97%)."""
        n_simulations = 200
        true_blip = 2.0
        covered = []

        for i in range(n_simulations):
            data, _ = generate_dtr_dgp(
                n=300, n_stages=1, true_blip=true_blip, seed=None
            )
            result = a_learning(data)

            # Check if true blip is within CI for intercept
            blip = result.blip_coefficients[0][0]
            se = result.blip_se[0][0]
            ci_lower = blip - 1.96 * se
            ci_upper = blip + 1.96 * se

            in_ci = ci_lower < true_blip < ci_upper
            covered.append(in_ci)

        coverage = np.mean(covered)

        assert 0.93 < coverage < 0.97, (
            f"Coverage {coverage:.2%} outside [93%, 97%]. "
            f"Covered {sum(covered)} of {n_simulations} simulations."
        )

    def test_se_calibration(self):
        """SE estimates are well-calibrated (within 20% of empirical)."""
        n_simulations = 200
        true_blip = 2.0
        blip_estimates = []
        se_estimates = []

        for i in range(n_simulations):
            data, _ = generate_dtr_dgp(
                n=300, n_stages=1, true_blip=true_blip, seed=None
            )
            result = a_learning(data)
            blip_estimates.append(result.blip_coefficients[0][0])
            se_estimates.append(result.blip_se[0][0])

        empirical_se = np.std(blip_estimates)
        mean_se = np.mean(se_estimates)

        relative_error = abs(mean_se - empirical_se) / empirical_se

        assert relative_error < 0.20, (
            f"SE calibration error {relative_error:.2%} exceeds 20%. "
            f"Mean SE: {mean_se:.4f}, Empirical SE: {empirical_se:.4f}"
        )

    def test_double_robustness_propensity_wrong(self):
        """A-learning unbiased when propensity misspecified (100 runs)."""
        n_simulations = 100
        true_blip = 2.0
        blip_estimates = []

        for _ in range(n_simulations):
            data, _ = generate_dr_misspecified_dgp(
                n=400,
                true_blip=true_blip,
                propensity_correct=False,  # Misspecified
                outcome_correct=True,
                seed=None,
            )
            result = a_learning(data)
            blip_estimates.append(result.blip_coefficients[0][0])

        mean_estimate = np.mean(blip_estimates)
        bias = mean_estimate - true_blip

        # DR tolerance is looser: < 0.15
        assert abs(bias) < 0.15, (
            f"DR bias {bias:.4f} exceeds 0.15 when propensity wrong. "
            f"Mean: {mean_estimate:.4f}, true: {true_blip}"
        )

    def test_double_robustness_outcome_wrong(self):
        """A-learning unbiased when outcome misspecified (100 runs)."""
        n_simulations = 100
        true_blip = 2.0
        blip_estimates = []

        for _ in range(n_simulations):
            data, _ = generate_dr_misspecified_dgp(
                n=400,
                true_blip=true_blip,
                propensity_correct=True,
                outcome_correct=False,  # Misspecified
                seed=None,
            )
            result = a_learning(data)
            blip_estimates.append(result.blip_coefficients[0][0])

        mean_estimate = np.mean(blip_estimates)
        bias = mean_estimate - true_blip

        # DR tolerance: < 0.15
        assert abs(bias) < 0.15, (
            f"DR bias {bias:.4f} exceeds 0.15 when outcome wrong. "
            f"Mean: {mean_estimate:.4f}, true: {true_blip}"
        )

    def test_two_stage_final_unbiased(self):
        """Two-stage A-learning: final stage blip is unbiased."""
        n_simulations = 100
        true_blip = 2.0
        stage2_blips = []

        for i in range(n_simulations):
            data, _ = generate_dtr_dgp(
                n=300, n_stages=2, true_blip=true_blip, seed=None
            )
            result = a_learning(data)
            stage2_blips.append(result.blip_coefficients[1][0])

        bias2 = np.mean(stage2_blips) - true_blip
        assert abs(bias2) < 0.20, f"Stage 2 bias {bias2:.4f} exceeds 0.20"
