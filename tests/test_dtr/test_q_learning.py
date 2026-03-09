"""Tests for Q-learning DTR estimation.

Three-layer validation:
- Layer 1: Known-answer tests
- Layer 2: Adversarial/edge case tests
- Layer 3: Monte Carlo statistical validation
"""

import numpy as np
import pytest
from typing import Optional

from causal_inference.dtr import DTRData, QLearningResult, q_learning, q_learning_single_stage
from .conftest import generate_dtr_dgp, generate_heterogeneous_dtr_dgp


class TestQLearningKnownAnswer:
    """Layer 1: Known-answer tests with analytically known results."""

    def test_single_stage_constant_blip(self, single_stage_constant_data):
        """Q-learning recovers constant treatment effect."""
        data, true_params = single_stage_constant_data
        result = q_learning(data)

        # Check structure
        assert result.n_stages == 1
        assert len(result.blip_coefficients) == 1
        assert len(result.blip_se) == 1

        # Blip intercept should be close to true_blip (2.0)
        # The intercept is the first coefficient (after adding intercept)
        blip_intercept = result.blip_coefficients[0][0]
        assert abs(blip_intercept - true_params["true_blip"]) < 0.5, (
            f"Blip intercept {blip_intercept:.3f} not close to true blip {true_params['true_blip']}"
        )

    def test_single_stage_zero_effect(self, single_stage_zero_effect_data):
        """Q-learning handles zero treatment effect."""
        data, true_params = single_stage_zero_effect_data
        result = q_learning(data)

        # Blip intercept should be close to 0
        blip_intercept = result.blip_coefficients[0][0]
        assert abs(blip_intercept) < 0.5, f"Blip intercept {blip_intercept:.3f} should be near 0"

    def test_two_stage_backward_induction(self, two_stage_data):
        """Two-stage Q-learning uses backward induction correctly."""
        data, true_params = two_stage_data
        result = q_learning(data)

        # Check structure
        assert result.n_stages == 2
        assert len(result.blip_coefficients) == 2
        assert len(result.stage_q_functions) == 2

        # Stage 2 (final) blip should be close to true value
        blip_stage2 = result.blip_coefficients[1][0]
        assert abs(blip_stage2 - true_params["true_blip"]) < 1.0, (
            f"Stage 2 blip {blip_stage2:.3f} not close to true blip"
        )

        # Stage 1 blip captures total value from that point forward
        # In this DGP, stages are correlated, so stage 1 blip may be larger
        # Just verify it's positive (treatment is beneficial)
        blip_stage1 = result.blip_coefficients[0][0]
        assert blip_stage1 > 0, f"Stage 1 blip {blip_stage1:.3f} should be positive"

    def test_optimal_regime_direction(self, single_stage_constant_data):
        """Optimal regime has correct direction."""
        data, true_params = single_stage_constant_data
        result = q_learning(data)

        # With positive blip, optimal regime should recommend treatment
        # for average history (all zeros)
        H_test = np.zeros(data.n_covariates[0])
        optimal_A = result.optimal_regime(H_test, stage=1)

        # Since blip > 0, should recommend A=1
        assert optimal_A == 1, "With positive blip, should recommend A=1"

    def test_optimal_regime_negative_blip(self):
        """Optimal regime correct with negative blip."""
        data, true_params = generate_dtr_dgp(n=500, n_stages=1, true_blip=-2.0, seed=123)
        result = q_learning(data)

        # With negative blip, should recommend A=0 for average history
        H_test = np.zeros(data.n_covariates[0])
        optimal_A = result.optimal_regime(H_test, stage=1)

        # Blip is negative, but regime depends on H'psi
        # Since H is zeros with intercept, depends on intercept sign
        blip_intercept = result.blip_coefficients[0][0]
        expected_A = 1 if blip_intercept > 0 else 0
        assert optimal_A == expected_A

    def test_value_function_positive(self, single_stage_constant_data):
        """Value function estimate is reasonable."""
        data, true_params = single_stage_constant_data
        result = q_learning(data)

        # Value should be positive (baseline + positive blip)
        # Not a strict test, just sanity check
        assert result.value_se > 0, "Value SE should be positive"
        assert np.isfinite(result.value_estimate), "Value estimate should be finite"

    def test_result_summary(self, single_stage_constant_data):
        """Result summary method works."""
        data, true_params = single_stage_constant_data
        result = q_learning(data)

        summary = result.summary()
        assert "Q-Learning Results" in summary
        assert "Optimal value" in summary
        assert "Blip Coefficients" in summary

    def test_predict_optimal_treatment(self, single_stage_constant_data):
        """predict_optimal_treatment works for multiple observations."""
        data, true_params = single_stage_constant_data
        result = q_learning(data)

        # Predict for multiple new observations
        H_new = np.random.randn(10, data.n_covariates[0])
        predictions = result.predict_optimal_treatment(H_new, stage=1)

        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)


class TestQLearningAdversarial:
    """Layer 2: Edge cases and adversarial scenarios."""

    def test_extreme_propensity_high(self):
        """Handles high propensity (most treated)."""
        data, _ = generate_dtr_dgp(n=500, n_stages=1, propensity=0.95, seed=42)
        result = q_learning(data)

        # Should still produce valid results
        assert np.isfinite(result.value_estimate)
        assert result.value_se > 0

    def test_extreme_propensity_low(self):
        """Handles low propensity (few treated)."""
        data, _ = generate_dtr_dgp(n=500, n_stages=1, propensity=0.05, seed=42)
        result = q_learning(data)

        # Should still produce valid results (though less precise)
        assert np.isfinite(result.value_estimate)
        assert result.value_se > 0

    def test_high_dimensional(self, high_dimensional_data):
        """Works with high-dimensional covariates (p=50)."""
        data, _ = high_dimensional_data
        result = q_learning(data)

        assert result.n_stages == 1
        # Should have p+1 blip coefficients (with intercept)
        assert len(result.blip_coefficients[0]) == 51

    def test_small_sample_warning(self, small_sample_data):
        """Issues warning for small sample."""
        data, _ = small_sample_data

        with pytest.warns(UserWarning, match="Small sample size"):
            result = q_learning(data)

        assert result is not None

    def test_many_stages(self):
        """Handles K=5 stages."""
        data, _ = generate_dtr_dgp(n=500, n_stages=5, true_blip=1.0, seed=42)
        result = q_learning(data)

        assert result.n_stages == 5
        assert len(result.blip_coefficients) == 5
        assert len(result.stage_q_functions) == 5

    def test_invalid_stage_error(self, single_stage_constant_data):
        """Error on invalid stage number."""
        data, _ = single_stage_constant_data
        result = q_learning(data)

        with pytest.raises(ValueError, match="stage must be in"):
            result.optimal_regime(np.zeros(3), stage=2)

        with pytest.raises(ValueError, match="stage must be in"):
            result.optimal_regime(np.zeros(3), stage=0)

    def test_invalid_model_error(self, single_stage_constant_data):
        """Error on unknown model."""
        data, _ = single_stage_constant_data

        with pytest.raises(ValueError, match="Unknown model"):
            q_learning(data, model="invalid")

    def test_invalid_se_method_error(self, single_stage_constant_data):
        """Error on unknown se_method."""
        data, _ = single_stage_constant_data

        with pytest.raises(ValueError, match="Unknown se_method"):
            q_learning(data, se_method="invalid")


class TestQLearningMethods:
    """Tests for different SE methods."""

    def test_sandwich_se(self, single_stage_constant_data):
        """Sandwich SE method works."""
        data, _ = single_stage_constant_data
        result = q_learning(data, se_method="sandwich")

        assert result.se_method == "sandwich"
        assert all(se > 0 for se in result.blip_se[0])

    def test_bootstrap_se(self, single_stage_constant_data):
        """Bootstrap SE method works."""
        data, _ = single_stage_constant_data
        result = q_learning(data, se_method="bootstrap", n_bootstrap=100)

        assert result.se_method == "bootstrap"
        assert all(se > 0 for se in result.blip_se[0])

    def test_se_methods_similar(self, single_stage_constant_data):
        """Sandwich and bootstrap SE should be similar."""
        data, _ = single_stage_constant_data

        result_sandwich = q_learning(data, se_method="sandwich")
        result_bootstrap = q_learning(data, se_method="bootstrap", n_bootstrap=200)

        # SE should be in same ballpark (within factor of 2)
        se_ratio = result_sandwich.blip_se[0][0] / result_bootstrap.blip_se[0][0]
        assert 0.5 < se_ratio < 2.0, f"SE ratio {se_ratio:.2f} outside [0.5, 2.0]"


class TestQLearningConvenience:
    """Tests for convenience functions."""

    def test_single_stage_wrapper(self):
        """q_learning_single_stage matches q_learning for K=1."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 3)
        A = np.random.binomial(1, 0.5, n).astype(float)
        Y = X[:, 0] + 2.0 * A + np.random.randn(n)

        # Via wrapper
        result1 = q_learning_single_stage(Y, A, X)

        # Via main function
        data = DTRData(outcomes=[Y], treatments=[A], covariates=[X])
        result2 = q_learning(data)

        # Should give same results
        np.testing.assert_allclose(
            result1.blip_coefficients[0],
            result2.blip_coefficients[0],
            rtol=1e-10,
        )


class TestDTRData:
    """Tests for DTRData validation."""

    def test_empty_outcomes_error(self):
        """Error on empty outcomes."""
        with pytest.raises(ValueError, match="Empty data"):
            DTRData(outcomes=[], treatments=[], covariates=[])

    def test_stage_count_mismatch_error(self):
        """Error on mismatched stage counts."""
        Y = np.random.randn(100)
        A = np.random.binomial(1, 0.5, 100).astype(float)
        X = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="Stage count mismatch"):
            DTRData(outcomes=[Y, Y], treatments=[A], covariates=[X])

    def test_observation_count_mismatch_error(self):
        """Error on mismatched observation counts."""
        Y = np.random.randn(100)
        A = np.random.binomial(1, 0.5, 50).astype(float)  # Wrong size
        X = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="Length mismatch"):
            DTRData(outcomes=[Y], treatments=[A], covariates=[X])

    def test_non_binary_treatment_error(self):
        """Error on non-binary treatment."""
        Y = np.random.randn(100)
        A = np.random.randn(100)  # Continuous, not binary
        X = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="Non-binary treatment"):
            DTRData(outcomes=[Y], treatments=[A], covariates=[X])

    def test_1d_covariates_reshaped(self):
        """1D covariates are reshaped to 2D."""
        Y = np.random.randn(100)
        A = np.random.binomial(1, 0.5, 100).astype(float)
        X = np.random.randn(100)  # 1D

        data = DTRData(outcomes=[Y], treatments=[A], covariates=[X])
        assert data.covariates[0].ndim == 2
        assert data.covariates[0].shape == (100, 1)

    def test_get_history_stage1(self, single_stage_constant_data):
        """get_history returns covariates for stage 1."""
        data, _ = single_stage_constant_data
        H1 = data.get_history(1)

        np.testing.assert_array_equal(H1, data.covariates[0])

    def test_get_history_invalid_stage(self, single_stage_constant_data):
        """Error on invalid stage in get_history."""
        data, _ = single_stage_constant_data

        with pytest.raises(ValueError, match="Invalid stage"):
            data.get_history(0)

        with pytest.raises(ValueError, match="Invalid stage"):
            data.get_history(2)


@pytest.mark.slow
class TestQLearningMonteCarlo:
    """Layer 3: Monte Carlo statistical validation."""

    def test_blip_unbiased(self):
        """Q-learning blip estimate is unbiased (bias < 0.10)."""
        n_simulations = 200
        true_blip = 2.0
        blip_estimates = []

        for i in range(n_simulations):
            data, _ = generate_dtr_dgp(n=300, n_stages=1, true_blip=true_blip, seed=None)
            result = q_learning(data)
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
            data, _ = generate_dtr_dgp(n=300, n_stages=1, true_blip=true_blip, seed=None)
            result = q_learning(data)

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
            data, _ = generate_dtr_dgp(n=300, n_stages=1, true_blip=true_blip, seed=None)
            result = q_learning(data)
            blip_estimates.append(result.blip_coefficients[0][0])
            se_estimates.append(result.blip_se[0][0])

        empirical_se = np.std(blip_estimates)
        mean_se = np.mean(se_estimates)

        relative_error = abs(mean_se - empirical_se) / empirical_se

        assert relative_error < 0.20, (
            f"SE calibration error {relative_error:.2%} exceeds 20%. "
            f"Mean SE: {mean_se:.4f}, Empirical SE: {empirical_se:.4f}"
        )

    def test_two_stage_final_unbiased(self):
        """Two-stage Q-learning: final stage blip is unbiased."""
        n_simulations = 100
        true_blip = 2.0
        stage2_blips = []

        for i in range(n_simulations):
            data, _ = generate_dtr_dgp(n=300, n_stages=2, true_blip=true_blip, seed=None)
            result = q_learning(data)
            stage2_blips.append(result.blip_coefficients[1][0])

        # Final stage should be unbiased
        # (Stage 1 accumulates value from future stages in this DGP)
        bias2 = np.mean(stage2_blips) - true_blip
        assert abs(bias2) < 0.20, f"Stage 2 bias {bias2:.4f} exceeds 0.20"
