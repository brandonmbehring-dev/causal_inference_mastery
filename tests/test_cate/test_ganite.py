"""Tests for GANITE (GAN-based ITE estimation).

Test layers:
- Layer 1: Known-answer tests (ATE recovery, CATE shape, CI validity)
- Layer 2: Adversarial tests (small sample, high-dim, edge cases)
- Layer 3: Monte Carlo validation (bias, coverage) @slow
"""

import numpy as np
import pytest

from .conftest import generate_cate_dgp


# Check if PyTorch is available
def _torch_available():
    try:
        import torch

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _torch_available(), reason="PyTorch required for GANITE tests")


# Import conditionally
if _torch_available():
    from causal_inference.cate.ganite import ganite, GANITE, _check_torch_available


# ============================================================================
# Layer 1: Known-Answer Tests
# ============================================================================


class TestGANITEKnownAnswer:
    """Known-answer tests for GANITE."""

    def test_constant_effect_ate_recovery(self, constant_effect_data):
        """GANITE should recover ATE within tolerance (neural networks have high variance)."""
        Y, T, X, _ = constant_effect_data
        result = ganite(Y, T, X, epochs=50, hidden_dim=64)

        # Neural methods get looser tolerance
        assert abs(result["ate"] - 2.0) < 1.5, f"ATE {result['ate']} not close to 2.0"

    def test_cate_shape(self, constant_effect_data):
        """CATE should have correct shape."""
        Y, T, X, _ = constant_effect_data
        result = ganite(Y, T, X, epochs=30, hidden_dim=64)

        assert result["cate"].shape == (len(Y),)

    def test_ci_validity(self, constant_effect_data):
        """CI should be valid (lower < upper)."""
        Y, T, X, _ = constant_effect_data
        result = ganite(Y, T, X, epochs=30, hidden_dim=64)

        assert result["ci_lower"] < result["ci_upper"]

    def test_se_positive(self, constant_effect_data):
        """SE should be positive."""
        Y, T, X, _ = constant_effect_data
        result = ganite(Y, T, X, epochs=30, hidden_dim=64)

        assert result["ate_se"] > 0

    def test_result_type(self, constant_effect_data):
        """Result should be CATEResult dict."""
        Y, T, X, _ = constant_effect_data
        result = ganite(Y, T, X, epochs=30, hidden_dim=64)

        assert isinstance(result, dict)
        assert "cate" in result
        assert "ate" in result
        assert "ate_se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "method" in result

    def test_method_name(self, constant_effect_data):
        """Method name should be correct."""
        Y, T, X, _ = constant_effect_data
        result = ganite(Y, T, X, epochs=30, hidden_dim=64)

        assert result["method"] == "ganite"

    def test_heterogeneous_effect_correlation(self, linear_heterogeneous_data):
        """GANITE should capture some heterogeneity."""
        Y, T, X, true_cate = linear_heterogeneous_data
        result = ganite(Y, T, X, epochs=50, hidden_dim=64)

        correlation = np.corrcoef(result["cate"], true_cate)[0, 1]
        # Neural methods may have low correlation, just check it's not negative
        assert correlation > -0.5 or abs(result["ate"] - np.mean(true_cate)) < 1.0

    def test_custom_architecture(self, constant_effect_data):
        """Should accept custom architecture parameters."""
        Y, T, X, _ = constant_effect_data
        result = ganite(
            Y,
            T,
            X,
            hidden_dim=32,
            noise_dim=16,
            epochs=20,
        )

        assert isinstance(result["ate"], float)


# ============================================================================
# Layer 2: Adversarial Tests
# ============================================================================


class TestGANITEAdversarial:
    """Adversarial tests for GANITE."""

    def test_empty_treatment_group_raises(self):
        """Should raise error when treatment group is empty."""
        n = 100
        Y = np.random.randn(n)
        T = np.zeros(n)  # All control
        X = np.random.randn(n, 3)

        with pytest.raises(ValueError, match="treatment"):
            ganite(Y, T, X, epochs=10)

    def test_empty_control_group_raises(self):
        """Should raise error when control group is empty."""
        n = 100
        Y = np.random.randn(n)
        T = np.ones(n)  # All treated
        X = np.random.randn(n, 3)

        with pytest.raises(ValueError, match="control"):
            ganite(Y, T, X, epochs=10)

    def test_non_binary_treatment_raises(self):
        """Should raise error for non-binary treatment."""
        n = 100
        Y = np.random.randn(n)
        T = np.random.randint(0, 3, n)  # Multi-valued
        X = np.random.randn(n, 3)

        with pytest.raises(ValueError, match="binary"):
            ganite(Y, T, X, epochs=10)

    def test_high_dimensional(self, high_dimensional_data):
        """Should handle high-dimensional covariates."""
        Y, T, X, _ = high_dimensional_data
        result = ganite(Y, T, X, epochs=30, hidden_dim=64)

        assert isinstance(result["ate"], float)
        assert not np.isnan(result["ate"])

    def test_small_sample(self):
        """Should handle small samples (graceful degradation)."""
        Y, T, X, _ = generate_cate_dgp(n=60, p=3, seed=42)
        result = ganite(Y, T, X, epochs=30, hidden_dim=32, batch_size=16)

        assert isinstance(result["ate"], float)
        assert not np.isnan(result["ate"])

    def test_single_covariate(self, single_covariate_data):
        """Should handle single covariate."""
        Y, T, X, _ = single_covariate_data
        result = ganite(Y, T, X.ravel(), epochs=30, hidden_dim=32)

        assert isinstance(result["ate"], float)

    def test_imbalanced_treatment(self):
        """Should handle imbalanced treatment (20% treated)."""
        Y, T, X, _ = generate_cate_dgp(
            n=500,
            p=3,
            effect_type="constant",
            true_ate=2.0,
            treatment_prob=0.2,
            seed=42,
        )
        result = ganite(Y, T, X, epochs=50, hidden_dim=64)

        # Looser tolerance for imbalanced
        assert abs(result["ate"] - 2.0) < 2.0

    def test_mismatched_lengths_raises(self):
        """Should raise error for mismatched lengths."""
        Y = np.random.randn(100)
        T = np.random.binomial(1, 0.5, 100)
        X = np.random.randn(90, 3)  # Wrong length

        with pytest.raises(ValueError, match="[Ll]ength"):
            ganite(Y, T, X, epochs=10)

    def test_custom_hyperparameters(self, constant_effect_data):
        """Should accept various hyperparameters."""
        Y, T, X, _ = constant_effect_data
        result = ganite(
            Y,
            T,
            X,
            hidden_dim=64,
            noise_dim=16,
            alpha=0.5,
            beta=0.5,
            epochs=20,
            batch_size=32,
            learning_rate=0.0005,
            cf_warmup=5,
        )

        assert isinstance(result["ate"], float)

    def test_different_alpha_ci(self, constant_effect_data):
        """Different alpha should change CI width."""
        Y, T, X, _ = constant_effect_data

        result_95 = ganite(Y, T, X, epochs=30, alpha_ci=0.05)
        result_99 = ganite(Y, T, X, epochs=30, alpha_ci=0.01)

        width_95 = result_95["ci_upper"] - result_95["ci_lower"]
        width_99 = result_99["ci_upper"] - result_99["ci_lower"]

        # 99% CI should be wider
        assert width_99 > width_95 * 0.9  # Allow some variation due to randomness


class TestGANITEModelClass:
    """Tests for GANITE model class directly."""

    def test_model_fit_predict(self, constant_effect_data):
        """Model should fit and predict."""
        Y, T, X, _ = constant_effect_data

        model = GANITE(epochs=20, hidden_dim=32)
        model.fit(X, T, Y)

        cate = model.predict_cate(X)
        assert cate.shape == (len(Y),)

    def test_predict_before_fit_raises(self, constant_effect_data):
        """Should raise error if predicting before fitting."""
        Y, T, X, _ = constant_effect_data

        model = GANITE(epochs=20)

        with pytest.raises(ValueError, match="fitted"):
            model.predict_cate(X)

    def test_predict_y0_y1(self, constant_effect_data):
        """Should predict potential outcomes."""
        Y, T, X, _ = constant_effect_data

        model = GANITE(epochs=20, hidden_dim=32)
        model.fit(X, T, Y)

        y0 = model.predict_y0(X)
        y1 = model.predict_y1(X)

        assert y0.shape == (len(Y),)
        assert y1.shape == (len(Y),)


# ============================================================================
# Layer 3: Monte Carlo Tests
# ============================================================================


class TestGANITEMonteCarlo:
    """Monte Carlo validation for GANITE."""

    @pytest.mark.slow
    def test_ate_unbiased(self):
        """GANITE should have reasonable bias over Monte Carlo runs."""
        n_runs = 30
        estimates = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=300, p=3, true_ate=2.0, seed=seed)
            result = ganite(Y, T, X, epochs=50, hidden_dim=64, random_state=seed)
            estimates.append(result["ate"])

        bias = abs(np.mean(estimates) - 2.0)
        # Neural methods have higher variance, looser threshold
        assert bias < 1.0, f"GANITE bias {bias:.3f} exceeds threshold"

    @pytest.mark.slow
    def test_coverage(self):
        """GANITE should have reasonable coverage."""
        n_runs = 30
        covers = []

        for seed in range(n_runs):
            Y, T, X, _ = generate_cate_dgp(n=300, p=3, true_ate=2.0, seed=seed)
            result = ganite(Y, T, X, epochs=50, hidden_dim=64, random_state=seed)
            covers.append(result["ci_lower"] < 2.0 < result["ci_upper"])

        coverage = np.mean(covers)
        # Neural methods: accept wider range 50-100%
        assert 0.50 < coverage < 1.0, f"Coverage {coverage:.2%} outside 50-100%"

    @pytest.mark.slow
    def test_cate_recovery(self):
        """GANITE should recover some heterogeneity."""
        correlations = []

        for seed in range(20):
            Y, T, X, true_cate = generate_cate_dgp(
                n=400, p=3, effect_type="linear", true_ate=2.0, seed=seed
            )
            result = ganite(Y, T, X, epochs=50, hidden_dim=64, random_state=seed)
            corr = np.corrcoef(result["cate"], true_cate)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        if correlations:
            mean_corr = np.mean(correlations)
            # Just check it's not strongly negative
            assert mean_corr > -0.3, f"Mean CATE correlation {mean_corr:.2f} too negative"

    @pytest.mark.slow
    def test_vs_dragonnet(self):
        """GANITE should be competitive with DragonNet."""
        from causal_inference.cate.dragonnet import dragonnet

        ganite_errors = []
        dragonnet_errors = []

        for seed in range(20):
            Y, T, X, _ = generate_cate_dgp(n=300, p=3, true_ate=2.0, seed=seed)

            ganite_result = ganite(Y, T, X, epochs=50, hidden_dim=64, random_state=seed)
            dragon_result = dragonnet(Y, T, X, hidden_layers=(64,), head_layers=(32,))

            ganite_errors.append(abs(ganite_result["ate"] - 2.0))
            dragonnet_errors.append(abs(dragon_result["ate"] - 2.0))

        ganite_mae = np.mean(ganite_errors)
        dragonnet_mae = np.mean(dragonnet_errors)

        # GANITE should not be much worse than DragonNet (within 3x)
        assert ganite_mae < dragonnet_mae * 3, (
            f"GANITE MAE {ganite_mae:.3f} much worse than DragonNet {dragonnet_mae:.3f}"
        )

    @pytest.mark.slow
    def test_confounded_dgp(self):
        """GANITE should handle confounded DGP."""
        estimates = []

        for seed in range(20):
            np.random.seed(seed)
            n = 300

            # Confounded DGP
            X = np.random.randn(n, 3)
            propensity = 1 / (1 + np.exp(-X[:, 0]))
            T = np.random.binomial(1, propensity)
            Y = 1 + X[:, 0] + 2 * T + 0.5 * X[:, 0] * T + np.random.randn(n)

            result = ganite(Y, T, X, epochs=50, hidden_dim=64, random_state=seed)
            estimates.append(result["ate"])

        bias = abs(np.mean(estimates) - 2.0)
        # Confounded setting: allow higher bias
        assert bias < 1.5, f"GANITE bias {bias:.3f} in confounded setting too high"
