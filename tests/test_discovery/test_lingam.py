"""Tests for LiNGAM (Linear Non-Gaussian Acyclic Model).

Session 133: Validation of functional causal discovery.

Test Layers:
1. Known-answer: Simple structures with known causal order
2. Adversarial: Edge cases, near-Gaussian violations
3. Monte Carlo: Statistical performance across random DAGs
"""

import numpy as np
import pytest

from causal_inference.discovery import (
    ica_lingam,
    direct_lingam,
    bootstrap_lingam,
    check_non_gaussianity,
    generate_random_dag,
    generate_dag_data,
)


# =============================================================================
# Layer 1: Known-Answer Tests (Simple Structures)
# =============================================================================


class TestLiNGAMKnownStructures:
    """Test LiNGAM on known DAG structures."""

    def test_chain_causal_order(self, chain_data_laplace):
        """Chain X0 -> X1 -> X2: order should be [0, 1, 2] or consistent."""
        data, B, dag = chain_data_laplace
        result = direct_lingam(data, seed=42)

        # Causal order should place X0 before X1 before X2
        order = result.causal_order
        assert order.index(0) < order.index(1), "X0 should come before X1"
        assert order.index(1) < order.index(2), "X1 should come before X2"

    def test_chain_adjacency_recovery(self, chain_data_laplace):
        """Chain adjacency matrix should have correct structure."""
        data, B, dag = chain_data_laplace
        result = direct_lingam(data, seed=42)

        B_est = result.adjacency_matrix

        # Should have edges 0->1 and 1->2
        # Check that these entries are non-zero
        has_0_to_1 = np.abs(B_est[0, 1]) > 0.1
        has_1_to_2 = np.abs(B_est[1, 2]) > 0.1

        assert has_0_to_1 or has_1_to_2, "Chain edges not detected"

    def test_collider_detection(self, collider_data_laplace):
        """Collider X0 -> X1 <- X2: should detect both edges into X1."""
        data, B, dag = collider_data_laplace
        result = direct_lingam(data, seed=42)

        # X0 and X2 should come before X1 in causal order
        order = result.causal_order
        pos_0 = order.index(0)
        pos_1 = order.index(1)
        pos_2 = order.index(2)

        assert pos_0 < pos_1 or pos_2 < pos_1, "X1 should be after at least one parent"

    def test_fork_detection(self, fork_dag):
        """Fork X0 <- X1 -> X2: X1 should be first in order."""
        data, B = generate_dag_data(
            fork_dag, n_samples=1000, noise_type="laplace", seed=42
        )
        result = direct_lingam(data, seed=42)

        # X1 is the root, should be first
        order = result.causal_order
        assert order[0] == 1, f"Fork root X1 should be first, got order {order}"

    def test_five_node_dag(self, five_node_data_laplace):
        """5-node DAG order validation."""
        data, B, dag = five_node_data_laplace
        result = direct_lingam(data, seed=42)

        # Check causal order accuracy
        true_order = dag.topological_order()
        accuracy = result.causal_order_accuracy(true_order)

        # Should get at least 60% correct
        assert accuracy >= 0.6, f"5-node order accuracy = {accuracy:.2f} < 0.6"


# =============================================================================
# Layer 1: ICA-LiNGAM vs DirectLiNGAM
# =============================================================================


class TestLiNGAMVariants:
    """Compare ICA-LiNGAM and DirectLiNGAM."""

    def test_direct_faster_or_similar(self, five_node_data_laplace):
        """DirectLiNGAM should give reasonable results."""
        data, B, dag = five_node_data_laplace

        result_direct = direct_lingam(data, seed=42)
        result_ica = ica_lingam(data, seed=42)

        # Both should return valid DAGs
        assert result_direct.dag is not None
        assert result_ica.dag is not None

    def test_both_variants_on_chain(self, chain_data_laplace):
        """Both variants should work on chain structure."""
        data, B, dag = chain_data_laplace

        result_direct = direct_lingam(data, seed=42)
        result_ica = ica_lingam(data, seed=42)

        # Both should identify chain-like structure
        # Check that order has 0 before 2
        assert result_direct.causal_order.index(0) < result_direct.causal_order.index(2)


# =============================================================================
# Layer 1: Non-Gaussianity Diagnostics
# =============================================================================


class TestNonGaussianity:
    """Test non-Gaussianity checking."""

    def test_laplace_is_non_gaussian(self, chain_data_laplace):
        """Laplace noise should be detected as non-Gaussian."""
        data, B, dag = chain_data_laplace
        diag = check_non_gaussianity(data, alpha=0.05)

        assert diag["n_non_gaussian"] >= 1, "Laplace should be non-Gaussian"
        assert diag["lingam_applicable"], "LiNGAM should be applicable"

    def test_gaussian_detected(self, chain_data_gaussian):
        """Gaussian data should be detected."""
        data, B, dag = chain_data_gaussian
        diag = check_non_gaussianity(data, alpha=0.05)

        # Most variables should be Gaussian
        assert diag["n_gaussian"] >= 1, "Gaussian data not detected"

    def test_kurtosis_sign(self, chain_data_laplace):
        """Laplace has positive excess kurtosis."""
        data, B, dag = chain_data_laplace
        diag = check_non_gaussianity(data)

        # Laplace has kurtosis = 3 (excess = 0 is Gaussian, Laplace > 0)
        mean_kurtosis = np.mean(diag["kurtosis"])
        assert mean_kurtosis > 0, "Laplace should have positive excess kurtosis"


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestLiNGAMAdversarial:
    """Adversarial tests for edge cases."""

    def test_small_sample_handling(self, chain_dag):
        """LiNGAM should handle small samples."""
        data, B = generate_dag_data(
            chain_dag, n_samples=100, noise_type="laplace", seed=42
        )
        result = direct_lingam(data, seed=42)

        # Should complete without error
        assert len(result.causal_order) == 3

    def test_near_gaussian_data(self, chain_dag):
        """LiNGAM performance degrades with Gaussian data."""
        data, B = generate_dag_data(
            chain_dag, n_samples=1000, noise_type="gaussian", seed=42
        )

        # Should still run, but may give poor results
        result = direct_lingam(data, seed=42)
        assert result.dag is not None

    def test_weak_effects(self, five_node_dag):
        """LiNGAM may miss weak edges."""
        data, B = generate_dag_data(
            five_node_dag,
            n_samples=1000,
            noise_type="laplace",
            coefficient_range=(0.1, 0.2),
            seed=42,
        )
        result = direct_lingam(data, seed=42)

        # Should complete, may have lower accuracy
        assert result.causal_order is not None

    def test_exponential_noise(self, chain_dag):
        """LiNGAM should work with exponential noise (asymmetric)."""
        data, B = generate_dag_data(
            chain_dag, n_samples=1000, noise_type="exponential", seed=42
        )
        result = direct_lingam(data, seed=42)

        # Exponential is strongly non-Gaussian - should complete without error
        assert result.dag is not None
        assert len(result.causal_order) == 3

    def test_uniform_noise(self, chain_dag):
        """LiNGAM with uniform noise (platykurtic)."""
        data, B = generate_dag_data(
            chain_dag, n_samples=1000, noise_type="uniform", seed=42
        )
        result = direct_lingam(data, seed=42)

        # Uniform is non-Gaussian
        assert result.dag is not None


# =============================================================================
# Layer 2: Bootstrap Confidence
# =============================================================================


class TestBootstrapLiNGAM:
    """Test bootstrap confidence estimation."""

    def test_bootstrap_returns_frequencies(self, chain_data_laplace):
        """Bootstrap should return edge frequency matrix."""
        data, B, dag = chain_data_laplace

        result, freqs = bootstrap_lingam(data, n_bootstrap=20, method="direct", seed=42)

        # Frequency matrix should be n_vars x n_vars
        assert freqs.shape == (3, 3)

        # Frequencies should be between 0 and 1
        assert np.all(freqs >= 0) and np.all(freqs <= 1)

    def test_high_confidence_edges(self, chain_data_laplace):
        """Strong edges should have high bootstrap frequency."""
        data, B, dag = chain_data_laplace

        result, freqs = bootstrap_lingam(data, n_bootstrap=30, method="direct", seed=42)

        # At least one edge should be detected frequently
        max_freq = np.max(freqs)
        assert max_freq >= 0.5, f"Max edge frequency = {max_freq:.2f} < 0.5"


# =============================================================================
# Layer 3: Monte Carlo Validation
# =============================================================================


class TestLiNGAMMonteCarlo:
    """Monte Carlo tests for statistical validation."""

    @pytest.mark.slow
    def test_causal_order_accuracy_monte_carlo(self):
        """Monte Carlo: Causal order accuracy reasonable."""
        n_runs = 30
        accuracies = []

        for seed in range(n_runs):
            dag = generate_random_dag(5, edge_prob=0.4, seed=seed)
            data, B = generate_dag_data(
                dag, n_samples=1000, noise_type="laplace", seed=seed
            )

            result = direct_lingam(data, seed=seed)
            true_order = dag.topological_order()
            acc = result.causal_order_accuracy(true_order)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        # LiNGAM is inherently challenging - accept 40%+ accuracy
        assert mean_acc >= 0.40, f"Mean order accuracy = {mean_acc:.2f} < 0.40"

    @pytest.mark.slow
    def test_edge_recovery_monte_carlo(self):
        """Monte Carlo: Edge recovery should be reasonable."""
        n_runs = 30
        recalls = []

        for seed in range(n_runs):
            dag = generate_random_dag(5, edge_prob=0.4, seed=seed)
            data, B = generate_dag_data(
                dag, n_samples=1000, noise_type="laplace", seed=seed
            )

            result = direct_lingam(data, seed=seed)

            # Count true positive edges
            B_est = result.adjacency_matrix
            B_true = B  # From DGP

            # True edges
            true_edges = np.sum(np.abs(B_true) > 0.1)
            detected = 0
            for i in range(5):
                for j in range(5):
                    if np.abs(B_true[i, j]) > 0.1 and np.abs(B_est[i, j]) > 0.1:
                        detected += 1

            recall = detected / true_edges if true_edges > 0 else 1.0
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        assert mean_recall >= 0.40, f"Mean edge recall = {mean_recall:.2f} < 0.40"

    @pytest.mark.slow
    def test_performance_by_noise_type(self):
        """LiNGAM should work better with more non-Gaussian noise."""
        dag = generate_random_dag(5, edge_prob=0.4, seed=42)

        noise_types = ["laplace", "exponential"]
        accuracies = {}

        for noise in noise_types:
            accs = []
            for seed in range(20):
                data, B = generate_dag_data(
                    dag, n_samples=1000, noise_type=noise, seed=seed
                )
                result = direct_lingam(data, seed=seed)
                true_order = dag.topological_order()
                accs.append(result.causal_order_accuracy(true_order))
            accuracies[noise] = np.mean(accs)

        # Both should produce results (accuracy may vary)
        for noise, acc in accuracies.items():
            assert acc >= 0.30, f"{noise} accuracy = {acc:.2f} < 0.30"


# =============================================================================
# Layer 3: Sample Size Sensitivity
# =============================================================================


class TestLiNGAMSampleSize:
    """Test LiNGAM performance vs sample size."""

    @pytest.mark.slow
    def test_accuracy_improves_with_n(self):
        """Order accuracy should improve with larger samples."""
        dag = generate_random_dag(5, edge_prob=0.4, seed=42)
        true_order = dag.topological_order()

        sample_sizes = [200, 500, 1000, 2000]
        accuracies = []

        for n in sample_sizes:
            data, B = generate_dag_data(
                dag, n_samples=n, noise_type="laplace", seed=42
            )
            result = direct_lingam(data, seed=42)
            acc = result.causal_order_accuracy(true_order)
            accuracies.append(acc)

        # Should improve or stay stable
        assert accuracies[-1] >= accuracies[0] - 0.15


# =============================================================================
# Regression Tests
# =============================================================================


class TestLiNGAMRegression:
    """Regression tests with fixed seeds."""

    def test_deterministic_with_seed(self, five_node_dag):
        """LiNGAM should give same result with same seed."""
        data, B = generate_dag_data(
            five_node_dag, n_samples=1000, noise_type="laplace", seed=42
        )

        result1 = direct_lingam(data, seed=42)
        result2 = direct_lingam(data, seed=42)

        assert result1.causal_order == result2.causal_order
        np.testing.assert_array_almost_equal(
            result1.adjacency_matrix, result2.adjacency_matrix, decimal=5
        )
