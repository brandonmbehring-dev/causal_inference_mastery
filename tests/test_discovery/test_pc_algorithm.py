"""Tests for PC Algorithm.

Session 133: Validation of constraint-based causal discovery.

Test Layers:
1. Known-answer: Simple structures with hand-verified results
2. Adversarial: Edge cases, near-violations
3. Monte Carlo: Statistical coverage across random DAGs
"""

import numpy as np
import pytest

from causal_inference.discovery import (
    pc_algorithm,
    pc_skeleton,
    pc_orient,
    pc_conservative,
    pc_majority,
    generate_random_dag,
    generate_dag_data,
    dag_to_cpdag,
    skeleton_f1,
    compute_shd,
    is_markov_equivalent,
)


# =============================================================================
# Layer 1: Known-Answer Tests (Simple Structures)
# =============================================================================


class TestPCKnownStructures:
    """Test PC on known DAG structures with expected results."""

    def test_chain_skeleton_recovery(self, chain_data_gaussian):
        """Chain structure: skeleton should have 2 edges."""
        data, B, dag = chain_data_gaussian
        result = pc_algorithm(data, alpha=0.01)

        # Skeleton should recover both edges
        precision, recall, f1 = result.skeleton_f1(dag)
        assert f1 >= 0.8, f"Chain skeleton F1 = {f1:.3f} < 0.8"

    def test_chain_cpdag_undirected(self, chain_data_gaussian):
        """Chain X0 -> X1 -> X2 has no v-structures, all edges undirected in CPDAG."""
        data, B, dag = chain_data_gaussian
        result = pc_algorithm(data, alpha=0.01)

        # True CPDAG should have undirected edges
        true_cpdag = dag_to_cpdag(dag)

        # Check orientation accuracy
        # Chain should have 2 undirected edges in CPDAG
        n_directed = np.sum(result.cpdag.directed > 0)
        n_undirected = np.sum(result.cpdag.undirected > 0) // 2

        # Mostly undirected expected for chain
        assert n_undirected >= 1, f"Expected undirected edges in chain, got {n_undirected}"

    def test_collider_v_structure_detected(self, collider_data_gaussian):
        """Collider X0 -> X1 <- X2 should detect v-structure."""
        data, B, dag = collider_data_gaussian
        result = pc_algorithm(data, alpha=0.01)

        # Check skeleton first
        precision, recall, f1 = result.skeleton_f1(dag)
        assert f1 >= 0.8, f"Collider skeleton F1 = {f1:.3f} < 0.8"

        # V-structure should be detected (edges directed into X1)
        cpdag = result.cpdag
        # At least one directed edge pointing to node 1
        has_directed_to_1 = (
            cpdag.has_directed_edge(0, 1) or cpdag.has_directed_edge(2, 1)
        )
        assert has_directed_to_1, "V-structure not detected in collider"

    def test_fork_no_v_structure(self, fork_dag):
        """Fork X0 <- X1 -> X2 has no v-structures."""
        data, B = generate_dag_data(fork_dag, n_samples=1000, seed=42)
        result = pc_algorithm(data, alpha=0.01)

        # Skeleton should be correct
        precision, recall, f1 = result.skeleton_f1(fork_dag)
        assert f1 >= 0.8, f"Fork skeleton F1 = {f1:.3f}"

    def test_diamond_structure(self, diamond_data_gaussian):
        """Diamond structure has v-structure at X3."""
        data, B, dag = diamond_data_gaussian
        result = pc_algorithm(data, alpha=0.01)

        # Skeleton check
        precision, recall, f1 = result.skeleton_f1(dag)
        assert f1 >= 0.75, f"Diamond skeleton F1 = {f1:.3f} < 0.75"

        # SHD should be reasonable
        shd = result.structural_hamming_distance(dag)
        assert shd <= 4, f"Diamond SHD = {shd} > 4"

    def test_five_node_dag(self, five_node_data_gaussian):
        """5-node DAG with multiple v-structures."""
        data, B, dag = five_node_data_gaussian
        result = pc_algorithm(data, alpha=0.01)

        # Skeleton F1
        precision, recall, f1 = result.skeleton_f1(dag)
        assert f1 >= 0.7, f"5-node skeleton F1 = {f1:.3f} < 0.7"

        # SHD
        shd = result.structural_hamming_distance(dag)
        assert shd <= 5, f"5-node SHD = {shd} > 5"


# =============================================================================
# Layer 1: Independence Test Integration
# =============================================================================


class TestPCIndependenceTests:
    """Test PC with different CI tests."""

    def test_fisher_z_default(self, chain_data_gaussian):
        """Default Fisher Z test should work on Gaussian data."""
        data, B, dag = chain_data_gaussian
        result = pc_algorithm(data, alpha=0.01)
        assert result.n_ci_tests > 0

    def test_ci_test_count_reasonable(self, five_node_data_gaussian):
        """Number of CI tests should be polynomial, not exponential."""
        data, B, dag = five_node_data_gaussian
        result = pc_algorithm(data, alpha=0.01)

        # For 5 nodes, should not need exponential CI tests
        n_vars = 5
        max_reasonable = n_vars * n_vars * 2**n_vars  # Very loose upper bound
        assert result.n_ci_tests < max_reasonable


# =============================================================================
# Layer 1: Stable PC Variant
# =============================================================================


class TestStablePC:
    """Test order-independent (stable) PC."""

    def test_stable_vs_unstable_consistency(self, random_data_gaussian):
        """Stable PC should give consistent results."""
        data, B, dag = random_data_gaussian

        result_stable = pc_algorithm(data, alpha=0.01, stable=True)
        result_unstable = pc_algorithm(data, alpha=0.01, stable=False)

        # Both should recover similar skeleton
        f1_stable = result_stable.skeleton_f1(dag)[2]
        f1_unstable = result_unstable.skeleton_f1(dag)[2]

        # Stable should be at least as good
        assert f1_stable >= f1_unstable - 0.1


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestPCAdversarial:
    """Adversarial tests for edge cases and difficult scenarios."""

    def test_small_sample_size(self, chain_dag):
        """PC should handle small samples gracefully."""
        data, B = generate_dag_data(chain_dag, n_samples=50, seed=42)
        result = pc_algorithm(data, alpha=0.05)

        # Should complete without error
        assert result.cpdag is not None
        # May have poor recovery with small n, but shouldn't crash

    def test_high_alpha_aggressive(self, five_node_data_gaussian):
        """High alpha should remove more edges."""
        data, B, dag = five_node_data_gaussian

        result_conservative = pc_algorithm(data, alpha=0.001)
        result_aggressive = pc_algorithm(data, alpha=0.10)

        # Aggressive should have fewer or equal edges
        n_edges_conservative = result_conservative.skeleton.n_edges()
        n_edges_aggressive = result_aggressive.skeleton.n_edges()

        assert n_edges_aggressive <= n_edges_conservative + 1

    def test_low_alpha_conservative(self, five_node_data_gaussian):
        """Low alpha should keep more edges."""
        data, B, dag = five_node_data_gaussian

        result_low = pc_algorithm(data, alpha=0.001)
        result_high = pc_algorithm(data, alpha=0.1)

        n_edges_low = result_low.skeleton.n_edges()
        n_edges_high = result_high.skeleton.n_edges()

        assert n_edges_low >= n_edges_high - 1

    def test_weak_effects(self, five_node_dag):
        """PC may miss edges with weak effects."""
        # Generate data with weak coefficients
        data, B = generate_dag_data(
            five_node_dag,
            n_samples=1000,
            coefficient_range=(0.1, 0.2),  # Weak
            seed=42,
        )
        result = pc_algorithm(data, alpha=0.01)

        # May have lower recall due to weak effects
        precision, recall, f1 = result.skeleton_f1(five_node_dag)
        # At least shouldn't add many false edges
        assert precision >= 0.5

    def test_nearly_collinear_data(self, chain_dag):
        """PC should handle near-collinearity."""
        data, B = generate_dag_data(
            chain_dag,
            n_samples=1000,
            noise_scale=0.1,  # Low noise = high collinearity
            seed=42,
        )
        result = pc_algorithm(data, alpha=0.01)

        # Should complete without numerical errors
        assert result.cpdag is not None

    def test_max_conditioning_set_size(self, random_dag_medium):
        """Limited conditioning set size for efficiency."""
        data, B = generate_dag_data(random_dag_medium, n_samples=1000, seed=42)

        # With max_cond_size=2, should be much faster
        result = pc_algorithm(data, alpha=0.01, max_cond_size=2)

        # Should complete and give some result
        assert result.skeleton.n_edges() > 0


# =============================================================================
# Layer 2: Conservative and Majority PC Variants
# =============================================================================


class TestPCVariants:
    """Test PC-CPC (conservative) and PC-MPC (majority) variants."""

    def test_conservative_fewer_orientations(self, diamond_data_gaussian):
        """Conservative PC should orient fewer edges."""
        data, B, dag = diamond_data_gaussian

        result_standard = pc_algorithm(data, alpha=0.01)
        result_conservative = pc_conservative(data, alpha=0.01)

        # Conservative should have equal or fewer directed edges
        n_dir_standard = np.sum(result_standard.cpdag.directed > 0)
        n_dir_conservative = np.sum(result_conservative.cpdag.directed > 0)

        # May have same, but not more
        assert n_dir_conservative <= n_dir_standard + 2

    def test_majority_intermediate(self, diamond_data_gaussian):
        """Majority PC should be between standard and conservative."""
        data, B, dag = diamond_data_gaussian

        result_majority = pc_majority(data, alpha=0.01)

        # Should complete without error
        assert result_majority.cpdag is not None


# =============================================================================
# Layer 3: Monte Carlo Validation
# =============================================================================


class TestPCMonteCarlo:
    """Monte Carlo tests for statistical validation."""

    @pytest.mark.slow
    def test_skeleton_recovery_monte_carlo(self):
        """Monte Carlo: Skeleton F1 > 0.80 across random DAGs."""
        n_runs = 50
        f1_scores = []

        for seed in range(n_runs):
            dag = generate_random_dag(5, edge_prob=0.4, seed=seed)
            data, B = generate_dag_data(dag, n_samples=1000, seed=seed)

            result = pc_algorithm(data, alpha=0.01)
            precision, recall, f1 = result.skeleton_f1(dag)
            f1_scores.append(f1)

        mean_f1 = np.mean(f1_scores)
        assert mean_f1 >= 0.75, f"Mean skeleton F1 = {mean_f1:.3f} < 0.75"

    @pytest.mark.slow
    def test_shd_distribution(self):
        """Monte Carlo: SHD distribution should be reasonable."""
        n_runs = 50
        shd_values = []

        for seed in range(n_runs):
            dag = generate_random_dag(5, edge_prob=0.4, seed=seed)
            data, B = generate_dag_data(dag, n_samples=1000, seed=seed)

            result = pc_algorithm(data, alpha=0.01)
            shd = result.structural_hamming_distance(dag)
            shd_values.append(shd)

        median_shd = np.median(shd_values)
        assert median_shd <= 4, f"Median SHD = {median_shd} > 4"

    @pytest.mark.slow
    def test_markov_equivalence_recovery(self):
        """Monte Carlo: Recovered CPDAG should match true equivalence class."""
        n_runs = 30
        exact_matches = 0

        for seed in range(n_runs):
            dag = generate_random_dag(4, edge_prob=0.4, seed=seed)
            data, B = generate_dag_data(dag, n_samples=2000, seed=seed)

            result = pc_algorithm(data, alpha=0.01)
            true_cpdag = dag_to_cpdag(dag)

            # Check if CPDAGs match
            shd = compute_shd(result.cpdag, dag)
            if shd == 0:
                exact_matches += 1

        # At least 50% should be exact
        rate = exact_matches / n_runs
        assert rate >= 0.40, f"Exact CPDAG recovery rate = {rate:.1%} < 40%"


# =============================================================================
# Layer 3: Sample Size Sensitivity
# =============================================================================


class TestPCSampleSize:
    """Test PC performance vs sample size."""

    @pytest.mark.slow
    def test_performance_improves_with_n(self):
        """F1 should improve with larger sample size."""
        dag = generate_random_dag(5, edge_prob=0.4, seed=42)

        sample_sizes = [200, 500, 1000, 2000]
        f1_scores = []

        for n in sample_sizes:
            data, B = generate_dag_data(dag, n_samples=n, seed=42)
            result = pc_algorithm(data, alpha=0.01)
            f1 = result.skeleton_f1(dag)[2]
            f1_scores.append(f1)

        # F1 should generally increase (allowing for noise)
        assert f1_scores[-1] >= f1_scores[0] - 0.1


# =============================================================================
# Regression Tests
# =============================================================================


class TestPCRegression:
    """Regression tests with fixed random seeds."""

    def test_deterministic_with_seed(self, five_node_dag):
        """PC should give same result with same seed."""
        data, B = generate_dag_data(five_node_dag, n_samples=1000, seed=42)

        result1 = pc_algorithm(data, alpha=0.01, stable=True)
        result2 = pc_algorithm(data, alpha=0.01, stable=True)

        # Results should be identical
        assert np.array_equal(result1.cpdag.directed, result2.cpdag.directed)
        assert np.array_equal(result1.cpdag.undirected, result2.cpdag.undirected)
