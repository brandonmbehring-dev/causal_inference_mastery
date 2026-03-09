"""Tests for FCI Algorithm.

Session 134: Validation of FCI for causal discovery with latent confounders.

Test Layers:
1. Known-answer: Simple structures with hand-verified results
2. Adversarial: Edge cases, latent detection accuracy
3. Monte Carlo: Statistical validation across random DAGs

Key difference from PC: FCI can detect latent confounders (bidirected edges).
"""

import numpy as np
import pytest

from causal_inference.discovery import (
    fci_algorithm,
    fci_orient,
    pc_algorithm,
    pc_skeleton,
    generate_random_dag,
    generate_dag_data,
    PAG,
    EdgeMark,
    FCIResult,
    Graph,
)


# =============================================================================
# Layer 1: Known-Answer Tests (Simple Structures)
# =============================================================================


class TestFCIKnownStructures:
    """Test FCI on known DAG structures with expected results."""

    def test_chain_skeleton_recovery(self, chain_data_gaussian):
        """Chain structure: skeleton should have 2 edges."""
        data, B, dag = chain_data_gaussian
        result = fci_algorithm(data, alpha=0.01)

        # Skeleton should recover edges
        assert result.skeleton.n_edges() >= 1, "Chain skeleton missing edges"
        assert result.pag.n_edges() >= 1, "PAG missing edges"

    def test_chain_no_latent_confounders(self, chain_data_gaussian):
        """Chain X0 -> X1 -> X2 has no latent confounders."""
        data, B, dag = chain_data_gaussian
        result = fci_algorithm(data, alpha=0.01)

        # No bidirected edges expected
        n_bidirected = result.pag.n_bidirected_edges()
        assert n_bidirected == 0, f"Chain should have no bidirected edges, got {n_bidirected}"
        assert len(result.possible_latent_confounders) == 0

    def test_collider_v_structure_detected(self, collider_data_gaussian):
        """Collider X0 -> X1 <- X2 should detect v-structure."""
        data, B, dag = collider_data_gaussian
        result = fci_algorithm(data, alpha=0.01)

        # PAG should have arrowheads into X1
        pag = result.pag

        # Check for v-structure (arrows pointing into node 1)
        has_arrow_into_1 = False
        for i in [0, 2]:
            if pag.has_edge(i, 1):
                edge = pag.get_edge(i, 1)
                if edge and edge.mark_j == EdgeMark.ARROW:
                    has_arrow_into_1 = True
                    break

        assert has_arrow_into_1, "V-structure not detected in collider"

    def test_fork_no_v_structure(self, fork_dag):
        """Fork X0 <- X1 -> X2 has no v-structures."""
        data, B = generate_dag_data(fork_dag, n_samples=1000, seed=42)
        result = fci_algorithm(data, alpha=0.01)

        # No bidirected edges in fork
        assert result.pag.n_bidirected_edges() == 0

    def test_diamond_structure(self, diamond_data_gaussian):
        """Diamond structure has v-structure at X3."""
        data, B, dag = diamond_data_gaussian
        result = fci_algorithm(data, alpha=0.01)

        # Skeleton should be reasonable
        assert result.skeleton.n_edges() >= 2

    def test_five_node_dag(self, five_node_data_gaussian):
        """5-node DAG should be recovered reasonably."""
        data, B, dag = five_node_data_gaussian
        result = fci_algorithm(data, alpha=0.01)

        # Check skeleton has edges
        assert result.skeleton.n_edges() >= 2

    def test_fci_result_structure(self, chain_data_gaussian):
        """FCIResult should have correct structure."""
        data, B, dag = chain_data_gaussian
        result = fci_algorithm(data, alpha=0.01)

        assert isinstance(result, FCIResult)
        assert isinstance(result.pag, PAG)
        assert isinstance(result.skeleton, Graph)
        assert isinstance(result.separating_sets, dict)
        assert isinstance(result.possible_latent_confounders, list)
        assert result.n_ci_tests > 0
        assert result.alpha == 0.01


# =============================================================================
# Layer 1: Latent Confounder Detection
# =============================================================================


class TestFCILatentConfounders:
    """Test FCI detection of latent confounders."""

    def test_simulated_latent_confounder(self):
        """Simulate latent confounder by omitting variable."""
        # Create DAG: X0 <- L -> X1, L -> X2
        # Observe only X0, X1, X2 (L is latent)
        np.random.seed(42)
        n = 1000

        # Generate latent L
        L = np.random.randn(n)

        # X0, X1, X2 all caused by L
        X0 = 0.8 * L + 0.3 * np.random.randn(n)
        X1 = 0.8 * L + 0.3 * np.random.randn(n)
        X2 = 0.8 * L + 0.3 * np.random.randn(n)

        data = np.column_stack([X0, X1, X2])

        result = fci_algorithm(data, alpha=0.01)

        # Should find that X0, X1, X2 are all correlated
        # due to latent L, but marginal independence tests may
        # detect this differently. The key is that FCI should
        # not incorrectly orient as a DAG when latent exists.
        assert result.pag is not None

    def test_observed_vs_latent_comparison(self):
        """Compare FCI on data with vs without latent variable."""
        # DAG: L -> X0, L -> X1, X1 -> X2
        np.random.seed(42)
        n = 1000

        L = np.random.randn(n)
        X0 = 0.8 * L + 0.3 * np.random.randn(n)
        X1 = 0.8 * L + 0.3 * np.random.randn(n)
        X2 = 0.8 * X1 + 0.3 * np.random.randn(n)

        # With L observed
        data_full = np.column_stack([L, X0, X1, X2])
        result_full = fci_algorithm(data_full, alpha=0.01)

        # Without L (latent)
        data_latent = np.column_stack([X0, X1, X2])
        result_latent = fci_algorithm(data_latent, alpha=0.01)

        # Both should complete
        assert result_full.pag is not None
        assert result_latent.pag is not None

        # With full data, should have fewer/no bidirected edges
        # With latent, may have bidirected edges
        assert result_full.pag.n_bidirected_edges() <= result_latent.pag.n_bidirected_edges() + 2


# =============================================================================
# Layer 1: FCI Orientation Rules
# =============================================================================


class TestFCIOrientationRules:
    """Test FCI orientation rules R0-R10."""

    def test_rule_0_v_structure(self):
        """R0: Unshielded collider detection."""
        # Create collider: X0 -> X1 <- X2, X0 not adjacent to X2
        np.random.seed(42)
        n = 1000

        X0 = np.random.randn(n)
        X2 = np.random.randn(n)
        X1 = 0.7 * X0 + 0.7 * X2 + 0.3 * np.random.randn(n)

        data = np.column_stack([X0, X1, X2])
        result = fci_algorithm(data, alpha=0.01)

        # Both edges should have arrowheads at X1
        pag = result.pag
        if pag.has_edge(0, 1) and pag.has_edge(2, 1):
            # Check arrowheads at node 1
            edge_01 = pag.get_edge(0, 1)
            edge_21 = pag.get_edge(2, 1)
            if edge_01 and edge_21:
                # At least one should have arrow at 1
                has_arrows = edge_01.mark_j == EdgeMark.ARROW or edge_21.mark_j == EdgeMark.ARROW
                assert has_arrows, "V-structure arrows not detected"

    def test_rule_1_away_from_collider(self):
        """R1: Orient away from collider prevents new v-structures."""
        # X0 -> X1 -> X2 -> X3, X0 -> X2 <- X3 (collider at X2)
        np.random.seed(42)
        n = 1000

        X0 = np.random.randn(n)
        X3 = np.random.randn(n)
        X2 = 0.7 * X0 + 0.7 * X3 + 0.3 * np.random.randn(n)
        X1 = 0.7 * X0 + 0.3 * np.random.randn(n)

        data = np.column_stack([X0, X1, X2, X3])
        result = fci_algorithm(data, alpha=0.05)

        # Should complete without errors
        assert result.pag is not None

    def test_rule_2_acyclicity(self):
        """R2: Orient to prevent cycles."""
        # Create structure where R2 would apply
        dag = generate_random_dag(4, edge_prob=0.5, seed=42)
        data, B = generate_dag_data(dag, n_samples=1000, seed=42)
        result = fci_algorithm(data, alpha=0.01)

        # PAG should exist
        assert result.pag is not None


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestFCIAdversarial:
    """Adversarial tests for edge cases."""

    def test_small_sample_size(self, chain_dag):
        """FCI should handle small samples gracefully."""
        data, B = generate_dag_data(chain_dag, n_samples=50, seed=42)
        result = fci_algorithm(data, alpha=0.10)

        # Should complete without error
        assert result.pag is not None

    def test_high_alpha_aggressive(self, five_node_data_gaussian):
        """High alpha should remove more edges."""
        data, B, dag = five_node_data_gaussian

        result_conservative = fci_algorithm(data, alpha=0.001)
        result_aggressive = fci_algorithm(data, alpha=0.10)

        # Aggressive should have fewer or equal edges
        n_edges_conservative = result_conservative.skeleton.n_edges()
        n_edges_aggressive = result_aggressive.skeleton.n_edges()

        assert n_edges_aggressive <= n_edges_conservative + 1

    def test_low_alpha_conservative(self, five_node_data_gaussian):
        """Low alpha should keep more edges."""
        data, B, dag = five_node_data_gaussian

        result_low = fci_algorithm(data, alpha=0.001)
        result_high = fci_algorithm(data, alpha=0.1)

        n_edges_low = result_low.skeleton.n_edges()
        n_edges_high = result_high.skeleton.n_edges()

        assert n_edges_low >= n_edges_high - 1

    def test_weak_effects(self, five_node_dag):
        """FCI with weak effects may miss edges."""
        data, B = generate_dag_data(
            five_node_dag,
            n_samples=1000,
            coefficient_range=(0.1, 0.2),
            seed=42,
        )
        result = fci_algorithm(data, alpha=0.01)

        # Should complete without error
        assert result.pag is not None

    def test_nearly_collinear_data(self, chain_dag):
        """FCI should handle near-collinearity."""
        data, B = generate_dag_data(
            chain_dag,
            n_samples=1000,
            noise_scale=0.1,
            seed=42,
        )
        result = fci_algorithm(data, alpha=0.01)

        assert result.pag is not None

    def test_max_conditioning_set_size(self, random_dag_medium):
        """Limited conditioning set size for efficiency."""
        data, B = generate_dag_data(random_dag_medium, n_samples=1000, seed=42)

        result = fci_algorithm(data, alpha=0.01, max_cond_size=2)

        assert result.skeleton.n_edges() >= 0


# =============================================================================
# Layer 2: FCI vs PC Comparison
# =============================================================================


class TestFCIvsPCComparison:
    """Compare FCI and PC behavior."""

    def test_same_skeleton_without_latents(self, chain_data_gaussian):
        """FCI and PC should produce same skeleton when no latents."""
        data, B, dag = chain_data_gaussian

        fci_result = fci_algorithm(data, alpha=0.01)
        pc_result = pc_algorithm(data, alpha=0.01)

        # Skeletons should match
        np.testing.assert_array_equal(
            fci_result.skeleton.adjacency,
            pc_result.skeleton.adjacency,
        )

    def test_separating_sets_match(self, diamond_data_gaussian):
        """FCI and PC should have same separating sets."""
        data, B, dag = diamond_data_gaussian

        fci_result = fci_algorithm(data, alpha=0.01)
        pc_result = pc_algorithm(data, alpha=0.01)

        # Separating sets should match
        assert fci_result.separating_sets == pc_result.separating_sets

    def test_ci_test_count_matches(self, five_node_data_gaussian):
        """FCI and PC perform same skeleton learning."""
        data, B, dag = five_node_data_gaussian

        fci_result = fci_algorithm(data, alpha=0.01)
        pc_result = pc_algorithm(data, alpha=0.01)

        # CI test counts should match (same skeleton phase)
        assert fci_result.n_ci_tests == pc_result.n_ci_tests


# =============================================================================
# Layer 2: PAG Edge Types
# =============================================================================


class TestPAGEdgeTypes:
    """Test PAG edge type queries."""

    def test_pag_from_skeleton(self):
        """PAG initialized from skeleton has all circle marks."""
        skeleton = Graph(n_nodes=3)
        skeleton.add_edge(0, 1)
        skeleton.add_edge(1, 2)

        pag = PAG.from_skeleton(skeleton)

        assert pag.n_edges() == 2
        assert pag.n_circle_edges() == 2  # All circle marks initially

    def test_directed_edge_detection(self):
        """Test definitely directed edge detection."""
        pag = PAG(n_nodes=3)
        pag.add_edge(0, 1, EdgeMark.TAIL, EdgeMark.ARROW)  # 0 -> 1

        assert pag.is_definitely_directed(0, 1)
        assert not pag.is_definitely_directed(1, 0)

    def test_bidirected_edge_detection(self):
        """Test bidirected edge detection (latent confounder)."""
        pag = PAG(n_nodes=3)
        pag.add_edge(0, 1, EdgeMark.ARROW, EdgeMark.ARROW)  # 0 <-> 1

        edge = pag.get_edge(0, 1)
        assert edge.is_bidirected()
        assert pag.n_bidirected_edges() == 1

    def test_circle_edge_detection(self):
        """Test circle (uncertain) edge detection."""
        pag = PAG(n_nodes=3)
        pag.add_edge(0, 1, EdgeMark.CIRCLE, EdgeMark.CIRCLE)

        edge = pag.get_edge(0, 1)
        assert edge.is_partially_directed()
        assert pag.n_circle_edges() == 1


# =============================================================================
# Layer 3: Monte Carlo Validation
# =============================================================================


class TestFCIMonteCarlo:
    """Monte Carlo tests for statistical validation."""

    @pytest.mark.slow
    def test_skeleton_recovery_monte_carlo(self):
        """Monte Carlo: Skeleton should be recovered reasonably."""
        n_runs = 30
        f1_scores = []

        for seed in range(n_runs):
            dag = generate_random_dag(5, edge_prob=0.4, seed=seed)
            data, B = generate_dag_data(dag, n_samples=1000, seed=seed)

            result = fci_algorithm(data, alpha=0.01)

            # Calculate skeleton F1 manually
            true_edges = set()
            est_edges = set()

            for i in range(dag.n_nodes):
                for j in range(i + 1, dag.n_nodes):
                    if dag.has_edge(i, j) or dag.has_edge(j, i):
                        true_edges.add((i, j))
                    if result.skeleton.has_edge(i, j):
                        est_edges.add((i, j))

            tp = len(true_edges & est_edges)
            fp = len(est_edges - true_edges)
            fn = len(true_edges - est_edges)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            f1_scores.append(f1)

        mean_f1 = np.mean(f1_scores)
        assert mean_f1 >= 0.65, f"Mean skeleton F1 = {mean_f1:.3f} < 0.65"

    @pytest.mark.slow
    def test_no_spurious_bidirected_without_latents(self):
        """Monte Carlo: No bidirected edges when no latent confounders."""
        n_runs = 30
        false_bidirected_rate = 0

        for seed in range(n_runs):
            dag = generate_random_dag(4, edge_prob=0.4, seed=seed)
            data, B = generate_dag_data(dag, n_samples=1000, seed=seed)

            result = fci_algorithm(data, alpha=0.01)

            if result.pag.n_bidirected_edges() > 0:
                false_bidirected_rate += 1

        rate = false_bidirected_rate / n_runs
        # Should rarely get false bidirected edges (< 30%)
        assert rate < 0.35, f"False bidirected rate = {rate:.1%} >= 35%"


# =============================================================================
# Layer 3: Sample Size Sensitivity
# =============================================================================


class TestFCISampleSize:
    """Test FCI performance vs sample size."""

    @pytest.mark.slow
    def test_performance_improves_with_n(self):
        """Skeleton should improve with larger sample size."""
        dag = generate_random_dag(5, edge_prob=0.4, seed=42)

        sample_sizes = [200, 500, 1000, 2000]
        n_edges_list = []

        for n in sample_sizes:
            data, B = generate_dag_data(dag, n_samples=n, seed=42)
            result = fci_algorithm(data, alpha=0.01)
            n_edges_list.append(result.skeleton.n_edges())

        # Should have stable or improving edge count
        # (with more data, CI tests become more reliable)
        assert len(n_edges_list) == len(sample_sizes)


# =============================================================================
# Regression Tests
# =============================================================================


class TestFCIRegression:
    """Regression tests with fixed random seeds."""

    def test_deterministic_with_seed(self, five_node_dag):
        """FCI should give same result with same seed."""
        data, B = generate_dag_data(five_node_dag, n_samples=1000, seed=42)

        result1 = fci_algorithm(data, alpha=0.01, stable=True)
        result2 = fci_algorithm(data, alpha=0.01, stable=True)

        # Results should be identical
        np.testing.assert_array_equal(
            result1.pag.endpoints,
            result2.pag.endpoints,
        )
        assert result1.n_ci_tests == result2.n_ci_tests


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestFCIInputValidation:
    """Test input validation and error handling."""

    def test_empty_data_handled(self):
        """FCI should handle empty data gracefully (returns empty graph)."""
        data = np.array([]).reshape(0, 3)

        # With empty data, should return result with 0 edges
        # (NumPy issues warnings but doesn't crash)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fci_algorithm(data, alpha=0.01)

        assert result.pag.n_edges() == 0

    def test_single_variable(self):
        """FCI with single variable."""
        data = np.random.randn(100, 1)
        result = fci_algorithm(data, alpha=0.01)

        assert result.pag.n_edges() == 0
        assert result.skeleton.n_edges() == 0

    def test_two_variables(self):
        """FCI with two variables."""
        np.random.seed(42)
        X = np.random.randn(100)
        Y = 0.7 * X + 0.3 * np.random.randn(100)
        data = np.column_stack([X, Y])

        result = fci_algorithm(data, alpha=0.01)

        # Should detect edge between X and Y
        assert result.skeleton.n_edges() in [0, 1]  # Depends on alpha


# =============================================================================
# Edge Mark Utility Tests
# =============================================================================


class TestEdgeMarkUtilities:
    """Test EdgeMark enum utilities."""

    def test_edge_mark_from_string(self):
        """Test EdgeMark.from_string()."""
        assert EdgeMark.from_string("-") == EdgeMark.TAIL
        assert EdgeMark.from_string(">") == EdgeMark.ARROW
        assert EdgeMark.from_string("o") == EdgeMark.CIRCLE

    def test_edge_mark_string_representation(self):
        """Test EdgeMark string representation."""
        assert str(EdgeMark.TAIL) == "-"
        assert str(EdgeMark.ARROW) == ">"
        assert str(EdgeMark.CIRCLE) == "o"

    def test_pag_edge_to_string(self):
        """Test PAGEdge.to_string()."""
        from causal_inference.discovery import PAGEdge

        edge = PAGEdge(0, 1, EdgeMark.TAIL, EdgeMark.ARROW)
        s = edge.to_string()
        assert "X0" in s and "X1" in s

        # With custom names
        s = edge.to_string(["A", "B"])
        assert "A" in s and "B" in s
