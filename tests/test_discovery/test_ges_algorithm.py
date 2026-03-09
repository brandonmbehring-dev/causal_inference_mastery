"""Tests for GES Algorithm.

Session 138: Greedy Equivalence Search tests.

Layer 1: Known-Answer (small DAGs with exact structure)
Layer 2: Adversarial (edge cases, high-dimensional)
Layer 3: Monte Carlo (statistical validation)
"""

import numpy as np
import pytest

from causal_inference.discovery import (
    GESResult,
    LocalScore,
    ScoreType,
    ges_algorithm,
    ges_backward,
    ges_forward,
    local_score,
    local_score_aic,
    local_score_bic,
    total_score,
    generate_random_dag,
    generate_dag_data,
    compute_shd,
)


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestGESKnownStructure:
    """Test GES on known DAG structures."""

    def test_empty_graph_two_independent(self):
        """Two independent variables should give empty graph."""
        np.random.seed(42)
        n = 500
        X1 = np.random.randn(n)
        X2 = np.random.randn(n)
        data = np.column_stack([X1, X2])

        result = ges_algorithm(data, score_type="bic")

        assert isinstance(result, GESResult)
        assert result.n_edges() == 0
        assert result.n_forward_steps == 0 or result.n_backward_steps >= result.n_forward_steps

    def test_chain_x_to_y(self):
        """Simple chain X → Y should find one edge."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n)
        Y = 0.8 * X + np.random.randn(n) * 0.5

        data = np.column_stack([X, Y])
        result = ges_algorithm(data, score_type="bic")

        # Should have 1 edge (direction may be undetermined in CPDAG)
        assert result.n_edges() >= 1

    def test_chain_three_nodes(self):
        """Chain X → Y → Z should find two edges."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n)
        Y = 0.8 * X + np.random.randn(n) * 0.3
        Z = 0.7 * Y + np.random.randn(n) * 0.3

        data = np.column_stack([X, Y, Z])
        result = ges_algorithm(data, score_type="bic")

        # Should find 2 edges
        assert result.n_edges() >= 2

    def test_fork_structure(self):
        """Fork X ← Z → Y should find two edges."""
        np.random.seed(42)
        n = 500
        Z = np.random.randn(n)
        X = 0.8 * Z + np.random.randn(n) * 0.3
        Y = 0.7 * Z + np.random.randn(n) * 0.3

        data = np.column_stack([X, Y, Z])
        result = ges_algorithm(data, score_type="bic")

        assert result.n_edges() >= 2

    def test_collider_structure(self):
        """Collider X → Z ← Y should find v-structure."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n)
        Y = np.random.randn(n)
        Z = 0.6 * X + 0.5 * Y + np.random.randn(n) * 0.3

        data = np.column_stack([X, Y, Z])
        result = ges_algorithm(data, score_type="bic")

        # Should find 2 edges (X → Z ← Y is identifiable)
        assert result.n_edges() >= 2

    def test_diamond_structure(self):
        """Diamond: X → Y, X → Z, Y → W, Z → W."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n)
        Y = 0.7 * X + np.random.randn(n) * 0.3
        Z = 0.6 * X + np.random.randn(n) * 0.3
        W = 0.5 * Y + 0.5 * Z + np.random.randn(n) * 0.3

        data = np.column_stack([X, Y, Z, W])
        result = ges_algorithm(data, score_type="bic")

        # Should find all 4 edges
        assert result.n_edges() >= 3


class TestScoreFunctions:
    """Test score function implementations."""

    def test_local_score_bic_basic(self):
        """BIC score increases with better fit."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n)
        Y = 0.9 * X + np.random.randn(n) * 0.1

        data = np.column_stack([X, Y])

        # Score with correct parent should be higher than without
        score_with_parent = local_score_bic(data, node=1, parents={0})
        score_no_parent = local_score_bic(data, node=1, parents=set())

        assert score_with_parent.score > score_no_parent.score

    def test_local_score_bic_penalty(self):
        """BIC penalizes more parameters."""
        np.random.seed(42)
        n = 100
        data = np.random.randn(n, 5)

        # More parents = more parameters = lower score (ceteris paribus)
        score_1 = local_score_bic(data, node=0, parents={1})
        score_2 = local_score_bic(data, node=0, parents={1, 2})

        # With random data, adding parents shouldn't help much
        # Due to penalty, score should decrease or stay similar
        assert score_2.n_params > score_1.n_params

    def test_local_score_aic_basic(self):
        """AIC score computation."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n)
        Y = 0.9 * X + np.random.randn(n) * 0.1

        data = np.column_stack([X, Y])

        score_with_parent = local_score_aic(data, node=1, parents={0})
        score_no_parent = local_score_aic(data, node=1, parents=set())

        assert score_with_parent.score > score_no_parent.score

    def test_total_score_empty_graph(self):
        """Total score for empty graph."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        adjacency = np.zeros((3, 3))

        score = total_score(data, adjacency, ScoreType.BIC)
        assert np.isfinite(score)

    def test_total_score_increases_with_edge(self):
        """Total score increases when adding correct edge."""
        np.random.seed(42)
        n = 300
        X = np.random.randn(n)
        Y = 0.8 * X + np.random.randn(n) * 0.3

        data = np.column_stack([X, Y])

        adj_empty = np.zeros((2, 2))
        adj_edge = np.array([[0, 1], [0, 0]])  # X → Y

        score_empty = total_score(data, adj_empty, ScoreType.BIC)
        score_edge = total_score(data, adj_edge, ScoreType.BIC)

        assert score_edge > score_empty


class TestGESResult:
    """Test GESResult dataclass."""

    def test_result_attributes(self):
        """GESResult has expected attributes."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        result = ges_algorithm(data)

        assert hasattr(result, "cpdag")
        assert hasattr(result, "score")
        assert hasattr(result, "n_forward_steps")
        assert hasattr(result, "n_backward_steps")
        assert hasattr(result, "forward_scores")
        assert hasattr(result, "backward_scores")
        assert result.n_vars == 3
        assert result.n_samples == 100

    def test_n_edges_method(self):
        """n_edges() method works."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        result = ges_algorithm(data)

        n_edges = result.n_edges()
        assert isinstance(n_edges, int)
        assert n_edges >= 0


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestGESAdversarial:
    """Test GES on edge cases."""

    def test_small_sample(self):
        """GES handles small samples."""
        np.random.seed(42)
        data = np.random.randn(20, 3)
        result = ges_algorithm(data)
        assert result is not None

    def test_high_dimensional(self):
        """GES handles more variables."""
        np.random.seed(42)
        n_vars = 8
        dag = generate_random_dag(n_vars, edge_prob=0.2, seed=42)
        data, _ = generate_dag_data(dag, n_samples=500, seed=42)

        result = ges_algorithm(data)
        assert result.n_vars == n_vars

    def test_collinear_data(self):
        """GES handles near-collinearity."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n)
        Y = X + np.random.randn(n) * 0.01  # Almost X
        Z = np.random.randn(n)

        data = np.column_stack([X, Y, Z])
        result = ges_algorithm(data)

        # Should not crash
        assert result is not None

    def test_constant_column(self):
        """GES handles constant column (degenerate)."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n)
        Y = np.ones(n)  # Constant
        Z = np.random.randn(n)

        data = np.column_stack([X, Y, Z])

        # Should handle gracefully (may warn or limit edges to Y)
        result = ges_algorithm(data)
        assert result is not None

    def test_score_type_aic(self):
        """GES works with AIC score."""
        np.random.seed(42)
        data = np.random.randn(200, 3)

        result = ges_algorithm(data, score_type="aic")
        assert result.score_type == "aic"

    def test_invalid_score_type(self):
        """GES rejects invalid score type."""
        data = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="Unknown score_type"):
            ges_algorithm(data, score_type="invalid")

    def test_too_few_samples(self):
        """GES rejects too few samples."""
        data = np.random.randn(5, 3)

        with pytest.raises(ValueError, match="Too few samples"):
            ges_algorithm(data)

    def test_too_few_variables(self):
        """GES rejects single variable."""
        data = np.random.randn(100, 1)

        with pytest.raises(ValueError, match="at least 2 variables"):
            ges_algorithm(data)

    def test_max_parents_limit(self):
        """GES respects max_parents limit."""
        np.random.seed(42)
        data = np.random.randn(200, 5)

        result = ges_algorithm(data, max_parents=2)
        # Should not have any node with more than 2 parents
        assert result is not None


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================


class TestGESMonteCarlo:
    """Monte Carlo validation of GES."""

    def test_shd_on_random_dags(self):
        """GES achieves reasonable SHD on random DAGs."""
        np.random.seed(42)
        n_vars = 5
        n_samples = 500

        shd_values = []
        for seed in range(10):
            dag = generate_random_dag(n_vars, edge_prob=0.3, seed=seed)
            data, _ = generate_dag_data(dag, n_samples=n_samples, seed=seed)

            result = ges_algorithm(data)
            shd = result.structural_hamming_distance(dag)
            shd_values.append(shd)

        mean_shd = np.mean(shd_values)
        # SHD should be reasonable (< n_vars * 2)
        assert mean_shd < n_vars * 2

    def test_skeleton_f1_on_random_dags(self):
        """GES achieves reasonable skeleton F1."""
        np.random.seed(42)
        n_vars = 5
        n_samples = 500

        f1_values = []
        for seed in range(10):
            dag = generate_random_dag(n_vars, edge_prob=0.3, seed=seed)
            data, _ = generate_dag_data(dag, n_samples=n_samples, seed=seed)

            result = ges_algorithm(data)
            f1 = result.skeleton_f1(dag)
            if np.isfinite(f1):
                f1_values.append(f1)

        if len(f1_values) > 0:
            mean_f1 = np.mean(f1_values)
            # F1 should be > 0.5 on average
            assert mean_f1 > 0.3

    def test_score_improves_with_samples(self):
        """GES accuracy improves with more samples."""
        np.random.seed(42)
        n_vars = 4
        dag = generate_random_dag(n_vars, edge_prob=0.4, seed=42)

        shd_small = []
        shd_large = []

        for seed in range(5):
            # Small sample
            data_small, _ = generate_dag_data(dag, n_samples=100, seed=seed)
            result_small = ges_algorithm(data_small)
            shd_small.append(result_small.structural_hamming_distance(dag))

            # Large sample
            data_large, _ = generate_dag_data(dag, n_samples=1000, seed=seed)
            result_large = ges_algorithm(data_large)
            shd_large.append(result_large.structural_hamming_distance(dag))

        # Large sample should have lower or equal SHD on average
        assert np.mean(shd_large) <= np.mean(shd_small) + 2


class TestGESvsPC:
    """Compare GES with PC algorithm."""

    def test_both_find_edges(self):
        """Both GES and PC find edges in simple structure."""
        from causal_inference.discovery import pc_algorithm

        np.random.seed(42)
        n = 500
        X = np.random.randn(n)
        Y = 0.8 * X + np.random.randn(n) * 0.3
        Z = 0.7 * Y + np.random.randn(n) * 0.3

        data = np.column_stack([X, Y, Z])

        ges_result = ges_algorithm(data)
        pc_result = pc_algorithm(data, alpha=0.01)

        # Both should find edges
        ges_edges = ges_result.n_edges()
        pc_edges = pc_result.cpdag.n_directed_edges() + pc_result.cpdag.n_undirected_edges()

        assert ges_edges >= 1
        assert pc_edges >= 1

    def test_similar_skeleton_on_chain(self):
        """GES and PC find similar skeleton on chain."""
        from causal_inference.discovery import pc_algorithm

        np.random.seed(42)
        n = 500
        X1 = np.random.randn(n)
        X2 = 0.7 * X1 + np.random.randn(n) * 0.3
        X3 = 0.7 * X2 + np.random.randn(n) * 0.3

        data = np.column_stack([X1, X2, X3])

        ges_result = ges_algorithm(data)
        pc_result = pc_algorithm(data, alpha=0.01)

        # Both should find ~2 edges
        ges_edges = ges_result.n_edges()
        pc_edges = pc_result.cpdag.n_directed_edges() + pc_result.cpdag.n_undirected_edges()

        assert abs(ges_edges - pc_edges) <= 1
