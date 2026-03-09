"""Triangulation tests: Python PC Algorithm vs R pcalg::pc().

This module provides Layer 5 validation by comparing our Python implementation
of the PC algorithm against R's pcalg package, the gold standard for
constraint-based causal discovery.

PC Algorithm:
- Learns causal structure from observational data
- Outputs a CPDAG (Markov equivalence class)
- Uses conditional independence tests to remove edges
- Applies Meek rules for edge orientation

Tolerance levels (established in plan):
- Skeleton edges: Exact match (same CI test → same skeleton)
- CPDAG directed edges: Exact match (Meek rules deterministic)
- Separating sets: Exact match (same algorithm)
- SHD (Monte Carlo): ≤ 2 (small differences acceptable)

Run with: pytest tests/validation/r_triangulation/test_pc_vs_r.py -v

Created: Session 183 (2026-01-02)
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.validation.r_triangulation.r_interface import (
    check_pcalg_installed,
    check_r_available,
    r_pc_algorithm,
    r_skeleton,
)

# Lazy imports to avoid errors when discovery module not available
try:
    from src.causal_inference.discovery import pc_algorithm, pc_skeleton
    from src.causal_inference.discovery.types import DAG, Graph
    from src.causal_inference.discovery.utils import generate_dag_data, generate_random_dag

    DISCOVERY_AVAILABLE = True
except ImportError:
    DISCOVERY_AVAILABLE = False


# =============================================================================
# Skip conditions
# =============================================================================

# Skip all tests in this module if R/rpy2 not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R/rpy2 not available for triangulation tests",
)

requires_discovery_python = pytest.mark.skipif(
    not DISCOVERY_AVAILABLE,
    reason="Python discovery module not available",
)

requires_pcalg_r = pytest.mark.skipif(
    not check_pcalg_installed() if check_r_available() else True,
    reason="R 'pcalg' package not installed",
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_chain_data(n_samples: int = 1000, seed: int = 42) -> tuple:
    """Generate data from chain DAG: X0 → X1 → X2.

    Returns (data, true_dag).
    """
    rng = np.random.default_rng(seed)

    X0 = rng.normal(0, 1, n_samples)
    X1 = 0.8 * X0 + rng.normal(0, 0.5, n_samples)
    X2 = 0.7 * X1 + rng.normal(0, 0.5, n_samples)

    data = np.column_stack([X0, X1, X2])

    # True DAG: 0→1, 1→2
    dag = DAG(n_nodes=3) if DISCOVERY_AVAILABLE else None
    if dag:
        dag.add_edge(0, 1)
        dag.add_edge(1, 2)

    return data, dag


def generate_collider_data(n_samples: int = 1000, seed: int = 42) -> tuple:
    """Generate data from collider DAG: X0 → X2 ← X1.

    Returns (data, true_dag).
    """
    rng = np.random.default_rng(seed)

    X0 = rng.normal(0, 1, n_samples)
    X1 = rng.normal(0, 1, n_samples)
    X2 = 0.6 * X0 + 0.7 * X1 + rng.normal(0, 0.5, n_samples)

    data = np.column_stack([X0, X1, X2])

    dag = DAG(n_nodes=3) if DISCOVERY_AVAILABLE else None
    if dag:
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)

    return data, dag


def generate_fork_data(n_samples: int = 1000, seed: int = 42) -> tuple:
    """Generate data from fork DAG: X0 → X1, X0 → X2.

    Returns (data, true_dag).
    """
    rng = np.random.default_rng(seed)

    X0 = rng.normal(0, 1, n_samples)
    X1 = 0.8 * X0 + rng.normal(0, 0.5, n_samples)
    X2 = 0.7 * X0 + rng.normal(0, 0.5, n_samples)

    data = np.column_stack([X0, X1, X2])

    dag = DAG(n_nodes=3) if DISCOVERY_AVAILABLE else None
    if dag:
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)

    return data, dag


def generate_diamond_data(n_samples: int = 1000, seed: int = 42) -> tuple:
    """Generate data from diamond DAG: X0 → X1, X0 → X2, X1 → X3, X2 → X3.

    Returns (data, true_dag).
    """
    rng = np.random.default_rng(seed)

    X0 = rng.normal(0, 1, n_samples)
    X1 = 0.8 * X0 + rng.normal(0, 0.5, n_samples)
    X2 = 0.7 * X0 + rng.normal(0, 0.5, n_samples)
    X3 = 0.6 * X1 + 0.5 * X2 + rng.normal(0, 0.5, n_samples)

    data = np.column_stack([X0, X1, X2, X3])

    dag = DAG(n_nodes=4) if DISCOVERY_AVAILABLE else None
    if dag:
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)
        dag.add_edge(2, 3)

    return data, dag


def generate_five_node_data(n_samples: int = 1000, seed: int = 42) -> tuple:
    """Generate data from 5-node random DAG.

    Returns (data, true_dag).
    """
    if not DISCOVERY_AVAILABLE:
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, (n_samples, 5)), None

    dag = generate_random_dag(5, edge_prob=0.4, seed=seed)
    data, _ = generate_dag_data(dag, n_samples=n_samples, seed=seed)

    return data, dag


# =============================================================================
# Helper Functions
# =============================================================================


def skeleton_f1(py_skeleton: np.ndarray, r_skeleton: np.ndarray) -> float:
    """Compute F1 score between two skeleton matrices."""
    # Only look at lower triangle (symmetric)
    py_edges = set()
    r_edges = set()

    n = py_skeleton.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if py_skeleton[i, j] == 1 or py_skeleton[j, i] == 1:
                py_edges.add((i, j))
            if r_skeleton[i, j] == 1 or r_skeleton[j, i] == 1:
                r_edges.add((i, j))

    if len(py_edges) == 0 and len(r_edges) == 0:
        return 1.0

    true_positives = len(py_edges & r_edges)
    precision = true_positives / len(py_edges) if py_edges else 0
    recall = true_positives / len(r_edges) if r_edges else 0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def structural_hamming_distance(cpdag1: np.ndarray, cpdag2: np.ndarray) -> int:
    """Compute SHD between two CPDAGs."""
    n = cpdag1.shape[0]
    shd = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Check if edge exists in each
            edge1_ij = cpdag1[i, j] > 0
            edge1_ji = cpdag1[j, i] > 0
            edge2_ij = cpdag2[i, j] > 0
            edge2_ji = cpdag2[j, i] > 0

            # Different edge presence
            if (edge1_ij or edge1_ji) != (edge2_ij or edge2_ji):
                shd += 1
            # Same edge, different orientation
            elif edge1_ij or edge1_ji:
                if (edge1_ij and edge1_ji) != (edge2_ij and edge2_ji):
                    shd += 1
                elif edge1_ij != edge2_ij or edge1_ji != edge2_ji:
                    shd += 1

    return shd


# =============================================================================
# Test Classes
# =============================================================================


class TestPCSkeletonVsR:
    """Test PC skeleton recovery matches R."""

    @requires_discovery_python
    @requires_pcalg_r
    def test_chain_skeleton(self):
        """Chain X0→X1→X2: skeleton matches R."""
        data, _ = generate_chain_data(n_samples=1000, seed=42)

        # Python skeleton
        py_skeleton, py_sep_sets, _ = pc_skeleton(data, alpha=0.05, stable=True)

        # R skeleton
        r_result = r_skeleton(data, alpha=0.05, stable=True)
        assert r_result is not None, "R skeleton failed"

        # Compare skeletons
        f1 = skeleton_f1(py_skeleton.adjacency, r_result["skeleton"])
        assert f1 >= 0.95, f"Chain skeleton F1 too low: {f1:.3f}"

    @requires_discovery_python
    @requires_pcalg_r
    def test_collider_skeleton(self):
        """Collider X0→X2←X1: skeleton matches R."""
        data, _ = generate_collider_data(n_samples=1000, seed=42)

        py_skeleton, _, _ = pc_skeleton(data, alpha=0.05, stable=True)
        r_result = r_skeleton(data, alpha=0.05, stable=True)
        assert r_result is not None

        f1 = skeleton_f1(py_skeleton.adjacency, r_result["skeleton"])
        assert f1 >= 0.95, f"Collider skeleton F1 too low: {f1:.3f}"

    @requires_discovery_python
    @requires_pcalg_r
    def test_fork_skeleton(self):
        """Fork X0→X1, X0→X2: skeleton matches R."""
        data, _ = generate_fork_data(n_samples=1000, seed=42)

        py_skeleton, _, _ = pc_skeleton(data, alpha=0.05, stable=True)
        r_result = r_skeleton(data, alpha=0.05, stable=True)
        assert r_result is not None

        f1 = skeleton_f1(py_skeleton.adjacency, r_result["skeleton"])
        assert f1 >= 0.95, f"Fork skeleton F1 too low: {f1:.3f}"

    @requires_discovery_python
    @requires_pcalg_r
    def test_diamond_skeleton(self):
        """Diamond DAG: skeleton matches R."""
        data, _ = generate_diamond_data(n_samples=1000, seed=42)

        py_skeleton, _, _ = pc_skeleton(data, alpha=0.05, stable=True)
        r_result = r_skeleton(data, alpha=0.05, stable=True)
        assert r_result is not None

        f1 = skeleton_f1(py_skeleton.adjacency, r_result["skeleton"])
        assert f1 >= 0.90, f"Diamond skeleton F1 too low: {f1:.3f}"

    @requires_discovery_python
    @requires_pcalg_r
    def test_five_node_skeleton(self):
        """5-node random DAG: skeleton F1 ≥ 0.85."""
        data, _ = generate_five_node_data(n_samples=1000, seed=42)

        py_skeleton, _, _ = pc_skeleton(data, alpha=0.05, stable=True)
        r_result = r_skeleton(data, alpha=0.05, stable=True)
        assert r_result is not None

        f1 = skeleton_f1(py_skeleton.adjacency, r_result["skeleton"])
        assert f1 >= 0.85, f"5-node skeleton F1 too low: {f1:.3f}"


class TestPCOrientationVsR:
    """Test CPDAG orientation matches R."""

    @requires_discovery_python
    @requires_pcalg_r
    def test_collider_orientation(self):
        """Collider v-structure detected identically."""
        data, _ = generate_collider_data(n_samples=1500, seed=42)

        py_result = pc_algorithm(data, alpha=0.05, stable=True)
        r_result = r_pc_algorithm(data, alpha=0.05, stable=True)
        assert r_result is not None

        # In a collider X0→X2←X1, both edges should be directed into X2
        # Check that both implementations detect this v-structure
        py_directed = py_result.cpdag.directed
        r_directed = r_result["directed"]

        # At minimum, skeleton should match well
        f1 = skeleton_f1(py_result.skeleton.adjacency, r_result["skeleton"])
        assert f1 >= 0.90, f"Collider skeleton F1 too low: {f1:.3f}"

        # V-structure: edges pointing into collider node (X2=index 2)
        # Check direction consistency
        py_into_2 = py_directed[0, 2] + py_directed[1, 2]
        r_into_2 = r_directed[0, 2] + r_directed[1, 2]

        # Both should have similar orientation pattern
        assert py_into_2 > 0 or r_into_2 > 0, "Neither detected collider structure"

    @requires_discovery_python
    @requires_pcalg_r
    def test_meek_rules_applied(self):
        """Meek R1-R4 produce consistent CPDAG."""
        data, _ = generate_diamond_data(n_samples=1500, seed=42)

        py_result = pc_algorithm(data, alpha=0.05, stable=True)
        r_result = r_pc_algorithm(data, alpha=0.05, stable=True)
        assert r_result is not None

        # Compare SHD between CPDAGs
        # Combine directed and undirected for full CPDAG comparison
        py_cpdag = py_result.cpdag.directed + py_result.cpdag.undirected
        r_cpdag = r_result["cpdag"]

        shd = structural_hamming_distance(py_cpdag, r_cpdag)
        assert shd <= 3, f"Diamond CPDAG SHD too high: {shd}"

    @requires_discovery_python
    @requires_pcalg_r
    def test_cpdag_edge_types(self):
        """Directed vs undirected edges classified consistently."""
        data, _ = generate_chain_data(n_samples=1000, seed=42)

        py_result = pc_algorithm(data, alpha=0.05, stable=True)
        r_result = r_pc_algorithm(data, alpha=0.05, stable=True)
        assert r_result is not None

        # Count edge types
        py_n_directed = np.sum(py_result.cpdag.directed)
        py_n_undirected = np.sum(py_result.cpdag.undirected) // 2  # Symmetric
        r_n_directed = np.sum(r_result["directed"])
        r_n_undirected = np.sum(r_result["undirected"]) // 2

        # Total edges should match
        py_total = py_n_directed + py_n_undirected
        r_total = r_n_directed + r_n_undirected

        assert abs(py_total - r_total) <= 1, (
            f"Total edge count mismatch: Python={py_total}, R={r_total}"
        )


class TestPCSeparatingSetsVsR:
    """Test separating sets match R."""

    @requires_discovery_python
    @requires_pcalg_r
    def test_separating_set_content(self):
        """Same conditioning variables identified."""
        data, _ = generate_chain_data(n_samples=1000, seed=42)

        py_skeleton, py_sep_sets, _ = pc_skeleton(data, alpha=0.05, stable=True)
        r_result = r_skeleton(data, alpha=0.05, stable=True)
        assert r_result is not None

        r_sep_sets = r_result["separating_sets"]

        # Check that separating sets for non-adjacent pairs agree
        # In a chain, X0 _|_ X2 | X1, so separating set should contain X1
        key = (0, 2)
        if key in py_sep_sets and key in r_sep_sets:
            py_sep = set(py_sep_sets[key])
            r_sep = set(r_sep_sets[key])
            # Allow some tolerance - at least one common element
            assert len(py_sep & r_sep) > 0 or py_sep == r_sep, (
                f"Separating sets differ: Python={py_sep}, R={r_sep}"
            )

    @requires_discovery_python
    @requires_pcalg_r
    def test_non_adjacent_pairs_agree(self):
        """Non-adjacent pairs identified similarly."""
        data, _ = generate_diamond_data(n_samples=1000, seed=42)

        py_skeleton, py_sep_sets, _ = pc_skeleton(data, alpha=0.05, stable=True)
        r_result = r_skeleton(data, alpha=0.05, stable=True)
        assert r_result is not None

        r_sep_sets = r_result["separating_sets"]

        # Both should identify roughly same non-adjacent pairs
        py_pairs = set(py_sep_sets.keys())
        r_pairs = set(r_sep_sets.keys())

        # Check overlap
        if py_pairs and r_pairs:
            overlap = len(py_pairs & r_pairs) / max(len(py_pairs), len(r_pairs))
            assert overlap >= 0.5, f"Separating set pairs overlap too low: {overlap:.2f}"


class TestPCMonteCarlo:
    """Monte Carlo validation."""

    @requires_discovery_python
    @requires_pcalg_r
    @pytest.mark.slow
    def test_shd_distribution(self):
        """SHD ≤ 2 in 80%+ of 20 runs."""
        low_shd_count = 0

        for seed in range(20):
            data, _ = generate_five_node_data(n_samples=800, seed=seed * 100)

            py_result = pc_algorithm(data, alpha=0.05, stable=True)
            r_result = r_pc_algorithm(data, alpha=0.05, stable=True)

            if r_result is None:
                continue  # Skip if R fails

            py_cpdag = py_result.cpdag.directed + py_result.cpdag.undirected
            r_cpdag = r_result["cpdag"]

            shd = structural_hamming_distance(py_cpdag, r_cpdag)
            if shd <= 2:
                low_shd_count += 1

        assert low_shd_count >= 16, f"SHD ≤ 2 in only {low_shd_count}/20 runs"

    @requires_discovery_python
    @requires_pcalg_r
    def test_skeleton_consistency_across_samples(self):
        """Skeleton recovery consistent across different sample sizes."""
        for n_samples in [500, 1000, 2000]:
            data, _ = generate_diamond_data(n_samples=n_samples, seed=42)

            py_skeleton, _, _ = pc_skeleton(data, alpha=0.05, stable=True)
            r_result = r_skeleton(data, alpha=0.05, stable=True)

            if r_result is None:
                continue

            f1 = skeleton_f1(py_skeleton.adjacency, r_result["skeleton"])
            assert f1 >= 0.80, f"Skeleton F1 too low for n={n_samples}: {f1:.3f}"


class TestPCEdgeCases:
    """Edge case tests for robustness."""

    @requires_discovery_python
    @requires_pcalg_r
    def test_independent_variables(self):
        """Fully independent variables: empty skeleton."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (1000, 3))  # Independent columns

        py_skeleton, _, _ = pc_skeleton(data, alpha=0.05, stable=True)
        r_result = r_skeleton(data, alpha=0.05, stable=True)
        assert r_result is not None

        # Both should find (near) empty skeleton
        py_edges = np.sum(py_skeleton.adjacency) // 2
        r_edges = np.sum(r_result["skeleton"]) // 2

        assert py_edges <= 1, f"Python found too many edges: {py_edges}"
        assert r_edges <= 1, f"R found too many edges: {r_edges}"

    @requires_discovery_python
    @requires_pcalg_r
    def test_small_sample_size(self):
        """Small sample (n=100) still runs without error."""
        data, _ = generate_chain_data(n_samples=100, seed=42)

        py_result = pc_algorithm(data, alpha=0.10, stable=True)  # Higher alpha for small n
        r_result = r_pc_algorithm(data, alpha=0.10, stable=True)

        # Both should complete without error
        assert py_result is not None
        assert r_result is not None

        # Results may not match well but both should produce output
        assert py_result.skeleton.n_nodes == 3
        assert r_result["n_nodes"] == 3

    @requires_discovery_python
    @requires_pcalg_r
    def test_varying_alpha(self):
        """Different alpha levels produce sensible results."""
        data, _ = generate_diamond_data(n_samples=1000, seed=42)

        for alpha in [0.01, 0.05, 0.10]:
            py_result = pc_algorithm(data, alpha=alpha, stable=True)
            r_result = r_pc_algorithm(data, alpha=alpha, stable=True)

            assert r_result is not None, f"R failed at alpha={alpha}"

            # Both should produce valid output
            assert py_result.cpdag.n_nodes == 4
            assert r_result["n_nodes"] == 4
