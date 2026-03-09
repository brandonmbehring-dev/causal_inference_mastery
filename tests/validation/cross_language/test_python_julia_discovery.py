"""
Cross-language validation tests for Causal Discovery algorithms.

Tests Python ↔ Julia parity for PC algorithm and LiNGAM.

Tolerance Strategy:
- Skeleton edges: Exact match (deterministic with fixed data)
- CPDAG directed: Exact match (same Meek rules)
- F1 scores: rtol=0.05 (float precision)
- Causal order: Element-wise match
- Adjacency coefficients: rtol=0.10 (estimation noise)

Session 134: Cross-language parity for Session 133 discovery methods.
"""

import numpy as np
import pytest

from src.causal_inference.discovery import (
    pc_algorithm,
    direct_lingam,
    generate_random_dag,
    generate_dag_data,
    skeleton_f1,
    compute_shd,
    fisher_z_test,
)
from tests.validation.cross_language.julia_interface import (
    is_julia_available,
    julia_generate_random_dag,
    julia_generate_dag_data,
    julia_pc_algorithm,
    julia_direct_lingam,
    julia_skeleton_f1,
    julia_compute_shd,
    julia_fisher_z_test,
)


pytestmark = pytest.mark.skipif(
    not is_julia_available(), reason="Julia not available for cross-validation"
)


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_discovery_test_data(
    n_vars: int = 5,
    n_samples: int = 1000,
    edge_prob: float = 0.3,
    noise_type: str = "gaussian",
    seed: int = 42,
):
    """
    Generate data for discovery cross-language testing.

    Parameters
    ----------
    n_vars : int
        Number of variables
    n_samples : int
        Number of samples
    edge_prob : float
        Edge probability for random DAG
    noise_type : str
        Noise distribution
    seed : int
        Random seed

    Returns
    -------
    dict
        Contains data, true_dag, adjacency_matrix
    """
    dag = generate_random_dag(n_vars, edge_prob=edge_prob, seed=seed)
    data, B = generate_dag_data(dag, n_samples=n_samples, noise_type=noise_type, seed=seed)

    return {
        "data": data,
        "true_dag": dag,
        "true_adjacency": dag.adjacency_matrix,
        "B": B,
        "n_vars": n_vars,
        "n_samples": n_samples,
    }


# =============================================================================
# PC Algorithm Skeleton Parity Tests
# =============================================================================


class TestPCSkeletonParity:
    """Test PC algorithm skeleton recovery matches between Python and Julia."""

    def test_skeleton_edges_match(self):
        """Skeleton adjacency should match exactly."""
        test_data = generate_discovery_test_data(n_vars=5, n_samples=1000, seed=42)

        py_result = pc_algorithm(test_data["data"], alpha=0.01)
        jl_result = julia_pc_algorithm(test_data["data"], alpha=0.01)

        # Skeleton should match (symmetric adjacency)
        py_skeleton = py_result.skeleton.adjacency_matrix
        jl_skeleton = jl_result["skeleton"]

        np.testing.assert_array_equal(
            py_skeleton, jl_skeleton, err_msg="Skeleton adjacency mismatch between Python and Julia"
        )

    def test_n_ci_tests_match(self):
        """Number of CI tests should match."""
        test_data = generate_discovery_test_data(n_vars=5, n_samples=1000, seed=42)

        py_result = pc_algorithm(test_data["data"], alpha=0.01)
        jl_result = julia_pc_algorithm(test_data["data"], alpha=0.01)

        assert py_result.n_ci_tests == jl_result["n_ci_tests"], (
            f"CI test count mismatch: Python={py_result.n_ci_tests}, Julia={jl_result['n_ci_tests']}"
        )

    def test_skeleton_f1_parity(self):
        """Skeleton F1 metrics should match."""
        test_data = generate_discovery_test_data(n_vars=5, n_samples=1000, seed=42)

        py_result = pc_algorithm(test_data["data"], alpha=0.01)

        # Python F1
        py_prec, py_rec, py_f1 = skeleton_f1(py_result.skeleton, test_data["true_dag"])

        # Julia F1
        jl_result = julia_skeleton_f1(
            py_result.skeleton.adjacency_matrix, test_data["true_adjacency"]
        )

        np.testing.assert_allclose(py_prec, jl_result["precision"], rtol=0.05)
        np.testing.assert_allclose(py_rec, jl_result["recall"], rtol=0.05)
        np.testing.assert_allclose(py_f1, jl_result["f1"], rtol=0.05)


# =============================================================================
# PC Algorithm CPDAG Orientation Parity Tests
# =============================================================================


class TestPCOrientationParity:
    """Test PC algorithm CPDAG orientation matches between Python and Julia."""

    def test_cpdag_directed_edges_match(self):
        """CPDAG directed edges should match."""
        test_data = generate_discovery_test_data(n_vars=5, n_samples=1000, seed=42)

        py_result = pc_algorithm(test_data["data"], alpha=0.01)
        jl_result = julia_pc_algorithm(test_data["data"], alpha=0.01)

        # Directed edges should match
        py_directed = py_result.cpdag.directed_matrix
        jl_directed = jl_result["cpdag_directed"]

        np.testing.assert_array_equal(
            py_directed, jl_directed, err_msg="CPDAG directed edges mismatch"
        )

    def test_cpdag_undirected_edges_match(self):
        """CPDAG undirected edges should match."""
        test_data = generate_discovery_test_data(n_vars=5, n_samples=1000, seed=42)

        py_result = pc_algorithm(test_data["data"], alpha=0.01)
        jl_result = julia_pc_algorithm(test_data["data"], alpha=0.01)

        # Undirected edges should match
        py_undirected = py_result.cpdag.undirected_matrix
        jl_undirected = jl_result["cpdag_undirected"]

        np.testing.assert_array_equal(
            py_undirected, jl_undirected, err_msg="CPDAG undirected edges mismatch"
        )

    def test_shd_parity(self):
        """Structural Hamming Distance should match."""
        test_data = generate_discovery_test_data(n_vars=5, n_samples=1000, seed=42)

        py_result = pc_algorithm(test_data["data"], alpha=0.01)

        # Python SHD
        py_shd = compute_shd(py_result.cpdag, test_data["true_dag"])

        # Julia SHD
        jl_shd = julia_compute_shd(
            py_result.cpdag.directed_matrix,
            py_result.cpdag.undirected_matrix,
            test_data["true_adjacency"],
        )

        assert py_shd == jl_shd["shd"], f"SHD mismatch: Python={py_shd}, Julia={jl_shd['shd']}"


# =============================================================================
# LiNGAM Causal Order Parity Tests
# =============================================================================


class TestLiNGAMCausalOrderParity:
    """Test LiNGAM causal order matches between Python and Julia."""

    def test_causal_order_matches(self):
        """Causal order should match element-wise."""
        # Use Laplace noise for LiNGAM
        test_data = generate_discovery_test_data(
            n_vars=4, n_samples=1000, noise_type="laplace", seed=42
        )

        py_result = direct_lingam(test_data["data"], seed=42)
        jl_result = julia_direct_lingam(test_data["data"], seed=42)

        # Causal order should match
        np.testing.assert_array_equal(
            py_result.causal_order,
            jl_result["causal_order"],
            err_msg="Causal order mismatch between Python and Julia",
        )

    def test_dag_structure_matches(self):
        """DAG structure (edges) should match."""
        test_data = generate_discovery_test_data(
            n_vars=4, n_samples=1000, noise_type="laplace", seed=42
        )

        py_result = direct_lingam(test_data["data"], seed=42)
        jl_result = julia_direct_lingam(test_data["data"], seed=42)

        # DAG adjacency should match
        np.testing.assert_array_equal(
            py_result.dag.adjacency_matrix,
            jl_result["dag_adjacency"],
            err_msg="DAG adjacency mismatch",
        )

    def test_adjacency_coefficients_similar(self):
        """Weighted adjacency coefficients should be close."""
        test_data = generate_discovery_test_data(
            n_vars=4, n_samples=1000, noise_type="laplace", seed=42
        )

        py_result = direct_lingam(test_data["data"], seed=42)
        jl_result = julia_direct_lingam(test_data["data"], seed=42)

        # Coefficients should be close (rtol=0.10 for estimation noise)
        np.testing.assert_allclose(
            py_result.adjacency_matrix,
            jl_result["adjacency_matrix"],
            rtol=0.10,
            atol=0.05,
            err_msg="Adjacency coefficients differ significantly",
        )


# =============================================================================
# Discovery Metrics Parity Tests
# =============================================================================


class TestDiscoveryMetricsParity:
    """Test discovery evaluation metrics match between Python and Julia."""

    def test_dag_generation_parity(self):
        """Random DAG generation should produce same structure with same seed."""
        py_dag = generate_random_dag(5, edge_prob=0.3, seed=42)
        jl_dag = julia_generate_random_dag(5, edge_prob=0.3, seed=42)

        np.testing.assert_array_equal(
            py_dag.adjacency_matrix,
            jl_dag["adjacency_matrix"],
            err_msg="Random DAG generation differs between Python and Julia",
        )

    def test_data_generation_parity(self):
        """Data generation should produce same output with same seed."""
        py_dag = generate_random_dag(4, edge_prob=0.4, seed=42)

        py_data, py_B = generate_dag_data(py_dag, n_samples=100, seed=42)
        jl_result = julia_generate_dag_data(
            py_dag.adjacency_matrix, n_samples=100, noise_type="gaussian", seed=42
        )

        # Data should match closely (floating point precision)
        np.testing.assert_allclose(
            py_data, jl_result["data"], rtol=1e-10, err_msg="Data generation differs"
        )

        # Coefficient matrix should match
        np.testing.assert_allclose(
            py_B, jl_result["B"], rtol=1e-10, err_msg="Coefficient matrix differs"
        )


# =============================================================================
# Independence Tests Parity
# =============================================================================


class TestIndependenceTestParity:
    """Test conditional independence tests match between Python and Julia."""

    def test_fisher_z_unconditional(self):
        """Unconditional Fisher Z test should match."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n)
        Y = 0.7 * X + 0.3 * np.random.randn(n)
        data = np.column_stack([X, Y])

        py_result = fisher_z_test(data, 0, 1, alpha=0.05)
        jl_result = julia_fisher_z_test(data, 0, 1, alpha=0.05)

        np.testing.assert_allclose(py_result.pvalue, jl_result["pvalue"], rtol=0.01)
        np.testing.assert_allclose(py_result.statistic, jl_result["statistic"], rtol=0.01)
        assert py_result.independent == jl_result["independent"]

    def test_fisher_z_conditional(self):
        """Conditional Fisher Z test should match."""
        np.random.seed(42)
        n = 500
        Z = np.random.randn(n)
        X = 0.7 * Z + 0.3 * np.random.randn(n)
        Y = 0.7 * Z + 0.3 * np.random.randn(n)
        data = np.column_stack([X, Y, Z])

        py_result = fisher_z_test(data, 0, 1, conditioning_set=[2], alpha=0.05)
        jl_result = julia_fisher_z_test(data, 0, 1, conditioning_set=[2], alpha=0.05)

        np.testing.assert_allclose(py_result.pvalue, jl_result["pvalue"], rtol=0.01)
        assert py_result.independent == jl_result["independent"]


# =============================================================================
# Larger DAG Parity Tests
# =============================================================================


class TestLargerDAGParity:
    """Test parity on larger DAGs (10+ nodes)."""

    @pytest.mark.slow
    def test_pc_10_node_dag(self):
        """PC algorithm should match on 10-node DAG."""
        test_data = generate_discovery_test_data(
            n_vars=10, n_samples=2000, edge_prob=0.25, seed=123
        )

        py_result = pc_algorithm(test_data["data"], alpha=0.01)
        jl_result = julia_pc_algorithm(test_data["data"], alpha=0.01)

        # Skeleton match
        np.testing.assert_array_equal(py_result.skeleton.adjacency_matrix, jl_result["skeleton"])

        # CPDAG match
        np.testing.assert_array_equal(py_result.cpdag.directed_matrix, jl_result["cpdag_directed"])

    @pytest.mark.slow
    def test_lingam_8_node_dag(self):
        """LiNGAM should match on 8-node DAG."""
        test_data = generate_discovery_test_data(
            n_vars=8, n_samples=2000, edge_prob=0.3, noise_type="laplace", seed=123
        )

        py_result = direct_lingam(test_data["data"], seed=123)
        jl_result = julia_direct_lingam(test_data["data"], seed=123)

        # Causal order should match
        np.testing.assert_array_equal(py_result.causal_order, jl_result["causal_order"])
