"""Fixtures for causal discovery tests.

Session 133: Test data generation for PC algorithm and LiNGAM.

Provides known DAGs and generated data for validating discovery algorithms.
"""

import numpy as np
import pytest

from causal_inference.discovery import (
    DAG,
    CPDAG,
    Graph,
    generate_random_dag,
    generate_dag_data,
    dag_to_cpdag,
)


# =============================================================================
# Known Structure Fixtures
# =============================================================================


@pytest.fixture
def chain_dag():
    """Simple chain: X0 -> X1 -> X2.

    All edges should be undirected in CPDAG (no v-structures).
    """
    dag = DAG(n_nodes=3, node_names=["X0", "X1", "X2"])
    dag.add_edge(0, 1)  # X0 -> X1
    dag.add_edge(1, 2)  # X1 -> X2
    return dag


@pytest.fixture
def fork_dag():
    """Fork structure: X0 <- X1 -> X2.

    All edges undirected in CPDAG (no v-structures).
    """
    dag = DAG(n_nodes=3, node_names=["X0", "X1", "X2"])
    dag.add_edge(1, 0)  # X1 -> X0
    dag.add_edge(1, 2)  # X1 -> X2
    return dag


@pytest.fixture
def collider_dag():
    """Collider (v-structure): X0 -> X1 <- X2.

    This is a v-structure, so edges X0 -> X1 and X2 -> X1 are
    directed (compelled) in the CPDAG.
    """
    dag = DAG(n_nodes=3, node_names=["X0", "X1", "X2"])
    dag.add_edge(0, 1)  # X0 -> X1
    dag.add_edge(2, 1)  # X2 -> X1
    return dag


@pytest.fixture
def diamond_dag():
    """Diamond structure: X0 -> X1, X0 -> X2, X1 -> X3, X2 -> X3.

    Has v-structure at X3.
    """
    dag = DAG(n_nodes=4, node_names=["X0", "X1", "X2", "X3"])
    dag.add_edge(0, 1)  # X0 -> X1
    dag.add_edge(0, 2)  # X0 -> X2
    dag.add_edge(1, 3)  # X1 -> X3
    dag.add_edge(2, 3)  # X2 -> X3
    return dag


@pytest.fixture
def five_node_dag():
    """5-node DAG for moderate complexity tests.

    Structure:
    X0 -> X1 -> X3
    X0 -> X2 -> X3
    X3 -> X4
    """
    dag = DAG(n_nodes=5, node_names=[f"X{i}" for i in range(5)])
    dag.add_edge(0, 1)
    dag.add_edge(0, 2)
    dag.add_edge(1, 3)
    dag.add_edge(2, 3)
    dag.add_edge(3, 4)
    return dag


@pytest.fixture
def random_dag_small():
    """Random 6-node DAG with moderate density."""
    return generate_random_dag(6, edge_prob=0.4, seed=42)


@pytest.fixture
def random_dag_medium():
    """Random 10-node DAG for scalability tests."""
    return generate_random_dag(10, edge_prob=0.3, seed=123)


# =============================================================================
# Data Generation Fixtures
# =============================================================================


@pytest.fixture
def chain_data_gaussian(chain_dag):
    """Gaussian data from chain DAG."""
    data, B = generate_dag_data(
        chain_dag,
        n_samples=1000,
        noise_type="gaussian",
        seed=42,
    )
    return data, B, chain_dag


@pytest.fixture
def chain_data_laplace(chain_dag):
    """Non-Gaussian (Laplace) data from chain DAG."""
    data, B = generate_dag_data(
        chain_dag,
        n_samples=1000,
        noise_type="laplace",
        seed=42,
    )
    return data, B, chain_dag


@pytest.fixture
def collider_data_gaussian(collider_dag):
    """Gaussian data from collider DAG."""
    data, B = generate_dag_data(
        collider_dag,
        n_samples=1000,
        noise_type="gaussian",
        seed=42,
    )
    return data, B, collider_dag


@pytest.fixture
def collider_data_laplace(collider_dag):
    """Non-Gaussian data from collider DAG."""
    data, B = generate_dag_data(
        collider_dag,
        n_samples=1000,
        noise_type="laplace",
        seed=42,
    )
    return data, B, collider_dag


@pytest.fixture
def diamond_data_gaussian(diamond_dag):
    """Gaussian data from diamond DAG."""
    data, B = generate_dag_data(
        diamond_dag,
        n_samples=1000,
        noise_type="gaussian",
        seed=42,
    )
    return data, B, diamond_dag


@pytest.fixture
def five_node_data_gaussian(five_node_dag):
    """Gaussian data from 5-node DAG."""
    data, B = generate_dag_data(
        five_node_dag,
        n_samples=1000,
        noise_type="gaussian",
        seed=42,
    )
    return data, B, five_node_dag


@pytest.fixture
def five_node_data_laplace(five_node_dag):
    """Non-Gaussian data from 5-node DAG."""
    data, B = generate_dag_data(
        five_node_dag,
        n_samples=1000,
        noise_type="laplace",
        seed=42,
    )
    return data, B, five_node_dag


@pytest.fixture
def random_data_gaussian(random_dag_small):
    """Gaussian data from random DAG."""
    data, B = generate_dag_data(
        random_dag_small,
        n_samples=2000,
        noise_type="gaussian",
        seed=42,
    )
    return data, B, random_dag_small


@pytest.fixture
def random_data_laplace(random_dag_small):
    """Non-Gaussian data from random DAG."""
    data, B = generate_dag_data(
        random_dag_small,
        n_samples=2000,
        noise_type="laplace",
        seed=42,
    )
    return data, B, random_dag_small


# =============================================================================
# High-Dimensional Fixtures
# =============================================================================


@pytest.fixture
def highdim_sparse_dag():
    """20-node sparse DAG for high-dimensional tests."""
    return generate_random_dag(20, edge_prob=0.15, seed=999)


@pytest.fixture
def highdim_sparse_data(highdim_sparse_dag):
    """Data from high-dimensional sparse DAG."""
    data, B = generate_dag_data(
        highdim_sparse_dag,
        n_samples=5000,
        noise_type="laplace",
        seed=999,
    )
    return data, B, highdim_sparse_dag


# =============================================================================
# Helper Functions
# =============================================================================


def generate_faithful_data(dag, n_samples=1000, seed=42):
    """Generate data satisfying faithfulness assumption.

    Avoids coefficient cancellation by ensuring strong effects.
    """
    return generate_dag_data(
        dag,
        n_samples=n_samples,
        coefficient_range=(0.7, 1.5),  # Strong effects
        noise_scale=1.0,
        seed=seed,
    )


def generate_near_unfaithful_data(dag, n_samples=1000, seed=42):
    """Generate data near faithfulness violation.

    Uses small coefficients that may cause near-cancellation.
    """
    return generate_dag_data(
        dag,
        n_samples=n_samples,
        coefficient_range=(0.1, 0.3),  # Weak effects
        noise_scale=1.0,
        seed=seed,
    )
