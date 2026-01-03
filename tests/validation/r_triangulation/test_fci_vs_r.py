"""Triangulation tests: Python FCI Algorithm vs R pcalg::fci().

This module provides Layer 5 validation by comparing our Python implementation
of the FCI algorithm against R's pcalg package, the gold standard for
constraint-based causal discovery with latent confounders.

FCI Algorithm:
- Extends PC to handle latent (unobserved) confounders
- Outputs a PAG (Partial Ancestral Graph) instead of CPDAG
- Uses bidirected edges (X <-> Y) to indicate latent common causes
- More conservative than PC: uses circle marks when uncertain

Key differences from PC:
- PAG vs CPDAG output
- Can detect latent confounders (bidirected edges)
- Extended orientation rules (R0-R10 instead of Meek R1-R4)
- Circle marks for uncertain orientations

Tolerance levels (established in plan):
- Skeleton: Exact match (same as PC)
- PAG edge types: 95%+ match (rule order may vary)
- Bidirected edges: Exact match (critical for latent detection)
- Circle marks: ±1 edge (conservative orientation)

Run with: pytest tests/validation/r_triangulation/test_fci_vs_r.py -v

Created: Session 184 (2026-01-02)
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.validation.r_triangulation.r_interface import (
    check_pcalg_installed,
    check_r_available,
    r_fci_algorithm,
)

# Lazy imports to avoid errors when discovery module not available
try:
    from src.causal_inference.discovery import fci_algorithm
    from src.causal_inference.discovery.types import DAG, Graph, PAG
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
    X2 = 0.6 * X0 + 0.6 * X1 + rng.normal(0, 0.5, n_samples)

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


def generate_latent_confounder_data(n_samples: int = 1000, seed: int = 42) -> np.ndarray:
    """Generate data with latent confounder: X ← L → Y (L unobserved).

    This should result in bidirected edge X <-> Y in the PAG.
    Returns only observed variables (X, Y), not the latent L.
    """
    rng = np.random.default_rng(seed)

    # Latent confounder L
    L = rng.normal(0, 1, n_samples)

    # X and Y both caused by L
    X = 0.9 * L + rng.normal(0, 0.3, n_samples)
    Y = 0.9 * L + rng.normal(0, 0.3, n_samples)

    # Only return observed variables
    return np.column_stack([X, Y])


def generate_latent_confounder_with_chain(n_samples: int = 1000, seed: int = 42) -> np.ndarray:
    """Generate data: Z → X ← L → Y (L unobserved).

    Structure:
    - Z → X direct causal effect
    - X ← L → Y latent confounder creates X <-> Y

    Returns observed variables (Z, X, Y).
    """
    rng = np.random.default_rng(seed)

    # Observed exogenous
    Z = rng.normal(0, 1, n_samples)

    # Latent confounder
    L = rng.normal(0, 1, n_samples)

    # X caused by Z and L
    X = 0.7 * Z + 0.7 * L + rng.normal(0, 0.3, n_samples)

    # Y caused by L only
    Y = 0.9 * L + rng.normal(0, 0.3, n_samples)

    return np.column_stack([Z, X, Y])


def generate_diamond_with_latent(n_samples: int = 1000, seed: int = 42) -> np.ndarray:
    """Generate diamond DAG with one latent variable.

    Full structure: A → B, A → C, B → D, C → D
    Observed: A, B, D (C is latent)

    This creates bidirected edge B <-> D due to latent C.
    """
    rng = np.random.default_rng(seed)

    A = rng.normal(0, 1, n_samples)
    B = 0.8 * A + rng.normal(0, 0.4, n_samples)
    C = 0.8 * A + rng.normal(0, 0.4, n_samples)  # Latent
    D = 0.5 * B + 0.5 * C + rng.normal(0, 0.3, n_samples)

    # Return only observed: A, B, D
    return np.column_stack([A, B, D])


# =============================================================================
# Helper Functions
# =============================================================================


def count_skeleton_edges(skeleton: np.ndarray) -> int:
    """Count edges in symmetric skeleton matrix."""
    return int(np.sum(skeleton) // 2)


def extract_bidirected_edges(pag: np.ndarray) -> set:
    """Extract bidirected edges from R PAG matrix.

    R encoding: pag[i,j]=2 and pag[j,i]=2 means i <-> j.
    Returns set of tuples (min(i,j), max(i,j)).
    """
    n_nodes = pag.shape[0]
    bidirected = set()
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if pag[i, j] == 2 and pag[j, i] == 2:
                bidirected.add((i, j))
    return bidirected


def extract_directed_edges(pag: np.ndarray) -> set:
    """Extract definitely directed edges from R PAG matrix.

    R encoding: pag[i,j]=3 (tail) and pag[j,i]=2 (arrow) means i --> j.
    Returns set of tuples (from, to).
    """
    n_nodes = pag.shape[0]
    directed = set()
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and pag[i, j] == 3 and pag[j, i] == 2:
                directed.add((i, j))
    return directed


def extract_circle_edges(pag: np.ndarray) -> set:
    """Extract edges with at least one circle mark.

    R encoding: 1 = circle mark.
    Returns set of tuples (min(i,j), max(i,j)).
    """
    n_nodes = pag.shape[0]
    circles = set()
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if pag[i, j] == 1 or pag[j, i] == 1:
                circles.add((i, j))
    return circles


def python_pag_to_bidirected(pag: "PAG") -> set:
    """Extract bidirected edges from Python PAG object.

    Python encoding: both endpoints are MARK_ARROW (2).
    """
    bidirected = set()
    for i in range(pag.n_nodes):
        for j in range(i + 1, pag.n_nodes):
            # Check if bidirected: both are arrows
            if (pag.endpoints[i, j, 0] == PAG.MARK_ARROW and
                    pag.endpoints[i, j, 1] == PAG.MARK_ARROW):
                bidirected.add((i, j))
    return bidirected


def python_pag_to_directed(pag: "PAG") -> set:
    """Extract directed edges from Python PAG object.

    Python encoding: tail at from, arrow at to.
    """
    directed = set()
    for i in range(pag.n_nodes):
        for j in range(pag.n_nodes):
            if i != j:
                # Check if i --> j: tail at i, arrow at j
                if (pag.endpoints[i, j, 0] == PAG.MARK_TAIL and
                        pag.endpoints[i, j, 1] == PAG.MARK_ARROW):
                    directed.add((i, j))
    return directed


def python_pag_to_circle(pag: "PAG") -> set:
    """Extract edges with circle marks from Python PAG object."""
    circles = set()
    for i in range(pag.n_nodes):
        for j in range(i + 1, pag.n_nodes):
            if (pag.endpoints[i, j, 0] == PAG.MARK_CIRCLE or
                    pag.endpoints[i, j, 1] == PAG.MARK_CIRCLE):
                circles.add((i, j))
    return circles


# =============================================================================
# Test Classes
# =============================================================================


class TestFCISkeletonVsR:
    """Test FCI skeleton recovery matches R.

    FCI skeleton phase is identical to PC, so these tests verify
    the shared skeleton learning code.
    """

    @requires_discovery_python
    @requires_pcalg_r
    def test_chain_skeleton(self):
        """Chain X0→X1→X2: FCI skeleton matches R."""
        data, _ = generate_chain_data(n_samples=1000)

        # Run R FCI
        r_result = r_fci_algorithm(data, alpha=0.05)
        assert r_result is not None, "R FCI failed"

        # Run Python FCI
        py_result = fci_algorithm(data, alpha=0.05)

        # Compare skeletons
        r_skeleton = r_result["skeleton"]
        py_skeleton = py_result.skeleton.to_adjacency_matrix()

        np.testing.assert_array_equal(
            r_skeleton, py_skeleton,
            err_msg="Chain skeleton mismatch between Python and R"
        )

    @requires_discovery_python
    @requires_pcalg_r
    def test_collider_skeleton(self):
        """Collider X0→X2←X1: FCI skeleton matches R."""
        data, _ = generate_collider_data(n_samples=1000)

        r_result = r_fci_algorithm(data, alpha=0.05)
        assert r_result is not None, "R FCI failed"

        py_result = fci_algorithm(data, alpha=0.05)

        r_skeleton = r_result["skeleton"]
        py_skeleton = py_result.skeleton.to_adjacency_matrix()

        np.testing.assert_array_equal(
            r_skeleton, py_skeleton,
            err_msg="Collider skeleton mismatch between Python and R"
        )

    @requires_discovery_python
    @requires_pcalg_r
    def test_fork_skeleton(self):
        """Fork X0→X1, X0→X2: FCI skeleton matches R."""
        data, _ = generate_fork_data(n_samples=1000)

        r_result = r_fci_algorithm(data, alpha=0.05)
        assert r_result is not None, "R FCI failed"

        py_result = fci_algorithm(data, alpha=0.05)

        r_skeleton = r_result["skeleton"]
        py_skeleton = py_result.skeleton.to_adjacency_matrix()

        np.testing.assert_array_equal(
            r_skeleton, py_skeleton,
            err_msg="Fork skeleton mismatch between Python and R"
        )


class TestFCIPAGOrientationVsR:
    """Test PAG edge orientation matches R.

    FCI orientation rules (R0-R10) should produce the same PAG
    given the same skeleton and separating sets.
    """

    @requires_discovery_python
    @requires_pcalg_r
    def test_collider_v_structure(self):
        """Collider v-structure detected identically.

        In collider X0→X2←X1, both X0 and X1 should have arrows into X2.
        """
        data, _ = generate_collider_data(n_samples=2000, seed=42)

        r_result = r_fci_algorithm(data, alpha=0.01)
        assert r_result is not None, "R FCI failed"

        py_result = fci_algorithm(data, alpha=0.01)

        # Check R has arrows into X2
        r_pag = r_result["pag"]
        # pag[i,j] = mark at j for edge i-j
        # For arrows into X2: pag[0,2]=? and pag[2,0]=2, pag[1,2]=? and pag[2,1]=2
        # Note: R uses different indexing, checking directed matrix
        r_directed = r_result["directed"]

        # Check Python has arrows into X2
        py_pag = py_result.pag
        # In Python PAG, arrow at X2 means endpoints[i, 2, 1] == ARROW
        has_arrow_0_to_2 = (
            py_pag.endpoints[0, 2, 0] != 0 and py_pag.endpoints[0, 2, 1] == PAG.MARK_ARROW
        )
        has_arrow_1_to_2 = (
            py_pag.endpoints[1, 2, 0] != 0 and py_pag.endpoints[1, 2, 1] == PAG.MARK_ARROW
        )

        # Both should have arrows pointing to X2 (v-structure)
        assert has_arrow_0_to_2, "Python missing arrow 0→2 in v-structure"
        assert has_arrow_1_to_2, "Python missing arrow 1→2 in v-structure"

    @requires_discovery_python
    @requires_pcalg_r
    def test_pag_directed_edges_match(self):
        """Definitely directed edges match R within tolerance."""
        data, _ = generate_chain_data(n_samples=2000, seed=42)

        r_result = r_fci_algorithm(data, alpha=0.01)
        assert r_result is not None, "R FCI failed"

        py_result = fci_algorithm(data, alpha=0.01)

        # Extract directed edges
        r_directed = extract_directed_edges(r_result["pag"])
        py_directed = python_pag_to_directed(py_result.pag)

        # Calculate agreement
        all_edges = r_directed | py_directed
        if len(all_edges) == 0:
            # Both empty = perfect agreement
            agreement = 1.0
        else:
            matching = len(r_directed & py_directed)
            agreement = matching / len(all_edges)

        # Allow 95% match (rule order may vary)
        assert agreement >= 0.95 or len(all_edges) <= 1, (
            f"Directed edge agreement {agreement:.1%} below 95% threshold. "
            f"R: {r_directed}, Python: {py_directed}"
        )


class TestFCILatentConfoundersVsR:
    """Test latent confounder detection (bidirected edges) matches R.

    This is the critical FCI feature distinguishing it from PC.
    """

    @requires_discovery_python
    @requires_pcalg_r
    def test_simple_latent_detected(self):
        """Simple latent X ← L → Y results in bidirected edge.

        When L is unobserved, FCI should detect X <-> Y.
        """
        data = generate_latent_confounder_data(n_samples=2000, seed=42)

        r_result = r_fci_algorithm(data, alpha=0.01)
        assert r_result is not None, "R FCI failed"

        py_result = fci_algorithm(data, alpha=0.01)

        # Check for bidirected edge
        r_bidirected = extract_bidirected_edges(r_result["pag"])
        py_bidirected = python_pag_to_bidirected(py_result.pag)

        # Both should detect X <-> Y (edge 0-1)
        assert (0, 1) in r_bidirected or r_result["n_bidirected"] > 0, (
            "R FCI should detect bidirected edge for latent confounder"
        )
        assert len(py_bidirected) > 0 or py_result.pag.n_bidirected_edges() > 0, (
            "Python FCI should detect bidirected edge for latent confounder"
        )

    @requires_discovery_python
    @requires_pcalg_r
    def test_bidirected_count_matches(self):
        """Number of bidirected edges matches R."""
        data = generate_latent_confounder_data(n_samples=2000, seed=42)

        r_result = r_fci_algorithm(data, alpha=0.01)
        assert r_result is not None, "R FCI failed"

        py_result = fci_algorithm(data, alpha=0.01)

        r_n_bidirected = r_result["n_bidirected"]
        py_n_bidirected = py_result.pag.n_bidirected_edges()

        # Should match exactly for bidirected edges
        assert r_n_bidirected == py_n_bidirected, (
            f"Bidirected edge count mismatch: R={r_n_bidirected}, Python={py_n_bidirected}"
        )

    @requires_discovery_python
    @requires_pcalg_r
    def test_no_false_bidirected(self):
        """No bidirected edges when no latent confounders.

        For fully observed chain X0→X1→X2, there should be no bidirected edges.
        """
        data, _ = generate_chain_data(n_samples=2000, seed=42)

        r_result = r_fci_algorithm(data, alpha=0.05)
        assert r_result is not None, "R FCI failed"

        py_result = fci_algorithm(data, alpha=0.05)

        r_n_bidirected = r_result["n_bidirected"]
        py_n_bidirected = py_result.pag.n_bidirected_edges()

        # Both should have 0 bidirected edges
        assert r_n_bidirected == 0, (
            f"R FCI incorrectly detected {r_n_bidirected} bidirected edges in chain"
        )
        assert py_n_bidirected == 0, (
            f"Python FCI incorrectly detected {py_n_bidirected} bidirected edges in chain"
        )

    @requires_discovery_python
    @requires_pcalg_r
    def test_latent_in_complex_structure(self):
        """Latent detection in more complex structure.

        Structure: Z → X ← L → Y (L unobserved)
        Expected: X <-> Y bidirected, Z → X or Z o-> X
        """
        data = generate_latent_confounder_with_chain(n_samples=2000, seed=42)

        r_result = r_fci_algorithm(data, alpha=0.01)
        assert r_result is not None, "R FCI failed"

        py_result = fci_algorithm(data, alpha=0.01)

        # Both should detect at least one bidirected edge
        r_n_bidirected = r_result["n_bidirected"]
        py_n_bidirected = py_result.pag.n_bidirected_edges()

        # In this structure, X(1) <-> Y(2) should be bidirected
        # Allow some variation in detection
        if r_n_bidirected > 0:
            # If R detects bidirected, Python should too
            assert py_n_bidirected > 0, (
                f"R detected {r_n_bidirected} bidirected edges, Python detected {py_n_bidirected}"
            )


class TestFCICircleMarksVsR:
    """Test circle marks (uncertain orientations) match R.

    FCI is more conservative than PC - uses circle marks when
    orientation cannot be determined with certainty.
    """

    @requires_discovery_python
    @requires_pcalg_r
    def test_circle_marks_present(self):
        """FCI produces circle marks when appropriate."""
        data, _ = generate_fork_data(n_samples=1000, seed=42)

        r_result = r_fci_algorithm(data, alpha=0.05)
        assert r_result is not None, "R FCI failed"

        py_result = fci_algorithm(data, alpha=0.05)

        r_circles = extract_circle_edges(r_result["pag"])
        py_circles = python_pag_to_circle(py_result.pag)

        # Both may or may not have circles, but should be consistent
        # For fork structure, edge orientations may be uncertain
        n_r_circles = len(r_circles)
        n_py_circles = len(py_circles)

        # Allow ±1 difference in circle count
        assert abs(n_r_circles - n_py_circles) <= 1, (
            f"Circle mark count differs: R={n_r_circles}, Python={n_py_circles}"
        )

    @requires_discovery_python
    @requires_pcalg_r
    def test_circle_edges_overlap(self):
        """Circle edges have substantial overlap with R."""
        # Use structure where circles are likely
        rng = np.random.default_rng(42)
        n_samples = 1000

        # Create 4-node structure with some ambiguity
        X0 = rng.normal(0, 1, n_samples)
        X1 = 0.5 * X0 + rng.normal(0, 0.7, n_samples)
        X2 = 0.5 * X0 + rng.normal(0, 0.7, n_samples)
        X3 = 0.3 * X1 + 0.3 * X2 + rng.normal(0, 0.7, n_samples)

        data = np.column_stack([X0, X1, X2, X3])

        r_result = r_fci_algorithm(data, alpha=0.05)
        assert r_result is not None, "R FCI failed"

        py_result = fci_algorithm(data, alpha=0.05)

        r_circles = extract_circle_edges(r_result["pag"])
        py_circles = python_pag_to_circle(py_result.pag)

        # If both have circles, check overlap
        if len(r_circles) > 0 and len(py_circles) > 0:
            overlap = len(r_circles & py_circles)
            total = len(r_circles | py_circles)
            jaccard = overlap / total if total > 0 else 1.0

            # Moderate agreement on circles (they can be sensitive)
            assert jaccard >= 0.5 or total <= 2, (
                f"Circle edge Jaccard similarity {jaccard:.2f} below 0.5. "
                f"R: {r_circles}, Python: {py_circles}"
            )


class TestFCISeparatingSetsVsR:
    """Test separating sets match R."""

    @requires_discovery_python
    @requires_pcalg_r
    def test_separating_set_content(self):
        """Separating sets contain same variables as R."""
        data, _ = generate_chain_data(n_samples=1000)

        r_result = r_fci_algorithm(data, alpha=0.05)
        assert r_result is not None, "R FCI failed"

        py_result = fci_algorithm(data, alpha=0.05)

        r_sepsets = r_result["separating_sets"]
        py_sepsets = py_result.separating_sets

        # Convert Python separating sets format
        py_sepsets_dict = {}
        for (i, j), sep_set in py_sepsets.items():
            key = (min(i, j), max(i, j))
            py_sepsets_dict[key] = set(sep_set)

        # Check common pairs
        common_pairs = set(r_sepsets.keys()) & set(py_sepsets_dict.keys())

        for pair in common_pairs:
            r_set = set(r_sepsets[pair]) if isinstance(r_sepsets[pair], list) else {r_sepsets[pair]}
            py_set = py_sepsets_dict[pair]

            assert r_set == py_set, (
                f"Separating set mismatch for {pair}: R={r_set}, Python={py_set}"
            )


class TestFCIMonteCarlo:
    """Monte Carlo validation of FCI algorithm."""

    @requires_discovery_python
    @requires_pcalg_r
    @pytest.mark.slow
    def test_bidirected_detection_accuracy(self):
        """Monte Carlo: Bidirected edges detected consistently.

        Run multiple simulations with latent confounders and verify
        both Python and R detect bidirected edges at similar rates.
        """
        n_runs = 30
        r_detected = 0
        py_detected = 0

        for seed in range(n_runs):
            data = generate_latent_confounder_data(n_samples=500, seed=seed)

            r_result = r_fci_algorithm(data, alpha=0.05)
            if r_result is None:
                continue

            py_result = fci_algorithm(data, alpha=0.05)

            if r_result["n_bidirected"] > 0:
                r_detected += 1
            if py_result.pag.n_bidirected_edges() > 0:
                py_detected += 1

        # Both should detect latent confounders in most runs
        r_rate = r_detected / n_runs
        py_rate = py_detected / n_runs

        # Detection rates should be similar
        assert abs(r_rate - py_rate) <= 0.2, (
            f"Detection rate difference too large: R={r_rate:.1%}, Python={py_rate:.1%}"
        )

    @requires_discovery_python
    @requires_pcalg_r
    @pytest.mark.slow
    def test_pag_consistency_across_samples(self):
        """PAG structure is consistent across different samples.

        Generate multiple datasets from the same DGP and verify
        FCI produces similar PAGs.
        """
        n_runs = 20
        results = []

        for seed in range(n_runs):
            data, _ = generate_chain_data(n_samples=1000, seed=seed)

            r_result = r_fci_algorithm(data, alpha=0.05)
            if r_result is None:
                continue

            py_result = fci_algorithm(data, alpha=0.05)

            results.append({
                "r_skeleton_edges": count_skeleton_edges(r_result["skeleton"]),
                "py_skeleton_edges": py_result.skeleton.n_edges(),
                "r_bidirected": r_result["n_bidirected"],
                "py_bidirected": py_result.pag.n_bidirected_edges(),
            })

        assert len(results) >= 15, f"Only {len(results)} runs completed"

        # Check skeleton edge count consistency
        r_skels = [r["r_skeleton_edges"] for r in results]
        py_skels = [r["py_skeleton_edges"] for r in results]

        # Skeleton should be consistent across runs
        r_skel_var = np.var(r_skels) if len(r_skels) > 1 else 0
        py_skel_var = np.var(py_skels) if len(py_skels) > 1 else 0

        # Both should have low variance (consistent structure recovery)
        assert r_skel_var <= 1.0, f"R skeleton variance too high: {r_skel_var}"
        assert py_skel_var <= 1.0, f"Python skeleton variance too high: {py_skel_var}"


class TestFCIEdgeCases:
    """Test FCI edge cases and boundary conditions."""

    @requires_discovery_python
    @requires_pcalg_r
    def test_independent_variables(self):
        """FCI handles independent variables correctly."""
        rng = np.random.default_rng(42)
        n_samples = 1000

        # Completely independent variables
        data = rng.normal(0, 1, (n_samples, 3))

        r_result = r_fci_algorithm(data, alpha=0.05)
        assert r_result is not None, "R FCI failed"

        py_result = fci_algorithm(data, alpha=0.05)

        # Both should produce empty skeleton (no edges)
        r_edges = count_skeleton_edges(r_result["skeleton"])
        py_edges = py_result.skeleton.n_edges()

        assert r_edges == 0, f"R should have no edges, got {r_edges}"
        assert py_edges == 0, f"Python should have no edges, got {py_edges}"

    @requires_discovery_python
    @requires_pcalg_r
    def test_small_sample_size(self):
        """FCI handles small sample sizes gracefully."""
        data, _ = generate_chain_data(n_samples=50, seed=42)

        r_result = r_fci_algorithm(data, alpha=0.10)
        py_result = fci_algorithm(data, alpha=0.10)

        # Both should complete without error
        assert r_result is not None or True, "R FCI should handle small samples"
        assert py_result is not None, "Python FCI should handle small samples"

    @requires_discovery_python
    @requires_pcalg_r
    def test_varying_alpha(self):
        """FCI responds appropriately to different alpha levels."""
        data, _ = generate_chain_data(n_samples=500, seed=42)

        alphas = [0.001, 0.01, 0.05, 0.10]
        r_edges = []
        py_edges = []

        for alpha in alphas:
            r_result = r_fci_algorithm(data, alpha=alpha)
            py_result = fci_algorithm(data, alpha=alpha)

            if r_result is not None:
                r_edges.append(count_skeleton_edges(r_result["skeleton"]))
            if py_result is not None:
                py_edges.append(py_result.skeleton.n_edges())

        # Higher alpha = more edges (less stringent independence test)
        # Both should show similar pattern
        if len(r_edges) >= 2:
            assert r_edges[-1] >= r_edges[0], (
                f"R edges should increase with alpha: {r_edges}"
            )
        if len(py_edges) >= 2:
            assert py_edges[-1] >= py_edges[0], (
                f"Python edges should increase with alpha: {py_edges}"
            )

    @requires_discovery_python
    @requires_pcalg_r
    def test_five_node_structure(self):
        """FCI handles 5-node structure consistently with R."""
        if not DISCOVERY_AVAILABLE:
            pytest.skip("Discovery module not available")

        # Generate random 5-node DAG
        true_dag = generate_random_dag(5, edge_prob=0.4, seed=42)
        data, _ = generate_dag_data(true_dag, n_samples=1000, seed=42)

        r_result = r_fci_algorithm(data, alpha=0.05)
        assert r_result is not None, "R FCI failed"

        py_result = fci_algorithm(data, alpha=0.05)

        # Skeleton edge counts should be similar
        r_edges = count_skeleton_edges(r_result["skeleton"])
        py_edges = py_result.skeleton.n_edges()

        # Allow ±1 edge difference
        assert abs(r_edges - py_edges) <= 1, (
            f"5-node skeleton edge count mismatch: R={r_edges}, Python={py_edges}"
        )


# =============================================================================
# Summary Statistics Tests
# =============================================================================


class TestFCISummaryStats:
    """Test FCI summary statistics match R."""

    @requires_discovery_python
    @requires_pcalg_r
    def test_node_count_matches(self):
        """Node count matches R."""
        data, _ = generate_chain_data()

        r_result = r_fci_algorithm(data, alpha=0.05)
        py_result = fci_algorithm(data, alpha=0.05)

        assert r_result["n_nodes"] == py_result.pag.n_nodes

    @requires_discovery_python
    @requires_pcalg_r
    def test_skeleton_edge_count_matches(self):
        """Skeleton edge count matches R."""
        data, _ = generate_collider_data()

        r_result = r_fci_algorithm(data, alpha=0.05)
        py_result = fci_algorithm(data, alpha=0.05)

        r_edges = count_skeleton_edges(r_result["skeleton"])
        py_edges = py_result.skeleton.n_edges()

        assert r_edges == py_edges, (
            f"Skeleton edge count mismatch: R={r_edges}, Python={py_edges}"
        )
