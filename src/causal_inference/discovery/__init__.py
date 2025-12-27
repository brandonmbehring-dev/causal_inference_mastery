"""Causal Discovery Module.

Session 133: Learn causal structure from observational data.

Two complementary approaches:
1. **PC Algorithm** (constraint-based): Tests conditional independence,
   outputs CPDAG (Markov equivalence class)
2. **LiNGAM** (functional): Exploits non-Gaussianity via ICA,
   outputs unique DAG

Algorithms
----------
PC Algorithm
    - pc_algorithm : Full PC with skeleton + orientation
    - pc_skeleton : Learn undirected skeleton
    - pc_orient : Orient skeleton to CPDAG
    - pc_conservative : Conservative v-structure orientation
    - pc_majority : Majority-rule v-structure orientation

LiNGAM
    - ica_lingam : ICA-based causal discovery
    - direct_lingam : DirectLiNGAM (faster)
    - bootstrap_lingam : Bootstrap confidence estimation

Types
-----
Graph : Undirected graph (skeleton)
DAG : Directed acyclic graph
CPDAG : Completed partially directed graph (equivalence class)
PCResult : PC algorithm output
LiNGAMResult : LiNGAM output

Utilities
---------
generate_random_dag : Generate random DAG for testing
generate_dag_data : Generate data from linear SCM
dag_to_cpdag : Convert DAG to its CPDAG
skeleton_f1 : Skeleton recovery metrics
compute_shd : Structural Hamming Distance
is_markov_equivalent : Check Markov equivalence

Independence Tests
------------------
fisher_z_test : Fisher's Z for Gaussian data
partial_correlation_test : Partial correlation CI test
g_squared_test : G² for categorical data
kernel_ci_test : Kernel-based CI test

Example
-------
>>> from causal_inference.discovery import pc_algorithm, direct_lingam
>>> from causal_inference.discovery import generate_random_dag, generate_dag_data
>>>
>>> # Generate ground truth
>>> true_dag = generate_random_dag(5, edge_prob=0.3, seed=42)
>>> data, B = generate_dag_data(true_dag, n_samples=1000, seed=42)
>>>
>>> # PC Algorithm (Gaussian data)
>>> pc_result = pc_algorithm(data, alpha=0.01)
>>> print(f"SHD: {pc_result.structural_hamming_distance(true_dag)}")
>>>
>>> # LiNGAM (non-Gaussian data)
>>> data_ng, _ = generate_dag_data(true_dag, n_samples=1000,
...                                 noise_type="laplace", seed=42)
>>> lingam_result = direct_lingam(data_ng)
>>> print(f"Order accuracy: {lingam_result.causal_order_accuracy(true_dag.topological_order()):.2f}")

References
----------
- Spirtes, Glymour, Scheines (2000). Causation, Prediction, and Search.
- Shimizu et al. (2006). A linear non-Gaussian acyclic model.
- Shimizu et al. (2011). DirectLiNGAM: A direct method.
"""

from .types import (
    CITestResult,
    CPDAG,
    DAG,
    EdgeMark,
    FCIResult,
    Graph,
    LiNGAMResult,
    PAG,
    PAGEdge,
    PCResult,
)

from .independence_tests import (
    ci_test,
    fisher_z_test,
    g_squared_test,
    kernel_ci_test,
    partial_correlation,
    partial_correlation_test,
)

from .pc_algorithm import (
    pc_algorithm,
    pc_conservative,
    pc_majority,
    pc_orient,
    pc_skeleton,
)

from .lingam import (
    bootstrap_lingam,
    check_non_gaussianity,
    direct_lingam,
    ica_lingam,
)

from .fci_algorithm import (
    fci_algorithm,
    fci_orient,
)

from .utils import (
    compute_shd,
    dag_to_cpdag,
    generate_dag_data,
    generate_random_dag,
    is_markov_equivalent,
    orientation_accuracy,
    skeleton_f1,
)

__all__ = [
    # Types
    "Graph",
    "DAG",
    "CPDAG",
    "PAG",
    "PAGEdge",
    "EdgeMark",
    "PCResult",
    "LiNGAMResult",
    "FCIResult",
    "CITestResult",
    # PC Algorithm
    "pc_algorithm",
    "pc_skeleton",
    "pc_orient",
    "pc_conservative",
    "pc_majority",
    # FCI Algorithm
    "fci_algorithm",
    "fci_orient",
    # LiNGAM
    "ica_lingam",
    "direct_lingam",
    "bootstrap_lingam",
    "check_non_gaussianity",
    # Independence Tests
    "ci_test",
    "fisher_z_test",
    "partial_correlation",
    "partial_correlation_test",
    "g_squared_test",
    "kernel_ci_test",
    # Utilities
    "generate_random_dag",
    "generate_dag_data",
    "dag_to_cpdag",
    "skeleton_f1",
    "orientation_accuracy",
    "compute_shd",
    "is_markov_equivalent",
]
