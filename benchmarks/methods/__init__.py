"""Method-specific benchmark functions.

Each module in this package provides benchmark_* functions for a method family.
These functions generate data, run the method, and return BenchmarkResult.

Session 131: Complete benchmark coverage for 22 method families.
"""

from __future__ import annotations

from typing import List, Dict, Callable

from benchmarks.utils import BenchmarkResult

# All implemented benchmark modules
IMPLEMENTED_FAMILIES = [
    # Session 130: Core methods
    "rct",
    "observational",
    "psm",
    "did",
    "iv",
    "rdd",
    # Session 131: Advanced methods
    "scm",
    "cate",
    "rkd",
    "panel",
    "sensitivity",
    "bayesian",
    "principal_strat",
    "bounds",
    "qte",
    "bunching",
    "selection",
    "mte",
    "mediation",
    "control_function",
    "shift_share",
    "dtr",
]


def get_implemented_families() -> List[str]:
    """Get list of families with implemented benchmarks."""
    return IMPLEMENTED_FAMILIES.copy()


def get_all_benchmarks() -> Dict[str, Dict[str, Callable]]:
    """Get all benchmark functions organized by family.

    Returns
    -------
    Dict[str, Dict[str, Callable]]
        Nested dict: family -> {method_name: benchmark_function}
    """
    all_benchmarks = {}

    for family in IMPLEMENTED_FAMILIES:
        try:
            module = __import__(
                f"benchmarks.methods.{family}",
                fromlist=["BENCHMARKS"]
            )
            all_benchmarks[family] = getattr(module, "BENCHMARKS", {})
        except ImportError:
            # Module not yet implemented
            all_benchmarks[family] = {}

    return all_benchmarks


def get_family_benchmarks(family: str) -> Dict[str, Callable]:
    """Get benchmark functions for a specific family.

    Parameters
    ----------
    family : str
        Method family name.

    Returns
    -------
    Dict[str, Callable]
        Dict of method_name -> benchmark_function.

    Raises
    ------
    ValueError
        If family not found.
    """
    if family not in IMPLEMENTED_FAMILIES:
        raise ValueError(
            f"Unknown family '{family}'. "
            f"Available: {IMPLEMENTED_FAMILIES}"
        )

    try:
        module = __import__(
            f"benchmarks.methods.{family}",
            fromlist=["BENCHMARKS"]
        )
        return getattr(module, "BENCHMARKS", {})
    except ImportError as e:
        raise ImportError(f"Could not load benchmarks for '{family}': {e}")


def list_all_methods() -> List[str]:
    """List all benchmark method names across all families.

    Returns
    -------
    List[str]
        Flat list of method names.
    """
    all_methods = []
    for family, benchmarks in get_all_benchmarks().items():
        all_methods.extend(benchmarks.keys())
    return all_methods


def count_benchmarks() -> Dict[str, int]:
    """Count benchmarks per family.

    Returns
    -------
    Dict[str, int]
        Family -> count mapping.
    """
    return {
        family: len(benchmarks)
        for family, benchmarks in get_all_benchmarks().items()
    }
