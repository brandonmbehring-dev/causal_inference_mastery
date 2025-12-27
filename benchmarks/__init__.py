"""Performance benchmarking infrastructure for causal inference methods.

This package provides comprehensive benchmarking for all 26 method families,
supporting:
1. Interview Portfolio - Polished visualizations
2. Practical Guidance - Sample size recommendations
3. Regression Testing - CI-integrated timing assertions
4. Cross-Language Comparison - Python vs Julia

Usage:
    # Run all benchmarks
    python -m benchmarks.runner --all

    # Run specific family
    python -m benchmarks.runner --family rct

    # Run at specific sample size
    python -m benchmarks.runner --family did --n 1000

Sessions 130-131 Implementation
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Brandon Behring"

# Core infrastructure
from benchmarks.config import BenchmarkConfig, DEFAULT_CONFIG
from benchmarks.utils import (
    BenchmarkResult,
    time_function,
    measure_memory,
    format_results_table,
    save_results_json,
    load_results_json,
)
from benchmarks.runner import BenchmarkRunner

__all__ = [
    # Version
    "__version__",
    # Config
    "BenchmarkConfig",
    "DEFAULT_CONFIG",
    # Utils
    "BenchmarkResult",
    "time_function",
    "measure_memory",
    "format_results_table",
    "save_results_json",
    "load_results_json",
    # Runner
    "BenchmarkRunner",
]
