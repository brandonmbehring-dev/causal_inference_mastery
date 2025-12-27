"""Cross-language benchmark comparison infrastructure.

Session 132: Python vs Julia performance comparison.

This package provides tools to benchmark the same causal inference
methods in both Python and Julia, computing speedup factors.

Main Components
---------------
CrossLanguageBenchmarkRunner
    Orchestrates benchmarks across both languages
CrossLanguageResult
    Container for comparison results

Example
-------
>>> from benchmarks.cross_language import CrossLanguageBenchmarkRunner
>>> runner = CrossLanguageBenchmarkRunner()
>>> results = runner.benchmark_family("rct", sample_sizes=[100, 1000])
>>> print(results.summary())
"""

from .runner import (
    CrossLanguageBenchmarkRunner,
    CrossLanguageResult,
    is_julia_available,
)

__all__ = [
    "CrossLanguageBenchmarkRunner",
    "CrossLanguageResult",
    "is_julia_available",
]
