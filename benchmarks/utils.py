"""Benchmark utility functions for timing and memory measurement.

Provides core infrastructure for:
- Timing functions with warmup and repetitions
- Memory profiling using tracemalloc
- Result formatting and serialization
"""

from __future__ import annotations

import gc
import json
import time
import tracemalloc
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Dict, Union
import numpy as np


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run.

    Attributes
    ----------
    method_name : str
        Name of the benchmarked method.
    family : str
        Method family (e.g., "rct", "did").
    sample_size : int
        Sample size used for benchmark.
    median_time_ms : float
        Median execution time in milliseconds.
    min_time_ms : float
        Minimum execution time.
    max_time_ms : float
        Maximum execution time.
    std_time_ms : float
        Standard deviation of execution times.
    memory_peak_kb : float
        Peak memory usage in kilobytes.
    n_repetitions : int
        Number of repetitions used.
    language : str
        "python" or "julia".
    timestamp : str
        ISO format timestamp of when benchmark was run.
    """

    method_name: str
    family: str
    sample_size: int
    median_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    memory_peak_kb: float
    n_repetitions: int
    language: str = "python"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkResult":
        """Create from dictionary."""
        return cls(**d)

    @property
    def speed_category(self) -> str:
        """Categorize speed: fast, medium, slow, very_slow."""
        if self.median_time_ms < 10:
            return "fast"
        elif self.median_time_ms < 100:
            return "medium"
        elif self.median_time_ms < 1000:
            return "slow"
        else:
            return "very_slow"


def time_function(
    func: Callable,
    *args,
    n_repetitions: int = 10,
    n_warmup: int = 1,
    **kwargs,
) -> Dict[str, float]:
    """Time a function with warmup and multiple repetitions.

    Parameters
    ----------
    func : Callable
        Function to time.
    *args
        Positional arguments for func.
    n_repetitions : int
        Number of timed repetitions (default 10).
    n_warmup : int
        Number of warmup runs to exclude (default 1).
    **kwargs
        Keyword arguments for func.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: median_ms, min_ms, max_ms, std_ms, times_ms.

    Examples
    --------
    >>> def slow_func(n):
    ...     return sum(range(n))
    >>> result = time_function(slow_func, 1000000, n_repetitions=5)
    >>> result['median_ms'] > 0
    True
    """
    # Warmup runs (exclude from timing)
    for _ in range(n_warmup):
        gc.collect()
        _ = func(*args, **kwargs)

    # Timed runs
    times_ms = []
    for _ in range(n_repetitions):
        gc.collect()
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)

    times_array = np.array(times_ms)

    return {
        "median_ms": float(np.median(times_array)),
        "min_ms": float(np.min(times_array)),
        "max_ms": float(np.max(times_array)),
        "std_ms": float(np.std(times_array)),
        "times_ms": times_ms,
    }


def measure_memory(
    func: Callable,
    *args,
    **kwargs,
) -> Dict[str, float]:
    """Measure peak memory usage of a function.

    Uses tracemalloc to measure Python memory allocations.

    Parameters
    ----------
    func : Callable
        Function to measure.
    *args
        Positional arguments for func.
    **kwargs
        Keyword arguments for func.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: peak_kb, current_kb.

    Examples
    --------
    >>> def allocate():
    ...     return [0] * 10000
    >>> result = measure_memory(allocate)
    >>> result['peak_kb'] > 0
    True
    """
    gc.collect()
    tracemalloc.start()

    _ = func(*args, **kwargs)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "peak_kb": peak / 1024,
        "current_kb": current / 1024,
    }


def benchmark_method(
    func: Callable,
    method_name: str,
    family: str,
    sample_size: int,
    *args,
    n_repetitions: int = 10,
    n_warmup: int = 1,
    language: str = "python",
    **kwargs,
) -> BenchmarkResult:
    """Full benchmark of a method: timing + memory.

    Parameters
    ----------
    func : Callable
        Function to benchmark.
    method_name : str
        Name of the method.
    family : str
        Method family name.
    sample_size : int
        Sample size being benchmarked.
    *args
        Positional arguments for func.
    n_repetitions : int
        Number of timing repetitions.
    n_warmup : int
        Number of warmup runs.
    language : str
        "python" or "julia".
    **kwargs
        Keyword arguments for func.

    Returns
    -------
    BenchmarkResult
        Complete benchmark result.
    """
    # Time the function
    timing = time_function(
        func, *args,
        n_repetitions=n_repetitions,
        n_warmup=n_warmup,
        **kwargs,
    )

    # Measure memory
    memory = measure_memory(func, *args, **kwargs)

    return BenchmarkResult(
        method_name=method_name,
        family=family,
        sample_size=sample_size,
        median_time_ms=timing["median_ms"],
        min_time_ms=timing["min_ms"],
        max_time_ms=timing["max_ms"],
        std_time_ms=timing["std_ms"],
        memory_peak_kb=memory["peak_kb"],
        n_repetitions=n_repetitions,
        language=language,
    )


def format_results_table(
    results: List[BenchmarkResult],
    sort_by: str = "median_time_ms",
) -> str:
    """Format benchmark results as a table string.

    Parameters
    ----------
    results : List[BenchmarkResult]
        List of benchmark results.
    sort_by : str
        Column to sort by (default "median_time_ms").

    Returns
    -------
    str
        Formatted table string.
    """
    if not results:
        return "No results to display."

    # Sort results
    sorted_results = sorted(results, key=lambda r: getattr(r, sort_by))

    # Header
    lines = [
        f"{'Method':<25} {'Family':<12} {'N':>8} {'Median(ms)':>12} "
        f"{'Min(ms)':>10} {'Max(ms)':>10} {'Memory(KB)':>12} {'Speed':>10}",
        "=" * 110,
    ]

    # Speed category emoji
    speed_emoji = {
        "fast": "🟢",
        "medium": "🟡",
        "slow": "🟠",
        "very_slow": "🔴",
    }

    for r in sorted_results:
        emoji = speed_emoji.get(r.speed_category, "⚪")
        lines.append(
            f"{r.method_name:<25} {r.family:<12} {r.sample_size:>8} "
            f"{r.median_time_ms:>12.2f} {r.min_time_ms:>10.2f} "
            f"{r.max_time_ms:>10.2f} {r.memory_peak_kb:>12.1f} {emoji:>10}"
        )

    return "\n".join(lines)


def save_results_json(
    results: List[BenchmarkResult],
    path: Union[str, Path],
    include_metadata: bool = True,
) -> None:
    """Save benchmark results to JSON file.

    Parameters
    ----------
    results : List[BenchmarkResult]
        List of benchmark results.
    path : Union[str, Path]
        Output file path.
    include_metadata : bool
        Include timestamp and system info (default True).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "results": [r.to_dict() for r in results],
    }

    if include_metadata:
        import platform
        import sys

        data["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_results_json(path: Union[str, Path]) -> List[BenchmarkResult]:
    """Load benchmark results from JSON file.

    Parameters
    ----------
    path : Union[str, Path]
        Input file path.

    Returns
    -------
    List[BenchmarkResult]
        List of benchmark results.

    Raises
    ------
    FileNotFoundError
        If file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    return [BenchmarkResult.from_dict(r) for r in data["results"]]


def compare_results(
    current: List[BenchmarkResult],
    baseline: List[BenchmarkResult],
    tolerance_func: Optional[Callable[[float], float]] = None,
) -> Dict[str, Any]:
    """Compare current results against baseline for regression detection.

    Parameters
    ----------
    current : List[BenchmarkResult]
        Current benchmark results.
    baseline : List[BenchmarkResult]
        Baseline results to compare against.
    tolerance_func : Optional[Callable[[float], float]]
        Function to compute tolerance from median_ms.
        If None, uses default tolerance bands.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - regressions: List of method names that regressed
        - improvements: List of method names that improved
        - comparison: Dict of method -> {current, baseline, change_pct}
    """
    from benchmarks.config import get_tolerance

    if tolerance_func is None:
        tolerance_func = get_tolerance

    # Index baseline by (method_name, sample_size)
    baseline_index = {
        (r.method_name, r.sample_size): r for r in baseline
    }

    regressions = []
    improvements = []
    comparison = {}

    for curr in current:
        key = (curr.method_name, curr.sample_size)
        if key not in baseline_index:
            continue

        base = baseline_index[key]
        change_pct = (curr.median_time_ms - base.median_time_ms) / base.median_time_ms
        tolerance = tolerance_func(base.median_time_ms)

        comparison[f"{curr.method_name}@n={curr.sample_size}"] = {
            "current_ms": curr.median_time_ms,
            "baseline_ms": base.median_time_ms,
            "change_pct": change_pct * 100,
            "tolerance_pct": tolerance * 100,
            "status": "ok" if abs(change_pct) <= tolerance else (
                "regression" if change_pct > tolerance else "improvement"
            ),
        }

        if change_pct > tolerance:
            regressions.append(f"{curr.method_name}@n={curr.sample_size}")
        elif change_pct < -tolerance:
            improvements.append(f"{curr.method_name}@n={curr.sample_size}")

    return {
        "regressions": regressions,
        "improvements": improvements,
        "comparison": comparison,
    }
