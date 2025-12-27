"""Cross-language benchmark runner for Python vs Julia comparison.

This module provides the infrastructure to run identical benchmarks
in both Python and Julia, computing speedup factors.

Key Design Decisions:
1. Data generated in Python, passed to both languages (identical inputs)
2. Warmup runs excluded (accounts for Julia JIT)
3. 10 repetitions, report median for stability
4. Graceful degradation when Julia unavailable
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from benchmarks.dgp import DGP_REGISTRY
from benchmarks.utils import time_function


# Julia availability check
_JULIA_AVAILABLE: Optional[bool] = None


def is_julia_available() -> bool:
    """Check if Julia is available via juliacall.

    Returns
    -------
    bool
        True if Julia can be loaded.
    """
    global _JULIA_AVAILABLE

    if _JULIA_AVAILABLE is not None:
        return _JULIA_AVAILABLE

    try:
        from tests.validation.cross_language.julia_interface import is_julia_available as jl_check
        _JULIA_AVAILABLE = jl_check()
    except ImportError:
        _JULIA_AVAILABLE = False

    return _JULIA_AVAILABLE


@dataclass
class CrossLanguageResult:
    """Result from cross-language benchmark comparison.

    Attributes
    ----------
    method_name : str
        Name of the method benchmarked.
    family : str
        Method family (rct, iv, did, etc.).
    sample_size : int
        Sample size used.
    python_time_ms : float
        Median Python execution time in milliseconds.
    julia_time_ms : float
        Median Julia execution time in milliseconds (NaN if unavailable).
    speedup : float
        Python time / Julia time (>1 means Julia faster).
    python_memory_kb : float
        Python peak memory in KB.
    julia_memory_kb : float
        Julia memory in KB (NaN if unavailable).
    n_repetitions : int
        Number of timing repetitions.
    julia_available : bool
        Whether Julia was available for this benchmark.
    """

    method_name: str
    family: str
    sample_size: int
    python_time_ms: float
    julia_time_ms: float
    speedup: float
    python_memory_kb: float
    julia_memory_kb: float
    n_repetitions: int
    julia_available: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "method_name": self.method_name,
            "family": self.family,
            "sample_size": self.sample_size,
            "python_time_ms": self.python_time_ms,
            "julia_time_ms": self.julia_time_ms if not np.isnan(self.julia_time_ms) else None,
            "speedup": self.speedup if not np.isnan(self.speedup) else None,
            "python_memory_kb": self.python_memory_kb,
            "julia_memory_kb": self.julia_memory_kb if not np.isnan(self.julia_memory_kb) else None,
            "n_repetitions": self.n_repetitions,
            "julia_available": self.julia_available,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CrossLanguageResult":
        """Create from dictionary."""
        return cls(
            method_name=d["method_name"],
            family=d["family"],
            sample_size=d["sample_size"],
            python_time_ms=d["python_time_ms"],
            julia_time_ms=d["julia_time_ms"] if d["julia_time_ms"] is not None else np.nan,
            speedup=d["speedup"] if d["speedup"] is not None else np.nan,
            python_memory_kb=d["python_memory_kb"],
            julia_memory_kb=d["julia_memory_kb"] if d["julia_memory_kb"] is not None else np.nan,
            n_repetitions=d["n_repetitions"],
            julia_available=d["julia_available"],
        )


@dataclass
class CrossLanguageBenchmarkRunner:
    """Run identical benchmarks in Python and Julia.

    Parameters
    ----------
    sample_sizes : List[int]
        Sample sizes to benchmark.
    n_repetitions : int
        Number of timing repetitions per benchmark.
    n_warmup : int
        Number of warmup runs (excluded from timing).
    seed : int
        Random seed for reproducibility.

    Example
    -------
    >>> runner = CrossLanguageBenchmarkRunner(sample_sizes=[100, 1000])
    >>> results = runner.benchmark_method("rct", "simple_ate")
    >>> print(f"Speedup: {results[0].speedup:.1f}x")
    """

    sample_sizes: List[int] = field(default_factory=lambda: [100, 1000, 5000])
    n_repetitions: int = 10
    n_warmup: int = 2
    seed: int = 42
    results: List[CrossLanguageResult] = field(default_factory=list)

    def benchmark_method(
        self,
        family: str,
        method_name: str,
        python_func: Callable,
        julia_func: Optional[Callable] = None,
        data_kwargs: Optional[Dict] = None,
    ) -> List[CrossLanguageResult]:
        """Benchmark a single method across sample sizes.

        Parameters
        ----------
        family : str
            Method family name.
        method_name : str
            Method name.
        python_func : Callable
            Python benchmark function.
        julia_func : Optional[Callable]
            Julia wrapper function (from julia_benchmarks.py).
        data_kwargs : Optional[Dict]
            Additional kwargs for DGP generation.

        Returns
        -------
        List[CrossLanguageResult]
            Results for each sample size.
        """
        results = []
        julia_available = is_julia_available() and julia_func is not None

        for n in self.sample_sizes:
            # Generate data once (Python side)
            dgp_func = DGP_REGISTRY.get(family)
            if dgp_func is None:
                raise ValueError(f"No DGP for family '{family}'")

            kwargs = {"n": n, "seed": self.seed}
            if data_kwargs:
                kwargs.update(data_kwargs)

            data = dgp_func(**kwargs)

            # Time Python
            py_timing = self._time_python(python_func, data)

            # Time Julia (if available)
            if julia_available:
                jl_timing = self._time_julia(julia_func, data)
                julia_time_ms = jl_timing["median_ms"]
                julia_memory_kb = jl_timing.get("memory_kb", np.nan)
                speedup = py_timing["median_ms"] / julia_time_ms if julia_time_ms > 0 else np.nan
            else:
                julia_time_ms = np.nan
                julia_memory_kb = np.nan
                speedup = np.nan

            result = CrossLanguageResult(
                method_name=method_name,
                family=family,
                sample_size=n,
                python_time_ms=py_timing["median_ms"],
                julia_time_ms=julia_time_ms,
                speedup=speedup,
                python_memory_kb=py_timing.get("memory_kb", 0),
                julia_memory_kb=julia_memory_kb,
                n_repetitions=self.n_repetitions,
                julia_available=julia_available,
            )
            results.append(result)
            self.results.append(result)

        return results

    def _time_python(self, func: Callable, data: Dict) -> Dict[str, float]:
        """Time Python function execution.

        Parameters
        ----------
        func : Callable
            Function to time.
        data : Dict
            Data to pass to function.

        Returns
        -------
        Dict[str, float]
            Timing results with median_ms key.
        """
        times = []

        # Warmup
        for _ in range(self.n_warmup):
            func(**data)

        # Timed runs
        for _ in range(self.n_repetitions):
            start = time.perf_counter()
            func(**data)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        return {
            "median_ms": float(np.median(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "std_ms": float(np.std(times)),
            "times_ms": times,
        }

    def _time_julia(self, func: Callable, data: Dict) -> Dict[str, float]:
        """Time Julia function execution via wrapper.

        Parameters
        ----------
        func : Callable
            Julia wrapper function.
        data : Dict
            Data to pass to function.

        Returns
        -------
        Dict[str, float]
            Timing results with median_ms key.
        """
        times = []

        # Warmup (important for Julia JIT)
        for _ in range(self.n_warmup):
            try:
                func(**data)
            except Exception:
                pass

        # Timed runs
        for _ in range(self.n_repetitions):
            start = time.perf_counter()
            try:
                func(**data)
            except Exception as e:
                # Return NaN on error
                return {"median_ms": np.nan, "error": str(e)}
            end = time.perf_counter()
            times.append((end - start) * 1000)

        return {
            "median_ms": float(np.median(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "std_ms": float(np.std(times)),
            "times_ms": times,
        }

    def summary(self) -> str:
        """Generate summary table of results.

        Returns
        -------
        str
            Formatted summary table.
        """
        if not self.results:
            return "No results yet."

        lines = [
            "=" * 80,
            "CROSS-LANGUAGE BENCHMARK RESULTS",
            "=" * 80,
            f"{'Family':<15} {'Method':<25} {'N':>8} {'Python(ms)':>12} {'Julia(ms)':>12} {'Speedup':>10}",
            "-" * 80,
        ]

        for r in sorted(self.results, key=lambda x: (x.family, x.method_name, x.sample_size)):
            julia_str = f"{r.julia_time_ms:>12.2f}" if not np.isnan(r.julia_time_ms) else "         N/A"
            speedup_str = f"{r.speedup:>9.1f}x" if not np.isnan(r.speedup) else "       N/A"

            lines.append(
                f"{r.family:<15} {r.method_name:<25} {r.sample_size:>8} "
                f"{r.python_time_ms:>12.2f} {julia_str} {speedup_str}"
            )

        lines.append("=" * 80)

        # Summary statistics
        valid_speedups = [r.speedup for r in self.results if not np.isnan(r.speedup)]
        if valid_speedups:
            lines.append(f"\nSpeedup Statistics (n={len(valid_speedups)} comparisons):")
            lines.append(f"  Mean:   {np.mean(valid_speedups):.1f}x")
            lines.append(f"  Median: {np.median(valid_speedups):.1f}x")
            lines.append(f"  Min:    {np.min(valid_speedups):.1f}x")
            lines.append(f"  Max:    {np.max(valid_speedups):.1f}x")

        return "\n".join(lines)

    def to_json(self, filepath: Optional[Path] = None) -> str:
        """Export results to JSON.

        Parameters
        ----------
        filepath : Optional[Path]
            If provided, write to file.

        Returns
        -------
        str
            JSON string.
        """
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "sample_sizes": self.sample_sizes,
                "n_repetitions": self.n_repetitions,
                "seed": self.seed,
                "julia_available": is_julia_available(),
            },
            "results": [r.to_dict() for r in self.results],
        }

        json_str = json.dumps(data, indent=2)

        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_json(cls, filepath: Path) -> "CrossLanguageBenchmarkRunner":
        """Load results from JSON.

        Parameters
        ----------
        filepath : Path
            Path to JSON file.

        Returns
        -------
        CrossLanguageBenchmarkRunner
            Runner with loaded results.
        """
        with open(filepath) as f:
            data = json.load(f)

        runner = cls(
            sample_sizes=data["metadata"]["sample_sizes"],
            n_repetitions=data["metadata"]["n_repetitions"],
            seed=data["metadata"]["seed"],
        )

        runner.results = [CrossLanguageResult.from_dict(r) for r in data["results"]]

        return runner
