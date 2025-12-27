"""Benchmark runner with CLI interface.

Provides BenchmarkRunner class for executing benchmarks programmatically
and CLI for running from command line.

Usage:
    # Run all benchmarks
    python -m benchmarks.runner --all

    # Run specific family
    python -m benchmarks.runner --family rct

    # Run at specific sample sizes
    python -m benchmarks.runner --family did --sizes 100 1000

    # Quick CI run
    python -m benchmarks.runner --ci
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import importlib

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from benchmarks.config import (
    BenchmarkConfig,
    DEFAULT_CONFIG,
    CI_CONFIG,
    get_all_families,
    get_family_methods,
)
from benchmarks.utils import (
    BenchmarkResult,
    benchmark_method,
    format_results_table,
    save_results_json,
)


class BenchmarkRunner:
    """Execute benchmarks for causal inference methods.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration.

    Examples
    --------
    >>> runner = BenchmarkRunner()
    >>> results = runner.run_family("rct", sample_sizes=[100, 500])
    >>> print(format_results_table(results))
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.results: List[BenchmarkResult] = []

    def _progress(self, iterable, desc: str = ""):
        """Wrap iterable with progress bar if available and verbose."""
        if self.config.verbose and TQDM_AVAILABLE:
            return tqdm(iterable, desc=desc)
        return iterable

    def _log(self, message: str):
        """Print message if verbose mode."""
        if self.config.verbose:
            print(message)

    def run_single(
        self,
        method_name: str,
        family: str,
        sample_sizes: Optional[List[int]] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmark for a single method across sample sizes.

        Parameters
        ----------
        method_name : str
            Name of the method to benchmark.
        family : str
            Method family name.
        sample_sizes : Optional[List[int]]
            Sample sizes to test. If None, uses config default.

        Returns
        -------
        List[BenchmarkResult]
            Results for each sample size.
        """
        sample_sizes = sample_sizes or self.config.sample_sizes
        results = []

        # Import the benchmark function
        try:
            benchmark_module = importlib.import_module(
                f"benchmarks.methods.{family}"
            )
            benchmark_func = getattr(benchmark_module, f"benchmark_{method_name}")
        except (ImportError, AttributeError) as e:
            self._log(f"  Skipping {method_name}: {e}")
            return results

        for n in sample_sizes:
            try:
                result = benchmark_func(
                    n=n,
                    seed=self.config.seed,
                    n_repetitions=self.config.n_repetitions,
                    n_warmup=self.config.n_warmup,
                )
                results.append(result)
                self._log(
                    f"  {method_name} n={n}: {result.median_time_ms:.2f}ms"
                )
            except Exception as e:
                self._log(f"  {method_name} n={n}: ERROR - {e}")

        return results

    def run_family(
        self,
        family: str,
        sample_sizes: Optional[List[int]] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmarks for all methods in a family.

        Parameters
        ----------
        family : str
            Family name (e.g., "rct", "did").
        sample_sizes : Optional[List[int]]
            Sample sizes to test.

        Returns
        -------
        List[BenchmarkResult]
            Results for all methods in family.
        """
        self._log(f"\n{'='*60}")
        self._log(f"Benchmarking family: {family.upper()}")
        self._log(f"{'='*60}")

        results = []
        methods = get_family_methods(family)

        for method_name, _, _ in self._progress(methods, desc=f"{family}"):
            method_results = self.run_single(
                method_name=method_name,
                family=family,
                sample_sizes=sample_sizes,
            )
            results.extend(method_results)

        self.results.extend(results)
        return results

    def run_all(
        self,
        families: Optional[List[str]] = None,
        sample_sizes: Optional[List[int]] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmarks for all families.

        Parameters
        ----------
        families : Optional[List[str]]
            Families to benchmark. If None, runs all.
        sample_sizes : Optional[List[int]]
            Sample sizes to test.

        Returns
        -------
        List[BenchmarkResult]
            All benchmark results.
        """
        families = families or get_all_families()
        results = []

        self._log(f"\nRunning benchmarks for {len(families)} families")
        self._log(f"Sample sizes: {sample_sizes or self.config.sample_sizes}")
        self._log(f"Repetitions: {self.config.n_repetitions}")

        for family in families:
            try:
                family_results = self.run_family(
                    family=family,
                    sample_sizes=sample_sizes,
                )
                results.extend(family_results)
            except Exception as e:
                self._log(f"Error in family {family}: {e}")

        self.results = results
        return results

    def to_dataframe(self):
        """Convert results to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with all results.

        Raises
        ------
        ImportError
            If pandas not available.
        """
        import pandas as pd

        return pd.DataFrame([r.to_dict() for r in self.results])

    def save(self, path: Optional[str] = None) -> str:
        """Save results to JSON file.

        Parameters
        ----------
        path : Optional[str]
            Output path. If None, generates timestamped filename.

        Returns
        -------
        str
            Path where results were saved.
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{self.config.output_dir}/benchmark_{timestamp}.json"

        save_results_json(self.results, path)
        self._log(f"\nResults saved to: {path}")
        return path

    def print_summary(self):
        """Print formatted summary of results."""
        if not self.results:
            print("No results to display.")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)
        print(format_results_table(self.results))

        # Statistics
        times = [r.median_time_ms for r in self.results]
        print(f"\n{'='*80}")
        print(f"Total benchmarks: {len(self.results)}")
        print(f"Fastest: {min(times):.2f}ms")
        print(f"Slowest: {max(times):.2f}ms")

        # By category
        categories = {"fast": 0, "medium": 0, "slow": 0, "very_slow": 0}
        for r in self.results:
            categories[r.speed_category] += 1
        print(f"\nSpeed distribution:")
        print(f"  🟢 Fast (<10ms):     {categories['fast']}")
        print(f"  🟡 Medium (10-100ms): {categories['medium']}")
        print(f"  🟠 Slow (100ms-1s):   {categories['slow']}")
        print(f"  🔴 Very slow (>1s):   {categories['very_slow']}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark causal inference methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks.runner --all
  python -m benchmarks.runner --family rct --sizes 100 1000
  python -m benchmarks.runner --ci
  python -m benchmarks.runner --families rct did iv --output results.json
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--family",
        type=str,
        help="Run single family (e.g., rct, did, iv)",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        help="Run multiple families",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        help="Sample sizes to test (default: 100 500 1000 5000 10000)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        help="Number of timing repetitions (default: 10)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Use CI configuration (fast, smaller samples)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path for results JSON",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available families and methods",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("Available method families:")
        for family in get_all_families():
            methods = get_family_methods(family)
            print(f"\n{family.upper()}:")
            for name, _, _ in methods:
                print(f"  - {name}")
        return

    # Configure
    if args.ci:
        config = CI_CONFIG
    else:
        config = BenchmarkConfig(
            sample_sizes=args.sizes or DEFAULT_CONFIG.sample_sizes,
            n_repetitions=args.reps,
            verbose=not args.quiet,
        )

    runner = BenchmarkRunner(config)

    # Run benchmarks
    if args.all:
        runner.run_all()
    elif args.family:
        runner.run_family(args.family, sample_sizes=args.sizes)
    elif args.families:
        runner.run_all(families=args.families, sample_sizes=args.sizes)
    else:
        print("Specify --all, --family, or --families. Use --help for options.")
        return

    # Output
    runner.print_summary()

    if args.output or config.save_results:
        runner.save(args.output)


if __name__ == "__main__":
    main()
