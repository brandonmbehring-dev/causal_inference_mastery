#!/usr/bin/env python
"""Generate golden baseline JSON for benchmark regression testing.

Usage:
    python benchmarks/golden/generate_baseline.py [--output FILE]

This script runs all benchmarks at standard sample sizes and saves
the median timings to a JSON file for regression testing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.methods import get_all_benchmarks, IMPLEMENTED_FAMILIES


def generate_baseline(
    sample_sizes: tuple = (100, 1000),
    n_repetitions: int = 10,
    n_warmup: int = 2,
) -> Dict[str, Any]:
    """Generate baseline timings for all benchmarks.

    Parameters
    ----------
    sample_sizes : tuple
        Sample sizes to benchmark.
    n_repetitions : int
        Timing repetitions per benchmark.
    n_warmup : int
        Warmup runs.

    Returns
    -------
    Dict[str, Any]
        Baseline data with timings and metadata.
    """
    baseline = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "sample_sizes": list(sample_sizes),
            "n_repetitions": n_repetitions,
            "n_warmup": n_warmup,
        },
        "families": {},
    }

    all_benchmarks = get_all_benchmarks()
    total_methods = sum(len(b) for b in all_benchmarks.values())
    completed = 0

    for family, benchmarks in all_benchmarks.items():
        if not benchmarks:
            continue

        baseline["families"][family] = {}

        for method_name, benchmark_func in benchmarks.items():
            print(f"[{completed + 1}/{total_methods}] {family}/{method_name}...", end=" ")

            method_results = {}

            for n in sample_sizes:
                try:
                    # Handle different parameter conventions
                    if family in ("scm",):
                        result = benchmark_func(
                            n_control=max(5, n // 10),
                            n_periods=10,
                            n_repetitions=n_repetitions,
                            n_warmup=n_warmup,
                        )
                    elif family in ("panel",):
                        result = benchmark_func(
                            n_units=max(10, n // 5),
                            n_periods=5,
                            n_repetitions=n_repetitions,
                            n_warmup=n_warmup,
                        )
                    elif family in ("shift_share",):
                        result = benchmark_func(
                            n_regions=max(10, n // 10),
                            n_periods=5,
                            n_repetitions=n_repetitions,
                            n_warmup=n_warmup,
                        )
                    else:
                        result = benchmark_func(
                            n=n,
                            n_repetitions=n_repetitions,
                            n_warmup=n_warmup,
                        )

                    method_results[n] = {
                        "median_ms": result.median_time_ms,
                        "min_ms": result.min_time_ms,
                        "max_ms": result.max_time_ms,
                        "std_ms": result.std_time_ms,
                        "memory_kb": result.memory_peak_kb,
                        "speed_category": result.speed_category,
                    }
                except Exception as e:
                    method_results[n] = {"error": str(e)}
                    print(f"Error at n={n}: {e}")

            baseline["families"][family][method_name] = method_results
            completed += 1
            print("done")

    return baseline


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden baseline for benchmark regression testing"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).parent / "benchmark_baseline.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=[100, 1000],
        help="Sample sizes to benchmark",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Number of timing repetitions",
    )

    args = parser.parse_args()

    print(f"Generating baseline with sample sizes: {args.sample_sizes}")
    print(f"Repetitions per benchmark: {args.repetitions}")
    print("-" * 50)

    baseline = generate_baseline(
        sample_sizes=tuple(args.sample_sizes),
        n_repetitions=args.repetitions,
    )

    # Save to file
    with open(args.output, "w") as f:
        json.dump(baseline, f, indent=2)

    print("-" * 50)
    print(f"Baseline saved to: {args.output}")

    # Summary
    total = sum(
        len(methods) for methods in baseline["families"].values()
    )
    print(f"Total methods benchmarked: {total}")


if __name__ == "__main__":
    main()
