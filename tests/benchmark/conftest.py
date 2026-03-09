"""Pytest fixtures for benchmark tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import pytest

from benchmarks.config import BenchmarkConfig, CI_CONFIG, get_tolerance


# Paths
BENCHMARK_DIR = Path(__file__).parent.parent.parent / "benchmarks"
GOLDEN_DIR = BENCHMARK_DIR / "golden"
BASELINE_FILE = GOLDEN_DIR / "benchmark_baseline.json"


@pytest.fixture
def benchmark_config() -> BenchmarkConfig:
    """Provide CI-optimized benchmark configuration."""
    return CI_CONFIG


@pytest.fixture
def small_config() -> BenchmarkConfig:
    """Very small config for smoke tests."""
    return BenchmarkConfig(
        sample_sizes=[100],
        n_repetitions=3,
        n_warmup=1,
        timeout_seconds=30.0,
        verbose=False,
    )


@pytest.fixture
def baseline_timings() -> Optional[Dict[str, Any]]:
    """Load baseline timings from golden file if available.

    Returns None if baseline doesn't exist yet.
    """
    if not BASELINE_FILE.exists():
        return None

    with open(BASELINE_FILE) as f:
        data = json.load(f)

    # Convert to lookup format: {method_name: {n: median_ms}}
    timings = {}
    for result in data.get("results", []):
        method = result["method_name"]
        n = result["sample_size"]
        if method not in timings:
            timings[method] = {}
        timings[method][n] = result["median_time_ms"]

    return timings


def assert_timing_regression(
    current_ms: float,
    baseline_ms: float,
    method_name: str,
    sample_size: int,
    tolerance: Optional[float] = None,
) -> None:
    """Assert current timing is within tolerance of baseline.

    Parameters
    ----------
    current_ms : float
        Current median timing in milliseconds.
    baseline_ms : float
        Baseline median timing.
    method_name : str
        Method name for error message.
    sample_size : int
        Sample size for error message.
    tolerance : Optional[float]
        Tolerance as fraction. If None, uses default tolerance bands.

    Raises
    ------
    AssertionError
        If current timing exceeds tolerance.
    """
    if tolerance is None:
        tolerance = get_tolerance(baseline_ms)

    max_allowed = baseline_ms * (1 + tolerance)
    change_pct = (current_ms - baseline_ms) / baseline_ms * 100

    if current_ms > max_allowed:
        raise AssertionError(
            f"Performance regression detected!\n"
            f"Method: {method_name} @ n={sample_size}\n"
            f"Baseline: {baseline_ms:.2f}ms\n"
            f"Current:  {current_ms:.2f}ms\n"
            f"Change:   {change_pct:+.1f}%\n"
            f"Tolerance: ±{tolerance * 100:.0f}%\n"
            f"Max allowed: {max_allowed:.2f}ms"
        )


# Skip marker for CI environments without full baseline
skip_without_baseline = pytest.mark.skipif(
    not BASELINE_FILE.exists(),
    reason="Baseline file not found - run benchmarks first to create baseline",
)
