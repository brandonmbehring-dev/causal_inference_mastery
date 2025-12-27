"""Benchmark regression tests.

These tests verify that method timings haven't regressed beyond tolerance.
Run after establishing a baseline with `python -m benchmarks.runner --all`.

Usage:
    pytest tests/benchmark/test_benchmark_regression.py -v
    pytest tests/benchmark/test_benchmark_regression.py -v -m "not slow"
"""

from __future__ import annotations

import pytest

from benchmarks.config import BenchmarkConfig
from tests.benchmark.conftest import (
    assert_timing_regression,
    skip_without_baseline,
)


# =============================================================================
# Smoke Tests (Always Run)
# These verify benchmarks execute without error at small n
# =============================================================================


class TestBenchmarkSmokeTests:
    """Verify all benchmark functions run without error."""

    @pytest.mark.benchmark
    def test_rct_simple_ate_smoke(self, small_config):
        """simple_ate benchmark runs without error."""
        from benchmarks.methods.rct import benchmark_simple_ate

        result = benchmark_simple_ate(
            n=100,
            seed=42,
            n_repetitions=3,
            n_warmup=1,
        )
        assert result.median_time_ms > 0
        assert result.memory_peak_kb > 0
        assert result.method_name == "simple_ate"
        assert result.family == "rct"

    @pytest.mark.benchmark
    def test_rct_stratified_ate_smoke(self, small_config):
        """stratified_ate benchmark runs without error."""
        from benchmarks.methods.rct import benchmark_stratified_ate

        result = benchmark_stratified_ate(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_rct_regression_ate_smoke(self, small_config):
        """regression_ate benchmark runs without error."""
        from benchmarks.methods.rct import benchmark_regression_ate

        result = benchmark_regression_ate(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_rct_permutation_test_smoke(self, small_config):
        """permutation_test benchmark runs (slow due to many permutations)."""
        from benchmarks.methods.rct import benchmark_permutation_test

        result = benchmark_permutation_test(
            n=100,
            seed=42,
            n_repetitions=2,
            n_permutations=100,  # Reduced for smoke test
        )
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_rct_ipw_ate_smoke(self, small_config):
        """ipw_ate benchmark runs without error."""
        from benchmarks.methods.rct import benchmark_ipw_ate

        result = benchmark_ipw_ate(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_observational_ipw_smoke(self, small_config):
        """ipw_ate_obs benchmark runs without error."""
        from benchmarks.methods.observational import benchmark_ipw_ate_obs

        result = benchmark_ipw_ate_obs(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0
        assert result.family == "observational"

    @pytest.mark.benchmark
    def test_observational_dr_smoke(self, small_config):
        """dr_ate benchmark runs without error."""
        from benchmarks.methods.observational import benchmark_dr_ate

        result = benchmark_dr_ate(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_observational_tmle_smoke(self, small_config):
        """tmle_ate benchmark runs without error."""
        from benchmarks.methods.observational import benchmark_tmle_ate

        result = benchmark_tmle_ate(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_psm_smoke(self, small_config):
        """psm_ate benchmark runs without error."""
        from benchmarks.methods.psm import benchmark_psm_ate

        result = benchmark_psm_ate(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0
        assert result.family == "psm"

    @pytest.mark.benchmark
    def test_did_2x2_smoke(self, small_config):
        """did_2x2 benchmark runs without error."""
        from benchmarks.methods.did import benchmark_did_2x2

        result = benchmark_did_2x2(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0
        assert result.family == "did"

    @pytest.mark.benchmark
    def test_did_event_study_smoke(self, small_config):
        """event_study benchmark runs without error."""
        from benchmarks.methods.did import benchmark_event_study

        result = benchmark_event_study(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_iv_2sls_smoke(self, small_config):
        """two_stage_ls benchmark runs without error."""
        from benchmarks.methods.iv import benchmark_two_stage_ls

        result = benchmark_two_stage_ls(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0
        assert result.family == "iv"

    @pytest.mark.benchmark
    def test_iv_liml_smoke(self, small_config):
        """liml benchmark runs without error."""
        from benchmarks.methods.iv import benchmark_liml

        result = benchmark_liml(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_iv_fuller_smoke(self, small_config):
        """fuller benchmark runs without error."""
        from benchmarks.methods.iv import benchmark_fuller

        result = benchmark_fuller(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_iv_gmm_smoke(self, small_config):
        """gmm benchmark runs without error."""
        from benchmarks.methods.iv import benchmark_gmm

        result = benchmark_gmm(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_rdd_sharp_smoke(self, small_config):
        """sharp_rdd benchmark runs without error."""
        from benchmarks.methods.rdd import benchmark_sharp_rdd

        result = benchmark_sharp_rdd(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0
        assert result.family == "rdd"

    @pytest.mark.benchmark
    def test_rdd_fuzzy_smoke(self, small_config):
        """fuzzy_rdd benchmark runs without error."""
        from benchmarks.methods.rdd import benchmark_fuzzy_rdd

        result = benchmark_fuzzy_rdd(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_rdd_mccrary_smoke(self, small_config):
        """mccrary benchmark runs without error."""
        from benchmarks.methods.rdd import benchmark_mccrary

        result = benchmark_mccrary(n=100, seed=42, n_repetitions=3)
        assert result.median_time_ms > 0


# =============================================================================
# Regression Tests (Require Baseline)
# These compare against stored baseline timings
# =============================================================================


@skip_without_baseline
class TestBenchmarkRegression:
    """Regression tests comparing against baseline timings."""

    @pytest.mark.benchmark
    def test_simple_ate_regression(self, baseline_timings):
        """simple_ate timing within tolerance of baseline."""
        from benchmarks.methods.rct import benchmark_simple_ate

        if "simple_ate" not in baseline_timings:
            pytest.skip("No baseline for simple_ate")

        for n in [100, 1000]:
            if n not in baseline_timings["simple_ate"]:
                continue

            result = benchmark_simple_ate(n=n, seed=42, n_repetitions=5)
            assert_timing_regression(
                current_ms=result.median_time_ms,
                baseline_ms=baseline_timings["simple_ate"][n],
                method_name="simple_ate",
                sample_size=n,
            )

    @pytest.mark.benchmark
    def test_dr_ate_regression(self, baseline_timings):
        """dr_ate timing within tolerance of baseline."""
        from benchmarks.methods.observational import benchmark_dr_ate

        if "dr_ate" not in baseline_timings:
            pytest.skip("No baseline for dr_ate")

        for n in [100, 1000]:
            if n not in baseline_timings["dr_ate"]:
                continue

            result = benchmark_dr_ate(n=n, seed=42, n_repetitions=5)
            assert_timing_regression(
                current_ms=result.median_time_ms,
                baseline_ms=baseline_timings["dr_ate"][n],
                method_name="dr_ate",
                sample_size=n,
            )


# =============================================================================
# Scaling Tests
# These verify expected algorithmic complexity
# =============================================================================


class TestScalingBehavior:
    """Test that methods scale as expected with sample size."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_simple_ate_linear_scaling(self):
        """simple_ate should scale approximately linearly.

        Note: For very fast methods, fixed overhead can dominate at small n,
        so we use larger sample sizes to see scaling behavior.
        """
        from benchmarks.methods.rct import benchmark_simple_ate

        # Run at larger sample sizes where computation dominates overhead
        result_small = benchmark_simple_ate(n=10000, seed=42, n_repetitions=5)
        result_large = benchmark_simple_ate(n=100000, seed=42, n_repetitions=5)

        # Time should increase roughly 10x for 10x data
        # Allow 1.5-30x range (fast methods may have sub-linear scaling due to vectorization)
        ratio = result_large.median_time_ms / result_small.median_time_ms
        assert 1.5 < ratio < 30, (
            f"simple_ate scaling ratio {ratio:.1f}x outside expected range [1.5, 30] "
            f"for 10x sample size increase"
        )

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_permutation_linear_scaling(self):
        """permutation_test should scale roughly linearly with n."""
        from benchmarks.methods.rct import benchmark_permutation_test

        result_small = benchmark_permutation_test(
            n=500, seed=42, n_repetitions=3, n_permutations=100
        )
        result_large = benchmark_permutation_test(
            n=2000, seed=42, n_repetitions=3, n_permutations=100
        )

        # 4x sample size should give ~4x time (linear)
        ratio = result_large.median_time_ms / result_small.median_time_ms
        assert 2 < ratio < 10, (
            f"permutation_test scaling ratio {ratio:.1f}x outside expected range [2, 10] "
            f"for 4x sample size increase"
        )


# =============================================================================
# Utils Tests
# =============================================================================


class TestBenchmarkUtils:
    """Test benchmark utility functions."""

    def test_time_function(self):
        """time_function returns valid timing results."""
        from benchmarks.utils import time_function

        def simple_func():
            return sum(range(1000))

        result = time_function(simple_func, n_repetitions=5, n_warmup=1)

        assert "median_ms" in result
        assert "min_ms" in result
        assert "max_ms" in result
        assert "std_ms" in result
        assert "times_ms" in result
        assert len(result["times_ms"]) == 5
        assert result["min_ms"] <= result["median_ms"] <= result["max_ms"]

    def test_measure_memory(self):
        """measure_memory returns valid memory results."""
        from benchmarks.utils import measure_memory

        def allocate():
            return [0] * 100000

        result = measure_memory(allocate)

        assert "peak_kb" in result
        assert "current_kb" in result
        assert result["peak_kb"] > 0

    def test_benchmark_result_dataclass(self):
        """BenchmarkResult dataclass works correctly."""
        from benchmarks.utils import BenchmarkResult

        result = BenchmarkResult(
            method_name="test",
            family="test",
            sample_size=100,
            median_time_ms=10.5,
            min_time_ms=8.0,
            max_time_ms=15.0,
            std_time_ms=2.0,
            memory_peak_kb=100.0,
            n_repetitions=10,
        )

        assert result.speed_category == "medium"  # 10-100ms

        d = result.to_dict()
        assert d["method_name"] == "test"
        assert d["median_time_ms"] == 10.5

        result2 = BenchmarkResult.from_dict(d)
        assert result2.method_name == result.method_name

    def test_format_results_table(self):
        """format_results_table produces valid output."""
        from benchmarks.utils import BenchmarkResult, format_results_table

        results = [
            BenchmarkResult(
                method_name="fast_method",
                family="rct",
                sample_size=100,
                median_time_ms=1.0,
                min_time_ms=0.8,
                max_time_ms=1.2,
                std_time_ms=0.1,
                memory_peak_kb=50.0,
                n_repetitions=10,
            ),
            BenchmarkResult(
                method_name="slow_method",
                family="did",
                sample_size=100,
                median_time_ms=500.0,
                min_time_ms=450.0,
                max_time_ms=550.0,
                std_time_ms=30.0,
                memory_peak_kb=200.0,
                n_repetitions=10,
            ),
        ]

        table = format_results_table(results)
        assert "fast_method" in table
        assert "slow_method" in table
        assert "rct" in table
        assert "did" in table
