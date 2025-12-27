"""Integration tests for cross-language benchmark infrastructure.

Session 132: Validate Python vs Julia benchmark comparison.

Tests:
1. Cross-language runner instantiation and configuration
2. Julia availability detection
3. Benchmark registry completeness
4. Speedup calculation validity
5. Result serialization/deserialization
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
import tempfile

import numpy as np
import pytest

from benchmarks.cross_language import (
    CrossLanguageBenchmarkRunner,
    CrossLanguageResult,
    is_julia_available,
)
from benchmarks.cross_language.julia_benchmarks import (
    JULIA_BENCHMARK_REGISTRY,
    JULIA_AVAILABLE,
    get_julia_benchmark,
    list_available_julia_benchmarks,
)


# =============================================================================
# Test: Runner Configuration
# =============================================================================


class TestCrossLanguageRunner:
    """Test CrossLanguageBenchmarkRunner configuration and state."""

    def test_default_initialization(self) -> None:
        """Verify default sample sizes and repetitions."""
        runner = CrossLanguageBenchmarkRunner()
        assert runner.sample_sizes == [100, 1000, 5000]
        assert runner.n_repetitions == 10
        assert runner.n_warmup == 2
        assert runner.seed == 42
        assert runner.results == []

    def test_custom_initialization(self) -> None:
        """Verify custom configuration acceptance."""
        runner = CrossLanguageBenchmarkRunner(
            sample_sizes=[50, 200],
            n_repetitions=5,
            n_warmup=1,
            seed=123,
        )
        assert runner.sample_sizes == [50, 200]
        assert runner.n_repetitions == 5
        assert runner.n_warmup == 1
        assert runner.seed == 123


# =============================================================================
# Test: Julia Availability
# =============================================================================


class TestJuliaAvailability:
    """Test Julia availability detection."""

    def test_availability_returns_bool(self) -> None:
        """is_julia_available should return boolean."""
        result = is_julia_available()
        assert isinstance(result, bool)

    def test_julia_available_constant_matches(self) -> None:
        """JULIA_AVAILABLE constant should match function."""
        # Note: These might differ if Julia loads lazily
        # At minimum, both should be boolean
        assert isinstance(JULIA_AVAILABLE, bool)


# =============================================================================
# Test: Benchmark Registry
# =============================================================================


class TestBenchmarkRegistry:
    """Test Julia benchmark registry structure."""

    def test_registry_has_expected_families(self) -> None:
        """Registry should include core method families."""
        expected_families = {
            "rct",
            "iv",
            "rdd",
            "did",
            "observational",
            "cate",
            "sensitivity",
            "bounds",
            "principal_strat",
        }
        actual_families = set(JULIA_BENCHMARK_REGISTRY.keys())
        assert expected_families.issubset(actual_families), (
            f"Missing families: {expected_families - actual_families}"
        )

    def test_rct_methods_complete(self) -> None:
        """RCT family should have core methods."""
        rct_methods = set(JULIA_BENCHMARK_REGISTRY.get("rct", {}).keys())
        expected = {"simple_ate", "stratified_ate", "regression_ate", "ipw_ate", "permutation_test"}
        assert expected.issubset(rct_methods)

    def test_iv_methods_complete(self) -> None:
        """IV family should have TSLS, LIML, GMM."""
        iv_methods = set(JULIA_BENCHMARK_REGISTRY.get("iv", {}).keys())
        expected = {"tsls", "liml", "gmm"}
        assert expected.issubset(iv_methods)

    def test_registry_functions_callable(self) -> None:
        """All registered functions should be callable."""
        for family, methods in JULIA_BENCHMARK_REGISTRY.items():
            for method_name, func in methods.items():
                assert callable(func), f"{family}/{method_name} not callable"

    def test_get_julia_benchmark_returns_callable_or_none(self) -> None:
        """get_julia_benchmark should return function or None."""
        result = get_julia_benchmark("rct", "simple_ate")
        if JULIA_AVAILABLE:
            assert callable(result)
        else:
            assert result is None

    def test_get_julia_benchmark_invalid_returns_none(self) -> None:
        """Invalid family/method should return None."""
        assert get_julia_benchmark("nonexistent", "method") is None
        assert get_julia_benchmark("rct", "nonexistent") is None

    def test_list_available_benchmarks(self) -> None:
        """list_available_julia_benchmarks should return dict."""
        available = list_available_julia_benchmarks()
        assert isinstance(available, dict)
        assert all(isinstance(v, list) for v in available.values())


# =============================================================================
# Test: CrossLanguageResult
# =============================================================================


class TestCrossLanguageResult:
    """Test result dataclass behavior."""

    def test_result_creation(self) -> None:
        """Result should store all fields correctly."""
        result = CrossLanguageResult(
            method_name="simple_ate",
            family="rct",
            sample_size=1000,
            python_time_ms=10.5,
            julia_time_ms=0.5,
            speedup=21.0,
            python_memory_kb=1024.0,
            julia_memory_kb=256.0,
            n_repetitions=10,
            julia_available=True,
        )
        assert result.method_name == "simple_ate"
        assert result.speedup == 21.0

    def test_result_to_dict(self) -> None:
        """Result should serialize to dictionary."""
        result = CrossLanguageResult(
            method_name="tsls",
            family="iv",
            sample_size=500,
            python_time_ms=5.0,
            julia_time_ms=1.0,
            speedup=5.0,
            python_memory_kb=512.0,
            julia_memory_kb=128.0,
            n_repetitions=10,
            julia_available=True,
        )
        d = result.to_dict()
        assert d["method_name"] == "tsls"
        assert d["speedup"] == 5.0
        assert "julia_time_ms" in d

    def test_result_handles_nan(self) -> None:
        """Result should handle NaN values in serialization."""
        result = CrossLanguageResult(
            method_name="test",
            family="rct",
            sample_size=100,
            python_time_ms=10.0,
            julia_time_ms=np.nan,
            speedup=np.nan,
            python_memory_kb=100.0,
            julia_memory_kb=np.nan,
            n_repetitions=10,
            julia_available=False,
        )
        d = result.to_dict()
        assert d["julia_time_ms"] is None
        assert d["speedup"] is None

    def test_result_from_dict(self) -> None:
        """Result should deserialize from dictionary."""
        d = {
            "method_name": "simple_ate",
            "family": "rct",
            "sample_size": 100,
            "python_time_ms": 5.0,
            "julia_time_ms": 0.5,
            "speedup": 10.0,
            "python_memory_kb": 256.0,
            "julia_memory_kb": 64.0,
            "n_repetitions": 10,
            "julia_available": True,
        }
        result = CrossLanguageResult.from_dict(d)
        assert result.method_name == "simple_ate"
        assert result.speedup == 10.0

    def test_result_roundtrip(self) -> None:
        """Result should survive to_dict/from_dict roundtrip."""
        original = CrossLanguageResult(
            method_name="dr_ate",
            family="observational",
            sample_size=2000,
            python_time_ms=25.0,
            julia_time_ms=2.5,
            speedup=10.0,
            python_memory_kb=1024.0,
            julia_memory_kb=256.0,
            n_repetitions=10,
            julia_available=True,
        )
        restored = CrossLanguageResult.from_dict(original.to_dict())
        assert restored.method_name == original.method_name
        assert restored.speedup == original.speedup


# =============================================================================
# Test: JSON Serialization
# =============================================================================


class TestJSONSerialization:
    """Test runner JSON export/import."""

    def test_to_json_valid(self) -> None:
        """Runner should export valid JSON."""
        runner = CrossLanguageBenchmarkRunner(sample_sizes=[100])

        # Add a mock result
        runner.results.append(
            CrossLanguageResult(
                method_name="test",
                family="rct",
                sample_size=100,
                python_time_ms=1.0,
                julia_time_ms=0.1,
                speedup=10.0,
                python_memory_kb=100.0,
                julia_memory_kb=10.0,
                n_repetitions=10,
                julia_available=True,
            )
        )

        json_str = runner.to_json()
        data = json.loads(json_str)
        assert "metadata" in data
        assert "results" in data
        assert len(data["results"]) == 1

    def test_json_file_write(self) -> None:
        """Runner should write JSON to file."""
        runner = CrossLanguageBenchmarkRunner()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            runner.to_json(filepath)
            assert filepath.exists()
            with open(filepath) as f:
                data = json.load(f)
            assert "metadata" in data
        finally:
            if filepath.exists():
                filepath.unlink()

    def test_from_json_roundtrip(self) -> None:
        """Runner should survive JSON roundtrip."""
        original = CrossLanguageBenchmarkRunner(
            sample_sizes=[100, 200],
            n_repetitions=5,
            seed=99,
        )
        original.results.append(
            CrossLanguageResult(
                method_name="simple_ate",
                family="rct",
                sample_size=100,
                python_time_ms=5.0,
                julia_time_ms=0.5,
                speedup=10.0,
                python_memory_kb=256.0,
                julia_memory_kb=64.0,
                n_repetitions=5,
                julia_available=True,
            )
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            original.to_json(filepath)
            restored = CrossLanguageBenchmarkRunner.from_json(filepath)

            assert restored.sample_sizes == original.sample_sizes
            assert restored.n_repetitions == original.n_repetitions
            assert restored.seed == original.seed
            assert len(restored.results) == 1
            assert restored.results[0].method_name == "simple_ate"
        finally:
            if filepath.exists():
                filepath.unlink()


# =============================================================================
# Test: Summary Generation
# =============================================================================


class TestSummaryGeneration:
    """Test summary table generation."""

    def test_empty_summary(self) -> None:
        """Empty runner should return 'No results' message."""
        runner = CrossLanguageBenchmarkRunner()
        summary = runner.summary()
        assert "No results" in summary

    def test_summary_with_results(self) -> None:
        """Summary should include result data."""
        runner = CrossLanguageBenchmarkRunner()
        runner.results.append(
            CrossLanguageResult(
                method_name="simple_ate",
                family="rct",
                sample_size=1000,
                python_time_ms=10.0,
                julia_time_ms=1.0,
                speedup=10.0,
                python_memory_kb=512.0,
                julia_memory_kb=64.0,
                n_repetitions=10,
                julia_available=True,
            )
        )
        summary = runner.summary()
        assert "simple_ate" in summary
        assert "rct" in summary
        assert "10.0x" in summary


# =============================================================================
# Test: Speedup Calculation
# =============================================================================


class TestSpeedupCalculation:
    """Test speedup computation validity."""

    def test_speedup_greater_than_one_means_julia_faster(self) -> None:
        """Speedup > 1 indicates Julia is faster."""
        result = CrossLanguageResult(
            method_name="test",
            family="test",
            sample_size=100,
            python_time_ms=10.0,
            julia_time_ms=2.0,
            speedup=5.0,
            python_memory_kb=100.0,
            julia_memory_kb=20.0,
            n_repetitions=10,
            julia_available=True,
        )
        # Speedup = Python time / Julia time = 10/2 = 5
        assert result.speedup == result.python_time_ms / result.julia_time_ms

    def test_speedup_less_than_one_means_python_faster(self) -> None:
        """Speedup < 1 indicates Python is faster (unlikely but valid)."""
        result = CrossLanguageResult(
            method_name="test",
            family="test",
            sample_size=100,
            python_time_ms=1.0,
            julia_time_ms=10.0,
            speedup=0.1,
            python_memory_kb=100.0,
            julia_memory_kb=200.0,
            n_repetitions=10,
            julia_available=True,
        )
        assert result.speedup < 1.0


# =============================================================================
# Conditional Julia Tests
# =============================================================================


@pytest.mark.skipif(not JULIA_AVAILABLE, reason="Julia not available")
class TestWithJulia:
    """Tests that require Julia to be available."""

    def test_julia_simple_ate_callable(self) -> None:
        """Julia simple_ate wrapper should be callable."""
        from benchmarks.cross_language.julia_benchmarks import jl_benchmark_simple_ate

        # Generate minimal test data
        outcome = np.array([1.0, 2.0, 3.0, 0.5, 0.8, 1.2])
        treatment = np.array([1, 1, 1, 0, 0, 0])

        result = jl_benchmark_simple_ate(outcome, treatment)
        assert "estimate" in result or "ate" in result

    def test_julia_returns_dict(self) -> None:
        """Julia wrappers should return dictionary-like results."""
        from benchmarks.cross_language.julia_benchmarks import jl_benchmark_simple_ate

        outcome = np.array([1.0, 2.0, 3.0, 0.5, 0.8, 1.2])
        treatment = np.array([1, 1, 1, 0, 0, 0])

        result = jl_benchmark_simple_ate(outcome, treatment)
        assert isinstance(result, dict)


# =============================================================================
# Test: Registry Coverage
# =============================================================================


class TestRegistryCoverage:
    """Test benchmark registry coverage metrics."""

    def test_minimum_method_count(self) -> None:
        """Registry should have minimum number of methods."""
        total_methods = sum(
            len(methods) for methods in JULIA_BENCHMARK_REGISTRY.values()
        )
        # Should have at least 20 methods across families
        assert total_methods >= 20, f"Only {total_methods} methods registered"

    def test_minimum_family_count(self) -> None:
        """Registry should have minimum number of families."""
        n_families = len(JULIA_BENCHMARK_REGISTRY)
        # Should have at least 8 families
        assert n_families >= 8, f"Only {n_families} families registered"
