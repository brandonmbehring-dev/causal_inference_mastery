"""Golden Reference Regression Tests.

This module tests that current implementations produce results matching
the frozen golden reference (tests/golden_results/python_golden_results.json).

Purpose:
- Catch unintended regressions in estimator behavior
- Validate that refactoring preserves numerical results
- Ensure reproducibility across versions

Created: Session 166 (Independent Audit 2026-01-01)
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.causal_inference.rct.estimators import simple_ate
from src.causal_inference.rct.estimators_stratified import stratified_ate
from src.causal_inference.rct.estimators_regression import regression_adjusted_ate
from src.causal_inference.rct.estimators_ipw import ipw_ate


# Load golden results at module level
GOLDEN_PATH = Path(__file__).parent.parent / "golden_results" / "python_golden_results.json"


@pytest.fixture(scope="module")
def golden_results():
    """Load golden reference results."""
    if not GOLDEN_PATH.exists():
        pytest.skip(f"Golden results file not found: {GOLDEN_PATH}")
    with open(GOLDEN_PATH) as f:
        return json.load(f)


class TestGoldenReferenceRCT:
    """Test RCT estimators against golden reference."""

    def test_balanced_rct_simple_ate(self, golden_results):
        """Test simple_ate matches golden reference for balanced RCT."""
        data = golden_results["balanced_rct"]["data"]
        expected = golden_results["balanced_rct"]["simple_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])

        result = simple_ate(outcomes, treatment)

        # Use rtol=1e-10 for numerical precision (not exact equality due to float representation)
        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"ATE mismatch: got {result['estimate']}, expected {expected['estimate']}"
        )
        assert np.isclose(result["se"], expected["se"], rtol=1e-10), (
            f"SE mismatch: got {result['se']}, expected {expected['se']}"
        )

    def test_stratified_rct_stratified_ate(self, golden_results):
        """Test stratified_ate matches golden reference."""
        data = golden_results["stratified_rct"]["data"]
        expected = golden_results["stratified_rct"]["stratified_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])
        strata = np.array(data["strata"])

        result = stratified_ate(outcomes, treatment, strata)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"ATE mismatch: got {result['estimate']}, expected {expected['estimate']}"
        )
        assert np.isclose(result["se"], expected["se"], rtol=1e-10), (
            f"SE mismatch: got {result['se']}, expected {expected['se']}"
        )

    def test_regression_rct_regression_adjusted_ate(self, golden_results):
        """Test regression_adjusted_ate matches golden reference."""
        data = golden_results["regression_rct"]["data"]
        expected = golden_results["regression_rct"]["regression_adjusted_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])
        covariate = np.array(data["covariate"])

        result = regression_adjusted_ate(outcomes, treatment, covariate)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"ATE mismatch: got {result["estimate"]}, expected {expected['ate']}"
        )
        assert np.isclose(result["se"], expected["se"], rtol=1e-10), (
            f"SE mismatch: got {result["se"]}, expected {expected['se']}"
        )

    def test_ipw_varying_ipw_ate(self, golden_results):
        """Test ipw_ate matches golden reference with varying propensity."""
        data = golden_results["ipw_varying"]["data"]
        expected = golden_results["ipw_varying"]["ipw_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])
        propensity = np.array(data["propensity"])

        result = ipw_ate(outcomes, treatment, propensity)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"ATE mismatch: got {result["estimate"]}, expected {expected['ate']}"
        )
        assert np.isclose(result["se"], expected["se"], rtol=1e-10), (
            f"SE mismatch: got {result["se"]}, expected {expected['se']}"
        )

    def test_large_sample_simple_ate(self, golden_results):
        """Test simple_ate matches golden reference for large sample."""
        data = golden_results["large_sample"]["data"]
        expected = golden_results["large_sample"]["simple_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])

        result = simple_ate(outcomes, treatment)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"ATE mismatch: got {result["estimate"]}, expected {expected['ate']}"
        )
        assert np.isclose(result["se"], expected["se"], rtol=1e-10), (
            f"SE mismatch: got {result["se"]}, expected {expected['se']}"
        )

    def test_large_sample_regression_adjusted_ate(self, golden_results):
        """Test regression_adjusted_ate matches golden reference for large sample."""
        data = golden_results["large_sample"]["data"]
        expected = golden_results["large_sample"]["regression_adjusted_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])
        covariate = np.array(data["covariate"])

        result = regression_adjusted_ate(outcomes, treatment, covariate)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"ATE mismatch: got {result["estimate"]}, expected {expected['ate']}"
        )
        assert np.isclose(result["se"], expected["se"], rtol=1e-10), (
            f"SE mismatch: got {result["se"]}, expected {expected['se']}"
        )

    def test_large_sample_ipw_ate(self, golden_results):
        """Test ipw_ate matches golden reference for large sample."""
        data = golden_results["large_sample"]["data"]
        expected = golden_results["large_sample"]["ipw_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])
        propensity = np.array(data["propensity"])

        result = ipw_ate(outcomes, treatment, propensity)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"ATE mismatch: got {result["estimate"]}, expected {expected['ate']}"
        )
        assert np.isclose(result["se"], expected["se"], rtol=1e-10), (
            f"SE mismatch: got {result["se"]}, expected {expected['se']}"
        )


class TestGoldenReferenceMetadata:
    """Validate golden reference file structure."""

    def test_golden_file_exists(self):
        """Verify golden results file exists."""
        assert GOLDEN_PATH.exists(), f"Golden results file missing: {GOLDEN_PATH}"

    def test_golden_file_has_expected_cases(self, golden_results):
        """Verify expected test cases are present."""
        expected_cases = [
            "balanced_rct",
            "stratified_rct",
            "regression_rct",
            "permutation_small",
            "ipw_varying",
            "large_sample",
        ]
        for case in expected_cases:
            assert case in golden_results, f"Missing test case: {case}"

    def test_golden_file_has_descriptions(self, golden_results):
        """Verify all test cases have descriptions."""
        for case_name, case_data in golden_results.items():
            assert "description" in case_data, f"Missing description for {case_name}"
            assert len(case_data["description"]) > 0, f"Empty description for {case_name}"

    def test_golden_file_has_data(self, golden_results):
        """Verify all test cases have input data."""
        for case_name, case_data in golden_results.items():
            assert "data" in case_data, f"Missing data for {case_name}"
            assert "treatment" in case_data["data"], f"Missing treatment for {case_name}"
            assert "outcomes" in case_data["data"], f"Missing outcomes for {case_name}"
