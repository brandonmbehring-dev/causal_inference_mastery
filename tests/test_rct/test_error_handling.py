"""Test error handling for RCT estimators.

Following Brandon's principle: NEVER FAIL SILENTLY.
Every error must be explicit with diagnostic information.

~30% of tests should be error handling tests.
"""

import numpy as np
import pytest

from src.causal_inference.rct.estimators import simple_ate


class TestSimpleATEErrorHandling:
    """Test that simple_ate fails fast with clear error messages."""

    def test_empty_input_fails_fast(self, empty_arrays):
        """
        Test that empty inputs raise explicit ValueError.

        Following CODING_STANDARDS.md: Error messages must include:
        - CRITICAL ERROR prefix
        - What went wrong
        - Context (function name, parameter values)
        - Expected vs Got
        """
        with pytest.raises(ValueError) as exc_info:
            simple_ate(
                outcomes=empty_arrays["outcomes"],
                treatment=empty_arrays["treatment"]
            )

        error_msg = str(exc_info.value)

        # Check error message contains required information
        assert "CRITICAL ERROR" in error_msg
        assert "empty" in error_msg.lower()
        assert "simple_ate" in error_msg or "function" in error_msg.lower()

    def test_mismatched_lengths_fails_fast(self, mismatched_arrays):
        """
        Test that arrays with different lengths raise explicit ValueError.

        Expected error format:
        "CRITICAL ERROR: Arrays have different lengths.
         Function: simple_ate.
         Expected: Same length arrays.
         Got: len(outcomes)=3, len(treatment)=2."
        """
        with pytest.raises(ValueError) as exc_info:
            simple_ate(
                outcomes=mismatched_arrays["outcomes"],
                treatment=mismatched_arrays["treatment"]
            )

        error_msg = str(exc_info.value)

        assert "CRITICAL ERROR" in error_msg
        assert "different lengths" in error_msg.lower() or "length" in error_msg.lower()
        assert "3" in error_msg  # outcomes length
        assert "2" in error_msg  # treatment length

    def test_nan_values_fail_fast(self, arrays_with_nan):
        """
        Test that NaN values raise explicit ValueError.

        NaN values indicate data quality issues and should never be silently ignored.
        """
        with pytest.raises(ValueError) as exc_info:
            simple_ate(
                outcomes=arrays_with_nan["outcomes"],
                treatment=arrays_with_nan["treatment"]
            )

        error_msg = str(exc_info.value)

        assert "CRITICAL ERROR" in error_msg
        assert "nan" in error_msg.lower() or "missing" in error_msg.lower()

    def test_all_treated_fails_fast(self, all_treated):
        """
        Test that data with no control units raises explicit ValueError.

        Cannot estimate treatment effect without control group.
        """
        with pytest.raises(ValueError) as exc_info:
            simple_ate(
                outcomes=all_treated["outcomes"],
                treatment=all_treated["treatment"]
            )

        error_msg = str(exc_info.value)

        assert "CRITICAL ERROR" in error_msg
        assert ("control" in error_msg.lower() or
                "no variation" in error_msg.lower() or
                "all treated" in error_msg.lower())

    def test_all_control_fails_fast(self, all_control):
        """
        Test that data with no treated units raises explicit ValueError.

        Cannot estimate treatment effect without treated group.
        """
        with pytest.raises(ValueError) as exc_info:
            simple_ate(
                outcomes=all_control["outcomes"],
                treatment=all_control["treatment"]
            )

        error_msg = str(exc_info.value)

        assert "CRITICAL ERROR" in error_msg
        assert ("treated" in error_msg.lower() or
                "no variation" in error_msg.lower() or
                "all control" in error_msg.lower())

    def test_invalid_alpha_fails_fast(self, simple_rct_data):
        """
        Test that invalid alpha values raise explicit ValueError.

        Alpha must be in (0, 1).
        """
        # Test alpha = 0
        with pytest.raises(ValueError) as exc_info:
            simple_ate(
                outcomes=simple_rct_data["outcomes"],
                treatment=simple_rct_data["treatment"],
                alpha=0.0
            )

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "alpha" in error_msg.lower()

        # Test alpha >= 1
        with pytest.raises(ValueError) as exc_info:
            simple_ate(
                outcomes=simple_rct_data["outcomes"],
                treatment=simple_rct_data["treatment"],
                alpha=1.0
            )

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "alpha" in error_msg.lower()

        # Test negative alpha
        with pytest.raises(ValueError) as exc_info:
            simple_ate(
                outcomes=simple_rct_data["outcomes"],
                treatment=simple_rct_data["treatment"],
                alpha=-0.05
            )

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert "alpha" in error_msg.lower()

    def test_non_binary_treatment_fails_fast(self):
        """
        Test that non-binary treatment values raise explicit ValueError.

        Treatment must be 0 or 1.
        """
        outcomes = np.array([1.0, 2.0, 3.0, 4.0])
        treatment = np.array([0, 1, 2, 3])  # Invalid: not binary

        with pytest.raises(ValueError) as exc_info:
            simple_ate(outcomes=outcomes, treatment=treatment)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert ("binary" in error_msg.lower() or
                "0 or 1" in error_msg or
                "treatment" in error_msg.lower())

    def test_infinite_values_fail_fast(self):
        """
        Test that infinite values raise explicit ValueError.
        """
        outcomes = np.array([1.0, 2.0, np.inf, 4.0])
        treatment = np.array([1, 1, 0, 0])

        with pytest.raises(ValueError) as exc_info:
            simple_ate(outcomes=outcomes, treatment=treatment)

        error_msg = str(exc_info.value)
        assert "CRITICAL ERROR" in error_msg
        assert ("inf" in error_msg.lower() or
                "infinite" in error_msg.lower())


class TestInputValidation:
    """Test that inputs are validated before computation."""

    def test_treatment_converted_to_array(self):
        """
        Test that list inputs are converted to numpy arrays.

        Should accept list but internally convert to numpy array.
        """
        # Inline deterministic data as lists (not arrays)
        outcomes = [7.0, 5.0, 3.0, 1.0]
        treatment = [1, 1, 0, 0]

        result = simple_ate(outcomes=outcomes, treatment=treatment)

        # Should compute correct result after conversion
        # Hand-calculated: (7+5)/2 - (3+1)/2 = 6.0 - 2.0 = 4.0
        assert np.isclose(result["estimate"], 4.0)

    def test_boolean_treatment_accepted(self):
        """
        Test that boolean treatment values are accepted and converted.

        Treatment as [True, True, False, False] should work.
        """
        outcomes = np.array([7.0, 5.0, 3.0, 1.0])
        treatment = np.array([True, True, False, False])

        result = simple_ate(outcomes=outcomes, treatment=treatment)

        # Should give same result as [1, 1, 0, 0]
        assert np.isclose(result["estimate"], 4.0)


# Marker for pytest
pytestmark = pytest.mark.unit
