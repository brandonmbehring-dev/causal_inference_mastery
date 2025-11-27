"""
Tests for shared validation utilities.

These tests validate the validation module that consolidates duplicate logic
across 9+ causal inference modules.
"""

import numpy as np
import pytest

from src.causal_inference.utils.validation import (
    validate_arrays_same_length,
    validate_finite,
    validate_binary,
    validate_not_empty,
    validate_has_variation,
    validate_in_range,
    validate_treatment_outcome,
    validate_iv_inputs,
    validate_did_inputs,
)


class TestValidateArraysSameLength:
    """Test validate_arrays_same_length()."""

    def test_same_length_arrays(self):
        """Should not raise for arrays with same length."""
        Y = np.array([1, 2, 3])
        D = np.array([0, 1, 0])
        X = np.array([1.0, 2.0, 3.0])

        # Should not raise
        validate_arrays_same_length(Y=Y, D=D, X=X)

    def test_different_length_arrays(self):
        """Should raise ValueError for different length arrays."""
        Y = np.array([1, 2, 3])
        D = np.array([0, 1])  # Length 2

        with pytest.raises(ValueError, match="same length"):
            validate_arrays_same_length(Y=Y, D=D)

    def test_empty_input(self):
        """Should not raise for empty input."""
        validate_arrays_same_length()

    def test_single_array(self):
        """Should not raise for single array."""
        Y = np.array([1, 2, 3])
        validate_arrays_same_length(Y=Y)


class TestValidateFinite:
    """Test validate_finite()."""

    def test_finite_array(self):
        """Should not raise for finite values."""
        Y = np.array([1.0, 2.0, 3.0])
        validate_finite(Y, "Y")

    def test_nan_values(self):
        """Should raise ValueError for NaN values."""
        Y = np.array([1.0, np.nan, 3.0])

        with pytest.raises(ValueError, match="non-finite"):
            validate_finite(Y, "Y")

    def test_inf_values(self):
        """Should raise ValueError for Inf values."""
        Y = np.array([1.0, np.inf, 3.0])

        with pytest.raises(ValueError, match="non-finite"):
            validate_finite(Y, "Y")

    def test_both_nan_and_inf(self):
        """Should raise ValueError for both NaN and Inf."""
        Y = np.array([np.nan, np.inf, 3.0])

        with pytest.raises(ValueError, match="non-finite"):
            validate_finite(Y, "Y")


class TestValidateBinary:
    """Test validate_binary()."""

    def test_binary_01(self):
        """Should not raise for binary array with 0 and 1."""
        D = np.array([0, 1, 1, 0])
        validate_binary(D, "D")

    def test_only_zeros(self):
        """Should not raise for array with only 0."""
        D = np.array([0, 0, 0])
        validate_binary(D, "D")

    def test_only_ones(self):
        """Should not raise for array with only 1."""
        D = np.array([1, 1, 1])
        validate_binary(D, "D")

    def test_non_binary_values(self):
        """Should raise ValueError for non-binary values."""
        D = np.array([0, 1, 2])

        with pytest.raises(ValueError, match="binary"):
            validate_binary(D, "D")

    def test_negative_values(self):
        """Should raise ValueError for negative values."""
        D = np.array([-1, 0, 1])

        with pytest.raises(ValueError, match="binary"):
            validate_binary(D, "D")


class TestValidateNotEmpty:
    """Test validate_not_empty()."""

    def test_non_empty_array(self):
        """Should not raise for non-empty array."""
        Y = np.array([1, 2, 3])
        validate_not_empty(Y, "Y")

    def test_empty_array(self):
        """Should raise ValueError for empty array."""
        Y = np.array([])

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_not_empty(Y, "Y")


class TestValidateHasVariation:
    """Test validate_has_variation()."""

    def test_array_with_variation(self):
        """Should not raise for array with variation."""
        Y = np.array([1, 2, 3])
        validate_has_variation(Y, "Y")

    def test_constant_array(self):
        """Should raise ValueError for constant array."""
        Y = np.array([1, 1, 1])

        with pytest.raises(ValueError, match="no variation"):
            validate_has_variation(Y, "Y")

    def test_2d_array_with_variation(self):
        """Should not raise for 2D array with variation in all columns."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        validate_has_variation(X, "X", axis=0)

    def test_2d_array_constant_column(self):
        """Should raise ValueError for 2D array with constant column."""
        X = np.array([[1, 2], [1, 4], [1, 6]])  # First column constant

        with pytest.raises(ValueError, match="constant columns"):
            validate_has_variation(X, "X", axis=0)


class TestValidateInRange:
    """Test validate_in_range()."""

    def test_values_in_range_inclusive(self):
        """Should not raise for values in range (inclusive)."""
        propensity = np.array([0.0, 0.5, 1.0])
        validate_in_range(propensity, "propensity", 0.0, 1.0, inclusive=True)

    def test_values_below_range(self):
        """Should raise ValueError for values below range."""
        propensity = np.array([-0.1, 0.5, 1.0])

        with pytest.raises(ValueError, match="must be in"):
            validate_in_range(propensity, "propensity", 0.0, 1.0)

    def test_values_above_range(self):
        """Should raise ValueError for values above range."""
        propensity = np.array([0.0, 0.5, 1.2])

        with pytest.raises(ValueError, match="must be in"):
            validate_in_range(propensity, "propensity", 0.0, 1.0)

    def test_exclusive_range(self):
        """Should raise ValueError for boundary values with exclusive range."""
        alpha = np.array([0.0, 0.05, 1.0])

        with pytest.raises(ValueError, match="must be in"):
            validate_in_range(alpha, "alpha", 0.0, 1.0, inclusive=False)


class TestValidateTreatmentOutcome:
    """Test validate_treatment_outcome()."""

    def test_valid_treatment_outcome(self):
        """Should not raise for valid treatment and outcome."""
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        D = np.array([0, 0, 1, 1])

        validate_treatment_outcome(Y, D)

    def test_different_lengths(self):
        """Should raise ValueError for different lengths."""
        Y = np.array([1.0, 2.0, 3.0])
        D = np.array([0, 1])

        with pytest.raises(ValueError, match="same length"):
            validate_treatment_outcome(Y, D)

    def test_nan_in_outcome(self):
        """Should raise ValueError for NaN in outcome."""
        Y = np.array([1.0, np.nan, 3.0, 4.0])
        D = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="non-finite"):
            validate_treatment_outcome(Y, D)

    def test_non_binary_treatment(self):
        """Should raise ValueError for non-binary treatment."""
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        D = np.array([0, 1, 2, 1])

        with pytest.raises(ValueError, match="binary"):
            validate_treatment_outcome(Y, D)

    def test_no_treated_units(self):
        """Should raise ValueError when all units are control."""
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        D = np.array([0, 0, 0, 0])

        with pytest.raises(ValueError, match="No treated units"):
            validate_treatment_outcome(Y, D)

    def test_no_control_units(self):
        """Should raise ValueError when all units are treated."""
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        D = np.array([1, 1, 1, 1])

        with pytest.raises(ValueError, match="No control units"):
            validate_treatment_outcome(Y, D)

    def test_constant_outcome(self):
        """Should raise ValueError for constant outcome."""
        Y = np.array([1.0, 1.0, 1.0, 1.0])
        D = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="no variation"):
            validate_treatment_outcome(Y, D)


class TestValidateIVInputs:
    """Test validate_iv_inputs()."""

    def test_valid_iv_inputs(self):
        """Should not raise for valid IV inputs."""
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        D = np.array([0.5, 1.0, 1.5, 2.0])
        Z = np.array([0, 0, 1, 1])

        validate_iv_inputs(Y, D, Z)

    def test_with_controls(self):
        """Should not raise for valid IV inputs with controls."""
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        D = np.array([0.5, 1.0, 1.5, 2.0])
        Z = np.array([0, 0, 1, 1])
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])

        validate_iv_inputs(Y, D, Z, X)

    def test_different_lengths(self):
        """Should raise ValueError for different lengths."""
        Y = np.array([1.0, 2.0, 3.0])
        D = np.array([0.5, 1.0, 1.5, 2.0])
        Z = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="same length"):
            validate_iv_inputs(Y, D, Z)

    def test_nan_in_instrument(self):
        """Should raise ValueError for NaN in instrument."""
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        D = np.array([0.5, 1.0, 1.5, 2.0])
        Z = np.array([0, np.nan, 1, 1])

        with pytest.raises(ValueError, match="non-finite"):
            validate_iv_inputs(Y, D, Z)

    def test_order_condition_fails(self):
        """Should raise ValueError when # instruments < # endogenous."""
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        D = np.array([[0.5, 1.0], [1.0, 1.5], [1.5, 2.0], [2.0, 2.5]])  # 2 endogenous
        Z = np.array([0, 0, 1, 1])  # 1 instrument

        with pytest.raises(ValueError, match="Order condition fails"):
            validate_iv_inputs(Y, D, Z)

    def test_no_variation_in_outcome(self):
        """Should raise ValueError for constant outcome."""
        Y = np.array([1.0, 1.0, 1.0, 1.0])
        D = np.array([0.5, 1.0, 1.5, 2.0])
        Z = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="no variation"):
            validate_iv_inputs(Y, D, Z)


class TestValidateDidInputs:
    """Test validate_did_inputs()."""

    def test_valid_did_inputs(self):
        """Should not raise for valid DiD inputs."""
        outcomes = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        treatment = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        post = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        unit_id = np.array([1, 1, 2, 2, 3, 3, 4, 4])

        validate_did_inputs(outcomes, treatment, post, unit_id)

    def test_different_lengths(self):
        """Should raise ValueError for different lengths."""
        outcomes = np.array([1, 2, 3, 4])
        treatment = np.array([0, 0, 1, 1])
        post = np.array([0, 1, 0])  # Length 3
        unit_id = np.array([1, 1, 2, 2])

        with pytest.raises(ValueError, match="same length"):
            validate_did_inputs(outcomes, treatment, post, unit_id)

    def test_non_binary_treatment(self):
        """Should raise ValueError for non-binary treatment."""
        outcomes = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        treatment = np.array([0, 0, 0, 0, 1, 1, 2, 2])  # Has value 2
        post = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        unit_id = np.array([1, 1, 2, 2, 3, 3, 4, 4])

        with pytest.raises(ValueError, match="binary"):
            validate_did_inputs(outcomes, treatment, post, unit_id)

    def test_non_binary_post(self):
        """Should raise ValueError for non-binary post."""
        outcomes = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        treatment = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        post = np.array([0, 1, 0, 2, 0, 1, 0, 1])  # Has value 2
        unit_id = np.array([1, 1, 2, 2, 3, 3, 4, 4])

        with pytest.raises(ValueError, match="binary"):
            validate_did_inputs(outcomes, treatment, post, unit_id)

    def test_no_treated_units(self):
        """Should raise ValueError when all units are control."""
        outcomes = np.array([1, 2, 3, 4])
        treatment = np.array([0, 0, 0, 0])
        post = np.array([0, 1, 0, 1])
        unit_id = np.array([1, 1, 2, 2])

        with pytest.raises(ValueError, match="No treated units"):
            validate_did_inputs(outcomes, treatment, post, unit_id)

    def test_no_pre_period(self):
        """Should raise ValueError when all periods are post."""
        outcomes = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        treatment = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        post = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # All post
        unit_id = np.array([1, 1, 2, 2, 3, 3, 4, 4])

        with pytest.raises(ValueError, match="No pre-treatment periods"):
            validate_did_inputs(outcomes, treatment, post, unit_id)

    def test_nan_in_outcomes(self):
        """Should raise ValueError for NaN in outcomes."""
        outcomes = np.array([1, 2, np.nan, 4, 5, 6, 7, 8])
        treatment = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        post = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        unit_id = np.array([1, 1, 2, 2, 3, 3, 4, 4])

        with pytest.raises(ValueError, match="non-finite"):
            validate_did_inputs(outcomes, treatment, post, unit_id)
