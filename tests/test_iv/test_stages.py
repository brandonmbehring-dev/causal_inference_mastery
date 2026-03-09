"""
Tests for three stages of IV regression (FirstStage, ReducedForm, SecondStage).

Validates:
- Stage separation (manual 3-stage = automatic 2SLS)
- Wald estimator identity (reduced form = first stage × second stage)
- First-stage diagnostics (F-stat, partial R²)
- Integration with TwoStageLeastSquares
"""

import numpy as np
import pytest

from src.causal_inference.iv import (
    TwoStageLeastSquares,
    FirstStage,
    ReducedForm,
    SecondStage,
)


class TestFirstStage:
    """Test first-stage regression."""

    def test_first_stage_f_statistic_strong_iv(self, iv_strong_instrument):
        """Test first-stage F-statistic with strong instrument."""
        Y, D, Z, X, _ = iv_strong_instrument

        first = FirstStage()
        first.fit(D, Z, X)

        # F-statistic should be > 20 (strong instrument)
        assert first.f_statistic_ > 20, f"Expected F > 20, got {first.f_statistic_:.2f}"

    def test_first_stage_f_statistic_weak_iv(self, iv_weak_instrument):
        """Test first-stage F-statistic with weak instrument."""
        Y, D, Z, X, _ = iv_weak_instrument

        first = FirstStage()
        first.fit(D, Z, X)

        # F-statistic should be < 12 (weak instrument)
        assert first.f_statistic_ < 12, f"Expected F < 12, got {first.f_statistic_:.2f}"

    def test_first_stage_partial_r2_bounds(self, iv_just_identified):
        """Test that partial R² is between 0 and 1."""
        Y, D, Z, X, _ = iv_just_identified

        first = FirstStage()
        first.fit(D, Z, X)

        assert 0 < first.partial_r2_ < 1, (
            f"Partial R² must be in (0, 1), got {first.partial_r2_:.4f}"
        )

    def test_first_stage_fitted_values_length(self, iv_just_identified):
        """Test that fitted values have correct length."""
        Y, D, Z, X, _ = iv_just_identified

        first = FirstStage()
        first.fit(D, Z, X)

        assert len(first.fitted_values_) == len(D), (
            f"Expected {len(D)} fitted values, got {len(first.fitted_values_)}"
        )

    def test_first_stage_predict(self, iv_just_identified):
        """Test first-stage prediction on new data."""
        Y, D, Z, X, _ = iv_just_identified

        first = FirstStage()
        first.fit(D, Z, X)

        # Predict on same data (should match fitted values)
        D_hat = first.predict(Z, X)

        assert np.allclose(D_hat, first.fitted_values_), (
            "Predictions should match fitted values on training data"
        )


class TestReducedForm:
    """Test reduced-form regression."""

    def test_reduced_form_fit(self, iv_just_identified):
        """Test reduced-form regression runs without error."""
        Y, D, Z, X, _ = iv_just_identified

        reduced = ReducedForm()
        reduced.fit(Y, Z, X)

        # Should have coefficients
        assert reduced.coef_ is not None
        assert len(reduced.coef_) > 0

    def test_reduced_form_r2_bounds(self, iv_just_identified):
        """Test that reduced-form R² is between 0 and 1."""
        Y, D, Z, X, _ = iv_just_identified

        reduced = ReducedForm()
        reduced.fit(Y, Z, X)

        assert 0 <= reduced.r2_ <= 1, f"R² must be in [0, 1], got {reduced.r2_:.4f}"

    def test_reduced_form_with_controls(self, iv_with_controls):
        """Test reduced form with exogenous controls."""
        Y, D, Z, X, _ = iv_with_controls

        reduced = ReducedForm()
        reduced.fit(Y, Z, X)

        # Should have coefficients for Z and X
        # Z: 1 instrument, X: 2 controls → 3 coefficients total
        assert len(reduced.coef_) == 3, (
            f"Expected 3 coefficients [Z, X1, X2], got {len(reduced.coef_)}"
        )


class TestSecondStage:
    """Test second-stage regression."""

    def test_second_stage_fit(self, iv_just_identified):
        """Test second-stage regression runs without error."""
        Y, D, Z, X, _ = iv_just_identified

        # First get D_hat from first stage
        first = FirstStage()
        first.fit(D, Z, X)
        D_hat = first.fitted_values_

        # Fit second stage
        second = SecondStage()
        second.fit(Y, D_hat, X)

        # Should have coefficient
        assert second.coef_ is not None
        assert len(second.coef_) > 0

    def test_second_stage_naive_se_warning(self, iv_just_identified):
        """Test that second-stage SEs are marked as naive."""
        Y, D, Z, X, _ = iv_just_identified

        first = FirstStage()
        first.fit(D, Z, X)
        D_hat = first.fitted_values_

        second = SecondStage()
        second.fit(Y, D_hat, X)

        # Attribute should be se_naive_ (not se_)
        assert hasattr(second, "se_naive_"), "Second stage should have se_naive_ attribute"
        assert not hasattr(second, "se_"), (
            "Second stage should NOT have se_ attribute (to prevent confusion)"
        )


class TestWaldEstimatorIdentity:
    """Test Wald estimator identity: γ = π × β (reduced form = first × second)."""

    def test_wald_identity_just_identified(self, iv_just_identified):
        """Test Wald identity with just-identified IV."""
        Y, D, Z, X, _ = iv_just_identified

        # Fit all three stages
        first = FirstStage().fit(D, Z, X)
        reduced = ReducedForm().fit(Y, Z, X)
        second = SecondStage().fit(Y, first.fitted_values_, X)

        # Extract coefficients (first coefficient is for instrument Z)
        pi = first.coef_[0]  # Effect of Z on D
        gamma = reduced.coef_[0]  # Effect of Z on Y (reduced form)
        beta = second.coef_[0]  # Effect of D on Y (structural)

        # Wald identity: γ = π × β
        assert np.isclose(gamma, pi * beta, rtol=0.01), (
            f"Wald identity failed: γ={gamma:.4f}, π×β={pi * beta:.4f}"
        )

    def test_wald_identity_strong_instrument(self, iv_strong_instrument):
        """Test Wald identity with strong instrument."""
        Y, D, Z, X, _ = iv_strong_instrument

        first = FirstStage().fit(D, Z, X)
        reduced = ReducedForm().fit(Y, Z, X)
        second = SecondStage().fit(Y, first.fitted_values_, X)

        pi = first.coef_[0]
        gamma = reduced.coef_[0]
        beta = second.coef_[0]

        # Should hold tightly with strong instrument
        assert np.isclose(gamma, pi * beta, rtol=0.005), (
            f"Wald identity failed: γ={gamma:.4f}, π×β={pi * beta:.4f}"
        )


class TestStageSeparation:
    """Test that manual 3-stage equals automatic 2SLS."""

    def test_manual_equals_automatic_just_identified(self, iv_just_identified):
        """Test manual 3-stage = automatic 2SLS (just-identified)."""
        Y, D, Z, X, _ = iv_just_identified

        # Manual 3-stage
        first = FirstStage().fit(D, Z, X)
        D_hat = first.fitted_values_
        second = SecondStage().fit(Y, D_hat, X)
        beta_manual = second.coef_[0]

        # Automatic 2SLS
        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)
        beta_auto = iv.coef_[0]

        # Coefficients should match (SEs won't, because second stage uses naive SEs)
        assert np.isclose(beta_manual, beta_auto, rtol=0.001), (
            f"Manual 3-stage ({beta_manual:.4f}) != automatic 2SLS ({beta_auto:.4f})"
        )

    def test_manual_equals_automatic_over_identified(self, iv_over_identified):
        """Test manual 3-stage = automatic 2SLS (over-identified)."""
        Y, D, Z, X, _ = iv_over_identified

        # Manual 3-stage
        first = FirstStage().fit(D, Z, X)
        D_hat = first.fitted_values_
        second = SecondStage().fit(Y, D_hat, X)
        beta_manual = second.coef_[0]

        # Automatic 2SLS
        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)
        beta_auto = iv.coef_[0]

        # Should match even with 2 instruments
        assert np.isclose(beta_manual, beta_auto, rtol=0.001), (
            f"Manual 3-stage ({beta_manual:.4f}) != automatic 2SLS ({beta_auto:.4f})"
        )

    def test_manual_equals_automatic_with_controls(self, iv_with_controls):
        """Test manual 3-stage = automatic 2SLS (with controls)."""
        Y, D, Z, X, _ = iv_with_controls

        # Manual 3-stage
        first = FirstStage().fit(D, Z, X)
        D_hat = first.fitted_values_
        second = SecondStage().fit(Y, D_hat, X)
        beta_manual = second.coef_[0]

        # Automatic 2SLS
        iv = TwoStageLeastSquares(inference="robust")
        iv.fit(Y, D, Z, X)
        beta_auto = iv.coef_[0]

        # Should match with controls
        assert np.isclose(beta_manual, beta_auto, rtol=0.001), (
            f"Manual 3-stage ({beta_manual:.4f}) != automatic 2SLS ({beta_auto:.4f})"
        )


class TestSummaryMethods:
    """Test summary output for all three stages."""

    def test_first_stage_summary(self, iv_just_identified):
        """Test first-stage summary returns DataFrame."""
        Y, D, Z, X, _ = iv_just_identified

        first = FirstStage().fit(D, Z, X)
        summary = first.summary()

        assert isinstance(summary, __import__("pandas").DataFrame), "Summary should be DataFrame"
        assert "coef" in summary.columns
        assert "se" in summary.columns

    def test_reduced_form_summary(self, iv_just_identified):
        """Test reduced-form summary returns DataFrame."""
        Y, D, Z, X, _ = iv_just_identified

        reduced = ReducedForm().fit(Y, Z, X)
        summary = reduced.summary()

        assert isinstance(summary, __import__("pandas").DataFrame), "Summary should be DataFrame"
        assert "coef" in summary.columns

    def test_second_stage_summary(self, iv_just_identified):
        """Test second-stage summary returns DataFrame."""
        Y, D, Z, X, _ = iv_just_identified

        first = FirstStage().fit(D, Z, X)
        second = SecondStage().fit(Y, first.fitted_values_, X)
        summary = second.summary()

        assert isinstance(summary, __import__("pandas").DataFrame), "Summary should be DataFrame"
        assert "coef" in summary.columns
        assert "se_naive" in summary.columns, "Second stage should mark SEs as naive"
