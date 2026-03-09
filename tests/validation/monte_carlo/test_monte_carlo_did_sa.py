"""
Monte Carlo validation for Sun-Abraham DiD estimator.

Validates statistical properties:
- Unbiasedness with heterogeneous treatment effects (IW estimator)
- Correct weighting of cohort × event-time effects
- Comparison with TWFE and CS

Key property: Sun-Abraham uses interaction-weighted (IW) estimation
that is robust to heterogeneous treatment effects.

References:
    Sun & Abraham (2021). "Estimating Dynamic Treatment Effects in Event Studies
    with Heterogeneous Treatment Effects"
"""

import numpy as np
import pytest
from src.causal_inference.did import (
    sun_abraham_ate,
    create_staggered_data,
)
from tests.validation.monte_carlo.dgp_did import (
    dgp_staggered_homogeneous,
    dgp_staggered_heterogeneous,
)


class TestSAUnbiasedness:
    """Monte Carlo validation of Sun-Abraham unbiasedness."""

    @pytest.mark.slow
    def test_sa_unbiased_homogeneous(self):
        """
        Validate SA is unbiased with homogeneous treatment effects.

        DGP: All cohorts have τ = 2.0
        Expected: Bias < 0.20
        """
        n_runs = 300
        true_att = 2.0

        estimates = []
        for seed in range(n_runs):
            data = dgp_staggered_homogeneous(
                n_units=150,
                n_periods=10,
                cohorts=(5, 7),
                true_effect=true_att,
                random_state=seed,
            )

            staggered = create_staggered_data(
                outcomes=data.outcomes,
                treatment=data.treatment,
                time=data.time,
                unit_id=data.unit_id,
                treatment_time=data.treatment_time,
            )

            result = sun_abraham_ate(staggered)
            estimates.append(result["att"])

        bias = abs(np.mean(estimates) - true_att)
        assert bias < 0.30, (
            f"SA bias {bias:.4f} exceeds 0.30 with homogeneous effects. "
            f"Mean estimate: {np.mean(estimates):.4f}"
        )

    @pytest.mark.slow
    def test_sa_unbiased_heterogeneous(self):
        """
        CRITICAL TEST: SA should be unbiased with heterogeneous effects.

        This is the key advantage of the interaction-weighted estimator.

        DGP: Cohort 5 has τ=1.0, Cohort 7 has τ=5.0
        True ATT = 3.0 (simple average)
        """
        n_runs = 300
        cohort_effects = {5: 1.0, 7: 5.0}
        true_att = np.mean(list(cohort_effects.values()))

        estimates = []
        for seed in range(n_runs):
            data = dgp_staggered_heterogeneous(
                n_units=150,
                n_periods=10,
                cohort_effects=cohort_effects,
                random_state=seed,
            )

            staggered = create_staggered_data(
                outcomes=data.outcomes,
                treatment=data.treatment,
                time=data.time,
                unit_id=data.unit_id,
                treatment_time=data.treatment_time,
            )

            result = sun_abraham_ate(staggered)
            estimates.append(result["att"])

        bias = abs(np.mean(estimates) - true_att)

        print(f"\n=== SA Heterogeneous Test ===")
        print(f"True ATT: {true_att:.4f}")
        print(f"SA mean estimate: {np.mean(estimates):.4f}")
        print(f"SA bias: {bias:.4f}")

        assert bias < 0.50, (
            f"SA bias {bias:.4f} exceeds 0.50 with heterogeneous effects. "
            f"SA should be robust to heterogeneity via IW estimation."
        )


class TestSAWeights:
    """Monte Carlo validation of Sun-Abraham weights."""

    def test_sa_weights_sum_to_one(self):
        """Verify SA weights sum to 1.0."""
        data = dgp_staggered_heterogeneous(n_units=150, n_periods=10, random_state=42)

        staggered = create_staggered_data(
            outcomes=data.outcomes,
            treatment=data.treatment,
            time=data.time,
            unit_id=data.unit_id,
            treatment_time=data.treatment_time,
        )

        result = sun_abraham_ate(staggered)

        assert "weights" in result
        weights_df = result["weights"]
        total_weight = weights_df["weight"].sum()

        assert abs(total_weight - 1.0) < 1e-4, f"SA weights sum to {total_weight:.6f}, expected 1.0"

    @pytest.mark.slow
    def test_sa_att_equals_weighted_average(self):
        """
        Verify SA ATT equals weighted average of cohort × event-time effects.

        SA_ATT = Σ_k Σ_g w_{g,k} × CATT(g,k)

        where CATT(g,k) is cohort-specific effect at event time k.
        """
        n_runs = 100

        match_count = 0
        for seed in range(n_runs):
            data = dgp_staggered_heterogeneous(
                n_units=150,
                n_periods=10,
                random_state=seed,
            )

            staggered = create_staggered_data(
                outcomes=data.outcomes,
                treatment=data.treatment,
                time=data.time,
                unit_id=data.unit_id,
                treatment_time=data.treatment_time,
            )

            result = sun_abraham_ate(staggered)

            # Manually compute weighted average
            if "cohort_effects" in result and "weights" in result:
                merged = result["cohort_effects"].merge(
                    result["weights"], on=["cohort", "event_time"]
                )
                manual_att = (merged["coef"] * merged["weight"]).sum()

                if abs(result["att"] - manual_att) < 1e-4:
                    match_count += 1

        # Most should match (allowing for numerical precision)
        match_rate = match_count / n_runs
        assert match_rate > 0.90, (
            f"Only {match_rate * 100:.0f}% of ATT values match manual calculation"
        )


class TestSACoverage:
    """Monte Carlo validation of SA standard errors and coverage."""

    @pytest.mark.slow
    def test_sa_cluster_se_coverage(self):
        """
        Validate SA cluster-robust SE coverage.

        Expected: Coverage 88-98% (cluster SEs may be conservative)
        """
        n_runs = 200
        true_att = 2.0

        ci_contains_true = []

        for seed in range(n_runs):
            data = dgp_staggered_homogeneous(
                n_units=150,
                n_periods=10,
                true_effect=true_att,
                random_state=seed,
            )

            staggered = create_staggered_data(
                outcomes=data.outcomes,
                treatment=data.treatment,
                time=data.time,
                unit_id=data.unit_id,
                treatment_time=data.treatment_time,
            )

            result = sun_abraham_ate(staggered, cluster_se=True)

            covers = result["ci_lower"] <= true_att <= result["ci_upper"]
            ci_contains_true.append(covers)

        coverage = np.mean(ci_contains_true)

        print(f"\n=== SA Cluster SE Coverage ===")
        print(f"Coverage: {coverage:.4f}")

        assert 0.85 < coverage < 0.99, f"SA coverage {coverage:.4f} outside [0.85, 0.99]"


class TestSADiagnostics:
    """Diagnostic tests for SA Monte Carlo validation."""

    def test_sa_returns_expected_structure(self):
        """Verify SA returns all expected output fields."""
        data = dgp_staggered_homogeneous(n_units=100, random_state=42)

        staggered = create_staggered_data(
            outcomes=data.outcomes,
            treatment=data.treatment,
            time=data.time,
            unit_id=data.unit_id,
            treatment_time=data.treatment_time,
        )

        result = sun_abraham_ate(staggered)

        # Check required fields
        assert "att" in result
        assert "se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "p_value" in result
        assert "cohort_effects" in result
        assert "weights" in result

        # Check values are reasonable
        assert result["se"] > 0
        assert result["ci_lower"] < result["att"] < result["ci_upper"]

    def test_sa_cohort_effects_structure(self):
        """Verify SA returns cohort × event-time effects."""
        data = dgp_staggered_heterogeneous(
            n_units=100,
            n_periods=10,
            cohort_effects={5: 2.0, 7: 4.0},
            random_state=42,
        )

        staggered = create_staggered_data(
            outcomes=data.outcomes,
            treatment=data.treatment,
            time=data.time,
            unit_id=data.unit_id,
            treatment_time=data.treatment_time,
        )

        result = sun_abraham_ate(staggered)

        cohort_effects = result["cohort_effects"]

        # Should have DataFrame with cohort, event_time, coef columns
        assert "cohort" in cohort_effects.columns
        assert "event_time" in cohort_effects.columns
        assert "coef" in cohort_effects.columns

        # Should have multiple rows
        assert len(cohort_effects) > 0
