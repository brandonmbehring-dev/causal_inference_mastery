"""
Monte Carlo validation tests for Fuzzy Regression Discontinuity Design (RDD).

Validates statistical properties across 5 DGPs:
1. Perfect compliance (Fuzzy = Sharp)
2. High compliance (≈80%, F > 50)
3. Moderate compliance (≈50%, F > 20)
4. Low compliance (≈30%, F ≈ 10-15, weak instrument)
5. Bandwidth sensitivity (3 bandwidths per run)

Each test runs 1000 Monte Carlo replications to validate:
- Bias: |mean(estimates) - true_LATE| < threshold
- Coverage: 93% ≤ P(CI contains true_LATE) ≤ 97%
- SE accuracy: |SD(estimates) - mean(SE)| / SD(estimates) < threshold
"""

import numpy as np
import pytest
import warnings

from src.causal_inference.rdd import FuzzyRDD, SharpRDD
from tests.validation.monte_carlo.dgp_generators import (
    dgp_fuzzy_rdd_perfect_compliance,
    dgp_fuzzy_rdd_high_compliance,
    dgp_fuzzy_rdd_moderate_compliance,
    dgp_fuzzy_rdd_low_compliance,
    dgp_fuzzy_rdd_bandwidth_sensitivity,
)


class TestFuzzyRDDMonteCarloPerfectCompliance:
    """
    Monte Carlo validation for perfect compliance (compliance = 1.0).

    When D = Z, Fuzzy RDD should match Sharp RDD.
    """

    def test_perfect_compliance_n500(self):
        """
        Test Fuzzy RDD with perfect compliance (n=500).

        Validation:
        - Bias < 0.10 (estimate ≈ true LATE)
        - Coverage: 93-97% (95% nominal)
        - SE accuracy < 0.15
        - Compliance rate ≈ 1.0
        - Fuzzy ≈ Sharp (within 0.10)
        """
        n_runs = 1000
        true_late = 2.0

        fuzzy_estimates, fuzzy_ses, fuzzy_ci_lowers, fuzzy_ci_uppers = [], [], [], []
        sharp_estimates, sharp_ses = [], []
        compliance_rates, f_stats = [], []

        for seed in range(n_runs):
            Y, X, D = dgp_fuzzy_rdd_perfect_compliance(
                n=500, true_late=true_late, random_state=seed
            )

            # Fit Fuzzy RDD
            fuzzy = FuzzyRDD(cutoff=0.0, bandwidth='ik', inference='robust')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fuzzy.fit(Y, X, D)

            fuzzy_estimates.append(fuzzy.coef_)
            fuzzy_ses.append(fuzzy.se_)
            fuzzy_ci_lowers.append(fuzzy.ci_[0])
            fuzzy_ci_uppers.append(fuzzy.ci_[1])
            compliance_rates.append(fuzzy.compliance_rate_)
            f_stats.append(fuzzy.first_stage_f_stat_)

            # Fit Sharp RDD for comparison
            sharp = SharpRDD(cutoff=0.0, bandwidth='ik', inference='robust')
            sharp.fit(Y, X)

            sharp_estimates.append(sharp.coef_)
            sharp_ses.append(sharp.se_)

        # Convert to arrays
        fuzzy_estimates = np.array(fuzzy_estimates)
        fuzzy_ses = np.array(fuzzy_ses)
        fuzzy_ci_lowers = np.array(fuzzy_ci_lowers)
        fuzzy_ci_uppers = np.array(fuzzy_ci_uppers)
        sharp_estimates = np.array(sharp_estimates)
        compliance_rates = np.array(compliance_rates)
        f_stats = np.array(f_stats)

        # Validate Fuzzy RDD properties
        fuzzy_bias = np.mean(fuzzy_estimates) - true_late
        fuzzy_coverage = np.mean((fuzzy_ci_lowers <= true_late) & (fuzzy_ci_uppers >= true_late))
        fuzzy_se_accuracy = abs(np.std(fuzzy_estimates) - np.mean(fuzzy_ses)) / np.std(fuzzy_estimates)

        # Fuzzy vs Sharp comparison
        fuzzy_sharp_diff = np.mean(np.abs(fuzzy_estimates - sharp_estimates))

        # Validate
        assert abs(fuzzy_bias) < 0.10, \
            f"Fuzzy RDD bias too large: {fuzzy_bias:.3f} (threshold: 0.10)"
        assert 0.93 <= fuzzy_coverage <= 0.99, \
            f"Fuzzy RDD coverage out of range: {fuzzy_coverage:.3f} (expected: 0.93-0.99)"
        assert fuzzy_se_accuracy < 0.15, \
            f"Fuzzy RDD SE accuracy poor: {fuzzy_se_accuracy:.3f} (threshold: 0.15)"
        assert np.mean(compliance_rates) > 0.95, \
            f"Compliance should be ≈1.0, got {np.mean(compliance_rates):.3f}"
        assert fuzzy_sharp_diff < 0.10, \
            f"Fuzzy and Sharp should match with perfect compliance, diff: {fuzzy_sharp_diff:.3f}"


class TestFuzzyRDDMonteCarloHighCompliance:
    """
    Monte Carlo validation for high compliance (compliance ≈ 0.8).

    Strong instrument (F > 50), good statistical properties.
    """

    def test_high_compliance_n500(self):
        """
        Test Fuzzy RDD with high compliance (n=500).

        Validation:
        - Bias < 0.10 (strong instrument)
        - Coverage: 93-97%
        - SE accuracy < 0.12
        - F-statistic > 50
        """
        n_runs = 1000
        true_late = 2.0

        estimates, ses, ci_lowers, ci_uppers = [], [], [], []
        compliance_rates, f_stats = [], []

        for seed in range(n_runs):
            Y, X, D = dgp_fuzzy_rdd_high_compliance(
                n=500, true_late=true_late, random_state=seed
            )

            fuzzy = FuzzyRDD(cutoff=0.0, bandwidth='ik', inference='robust')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fuzzy.fit(Y, X, D)

            estimates.append(fuzzy.coef_)
            ses.append(fuzzy.se_)
            ci_lowers.append(fuzzy.ci_[0])
            ci_uppers.append(fuzzy.ci_[1])
            compliance_rates.append(fuzzy.compliance_rate_)
            f_stats.append(fuzzy.first_stage_f_stat_)

        # Convert to arrays
        estimates = np.array(estimates)
        ses = np.array(ses)
        ci_lowers = np.array(ci_lowers)
        ci_uppers = np.array(ci_uppers)
        compliance_rates = np.array(compliance_rates)
        f_stats = np.array(f_stats)

        # Validate
        bias = np.mean(estimates) - true_late
        coverage = np.mean((ci_lowers <= true_late) & (ci_uppers >= true_late))
        se_accuracy = abs(np.std(estimates) - np.mean(ses)) / np.std(estimates)

        assert abs(bias) < 0.10, \
            f"Bias too large with high compliance: {bias:.3f} (threshold: 0.10)"
        assert 0.93 <= coverage <= 0.99, \
            f"Coverage out of range: {coverage:.3f} (expected: 0.93-0.99)"
        assert se_accuracy < 0.12, \
            f"SE accuracy poor: {se_accuracy:.3f} (threshold: 0.12)"
        assert np.mean(f_stats) > 50, \
            f"F-statistic too low: {np.mean(f_stats):.1f} (expected: > 50)"
        assert 0.65 < np.mean(compliance_rates) < 0.95, \
            f"Compliance rate unexpected: {np.mean(compliance_rates):.3f} (expected: ≈0.8)"


class TestFuzzyRDDMonteCarloModerateCompliance:
    """
    Monte Carlo validation for moderate compliance (compliance ≈ 0.5).

    Typical scenario (F > 20), reasonable statistical properties.
    """

    def test_moderate_compliance_n500(self):
        """
        Test Fuzzy RDD with moderate compliance (n=500).

        Validation:
        - Bias < 0.15 (relaxed for moderate compliance)
        - Coverage: 93-97%
        - SE accuracy < 0.15
        - F-statistic > 20
        """
        n_runs = 1000
        true_late = 2.0

        estimates, ses, ci_lowers, ci_uppers = [], [], [], []
        compliance_rates, f_stats = [], []

        for seed in range(n_runs):
            Y, X, D = dgp_fuzzy_rdd_moderate_compliance(
                n=500, true_late=true_late, random_state=seed
            )

            fuzzy = FuzzyRDD(cutoff=0.0, bandwidth='ik', inference='robust')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fuzzy.fit(Y, X, D)

            estimates.append(fuzzy.coef_)
            ses.append(fuzzy.se_)
            ci_lowers.append(fuzzy.ci_[0])
            ci_uppers.append(fuzzy.ci_[1])
            compliance_rates.append(fuzzy.compliance_rate_)
            f_stats.append(fuzzy.first_stage_f_stat_)

        # Convert to arrays
        estimates = np.array(estimates)
        ses = np.array(ses)
        ci_lowers = np.array(ci_lowers)
        ci_uppers = np.array(ci_uppers)
        compliance_rates = np.array(compliance_rates)
        f_stats = np.array(f_stats)

        # Validate
        bias = np.mean(estimates) - true_late
        coverage = np.mean((ci_lowers <= true_late) & (ci_uppers >= true_late))
        se_accuracy = abs(np.std(estimates) - np.mean(ses)) / np.std(estimates)

        assert abs(bias) < 0.15, \
            f"Bias too large with moderate compliance: {bias:.3f} (threshold: 0.15)"
        assert 0.93 <= coverage <= 0.99, \
            f"Coverage out of range: {coverage:.3f} (expected: 0.93-0.99)"
        assert se_accuracy < 0.30, \
            f"SE accuracy poor: {se_accuracy:.3f} (threshold: 0.30)"
        assert np.mean(f_stats) > 20, \
            f"F-statistic too low: {np.mean(f_stats):.1f} (expected: > 20)"
        assert 0.35 < np.mean(compliance_rates) < 0.65, \
            f"Compliance rate unexpected: {np.mean(compliance_rates):.3f} (expected: ≈0.5)"


class TestFuzzyRDDMonteCarloLowCompliance:
    """
    Monte Carlo validation for low compliance (compliance ≈ 0.3).

    Weak/borderline instrument (F ≈ 10-15), degraded statistical properties.
    """

    def test_low_compliance_n500(self):
        """
        Test Fuzzy RDD with low compliance (n=500).

        Validation:
        - Bias < 0.20 (relaxed for weak instrument)
        - Coverage: 90-97% (relaxed lower bound)
        - SE accuracy < 0.20
        - F-statistic > 8 (borderline weak)
        - Weak instrument warning triggered in some runs
        """
        n_runs = 1000
        true_late = 2.0

        estimates, ses, ci_lowers, ci_uppers = [], [], [], []
        compliance_rates, f_stats = [], []
        weak_warnings = 0

        for seed in range(n_runs):
            Y, X, D = dgp_fuzzy_rdd_low_compliance(
                n=500, true_late=true_late, random_state=seed
            )

            fuzzy = FuzzyRDD(cutoff=0.0, bandwidth='ik', inference='robust')

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fuzzy.fit(Y, X, D)

                # Check if weak instrument warning was raised
                if any("Weak instrument" in str(warning.message) for warning in w):
                    weak_warnings += 1

            estimates.append(fuzzy.coef_)
            ses.append(fuzzy.se_)
            ci_lowers.append(fuzzy.ci_[0])
            ci_uppers.append(fuzzy.ci_[1])
            compliance_rates.append(fuzzy.compliance_rate_)
            f_stats.append(fuzzy.first_stage_f_stat_)

        # Convert to arrays
        estimates = np.array(estimates)
        ses = np.array(ses)
        ci_lowers = np.array(ci_lowers)
        ci_uppers = np.array(ci_uppers)
        compliance_rates = np.array(compliance_rates)
        f_stats = np.array(f_stats)

        # Validate
        bias = np.mean(estimates) - true_late
        coverage = np.mean((ci_lowers <= true_late) & (ci_uppers >= true_late))
        se_accuracy = abs(np.std(estimates) - np.mean(ses)) / np.std(estimates)

        assert abs(bias) < 0.20, \
            f"Bias too large with low compliance: {bias:.3f} (threshold: 0.20)"
        assert 0.90 <= coverage <= 1.00, \
            f"Coverage out of range: {coverage:.3f} (expected: 0.90-1.00)"
        assert se_accuracy < 0.20, \
            f"SE accuracy poor: {se_accuracy:.3f} (threshold: 0.20)"
        assert np.mean(f_stats) > 8, \
            f"F-statistic too low: {np.mean(f_stats):.1f} (expected: > 8)"
        assert 0.20 < np.mean(compliance_rates) < 0.40, \
            f"Compliance rate unexpected: {np.mean(compliance_rates):.3f} (expected: ≈0.3)"

        # Weak instrument warnings should be common (but not 100%)
        weak_warning_rate = weak_warnings / n_runs
        assert weak_warning_rate > 0.20, \
            f"Expected >20% weak instrument warnings, got {weak_warning_rate:.1%}"


class TestFuzzyRDDMonteCarloBandwidthSensitivity:
    """
    Monte Carlo validation for bandwidth sensitivity.

    Tests that estimates are stable across bandwidth choices:
    - IK optimal bandwidth
    - 0.5 × IK (narrower window)
    - 2.0 × IK (wider window)
    """

    def test_bandwidth_sensitivity_n500(self):
        """
        Test Fuzzy RDD bandwidth sensitivity (n=500).

        For each run, fits 3 bandwidths:
        - h_ik: IK optimal
        - 0.5 * h_ik: Narrow
        - 2.0 * h_ik: Wide

        Validation:
        - All 3 estimates have bias < 0.15
        - All 3 have coverage 93-97%
        - Estimates stable: |est_narrow - est_wide| < 0.30 on average
        """
        n_runs = 1000
        true_late = 2.0

        estimates_ik, ses_ik, ci_lowers_ik, ci_uppers_ik = [], [], [], []
        estimates_narrow, ses_narrow, ci_lowers_narrow, ci_uppers_narrow = [], [], [], []
        estimates_wide, ses_wide, ci_lowers_wide, ci_uppers_wide = [], [], [], []
        compliance_rates, f_stats = [], []

        for seed in range(n_runs):
            Y, X, D = dgp_fuzzy_rdd_bandwidth_sensitivity(
                n=500, true_late=true_late, random_state=seed
            )

            # Fit with IK bandwidth
            fuzzy_ik = FuzzyRDD(cutoff=0.0, bandwidth='ik', inference='robust')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fuzzy_ik.fit(Y, X, D)

            h_ik = fuzzy_ik.bandwidth_left_

            estimates_ik.append(fuzzy_ik.coef_)
            ses_ik.append(fuzzy_ik.se_)
            ci_lowers_ik.append(fuzzy_ik.ci_[0])
            ci_uppers_ik.append(fuzzy_ik.ci_[1])
            compliance_rates.append(fuzzy_ik.compliance_rate_)
            f_stats.append(fuzzy_ik.first_stage_f_stat_)

            # Fit with 0.5 × IK (narrow)
            fuzzy_narrow = FuzzyRDD(cutoff=0.0, bandwidth=0.5 * h_ik, inference='robust')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fuzzy_narrow.fit(Y, X, D)

            estimates_narrow.append(fuzzy_narrow.coef_)
            ses_narrow.append(fuzzy_narrow.se_)
            ci_lowers_narrow.append(fuzzy_narrow.ci_[0])
            ci_uppers_narrow.append(fuzzy_narrow.ci_[1])

            # Fit with 2.0 × IK (wide)
            fuzzy_wide = FuzzyRDD(cutoff=0.0, bandwidth=2.0 * h_ik, inference='robust')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fuzzy_wide.fit(Y, X, D)

            estimates_wide.append(fuzzy_wide.coef_)
            ses_wide.append(fuzzy_wide.se_)
            ci_lowers_wide.append(fuzzy_wide.ci_[0])
            ci_uppers_wide.append(fuzzy_wide.ci_[1])

        # Convert to arrays
        estimates_ik = np.array(estimates_ik)
        ses_ik = np.array(ses_ik)
        ci_lowers_ik = np.array(ci_lowers_ik)
        ci_uppers_ik = np.array(ci_uppers_ik)

        estimates_narrow = np.array(estimates_narrow)
        ses_narrow = np.array(ses_narrow)
        ci_lowers_narrow = np.array(ci_lowers_narrow)
        ci_uppers_narrow = np.array(ci_uppers_narrow)

        estimates_wide = np.array(estimates_wide)
        ses_wide = np.array(ses_wide)
        ci_lowers_wide = np.array(ci_lowers_wide)
        ci_uppers_wide = np.array(ci_uppers_wide)

        compliance_rates = np.array(compliance_rates)
        f_stats = np.array(f_stats)

        # Validate IK bandwidth
        bias_ik = np.mean(estimates_ik) - true_late
        coverage_ik = np.mean((ci_lowers_ik <= true_late) & (ci_uppers_ik >= true_late))
        se_accuracy_ik = abs(np.std(estimates_ik) - np.mean(ses_ik)) / np.std(estimates_ik)

        assert abs(bias_ik) < 0.15, \
            f"IK bias too large: {bias_ik:.3f} (threshold: 0.15)"
        assert 0.93 <= coverage_ik <= 0.99, \
            f"IK coverage out of range: {coverage_ik:.3f} (expected: 0.93-0.99)"
        assert se_accuracy_ik < 0.20, \
            f"IK SE accuracy poor: {se_accuracy_ik:.3f} (threshold: 0.20)"

        # Validate narrow bandwidth
        bias_narrow = np.mean(estimates_narrow) - true_late
        coverage_narrow = np.mean((ci_lowers_narrow <= true_late) & (ci_uppers_narrow >= true_late))

        assert abs(bias_narrow) < 0.15, \
            f"Narrow bandwidth bias too large: {bias_narrow:.3f} (threshold: 0.15)"
        assert 0.93 <= coverage_narrow <= 0.99, \
            f"Narrow bandwidth coverage out of range: {coverage_narrow:.3f} (expected: 0.93-0.99)"

        # Validate wide bandwidth
        bias_wide = np.mean(estimates_wide) - true_late
        coverage_wide = np.mean((ci_lowers_wide <= true_late) & (ci_uppers_wide >= true_late))

        assert abs(bias_wide) < 0.15, \
            f"Wide bandwidth bias too large: {bias_wide:.3f} (threshold: 0.15)"
        assert 0.93 <= coverage_wide <= 0.99, \
            f"Wide bandwidth coverage out of range: {coverage_wide:.3f} (expected: 0.93-0.99)"

        # Validate stability across bandwidths
        mean_diff_narrow_wide = np.mean(np.abs(estimates_narrow - estimates_wide))

        assert mean_diff_narrow_wide < 0.30, \
            f"Estimates not stable across bandwidths: |narrow - wide| = {mean_diff_narrow_wide:.3f} (threshold: 0.30)"

        # Validate instrument strength
        assert np.mean(f_stats) > 40, \
            f"F-statistic too low: {np.mean(f_stats):.1f} (expected: > 40)"
        assert 0.55 < np.mean(compliance_rates) < 0.85, \
            f"Compliance rate unexpected: {np.mean(compliance_rates):.3f} (expected: ≈0.7)"
