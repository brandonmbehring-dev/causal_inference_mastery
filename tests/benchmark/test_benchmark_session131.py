"""Session 131 Benchmark Smoke Tests.

These tests verify that all Session 131 benchmark functions execute without error.
Each test runs at small sample size with minimal repetitions.

Usage:
    pytest tests/benchmark/test_benchmark_session131.py -v
    pytest tests/benchmark/test_benchmark_session131.py -v -m "not slow"
"""

from __future__ import annotations

import pytest


# =============================================================================
# SCM Benchmarks
# =============================================================================


class TestSCMBenchmarks:
    """Smoke tests for Synthetic Control Method benchmarks."""

    @pytest.mark.benchmark
    def test_synthetic_control_smoke(self, small_config):
        """synthetic_control benchmark runs without error."""
        from benchmarks.methods.scm import benchmark_synthetic_control

        result = benchmark_synthetic_control(
            n_control=10,
            n_periods=10,
            seed=42,
            n_repetitions=2,
        )
        assert result.median_time_ms > 0
        assert result.family == "scm"

    @pytest.mark.benchmark
    def test_augmented_scm_smoke(self, small_config):
        """augmented_scm benchmark runs without error."""
        from benchmarks.methods.scm import benchmark_augmented_scm

        result = benchmark_augmented_scm(
            n_control=10,
            n_periods=10,
            seed=42,
            n_repetitions=2,
        )
        assert result.median_time_ms > 0


# =============================================================================
# CATE Benchmarks
# =============================================================================


class TestCATEBenchmarks:
    """Smoke tests for CATE meta-learner benchmarks."""

    @pytest.mark.benchmark
    def test_s_learner_smoke(self, small_config):
        """s_learner benchmark runs without error."""
        from benchmarks.methods.cate import benchmark_s_learner

        result = benchmark_s_learner(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0
        assert result.family == "cate"

    @pytest.mark.benchmark
    def test_t_learner_smoke(self, small_config):
        """t_learner benchmark runs without error."""
        from benchmarks.methods.cate import benchmark_t_learner

        result = benchmark_t_learner(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_x_learner_smoke(self, small_config):
        """x_learner benchmark runs without error."""
        from benchmarks.methods.cate import benchmark_x_learner

        result = benchmark_x_learner(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_r_learner_smoke(self, small_config):
        """r_learner benchmark runs without error."""
        from benchmarks.methods.cate import benchmark_r_learner

        result = benchmark_r_learner(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_double_ml_smoke(self, small_config):
        """double_ml benchmark runs without error."""
        from benchmarks.methods.cate import benchmark_double_ml

        result = benchmark_double_ml(n=100, seed=42, n_repetitions=2, n_folds=2)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_causal_forest_smoke(self, small_config):
        """causal_forest benchmark runs (slower due to tree fitting)."""
        from benchmarks.methods.cate import benchmark_causal_forest

        result = benchmark_causal_forest(n=100, seed=42, n_repetitions=2, n_trees=10)
        assert result.median_time_ms > 0


# =============================================================================
# RKD Benchmarks
# =============================================================================


class TestRKDBenchmarks:
    """Smoke tests for Regression Kink Design benchmarks."""

    @pytest.mark.benchmark
    def test_sharp_rkd_smoke(self, small_config):
        """sharp_rkd benchmark runs without error."""
        from benchmarks.methods.rkd import benchmark_sharp_rkd

        result = benchmark_sharp_rkd(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0
        assert result.family == "rkd"

    @pytest.mark.benchmark
    def test_fuzzy_rkd_smoke(self, small_config):
        """fuzzy_rkd benchmark runs without error."""
        from benchmarks.methods.rkd import benchmark_fuzzy_rkd

        result = benchmark_fuzzy_rkd(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0


# =============================================================================
# Panel Benchmarks
# =============================================================================


class TestPanelBenchmarks:
    """Smoke tests for Panel Data benchmarks."""

    @pytest.mark.benchmark
    def test_dml_cre_smoke(self, small_config):
        """dml_cre benchmark runs without error."""
        from benchmarks.methods.panel import benchmark_dml_cre

        result = benchmark_dml_cre(n_units=20, n_periods=4, seed=42, n_repetitions=2, n_folds=2)
        assert result.median_time_ms > 0
        assert result.family == "panel"

    @pytest.mark.benchmark
    def test_dml_cre_continuous_smoke(self, small_config):
        """dml_cre_continuous benchmark runs without error."""
        from benchmarks.methods.panel import benchmark_dml_cre_continuous

        result = benchmark_dml_cre_continuous(
            n_units=20, n_periods=4, seed=42, n_repetitions=2, n_folds=2
        )
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_panel_rif_qte_smoke(self, small_config):
        """panel_rif_qte benchmark runs without error."""
        from benchmarks.methods.panel import benchmark_panel_rif_qte

        result = benchmark_panel_rif_qte(n_units=20, n_periods=4, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0


# =============================================================================
# Sensitivity Benchmarks
# =============================================================================


class TestSensitivityBenchmarks:
    """Smoke tests for Sensitivity Analysis benchmarks."""

    @pytest.mark.benchmark
    def test_e_value_smoke(self, small_config):
        """e_value benchmark runs without error."""
        from benchmarks.methods.sensitivity import benchmark_e_value

        result = benchmark_e_value(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0
        assert result.family == "sensitivity"

    @pytest.mark.benchmark
    def test_rosenbaum_bounds_smoke(self, small_config):
        """rosenbaum_bounds benchmark runs without error."""
        from benchmarks.methods.sensitivity import benchmark_rosenbaum_bounds

        result = benchmark_rosenbaum_bounds(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0


# =============================================================================
# Bayesian Benchmarks
# =============================================================================


class TestBayesianBenchmarks:
    """Smoke tests for Bayesian Causal Inference benchmarks."""

    @pytest.mark.benchmark
    def test_bayesian_ate_smoke(self, small_config):
        """bayesian_ate benchmark runs without error."""
        from benchmarks.methods.bayesian import benchmark_bayesian_ate

        result = benchmark_bayesian_ate(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0
        assert result.family == "bayesian"

    @pytest.mark.benchmark
    def test_bayesian_propensity_smoke(self, small_config):
        """bayesian_propensity benchmark runs without error."""
        from benchmarks.methods.bayesian import benchmark_bayesian_propensity

        result = benchmark_bayesian_propensity(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_bayesian_dr_ate_smoke(self, small_config):
        """bayesian_dr_ate benchmark runs without error."""
        from benchmarks.methods.bayesian import benchmark_bayesian_dr_ate

        result = benchmark_bayesian_dr_ate(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0


# =============================================================================
# Principal Stratification Benchmarks
# =============================================================================


class TestPrincipalStratBenchmarks:
    """Smoke tests for Principal Stratification benchmarks."""

    @pytest.mark.benchmark
    def test_cace_2sls_smoke(self, small_config):
        """cace_2sls benchmark runs without error."""
        from benchmarks.methods.principal_strat import benchmark_cace_2sls

        result = benchmark_cace_2sls(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0
        assert result.family == "principal_strat"

    @pytest.mark.benchmark
    def test_cace_em_smoke(self, small_config):
        """cace_em benchmark runs without error."""
        from benchmarks.methods.principal_strat import benchmark_cace_em

        result = benchmark_cace_em(n=100, seed=42, n_repetitions=2, max_iter=20)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_sace_bounds_smoke(self, small_config):
        """sace_bounds benchmark runs without error."""
        from benchmarks.methods.principal_strat import benchmark_sace_bounds

        result = benchmark_sace_bounds(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0


# =============================================================================
# Bounds Benchmarks
# =============================================================================


class TestBoundsBenchmarks:
    """Smoke tests for Partial Identification / Bounds benchmarks."""

    @pytest.mark.benchmark
    def test_manski_worst_case_smoke(self, small_config):
        """manski_worst_case benchmark runs without error."""
        from benchmarks.methods.bounds import benchmark_manski_worst_case

        result = benchmark_manski_worst_case(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0
        assert result.family == "bounds"

    @pytest.mark.benchmark
    def test_manski_mtr_smoke(self, small_config):
        """manski_mtr benchmark runs without error."""
        from benchmarks.methods.bounds import benchmark_manski_mtr

        result = benchmark_manski_mtr(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0

    @pytest.mark.benchmark
    def test_lee_bounds_smoke(self, small_config):
        """lee_bounds benchmark runs without error."""
        from benchmarks.methods.bounds import benchmark_lee_bounds

        result = benchmark_lee_bounds(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0


# =============================================================================
# QTE Benchmarks
# =============================================================================


class TestQTEBenchmarks:
    """Smoke tests for Quantile Treatment Effects benchmarks."""

    @pytest.mark.benchmark
    def test_unconditional_qte_smoke(self, small_config):
        """unconditional_qte benchmark runs without error."""
        from benchmarks.methods.qte import benchmark_unconditional_qte

        result = benchmark_unconditional_qte(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0
        assert result.family == "qte"

    @pytest.mark.benchmark
    def test_conditional_qte_smoke(self, small_config):
        """conditional_qte benchmark runs without error."""
        from benchmarks.methods.qte import benchmark_conditional_qte

        result = benchmark_conditional_qte(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0


# =============================================================================
# Bunching Benchmarks
# =============================================================================


class TestBunchingBenchmarks:
    """Smoke tests for Bunching Estimator benchmarks."""

    @pytest.mark.benchmark
    def test_bunching_estimator_smoke(self, small_config):
        """bunching_estimator benchmark runs without error."""
        from benchmarks.methods.bunching import benchmark_bunching_estimator

        result = benchmark_bunching_estimator(n=500, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0
        assert result.family == "bunching"


# =============================================================================
# Selection Benchmarks
# =============================================================================


class TestSelectionBenchmarks:
    """Smoke tests for Selection Correction benchmarks."""

    @pytest.mark.benchmark
    def test_heckman_two_step_smoke(self, small_config):
        """heckman_two_step benchmark runs without error."""
        from benchmarks.methods.selection import benchmark_heckman_two_step

        result = benchmark_heckman_two_step(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0
        assert result.family == "selection"


# =============================================================================
# MTE Benchmarks
# =============================================================================


class TestMTEBenchmarks:
    """Smoke tests for Marginal Treatment Effects benchmarks."""

    @pytest.mark.benchmark
    def test_local_iv_smoke(self, small_config):
        """local_iv benchmark runs without error."""
        from benchmarks.methods.mte import benchmark_local_iv

        result = benchmark_local_iv(n=200, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0
        assert result.family == "mte"


# =============================================================================
# Mediation Benchmarks
# =============================================================================


class TestMediationBenchmarks:
    """Smoke tests for Mediation Analysis benchmarks."""

    @pytest.mark.benchmark
    def test_baron_kenny_smoke(self, small_config):
        """baron_kenny benchmark runs without error."""
        from benchmarks.methods.mediation import benchmark_baron_kenny

        result = benchmark_baron_kenny(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0
        assert result.family == "mediation"

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_mediation_analysis_smoke(self, small_config):
        """mediation_analysis benchmark runs (slower due to bootstrap)."""
        from benchmarks.methods.mediation import benchmark_mediation_analysis

        result = benchmark_mediation_analysis(n=100, seed=42, n_repetitions=2, n_bootstrap=20)
        assert result.median_time_ms > 0


# =============================================================================
# Control Function Benchmarks
# =============================================================================


class TestControlFunctionBenchmarks:
    """Smoke tests for Control Function benchmarks."""

    @pytest.mark.benchmark
    def test_control_function_ate_smoke(self, small_config):
        """control_function_ate benchmark runs without error."""
        from benchmarks.methods.control_function import benchmark_control_function_ate

        result = benchmark_control_function_ate(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0
        assert result.family == "control_function"


# =============================================================================
# Shift-Share Benchmarks
# =============================================================================


class TestShiftShareBenchmarks:
    """Smoke tests for Shift-Share IV benchmarks."""

    @pytest.mark.benchmark
    def test_shift_share_iv_smoke(self, small_config):
        """shift_share_iv benchmark runs without error."""
        from benchmarks.methods.shift_share import benchmark_shift_share_iv

        result = benchmark_shift_share_iv(
            n_regions=20, n_industries=5, n_periods=5, seed=42, n_repetitions=2
        )
        assert result.median_time_ms > 0
        assert result.family == "shift_share"


# =============================================================================
# DTR Benchmarks
# =============================================================================


class TestDTRBenchmarks:
    """Smoke tests for Dynamic Treatment Regime benchmarks."""

    @pytest.mark.benchmark
    def test_q_learning_single_stage_smoke(self, small_config):
        """q_learning_single_stage benchmark runs without error."""
        from benchmarks.methods.dtr import benchmark_q_learning_single_stage

        result = benchmark_q_learning_single_stage(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0
        assert result.family == "dtr"

    @pytest.mark.benchmark
    def test_a_learning_single_stage_smoke(self, small_config):
        """a_learning_single_stage benchmark runs without error."""
        from benchmarks.methods.dtr import benchmark_a_learning_single_stage

        result = benchmark_a_learning_single_stage(n=100, seed=42, n_repetitions=2)
        assert result.median_time_ms > 0
