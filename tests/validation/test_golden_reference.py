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

from causal_inference.rct.estimators import simple_ate
from causal_inference.rct.estimators_stratified import stratified_ate
from causal_inference.rct.estimators_regression import regression_adjusted_ate
from causal_inference.rct.estimators_ipw import ipw_ate


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
            # Not all cases have treatment/outcomes (IV uses y, d, z)
            assert len(case_data["data"]) > 0, f"Empty data for {case_name}"


# =============================================================================
# PSM Golden Reference Tests (Session 180)
# =============================================================================


class TestGoldenReferencePSM:
    """Test PSM estimators against golden reference."""

    def test_psm_observable_confounding(self, golden_results):
        """Test psm_ate matches golden reference for observable confounding case."""
        from src.causal_inference.psm import psm_ate

        if "psm_observable_confounding" not in golden_results:
            pytest.skip("PSM golden reference not available")

        data = golden_results["psm_observable_confounding"]["data"]
        expected = golden_results["psm_observable_confounding"]["psm_ate"]

        treatment = np.array(data["treatment"])
        outcomes = np.array(data["outcomes"])
        covariates = np.array(data["covariates"])

        result = psm_ate(outcomes, treatment, covariates)

        # PSM has some stochastic component, use rtol=1e-6 for near-exact match
        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-6), (
            f"PSM estimate mismatch: got {result['estimate']}, expected {expected['estimate']}"
        )


# =============================================================================
# IV Golden Reference Tests (Session 180)
# =============================================================================


class TestGoldenReferenceIV:
    """Test IV estimators against golden reference."""

    def test_iv_tsls(self, golden_results):
        """Test 2SLS matches golden reference."""
        from src.causal_inference.iv import TwoStageLeastSquares

        if "iv_single_instrument" not in golden_results:
            pytest.skip("IV golden reference not available")

        data = golden_results["iv_single_instrument"]["data"]
        expected = golden_results["iv_single_instrument"]["tsls"]

        y = np.array(data["y"])
        d = np.array(data["d"]).reshape(-1, 1)
        z = np.array(data["z"]).reshape(-1, 1)

        tsls = TwoStageLeastSquares(inference="robust")
        tsls.fit(y, d, z)

        assert np.isclose(tsls.coef_[0], expected["coefficient"], rtol=1e-10), (
            f"2SLS coef mismatch: got {tsls.coef_[0]}, expected {expected['coefficient']}"
        )

    def test_iv_liml(self, golden_results):
        """Test LIML matches golden reference."""
        from src.causal_inference.iv import LIML

        if "iv_single_instrument" not in golden_results:
            pytest.skip("IV golden reference not available")

        data = golden_results["iv_single_instrument"]["data"]
        expected = golden_results["iv_single_instrument"]["liml"]

        y = np.array(data["y"])
        d = np.array(data["d"]).reshape(-1, 1)
        z = np.array(data["z"]).reshape(-1, 1)

        liml = LIML(inference="robust")
        liml.fit(y, d, z)

        assert np.isclose(liml.coef_[0], expected["coefficient"], rtol=1e-10), (
            f"LIML coef mismatch: got {liml.coef_[0]}, expected {expected['coefficient']}"
        )


# =============================================================================
# DiD Golden Reference Tests (Session 180)
# =============================================================================


class TestGoldenReferenceDiD:
    """Test DiD estimators against golden reference."""

    def test_did_classic_2x2(self, golden_results):
        """Test did_2x2 matches golden reference."""
        from src.causal_inference.did import did_2x2

        if "did_classic_2x2" not in golden_results:
            pytest.skip("DiD golden reference not available")

        data = golden_results["did_classic_2x2"]["data"]
        expected = golden_results["did_classic_2x2"]["did_2x2"]

        outcome = np.array(data["outcome"])
        treatment = np.array(data["treatment"])
        post = np.array(data["post"])
        unit_id = np.array(data["unit_id"])

        result = did_2x2(
            outcomes=outcome,
            treatment=treatment,
            post=post,
            unit_id=unit_id,
        )

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-10), (
            f"DiD estimate mismatch: got {result['estimate']}, expected {expected['estimate']}"
        )


# =============================================================================
# RDD Golden Reference Tests (Session 180)
# =============================================================================


class TestGoldenReferenceRDD:
    """Test RDD estimators against golden reference."""

    def test_rdd_sharp(self, golden_results):
        """Test SharpRDD matches golden reference."""
        from src.causal_inference.rdd import SharpRDD

        if "rdd_sharp" not in golden_results:
            pytest.skip("RDD sharp golden reference not available")

        data = golden_results["rdd_sharp"]["data"]
        expected = golden_results["rdd_sharp"]["sharp_rdd"]

        outcome = np.array(data["outcome"])
        running = np.array(data["running"])

        rdd = SharpRDD(cutoff=0.0, bandwidth=expected["bandwidth"], inference="robust")
        rdd.fit(outcome, running)

        assert np.isclose(rdd.coef_, expected["estimate"], rtol=1e-10), (
            f"Sharp RDD estimate mismatch: got {rdd.coef_}, expected {expected['estimate']}"
        )

    def test_rdd_fuzzy(self, golden_results):
        """Test FuzzyRDD matches golden reference."""
        from src.causal_inference.rdd import FuzzyRDD

        if "rdd_fuzzy" not in golden_results:
            pytest.skip("RDD fuzzy golden reference not available")

        data = golden_results["rdd_fuzzy"]["data"]
        expected = golden_results["rdd_fuzzy"]["fuzzy_rdd"]

        outcome = np.array(data["outcome"])
        running = np.array(data["running"])
        treatment = np.array(data["treatment"])

        fuzzy = FuzzyRDD(cutoff=0.0, bandwidth=expected["bandwidth"], inference="robust")
        fuzzy.fit(outcome, running, treatment)

        assert np.isclose(fuzzy.coef_, expected["estimate"], rtol=1e-10), (
            f"Fuzzy RDD estimate mismatch: got {fuzzy.coef_}, expected {expected['estimate']}"
        )


# =============================================================================
# Session 185: Extended Golden Reference Tests
# =============================================================================


class TestGoldenReferenceObservational:
    """Test Observational estimators against golden reference."""

    def test_ipw_ate_observational(self, golden_results):
        """Test IPW ATE matches golden reference."""
        from src.causal_inference.observational import ipw_ate_observational

        if "observational_confounding" not in golden_results:
            pytest.skip("Observational golden reference not available")

        data = golden_results["observational_confounding"]["data"]
        expected = golden_results["observational_confounding"]["ipw_ate"]

        outcome = np.array(data["outcome"])
        treatment = np.array(data["treatment"])
        covariates = np.array(data["covariates"])

        result = ipw_ate_observational(outcome, treatment, covariates)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-6), (
            f"IPW estimate mismatch: got {result['estimate']}, expected {expected['estimate']}"
        )

    def test_dr_ate(self, golden_results):
        """Test DR ATE matches golden reference."""
        from src.causal_inference.observational import dr_ate

        if "observational_confounding" not in golden_results:
            pytest.skip("Observational golden reference not available")

        data = golden_results["observational_confounding"]["data"]
        expected = golden_results["observational_confounding"]["dr_ate"]

        outcome = np.array(data["outcome"])
        treatment = np.array(data["treatment"])
        covariates = np.array(data["covariates"])

        result = dr_ate(outcome, treatment, covariates)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-6), (
            f"DR estimate mismatch: got {result['estimate']}, expected {expected['estimate']}"
        )


class TestGoldenReferenceSensitivity:
    """Test Sensitivity analysis against golden reference."""

    def test_e_value(self, golden_results):
        """Test E-value matches golden reference."""
        from src.causal_inference.sensitivity import e_value

        if "sensitivity_evalue" not in golden_results:
            pytest.skip("Sensitivity golden reference not available")

        data = golden_results["sensitivity_evalue"]["data"]
        expected = golden_results["sensitivity_evalue"]["e_value"]

        result = e_value(
            estimate=data["estimate"],
            ci_lower=data["ci_lower"],
            effect_type=data["effect_type"],
        )

        assert np.isclose(result["e_value"], expected["e_value"], rtol=1e-10), (
            f"E-value mismatch: got {result['e_value']}, expected {expected['e_value']}"
        )


class TestGoldenReferenceCATE:
    """Test CATE meta-learners against golden reference."""

    def test_s_learner(self, golden_results):
        """Test S-learner matches golden reference."""
        from src.causal_inference.cate.meta_learners import s_learner

        if "cate_heterogeneous" not in golden_results:
            pytest.skip("CATE golden reference not available")

        data = golden_results["cate_heterogeneous"]["data"]
        expected = golden_results["cate_heterogeneous"]["s_learner"]

        outcome = np.array(data["outcome"])
        treatment = np.array(data["treatment"])
        covariates = np.array(data["covariates"])

        result = s_learner(outcome, treatment, covariates)

        # CATE uses ML models, use rtol=1e-6 for near-match
        assert np.isclose(result["ate"], expected["ate"], rtol=1e-6), (
            f"S-learner ATE mismatch: got {result['ate']}, expected {expected['ate']}"
        )

    def test_t_learner(self, golden_results):
        """Test T-learner matches golden reference."""
        from src.causal_inference.cate.meta_learners import t_learner

        if "cate_heterogeneous" not in golden_results:
            pytest.skip("CATE golden reference not available")

        data = golden_results["cate_heterogeneous"]["data"]
        expected = golden_results["cate_heterogeneous"]["t_learner"]

        outcome = np.array(data["outcome"])
        treatment = np.array(data["treatment"])
        covariates = np.array(data["covariates"])

        result = t_learner(outcome, treatment, covariates)

        assert np.isclose(result["ate"], expected["ate"], rtol=1e-6), (
            f"T-learner ATE mismatch: got {result['ate']}, expected {expected['ate']}"
        )


class TestGoldenReferenceRKD:
    """Test RKD estimators against golden reference."""

    def test_sharp_rkd(self, golden_results):
        """Test SharpRKD matches golden reference."""
        from src.causal_inference.rkd import SharpRKD

        if "rkd_sharp" not in golden_results:
            pytest.skip("RKD golden reference not available")

        data = golden_results["rkd_sharp"]["data"]
        expected = golden_results["rkd_sharp"]["sharp_rkd"]

        outcome = np.array(data["outcome"])
        running = np.array(data["running"])
        treatment_intensity = np.array(data["treatment_intensity"])

        rkd = SharpRKD(cutoff=0.0, bandwidth=1.0)
        result = rkd.fit(outcome, running, treatment_intensity)

        # Result is a dataclass, use attribute access
        assert np.isclose(result.estimate, expected["estimate"], rtol=1e-10), (
            f"RKD estimate mismatch: got {result.estimate}, expected {expected['estimate']}"
        )


class TestGoldenReferenceSCM:
    """Test SCM estimators against golden reference."""

    def test_synthetic_control(self, golden_results):
        """Test synthetic_control matches golden reference."""
        from src.causal_inference.scm import synthetic_control

        if "scm_basic" not in golden_results:
            pytest.skip("SCM golden reference not available")

        data = golden_results["scm_basic"]["data"]
        expected = golden_results["scm_basic"]["synthetic_control"]

        outcomes = np.array(data["outcomes"])
        treatment = np.array(data["treatment"])
        treatment_period = data["treatment_period"]

        result = synthetic_control(outcomes, treatment, treatment_period)

        # synthetic_control returns a dict (not dataclass)
        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-6), (
            f"SCM estimate mismatch: got {result['estimate']}, expected {expected['estimate']}"
        )


class TestGoldenReferenceTimeSeries:
    """Test Time Series estimators against golden reference."""

    def test_var_estimate(self, golden_results):
        """Test VAR estimation matches golden reference."""
        from src.causal_inference.timeseries import var_estimate

        if "timeseries_var" not in golden_results:
            pytest.skip("Time series golden reference not available")

        data = golden_results["timeseries_var"]["data"]
        expected = golden_results["timeseries_var"]["var_estimate"]

        Y = np.array(data["Y"])

        result = var_estimate(Y, lags=1)

        # Result is a dataclass, use attribute access
        assert np.allclose(result.coefficients, expected["coefficients"], rtol=1e-10), (
            f"VAR coefficients mismatch"
        )

    def test_granger_causality(self, golden_results):
        """Test Granger causality matches golden reference."""
        from src.causal_inference.timeseries import granger_causality

        if "timeseries_var" not in golden_results:
            pytest.skip("Time series golden reference not available")

        data = golden_results["timeseries_var"]["data"]
        expected = golden_results["timeseries_var"]["granger_causality"]

        Y = np.array(data["Y"])

        result = granger_causality(Y, lags=1, cause_idx=1, effect_idx=0)

        # Result is a dataclass, use attribute access
        assert np.isclose(result.f_statistic, expected["f_statistic"], rtol=1e-10), (
            f"Granger F-stat mismatch: got {result.f_statistic}, expected {expected['f_statistic']}"
        )


class TestGoldenReferenceVECM:
    """Test VECM estimators against golden reference."""

    def test_vecm_estimate(self, golden_results):
        """Test VECM estimation matches golden reference."""
        from src.causal_inference.timeseries.vecm import vecm_estimate

        if "vecm_cointegrated" not in golden_results:
            pytest.skip("VECM golden reference not available")

        data = golden_results["vecm_cointegrated"]["data"]
        expected = golden_results["vecm_cointegrated"]["vecm_estimate"]

        Y = np.array(data["Y"])

        result = vecm_estimate(Y, coint_rank=1, lags=1)

        # Result is a dataclass, use attribute access
        assert np.allclose(result.alpha, expected["alpha"], rtol=1e-6), (
            f"VECM alpha mismatch"
        )


class TestGoldenReferenceSelection:
    """Test Selection models against golden reference."""

    def test_heckman_two_step(self, golden_results):
        """Test Heckman two-step matches golden reference."""
        from src.causal_inference.selection import heckman_two_step

        if "selection_heckman" not in golden_results:
            pytest.skip("Selection golden reference not available")

        data = golden_results["selection_heckman"]["data"]
        expected = golden_results["selection_heckman"]["heckman_two_step"]

        outcome = np.array(data["outcome"])
        X = np.array(data["X"])
        Z = np.array(data["Z"])
        selected = np.array(data["selected"])

        selection_covariates = np.column_stack([X, Z])

        result = heckman_two_step(
            outcome=outcome,
            selected=selected,
            selection_covariates=selection_covariates,
            outcome_covariates=X,
        )

        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-4), (
            f"Heckman estimate mismatch: got {result['estimate']}, expected {expected['estimate']}"
        )


class TestGoldenReferenceBounds:
    """Test Bounds estimators against golden reference."""

    def test_manski_bounds(self, golden_results):
        """Test Manski bounds match golden reference."""
        from src.causal_inference.bounds import manski_worst_case

        if "bounds_partial_id" not in golden_results:
            pytest.skip("Bounds golden reference not available")

        data = golden_results["bounds_partial_id"]["data"]
        expected = golden_results["bounds_partial_id"]["manski_bounds"]

        # Use complete outcome array (not outcome_observed with NaN)
        outcome = np.array(data["outcome"])
        treatment = np.array(data["treatment"])

        result = manski_worst_case(outcome, treatment, outcome_support=(-5.0, 5.0))

        # ManskiBoundsResult is a TypedDict with 'bounds_lower'/'bounds_upper' keys
        assert np.isclose(result["bounds_lower"], expected["bounds_lower"], rtol=1e-6), (
            f"Manski lower bound mismatch"
        )
        assert np.isclose(result["bounds_upper"], expected["bounds_upper"], rtol=1e-6), (
            f"Manski upper bound mismatch"
        )


class TestGoldenReferenceMediation:
    """Test Mediation analysis against golden reference."""

    def test_mediation_analysis(self, golden_results):
        """Test mediation analysis matches golden reference."""
        from src.causal_inference.mediation import mediation_analysis

        if "mediation_basic" not in golden_results:
            pytest.skip("Mediation golden reference not available")

        data = golden_results["mediation_basic"]["data"]
        expected = golden_results["mediation_basic"]["mediation_analysis"]

        outcome = np.array(data["outcome"])
        treatment = np.array(data["treatment"])
        mediator = np.array(data["mediator"])

        result = mediation_analysis(outcome, treatment, mediator, random_state=42)

        # Mediation has stochastic component
        assert np.isclose(result["total_effect"], expected["total_effect"], rtol=0.1), (
            f"Mediation total effect mismatch"
        )


class TestGoldenReferenceMTE:
    """Test MTE/LATE estimators against golden reference."""

    def test_late_estimator(self, golden_results):
        """Test LATE estimation matches golden reference."""
        from src.causal_inference.mte import late_estimator

        if "mte_late" not in golden_results:
            pytest.skip("MTE golden reference not available")

        data = golden_results["mte_late"]["data"]
        expected = golden_results["mte_late"]["late_estimate"]

        outcome = np.array(data["outcome"])
        treatment = np.array(data["treatment"])
        instrument = np.array(data["instrument"])

        result = late_estimator(outcome, treatment, instrument)

        assert np.isclose(result["late"], expected["late"], rtol=1e-6), (
            f"LATE mismatch: got {result['late']}, expected {expected['late']}"
        )


class TestGoldenReferenceQTE:
    """Test QTE estimators against golden reference."""

    def test_unconditional_qte_median(self, golden_results):
        """Test unconditional QTE at median matches golden reference."""
        from src.causal_inference.qte import unconditional_qte

        if "qte_unconditional" not in golden_results:
            pytest.skip("QTE golden reference not available")

        data = golden_results["qte_unconditional"]["data"]
        expected = golden_results["qte_unconditional"]["qte_50"]

        outcome = np.array(data["outcome"])
        treatment = np.array(data["treatment"])

        result = unconditional_qte(outcome, treatment, quantile=0.50, random_state=42)

        # QTEResult is a TypedDict with 'tau_q' key (quantile treatment effect)
        assert np.isclose(result["tau_q"], expected["tau_q"], rtol=0.1), (
            f"QTE50 mismatch: got {result['tau_q']}, expected {expected['tau_q']}"
        )


class TestGoldenReferenceControlFunction:
    """Test Control Function estimators against golden reference."""

    def test_control_function_ate(self, golden_results):
        """Test control function ATE matches golden reference."""
        from src.causal_inference.control_function import control_function_ate

        if "control_function_2step" not in golden_results:
            pytest.skip("Control function golden reference not available")

        data = golden_results["control_function_2step"]["data"]
        expected = golden_results["control_function_2step"]["control_function"]

        outcome = np.array(data["outcome"])
        treatment = np.array(data["treatment"])
        covariates = np.array(data["covariates"])
        instrument = np.array(data["instrument"])

        result = control_function_ate(outcome, treatment, instrument, covariates, random_state=42)

        assert np.isclose(result["estimate"], expected["estimate"], rtol=0.1), (
            f"CF estimate mismatch: got {result['estimate']}, expected {expected['estimate']}"
        )


class TestGoldenReferenceBayesian:
    """Test Bayesian estimators against golden reference."""

    def test_bayesian_ate(self, golden_results):
        """Test Bayesian ATE matches golden reference."""
        from src.causal_inference.bayesian import bayesian_ate

        if "bayesian_ate" not in golden_results:
            pytest.skip("Bayesian golden reference not available")

        data = golden_results["bayesian_ate"]["data"]
        expected = golden_results["bayesian_ate"]["bayesian_ate"]

        outcome = np.array(data["outcome"])
        treatment = np.array(data["treatment"])

        result = bayesian_ate(outcome, treatment, n_posterior_samples=1000)

        # Bayesian has MCMC variance, use rtol=0.2
        assert np.isclose(result["posterior_mean"], expected["posterior_mean"], rtol=0.2), (
            f"Bayesian ATE mismatch: got {result['posterior_mean']}, expected {expected['posterior_mean']}"
        )


class TestGoldenReferenceDiscovery:
    """Test Discovery (causal structure learning) against golden reference."""

    def test_pc_algorithm(self, golden_results):
        """Test PC algorithm matches golden reference."""
        from src.causal_inference.discovery import pc_algorithm

        if "discovery_pc" not in golden_results:
            pytest.skip("Discovery golden reference not available")

        data = golden_results["discovery_pc"]["data"]
        expected = golden_results["discovery_pc"]["pc_algorithm"]

        X = np.array(data["X"])
        Y = np.array(data["Y"])
        Z = np.array(data["Z"])
        data_matrix = np.column_stack([X, Y, Z])

        result = pc_algorithm(data_matrix, alpha=0.05)

        # PCResult is a dataclass. Count edges from skeleton adjacency matrix.
        # Skeleton adjacency is symmetric for undirected graph, so edges = sum/2
        skeleton_adj = np.array(result.skeleton.adjacency)
        n_edges = np.sum(skeleton_adj) // 2
        expected_adj = np.array(expected["skeleton"]["adjacency"])
        expected_n_edges = np.sum(expected_adj) // 2

        assert n_edges == expected_n_edges, (
            f"PC edge count mismatch: got {n_edges}, expected {expected_n_edges}"
        )


class TestGoldenReferenceBunching:
    """Test Bunching estimators against golden reference."""

    def test_bunching_excess_mass(self, golden_results):
        """Test bunching estimator matches golden reference."""
        from src.causal_inference.bunching import bunching_estimator

        if "bunching_kink" not in golden_results:
            pytest.skip("Bunching golden reference not available")

        data = golden_results["bunching_kink"]["data"]
        expected = golden_results["bunching_kink"]["bunching"]

        income = np.array(data["income"])
        kink_point = data["kink_point"]
        bunching_width = data["bunching_width"]

        result = bunching_estimator(
            data=income,
            kink_point=kink_point,
            bunching_width=bunching_width,
            n_bins=60,
            t1_rate=0.25,
            t2_rate=0.35,
            n_bootstrap=50,
            random_state=42,
        )

        # BunchingResult is a dataclass, use attribute access
        assert np.isclose(result.excess_mass, expected["excess_mass"], rtol=0.2), (
            f"Bunching excess mass mismatch: got {result.excess_mass}, expected {expected['excess_mass']}"
        )


class TestGoldenReferenceShiftShare:
    """Test Shift-Share IV against golden reference."""

    def test_shift_share_estimator(self, golden_results):
        """Test shift-share IV matches golden reference."""
        from src.causal_inference.shift_share import shift_share_iv

        if "shift_share_bartik" not in golden_results:
            pytest.skip("Shift-Share golden reference not available")

        data = golden_results["shift_share_bartik"]["data"]
        expected = golden_results["shift_share_bartik"]["shift_share_iv"]

        Y = np.array(data["Y"])
        D = np.array(data["D"])  # Treatment is stored as 'D'
        shares = np.array(data["shares"])
        shocks = np.array(data["shocks"])

        result = shift_share_iv(Y, D, shares, shocks)

        # Shift-share may return dict or dataclass
        estimate = result["estimate"] if isinstance(result, dict) else result.estimate

        assert np.isclose(estimate, expected["estimate"], rtol=0.15), (
            f"Shift-share estimate mismatch: got {estimate}, expected {expected['estimate']}"
        )


class TestGoldenReferenceDTR:
    """Test Dynamic Treatment Regimes against golden reference."""

    def test_q_learning(self, golden_results):
        """Test Q-learning matches golden reference."""
        from src.causal_inference.dtr import q_learning

        if "dtr_two_stage" not in golden_results:
            pytest.skip("DTR golden reference not available")

        data = golden_results["dtr_two_stage"]["data"]
        expected = golden_results["dtr_two_stage"]["q_learning"]

        # Data uses stage-specific keys: X1, X2, A1, A2, Y1, Y2
        X1 = np.array(data["X1"])
        X2 = np.array(data["X2"])
        A1 = np.array(data["A1"])
        A2 = np.array(data["A2"])
        Y1 = np.array(data["Y1"])
        Y2 = np.array(data["Y2"])

        result = q_learning(
            covariates=[X1, X2],
            treatments=[A1, A2],
            outcomes=[Y1, Y2],
        )

        # Q-learning returns dict with 'value_estimate'
        assert np.isclose(result["value_estimate"], expected["value_estimate"], rtol=0.15), (
            f"Q-learning value estimate mismatch"
        )


class TestGoldenReferenceDynamicDML:
    """Test Dynamic DML against golden reference."""

    def test_dynamic_dml(self, golden_results):
        """Test Dynamic DML matches golden reference."""
        from src.causal_inference.dynamic.dynamic_dml import dynamic_dml

        if "dynamic_dml_panel" not in golden_results:
            pytest.skip("Dynamic DML golden reference not available")

        data = golden_results["dynamic_dml_panel"]["data"]
        expected = golden_results["dynamic_dml_panel"]["dynamic_dml"]

        outcome = np.array(data["outcome"])
        treatment = np.array(data["treatment"])
        states = np.array(data["states"])  # covariates stored as 'states'

        result = dynamic_dml(
            outcome=outcome,
            treatment=treatment,
            states=states,
            max_lag=expected["max_lag"],
        )

        # Dynamic DML returns theta (treatment effect)
        assert np.isclose(result["theta"], expected["theta"], rtol=0.2), (
            f"Dynamic DML theta mismatch: got {result['theta']}, expected {expected['theta']}"
        )
