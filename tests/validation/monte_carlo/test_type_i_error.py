"""
Type I Error Verification Tests

Validates that estimators correctly control Type I error rate at nominal level.
Under the null hypothesis (true effect = 0), the rejection rate should be ~5%.

Phase 1: 5 core estimators (one per method family)
- SimpleATE (RCT)
- IPW (Observational)
- DiD 2x2 (DiD)
- 2SLS (IV)
- SharpRDD (RDD)

Target: Rejection rate between 3% and 7% (5% +/- 2%)

Session 158 Update:
- Modernized imports to match current module structure
- Updated API usage for class-based estimators (TwoStageLeastSquares, SharpRDD)
- Aligned DGP function names with actual codebase
"""

import numpy as np
import pytest
from typing import Tuple, Dict, Any

# Import estimators - using correct module paths
from src.causal_inference.rct import simple_ate
from src.causal_inference.observational import ipw_ate_observational
from src.causal_inference.did import did_2x2
from src.causal_inference.iv import TwoStageLeastSquares
from src.causal_inference.rdd import SharpRDD

# Import DGP generators - using correct function names
from tests.validation.monte_carlo.dgp_generators import dgp_simple_rct
from tests.validation.monte_carlo.dgp_did import dgp_did_2x2_simple
from tests.validation.monte_carlo.dgp_iv import dgp_iv_strong
from tests.validation.monte_carlo.dgp_rdd import dgp_rdd_zero_effect


# Configuration
N_SIMULATIONS = 2000  # Sufficient for Type I error estimation
ALPHA = 0.05  # Nominal significance level
TYPE_I_LOWER = 0.03  # 5% - 2%
TYPE_I_UPPER = 0.07  # 5% + 2%


def _count_rejections_dict(results: list, true_effect: float = 0.0) -> Tuple[int, float]:
    """
    Count rejections where CI excludes true effect (dict-based results).

    Returns:
        Tuple of (rejection_count, rejection_rate)
    """
    rejections = 0
    for result in results:
        # Handle dict results (simple_ate returns dict)
        if isinstance(result, dict):
            ci_lower = result.get("ci_lower", result.get("conf_int_lower"))
            ci_upper = result.get("ci_upper", result.get("conf_int_upper"))
            if ci_lower is not None and ci_upper is not None:
                if ci_lower > true_effect or ci_upper < true_effect:
                    rejections += 1
        # Handle object results with attributes
        elif hasattr(result, "ci_lower") and hasattr(result, "ci_upper"):
            if result.ci_lower > true_effect or result.ci_upper < true_effect:
                rejections += 1
        elif hasattr(result, "conf_int_lower") and hasattr(result, "conf_int_upper"):
            if result.conf_int_lower > true_effect or result.conf_int_upper < true_effect:
                rejections += 1
        elif hasattr(result, "ci_"):
            # SharpRDD returns tuple (lower, upper)
            ci_lower, ci_upper = result.ci_
            if ci_lower > true_effect or ci_upper < true_effect:
                rejections += 1

    rejection_rate = rejections / len(results) if results else 0.0
    return rejections, rejection_rate


# =============================================================================
# RCT: SimpleATE
# =============================================================================


@pytest.mark.monte_carlo
@pytest.mark.type_i_error
def test_type_i_error_simple_ate():
    """
    Type I error test for SimpleATE.

    Under null (true_ate=0), rejection rate should be ~5%.
    """
    np.random.seed(42)
    results = []

    for i in range(N_SIMULATIONS):
        # Generate RCT data with NO effect
        y, t = dgp_simple_rct(n=200, true_ate=0.0, random_state=42 + i)

        try:
            result = simple_ate(y, t)
            results.append(result)
        except Exception:
            # Skip failed iterations (shouldn't happen for RCT)
            continue

    rejections, rejection_rate = _count_rejections_dict(results, true_effect=0.0)

    assert TYPE_I_LOWER < rejection_rate < TYPE_I_UPPER, (
        f"Type I error rate {rejection_rate:.3f} ({rejections}/{len(results)}) "
        f"outside acceptable range [{TYPE_I_LOWER}, {TYPE_I_UPPER}]"
    )


# =============================================================================
# Observational: IPW
# =============================================================================


def _generate_ipw_null_dgp(
    n: int = 500, random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate observational data with no treatment effect."""
    rng = np.random.RandomState(random_state)

    # Covariates
    x = rng.randn(n, 2)

    # Propensity score (treatment depends on covariates)
    ps_true = 1 / (1 + np.exp(-0.5 * x[:, 0] - 0.3 * x[:, 1]))
    t = (rng.rand(n) < ps_true).astype(float)

    # Outcome with NO treatment effect (true_ate = 0)
    y = 1.0 + 0.5 * x[:, 0] + 0.3 * x[:, 1] + rng.randn(n)
    # Note: No t term, so true ATE = 0

    return y, t, x


@pytest.mark.monte_carlo
@pytest.mark.type_i_error
def test_type_i_error_ipw():
    """
    Type I error test for IPW.

    Under null (true_ate=0), rejection rate should be ~5%.
    """
    np.random.seed(42)
    results = []

    for i in range(N_SIMULATIONS):
        y, t, x = _generate_ipw_null_dgp(n=500, random_state=42 + i)

        try:
            result = ipw_ate_observational(y, t, x)
            results.append(result)
        except Exception:
            # Skip iterations with extreme propensity scores
            continue

    # Need sufficient successful iterations
    assert len(results) >= N_SIMULATIONS * 0.9, (
        f"Too many failed iterations: {N_SIMULATIONS - len(results)}"
    )

    rejections, rejection_rate = _count_rejections_dict(results, true_effect=0.0)

    assert TYPE_I_LOWER < rejection_rate < TYPE_I_UPPER, (
        f"Type I error rate {rejection_rate:.3f} ({rejections}/{len(results)}) "
        f"outside acceptable range [{TYPE_I_LOWER}, {TYPE_I_UPPER}]"
    )


# =============================================================================
# DiD: did_2x2
# =============================================================================


@pytest.mark.monte_carlo
@pytest.mark.type_i_error
def test_type_i_error_did_2x2():
    """
    Type I error test for DiD 2x2.

    Under null (true_effect=0), rejection rate should be ~5%.
    """
    np.random.seed(42)
    results = []

    for i in range(N_SIMULATIONS):
        # Generate DiD data with NO effect
        data = dgp_did_2x2_simple(
            n_treated=50,
            n_control=50,
            n_pre=1,
            n_post=1,
            true_att=0.0,  # NULL hypothesis
            sigma=1.0,
            random_state=42 + i,
        )

        try:
            result = did_2x2(
                outcomes=data.outcomes,
                treatment=data.treatment,
                post=data.post,
                unit_id=data.unit_id,
            )
            results.append(result)
        except Exception:
            continue

    assert len(results) >= N_SIMULATIONS * 0.9, (
        f"Too many failed iterations: {N_SIMULATIONS - len(results)}"
    )

    rejections, rejection_rate = _count_rejections_dict(results, true_effect=0.0)

    assert TYPE_I_LOWER < rejection_rate < TYPE_I_UPPER, (
        f"Type I error rate {rejection_rate:.3f} ({rejections}/{len(results)}) "
        f"outside acceptable range [{TYPE_I_LOWER}, {TYPE_I_UPPER}]"
    )


# =============================================================================
# IV: TwoStageLeastSquares
# =============================================================================


@pytest.mark.monte_carlo
@pytest.mark.type_i_error
def test_type_i_error_2sls():
    """
    Type I error test for 2SLS.

    Under null (true_effect=0), rejection rate should be ~5%.
    Uses strong instrument to ensure valid inference.
    """
    np.random.seed(42)
    results = []

    for i in range(N_SIMULATIONS):
        # Generate IV data with NO effect
        data = dgp_iv_strong(
            n=500,
            true_beta=0.0,  # NULL hypothesis
            endogeneity_rho=0.5,  # Still has endogeneity
            random_state=42 + i,
        )

        try:
            tsls = TwoStageLeastSquares(inference="robust")
            tsls.fit(data.Y, data.D, data.Z)
            results.append(tsls)
        except Exception:
            continue

    assert len(results) >= N_SIMULATIONS * 0.9, (
        f"Too many failed iterations: {N_SIMULATIONS - len(results)}"
    )

    # Count rejections using p-value or CI
    rejections = 0
    for result in results:
        # TwoStageLeastSquares has coef_, se_, ci_ attributes
        if hasattr(result, "ci_") and result.ci_ is not None:
            ci_lower, ci_upper = result.ci_[0]  # First coefficient CI
            if ci_lower > 0.0 or ci_upper < 0.0:
                rejections += 1
        elif hasattr(result, "p_value_") and result.p_value_ is not None:
            if result.p_value_[0] < ALPHA:
                rejections += 1

    rejection_rate = rejections / len(results)

    assert TYPE_I_LOWER < rejection_rate < TYPE_I_UPPER, (
        f"Type I error rate {rejection_rate:.3f} ({rejections}/{len(results)}) "
        f"outside acceptable range [{TYPE_I_LOWER}, {TYPE_I_UPPER}]"
    )


# =============================================================================
# RDD: SharpRDD
# =============================================================================


@pytest.mark.monte_carlo
@pytest.mark.type_i_error
def test_type_i_error_sharp_rdd():
    """
    Type I error test for Sharp RDD.

    Under null (true_effect=0), rejection rate should be ~5%.
    """
    np.random.seed(42)
    results = []

    for i in range(N_SIMULATIONS):
        # Generate RDD data with NO effect
        data = dgp_rdd_zero_effect(
            n=1000,  # Need more obs for RDD
            cutoff=0.0,
            slope=1.0,
            error_sd=1.0,
            random_state=42 + i,
        )

        try:
            rdd = SharpRDD(cutoff=data.cutoff, bandwidth="ik", inference="robust")
            rdd.fit(data.Y, data.X)
            results.append(rdd)
        except Exception:
            continue

    assert len(results) >= N_SIMULATIONS * 0.8, (
        f"Too many failed iterations: {N_SIMULATIONS - len(results)}"
    )

    # Count rejections
    rejections = 0
    for result in results:
        if hasattr(result, "ci_") and result.ci_ is not None:
            ci_lower, ci_upper = result.ci_
            if ci_lower > 0.0 or ci_upper < 0.0:
                rejections += 1
        elif hasattr(result, "p_value_") and result.p_value_ is not None:
            if result.p_value_ < ALPHA:
                rejections += 1

    rejection_rate = rejections / len(results)

    # RDD can have slightly higher Type I error due to bandwidth selection
    # Use slightly wider bounds
    rdd_type_i_lower = 0.025
    rdd_type_i_upper = 0.085

    assert rdd_type_i_lower < rejection_rate < rdd_type_i_upper, (
        f"Type I error rate {rejection_rate:.3f} ({rejections}/{len(results)}) "
        f"outside acceptable range [{rdd_type_i_lower}, {rdd_type_i_upper}]"
    )


# =============================================================================
# Summary Test
# =============================================================================


@pytest.mark.monte_carlo
@pytest.mark.type_i_error
def test_type_i_error_summary():
    """
    Quick summary test that documents what Type I error tests cover.

    Use this for rapid verification during development.
    """
    estimators_tested = [
        "SimpleATE (RCT) - simple_ate()",
        "IPW (Observational) - ipw_ate_observational()",
        "DiD 2x2 (DiD) - did_2x2()",
        "2SLS (IV) - TwoStageLeastSquares",
        "SharpRDD (RDD) - SharpRDD",
    ]

    print("\n=== Type I Error Verification ===")
    print(f"Estimators: {len(estimators_tested)}")
    print(f"Simulations per test: {N_SIMULATIONS}")
    print(f"Nominal alpha: {ALPHA}")
    print(f"Acceptable range: [{TYPE_I_LOWER}, {TYPE_I_UPPER}]")
    print("=" * 40)

    for est in estimators_tested:
        print(f"  - {est}")

    # This test always passes - it's just for documentation
    assert True
