"""
Tests for Sign Restrictions SVAR (Uhlig 2005).

Session 161: Test suite for set-identified SVAR using sign restrictions.

Test Structure (3-layer validation):
1. Known-Answer Tests: Basic functionality, output shapes, constraint satisfaction
2. Adversarial Tests: Invalid inputs, conflicting constraints, edge cases
3. Monte Carlo Tests: Set width, coverage, sign correctness across draws
"""

import numpy as np
import pytest
import warnings
from typing import Tuple, List

from causal_inference.timeseries.sign_restrictions import (
    SignRestrictionConstraint,
    SignRestrictionResult,
    sign_restriction_svar,
    create_monetary_policy_constraints,
    check_cholesky_in_set,
    validate_constraints,
    _givens_rotation_matrix,
    _random_orthogonal_givens,
    _random_orthogonal_qr,
    _check_sign_constraints,
    _compute_irf_from_impact,
)
from causal_inference.timeseries import var_estimate, cholesky_svar
from causal_inference.timeseries.svar import vma_coefficients
from causal_inference.timeseries.svar_types import IdentificationMethod


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_var_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate stable VAR(2) data with known structure.

    Returns
    -------
    data : np.ndarray
        Shape (200, 3) VAR data
    A1 : np.ndarray
        First lag coefficient matrix
    """
    np.random.seed(42)
    n = 200
    k = 3

    A1 = np.array([[0.4, 0.1, 0.05], [0.15, 0.35, 0.1], [0.05, 0.1, 0.3]])

    data = np.zeros((n, k))
    for t in range(1, n):
        data[t, :] = A1 @ data[t - 1, :] + np.random.randn(k) * 0.5

    return data, A1


@pytest.fixture
def var_result(sample_var_data):
    """Pre-estimated VAR result."""
    data, _ = sample_var_data
    return var_estimate(data, lags=2)


@pytest.fixture
def basic_constraints() -> List[SignRestrictionConstraint]:
    """Basic sign constraints for testing."""
    return [
        # Shock 0 has positive impact on variable 0
        SignRestrictionConstraint(shock_idx=0, response_idx=0, horizon=0, sign=1),
        # Shock 0 has positive impact on variable 1 at horizon 0
        SignRestrictionConstraint(shock_idx=0, response_idx=1, horizon=0, sign=1),
    ]


@pytest.fixture
def cholesky_consistent_constraints(var_result) -> List[SignRestrictionConstraint]:
    """
    Constraints consistent with Cholesky ordering.

    Cholesky gives lower triangular B0_inv, so diagonal elements are positive.
    """
    # For Cholesky, the diagonal of B0_inv is positive
    # Impact of shock i on variable i at horizon 0 is positive
    return [
        SignRestrictionConstraint(shock_idx=0, response_idx=0, horizon=0, sign=1),
        SignRestrictionConstraint(shock_idx=1, response_idx=1, horizon=0, sign=1),
        SignRestrictionConstraint(shock_idx=2, response_idx=2, horizon=0, sign=1),
    ]


# =============================================================================
# Layer 1: Known-Answer Tests
# =============================================================================


class TestSignRestrictionBasic:
    """Basic functionality tests for sign_restriction_svar."""

    def test_basic_estimation(self, var_result, basic_constraints):
        """Sign restriction estimation runs without error."""
        result = sign_restriction_svar(
            var_result,
            basic_constraints,
            horizons=10,
            n_draws=1000,
            seed=42,
        )

        assert isinstance(result, SignRestrictionResult)
        assert result.identification == IdentificationMethod.SIGN
        assert result.n_vars == 3
        assert result.n_accepted > 0

    def test_output_shapes(self, var_result, basic_constraints):
        """All output arrays have correct shapes."""
        horizons = 15
        result = sign_restriction_svar(
            var_result,
            basic_constraints,
            horizons=horizons,
            n_draws=500,
            seed=42,
        )

        # IRF shapes: (n_vars, n_vars, horizons+1)
        assert result.irf_median.shape == (3, 3, horizons + 1)
        assert result.irf_lower.shape == (3, 3, horizons + 1)
        assert result.irf_upper.shape == (3, 3, horizons + 1)

        # Impact matrices
        assert result.B0_inv.shape == (3, 3)
        assert result.B0.shape == (3, 3)

        # Structural shocks
        T = var_result.residuals.shape[0]
        assert result.structural_shocks.shape == (T, 3)

    def test_constraints_satisfied(self, var_result, basic_constraints):
        """All accepted draws satisfy the sign constraints."""
        result = sign_restriction_svar(
            var_result,
            basic_constraints,
            horizons=10,
            n_draws=500,
            seed=42,
        )

        # Compute VMA coefficients
        Phi = vma_coefficients(var_result, result.horizons)

        # Check every accepted B0_inv
        for B0_inv in result.B0_inv_set:
            irf = _compute_irf_from_impact(Phi, B0_inv, result.horizons)
            assert _check_sign_constraints(irf, basic_constraints), (
                "Accepted draw should satisfy all constraints"
            )

    def test_cholesky_within_set(self, var_result, cholesky_consistent_constraints):
        """When constraints match Cholesky ordering, Cholesky is in the set."""
        # First check that Cholesky satisfies constraints
        cholesky_satisfies = check_cholesky_in_set(
            var_result, cholesky_consistent_constraints, horizons=10
        )

        if cholesky_satisfies:
            # Run sign restrictions
            result = sign_restriction_svar(
                var_result,
                cholesky_consistent_constraints,
                horizons=10,
                n_draws=500,
                seed=42,
            )

            # Acceptance rate should be high since Cholesky is valid
            assert result.acceptance_rate > 0.05, (
                "Cholesky-consistent constraints should have reasonable acceptance"
            )

    def test_set_bounds_bracket_median(self, var_result, basic_constraints):
        """Lower bound <= median <= upper bound everywhere."""
        result = sign_restriction_svar(
            var_result,
            basic_constraints,
            horizons=10,
            n_draws=500,
            seed=42,
        )

        assert np.all(result.irf_lower <= result.irf_median + 1e-10), (
            "Lower bound should be <= median"
        )
        assert np.all(result.irf_median <= result.irf_upper + 1e-10), (
            "Median should be <= upper bound"
        )

    def test_acceptance_rate_positive(self, var_result, basic_constraints):
        """At least some rotations are accepted."""
        result = sign_restriction_svar(
            var_result,
            basic_constraints,
            horizons=10,
            n_draws=500,
            seed=42,
        )

        assert result.n_accepted > 0
        assert result.acceptance_rate > 0
        assert result.acceptance_rate <= 1.0


class TestSignRestrictionConstraint:
    """Tests for SignRestrictionConstraint dataclass."""

    def test_valid_constraint(self):
        """Valid constraint creation."""
        c = SignRestrictionConstraint(shock_idx=0, response_idx=1, horizon=5, sign=1)
        assert c.shock_idx == 0
        assert c.response_idx == 1
        assert c.horizon == 5
        assert c.sign == 1

    def test_negative_sign(self):
        """Negative sign constraint."""
        c = SignRestrictionConstraint(shock_idx=0, response_idx=1, horizon=0, sign=-1)
        assert c.sign == -1

    def test_invalid_sign(self):
        """Invalid sign raises error."""
        with pytest.raises(ValueError, match="sign must be 1 or -1"):
            SignRestrictionConstraint(shock_idx=0, response_idx=1, horizon=0, sign=0)

    def test_negative_horizon(self):
        """Negative horizon raises error."""
        with pytest.raises(ValueError, match="horizon must be >= 0"):
            SignRestrictionConstraint(shock_idx=0, response_idx=1, horizon=-1, sign=1)

    def test_repr(self):
        """String representation is informative."""
        c = SignRestrictionConstraint(shock_idx=0, response_idx=1, horizon=5, sign=1)
        repr_str = repr(c)
        assert "shock=0" in repr_str
        assert "response=1" in repr_str
        assert "h=5" in repr_str


class TestRotationMatrices:
    """Tests for rotation matrix generation."""

    def test_givens_rotation_orthogonal(self):
        """Givens rotation is orthogonal."""
        G = _givens_rotation_matrix(3, 0, 1, np.pi / 4)

        # G @ G.T = I
        np.testing.assert_array_almost_equal(G @ G.T, np.eye(3))

        # det(G) = 1 for proper rotation
        np.testing.assert_almost_equal(np.linalg.det(G), 1.0)

    def test_random_orthogonal_givens(self):
        """Random orthogonal via Givens is orthogonal with det=1."""
        rng = np.random.default_rng(42)
        Q = _random_orthogonal_givens(4, rng)

        # Q @ Q.T = I
        np.testing.assert_array_almost_equal(Q @ Q.T, np.eye(4))

        # det(Q) = 1
        np.testing.assert_almost_equal(np.linalg.det(Q), 1.0, decimal=10)

    def test_random_orthogonal_qr(self):
        """Random orthogonal via QR is orthogonal with det=1."""
        rng = np.random.default_rng(42)
        Q = _random_orthogonal_qr(4, rng)

        # Q @ Q.T = I
        np.testing.assert_array_almost_equal(Q @ Q.T, np.eye(4))

        # det(Q) = +1 (corrected)
        np.testing.assert_almost_equal(np.linalg.det(Q), 1.0, decimal=10)

    def test_rotation_methods_both_work(self, var_result, basic_constraints):
        """Both rotation methods produce valid results."""
        result_givens = sign_restriction_svar(
            var_result,
            basic_constraints,
            horizons=10,
            n_draws=200,
            rotation_method="givens",
            seed=42,
        )
        result_qr = sign_restriction_svar(
            var_result, basic_constraints, horizons=10, n_draws=200, rotation_method="qr", seed=42
        )

        assert result_givens.n_accepted > 0
        assert result_qr.n_accepted > 0


# =============================================================================
# Layer 2: Adversarial Tests
# =============================================================================


class TestSignRestrictionAdversarial:
    """Adversarial tests for edge cases and invalid inputs."""

    def test_conflicting_constraints(self, var_result):
        """Conflicting constraints lead to no acceptance or error."""
        # Shock 0 must have both positive AND negative impact on var 0 at h=0
        conflicting = [
            SignRestrictionConstraint(shock_idx=0, response_idx=0, horizon=0, sign=1),
            SignRestrictionConstraint(shock_idx=0, response_idx=0, horizon=0, sign=-1),
        ]

        with pytest.raises(ValueError, match="No rotations satisfied"):
            sign_restriction_svar(
                var_result,
                conflicting,
                horizons=10,
                n_draws=500,
                seed=42,
            )

    def test_too_many_constraints_warning(self, var_result):
        """Very restrictive constraints trigger warning."""
        # Many constraints across all variables and horizons
        many_constraints = []
        for shock_idx in range(3):
            for response_idx in range(3):
                for h in range(5):
                    # Alternate signs to make it hard to satisfy
                    sign = 1 if (shock_idx + response_idx + h) % 2 == 0 else -1
                    many_constraints.append(
                        SignRestrictionConstraint(
                            shock_idx=shock_idx,
                            response_idx=response_idx,
                            horizon=h,
                            sign=sign,
                        )
                    )

        # This should either raise or warn about low acceptance
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = sign_restriction_svar(
                    var_result,
                    many_constraints,
                    horizons=10,
                    n_draws=1000,
                    seed=42,
                    min_acceptance_rate=0.01,
                )
                # Check for warning
                if result.acceptance_rate < 0.01:
                    assert any(
                        "low acceptance rate" in str(warning.message).lower() for warning in w
                    ), "Should warn about low acceptance"
        except ValueError:
            # Also acceptable - no rotations satisfied
            pass

    def test_invalid_horizon(self, var_result):
        """Horizon exceeding max raises error."""
        constraints = [
            SignRestrictionConstraint(
                shock_idx=0,
                response_idx=0,
                horizon=100,
                sign=1,  # Beyond horizons
            ),
        ]

        with pytest.raises(ValueError, match="horizon"):
            sign_restriction_svar(
                var_result,
                constraints,
                horizons=10,  # max is 10, constraint is 100
                n_draws=100,
                seed=42,
            )

    def test_invalid_shock_idx(self, var_result):
        """Out of bounds shock index raises error."""
        constraints = [
            SignRestrictionConstraint(
                shock_idx=10,
                response_idx=0,
                horizon=0,
                sign=1,  # n_vars=3
            ),
        ]

        with pytest.raises(ValueError, match="shock_idx"):
            sign_restriction_svar(
                var_result,
                constraints,
                horizons=10,
                n_draws=100,
                seed=42,
            )

    def test_invalid_response_idx(self, var_result):
        """Out of bounds response index raises error."""
        constraints = [
            SignRestrictionConstraint(
                shock_idx=0,
                response_idx=10,
                horizon=0,
                sign=1,  # n_vars=3
            ),
        ]

        with pytest.raises(ValueError, match="response_idx"):
            sign_restriction_svar(
                var_result,
                constraints,
                horizons=10,
                n_draws=100,
                seed=42,
            )

    def test_empty_constraints(self, var_result):
        """No constraints → all rotations accepted (with warning)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = sign_restriction_svar(
                var_result,
                [],  # Empty constraints
                horizons=10,
                n_draws=100,
                seed=42,
            )

            # Check for warning about empty constraints
            assert any("No constraints" in str(warning.message) for warning in w)

        # All rotations should be accepted
        assert result.acceptance_rate == 1.0

    def test_invalid_rotation_method(self, var_result, basic_constraints):
        """Invalid rotation method raises error."""
        with pytest.raises(ValueError, match="rotation_method"):
            sign_restriction_svar(
                var_result,
                basic_constraints,
                horizons=10,
                n_draws=100,
                rotation_method="invalid",
                seed=42,
            )


# =============================================================================
# Layer 3: Monte Carlo Tests
# =============================================================================


class TestSignRestrictionMonteCarlo:
    """Monte Carlo tests for statistical properties."""

    @pytest.mark.slow
    def test_set_width_decreases_with_constraints(self, var_result):
        """More constraints → narrower identified set."""
        # Few constraints
        few_constraints = [
            SignRestrictionConstraint(shock_idx=0, response_idx=0, horizon=0, sign=1),
        ]

        # More constraints
        more_constraints = [
            SignRestrictionConstraint(shock_idx=0, response_idx=0, horizon=0, sign=1),
            SignRestrictionConstraint(shock_idx=0, response_idx=0, horizon=1, sign=1),
            SignRestrictionConstraint(shock_idx=0, response_idx=1, horizon=0, sign=1),
            SignRestrictionConstraint(shock_idx=0, response_idx=1, horizon=1, sign=1),
        ]

        result_few = sign_restriction_svar(
            var_result, few_constraints, horizons=10, n_draws=2000, seed=42
        )
        result_more = sign_restriction_svar(
            var_result, more_constraints, horizons=10, n_draws=2000, seed=42
        )

        # Set width = upper - lower
        width_few = np.mean(result_few.irf_upper - result_few.irf_lower)
        width_more = np.mean(result_more.irf_upper - result_more.irf_lower)

        # More constraints should give equal or narrower set
        # Allow some tolerance for sampling variation
        assert width_more <= width_few * 1.1, (
            f"More constraints should narrow set: {width_more:.4f} vs {width_few:.4f}"
        )

    @pytest.mark.slow
    def test_true_dgp_in_set(self):
        """When DGP is known, true IRF should be within confidence set."""
        # Generate data from known SVAR structure
        np.random.seed(42)
        n = 500
        k = 2

        # True structural parameters
        B0_inv_true = np.array([[1.0, 0.0], [0.5, 1.0]])  # Lower triangular
        A1 = np.array([[0.5, 0.1], [0.2, 0.4]])

        # Generate structural shocks
        eps = np.random.randn(n, k)

        # Reduced form: u = B0_inv @ eps
        data = np.zeros((n, k))
        for t in range(1, n):
            data[t, :] = A1 @ data[t - 1, :] + B0_inv_true @ eps[t, :]

        # Estimate VAR
        var_result = var_estimate(data, lags=1)

        # Constraints consistent with true DGP
        # B0_inv_true has positive diagonal
        constraints = [
            SignRestrictionConstraint(shock_idx=0, response_idx=0, horizon=0, sign=1),
            SignRestrictionConstraint(shock_idx=1, response_idx=1, horizon=0, sign=1),
        ]

        # Run sign restrictions
        result = sign_restriction_svar(var_result, constraints, horizons=10, n_draws=3000, seed=42)

        # True IRF at impact
        true_irf_impact = B0_inv_true

        # Check that true values are within bounds (with tolerance for finite sample)
        for i in range(k):
            for j in range(k):
                lower = result.irf_lower[i, j, 0]
                upper = result.irf_upper[i, j, 0]
                true_val = true_irf_impact[i, j]

                # Allow some slack for finite sample bias
                assert lower - 0.5 <= true_val <= upper + 0.5, (
                    f"True IRF[{i},{j}]={true_val:.3f} outside [{lower:.3f}, {upper:.3f}]"
                )

    @pytest.mark.slow
    def test_sign_correctness_across_set(self, var_result, basic_constraints):
        """All accepted IRFs have correct signs at constrained points."""
        result = sign_restriction_svar(
            var_result, basic_constraints, horizons=10, n_draws=2000, seed=42
        )

        Phi = vma_coefficients(var_result, result.horizons)

        violations = 0
        for B0_inv in result.B0_inv_set:
            irf = _compute_irf_from_impact(Phi, B0_inv, result.horizons)

            for c in basic_constraints:
                val = irf[c.response_idx, c.shock_idx, c.horizon]
                if c.sign > 0 and val <= 0:
                    violations += 1
                if c.sign < 0 and val >= 0:
                    violations += 1

        assert violations == 0, f"Found {violations} sign violations in accepted set"


# =============================================================================
# Additional Tests
# =============================================================================


class TestMonetaryPolicyConstraints:
    """Tests for monetary policy constraint helper."""

    def test_create_constraints(self):
        """Helper creates valid constraints."""
        constraints = create_monetary_policy_constraints(
            money_shock_idx=0,
            output_idx=1,
            price_idx=2,
            max_horizon=4,
        )

        # Should have constraints for output and prices at each horizon
        assert len(constraints) == 10  # 5 horizons * 2 variables

        # Check types
        for c in constraints:
            assert isinstance(c, SignRestrictionConstraint)

    def test_with_interest_rate(self):
        """Including interest rate adds more constraints."""
        constraints = create_monetary_policy_constraints(
            money_shock_idx=0,
            output_idx=1,
            price_idx=2,
            interest_idx=3,
            max_horizon=4,
        )

        # 5 horizons * 3 variables
        assert len(constraints) == 15


class TestSignRestrictionResult:
    """Tests for SignRestrictionResult methods."""

    def test_get_irf_bounds(self, var_result, basic_constraints):
        """get_irf_bounds returns correct dictionary."""
        result = sign_restriction_svar(
            var_result, basic_constraints, horizons=10, n_draws=500, seed=42
        )

        bounds = result.get_irf_bounds(0, 0)

        assert "median" in bounds
        assert "lower" in bounds
        assert "upper" in bounds
        assert "horizon" in bounds

        assert bounds["median"].shape == (11,)
        np.testing.assert_array_equal(bounds["horizon"], np.arange(11))

    def test_to_irf_result(self, var_result, basic_constraints):
        """Conversion to standard IRFResult works."""
        result = sign_restriction_svar(
            var_result, basic_constraints, horizons=10, n_draws=500, seed=42
        )

        irf_result = result.to_irf_result()

        assert irf_result.horizons == 10
        assert irf_result.orthogonalized is True
        assert irf_result.n_bootstrap == result.n_accepted


class TestCheckCholeskyInSet:
    """Tests for check_cholesky_in_set utility."""

    def test_check_returns_bool(self, var_result, basic_constraints):
        """Function returns boolean."""
        result = check_cholesky_in_set(var_result, basic_constraints, horizons=10)
        assert isinstance(result, bool)

    def test_diagonal_positive_constraints(self, var_result):
        """Cholesky diagonal is always positive."""
        # Constraints requiring positive diagonal elements
        constraints = [
            SignRestrictionConstraint(shock_idx=i, response_idx=i, horizon=0, sign=1)
            for i in range(3)
        ]

        # Cholesky should satisfy this
        assert check_cholesky_in_set(var_result, constraints, horizons=10) is True


class TestValidateConstraints:
    """Tests for validate_constraints function."""

    def test_valid_constraints_pass(self):
        """Valid constraints pass validation."""
        constraints = [
            SignRestrictionConstraint(shock_idx=0, response_idx=1, horizon=5, sign=1),
        ]
        # Should not raise
        validate_constraints(constraints, n_vars=3, horizons=10)

    def test_invalid_shock_idx_fails(self):
        """Shock index out of bounds fails."""
        constraints = [
            SignRestrictionConstraint(shock_idx=5, response_idx=0, horizon=0, sign=1),
        ]
        with pytest.raises(ValueError, match="shock_idx"):
            validate_constraints(constraints, n_vars=3, horizons=10)

    def test_invalid_response_idx_fails(self):
        """Response index out of bounds fails."""
        constraints = [
            SignRestrictionConstraint(shock_idx=0, response_idx=5, horizon=0, sign=1),
        ]
        with pytest.raises(ValueError, match="response_idx"):
            validate_constraints(constraints, n_vars=3, horizons=10)

    def test_invalid_horizon_fails(self):
        """Horizon out of bounds fails."""
        constraints = [
            SignRestrictionConstraint(shock_idx=0, response_idx=0, horizon=20, sign=1),
        ]
        with pytest.raises(ValueError, match="horizon"):
            validate_constraints(constraints, n_vars=3, horizons=10)
