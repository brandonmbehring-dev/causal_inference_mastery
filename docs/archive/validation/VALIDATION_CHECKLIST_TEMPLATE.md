# Validation Checklist Template

**Purpose**: Ensure all causal inference methods meet research-grade quality standards through comprehensive six-layer validation.

**For**: Future methods (PSM, RDD, IV, DiD, etc.)

**Based On**: RCT validation architecture established in Phase 1

---

## Pre-Implementation Checklist

Before writing any estimator code:

- [ ] Read `docs/PYTHON_VALIDATION_ARCHITECTURE.md`
- [ ] Read `docs/standards/PHASE_COMPLETION_STANDARDS.md`
- [ ] Identify method-specific edge cases (e.g., PSM: no matches; RDD: bandwidth selection)
- [ ] Plan Data Generating Processes (DGPs) for Monte Carlo validation
- [ ] Create plan document in `docs/plans/active/`

---

## Layer 1: Known-Answer Tests ✅

**Minimum**: 10+ tests per estimator
**Coverage**: >90% line coverage per estimator file

**Checklist**:
- [ ] **Basic case**: Hand-calculated example with known outcome
- [ ] **Zero effect**: Treatment has no effect (ATE = 0)
- [ ] **Negative effect**: Treatment decreases outcome (ATE < 0)
- [ ] **Balanced design**: Equal sample sizes (if applicable)
- [ ] **Imbalanced design**: Unequal sample sizes
- [ ] **Small sample**: n = 20 (tests t-distribution if applicable)
- [ ] **Large sample**: n = 1000 (tests numerical stability)
- [ ] **Edge case 1**: Method-specific (e.g., PSM: caliper = 0)
- [ ] **Edge case 2**: Method-specific (e.g., RDD: bandwidth → ∞)
- [ ] **Edge case 3**: Method-specific (e.g., IV: weak instruments)

**Files**:
- [ ] Create `tests/test_[method]/test_[estimator].py`
- [ ] Create `tests/test_[method]/conftest.py` (shared fixtures)

**Example Structure**:
```python
def test_[estimator]_basic_case():
    """Basic case with hand-calculated expected value."""
    # Setup data
    # Call estimator
    # Assert: estimate == expected (rtol=1e-10)
    # Assert: se > 0
    # Assert: ci_lower < estimate < ci_upper

def test_[estimator]_zero_effect():
    """Zero treatment effect."""
    # ...

def test_[estimator]_small_sample():
    """Small sample (n=20) for t-distribution validation."""
    # ...
```

---

## Layer 2: Adversarial Tests 🛡️

**Minimum**: 10+ tests per estimator (50+ total for method)
**Goal**: Find bugs before they reach production

**Checklist**:
- [ ] **Extreme sample sizes**: n=2, n=3, n=10000, n=1000000
- [ ] **Extreme imbalance**: n1=1 n0=999, n1=999 n0=1
- [ ] **Extreme variance**: σ²=0.001, σ²=1000000
- [ ] **Numerical stability**: Values near machine precision (1e-15)
- [ ] **Outliers**: Extreme outcome values (1e6, -1e6)
- [ ] **Perfect separation**: All treated=100, all control=0
- [ ] **Tied values**: All outcomes identical
- [ ] **Missing data**: NaN, Inf, -Inf values (should error cleanly)
- [ ] **Method-specific edge 1**: (e.g., PSM: no matches within caliper)
- [ ] **Method-specific edge 2**: (e.g., RDD: all observations on one side)
- [ ] **Method-specific edge 3**: (e.g., IV: instruments = endogenous vars)

**Files**:
- [ ] Create `tests/validation/adversarial/test_[estimator]_adversarial.py`

**Example Structure**:
```python
class Test[Estimator]Adversarial:
    """Adversarial tests for [estimator]."""

    def test_minimum_sample_n2(self):
        """n=2 (n1=1, n0=1) - minimum possible sample."""
        # Test estimator doesn't crash
        # Either: returns valid result OR raises clear error

    def test_extreme_imbalance(self):
        """n1=1, n0=999 - extreme treatment imbalance."""
        # ...

    @pytest.mark.xfail(reason="Known issue: [description]")
    def test_known_bug(self):
        """Document known edge case bug."""
        # Test that SHOULD pass but doesn't yet
        # Will auto-fail when bug is fixed
```

---

## Layer 3: Monte Carlo Validation 📊

**Minimum**: 5+ tests per estimator
**Runs**: 1000 per test
**Performance**: All tests < 60 seconds total

**Checklist**:
- [ ] **DGP 1: Simple/baseline**: Standard case with known τ = 2.0
- [ ] **DGP 2: Heteroskedastic**: Different variances (σ²_1 ≠ σ²_0)
- [ ] **DGP 3: Small sample**: n = 20-50 for t-distribution validation
- [ ] **DGP 4: Large sample**: n = 1000-5000 for asymptotic validation
- [ ] **DGP 5: Method-specific**: (e.g., PSM: confounded treatment assignment)
- [ ] **Validation 1: Bias**: |mean(estimates) - 2.0| < 0.05
- [ ] **Validation 2: Coverage**: 93% < proportion(CI contains 2.0) < 97%
- [ ] **Validation 3: SE accuracy**: std(estimates) ≈ mean(SE) within 10-20%

**Files**:
- [ ] Create `tests/validation/monte_carlo/dgp_generators.py` (DGP functions)
- [ ] Create `tests/validation/monte_carlo/test_monte_carlo_[estimator].py`
- [ ] Update `tests/validation/utils.py` if new validation helpers needed

**Example DGP**:
```python
def dgp_[method]_simple(n=100, true_ate=2.0, random_state=None):
    """
    Simple DGP for [method] with known true ATE.

    Parameters
    ----------
    n : int
        Sample size
    true_ate : float
        True average treatment effect
    random_state : int or None
        Random seed for reproducibility

    Returns
    -------
    tuple
        (outcomes, treatment, [method-specific args])
    """
    rng = np.random.RandomState(random_state)
    # Generate data with known ATE
    # Return all necessary arrays
```

**Example Test**:
```python
def test_monte_carlo_[estimator]():
    """Validate [estimator] with 1000 runs."""
    n_runs = 1000
    true_ate = 2.0

    estimates, ses, ci_lowers, ci_uppers = [], [], [], []

    for seed in range(n_runs):
        data = dgp_[method]_simple(n=100, true_ate=true_ate, random_state=seed)
        result = [estimator](*data)
        estimates.append(result["estimate"])
        ses.append(result["se"])
        ci_lowers.append(result["ci_lower"])
        ci_uppers.append(result["ci_upper"])

    # Validate using helper
    validation = validate_monte_carlo_results(
        estimates, ses, ci_lowers, ci_uppers, true_ate=true_ate
    )

    assert validation["bias_ok"], f"Bias {validation['bias']:.4f} exceeds threshold"
    assert validation["coverage_ok"], f"Coverage {validation['coverage']:.4f} outside [0.93, 0.97]"
    assert validation["se_accuracy_ok"], f"SE accuracy {validation['se_accuracy']:.4f} exceeds threshold"
```

---

## Layer 4: Cross-Language Validation 🔄

**Status**: Optional (Julia→Python already exists for RCT)
**Tolerance**: rtol < 1e-10 (near machine precision)

**Checklist**:
- [ ] Decide: Is Python→Julia validation needed?
  - YES if: Python is primary implementation or for publication
  - NO if: Julia→Python sufficient and juliacall has issues
- [ ] If YES:
  - [ ] Create `tests/validation/cross_language/julia_interface.py` wrapper
  - [ ] Create `tests/validation/cross_language/test_python_julia_[estimator].py`
  - [ ] Test on basic case, stratified, with covariates, small sample, large sample
  - [ ] Mark as `pytest.mark.skipif(not is_julia_available())` for graceful skipping

**Example Test**:
```python
pytestmark = pytest.mark.skipif(
    not is_julia_available(),
    reason="Julia not available for cross-validation"
)

def test_python_julia_[estimator]():
    """Cross-validate Python vs Julia for [estimator]."""
    # Generate test data
    data = [test data]

    # Python estimate
    python_result = [python_estimator](*data)

    # Julia estimate
    julia_result = julia_[estimator](*data)

    # Assert agreement to machine precision
    assert np.isclose(python_result["estimate"], julia_result["estimate"], rtol=1e-10)
    assert np.isclose(python_result["se"], julia_result["se"], rtol=1e-10)
    assert np.isclose(python_result["ci_lower"], julia_result["ci_lower"], rtol=1e-10)
    assert np.isclose(python_result["ci_upper"], julia_result["ci_upper"], rtol=1e-10)
```

---

## Layer 5: R Triangulation 🔺

**Status**: Optional (deferred for Python unless publishing)

**Checklist**:
- [ ] Decide: Is R validation needed?
  - YES if: Publishing paper or validation layers 1-4 insufficient
  - NO if: Python validation layers 1-3 provide sufficient coverage
- [ ] If YES:
  - [ ] Create `validation/r_scripts/validate_[method].R`
  - [ ] Implement all estimators in R
  - [ ] Save results to CSV
  - [ ] Python test reads CSV and compares
  - [ ] Graceful skip if R unavailable

---

## Layer 6: Golden Reference Tests 📜

**Minimum**: 6 carefully designed datasets
**Storage**: `tests/golden_results/python_golden_results_[method].json`

**Checklist**:
- [ ] **Dataset 1: Balanced design**: Standard balanced case
- [ ] **Dataset 2: Imbalanced design**: Unequal sample sizes
- [ ] **Dataset 3: Small sample**: n=20 for t-distribution
- [ ] **Dataset 4: Large sample**: n=1000 for precision
- [ ] **Dataset 5: Method-specific 1**: (e.g., PSM: poor overlap)
- [ ] **Dataset 6: Method-specific 2**: (e.g., RDD: discontinuity at cutoff)
- [ ] Run all estimators on all 6 datasets
- [ ] Save results to JSON (estimates, SEs, CIs, diagnostics)
- [ ] Create test that validates against frozen JSON
- [ ] Update JSON ONLY when estimator changes (with justification)

**Example**:
```python
def test_golden_reference_[estimator]():
    """Validate [estimator] against golden reference results."""
    # Load golden results
    with open("tests/golden_results/python_golden_results_[method].json") as f:
        golden = json.load(f)

    # Run estimator on each dataset
    for dataset_name in ["balanced", "imbalanced", "small_sample", ...]:
        data = load_dataset(dataset_name)
        result = [estimator](*data)

        # Compare to frozen results
        expected = golden[dataset_name]["[estimator]"]
        assert np.isclose(result["estimate"], expected["estimate"], rtol=1e-12)
        assert np.isclose(result["se"], expected["se"], rtol=1e-12)
```

---

## Documentation Checklist 📚

**Required Files**:
- [ ] Update `README.md` with method status
- [ ] Update `docs/PYTHON_VALIDATION_ARCHITECTURE.md` if new patterns emerge
- [ ] Update `docs/ROADMAP.md` with completion evidence
- [ ] Create `CURRENT_WORK.md` entry documenting progress

**Estimator Docstrings** (all estimators):
- [ ] Mathematical foundation (equation for estimand)
- [ ] Variance estimator with justification
- [ ] Parameters with types
- [ ] Returns with descriptions
- [ ] Usage example
- [ ] Known limitations
- [ ] References (if applicable)

**Example Docstring Structure**:
```python
def [estimator](outcome, treatment, [other_args], alpha=0.05):
    """
    [Brief description of estimator]

    Mathematical Foundation
    -----------------------
    Estimates the Average Treatment Effect (ATE):
        τ = E[Y(1) - Y(0)]

    Using [method description]:
        τ_hat = [estimator formula]

    Variance Estimator
    ------------------
    Uses [variance type] variance:
        Var(τ_hat) = [variance formula]

    Justification: [Why this variance estimator]

    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes (n,)
    treatment : np.ndarray
        Treatment assignments (0/1), shape (n,)
    alpha : float, default=0.05
        Significance level for confidence interval

    Returns
    -------
    dict
        - estimate : float - ATE estimate
        - se : float - Standard error
        - ci_lower : float - Lower CI bound
        - ci_upper : float - Upper CI bound
        - [method-specific diagnostics]

    Examples
    --------
    >>> outcome = np.array([10, 8, 4, 2])
    >>> treatment = np.array([1, 1, 0, 0])
    >>> result = [estimator](outcome, treatment)
    >>> result["estimate"]
    6.0

    Notes
    -----
    Known limitations:
    - [Limitation 1]
    - [Limitation 2]

    References
    ----------
    - [Citation if applicable]
    """
```

---

## Verification Checklist ✅

**Before marking phase complete**:

- [ ] All Layer 1 tests passing (>90% coverage)
- [ ] All Layer 2 tests passing (or xfail with documented reason)
- [ ] All Layer 3 tests passing (bias, coverage, SE accuracy validated)
- [ ] Layer 4 complete OR rational decision to defer documented
- [ ] Layer 5 complete OR rational decision to defer documented
- [ ] Layer 6 golden reference created and validated
- [ ] All estimator docstrings complete
- [ ] Documentation files updated
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Check coverage: `pytest tests/ --cov=src/causal_inference/[method]`
- [ ] Verify no xfail tests without documented bugs
- [ ] Update plan document with completion timestamp
- [ ] Create git commit with evidence of completion

---

## Common Pitfalls 🚨

**Avoid These**:
1. ❌ Skipping adversarial tests ("my code handles edge cases")
   - Reality: Adversarial tests WILL find bugs (found n=1 bug in RCT)
2. ❌ Using tight Monte Carlo thresholds (e.g., coverage 94-96%)
   - Reality: Use 93-97% to account for statistical variation
3. ❌ Not documenting known bugs with xfail
   - Reality: xfail auto-fails when bug is fixed, forcing cleanup
4. ❌ Generating golden reference JSON without manual verification
   - Reality: Golden results should be manually validated ONCE
5. ❌ Implementing all estimators before any validation
   - Reality: Validate incrementally (1 estimator → full validation → next)
6. ❌ Relying only on known-answer tests
   - Reality: Need adversarial + Monte Carlo to catch statistical bugs

---

## Lessons Learned from RCT Phase

**What Worked**:
- ✅ Adversarial tests found real bugs (n=1 produces NaN SE)
- ✅ xfail pattern documents bugs and auto-alerts when fixed
- ✅ Monte Carlo validation caught coverage issues
- ✅ Validation utilities (`validate_monte_carlo_results()`) saved time
- ✅ Incremental validation prevented accumulating bugs

**What to Improve**:
- Coverage thresholds: Use 93-97% not 94-96% (accounts for MC variation)
- Fix 2 tests expecting old buggy behavior after fixes
- Document decisions to defer layers (e.g., Layer 4 for RCT)

---

## Time Estimates

**Typical duration for comprehensive validation** (based on RCT):
- Layer 1 (Known-Answer): 3-4 hours
- Layer 2 (Adversarial): 4-5 hours (most time-consuming)
- Layer 3 (Monte Carlo): 3-4 hours (DGP design + tests)
- Layer 4 (Cross-Language): 2-3 hours (if activated)
- Layer 5 (R Triangulation): Deferred
- Layer 6 (Golden Reference): 1-2 hours
- Documentation: 1-2 hours

**Total**: 12-16 hours for full six-layer validation per method

---

**Template Version**: 1.0
**Created**: 2025-11-20
**Based On**: RCT validation layers (Phases 1-5 completed)
