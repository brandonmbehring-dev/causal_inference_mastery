# Session 14: Sharp RDD

**Created**: 2025-11-22 01:30
**Updated**: 2025-11-22 04:15
**Status**: COMPLETED
**Estimated Duration**: 8-10 hours
**Actual Duration**: ~2.5 hours
**Estimated Lines**: ~700 lines (estimator ~350 lines, bandwidth ~200 lines, tests ~400 lines)
**Actual Lines**: 1,320 lines (implementation 550, tests 770)

---

## Objective

Implement Sharp Regression Discontinuity Design (RDD) estimator with local linear regression at cutoff, optimal bandwidth selection (Imbens-Kalyanaraman and Calonico-Cattaneo-Titiunik), and robust bias-corrected confidence intervals. This establishes the foundation for Phase 5, enabling causal inference in settings with treatment determined by a known threshold.

---

## Current State

**Files**:
- No RDD implementation exists yet in Python
- Julia Phase 5 has complete RDD implementation (reference for cross-validation)

**Capabilities**:
- ✅ RCT estimators complete (Session 4)
- ✅ Observational methods complete (Sessions 5-6: IPW, DR)
- ✅ PSM complete (Sessions 1-3, 7)
- ✅ DiD complete (Sessions 8-10)
- ✅ IV complete (Sessions 11-13)
- ❌ No RDD implementation

**References Available**:
- Imbens & Lemieux (2008) - Regression discontinuity designs: A guide to practice
- Cattaneo, Idrobo, Titiunik (2019) - A Practical Introduction to RDD
- Calonico et al. (2014) - Robust nonparametric confidence intervals for RDD
- Julia Phase 5 RDD implementation for cross-validation

---

## Target State

**Files to Create**:
- `src/causal_inference/rdd/sharp_rdd.py` (~350 lines)
- `src/causal_inference/rdd/bandwidth.py` (~200 lines)

**Expected Features**:
- Local linear regression on each side of cutoff
- Treatment effect = difference at cutoff (lim_{x↓c} E[Y|X=x] - lim_{x↑c} E[Y|X=x])
- Imbens-Kalyanaraman (IK) bandwidth selector
- Calonico-Cattaneo-Titiunik (CCT) bandwidth selector
- Robust bias-corrected confidence intervals (CCT 2014)
- Rectangular and triangular kernels
- Optimal bandwidth selection with cross-validation
- Summary table with diagnostics

**Test Coverage**:
- Layer 1 (Known-Answer): 10-12 tests
- Layer 2 (Adversarial): 8-10 tests
- Layer 3: Deferred to Session 16 (Monte Carlo with fuzzy RDD)
- Layer 4: Julia cross-validation

---

## Detailed Plan

### Phase 0: Plan Document Creation (10 minutes)
**Status**: ✅ COMPLETE (01:30)
**Tasks**:
- [x] Create plan document
- [x] Define objective and scope
- [x] Identify implementation approach

### Phase 1: Sharp RDD Estimator Implementation (3-4 hours)
**Status**: NOT_STARTED
**Estimated Start**: 01:40
**Estimated Completion**: 05:40

**Tasks**:
- [ ] Create `src/causal_inference/rdd/` directory
- [ ] Create `src/causal_inference/rdd/__init__.py`
- [ ] Implement `sharp_rdd.py`:
  - [ ] SharpRDD class with .fit() method
  - [ ] Local linear regression on left side of cutoff (x < c)
  - [ ] Local linear regression on right side of cutoff (x ≥ c)
  - [ ] Treatment effect = E[Y|X=c+] - E[Y|X=c-]
  - [ ] Weighted least squares with kernel weights
  - [ ] Rectangular kernel: K(u) = 1 if |u| ≤ 1, else 0
  - [ ] Triangular kernel: K(u) = (1 - |u|) if |u| ≤ 1, else 0
  - [ ] Robust bias-corrected CIs (CCT 2014)
  - [ ] Effective sample sizes (left/right of cutoff)
  - [ ] Summary table generation
- [ ] Input validation:
  - [ ] Check X contains observations on both sides of cutoff
  - [ ] Verify bandwidth > 0
  - [ ] Check sufficient observations near cutoff
  - [ ] Validate alpha bounds

**Expected Structure**:
```python
class SharpRDD:
    """
    Sharp Regression Discontinuity Design estimator.

    Uses local linear regression on each side of the cutoff to estimate
    the treatment effect at the discontinuity.

    Model:
        Left (x < c):  Y = α_L + β_L*(x - c) + ε_L
        Right (x ≥ c): Y = α_R + β_R*(x - c) + ε_R

    Treatment effect: τ = α_R - α_L (difference at cutoff)

    Parameters
    ----------
    cutoff : float
        Threshold value where treatment discontinuously changes
    bandwidth : float or str, default='ik'
        Bandwidth for local linear regression
        - float: Use specified bandwidth
        - 'ik': Imbens-Kalyanaraman optimal bandwidth
        - 'cct': Calonico-Cattaneo-Titiunik optimal bandwidth
    kernel : str, default='triangular'
        Kernel function: 'rectangular' or 'triangular'
    inference : str, default='robust'
        'standard' or 'robust' (bias-corrected CCT)
    alpha : float, default=0.05
        Significance level for confidence intervals

    Attributes (after .fit())
    --------------------------
    coef_ : float
        RDD treatment effect estimate (τ)
    se_ : float
        Standard error (robust if inference='robust')
    t_stat_ : float
        T-statistic
    p_value_ : float
        P-value (two-sided)
    ci_ : tuple
        (lower, upper) confidence interval
    bandwidth_left_ : float
        Bandwidth used on left side
    bandwidth_right_ : float
        Bandwidth used on right side (may differ with CCT)
    n_left_ : int
        Effective sample size on left
    n_right_ : int
        Effective sample size on right
    """

    def __init__(self, cutoff, bandwidth='ik', kernel='triangular',
                 inference='robust', alpha=0.05):
        pass

    def fit(self, Y, X):
        """
        Fit Sharp RDD estimator.

        Parameters
        ----------
        Y : array-like, shape (n,)
            Outcome variable
        X : array-like, shape (n,)
            Running variable (assignment variable)

        Returns
        -------
        self : SharpRDD
            Fitted estimator
        """
        pass

    def summary(self):
        """Return formatted results table."""
        pass

    def _local_linear_regression(self, Y, X, side, bandwidth, kernel):
        """
        Fit local linear regression on one side of cutoff.

        Returns: (intercept, slope, variance)
        """
        pass

    def _kernel_weight(self, u, kernel):
        """Compute kernel weight for normalized distance u."""
        pass
```

### Phase 2: Bandwidth Selection Implementation (2-2.5 hours)
**Status**: NOT_STARTED
**Estimated Start**: 05:40
**Estimated Completion**: 08:10

**Tasks**:
- [ ] Implement `bandwidth.py`:
  - [ ] `imbens_kalyanaraman_bandwidth()` - IK optimal bandwidth
    - [ ] Estimate second derivative of conditional mean (m''(c))
    - [ ] Estimate conditional variance σ²(c)
    - [ ] Compute optimal h = C_IK * n^(-1/5)
    - [ ] Use rule-of-thumb constants from IK (2012)
  - [ ] `cct_bandwidth()` - CCT optimal bandwidth
    - [ ] MSE-optimal bandwidth
    - [ ] Bias correction for coverage
    - [ ] Allow asymmetric bandwidths (left ≠ right)
    - [ ] CER-optimal bandwidth (coverage error rate)
  - [ ] `cross_validation_bandwidth()` - CV bandwidth selection
    - [ ] Leave-one-out cross-validation
    - [ ] Minimize MSE on left-out observations

**Expected Functions**:
```python
def imbens_kalyanaraman_bandwidth(Y, X, cutoff, kernel='triangular'):
    """
    Imbens-Kalyanaraman (2012) optimal bandwidth.

    Rule-of-thumb bandwidth based on minimizing asymptotic MSE
    of local linear estimator.

    h_IK = C_K * [σ²_c / (n * m''(c)²)]^(1/5)

    where:
    - σ²_c: Conditional variance near cutoff
    - m''(c): Second derivative of conditional mean at cutoff
    - C_K: Kernel-specific constant

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome variable
    X : array-like, shape (n,)
        Running variable
    cutoff : float
        RDD cutoff value
    kernel : str
        'rectangular' or 'triangular'

    Returns
    -------
    h_opt : float
        Optimal bandwidth
    """
    pass


def cct_bandwidth(Y, X, cutoff, kernel='triangular',
                  bias_correction=True):
    """
    Calonico-Cattaneo-Titiunik (2014) optimal bandwidth.

    MSE-optimal bandwidth with optional bias correction for
    robust confidence intervals.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome variable
    X : array-like, shape (n,)
        Running variable
    cutoff : float
        RDD cutoff value
    kernel : str
        'rectangular' or 'triangular'
    bias_correction : bool
        If True, use bias-corrected bandwidth

    Returns
    -------
    h_main : float
        Main bandwidth for point estimate
    h_bias : float
        Bias correction bandwidth (if bias_correction=True)
    """
    pass
```

### Phase 3: Test Suite Implementation (2-2.5 hours)
**Status**: NOT_STARTED
**Estimated Start**: 08:10
**Estimated Completion**: 10:40

**Tasks**:
- [ ] Create `tests/test_rdd/` directory
- [ ] Create `tests/test_rdd/__init__.py`
- [ ] Create `tests/test_rdd/conftest.py` with DGP fixtures
- [ ] Test Class 1: TestSharpRDDKnownAnswers (10-12 tests)
  - [ ] test_sharp_rdd_zero_effect - No discontinuity
  - [ ] test_sharp_rdd_known_jump - Known treatment effect at cutoff
  - [ ] test_linear_dgp - Y = X, no discontinuity
  - [ ] test_quadratic_dgp - Y = X², discontinuity at cutoff
  - [ ] test_rectangular_vs_triangular_kernel - Kernel comparison
  - [ ] test_bandwidth_sensitivity - Different bandwidths
  - [ ] test_ik_bandwidth_selection - IK automatic bandwidth
  - [ ] test_cct_bandwidth_selection - CCT automatic bandwidth
  - [ ] test_robust_ci_wider_than_standard - Bias-corrected CIs
  - [ ] test_effective_sample_sizes - n_left + n_right correct
  - [ ] test_summary_table - Summary generation
  - [ ] test_confidence_intervals - CI coverage
- [ ] Test Class 2: TestSharpRDDAdversarial (8-10 tests)
  - [ ] test_minimum_observations_near_cutoff - Edge case
  - [ ] test_sparse_data_near_cutoff - Few observations
  - [ ] test_all_observations_one_side - No control/treatment
  - [ ] test_cutoff_at_boundary - Cutoff at min/max of X
  - [ ] test_bandwidth_larger_than_range - h > range(X)
  - [ ] test_high_variance - Noisy outcomes
  - [ ] test_outliers - Robust to outliers?
  - [ ] test_non_monotonic_relationship - Y not monotonic in X
  - [ ] test_invalid_inputs - Error handling
- [ ] Fixtures:
  - [ ] Create sharp RDD DGP (Y = f(X) + τ*1{X≥c} + ε)
  - [ ] Create DGPs with different functional forms
  - [ ] Create DGP with no discontinuity (validation)

**Expected DGP Structure**:
```python
def generate_sharp_rdd_dgp(
    n=500,
    cutoff=0.0,
    true_effect=2.0,
    functional_form='linear',
    noise_sd=1.0,
    seed=None
):
    """
    Generate Sharp RDD data.

    DGP:
    - X ~ Uniform(-5, 5) (running variable)
    - Treatment: D = 1{X ≥ cutoff}
    - Outcome: Y = f(X) + τ*D + ε, ε ~ N(0, σ²)

    where f(X) is the functional form (linear, quadratic, cubic, etc.)

    Parameters
    ----------
    n : int
        Sample size
    cutoff : float
        Treatment threshold
    true_effect : float
        True treatment effect at cutoff
    functional_form : str
        'linear', 'quadratic', 'cubic', 'sine'
    noise_sd : float
        Standard deviation of noise
    seed : int, optional
        Random seed

    Returns
    -------
    Y : array, shape (n,)
        Outcome
    X : array, shape (n,)
        Running variable
    true_effect : float
        True treatment effect
    """
    pass
```

### Phase 4: Documentation (1-1.5 hours)
**Status**: NOT_STARTED
**Estimated Start**: 10:40
**Estimated Completion**: 12:10

**Tasks**:
- [ ] Create `docs/SESSION_14_SHARP_RDD_2025-11-22.md`
- [ ] Document Sharp RDD implementation approach
- [ ] Document bandwidth selection methods (IK vs CCT)
- [ ] Document test results (pass rate, coverage)
- [ ] Document kernel comparison (rectangular vs triangular)
- [ ] Update `src/causal_inference/rdd/README.md` with:
  - [ ] RDD overview
  - [ ] Use cases and examples
  - [ ] When to use Sharp vs Fuzzy RDD
  - [ ] Bandwidth selection guidance
  - [ ] Limitations and assumptions
- [ ] Update `docs/ROADMAP.md`: Mark Session 14 as COMPLETE

### Phase 5: Commit and Wrap-up (10 minutes)
**Status**: NOT_STARTED
**Estimated Start**: 12:10
**Estimated Completion**: 12:20

**Tasks**:
- [ ] Run full test suite: `pytest tests/test_rdd/ -v`
- [ ] Verify all tests passing
- [ ] Git add all files
- [ ] Git commit with comprehensive message
- [ ] Move plan to docs/plans/implemented/

---

## Decisions Made

*To be filled during implementation*

**Implementation Decisions**:
- (Will document: Local linear vs local constant, kernel choice, bandwidth defaults)

**Bandwidth Decisions**:
- (Will document: IK vs CCT tradeoffs, when to use each, CV as fallback)

**Testing Decisions**:
- (Will document: DGP functional forms, bandwidth ranges tested)

---

## Testing Strategy

**Coverage Goals**:
- Sharp RDD estimator: 90%+ (core implementation)
- Bandwidth selection: 85%+ (IK and CCT)
- Kernel functions: 100% (simple, critical)

**Validation Approach**:
- Layer 1 (Known-Answer): Test on known DGPs with analytical solutions
- Layer 2 (Adversarial): Test edge cases (sparse data, one-sided, outliers)
- Layer 3: Deferred to Session 16 (Monte Carlo)
- Layer 4 (Julia): Cross-validate estimates and bandwidths

**Key Tests**:
1. Zero effect DGP → RDD estimate ≈ 0
2. Known jump DGP → RDD estimate ≈ true effect
3. Linear DGP (no jump) → RDD estimate ≈ 0
4. IK bandwidth → reasonable (0.5 to 2.0 typical range)
5. CCT robust CI → wider than standard CI
6. Kernel comparison → triangular more efficient

---

## Success Criteria

- [ ] Sharp RDD estimator implemented and tested
- [ ] IK bandwidth selection working
- [ ] CCT bandwidth selection working
- [ ] Robust bias-corrected CIs functional
- [ ] 18-22 tests passing (10-12 known-answer + 8-10 adversarial)
- [ ] Coverage ≥ 90%
- [ ] Session summary document created
- [ ] ROADMAP updated
- [ ] All work committed to git

---

## Notes

**Time Estimate Justification**:
- Sharp RDD implementation: 3-4 hours (local linear, kernels, robust CIs)
- Bandwidth selection: 2-2.5 hours (IK + CCT algorithms)
- Test suite: 2-2.5 hours (18-22 tests, DGP creation)
- Documentation: 1-1.5 hours (session summary, README, roadmap update)
- Total: 8-10 hours

**Risks**:
- IK bandwidth may require numerical optimization (second derivative estimation)
- CCT bandwidth is complex (bias correction, asymmetric bandwidths)
- Robust CIs may require careful SE calculation

**Mitigation**:
- Use rule-of-thumb approximations for IK (simpler than full optimization)
- Implement basic CCT first, add refinements later
- Follow CCT (2014) paper formulas exactly for robust SEs

**Defer to Future Sessions**:
- McCrary density test (Session 15)
- Covariate balance checks (Session 15)
- Placebo cutoffs (Session 15)
- Fuzzy RDD (Session 16)
- Monte Carlo validation (Session 16)

---

## Methodological Concerns Addressed

**From METHODOLOGICAL_CONCERNS.md**:

- **CONCERN-22**: McCrary density test for manipulation
  - Status: NOT YET (deferred to Session 15)
  - This session: Sharp RDD only

- **CONCERN-23**: Bandwidth sensitivity analysis
  - Status: ✅ ADDRESSED in this session
  - Implementation: Multiple bandwidth selectors (IK, CCT, CV), test sensitivity

- **CONCERN-24**: Covariate balance checks
  - Status: NOT YET (deferred to Session 15)
  - This session: Sharp RDD only

---

## References

**Core Papers**:
- Imbens, Guido W., and Thomas Lemieux. 2008. "Regression Discontinuity Designs: A Guide to Practice." Journal of Econometrics 142(2): 615-635.
  - Overview of RDD methods
  - IK bandwidth selector

- Calonico, Sebastian, Matias D. Cattaneo, and Rocio Titiunik. 2014. "Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs." Econometrica 82(6): 2295-2326.
  - CCT bandwidth selector
  - Robust bias-corrected confidence intervals

- Lee, David S., and Thomas Lemieux. 2010. "Regression Discontinuity Designs in Economics." Journal of Economic Literature 48(2): 281-355.
  - RDD overview and applications

**Textbooks**:
- Cattaneo, Matias D., Nicolás Idrobo, and Rocío Titiunik. 2019. *A Practical Introduction to Regression Discontinuity Designs: Foundations*. Cambridge University Press.
  - Comprehensive RDD guide

**Software References**:
- rdrobust package (R/Stata) - Reference implementation by Cattaneo et al.
- Julia Phase 5 RDD implementation - Cross-validation reference

---

## Implementation Notes

### Local Linear Regression

**Model**:
```
Left side (x < c):  Y_i = α_L + β_L*(X_i - c) + ε_i
Right side (x ≥ c): Y_i = α_R + β_R*(X_i - c) + ε_i
```

**Treatment effect**: τ = α_R - α_L

**Why local linear (not local constant)**:
- Reduces bias when true function is non-linear
- Better boundary properties
- Standard practice in RDD literature

### Kernel Choice

**Triangular kernel** (default):
- K(u) = (1 - |u|) if |u| ≤ 1, else 0
- More efficient than rectangular
- Downweights observations far from cutoff

**Rectangular kernel**:
- K(u) = 1 if |u| ≤ 1, else 0
- Simpler, less efficient
- Useful for comparison/robustness

### Bandwidth Selection Tradeoffs

**Imbens-Kalyanaraman (IK)**:
- Pros: Simple, rule-of-thumb, fast
- Cons: May be too large (oversmoothing) in some cases
- Use when: Quick analysis, standard RDD

**Calonico-Cattaneo-Titiunik (CCT)**:
- Pros: MSE-optimal, bias correction, robust CIs
- Cons: More complex, computationally intensive
- Use when: Publication-quality results, robust inference

**Cross-Validation (CV)**:
- Pros: Data-driven, no parametric assumptions
- Cons: Slow, may select too small bandwidth
- Use when: Uncertain about functional form

---

## Mathematical Background

### Local Linear Regression Weights

For observation i with running variable X_i, kernel weight is:
```
w_i = K((X_i - c) / h)
```

where:
- K(·) is the kernel function
- h is the bandwidth
- c is the cutoff

### Weighted Least Squares

Minimize on each side:
```
Σ w_i * [Y_i - α - β*(X_i - c)]²
```

Solution (matrix form):
```
[α̂, β̂]' = (X'WX)^(-1) X'WY
```

where:
- X = [1, X_i - c] (design matrix)
- W = diag(w_1, ..., w_n) (weight matrix)

### RDD Estimate

```
τ̂ = α̂_R - α̂_L
```

### Standard Error

Variance of τ̂:
```
Var(τ̂) = Var(α̂_R) + Var(α̂_L) - 2*Cov(α̂_R, α̂_L)
```

Under independence (separate regressions):
```
Var(τ̂) = Var(α̂_R) + Var(α̂_L)
```

Robust SE accounts for heteroskedasticity using sandwich estimator.

---

**Session 14 Plan Status**: ✅ READY TO IMPLEMENT
**Next Step**: Begin Phase 1 - Sharp RDD Estimator Implementation
