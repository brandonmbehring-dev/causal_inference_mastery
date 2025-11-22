# Session 7: PSM Monte Carlo Validation

**Created**: 2025-11-22 00:30
**Updated**: 2025-11-22 00:30
**Status**: NOT_STARTED
**Estimated Duration**: 6-8 hours
**Estimated Lines**: ~350 lines (Monte Carlo tests)

---

## Objective

Complete Layer 3 (Monte Carlo) validation for PSM to finish Phase 2. Validate bias, coverage, and SE accuracy using hybrid DGPs (Julia + Python patterns). Compare with/without replacement, caliper variations, and verify Abadie-Imbens SE matches empirical SD.

---

## Current State

**Files**:
- Module: `src/causal_inference/psm/` (Sessions 1-3 complete)
- Estimators: PropensityScoreEstimator, Matching, Variance (Abadie-Imbens), Balance, psm_ate()
- Tests: Layers 1-2 complete (18 tests, 100% pass rate)

**Capabilities**:
- ✅ With/without replacement matching
- ✅ Caliper enforcement
- ✅ Abadie-Imbens (2006) variance
- ✅ Balance diagnostics (SMD, variance ratios)
- ❌ No Monte Carlo validation (Layer 3)

**References Available**:
- Abadie & Imbens (2006, 2008) - PSM variance theory
- Austin (2011) - Balance diagnostics
- Julia Phase 2 - Complete PSM implementation for cross-validation

---

## Target State

**File to Create**:
- `tests/validation/monte_carlo/test_monte_carlo_psm.py` (~350 lines)

**Expected Features**:
- 5 DGPs (2-3 from Julia, 2-3 Python-specific)
- 5000 runs per DGP (25,000 simulations total)
- Validation metrics: Bias < 0.10, Coverage 93-97.5%, SE accuracy < 15%
- Comparisons: With/without replacement, caliper variations
- Verification: AI SE matches empirical SD

**Test Coverage**:
- Homoskedastic DGP (n=100, 500, 1000)
- Heteroskedastic DGP
- High confounding DGP
- Weak overlap DGP
- With/without replacement comparison

---

## Detailed Plan

### Phase 0: Plan Document Creation (10 minutes)
**Status**: ✅ COMPLETE (00:30)
**Tasks**:
- [x] Create plan document
- [x] Define objective and scope
- [x] Identify DGPs and metrics

### Phase 1: DGP Design (1-1.5 hours)
**Status**: NOT_STARTED
**Estimated Start**: 00:40
**Estimated Completion**: 02:10

**Tasks**:
- [ ] Review Julia Phase 2 DGPs for replication candidates
- [ ] Select 2-3 DGPs from Julia for cross-validation
- [ ] Design 2-3 Python-specific DGPs (following Sessions 5-6 patterns)
- [ ] Document DGP specifications:
  - [ ] DGP 1: Homoskedastic, moderate confounding (n=500)
  - [ ] DGP 2: Heteroskedastic, high confounding (n=1000)
  - [ ] DGP 3: Weak overlap (n=500)
  - [ ] DGP 4: Julia cross-validation DGP
  - [ ] DGP 5: Large sample (n=1000)

**Expected DGP Structure**:
```python
def generate_psm_dgp_homoskedastic(n=500, seed=None):
    """
    Moderate confounding, homoskedastic errors.

    DGP:
    - X ~ N(0, 1)
    - P(T=1|X) = logit^(-1)(0.5 + 0.8*X)
    - Y₁ = 1.0 + 0.5*X + ε₁, ε₁ ~ N(0, 1)
    - Y₀ = 0.5*X + ε₀, ε₀ ~ N(0, 1)
    - True ATE = 1.0
    """
```

### Phase 2: Monte Carlo Test Implementation (3-4 hours)
**Status**: NOT_STARTED
**Estimated Start**: 02:10
**Estimated Completion**: 06:10

**Tasks**:
- [ ] Create `tests/validation/monte_carlo/test_monte_carlo_psm.py`
- [ ] Test Class 1: TestPSMMonteCarloHomoskedastic
  - [ ] test_psm_n100_with_replacement (5000 runs)
  - [ ] test_psm_n500_with_replacement (5000 runs)
  - [ ] test_psm_n1000_with_replacement (5000 runs)
- [ ] Test Class 2: TestPSMMonteCarloHeteroskedastic
  - [ ] test_psm_heteroskedastic (5000 runs)
- [ ] Test Class 3: TestPSMMonteCarloHighConfounding
  - [ ] test_psm_high_confounding (5000 runs)
- [ ] Test Class 4: TestPSMMonteCarloComparisons
  - [ ] test_with_vs_without_replacement
  - [ ] test_caliper_sensitivity
  - [ ] test_ai_se_vs_empirical_sd

**Monte Carlo Structure**:
```python
class TestPSMMonteCarloHomoskedastic:
    def test_psm_n500_with_replacement(self):
        """Test PSM with replacement on homoskedastic DGP."""
        results = []
        for i in range(5000):
            # Generate data
            Y, T, X, true_ate = generate_psm_dgp_homoskedastic(n=500, seed=i)

            # Estimate
            ate, se, _ = psm_ate(Y, T, X, with_replacement=True)

            # Store
            results.append({
                'ate': ate,
                'se': se,
                'ci_lower': ate - 1.96*se,
                'ci_upper': ate + 1.96*se,
                'covers': (ate - 1.96*se <= true_ate <= ate + 1.96*se)
            })

        # Validate
        df = pd.DataFrame(results)
        assert abs(df['ate'].mean() - true_ate) < 0.10  # Bias
        assert 0.93 <= df['covers'].mean() <= 0.975  # Coverage
        assert abs(df['se'].mean() / df['ate'].std() - 1.0) < 0.15  # SE accuracy
```

**Expected Metrics**:
- Bias: < 0.10 (relaxed vs RCT due to matching approximation)
- Coverage: 93-97.5% (nominal 95%)
- SE accuracy: AI SE / empirical SD within 15%

### Phase 3: Documentation (1-1.5 hours)
**Status**: NOT_STARTED
**Estimated Start**: 06:10
**Estimated Completion**: 07:40

**Tasks**:
- [ ] Create `docs/SESSION_7_PSM_MONTE_CARLO_2025-11-22.md`
- [ ] Document Monte Carlo results (bias, coverage, SE accuracy)
- [ ] Compare with/without replacement performance
- [ ] Compare caliper sensitivity
- [ ] Julia vs Python comparison (for replicated DGPs)
- [ ] PSM vs IPW vs DR comparison
- [ ] Update `docs/ROADMAP.md`: Mark Phase 2 as COMPLETE

### Phase 4: Commit and Wrap-up (10 minutes)
**Status**: NOT_STARTED
**Estimated Start**: 07:40
**Estimated Completion**: 07:50

**Tasks**:
- [ ] Run full test suite: `pytest tests/test_psm/ tests/validation/ -v`
- [ ] Verify all tests passing
- [ ] Git add all files
- [ ] Git commit with comprehensive message
- [ ] Move plan to docs/plans/implemented/

---

## Decisions Made

*To be filled during implementation*

**DGP Design Decisions**:
- (Will document: Which Julia DGPs replicated, Python-specific DGP rationale)

**Metric Thresholds**:
- (Will document: Why bias < 0.10, coverage 93-97.5%, etc.)

**Comparison Decisions**:
- (Will document: With/without replacement findings, caliper recommendations)

---

## Testing Strategy

**Coverage Goals**:
- PSM Monte Carlo: 5 DGPs × 5000 runs = 25,000 simulations
- Validate all PSM features: matching, variance, balance
- Compare configurations: with/without replacement, caliper values

**Validation Approach**:
- Bias: Empirical mean of estimates vs true ATE
- Coverage: Proportion of 95% CIs containing true ATE
- SE accuracy: Ratio of mean AI SE to empirical SD

**Cross-Validation** (if Julia DGPs replicated):
- Run same DGP in Julia and Python
- Compare bias, coverage, SE
- Document any differences

---

## Success Criteria

- [ ] 5 DGPs implemented and documented
- [ ] 25,000 Monte Carlo simulations completed
- [ ] All metrics within targets (bias, coverage, SE)
- [ ] With/without replacement compared
- [ ] Caliper sensitivity analyzed
- [ ] Session summary document created
- [ ] Phase 2 marked COMPLETE in roadmap
- [ ] All work committed to git

---

## Notes

**Time Estimate Justification**:
- DGP design: 1-1.5 hours (review Julia, design Python DGPs)
- Monte Carlo implementation: 3-4 hours (5 DGPs, comparison tests)
- Documentation: 1-1.5 hours (results, comparisons, roadmap update)
- Total: 6-8 hours

**Risks**:
- Monte Carlo simulations may take time to run (5000 runs × 5 DGPs)
- Julia cross-validation may require PyCall setup
- Caliper sensitivity may show unexpected patterns

**Mitigation**:
- Run simulations in background if needed
- Julia cross-validation optional (nice-to-have)
- Document all findings, even if unexpected
