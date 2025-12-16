# Phase 2: DiD Monte Carlo Validation Plan

**Created**: 2025-11-27 18:30
**Estimated Time**: 12-15 hours (likely 8-10 with existing patterns)
**Status**: Active

---

## Objective

Implement comprehensive Monte Carlo validation for Difference-in-Differences estimators to verify:
1. Classic 2×2 DiD unbiasedness
2. TWFE bias demonstration with staggered adoption
3. Callaway-Sant'Anna bias correction
4. Sun-Abraham bias correction
5. Event study pre-trends validation

---

## Current State

### Existing Infrastructure
- `tests/validation/monte_carlo/dgp_generators.py` - RCT, PSM, RDD DGPs (569 lines)
- `tests/validation/utils.py` - `validate_monte_carlo_results()` helper
- Pattern: 5,000 runs per test, bias < 0.05-0.10, coverage 93-97%

### DiD Estimators to Validate
| Estimator | Function | Purpose |
|-----------|----------|---------|
| Classic 2×2 | `did_2x2()` | Basic DiD with cluster SE |
| TWFE Staggered | `twfe_staggered()` | Document bias (negative weights) |
| Callaway-Sant'Anna | `callaway_santanna_ate()` | Heterogeneity-robust |
| Sun-Abraham | `sun_abraham_ate()` | IW estimator |
| Event Study | `event_study()` | Dynamic treatment effects |

### No existing DiD Monte Carlo tests

---

## Target State

### File Structure
```
tests/validation/monte_carlo/
├── dgp_generators.py          # Add DiD DGPs (extend)
├── dgp_did.py                 # NEW: DiD-specific DGPs
├── test_monte_carlo_did_2x2.py    # NEW: Classic DiD
├── test_monte_carlo_did_twfe.py   # NEW: TWFE bias demo
├── test_monte_carlo_did_cs.py     # NEW: Callaway-Sant'Anna
├── test_monte_carlo_did_sa.py     # NEW: Sun-Abraham
└── test_monte_carlo_event_study.py # NEW: Event study
```

### Test Count Target: 15 tests, 75,000 simulation runs

---

## Detailed Plan

### Phase 2.1: DGP Generators (2-3h)

Create `dgp_did.py` with:

1. **`dgp_did_2x2_simple()`** - Classic 2×2 DiD
   ```
   DGP:
     Unit-level: i = 1, ..., n (n_treated + n_control)
     Time: t ∈ {0, 1} (pre, post)
     Treatment: D_i = 1 for first n_treated units
     Y_it = α_i + λ_t + τ·D_i·Post_t + ε_it
     α_i ~ N(0, 1), λ_0=0, λ_1~N(0, 0.5), ε ~ N(0, σ)
   True ATT = τ = 2.0
   ```

2. **`dgp_did_2x2_heterogeneous()`** - Heteroskedastic errors
   ```
   Same as above but σ_treated ≠ σ_control
   Tests Neyman variance robustness
   ```

3. **`dgp_staggered_homogeneous()`** - Staggered adoption, same effect
   ```
   DGP:
     Units: n_units, T periods
     Cohorts: Treatment at t=g for cohort g
     Y_it = α_i + λ_t + τ·D_it + ε_it
     τ = 2.0 for all cohorts (homogeneous)
   TWFE should be unbiased here
   ```

4. **`dgp_staggered_heterogeneous()`** - Staggered, different effects
   ```
   DGP:
     Same structure but τ_g varies by cohort
     τ_g = g/2 (e.g., cohort 4: τ=2, cohort 8: τ=4)
   True ATT = weighted average of τ_g
   TWFE biased, CS/SA unbiased
   ```

5. **`dgp_event_study_pretrends()`** - With/without pre-trends
   ```
   DGP:
     Y_it = α_i + λ_t + Σ_k β_k·D_i·1{t-g_i=k} + ε_it
     β_k = 0 for k < 0 (true parallel trends)
     β_k = τ for k >= 0 (constant treatment effect)
   ```

6. **`dgp_event_study_dynamic()`** - Time-varying effects
   ```
   DGP:
     β_k = k·τ_step for k >= 0 (growing effect)
   Tests that event study captures dynamics
   ```

### Phase 2.2: Classic 2×2 DiD Tests (2h)

File: `test_monte_carlo_did_2x2.py`

| Test | n_runs | Sample | Validation |
|------|--------|--------|------------|
| test_did_2x2_unbiased | 5,000 | n=200, 50 treated | Bias < 0.10 |
| test_did_2x2_coverage | 5,000 | n=200 | Coverage 93-97% |
| test_did_2x2_cluster_se_valid | 2,000 | 20 clusters | SE accuracy < 15% |
| test_did_2x2_heteroskedastic | 3,000 | Unequal variance | Coverage OK |

### Phase 2.3: TWFE Bias Demonstration (2h)

File: `test_monte_carlo_did_twfe.py`

| Test | n_runs | Purpose |
|------|--------|---------|
| test_twfe_unbiased_homogeneous | 1,000 | Confirm TWFE works when effects same |
| test_twfe_biased_heterogeneous | 1,000 | Document bias with varying effects |
| test_twfe_bias_magnitude | 500 | Measure and document bias size |

**Key educational output**: Document that TWFE bias can be **negative** even when all true effects positive (due to negative weights on "forbidden comparisons").

### Phase 2.4: Callaway-Sant'Anna Tests (2-3h)

File: `test_monte_carlo_did_cs.py`

| Test | n_runs | Purpose |
|------|--------|---------|
| test_cs_unbiased_homogeneous | 500 | Basic unbiasedness |
| test_cs_unbiased_heterogeneous | 500 | Key: no bias with varying effects |
| test_cs_coverage | 500 | Bootstrap CI coverage |
| test_cs_aggregation_simple | 300 | Simple aggregation correct |
| test_cs_aggregation_dynamic | 300 | Event-time aggregation |

**Note**: CS uses bootstrap (slow), so fewer runs but larger samples.

### Phase 2.5: Sun-Abraham Tests (2h)

File: `test_monte_carlo_did_sa.py`

| Test | n_runs | Purpose |
|------|--------|---------|
| test_sa_unbiased_homogeneous | 500 | Basic IW estimator |
| test_sa_unbiased_heterogeneous | 500 | Key: robust to heterogeneity |
| test_sa_weights_correct | 200 | Verify IW weights sum to 1 |
| test_sa_cohort_effects_unbiased | 300 | Individual cohort-time effects |

### Phase 2.6: Event Study Tests (2h)

File: `test_monte_carlo_event_study.py`

| Test | n_runs | Purpose |
|------|--------|---------|
| test_event_study_pretrends_null | 1,000 | Pre-treatment β_k ≈ 0 |
| test_event_study_post_treatment | 1,000 | Post-treatment β_k ≈ τ |
| test_event_study_dynamic_effects | 500 | Captures time-varying τ_k |

---

## Quality Targets

| Metric | 2×2 DiD | TWFE (bias demo) | CS/SA | Event Study |
|--------|---------|------------------|-------|-------------|
| Bias | < 0.10 | Document bias | < 0.15 | < 0.10 per k |
| Coverage | 93-97% | - | 90-98% | 90-98% |
| SE Accuracy | < 15% | - | < 20% | < 20% |

---

## Decisions Made

1. **Separate DGP file**: Create `dgp_did.py` rather than adding to `dgp_generators.py` (modularity)
2. **Fewer runs for bootstrap methods**: CS/SA use bootstrap, so 500 runs instead of 5000 (runtime)
3. **TWFE tests are educational**: Document bias rather than assert unbiasedness
4. **Event study bins**: Focus on k ∈ {-3, -2, -1, 0, 1, 2, 3} (typical event window)

---

## Implementation Order

1. ✅ Plan document (this file)
2. [ ] `dgp_did.py` - DGP generators
3. [ ] `test_monte_carlo_did_2x2.py` - Classic DiD (highest priority)
4. [ ] `test_monte_carlo_did_twfe.py` - TWFE bias demo
5. [ ] `test_monte_carlo_did_cs.py` - Callaway-Sant'Anna
6. [ ] `test_monte_carlo_did_sa.py` - Sun-Abraham
7. [ ] `test_monte_carlo_event_study.py` - Event study
8. [ ] Documentation update (CURRENT_WORK.md, commit)

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| CS/SA bootstrap slow | Use n_bootstrap=100 for MC, smaller samples |
| TWFE bias magnitude varies | Document typical range, not exact value |
| Event study SE inflated | Known issue, focus on point estimates |
| Monte Carlo seed sensitivity | Use fixed seeds per test, verify stability |

---

## Notes

- Monte Carlo validation is computationally expensive (~10-15 minutes per test file)
- Consider marking slow tests with `@pytest.mark.slow`
- TDD protocol: Write failing tests first, then verify DGPs generate expected structure
