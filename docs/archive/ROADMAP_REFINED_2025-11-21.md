# Refined Roadmap: Causal Inference Implementation

Based on audit of Sessions 1-4 and discovery of existing RCT code.

---

## Phase 1: Foundation (Sessions 1-5) ✓ CURRENT

### Sessions 1-4: PSM Implementation ✓ COMPLETE
- Clean architecture with separation of concerns
- Three-layer testing strategy proven effective
- Monte Carlo validation revealed important limitations
- 77% test coverage achieved

### Session 5: RCT Estimators Testing [NEXT]
**Why**: 1,266 lines of RCT code exists at 0% coverage

**Scope** (4-6 hours):
1. Test `simple_ate` - baseline difference-in-means
2. Test `ipw_ate` - foundation for observational IPW
3. Test `regression_ate` - foundation for regression adjustment
4. Test `stratified_ate` - block randomization support

**Testing approach**:
- Layer 1: Known-answer tests (perfect RCT scenarios)
- Layer 2: Edge cases (small samples, perfect separation)
- Layer 3: Monte Carlo (1000 runs, verify unbiasedness)

**Expected outcome**:
- 80%+ coverage on RCT modules
- Validated foundation for observational extensions
- Clear understanding of what needs adaptation

---

## Phase 2: Observational Extensions (Sessions 6-8)

### Session 6: IPW for Observational Data
**Build on tested `ipw_ate`**:

**New components** (4-6 hours):
- Integrate propensity estimation (reuse from PSM)
- Weight diagnostics (effective sample size, CV of weights)
- Weight trimming at 1st/99th percentiles
- Weight stabilization (SW = P(T)/P(T|X))

**Testing**:
- Reuse PSM DGPs for consistency
- Add extreme weight scenarios
- Monte Carlo with same thresholds as PSM

### Session 7: Regression for Observational Data
**Build on tested `regression_ate`**:

**New components** (4-6 hours):
- Outcome models E[Y|X,T=1] and E[Y|X,T=0]
- G-computation for ATE
- Model diagnostics (residual plots, R²)
- Bootstrap for inference

**Testing**:
- Linear and nonlinear outcome models
- Model misspecification scenarios
- Monte Carlo validation

### Session 8: Doubly Robust Estimator
**Combine IPW + Regression**:

**Implementation** (6-8 hours):
- AIPW formula: combines IPW and regression
- Cross-fitting with 5-fold CV
- Influence function for variance

**Critical tests**:
- Correct propensity + wrong outcome → should work
- Wrong propensity + correct outcome → should work
- Both wrong → should fail
- Monte Carlo on all DGPs

---

## Phase 3: Advanced Methods (Sessions 9-12)

### Session 9: Matching Methods Comparison
- Optimal matching (solve assignment problem)
- Genetic matching (balance optimization)
- CEM (Coarsened Exact Matching)
- Compare to nearest-neighbor PSM

### Session 10: Machine Learning Methods
- Random forest for propensity/outcomes
- XGBoost/LightGBM integration
- Super Learner ensemble
- Targeted Learning (TMLE)

### Session 11: Sensitivity Analysis
- Rosenbaum bounds for hidden bias
- E-value for unmeasured confounding
- Simulation-based sensitivity
- Negative controls

### Session 12: Comparative Study
- All methods on same DGPs
- Bias-variance-computation tradeoffs
- Practical recommendations
- Final report with guidelines

---

## Phase 4: Production Tools (Sessions 13-15)

### Session 13: Diagnostic Dashboard
- Automated balance tables
- Love plots for SMD
- Propensity score distributions
- Common support visualization

### Session 14: Workflow Automation
- Config-driven estimator selection
- Automatic method comparison
- Report generation (LaTeX/HTML)
- Reproducibility tools

### Session 15: Package Release
- Documentation website
- Tutorial notebooks
- CI/CD pipeline
- PyPI release

---

## Key Principles (Learned from PSM)

1. **Test existing code first**: Don't leave 0% coverage code
2. **Three-layer testing**: Known-answer → Adversarial → Monte Carlo
3. **DGP design critical**: β_X affects bias dramatically
4. **Conservative is OK**: Better wide CIs than false precision
5. **Document limitations**: xfail tests preserve knowledge

---

## Success Metrics

### Per Session
- [ ] 80%+ test coverage
- [ ] Layer 1: 5+ known-answer tests
- [ ] Layer 2: 10+ adversarial tests
- [ ] Layer 3: 1000 Monte Carlo runs
- [ ] Documentation complete

### Overall
- [ ] All estimators validated
- [ ] Consistent API across methods
- [ ] Comparative analysis complete
- [ ] Production-ready package

---

## Timeline Estimate

- **Phase 1**: 1 week (Session 5)
- **Phase 2**: 2 weeks (Sessions 6-8)
- **Phase 3**: 3 weeks (Sessions 9-12)
- **Phase 4**: 2 weeks (Sessions 13-15)

**Total**: 8 weeks to production-ready package

---

## Immediate Next Steps

1. **Session 5**: Test existing RCT estimators
   - Start with `simple_ate` (baseline)
   - Verify unbiasedness in RCT setting
   - Build test fixtures for reuse

2. **Decision Point**: After Session 5
   - If RCT code solid → extend to observational
   - If RCT code problematic → rewrite with lessons learned
   - If time-constrained → focus on IPW + DR only

---

*Roadmap created: 2025-11-21*
*Based on: PSM implementation audit*
*Next session: RCT estimator validation*