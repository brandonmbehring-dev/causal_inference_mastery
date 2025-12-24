# Quick Reference

Copy/paste-ready commands and quick lookups for causal_inference_mastery.

---

## Method Selection (21 Families)

| Your Situation | Method |
|----------------|--------|
| Randomized experiment | RCT (simple, stratified, regression-adjusted) |
| Observational with confounders | IPW, Doubly Robust, PSM |
| Before/after comparison | Difference-in-Differences (DiD) |
| Endogenous variable | Instrumental Variables (IV) |
| Threshold/cutoff | Regression Discontinuity (RDD) |
| Single treated unit | Synthetic Control (SCM) |
| Treatment effect heterogeneity | CATE (S/T/X/R-learners, Causal Forest) |
| Sensitivity to confounding | E-values, Rosenbaum Bounds |
| Kink in policy | Regression Kink Design (RKD) |
| Bunching at threshold | Bunching Estimators |
| Sample selection/attrition | Heckman Selection |
| Assumptions violated | Manski/Lee Bounds |
| Distributional effects | Quantile Treatment Effects (QTE) |
| Marginal effects | MTE (Local IV, Policy Relevant) |
| Direct vs indirect effects | Mediation Analysis |
| Endogeneity, nonlinear | Control Function |
| Sector exposure + shocks | Shift-Share IV (Bartik) |

---

## Test Commands

### Python (run from repo root)

```bash
# Full test suite with coverage
pytest tests/ --cov=src/causal_inference --cov-report=term-missing --cov-fail-under=90

# Fast tests (skip slow/Monte Carlo)
pytest tests/ -m "not slow"

# Single module
pytest tests/test_rct/ -v
pytest tests/test_did/ -v
pytest tests/observational/ -v

# Specific test file
pytest tests/test_rct/test_simple_ate.py -v

# Cross-language validation
pytest tests/validation/cross_language/ -v
```

### Julia (run from julia/ directory)

```bash
# Full test suite
cd julia && julia --project test/runtests.jl

# Specific test file
cd julia && julia --project -e 'include("test/did/test_classic_did.jl")'

# With verbose output
cd julia && julia --project test/runtests.jl --verbose
```

### Code Quality (run from repo root)

```bash
# Format Python
black src/ tests/

# Lint Python
ruff src/ tests/

# Type check
mypy src/

# All pre-commit hooks
pre-commit run --all-files
```

---

## Validation Checklist (Core)

- [ ] Layer 1: Known-answer tests pass
- [ ] Layer 2: Adversarial tests pass
- [ ] Layer 3: Monte Carlo validation
- [ ] Layer 4: Cross-language parity

See [patterns/validation.md](patterns/validation.md) for Layers 5-6 (R Triangulation, Golden Reference).

---

## Monte Carlo Targets

| Method | Bias | Coverage | SE Accuracy |
|--------|------|----------|-------------|
| RCT | < 0.05 | 93-97% | < 10% |
| IPW/DR | < 0.10 | 93-97.5% | < 15% |
| PSM | < 0.30 | >= 95% | < 150% |
| DiD | < 0.10 | 93-97% | < 15% |
| IV | < 0.15 | 90-97% | < 20% |
| RDD | < 0.15 | 90-97% | < 20% |

---

## Diagnostic Commands (New Methods)

```python
# Heckman Selection - Mills ratio
from causal_inference.selection import heckman_two_stage
result = heckman_two_stage(Y, X, Z)
print(f"Lambda (selection): {result['lambda']:.3f} (p={result['lambda_pvalue']:.3f})")

# Manski Bounds - Width indicates identification strength
from causal_inference.bounds import manski_bounds
result = manski_bounds(Y, T)
print(f"Bounds: [{result['lower']:.3f}, {result['upper']:.3f}]")

# QTE - Effects across distribution
from causal_inference.qte import unconditional_qte
result = unconditional_qte(Y, T, quantiles=[0.25, 0.5, 0.75])
for q, est in zip(result['quantiles'], result['estimates']):
    print(f"QTE({q:.0%}): {est:.3f}")

# MTE - Marginal effects curve
from causal_inference.mte import marginal_treatment_effect
result = marginal_treatment_effect(Y, T, Z, X)
print(f"MTE at p=0.5: {result['mte_at_median']:.3f}")

# Mediation - Direct/Indirect decomposition
from causal_inference.mediation import mediation_analysis
result = mediation_analysis(Y, T, M, X)
print(f"NDE: {result['nde']:.3f}, NIE: {result['nie']:.3f}")
print(f"% Mediated: {result['proportion_mediated']:.1%}")

# Control Function - First-stage residual
from causal_inference.control_function import control_function
result = control_function(Y, T, Z, X)
print(f"CF Estimate: {result['estimate']:.3f}")

# Shift-Share - Rotemberg diagnostics
from causal_inference.shift_share import shift_share_iv
result = shift_share_iv(Y, T, shares, shocks)
print(f"Estimate: {result['estimate']:.3f}")
print(f"Negative weight share: {result['rotemberg']['negative_weight_share']:.1%}")
print(f"Top sector: {result['rotemberg']['top_5_sectors'][0]}")
```

---

## Interview Quick Answers

| Question | Answer |
|----------|--------|
| When would 2SLS fail? | Weak instruments (F < 10) |
| Why not just regression? | Selection bias from unobserved confounders |
| DiD vs RDD? | Time variation vs cutoff |
| When use bounds? | When standard assumptions fail |
| What does Heckman correct? | Sample selection bias (non-random attrition) |
| What are Rotemberg weights? | Which sectors drive shift-share IV estimate |
| QTE vs ATE? | Distributional effects vs mean effect |
| NDE vs NIE? | Direct effect vs effect through mediator |
| Control function vs 2SLS? | CF handles nonlinear models, heterogeneity |

---

## Key Files

| File | Purpose |
|------|---------|
| `CURRENT_WORK.md` | Session context (start here) |
| `CLAUDE.md` | AI assistant guidance |
| `pyproject.toml` | Python config |
| `julia/Project.toml` | Julia config |
| `tests/conftest.py` | Shared fixtures |
| `tests/golden_results/` | Reference JSON |

---

## Git Commit Format

```
type(scope): Short description

- Detail 1
- Detail 2

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `validate`

---

*Last updated: 2025-12-24 (Session 98 - 21 method families)*
