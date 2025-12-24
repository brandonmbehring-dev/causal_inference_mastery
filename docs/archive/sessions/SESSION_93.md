# Session 93: Control Function (Python)

**Date**: 2025-12-20
**Status**: ✅ COMPLETE

---

## Overview

Implemented full control function module with 102 tests passing.

## Files Created (~3,600 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `src/causal_inference/control_function/__init__.py` | 67 | Public API |
| `src/causal_inference/control_function/types.py` | 205 | TypedDicts |
| `src/causal_inference/control_function/control_function.py` | 686 | Linear CF estimator |
| `src/causal_inference/control_function/nonlinear.py` | 515 | Probit/Logit CF |
| `tests/test_control_function/` | ~1,500 | 102 tests |

## Key Features

1. **Linear CF**: Y = β₀ + β₁D + ρν̂ + u where ν̂ = first-stage residuals
2. **Murphy-Topel SE**: Corrected SEs for two-step estimation uncertainty
3. **Bootstrap Inference**: Paired bootstrap re-estimating both stages
4. **Endogeneity Test**: T-test on control coefficient (H0: ρ = 0)
5. **2SLS Equivalence**: Numerically matches 2SLS in linear models
6. **Nonlinear Extension**: Probit/Logit CF for binary outcomes

## Test Categories (102 tests)

| Category | Tests |
|----------|-------|
| CF-2SLS equivalence | 3 |
| Treatment effect recovery | 3 |
| Endogeneity detection | 4 |
| Standard errors | 4 |
| First-stage diagnostics | 5 |
| Adversarial | 31 |
| Monte Carlo | 15 |
| Nonlinear CF | 26 |
| Metadata/summary | 11 |

## Mathematical Foundation

**Why Control Function works:**
1. First stage: D = π₀ + π₁Z + ν
2. Second stage: Y = β₀ + β₁D + ρν̂ + u
3. ρ captures Cov(D, ε)/Var(ν)
4. If ρ = 0, no endogeneity → OLS consistent

**Why 2SLS fails for nonlinear:**
- Jensen's inequality: E[Φ(β*D̂)] ≠ Φ(β*E[D̂])
- Control function includes residuals directly, avoiding this
