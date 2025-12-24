# Sessions 85-89: Selection, QTE, and Bounds

**Date**: 2025-12-20
**Status**: ✅ COMPLETE

---

## Overview

Implemented Heckman Selection, Quantile Treatment Effects, and Partial Identification Bounds.

## Session 85: Heckman Selection

**Files**: `src/causal_inference/selection/`
- `heckman.py`: Two-stage selection model
- `types.py`: Result types
- `diagnostics.py`: Selection diagnostics

**Key Features**:
- Probit selection equation
- Inverse Mills ratio construction
- Outcome equation with selection correction
- Tests for selection (ρ significance)

**References**: Heckman (1979), Wooldridge (2010) Ch. 19

## Sessions 86-88: Bounds (Manski + Lee)

**Files**: `src/causal_inference/bounds/`
- `manski.py`: Worst-case, MTR, MTS, MTR+MTS, IV bounds
- `lee.py`: Lee (2009) attrition bounds
- `types.py`: Bounds result types

**Key Features**:
- No-assumptions bounds
- Monotone treatment response (MTR)
- Monotone treatment selection (MTS)
- Combined MTR+MTS bounds
- IV bounds
- Lee trimming procedure

**References**: Manski (1990, 2003), Lee (2009)

## Sessions 88-89: Quantile Treatment Effects

**Files**: `src/causal_inference/qte/`
- `unconditional.py`: Unconditional QTE
- `conditional.py`: Conditional QTE (quantile regression)
- `rif.py`: RIF-regression (Firpo, Fortin, Lemieux)
- `types.py`: QTE result types

**Key Features**:
- Unconditional QTE at specified quantiles
- Conditional QTE with covariates
- RIF-regression for unconditional effects
- Bootstrap standard errors

**References**: Koenker & Bassett (1978), Firpo, Fortin, Lemieux (2009)

## Total Tests

~300 tests across all sessions
