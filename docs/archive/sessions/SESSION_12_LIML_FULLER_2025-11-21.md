# Session 12: LIML & Fuller Estimators

**Date**: 2025-11-21
**Duration**: ~5 hours
**Status**: ✅ COMPLETE
**Phase**: 4 (Instrumental Variables)

---

## Overview

Implemented LIML (Limited Information Maximum Likelihood) and Fuller-k estimators, which are less biased than 2SLS with weak instruments.

---

## Implementation Summary

### LIML Estimator (`src/causal_inference/iv/liml.py`)
- k-class estimator with k = λ_min (smallest eigenvalue)
- Less biased than 2SLS when instruments are weak
- Approximately median-unbiased
- Higher variance than 2SLS but lower bias

### Fuller Estimator (`src/causal_inference/iv/fuller.py`)
- Fuller-1 and Fuller-4 variants
- k = λ_min - α/(n - K) where α ∈ {1, 4}
- Fuller-1: More aggressive bias reduction
- Fuller-4: Balance between bias and variance
- Recommended by Angrist & Pischke for weak IV

---

## Test Results

**Total Tests**: 35
**Passing**: 35
**Pass Rate**: 100%

**Test Categories**:
- LIML: 17 tests (correctness, weak IV performance, comparison to 2SLS)
- Fuller: 18 tests (Fuller-1, Fuller-4, weak IV scenarios)

---

## Files Created

**Implementation** (~640 lines total):
- `src/causal_inference/iv/liml.py` (320 lines)
- `src/causal_inference/iv/fuller.py` (320 lines)

**Tests**:
- `tests/test_iv/test_liml.py` (17 tests)
- `tests/test_iv/test_fuller.py` (18 tests)

---

## Key Features

1. **Weak IV Performance**: LIML and Fuller perform better than 2SLS with weak instruments
2. **Automatic Variant Selection**: Fuller-4 recommended as default
3. **k-Class Framework**: Unified interface for 2SLS (k=1), LIML (k=λ_min), Fuller (k=λ_min-α/(n-K))

---

## Validation

- Bias reduction validated with weak IV (F=5-10)
- LIML approximately median-unbiased
- Fuller-4 balances bias-variance tradeoff

---

## Session Statistics

- **Duration**: ~5 hours
- **Lines Added**: ~640 (implementation) + ~400 (tests)
- **Tests Created**: 35
- **Tests Passing**: 35 (100%)
- **Files Changed**: 4
