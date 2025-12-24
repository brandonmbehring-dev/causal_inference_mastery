# Sessions 90-91: Marginal Treatment Effects (MTE)

**Date**: 2025-12-20
**Status**: ✅ COMPLETE

---

## Overview

Implemented full MTE module in both Python and Julia with 171 total tests.

## Test Summary

| Language | Tests |
|----------|-------|
| Python | 93 |
| Julia | 63 |
| Cross-language | 15 |
| **Total** | **171** |

## Key Components

- `local_iv`: Local IV estimation along propensity score
- `late`: LATE decomposition from MTE
- `policy`: Policy-relevant treatment effects (PRTE)
- `diagnostics`: MTE diagnostics

## References

- Heckman & Vytlacil (2005). Structural Equations, Treatment Effects
