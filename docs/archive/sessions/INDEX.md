# Session Archive Index

**Last Updated**: 2025-12-23

Quick lookup for causal_inference_mastery session history.

---

## Quick Lookup by Method

| Method | Sessions | Key Files |
|--------|----------|-----------|
| **Phase 12-13: Advanced Methods** | | |
| Heckman Selection | 85 | `selection/` |
| Manski Bounds | 86-88 | `bounds/manski.py` |
| Lee Bounds | 87-88 | `bounds/lee.py` |
| QTE | 88-89 | `qte/` |
| MTE | 90-91 | `mte/` |
| Mediation | 92 | `mediation/` |
| Control Function | 93 | `control_function/` |
| Shift-Share | 94 | `shift_share/` |
| Julia Cross-Language | 95 | `julia_interface.py` |
| **Phase 11: RKD + Bunching** | | |
| Python RKD | 72-73 | `rkd/` |
| Julia RKD | 74-75 | `julia/src/rkd/` |
| Python Bunching | 76-77 | `bunching/` |
| Julia Bunching | 78, 81 | `julia/src/bunching/` |
| **Phase 10: Validation** | | |
| Julia Observational MC | 80 | `julia/test/observational/` |
| Context Engineering | 82 | `.claude/skills/`, MCP |
| Comprehensive Audit | 83 | `docs/AUDIT_RESULTS.md` |
| **Phase 9: CATE + Sensitivity** | | |
| CATE Meta-Learners | 42-46 | `cate/` |
| SCM | 46-47 | `scm/` |
| Sensitivity Analysis | 48-50 | `sensitivity/` |
| **Phase 5-8: Core Methods** | | |
| RCT | 4-6 | `rct/` |
| PSM | 7 | `psm/` |
| DiD | 8-10 | `did/` |
| IV | 11-13 | `iv/` |
| RDD | 14-16 | `rdd/` |

---

## Phase Summary

| Phase | Sessions | Focus | Status |
|-------|----------|-------|--------|
| 1-4 | 1-19 | RCT, PSM, DiD foundation | ✅ Archived |
| 5-8 | 20-41 | IV, RDD, SCM | ⚠️ Missing |
| 9-11 | 42-62 | CATE, Sensitivity, RKD, Bunching | ✅ Archived |
| 12-13 | 63-95 | Context, Audit, Advanced Methods | ✅ Archived |

---

## Session Quick Reference

### Sessions 85-95 (Phase 12-13: Advanced Methods)

| Session | Date | Focus | Tests |
|---------|------|-------|-------|
| 95 | 2025-12-20 | Julia Cross-Language Parity | 180 |
| 94 | 2025-12-20 | Shift-Share IV (Python) | 32 |
| 93 | 2025-12-20 | Control Function (Python) | 102 |
| 92 | 2025-12-20 | Mediation Analysis (Python) | 100 |
| 90-91 | 2025-12-20 | MTE (Python + Julia) | 171 |
| 85-89 | 2025-12-20 | Selection, QTE, Bounds | ~300 |

### Sessions 63-84 (Phase 12-13: Context + Audit)

| Session | Date | Focus | Tests |
|---------|------|-------|-------|
| 82 | 2025-12-19 | Context Engineering | 10 |
| 80-81 | 2025-12-19 | Julia Observational/Bunching MC | ~100 |
| 78 | 2025-12-19 | Julia Bunching + Cross-Lang | 124 |
| 76-77 | 2025-12-19 | Python Bunching | 104 |
| 74-75 | 2025-12-18 | Julia RKD | 302 |
| 72-73 | 2025-12-18 | Python RKD | ~150 |
| 63-71 | 2025-12-17 | RKD foundation | ~200 |

### Sessions 42-62 (Phase 9-11: CATE, Sensitivity)

| Session | Date | Focus | Tests |
|---------|------|-------|-------|
| 62 | 2025-12-17 | CATE Monte Carlo | 50 |
| 55-61 | 2025-12-17 | IV stages, VCov, McCrary | ~100 |
| 46-54 | 2025-12-16 | SCM, Sensitivity | ~150 |
| 42-45 | 2025-12-16 | CATE Meta-Learners | ~100 |

### Sessions 1-19 (Phase 1-4: Foundation)

See individual files in `docs/archive/sessions/SESSION_*.md`

---

## Total Statistics

| Metric | Value |
|--------|-------|
| Total Sessions | 95+ |
| Python Lines | ~27,000 |
| Julia Lines | ~25,000 |
| Total Tests | ~8,500 |
| Method Families | 21 |
| Pass Rate | 99%+ |

---

## File Naming Convention

- `SESSION_{N}.md` - Individual session archive
- `SESSION_{N}_{TOPIC}_{DATE}.md` - Legacy format (Sessions 1-19)

---

*For current work, see `CURRENT_WORK.md` in project root.*
