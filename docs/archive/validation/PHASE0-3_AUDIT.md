# Phase 0–3 Audit (causal_inference_mastery)

## Scope & Sources
- **Phase mapping used**: Phase 0 = statistical correctness + validation hardening (Session 19–21 fixes); Phase 1 = RCT foundation; Phase 2 = Observational methods (IPW, DR, PSM); Phase 3 = Difference-in-Differences (classic, event study, modern staggered). This aligns with `docs/ROADMAP.md` and the latest status in `CURRENT_WORK.md`.
- **Evidence reviewed**: `CURRENT_WORK.md`, `docs/ROADMAP.md`, `docs/ROADMAP_REFINED_2025-11-23.md`, session summaries (`docs/SESSION_4_RCT_*.md`, `docs/SESSION_5_IPW_*.md`, `docs/SESSION_6_DOUBLY_ROBUST_*.md`, `docs/SESSION_7_PSM_MONTE_CARLO_*.md`, `docs/SESSION_8_DID_*.md`, `docs/SESSION_9_EVENT_STUDY_*.md`, `docs/SESSION_10_MODERN_DID_*.md`, `docs/SESSION_18_STAGGERED_DID_BOOTSTRAP_FIX_2025-11-23.md`), validation docs (`docs/PYTHON_VALIDATION_ARCHITECTURE.md`, `docs/QUICK_FIXES_COMPLETE_2025-11-20.md`), and code/tests under `src/causal_inference/` and `tests/`.

## Status Snapshot
| Phase | Scope | Evidence (code/tests/docs) | Current state | Open risks |
|-------|-------|---------------------------|---------------|------------|
| 0 | Statistical correctness + validation infrastructure | t-distribution + permutation smoothing fixes, propensity clipping, wild bootstrap wiring (`tests/test_did/test_wild_bootstrap.py`, `docs/QUICK_FIXES_COMPLETE_2025-11-20.md`, `CURRENT_WORK.md`) | ✅ Applied; RCT coverage >95% (Quick Fixes) | Cross-language coverage thin (only `tests/validation/cross_language/test_python_julia_simple_ate.py`); IPW weight instability historically brittle but now warned/clipped |
| 1 | RCT estimators (simple, regression-adjusted, stratified, IPW, permutation) | `src/causal_inference/rct/`, 68 unit tests (`rg -c "def test_" tests/test_rct`), Monte Carlo in `tests/validation/monte_carlo/test_monte_carlo_{simple,regression_ate,stratified_ate,ipw_ate}.py`, docs `docs/SESSION_4_RCT_2025-11-21.md` | ✅ Complete; 3-layer validation present; coverage ~95% reported; Monte Carlo bias/coverage within targets | Cross-language parity only for simple_ate; permutation test still small-sample sensitive by design |
| 2 | Observational (propensity, IPW, DR) + PSM | Implementations in `src/causal_inference/observational/` and `src/causal_inference/psm/`; 81 IPW/DR unit tests + 5 PSM unit tests; Monte Carlo: IPW/DR (`tests/validation/monte_carlo/test_monte_carlo_ipw_*`, `test_monte_carlo_doubly_robust.py`), PSM (`test_monte_carlo_psm.py`); docs `docs/SESSION_5_IPW_2025-11-21.md`, `docs/SESSION_6_DOUBLY_ROBUST_2025-11-21.md`, `docs/SESSION_7_PSM_MONTE_CARLO_*.md` | ✅ Complete; double-robustness empirically validated; PSM Monte Carlo 4/5 passing with 1 expected xfail (limited overlap) | PSM known limitation with extreme lack of overlap; observational cross-language parity not implemented |
| 3 | Difference-in-Differences (classic, event study, modern CS/SA, TWFE) | Code in `src/causal_inference/did/`; 100 unit/integration tests (`rg -c "def test_" tests/test_did`); Monte Carlo 37 tests across TWFE/CS/SA/2x2/event study (`tests/validation/monte_carlo/test_monte_carlo_did_*`, `test_monte_carlo_event_study.py`); docs `docs/SESSION_8_DID_*`, `docs/SESSION_9_EVENT_STUDY_*`, `docs/SESSION_10_MODERN_DID_*`, `docs/SESSION_18_STAGGERED_DID_BOOTSTRAP_FIX_2025-11-23.md` | ✅ Complete; Session 20/18 resolved staggered tolerance issues; wild cluster bootstrap added (`tests/test_did/test_wild_bootstrap.py`); Monte Carlo targets met (Phase 2 MC in `CURRENT_WORK.md`) | Classic/Event PyCall parity still pending; cluster-SE small-cluster caveats documented (CONCERN-13); bootstrap variability requires tolerant thresholds |

## Phase Details

### Phase 0 — Statistical Correctness & Validation Baseline
- Fixed critical inference issues (z→t, permutation smoothing, stratified n=1 variance) documented in `docs/AUDIT_RECONCILIATION_2025-11-20.md` and addressed in `docs/QUICK_FIXES_COMPLETE_2025-11-20.md`.
- Validation layer build-out captured in `docs/PYTHON_VALIDATION_ARCHITECTURE.md`; adversarial suites live under `tests/validation/adversarial/` and Monte Carlo under `tests/validation/monte_carlo/`.
- Additional safety rails: propensity clipping + perfect-separation warnings (`tests/observational/test_propensity_clipping.py`), wild cluster bootstrap harness (`tests/test_did/test_wild_bootstrap.py`).
- Residual gaps: limited PyCall coverage on Python side; IPW instability mitigated but still documented as methodological concern (see CONCERN-5).

### Phase 1 — RCT Foundation
- Implementations: `src/causal_inference/rct/estimators*.py` covering simple, regression-adjusted, stratified, IPW, permutation.
- Tests: 68 functional/adversarial unit tests in `tests/test_rct/` plus Monte Carlo suites for simple/regression/stratified/IPW. Coverage reported at 95%+ in `docs/QUICK_FIXES_COMPLETE_2025-11-20.md`.
- Validation outcomes: Bias <0.05 and coverage 93–97% on RCT DGPs (per `docs/ROADMAP.md`); permutation small-sample thresholds relaxed to respect discreteness.
- Outstanding: Only simple_ate has Python→Julia parity test; consider extending cross-language coverage to regression/stratified/IPW to harden layer 4.

### Phase 2 — Observational Methods (IPW, DR, PSM)
- IPW/DR: Propensity estimation, trimming, stabilization, and double-robust estimator in `src/causal_inference/observational/`; 81 tests across propensity, IPW, DR, small-sample inference, perfect-separation handling.
- PSM: Matching/variance/balance stack in `src/causal_inference/psm/`; 5 unit tests in `tests/test_psm/` plus 5 Monte Carlo scenarios (`tests/validation/monte_carlo/test_monte_carlo_psm.py`). Abadie-Imbens variance intentionally conservative; limited-overlap scenario marked xfail.
- Empirical validation: Monte Carlo suites for observational estimators hit bias/coverage/SE targets with relaxed thresholds for confounded settings (see `docs/SESSION_7_PSM_MONTE_CARLO_*.md`).
- Risks: Residual bias for PSM under strong confounding; no cross-language parity for observational estimators; variance remains conservative (expected).

### Phase 3 — Difference-in-Differences
- Implementations: Classic 2×2 (`src/causal_inference/did/did_estimator.py`), event study (`src/causal_inference/did/event_study.py`), modern staggered CS/SA/TWFE (`src/causal_inference/did/{callaway_santanna.py,sun_abraham.py,staggered.py,comparison.py}`).
- Tests: 100 deterministic tests (`rg -c "def test_" tests/test_did`) covering known-answer, adversarial, staggered edge cases, and the new wild bootstrap harness. Monte Carlo layer includes 37 tests across TWFE/CS/SA/2×2/event-study files noted above.
- Recent fixes: Session 18/20 resolved staggered tolerance issues and bootstrap sample bugs; DiD suite reported fully passing in `CURRENT_WORK.md`.
- Gaps & cautions: PyCall parity exists for staggered DiD only; classic/event validation against Julia still pending. Cluster SEs with few clusters and pre-trends testing limitations tracked in `docs/METHODOLOGICAL_CONCERNS.md` (CONCERN-12/13).

## Cross-Phase Observations & Suggested Next Actions
- Validation breadth is strong (adversarial + Monte Carlo) but Python→Julia parity lags outside staggered DiD and simple_ate. Extending parity to RCT variants and classic/event DiD would close layer-4 gaps.
- PSM limitation (xfail limited-overlap case) remains the only deliberate failure across phases 0–3; keep documented thresholds and warn users in API docs.
- Monte Carlo suites are extensive (20+ files); consider marking slow tests and maintaining seeds to avoid flakiness as further phases add runtime.
- If prioritizing phase 3 hardening: add Julia parity tests for classic/event study, and document bootstrap variability expectations in `docs/SESSION_10_MODERN_DID_2025-11-21.md` or a new addendum.
