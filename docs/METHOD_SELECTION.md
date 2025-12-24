# Method Selection Decision Tree

A systematic guide for choosing the appropriate causal inference method.

---

## Quick Decision Flowchart

```
START: Do you want to estimate a causal effect?
│
├─ Step 1: Is treatment RANDOMIZED?
│   ├─ YES ──────────────────────────────────► RCT Methods
│   │                                           (SimpleATE, Neyman, Regression-adjusted)
│   └─ NO ──► Continue to Step 2
│
├─ Step 2: Do you have a valid INSTRUMENT?
│   ├─ YES ──────────────────────────────────► IV Methods
│   │   └─ Is instrument weak (F < 10)?        (2SLS, LIML, Anderson-Rubin)
│   │       ├─ YES ──► WeakIVRobust, LIML
│   │       └─ NO ───► 2SLS
│   └─ NO ──► Continue to Step 3
│
├─ Step 3: Is there a DISCONTINUITY in treatment assignment?
│   ├─ YES ──────────────────────────────────► RDD Methods
│   │   └─ Is assignment sharp or fuzzy?       (Sharp, Fuzzy, Local Linear)
│   │       ├─ SHARP ──► SharpRDD
│   │       └─ FUZZY ──► FuzzyRDD (combines RDD + IV)
│   └─ NO ──► Continue to Step 4
│
├─ Step 4: Do you have PRE/POST data with treatment/control groups?
│   ├─ YES ──────────────────────────────────► DiD Methods
│   │   └─ Is adoption staggered?              (Classic, TWFE, CS, SA)
│   │       ├─ NO (single treatment time) ──► ClassicDiD, TWFE
│   │       └─ YES (staggered) ──► Callaway-Sant'Anna, Sun-Abraham
│   └─ NO ──► Continue to Step 5
│
└─ Step 5: Can you assume SELECTION ON OBSERVABLES?
    ├─ YES ──────────────────────────────────► Observational Methods
    │   └─ What's your concern?                (IPW, DR, PSM, Regression)
    │       ├─ Model misspecification ──► DoublyRobust
    │       ├─ Need exact matches ──► PSM
    │       └─ General use ──► IPW or Regression
    └─ NO ──► Continue to Step 6

├─ Step 6: Is there SAMPLE SELECTION (non-random attrition)?
│   ├─ YES ──────────────────────────────────► Selection Methods
│   │                                           (Heckman Two-Stage)
│   └─ NO ──► Continue to Step 7

├─ Step 7: Do identification assumptions FAIL but you want bounds?
│   ├─ YES ──────────────────────────────────► Bounds Methods
│   │   └─ What can you assume?                (Manski, Lee)
│   │       ├─ Monotone response ──► Manski Monotone
│   │       ├─ IV setting ──► Manski IV Bounds
│   │       └─ Selection on treatment ──► Lee Bounds
│   └─ NO ──► Continue to Step 8

├─ Step 8: Need DISTRIBUTIONAL effects (not just mean)?
│   ├─ YES ──────────────────────────────────► QTE Methods
│   │                                           (Quantile Treatment Effects)
│   └─ NO ──► Continue to Step 9

├─ Step 9: Need MARGINAL treatment effects across propensity?
│   ├─ YES ──────────────────────────────────► MTE Methods
│   │                                           (Local IV, Policy Relevant)
│   └─ NO ──► Continue to Step 10

├─ Step 10: Need to decompose DIRECT vs INDIRECT effects?
│   ├─ YES ──────────────────────────────────► Mediation Analysis
│   │                                           (Baron-Kenny, NDE/NIE)
│   └─ NO ──► Continue to Step 11

├─ Step 11: Endogeneity with KNOWN FUNCTIONAL FORM?
│   ├─ YES ──────────────────────────────────► Control Function
│   │                                           (Linear, Probit/Logit)
│   └─ NO ──► Continue to Step 12

└─ Step 12: SHIFT-SHARE identification strategy available?
    ├─ YES ──────────────────────────────────► Shift-Share IV
    │                                           (Bartik Instruments)
    └─ NO ──► Reconsider identification or data
```

---

## Detailed Method Selection

### 1. RCT Methods (Randomized Experiments)

**When to use**: Treatment was randomly assigned

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `SimpleATE` | Basic experiments | Difference in means |
| `NeymanATE` | Conservative inference | Conservative variance (no covariance) |
| `RegressionAdjusted` | Improve precision | Control for pre-treatment covariates |
| `StratifiedATE` | Blocked designs | Account for stratification |

**Assumptions**:
- SUTVA (no interference, single treatment version)
- Random assignment

**Implementation**: `src/causal_inference/rct/`

---

### 2. IV Methods (Instrumental Variables)

**When to use**: You have an instrument that affects treatment but not outcome directly

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `TwoStageLeastSquares` | Strong instruments (F > 10) | Standard IV estimator |
| `LIML` | Weak instruments | More robust to weak IV |
| `AndersonRubin` | Very weak instruments | Valid inference regardless of F |
| `WeakIVRobust` | Suspected weak IV | CLR/AR confidence sets |

**Assumptions**:
- Relevance: Instrument predicts treatment (testable: F > 10)
- Exclusion: Instrument affects outcome only through treatment (untestable)
- Exogeneity: Instrument uncorrelated with unobservables (untestable)
- Monotonicity: No defiers (for LATE interpretation)

**Diagnostics**:
- First-stage F-statistic
- Stock-Yogo weak IV test
- Sargan/Hansen over-identification test (if multiple instruments)

**Implementation**: `src/causal_inference/iv/`

---

### 3. RDD Methods (Regression Discontinuity)

**When to use**: Treatment assigned based on a cutoff in a running variable

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `SharpRDD` | Deterministic cutoff | Local linear regression at cutoff |
| `FuzzyRDD` | Probabilistic cutoff | Combines RDD + IV |
| `LocalLinear` | Standard use | Linear fit in bandwidth |
| `LocalPolynomial` | Curvature | Higher-order polynomials |

**Assumptions**:
- Continuity: Potential outcomes continuous at cutoff
- No manipulation: Units cannot precisely manipulate running variable

**Diagnostics**:
- McCrary density test (manipulation check)
- Covariate balance at cutoff
- Bandwidth sensitivity analysis

**Implementation**: `src/causal_inference/rdd/`

---

### 4. DiD Methods (Difference-in-Differences)

**When to use**: You have panel data with treatment/control groups and pre/post periods

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `ClassicDiD` | 2 periods, 2 groups | Simple before-after comparison |
| `TWFE` | Multiple periods/groups | Two-way fixed effects regression |
| `CallawayASantAnna` | Staggered adoption | Robust to heterogeneous effects |
| `SunAbraham` | Event studies | Interaction-weighted estimator |
| `EventStudy` | Dynamic effects | Effect by time relative to treatment |

**When NOT to use TWFE**:
- Staggered adoption + heterogeneous treatment effects → Use CS or SA
- See CONCERN-11 in `docs/METHODOLOGICAL_CONCERNS.md`

**Assumptions**:
- Parallel trends: Treatment/control would trend similarly absent treatment
- No anticipation: Units don't change behavior before treatment
- SUTVA: No spillovers

**Diagnostics**:
- Pre-trends test (event study with pre-periods)
- Placebo tests

**Implementation**: `src/causal_inference/did/`

---

### 5. Observational Methods

**When to use**: No randomization, instrument, discontinuity, or panel structure

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `IPW` | Reweighting | Creates pseudo-population |
| `DoublyRobust` | Robustness | Consistent if either model correct |
| `OutcomeRegression` | Simple cases | Covariate adjustment |
| `PSM` | Exact matching | Match treated to controls |

**Assumptions**:
- Unconfoundedness: All confounders observed (untestable)
- Positivity/Overlap: 0 < P(T=1|X) < 1 for all X

**Diagnostics**:
- Covariate balance after weighting/matching
- Propensity score overlap plots
- Sensitivity analysis (Rosenbaum bounds)

**Implementation**: `src/causal_inference/observational/`, `src/causal_inference/psm/`

---

### 6. Selection Methods (Heckman)

**When to use**: Sample selection/attrition is non-random and correlated with outcome

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `HeckmanTwoStage` | Selection on observables | Mills ratio correction |
| `HeckmanML` | Maximum likelihood | Joint estimation |

**Assumptions**:
- Exclusion restriction: At least one variable affects selection but not outcome
- Normality: Errors are jointly normal (for two-stage)

**Key Output**:
- Lambda (λ): Mills ratio coefficient - measures selection bias
- Rho (ρ): Correlation between selection and outcome errors

**Diagnostics**:
- Significance of λ indicates selection bias present
- Compare with OLS to assess bias magnitude

**Implementation**: `src/causal_inference/selection/`

---

### 7. Bounds Methods (Manski, Lee)

**When to use**: Standard identification assumptions fail, want to bound treatment effects

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `ManskiWorstCase` | No assumptions | Widest bounds |
| `ManskiMonotone` | Monotone treatment response | Tighter bounds |
| `ManskiIV` | IV with violations | Bounds under partial compliance |
| `ManskiMTS` | Monotone treatment selection | Selection-based bounds |
| `ManskiMTR` | Monotone treatment response | Response-based bounds |
| `LeeBounds` | Sample selection | Trimming-based bounds |

**Key Insight**:
- Bounds trade off assumptions for width
- More assumptions → tighter bounds
- Always valid (no point estimates, no false precision)

**Diagnostics**:
- Bound width indicates identification strength
- Check if bounds include zero (significance)

**Implementation**: `src/causal_inference/bounds/`

---

### 8. QTE Methods (Quantile Treatment Effects)

**When to use**: Care about distributional effects, not just mean

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `UnconditionalQTE` | Population quantiles | Marginal distribution effects |
| `ConditionalQTE` | Heterogeneity by X | Conditional quantile regression |
| `RIF_QTE` | Recentered influence function | Unconditional effects with covariates |

**Key Insight**:
- QTE(τ) ≠ TE at quantile τ of treated
- Unconditional QTE: Effect on population distribution
- Conditional QTE: Effect at quantile given X

**Assumptions**:
- Rank invariance (for unconditional): Treatment doesn't change rank
- Standard causal assumptions (unconfoundedness, etc.)

**Implementation**: `src/causal_inference/qte/`

---

### 9. MTE Methods (Marginal Treatment Effects)

**When to use**: IV setting, want to understand heterogeneity across propensity to treat

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `LocalIV` | MTE at propensity p | Local slope of outcome on propensity |
| `PolicyRelevant` | Policy evaluation | Weighted average of MTEs |
| `LATERecovery` | LATE decomposition | MTE integral over compliers |

**Key Insight**:
- MTE(p) = E[Y₁ - Y₀ | P(Z) = p]
- LATE = weighted average of MTEs
- Can evaluate new policies by reweighting

**Assumptions**:
- Valid IV (relevance, exclusion)
- Monotonicity
- Sufficient variation in propensity

**Implementation**: `src/causal_inference/mte/`

---

### 10. Mediation Analysis

**When to use**: Want to decompose effect into direct and indirect paths

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `BaronKenny` | Traditional approach | Product of coefficients |
| `NDE_NIE` | Causal mediation | Natural direct/indirect effects |
| `CDE` | Controlled effects | Effect at fixed mediator |
| `MediationSensitivity` | Robustness | Sequential ignorability tests |

**Key Insight**:
- Total Effect = Direct Effect + Indirect Effect
- NDE: Effect if mediator held at control value
- NIE: Effect through mediator pathway

**Assumptions**:
- Sequential ignorability: No unmeasured T→M or M→Y confounding
- No treatment-mediator interaction (for simple decomposition)

**Diagnostics**:
- Sensitivity analysis for unmeasured confounding
- Proportion mediated

**Implementation**: `src/causal_inference/mediation/`

---

### 11. Control Function Methods

**When to use**: Endogeneity with known functional form, alternative to IV

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `ControlFunctionLinear` | Linear models | First-stage residual as control |
| `ControlFunctionProbit` | Binary treatment | Probit first stage |
| `ControlFunctionLogit` | Binary treatment | Logit first stage |

**Key Insight**:
- Two-step: (1) Predict treatment, (2) Include residual in outcome
- More flexible than 2SLS for nonlinear models
- Can handle heterogeneous effects

**Assumptions**:
- Valid exclusion restriction
- Correct functional form for first stage
- Additive separability of unobservables

**Implementation**: `src/causal_inference/control_function/`

---

### 12. Shift-Share IV Methods (Bartik)

**When to use**: Industry/sector exposure varies across locations + aggregate shocks

| Method | Best For | Key Feature |
|--------|----------|-------------|
| `ShiftShareIV` | Labor economics | Bartik instrument construction |
| `RotembergDiagnostics` | Validity checks | Weight decomposition |

**Instrument Construction**:
- Z_i = Σ_s (share_{i,s} × shock_s)
- Shares: Local exposure to sectors
- Shocks: National/aggregate sector changes

**Assumptions** (choose one):
- Exogenous shares (Goldsmith-Pinkham et al. 2020)
- Exogenous shocks (Borusyak et al. 2022)

**Diagnostics**:
- Rotemberg weights: Which sectors drive the estimate
- Negative weights: Potential monotonicity violations
- First-stage F-statistic

**Implementation**: `src/causal_inference/shift_share/`

---

## Method Comparison Matrix

| Method Family | Estimand | Identification | Internal Validity | External Validity |
|---------------|----------|----------------|-------------------|-------------------|
| RCT | ATE | Randomization | High | Depends on sample |
| IV | LATE | Exclusion + Relevance | Medium-High | Compliers only |
| RDD | LATE at cutoff | Continuity | High near cutoff | Local to cutoff |
| DiD | ATT | Parallel trends | Medium | Treated units |
| Observational | ATE/ATT | Selection on observables | Low-Medium | Sample-dependent |
| Selection | ATE (corrected) | Exclusion + normality | Medium | Selected sample |
| Bounds | Interval | Varies by assumption | High (conservative) | Sample-dependent |
| QTE | QTE(τ) | Rank invariance | Medium | Quantile-specific |
| MTE | MTE(p) | IV + monotonicity | Medium-High | Policy-dependent |
| Mediation | NDE/NIE | Sequential ignorability | Low-Medium | Pathway-specific |
| Control Function | ATE | Exclusion + form | Medium | Model-dependent |
| Shift-Share | LATE | Exogenous shares/shocks | Medium | Sector-weighted |

---

## Sample Size Guidelines

| Method | Minimum n | Recommended n | Notes |
|--------|-----------|---------------|-------|
| SimpleATE | 30/group | 100/group | Larger for small effects |
| IPW | 100 | 500+ | Need overlap |
| DiD | 50/group | 200/group | More for clustering |
| IV | 100 | 500+ | More for weak IV |
| RDD | 100 near cutoff | 500+ | Depends on bandwidth |

---

## Quick Reference: By Scenario

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| A/B test | SimpleATE or NeymanATE | Randomization ensures identification |
| Policy change at threshold | SharpRDD | Natural experiment at cutoff |
| Natural experiment with instrument | 2SLS | Exploit exogenous variation |
| New policy rollout over time | ClassicDiD | Compare pre/post, treatment/control |
| Staggered state policy adoption | Callaway-Sant'Anna | Handles heterogeneity correctly |
| Observational data, rich covariates | DoublyRobust | Best robustness properties |
| Need exact comparison units | PSM | Interpretable matches |
| Concerned about weak instrument | LIML or AndersonRubin | Robust to weak IV |
| Survey with non-response | Heckman Selection | Corrects for selection bias |
| Assumptions questionable | Manski/Lee Bounds | Honest about uncertainty |
| Care about inequality effects | QTE | Effects across distribution |
| Want to evaluate new policies | MTE | Policy-relevant treatment effects |
| Treatment works through mediator | Mediation Analysis | Decompose direct/indirect |
| Endogeneity, non-linear model | Control Function | More flexible than 2SLS |
| Labor/trade with sector exposure | Shift-Share IV | Bartik instrument |

---

## See Also

- `docs/TROUBLESHOOTING.md` - When methods fail
- `docs/FAILURE_MODES.md` - Method-specific failure patterns
- `docs/METHODOLOGICAL_CONCERNS.md` - Known issues and solutions
- `docs/QUICK_REFERENCE.md` - Copy/paste commands
- Query research-kb: `research_kb_get_concept "{method_name}"`

---

*Last updated: 2025-12-24 (Session 98 - 21 method families)*
