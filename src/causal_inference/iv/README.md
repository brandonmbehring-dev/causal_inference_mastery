# Instrumental Variables (IV) Module

Production-ready instrumental variables estimation for handling endogeneity bias in causal inference.

## Quick Start

```python
from causal_inference.iv import TwoStageLeastSquares
import numpy as np

# Generate example data
np.random.seed(42)
n = 1000
Z = np.random.normal(0, 1, n)  # Instrument
D = 2 * Z + np.random.normal(0, 1, n)  # Endogenous treatment
Y = 0.5 * D + np.random.normal(0, 1, n)  # Outcome

# Fit 2SLS with robust standard errors
iv = TwoStageLeastSquares(inference='robust')
iv.fit(Y, D, Z)

# View results
print(iv.summary())
print(f"Treatment effect: {iv.coef_[0]:.3f} (SE: {iv.se_[0]:.3f})")
print(f"First-stage F-statistic: {iv.first_stage_f_stat_:.2f}")
```

## Features

### Core Estimators

- **TwoStageLeastSquares**: Production 2SLS with correct standard errors
  - Standard (homoskedastic)
  - Robust (heteroskedasticity-robust)
  - Clustered (multi-way clustering)

- **LIML**: Limited Information Maximum Likelihood
  - Less biased than 2SLS with weak instruments
  - Higher variance in small samples
  - Recommended when F < 10

- **Fuller**: Modified LIML with bias correction
  - Fuller-1 (α=1): Most commonly recommended
  - Fuller-4 (α=4): More conservative
  - Better finite-sample properties than LIML

- **GMM**: Generalized Method of Moments
  - One-step GMM (equivalent to 2SLS)
  - Two-step efficient GMM with optimal weighting
  - Hansen J-test for overidentifying restrictions
  - Optimal for many instruments (q >> p)

### Educational Components

- **FirstStage**: Examine first-stage strength (F-stat, partial R²)
- **ReducedForm**: Direct effect of instruments on outcome
- **SecondStage**: Structural equation (naive SEs - use 2SLS for inference)

### Weak Instrument Diagnostics

- **classify_instrument_strength()**: Stock-Yogo classification
- **cragg_donald_statistic()**: Multivariate weak IV test
- **anderson_rubin_test()**: Weak-IV-robust confidence intervals
- **weak_instrument_summary()**: Comprehensive diagnostic table

## Common Use Cases

### 1. Basic IV Estimation

```python
from causal_inference.iv import TwoStageLeastSquares

# Just-identified: 1 instrument, 1 endogenous variable
iv = TwoStageLeastSquares(inference='robust')
iv.fit(Y, D, Z)

# Check instrument strength
if iv.first_stage_f_stat_ > 20:
    print("✓ Strong instrument (F > 20)")
    print(f"Treatment effect: {iv.coef_[0]:.3f}")
else:
    print("⚠ Weak instrument - consider alternatives")
```

### 2. IV with Exogenous Controls

```python
# With control variables X
iv = TwoStageLeastSquares(inference='robust')
iv.fit(Y, D, Z, X)

# View summary table
summary = iv.summary()
print(summary)
```

### 3. Over-Identified Models (Multiple Instruments)

```python
# Two instruments for one endogenous variable
Z = np.column_stack([Z1, Z2])

iv = TwoStageLeastSquares(inference='robust')
iv.fit(Y, D, Z, X)

# Check overidentification
print(f"Number of instruments: {iv.n_instruments_}")
print(f"Number of endogenous: {iv.n_endogenous_}")
print(f"Overidentified: {iv.n_instruments_ > iv.n_endogenous_}")
```

### 4. Weak Instrument Diagnostics

```python
from causal_inference.iv import (
    classify_instrument_strength,
    anderson_rubin_test,
    weak_instrument_summary
)

# Fit 2SLS
iv = TwoStageLeastSquares(inference='robust')
iv.fit(Y, D, Z)

# Classify instrument strength
classification, critical_value, interpretation = classify_instrument_strength(
    f_statistic=iv.first_stage_f_stat_,
    n_instruments=1,
    n_endogenous=1
)

print(f"Instrument classification: {classification}")
print(interpretation)

# Get Anderson-Rubin confidence interval (robust to weak instruments)
ar_stat, p_value, (ci_lower, ci_upper) = anderson_rubin_test(Y, D, Z)
print(f"AR 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Generate comprehensive diagnostic summary
summary = weak_instrument_summary(
    f_statistic=iv.first_stage_f_stat_,
    n_instruments=1,
    n_endogenous=1,
    ar_ci=(ci_lower, ci_upper)
)
print(summary)
```

### 5. Three-Stage Decomposition (Educational)

```python
from causal_inference.iv import FirstStage, ReducedForm, SecondStage

# First stage: D = π₀ + π₁Z + π₂X + ν
first = FirstStage()
first.fit(D, Z, X)
print(f"First-stage F-statistic: {first.f_statistic_:.2f}")
print(f"Partial R²: {first.partial_r2_:.3f}")

# Reduced form: Y = γ₀ + γ₁Z + γ₂X + u
reduced = ReducedForm()
reduced.fit(Y, Z, X)

# Second stage: Y = β₀ + β₁D̂ + β₂X + ε
second = SecondStage()
second.fit(Y, first.fitted_values_, X)

# Verify Wald identity: γ = π × β
pi = first.coef_[0]  # Effect of Z on D
gamma = reduced.coef_[0]  # Effect of Z on Y
beta = second.coef_[0]  # Effect of D on Y
print(f"Wald identity check: {gamma:.4f} ≈ {pi * beta:.4f}")

# WARNING: Second-stage SEs are NAIVE - use TwoStageLeastSquares for correct SEs
```

### 6. Clustered Standard Errors

```python
# Multi-way clustering (e.g., by state and year)
cluster_var = state_year_pairs  # shape (n, 2)

iv = TwoStageLeastSquares(inference='clustered', cluster_var=cluster_var)
iv.fit(Y, D, Z, X)

# Clustered SEs are larger than robust SEs
print(f"Robust SE: {robust_se:.4f}")
print(f"Clustered SE: {iv.se_[0]:.4f}")
```

### 7. LIML with Weak Instruments

```python
from causal_inference.iv import LIML

# When instruments are weak (F < 10), use LIML
# LIML is less biased than 2SLS but has higher variance

liml = LIML(inference='robust')
liml.fit(Y, D, Z, X)

print(f"LIML estimate: {liml.coef_[0]:.3f}")
print(f"LIML kappa: {liml.kappa_:.3f}")  # LIML k-class parameter

# Compare with 2SLS
tsls = TwoStageLeastSquares(inference='robust')
tsls.fit(Y, D, Z, X)

print(f"2SLS estimate: {tsls.coef_[0]:.3f}")
print(f"Difference: {abs(liml.coef_[0] - tsls.coef_[0]):.3f}")
```

### 8. Fuller Estimator (Recommended for Weak IV)

```python
from causal_inference.iv import Fuller

# Fuller-1 is often recommended over LIML
# Better finite-sample properties (lower MSE)

fuller = Fuller(alpha_param=1.0, inference='robust')  # Fuller-1
fuller.fit(Y, D, Z, X)

print(f"Fuller-1 estimate: {fuller.coef_[0]:.3f}")
print(f"Fuller kappa: {fuller.kappa_:.3f}")
print(f"LIML kappa (unadjusted): {fuller.kappa_liml_:.3f}")

# Fuller-4 is more conservative
fuller4 = Fuller(alpha_param=4.0, inference='robust')
fuller4.fit(Y, D, Z, X)
print(f"Fuller-4 estimate: {fuller4.coef_[0]:.3f}")
```

### 9. GMM with Overidentification Test

```python
from causal_inference.iv import GMM

# When you have many instruments (q > p)
# GMM is asymptotically more efficient than 2SLS

np.random.seed(42)
n = 1000
Z1 = np.random.normal(0, 1, n)
Z2 = np.random.normal(0, 1, n)
Z3 = np.random.normal(0, 1, n)
Z = np.column_stack([Z1, Z2, Z3])  # 3 instruments

D = 2*Z1 + 1.5*Z2 + Z3 + np.random.normal(0, 1, n)
Y = 0.5*D + np.random.normal(0, 1, n)

# Fit two-step GMM (efficient)
gmm = GMM(steps='two', inference='robust')
gmm.fit(Y, D, Z)

print(f"GMM estimate: {gmm.coef_[0]:.3f}")
print(f"Hansen J-statistic: {gmm.j_statistic_:.3f}")
print(f"J-test p-value: {gmm.j_pvalue_:.3f}")

# Interpret J-test
if gmm.j_pvalue_ > 0.05:
    print("✓ Overidentifying restrictions valid (instruments OK)")
else:
    print("✗ J-test rejects - some instruments may be invalid")

# Compare one-step vs two-step
gmm_one = GMM(steps='one', inference='robust')
gmm_one.fit(Y, D, Z)
print(f"One-step: {gmm_one.coef_[0]:.3f} (SE: {gmm_one.se_[0]:.3f})")
print(f"Two-step: {gmm.coef_[0]:.3f} (SE: {gmm.se_[0]:.3f})")
# Two-step is asymptotically more efficient
```

## When to Use Which Estimator

### Use 2SLS When:
- ✅ Instruments are strong (F > 20)
- ✅ Just-identified or modestly over-identified
- ✅ Standard IV application
- ✅ Large sample size (n > 1000)

### Use LIML When:
- ⚠️ **Weak instruments (F < 10)**: LIML less biased than 2SLS
- ⚠️ **Many instruments (q > 10)**: LIML more efficient
- ⚠️ **Moderate sample (n ≈ 500-1000)**: Acceptable variance
- ⚠️ **Over-identified models**: LIML handles better

### Use Fuller When:
- ✅ **Weak instruments + small sample**: Fuller-1 recommended
- ✅ **Best all-around choice**: Balances bias and variance
- ✅ **F between 10-20**: Fuller often outperforms both 2SLS and LIML
- ✅ **Conservative inference**: Use Fuller-4

### Use GMM When:
- ✅ **Many instruments (q >> p)**: GMM is asymptotically efficient
- ✅ **Want to test overidentifying restrictions**: Hansen J-test
- ✅ **Two-step for efficiency**: Optimal weighting matrix accounts for heteroskedasticity
- ✅ **Large sample (n > 500)**: GMM asymptotic properties hold

### Consider Alternatives When:
- ⚠️ **Very weak instruments (F < 5)**: Use Anderson-Rubin CI or find better instruments
- ⚠️ **Heteroskedasticity**: Already handled (use `inference='robust'`)
- ⚠️ **Clustered data**: Already handled (use `inference='clustered'`)

## Diagnostic Interpretation

### First-Stage F-Statistic

| F-Statistic | Classification | Recommendation |
|-------------|----------------|----------------|
| F > 20 | Strong | ✅ Use 2SLS with confidence |
| 10 < F ≤ 20 | Moderate | ⚠️ Check Anderson-Rubin CI |
| F ≤ 10 | Weak | ❌ Avoid 2SLS, use AR CI or find better instruments |

### Stock-Yogo Critical Values (10% maximal bias)

| Instruments (q) | Endogenous (p) | Critical Value |
|-----------------|----------------|----------------|
| 1 | 1 | 16.38 |
| 2 | 1 | 19.93 |
| 3 | 1 | 22.30 |
| 2 | 2 | 11.04 |

If F > critical value, reject null of weak instruments.

### Anderson-Rubin Confidence Intervals

- **Robust to weak instruments** (unlike 2SLS CIs)
- Typically wider than 2SLS CIs when instruments are weak
- If AR CI excludes zero, evidence of causal effect even with weak instruments
- **Current limitation**: Implemented for just-identified case (q=1, p=1)

## API Reference

### TwoStageLeastSquares

```python
TwoStageLeastSquares(
    inference='robust',  # 'standard', 'robust', or 'clustered'
    cluster_var=None,    # Required if inference='clustered'
    alpha=0.05           # Significance level for CIs
)
```

**Attributes (after `.fit()`):**
- `coef_`: Coefficient estimates
- `se_`: Standard errors
- `t_stats_`: T-statistics
- `p_values_`: P-values (two-sided)
- `ci_`: Confidence intervals
- `first_stage_f_stat_`: First-stage F-statistic
- `first_stage_r2_`: First-stage R²
- `n_obs_`: Sample size
- `n_instruments_`: Number of instruments
- `n_endogenous_`: Number of endogenous variables

**Methods:**
- `.fit(Y, D, Z, X=None)`: Fit 2SLS model
- `.summary()`: Return formatted results table

### FirstStage, ReducedForm, SecondStage

```python
# First stage
first = FirstStage()
first.fit(D, Z, X=None)

# Attributes: coef_, se_, f_statistic_, partial_r2_, r2_, fitted_values_, residuals_

# Reduced form
reduced = ReducedForm()
reduced.fit(Y, Z, X=None)

# Attributes: coef_, se_, r2_, fitted_values_, residuals_

# Second stage (educational only - use 2SLS for inference)
second = SecondStage()
second.fit(Y, D_hat, X=None)

# Attributes: coef_, se_naive_ (WARNING: INCORRECT FOR INFERENCE)
```

### Diagnostic Functions

```python
# Classify instrument strength
classification, critical_value, interpretation = classify_instrument_strength(
    f_statistic,
    n_instruments=1,
    n_endogenous=1,
    bias_threshold='10pct'  # '10pct', '15pct', or '20pct'
)

# Cragg-Donald statistic (multivariate weak IV test)
cd_stat = cragg_donald_statistic(Y, D, Z, X=None)

# Anderson-Rubin test and CI
ar_stat, p_value, (ci_lower, ci_upper) = anderson_rubin_test(
    Y, D, Z, X=None,
    alpha=0.05,
    n_grid=100
)

# Comprehensive summary
summary_df = weak_instrument_summary(
    f_statistic,
    n_instruments,
    n_endogenous,
    cragg_donald=None,  # Optional
    ar_ci=None          # Optional
)
```

## Known Limitations

1. **Anderson-Rubin test**: Currently implemented for just-identified case (q=1, p=1). Over-identified case (q>1) requires additional normalization (future enhancement).

2. **Multivariate endogenous variables**: Current implementation focuses on single endogenous variable (p=1). Extension to p>1 is straightforward but not yet implemented.

3. **Non-linear IV**: Module supports linear IV only. Non-linear IV estimation requires different methods.

## References

### Textbooks
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*, Chapter 4.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*, Chapter 5.

### Papers
- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV regression. In *Identification and Inference for Econometric Models* (pp. 80-108).
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single equation in a complete system of stochastic equations. *Annals of Mathematical Statistics*, 20(1), 46-63.
- Cragg, J. G., & Donald, S. G. (1993). Testing identifiability and specification in instrumental variable models. *Econometric Theory*, 9(2), 222-240.

## Examples Directory

See `examples/iv/` for complete worked examples:
- `01_basic_iv.py`: Simple IV estimation
- `02_weak_instruments.py`: Weak instrument diagnostics
- `03_returns_to_schooling.py`: Classic Angrist-Krueger application
- `04_three_stages.py`: Educational decomposition

## Getting Help

- **Documentation**: See module docstrings for detailed API docs
- **Issues**: Report bugs at [GitHub repository]
- **Questions**: See FAQ in `docs/iv/FAQ.md`

## Version

**Current version**: 0.3.0 (Session 13: GMM Estimator + Hansen J-Test)

**Session 11** (0.1.0): Core 2SLS, three stages, weak instrument diagnostics
**Session 12** (0.2.0): LIML, Fuller, 99 comprehensive tests
**Session 13** (0.3.0): GMM (one-step + two-step), Hansen J-test, 116 comprehensive tests

Next planned features (Session 14):
- AR test for over-identified case (q>1)
- Multivariate IV (p>1)
- Additional enhancements
