"""
    CausalEstimators

Production-quality causal inference estimators following SciML design patterns.

Implements the Problem-Estimator-Solution architecture for randomized controlled trials (RCTs).

# Quick Start

```julia
using CausalEstimators

# Create RCT data
outcomes = [10.0, 12.0, 11.0, 4.0, 5.0, 3.0]
treatment = [true, true, true, false, false, false]

# Specify problem
problem = RCTProblem(outcomes, treatment, nothing, nothing, (alpha=0.05,))

# Estimate ATE
solution = solve(problem, SimpleATE())

println("ATE: \$(solution.estimate) ± \$(solution.se)")
println("95% CI: [\$(solution.ci_lower), \$(solution.ci_upper)]")
```

# Estimators

- `SimpleATE`: Difference-in-means with Neyman variance
- `StratifiedATE`: Block randomization
- `RegressionATE`: ANCOVA adjustment
- `PermutationTest`: Fisher exact test
- `IPWATE`: Inverse probability weighting

# Architecture

Based on SciML's Problem-Estimator-Solution pattern:
- **Problem**: Immutable data specification (`RCTProblem`)
- **Estimator**: Algorithm choice (`SimpleATE`, `StratifiedATE`, etc.)
- **Solution**: Results with metadata (`RCTSolution`)
- **Interface**: Universal `solve(problem, estimator)` pattern

# References

- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences*. Cambridge University Press.
- SciML Documentation: https://docs.sciml.ai/
"""
module CausalEstimators

# Standard library
using Statistics
using LinearAlgebra
using Random

# External dependencies
using StatsBase
using Distributions
using Combinatorics
using GLM
using DataFrames

# Abstract types (hierarchy)
include("problems/rct_problem.jl")
include("solutions/rct_solution.jl")
include("solutions/permutation_test_solution.jl")

# Validation utilities
include("problems/validation.jl")

# Utility functions
include("utils/errors.jl")
include("utils/statistics.jl")

# Core solve interface
include("solve.jl")

# RCT estimators
include("estimators/rct/simple_ate.jl")
include("estimators/rct/stratified_ate.jl")
include("estimators/rct/regression_ate.jl")
include("estimators/rct/permutation_test.jl")
include("estimators/rct/ipw_ate.jl")

# PSM estimators
include("estimators/psm/problem.jl")
include("estimators/psm/propensity.jl")
include("estimators/psm/matching.jl")
include("estimators/psm/variance.jl")
include("estimators/psm/balance.jl")
include("estimators/psm/nearest_neighbor.jl")

# RDD types (Phase 3)
include("rdd/types.jl")
include("rdd/sharp_rdd.jl")
include("rdd/fuzzy_rdd.jl")
include("rdd/sensitivity.jl")
include("rdd/mccrary.jl")   # Session 57: CJM (2020) proper variance

# IV types (Phase 4)
include("iv/types.jl")
include("iv/diagnostics.jl")
include("iv/tsls.jl")
include("iv/liml.jl")
include("iv/gmm.jl")
include("iv/weak_iv_robust.jl")
include("iv/vcov.jl")      # Session 56: Variance-covariance estimators
include("iv/stages.jl")    # Session 56: First/Reduced/Second stage decomposition

# DiD types (Phase 5 - Sessions 15-17)
include("did/types.jl")
include("did/classic_did.jl")
include("did/event_study.jl")

# DiD staggered methods (Phase 5 - Session 18)
include("did/staggered.jl")

# Observational IPW/DR (Session 32+)
include("observational/types.jl")
include("observational/propensity.jl")
include("observational/ipw.jl")
include("observational/outcome_models.jl")
include("observational/doubly_robust.jl")

# CATE Meta-Learners (Session 44)
include("cate/types.jl")
include("cate/utils.jl")
include("cate/s_learner.jl")
include("cate/t_learner.jl")
include("cate/x_learner.jl")
include("cate/r_learner.jl")
include("cate/dml.jl")

# Synthetic Control Methods (Session 47)
include("scm/types.jl")
include("scm/weights.jl")
include("scm/inference.jl")
include("scm/synthetic_control.jl")
include("scm/augmented_scm.jl")

# Sensitivity Analysis (Session 51)
include("sensitivity/types.jl")
include("sensitivity/e_value.jl")
include("sensitivity/rosenbaum.jl")

# Exports

## Abstract types
export AbstractCausalProblem, AbstractCausalEstimator, AbstractCausalSolution
export AbstractRCTProblem, AbstractRCTEstimator, AbstractRCTSolution
export AbstractPSMProblem, AbstractPSMEstimator, AbstractPSMSolution
export AbstractRDDProblem, AbstractRDDEstimator, AbstractRDDSolution
export AbstractIVProblem, AbstractIVEstimator, AbstractIVSolution
export AbstractDiDProblem, AbstractDiDEstimator, AbstractDiDSolution
export AbstractObservationalProblem, AbstractObservationalEstimator, AbstractObservationalSolution
export AbstractCATEProblem, AbstractCATEEstimator, AbstractCATESolution
export AbstractSCMProblem, AbstractSCMEstimator, AbstractSCMSolution
export AbstractSensitivityProblem, AbstractSensitivityEstimator, AbstractSensitivitySolution

## Problem types
export RCTProblem, PSMProblem, RDDProblem, IVProblem, DiDProblem, StaggeredDiDProblem
export SCMProblem
export ObservationalProblem
export CATEProblem
export EValueProblem, RosenbaumProblem
export FirstStageProblem, ReducedFormProblem, SecondStageProblem  # Session 56

## Estimator types
export SimpleATE, StratifiedATE, RegressionATE, PermutationTest, IPWATE
export NearestNeighborPSM
export SharpRDD, FuzzyRDD
export TSLS, LIML, GMM, AndersonRubin, ConditionalLR, OLS
export ClassicDiD, EventStudy, StaggeredTWFE, CallawaySantAnna, SunAbraham
export ObservationalIPW, DoublyRobust
export SLearner, TLearner, XLearner, RLearner, DoubleMachineLearning
export SyntheticControl, AugmentedSC
export EValue, RosenbaumBounds

## Solution types
export RCTSolution, PSMSolution, RDDSolution, FuzzyRDDSolution, IVSolution, DiDSolution
export IPWSolution, DRSolution
export CATESolution
export SCMSolution
export EValueSolution, RosenbaumSolution
export EffectType, RR, OR, HR, SMD, ATE, effect_type_from_symbol
export FirstStageSolution, ReducedFormSolution, SecondStageSolution  # Session 56

## RDD utilities
export AbstractBandwidthSelector, IKBandwidth, CCTBandwidth
export RDDKernel, TriangularKernel, UniformKernel, EpanechnikovKernel
export kernel_function
export McCraryTest
export select_bandwidth, mccrary_test
export bandwidth_sensitivity, placebo_test, balance_test, donut_rdd, permutation_test

## McCrary density test (Session 57)
export McCraryProblem, McCrarySolution, McCraryDensityTest

## Core interface
export solve

## Utilities
export remake  # For sensitivity analysis

## PSM utilities
export estimate_propensity, check_common_support
export nearest_neighbor_match, compute_ate_from_matches
export abadie_imbens_variance, pairs_bootstrap_variance
export compute_standardized_mean_difference, compute_variance_ratio
export check_covariate_balance, balance_summary

## IV diagnostics
export first_stage_fstat, cragg_donald_stat
export stock_yogo_critical_value, weak_iv_warning

## Weak IV robust inference
export ar_confidence_set, ar_test_statistic

## IV variance-covariance (Session 56)
export compute_standard_vcov, compute_robust_vcov, compute_clustered_vcov, compute_vcov

## Observational propensity utilities
export estimate_propensity_scores, compute_propensity_auc
export trim_propensities, compute_ipw_weights, check_positivity

## Observational outcome model utilities
export fit_outcome_models, compute_r2

## SCM utilities
export compute_scm_weights, compute_pre_treatment_fit

## Sensitivity utilities
export compute_e_value, convert_to_rr, smd_to_rr, ate_to_rr
export compute_signed_rank_statistic

end # module CausalEstimators
