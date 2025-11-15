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
include("rdd/sensitivity.jl")

# IV types (Phase 4)
include("iv/types.jl")

# Exports

## Abstract types
export AbstractCausalProblem, AbstractCausalEstimator, AbstractCausalSolution
export AbstractRCTProblem, AbstractRCTEstimator, AbstractRCTSolution
export AbstractPSMProblem, AbstractPSMEstimator, AbstractPSMSolution
export AbstractRDDProblem, AbstractRDDEstimator, AbstractRDDSolution
export AbstractIVProblem, AbstractIVEstimator, AbstractIVSolution

## Problem types
export RCTProblem, PSMProblem, RDDProblem, IVProblem

## Estimator types
export SimpleATE, StratifiedATE, RegressionATE, PermutationTest, IPWATE
export NearestNeighborPSM
export SharpRDD

## Solution types
export RCTSolution, PSMSolution, RDDSolution, IVSolution

## RDD utilities
export AbstractBandwidthSelector, IKBandwidth, CCTBandwidth
export RDDKernel, TriangularKernel, UniformKernel, EpanechnikovKernel
export kernel_function
export McCraryTest
export select_bandwidth, mccrary_test
export bandwidth_sensitivity, placebo_test, balance_test, donut_rdd, permutation_test

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

end # module CausalEstimators
