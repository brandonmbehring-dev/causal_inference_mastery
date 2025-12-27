"""
Time Series Causal Inference Module

Sessions 135-137, 147: Granger causality, VAR, PCMCI, SVAR, and more.

This module provides:
- Granger causality tests (pairwise and multivariate)
- VAR (Vector Autoregression) estimation
- Stationarity tests (ADF, KPSS, Phillips-Perron)
- Lag selection via information criteria
- PCMCI algorithm for time-series causal discovery
- Structural VAR (SVAR) with Cholesky identification
- Impulse Response Functions (IRF) with bootstrap inference
- Forecast Error Variance Decomposition (FEVD) with bootstrap
- Cointegration tests (Johansen, Engle-Granger)
- Moving Block Bootstrap for time-dependent data

# Example: Granger Causality
```julia
using CausalEstimators
using Random
Random.seed!(42)
n = 200
x = randn(n)
y = zeros(n)
for t in 2:n
    y[t] = 0.5 * x[t-1] + 0.3 * y[t-1] + randn() * 0.5
end
data = hcat(y, x)
result = granger_causality(data, lags=2)
println("Granger causes: ", result.granger_causes)
```

# Example: PCMCI
```julia
using CausalEstimators
using Random
Random.seed!(42)
n = 200
data = zeros(n, 3)
data[:, 1] = randn(n)
for t in 2:n
    data[t, 2] = 0.6 * data[t-1, 1] + 0.3 * data[t-1, 2] + randn() * 0.5
    data[t, 3] = 0.5 * data[t-1, 2] + 0.2 * data[t-1, 3] + randn() * 0.5
end
result = pcmci(data, max_lag=2, alpha=0.05)
println("Found \$(length(result.links)) causal links")
```

# Example: Structural VAR
```julia
using CausalEstimators
using Random
Random.seed!(42)
data = randn(200, 3)
var_result = var_estimate(data, lags=2)
svar_result = cholesky_svar(var_result)
irf = compute_irf(svar_result, horizons=20)
println("IRF shape: \$(size(irf.irf))")
```
"""
module TimeSeries

# Types
include("types.jl")
using .TimeSeriesTypes

# VAR estimation
include("var.jl")
using .VAR

# Granger causality
include("granger.jl")
using .Granger

# PCMCI Types
include("pcmci_types.jl")
using .PCMCITypes

# PCMCI CI tests
include("ci_tests.jl")
using .CITests

# PCMCI algorithm
include("pcmci.jl")
using .PCMCI: pcmci, pc_stable_condition_selection, mci_test_all

# SVAR Types
include("svar_types.jl")
using .SVARTypes

# SVAR algorithm
include("svar.jl")
using .SVAR

# Stationarity tests (Session 147)
include("stationarity.jl")
using .Stationarity

# Cointegration tests (Session 147)
include("cointegration.jl")
using .Cointegration

# Bootstrap IRF/FEVD (Session 147)
include("bootstrap_irf.jl")
using .BootstrapIRF

# VECM (Session 149)
include("vecm.jl")

# Re-export types
export GrangerResult, VARResult, ADFResult, LagSelectionResult
export KPSSResult, PPResult, ConfirmatoryResult
export JohansenResult, EngleGrangerResult, VECMResult

# Re-export VAR functions
export var_estimate, var_forecast, var_residuals

# Re-export type accessor functions from TimeSeriesTypes
export n_vars, n_params_per_eq, get_lag_matrix, get_intercepts
export get_optimal_by_criterion

# Re-export Granger functions
export granger_causality, granger_causality_matrix, bidirectional_granger

# Re-export PCMCI types
export LinkType, TimeSeriesLink, LaggedDAG, PCMCIResult
export ConditionSelectionResult, CITestResult
export is_significant, is_lagged, add_edge!, remove_edge!, has_edge
export get_parents, get_children, n_edges, to_links
export get_lagged_dag, get_significant_links, get_parents_of

# Re-export PCMCI CI tests
export parcorr_test, run_ci_test

# Re-export PCMCI algorithm
export pcmci, pc_stable_condition_selection, mci_test_all

# Re-export SVAR types
export IdentificationMethod, SVARResult, IRFResult, FEVDResult
export is_just_identified, is_over_identified
export get_structural_coefficient, get_response, get_decomposition
export has_confidence_bands, validate_rows_sum_to_one

# Re-export SVAR functions
export cholesky_svar, long_run_svar, companion_form, vma_coefficients, structural_vma_coefficients
export check_stability, long_run_impact_matrix, verify_identification
export compute_irf, compute_fevd

# Re-export Stationarity functions
export adf_test, kpss_test, phillips_perron_test
export confirmatory_stationarity_test, difference_series, check_stationarity

# Re-export Cointegration functions
export johansen_test, engle_granger_test

# Re-export Bootstrap IRF/FEVD functions
export bootstrap_irf, moving_block_bootstrap_irf
export joint_confidence_bands, moving_block_bootstrap_irf_joint
export bootstrap_fevd

# Re-export VECM functions
export vecm_estimate, vecm_forecast, compute_error_correction_term

end # module
