"""
Main test runner for CausalEstimators.jl

Following SciML pattern with SafeTestsets for test isolation.

Run with: julia --project test/runtests.jl
"""

using Test
using SafeTestsets

@info "Starting CausalEstimators.jl test suite"

# Test module loading
@safetestset "Module Loading" begin
    using CausalEstimators
    @test true  # If module loads, test passes
end

# Test problem construction
@safetestset "Problem Construction" begin
    include("test_problems.jl")
end

# Test solution types
@safetestset "Solution Types" begin
    include("test_solutions.jl")
end

# RCT Estimators
@safetestset "RCT Estimators" begin
    @safetestset "SimpleATE" begin include("rct/test_simple_ate.jl") end
    @safetestset "StratifiedATE" begin include("rct/test_stratified_ate.jl") end
    @safetestset "RegressionATE" begin include("rct/test_regression_ate.jl") end
    @safetestset "PermutationTest" begin include("rct/test_permutation_test.jl") end
    @safetestset "IPWATE" begin include("rct/test_ipw_ate.jl") end
end

# PSM Estimators
@safetestset "PSM Estimators" begin
    @safetestset "Propensity Estimation" begin include("estimators/psm/test_propensity.jl") end
    @safetestset "Matching Algorithm" begin include("estimators/psm/test_matching.jl") end
    @safetestset "Balance Diagnostics" begin include("estimators/psm/test_balance.jl") end
    @safetestset "NearestNeighborPSM" begin include("estimators/psm/test_nearest_neighbor_psm.jl") end
    @safetestset "Monte Carlo Validation" begin include("estimators/psm/test_monte_carlo.jl") end
end

# Golden Reference Validation
@safetestset "Golden Reference Validation" begin
    include("rct/test_golden_reference.jl")
end

# RDD Estimators (Phase 3)
@safetestset "RDD Estimators" begin
    include("rdd/runtests.jl")
end

# Observational IPW/DR (Session 32+)
@safetestset "Observational Estimators" begin
    include("observational/runtests.jl")
end

# DiD Estimators (Session 63 validation)
@safetestset "DiD Estimators" begin
    include("did/runtests.jl")
end

# CATE Meta-Learners (Session 44)
@safetestset "CATE Estimators" begin
    include("cate/runtests.jl")
end

# Synthetic Control Methods (Session 47)
@safetestset "SCM Estimators" begin
    include("scm/runtests.jl")
end

# Sensitivity Analysis (Session 51)
@safetestset "Sensitivity Analysis" begin
    include("sensitivity/runtests.jl")
end

# RKD Estimators (Session 74)
@safetestset "RKD Estimators" begin
    include("rkd/runtests.jl")
end

# Bunching Estimation (Session 78)
@safetestset "Bunching Estimators" begin
    include("bunching/runtests.jl")
end

# Selection Models (Session 85)
@safetestset "Selection Models" begin
    include("selection/runtests.jl")
end

# QTE (Session 89)
@safetestset "QTE Estimators" begin
    include("qte/runtests.jl")
end

# MTE (Session 91)
@safetestset "MTE Estimators" begin
    include("mte/runtests.jl")
end

# Control Function (Session 95)
@safetestset "Control Function" begin
    include("control_function/runtests.jl")
end

# Partial Identification Bounds (Session 95)
@safetestset "Bounds" begin
    include("bounds/runtests.jl")
end

# Mediation Analysis (Session 95)
@safetestset "Mediation" begin
    include("mediation/runtests.jl")
end

# Shift-Share IV (Session 97)
@safetestset "Shift-Share IV" begin
    include("shift_share/test_shift_share.jl")
end

# Dynamic Treatment Regimes (Session 121)
@safetestset "DTR Estimators" begin
    include("dtr/runtests.jl")
end

# Time Series Causal Inference (Session 147, 149, 150)
@safetestset "Time Series" begin
    @safetestset "Stationarity" begin include("timeseries/test_stationarity.jl") end
    @safetestset "Cointegration" begin include("timeseries/test_cointegration.jl") end
    @safetestset "Bootstrap IRF" begin include("timeseries/test_bootstrap_irf.jl") end
    @safetestset "VECM" begin include("timeseries/test_vecm.jl") end
    @safetestset "Granger" begin include("timeseries/test_granger.jl") end
    @safetestset "VAR" begin include("timeseries/test_var.jl") end
    @safetestset "PCMCI" begin include("timeseries/test_pcmci.jl") end
end

@info "Test suite complete"
