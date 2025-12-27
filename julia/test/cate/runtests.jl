#=
CATE Module Test Runner

Tests for CATE (Conditional Average Treatment Effect) meta-learners:
- S-Learner: Single model approach
- T-Learner: Two models approach
- X-Learner: Cross-learner with propensity weighting
- R-Learner: Robinson transformation
- DML: Double Machine Learning with cross-fitting
- DragonNet: Neural CATE with shared representation (Session 152)
- IRM: Interactive Regression Model (Session 153)
=#

using Test
using CausalEstimators
using Random
using Statistics

@testset "CATE Module Tests" begin
    # Unit tests
    include("test_s_learner.jl")
    include("test_t_learner.jl")
    include("test_x_learner.jl")
    include("test_r_learner.jl")
    include("test_dml.jl")
    include("test_dml_continuous.jl")
    include("test_dragonnet.jl")  # Session 152: Neural CATE
    include("test_oml.jl")  # Session 153: OML/IRM

    # Validation tests (Monte Carlo and Adversarial)
    @testset "CATE Monte Carlo Validation" begin
        include("test_cate_montecarlo.jl")
    end
    @testset "CATE Adversarial Tests" begin
        include("test_cate_adversarial.jl")
    end
end
