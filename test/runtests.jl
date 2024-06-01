using SSM
using Distributions
using ForwardDiff
using LinearAlgebra
using Random
using StatsFuns
using Test

Random.seed!(1234)

"""
Tests for MixtureModels.jl
"""

# Test general properties of GaussianMixtureModel
function test_GaussianMixtureModel_properties(gmm::GaussianMixtureModel, k::Int, data_dim::Int)
    @test gmm.k == k
    @test size(gmm.μₖ) == (k, data_dim)

    for Σ in gmm.Σₖ
        @test size(Σ) == (data_dim, data_dim)
        @test ishermitian(Σ)
    end

    @test length(gmm.πₖ) == k
    @test sum(gmm.πₖ) ≈ 1.0
end

function testGaussianMixtureModel_EStep(gmm::GaussianMixtureModel, data::Union{Matrix{Float64}, Vector{Float64}})

    k::Int = gmm.k
    data_dim::Int = size(data, 2)
    
    # Run EStep
    class_probabilities = SSM.EStep(gmm, data)
    # Check dimensions
    @test size(class_probabilities) == (size(data, 1), k)
    # Check if the row sums are close to 1 (since they represent probabilities)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(class_probabilities, dims=2))
    
    test_GaussianMixtureModel_properties(gmm, k, data_dim)
end

function testGaussianMixtureModel_MStep(gmm::GaussianMixtureModel, data::Union{Matrix{Float64}, Vector{Float64}})

    k::Int = gmm.k
    data_dim::Int = size(data, 2)

    class_probabilities = SSM.EStep(gmm, data)

    # Run MStep
    SSM.MStep!(gmm, data, class_probabilities)

    test_GaussianMixtureModel_properties(gmm, k, data_dim)
end

function testGaussianMixtureModel_fit(gmm::GaussianMixtureModel, data::Union{Matrix{Float64}, Vector{Float64}})

    k::Int = gmm.k
    data_dim::Int = size(data, 2)

    # Run fit!
    fit!(gmm, data; maxiter=10, tol=1e-3)

    test_GaussianMixtureModel_properties(gmm, k, data_dim)
end

function test_log_likelihood(gmm::GaussianMixtureModel, data::Union{Matrix{Float64}, Vector{Float64}})

    # Calculate log-likelihood
    ll = log_likelihood(gmm, data)

    # Check if log-likelihood is a scalar
    @test size(ll) == ()

    # Log-likelihood should be a negative float
    @test ll < 0.0

    # Log-likelihood should monotonically increase with iterations (when using exact EM)

    #repeatedly applying fit! without initializtion, so first initialize means
    # Initialize k means of gmm
	gmm.μₖ = permutedims(kmeanspp_initialization(data, gmm.k))
    
    ll_prev = -Inf
    for i in 1:10
        fit!(gmm, data; maxiter=1, tol=1e-3, initialize_kmeans=false)
        ll = log_likelihood(gmm, data)
        @test ll > ll_prev || isapprox(ll, ll_prev; atol=1e-6)
        ll_prev = ll
    end
end

"""
Tests for PoissonMixtureModel
"""

# Test general properties of PoissonMixtureModel
function test_PoissonMixtureModel_properties(pmm::PoissonMixtureModel, k::Int)
    @test pmm.k == k
    @test length(pmm.λₖ) == k
    @test length(pmm.πₖ) == k
    @test sum(pmm.πₖ) ≈ 1.0
end

function testPoissonMixtureModel_EStep(pmm::PoissonMixtureModel, data::Union{Matrix{Int}, Vector{Int}})
    k::Int = pmm.k
    
    # Run EStep
    class_probabilities = SSM.EStep(pmm, data)
    # Check dimensions
    @test size(class_probabilities) == (size(data, 1), k)
    # Check if the row sums are close to 1 (since they represent probabilities)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(class_probabilities, dims=2))
    
    test_PoissonMixtureModel_properties(pmm, k)
end

function testPoissonMixtureModel_MStep(pmm::PoissonMixtureModel, data::Union{Matrix{Int}, Vector{Int}})
    k::Int = pmm.k

    class_probabilities = SSM.EStep(pmm, data)

    # Run MStep
    SSM.MStep!(pmm, data, class_probabilities)

    test_PoissonMixtureModel_properties(pmm, k)
end

function testPoissonMixtureModel_fit(pmm::PoissonMixtureModel, data::Union{Matrix{Int}, Vector{Int}})
    k::Int = pmm.k

    # Run fit!
    fit!(pmm, data; maxiter=10, tol=1e-3)

    test_PoissonMixtureModel_properties(pmm, k)
end

function test_log_likelihood(pmm::PoissonMixtureModel, data::Union{Matrix{Int}, Vector{Int}})
    # Calculate log-likelihood
    ll = log_likelihood(pmm, data)

    # Check if log-likelihood is a scalar
    @test size(ll) == ()

    # Log-likelihood should not necessarily be negative for Poisson models

    # Initialize λₖ with kmeans_init
    λₖ_matrix = permutedims(kmeanspp_initialization(Float64.(data), pmm.k))
    pmm.λₖ = vec(λₖ_matrix)

    # Log-likelihood should monotonically increase with iterations (when using exact EM)
    ll_prev = -Inf
    for i in 1:10
        fit!(pmm, data; maxiter=1, tol=1e-3, initialize_kmeans=false)
        ll = log_likelihood(pmm, data)
        @test ll > ll_prev || isapprox(ll, ll_prev; atol=1e-6)
        ll_prev = ll
    end
end




@testset "MixtureModels.jl Tests" begin
    # Test GaussianMixtureModel

    
    # Initialize test models


    # Standard GaussianMixtureModel model

    # Number of clusters
    k = 3
    # Dimension of data points
    data_dim = 2
    # Construct gmm
    standard_gmm = GaussianMixtureModel(k, data_dim)
    # Generate sample data
    standard_data = randn(10, data_dim)

    # Test constructor method of GaussianMixtureModel
    test_GaussianMixtureModel_properties(standard_gmm, k, data_dim)



    # Vector-data GaussianMixtureModel model

    # Number of clusters
    k = 2
    # Dimension of data points
    data_dim = 1
    # Construct gmm
    vector_gmm = GaussianMixtureModel(k, data_dim)
    # Generate sample data
    vector_data = randn(1000,)
    # Test constructor method of GaussianMixtureModel
    test_GaussianMixtureModel_properties(vector_gmm, k, data_dim)
  
    # Test EM methods of the GaussianMixtureModels

    # Paired data and GaussianMixtureModels to test
    tester_set = [
        (standard_gmm, standard_data), 
        (vector_gmm, vector_data),
        ]

    for (gmm, data) in tester_set
        k = gmm.k
        data_dim = size(data, 2)

        gmm = GaussianMixtureModel(k, data_dim)
        testGaussianMixtureModel_EStep(gmm, data)

        gmm = GaussianMixtureModel(k, data_dim)
        testGaussianMixtureModel_MStep(gmm, data)

        gmm = GaussianMixtureModel(k, data_dim)
        testGaussianMixtureModel_fit(gmm, data)

        gmm = GaussianMixtureModel(k, data_dim)
        test_log_likelihood(gmm, data)
    end
  
    # Test PoissonMixtureModel
    k = 3  # Number of clusters
    
    # Simulate some Poisson-distributed data using the sample function
    # First, define a temporary PMM for sampling purposes
    temp_pmm = PoissonMixtureModel(k)
    temp_pmm.λₖ = [5.0, 10.0, 15.0]  # Assign some λ values for generating data
    temp_pmm.πₖ = [1/3, 1/3, 1/3]  # Equal mixing coefficients for simplicity
    data = SSM.sample(temp_pmm, 300)  # Generate sample data
    
    standard_pmm = PoissonMixtureModel(k)
    
    # Conduct tests
    test_PoissonMixtureModel_properties(standard_pmm, k)
    
    tester_set = [(standard_pmm, data)]
    
    for (pmm, data) in tester_set
        pmm = PoissonMixtureModel(k)
        testPoissonMixtureModel_EStep(pmm, data)
        pmm = PoissonMixtureModel(k)
        testPoissonMixtureModel_MStep(pmm, data)
        pmm = PoissonMixtureModel(k)
        testPoissonMixtureModel_fit(pmm, data)
        pmm = PoissonMixtureModel(k)
        test_log_likelihood(pmm, data)
    end
end

"""
Tests for HiddenMarkovModels.jl
"""

function toy_HMM(k::Int=3, data_dim::Int=2, n::Int=1000)
    # create random data
    data = randn(n, data_dim)
    # fit hmm
    hmm = GaussianHMM(data, k)
    return hmm, data
end

function test_toy_HMM()
    hmm, data = toy_HMM()
    @test size(data, 2) == hmm.D
    @test size(data, 1) == 1000
    @test hmm.K == 3
end

function test_HMM_properties(hmm::GaussianHMM)
    @test isapprox(sum(hmm.A, dims=2), ones(hmm.K))
    @test typeof(hmm.B) == Vector{GaussianEmission}
    @test sum(hmm.πₖ) ≈ 1.0
end

function test_GaussianHMM_constructor()
    hmm, _ = toy_HMM()
    test_HMM_properties(hmm)
end

function test_HMM_forward_and_back()
    hmm, data = toy_HMM()
    α = SSM.forward(hmm, data)
    @test size(α) == (size(data, 1), hmm.K)
    β = SSM.backward(hmm, data)
    @test size(β) == (size(data, 1), hmm.K)
end

function test_HMM_gamma_xi()
    hmm, data = toy_HMM()
    α = SSM.forward(hmm, data)
    β = SSM.backward(hmm, data)
    γ = SSM.calculate_γ(hmm, α, β)
    ξ = SSM.calculate_ξ(hmm, α, β, data)
    @test size(γ) == (size(data, 1), hmm.K)
    @test size(ξ) == (size(data, 1) - 1, hmm.K, hmm.K)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(γ), dims=2))
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(ξ), dims=(2, 3)))
end

function test_HMM_E_step()
    hmm, data = toy_HMM()
    γ, ξ, α, β = SSM.E_step(hmm, data)
    @test size(γ) == (size(data, 1), hmm.K)
    @test size(ξ) == (size(data, 1) - 1, hmm.K, hmm.K)
end

function test_HMM_M_step()
    hmm, data = toy_HMM()
    # test indiviudal M-step functions
    γ, ξ, α, β = SSM.E_step(hmm, data)
    SSM.update_initial_state_distribution!(hmm, γ)
    @test sum(hmm.πₖ) ≈ 1.0
    SSM.update_transition_matrix!(hmm, γ, ξ)
    @test isapprox(sum(hmm.A, dims=2), ones(hmm.K))
    SSM.update_emission_models!(hmm, γ, data)
    @test typeof(hmm.B) == Vector{GaussianEmission}
    # test M-step
    γ, ξ, α, β = SSM.E_step(hmm, data)
    SSM.M_step!(hmm, γ, ξ, data)
    test_HMM_properties(hmm)
end

function test_HMM_EM()
    Random.seed!(1234)
    A = [0.7 0.2 0.1; 0.1 0.7 0.2; 0.2 0.1 0.7]
    means = [[0.0, 0.0], [-1.0, 2.0], [3.0, 2.5]]
    covs = [
        [0.1 0.0; 0.0 0.1], 
        [0.1 0.0; 0.0 0.1], 
        [0.1 0.0; 0.0 0.1]
    ]
    emissions_models = [GaussianEmission(mean, cov) for (mean, cov) in zip(means, covs)]
    simul_hmm = GaussianHMM(A, emissions_models, [0.33, 0.33, 0.34], 3, 2)
    states, observations = SSM.sample(simul_hmm, 10000)
    hmm = GaussianHMM(observations, 3)
    baumWelch!(hmm, observations, 100)
    pred_means = [hmm.B[i].μ for i in 1:3]
    @test sort(pred_means) ≈ sort(means) atol=0.2
    pred_covs = [hmm.B[i].Σ for i in 1:3]
    @test pred_covs ≈ covs atol=0.1
    best_path = viterbi(hmm, observations)
    @test length(best_path) == 10000
    @test all(x -> x in 1:3, best_path)
end
@testset "HiddenMarkovModels.jl Tests" begin
    test_toy_HMM()
    test_GaussianHMM_constructor()
    test_HMM_forward_and_back()
    test_HMM_gamma_xi()
    test_HMM_E_step()
    test_HMM_M_step()
    test_HMM_EM()
end

"""
Tests for LDS.jl
"""
# Create a toy example for all LDS tests. This example represents a pendulum in a frictionless environment.
g = 9.81 # gravity
l = 1.0 # length of pendulum
dt = 0.01 # time step
T = 10.0 # total time
# Discrete-time dynamics
A = [1.0 dt; -g/l*dt 1.0]
# Initial state
x0 = [0.0; 1.0]
# Time vector
t = 0:dt:T
# Define the LDS model parameters
H = I(2)  # Observation matrix (assuming direct observation)
Q = 0.00001 * I(2)  # Process noise covariance
observation_noise_std = 0.5
R = (observation_noise_std^2) * I(2)  # Observation noise covariance
p0 = 0.1*I(2)  # Initial state covariance
x0 = [0.0; 1.0]  # Initial state mean
# Generate true data
x = zeros(2, length(t))
x[:,1] = x0
for i = 2:length(t)
    x[:,i] = A*x[:,i-1]
end
# Generate noisy data
x_noisy = zeros(2, length(t))
x_noisy[:, 1] = x0

noise = rand(Normal(0, observation_noise_std), (2, length(t)))

for i in 2:length(t)
    x_noisy[:, i] = A * x[:, i-1] + noise[:, i]
end

function test_LDS_with_params()
    # Create the Kalman filter parameter vector
    kf = LDS(A,
             H,
             nothing,
             Q, 
             R, 
             x0, 
             p0, 
             nothing, 
             2, 
             2, 
             Vector([false, false, false, false, false, false, false, false]))
    # confirm parameters are set correctly
    @test kf.A == A
    @test kf.H == H
    @test kf.B === nothing
    @test kf.Q == Q
    @test kf.R == R
    @test kf.x0 == x0
    @test kf.p0 == p0
    @test kf.inputs === nothing
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == Vector([false, false, false, false, false, false, false, false])
    # run the filter
    x_filt, p_filt, x_pred, p_pred, v, F, K, ml = KalmanFilter(kf, x_noisy')
    # check dimensions
    @test size(x_filt) == (length(t), 2)
    @test size(p_filt) == (length(t), 2, 2)
    @test size(x_pred) == (length(t), 2)
    @test size(p_pred) == (length(t), 2, 2)
    @test size(v) == (length(t), 2)
    @test size(F) == (length(t), 2, 2)
    @test size(K) == (length(t), 2, 2)
    # run the smoother
    x_smooth, p_smooth = KalmanSmoother(kf, x_noisy')
    # check dimensions
    @test size(x_smooth) == (length(t), 2)
    @test size(p_smooth) == (length(t), 2, 2)
end

function test_LDS_without_params()
    # Create the Kalman filter without any params
    kf = LDS()
    # confirm parameters are set correctly
    @test kf.A !== nothing
    @test kf.H !== nothing
    @test kf.B === nothing
    @test kf.Q !== nothing
    @test kf.R !== nothing
    @test kf.x0 !== nothing
    @test kf.p0 !== nothing
    @test kf.inputs === nothing
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == fill(true, 7)
end

function test_LDS_EStep()
    # Create the Kalman filter parameter vector
    kf = LDS(A,
             H,
             nothing,
             Q, 
             R, 
             x0, 
             p0, 
             nothing, 
             2, 
             2, 
             Vector([true, true, true, true, true, true, true]))
    # run the EStep
    x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml = SSM.EStep(kf, x_noisy')
    # check dimensions
    @test size(x_smooth) == (length(t), 2)
    @test size(p_smooth) == (length(t), 2, 2)
    @test size(E_z) == (length(t), 2)
    @test size(E_zz) == (length(t), 2, 2)
    @test size(E_zz_prev) == (length(t), 2, 2)
    @test size(ml) == ()
end

function test_LDS_MStep!()
    # Create the Kalman filter parameter vector
    kf = LDS(A,
             H,
             nothing,
             Q, 
             R, 
             x0, 
             p0, 
             nothing, 
             2, 
             2, 
             Vector([true, true, true, true, true, true, true]))
    # run the EStep
    x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml = SSM.EStep(kf, x_noisy')
    # run the MStep
    SSM.MStep!(kf, E_z, E_zz, E_zz_prev, x_noisy')
    # check if the parameters are updated
    @test kf.A !== A
    @test kf.H !== H
    @test kf.B === nothing
    @test kf.Q !== Q
    @test kf.R !== R
    @test kf.x0 !== x0
    @test kf.p0 !== p0
    @test kf.inputs === nothing
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == Vector([true, true, true, true, true, true, true])
end

function test_LDS_EM()
    kf = LDS(A,
             H,
             nothing,
             Q, 
             R, 
             x0, 
             p0, 
             nothing, 
             2, 
             2, 
             Vector([true, true, true, true, true, true, true]))
    # run the EM
    for i in 1:10
        ml_prev = -Inf
        l, ml = SSM.KalmanFilterEM!(kf, x_noisy', 1)
        @test ml > ml_prev
        ml_prev = ml
    end
    # check if the parameters are updated
    @test kf.A !== A
    @test kf.H !== H
    @test kf.B === nothing
    @test kf.Q !== Q
    @test kf.R !== R
    @test kf.x0 !== x0
    @test kf.p0 !== p0
    @test kf.inputs === nothing
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == Vector([true, true, true, true, true, true, true]) 
end

@testset "LDS Tests" begin
    test_LDS_with_params()
    test_LDS_without_params()
    test_LDS_EStep()
    test_LDS_MStep!()
    test_LDS_EM()
end

function test_PLDS_constructor_with_params()
    # create a set of parameters to test with
    obs_dim = 10
    latent_dim = 5

    A = randn(latent_dim, latent_dim)
    C = randn(obs_dim, latent_dim)
    Q = I(latent_dim)
    x0 = randn(latent_dim)
    p0 = I(latent_dim)
    refrac = 1
    d = randn(obs_dim)
    D = randn(obs_dim, obs_dim)
    fit_bool=Vector([true, true, true, true, true, true, true])

    # create the PLDS model
    plds = PoissonLDS(A=A, C=C, Q=Q, D=D, d=d, x0=x0, p0=p0, refractory_period=1, obs_dim=obs_dim, latent_dim=latent_dim, fit_bool=fit_bool)

    # test model
    @test plds.A == A
    @test plds.C == C
    @test plds.Q == Q
    @test plds.x0 == x0
    @test plds.p0 == p0
    @test plds.d == d
    @test plds.D == D
    @test plds.refractory_period == 1
    @test plds.obs_dim == obs_dim
    @test plds.latent_dim == latent_dim
end

function test_PLDS_constructor_without_params()
    # create the PLDS model
    plds = PoissonLDS(;obs_dim=10, latent_dim=5)

    # test parameters are not empty
    @test !isempty(plds.A)
    @test !isempty(plds.C)
    @test !isempty(plds.Q)
    @test !isempty(plds.x0)
    @test !isempty(plds.p0)
    @test !isempty(plds.d)
    @test !isempty(plds.D)
    @test plds.refractory_period == 1
    @test plds.obs_dim == 10
    @test plds.latent_dim == 5

    # test dims of parameters
    @test size(plds.A) == (5, 5)
    @test size(plds.C) == (10, 5)
    @test size(plds.Q) == (5, 5)
    @test size(plds.x0) == (5,)
    @test size(plds.p0) == (5, 5)
    @test size(plds.d) == (10,)
    @test size(plds.D) == (10, 10)
end

function test_countspikes()
    # create a set of observations that is a matrix of spikes/events
    obs = [0 0 1; 1 1 1; 0 1 0]
    # count the spikes when window=1
    count = SSM.countspikes(obs, 1)
    # check the count
    @test count == [0 0 0; 0 0 1; 1 1 1]
    # count spikes when window=2
    count_2 = SSM.countspikes(obs, 2)
    # check the count
    @test count_2 == [0 0 0; 0 0 1; 1 1 2]
end

function test_logposterior()
    # create a plds model
    plds = PoissonLDS(;obs_dim=10, latent_dim=5)
    # create some observations
    obs = rand(Bool, 100, 10)
    # create latent state
    x = randn(100, 5)
    # calculate the log posterior
    logpost = SSM.logposterior(x, plds, obs)
    # check the dimensions
    @test logpost isa Float64
end

function test_gradient_plds()
    # create a plds model
    plds = PoissonLDS(;obs_dim=10, latent_dim=5)
    # create some observations
    obs = rand(Bool, 100, 10)
    # create initial latent state for gradient calculation
    x = randn(100, 5)
    # calculate the gradient
    grad = SSM.Gradient(x, plds, obs)
    # check the dimensions
    @test size(grad) == (100, 5)
    # check the gradients using autodiff
    obj(x) = x -> SSM.logposterior(x, plds, obs)
    grad_autodiff = ForwardDiff.gradient(obj(x), x)
    @test grad ≈ grad_autodiff atol=1e-6
end

function test_hessian_plds()
    # create a plds model
    plds = PoissonLDS(;obs_dim=10, latent_dim=5)
    # create some observations
    obs = rand(Bool, 100, 10)
    # create initial latent state for hessian calculation
    x = randn(100, 5)
    # calculate the hessian
    hess, main, super, sub = SSM.Hessian(x, plds, obs)
    # check the dimensions
    @test length(main) == 100
    @test length(super) == 99
    @test length(sub) == 99
    @test size(hess) == (500, 500)
    # check the hessian using autodiff
    function obj_logposterior(x::Vector)
        x = SSM.interleave_reshape(x, 100, 5)
        return SSM.logposterior(x, plds, obs)
    end
    hess_autodiff = ForwardDiff.hessian(obj_logposterior, reshape(x', 500))
    @test hess ≈ hess_autodiff atol=1e-6
end


@testset "PLDS Tests" begin
    test_PLDS_constructor_with_params()
    test_PLDS_constructor_without_params()
    test_countspikes()
    test_logposterior()
    test_gradient_plds()
    test_hessian_plds()
end

"""
Tests for Regression.jl
"""
function test_GaussianRegression_fit()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    y = X * true_β + randn(1000) * 0.5
    
    # Initialize and fit the model
    model = GaussianRegression()
    fit!(model, X[:, 2:end], y)
    
    # Check if the fitted coefficients are close to the true coefficients
    @test isapprox(model.β, true_β, atol=0.5)
    @test model.σ² > 0
    @test isapprox(model.σ², 0.25, atol=0.1)
end

function test_GaussianRegression_loglikelihood()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    y = X * true_β + randn(1000) * 0.5
    
    # Initialize and fit the model
    model = GaussianRegression()
    fit!(model, X[:, 2:end], y)
    # check if the fitted coefficients are close to the true coefficients
    @test isapprox(model.β, true_β, atol=0.5)
    @test model.σ² > 0
    @test isapprox(model.σ², 0.25, atol=0.1)
    
    # Check log likelihood
    loglik = SSM.loglikelihood(model, X[:, 2:end], y)
    @test loglik < 0

    # test loglikelihood on a single point
    loglik = SSM.loglikelihood(model, X[1, 2:end], y[1])
    @test loglik < 0
end

function test_GaussianRegression_empty_model()
    model = GaussianRegression()
    @test isempty(model.β)
    @test model.σ² == 0.0
end

function test_GaussianRegression_intercept()
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    y = X * true_β + randn(1000) * 0.5
    
    model = GaussianRegression(include_intercept=false)
    fit!(model, X[:, 2:end], y)
    @test length(model.β) == 2
    
    model_with_intercept = GaussianRegression()
    fit!(model_with_intercept, X[:, 2:end], y)
    @test length(model_with_intercept.β) == 3
    @test isapprox(model_with_intercept.β, true_β, atol=0.5)
end

@testset "GaussianRegression Tests" begin
    test_GaussianRegression_fit()
    test_GaussianRegression_loglikelihood()
    test_GaussianRegression_empty_model()
    test_GaussianRegression_intercept()
end

function test_BernoulliRegression_fit()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    p = logistic.(X * true_β)
    y = rand.(Bernoulli.(p))
    
    # Initialize and fit the model
    model = BernoulliRegression()
    fit!(model, X[:, 2:end], y)
    
    # Check if the fitted coefficients are reasonable
    @test length(model.β) == 3
    @test isapprox(model.β, true_β, atol=0.5)
end

function test_BernoulliRegression_loglikelihood()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    p = logistic.(X * true_β)
    y = rand.(Bernoulli.(p))
    
    # Initialize and fit the model
    model = BernoulliRegression()
    fit!(model, X[:, 2:end], y)
    # check if the fitted coefficients are close to the true coefficients
    @test length(model.β) == 3
    @test isapprox(model.β, true_β, atol=0.5)
    
    # Check log likelihood
    loglik = SSM.loglikelihood(model, X[:, 2:end], y)
    @test loglik < 0


    # test loglikelihood on a single point
    # loglik = SSM.loglikelihood(model, X[1, 2:end], y[1])
    # @test loglik < 0
end

function test_BernoulliRegression_empty_model()
    model = BernoulliRegression()
    @test isempty(model.β)
end

function test_BernoulliRegression_intercept()
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    p = logistic.(X * true_β)
    y = rand.(Bernoulli.(p))
    
    model = BernoulliRegression(include_intercept=false)
    fit!(model, X[:, 2:end], y)
    @test length(model.β) == 2
    
    model_with_intercept = BernoulliRegression()
    fit!(model_with_intercept, X[:, 2:end], y)
    @test length(model_with_intercept.β) == 3
    @test isapprox(model_with_intercept.β, true_β, atol=0.5)
end

@testset "BernoulliRegression Tests" begin
    test_BernoulliRegression_fit()
    test_BernoulliRegression_loglikelihood()
    test_BernoulliRegression_empty_model()
    test_BernoulliRegression_intercept()
end

function test_PoissonRegression_fit()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    λ = exp.(X * true_β)
    y = rand.(Poisson.(λ))
    
    # Initialize and fit the model
    model = PoissonRegression()
    fit!(model, X[:, 2:end], y)

    # Check if the fitted coefficients are close to the true coefficients
    @test length(model.β) == 3
    @test isapprox(model.β, true_β, atol=0.5)
end

function test_PoissonRegression_loglikelihood()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    λ = exp.(X * true_β)
    y = rand.(Poisson.(λ))
    
    # Initialize and fit the model
    model = PoissonRegression()
    fit!(model, X[:, 2:end], y)
    # check if the fitted coefficients are close to the true coefficients
    @test length(model.β) == 3
    @test isapprox(model.β, true_β, atol=0.5)
    
    # Check log likelihood
    loglik = SSM.loglikelihood(model, X[:, 2:end], y)
    @test loglik < 0

    # test loglikelihood on a single point
    # loglik = SSM.loglikelihood(model, X[1, 2:end], y[1])
    # @test loglik < 0
end

function test_PoissonRegression_empty_model()
    model = PoissonRegression()
    @test isempty(model.β)
end

function test_PoissonRegression_intercept()
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    λ = exp.(X * true_β)
    y = rand.(Poisson.(λ))
    
    model = PoissonRegression(include_intercept=false)
    fit!(model, X[:, 2:end], y)
    @test length(model.β) == 2
    
    model_with_intercept = PoissonRegression()
    fit!(model_with_intercept, X[:, 2:end], y)
    @test length(model_with_intercept.β) == 3
    @test isapprox(model_with_intercept.β, true_β, atol=0.5)
end
@testset "PoissonRegression Tests" begin
    test_PoissonRegression_fit()
    test_PoissonRegression_loglikelihood()
    test_PoissonRegression_empty_model()
    test_PoissonRegression_intercept()
end

"""
Tests for Emissions.jl
"""

function test_GaussianEmission()
    # Initialize Gaussian Emission Model
    gaussian_emission = GaussianEmission([0.0, 0.0], [1.0 0.0; 0.0 1.0])
    # Check if parameters are initialized correctly
    @test gaussian_emission.μ == [0.0, 0.0]
    @test gaussian_emission.Σ == [1.0 0.0; 0.0 1.0]
    # Generate random data
    data = randn(100, 2)
    # Calculate log-likelihood
    ll = SSM.loglikelihood(gaussian_emission, data[1, :])
    # Check if log-likelihood is a scalar
    @test size(ll) == ()
    # Log-likelihood should be a negative float
    @test ll < 0.0
    # Check sample emission
    sample = SSM.sample_emission(gaussian_emission)
    @test length(sample) == 2
    # Update emission model
    γ = ones(100)
    SSM.updateEmissionModel!(gaussian_emission, data, γ)
    # Check if parameters are updated correctly
    @test gaussian_emission.μ ≈ mean(data, dims=1)'
    @test gaussian_emission.Σ ≈ cov(data, corrected=false)
end

function test_regression_emissions()
    # generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    true_σ² = 0.5
    # gaussian glm response
    y = X * true_β + rand(Normal(0., sqrt(true_σ²)), 1000)
    # poisson glm response
    y_poisson = rand.(Poisson.(exp.(X * true_β)))
    y_poisson = convert(Vector{Float64}, y_poisson)
    # bernoulli glm response
    y_bernoulli = rand.(Bernoulli.(logistic.(X * true_β)))
    y_bernoulli = convert(Vector{Float64}, y_bernoulli)
    # initialize emission models
    gaussian_emission = RegressionEmissions(GaussianRegression(;include_intercept=false))
    poisson_emission = RegressionEmissions(PoissonRegression(;include_intercept=false))
    bernoulli_emission = RegressionEmissions(BernoulliRegression(;include_intercept=false))
    # update emission models
    SSM.update_emissions_model!(gaussian_emission, X, y)
    SSM.update_emissions_model!(poisson_emission, X, y_poisson)
    SSM.update_emissions_model!(bernoulli_emission, X, y_bernoulli)
    # check if parameters are updated correctly
    @test isapprox(gaussian_emission.regression.β, true_β, atol=0.5)
    @test isapprox(gaussian_emission.regression.σ², true_σ², atol=0.1)
    @test isapprox(poisson_emission.regression.β, true_β, atol=0.5)
    @test isapprox(bernoulli_emission.regression.β, true_β, atol=0.5)
    # test the loglikelihood
    # ll_gaussian = SSM.loglikelihood(gaussian_emission, X[1, :], y[1])
    # ll_poisson = SSM.loglikelihood(poisson_emission, X[1, :], y_poisson[1])
    # ll_bernoulli = SSM.loglikelihood(bernoulli_emission, X[1, :], y_bernoulli[1])
    # @test ll_gaussian < 0
    # @test ll_poisson < 0
    # @test ll_bernoulli < 0
end


@testset "Emissions.jl Tests" begin
    test_GaussianEmission()
    test_regression_emissions()
end

"""
Tests for Utilities.jl
"""

function test_euclidean_distance()
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    @test SSM.euclidean_distance(x, y) == sqrt(27.0)
end

function test_kmeanspp_initialization()
    # Generate random data
    data = randn(100, 2)
    # Initialize centroids
    k_means = 3
    centroids = kmeanspp_initialization(data, k_means)
    # Check dimensions
    @test size(centroids) == (2, k_means)
end

function test_kmeans_clustering()
    # Generate random data
    data = randn(100, 2)
    # Initialize centroids
    k_means = 3
    centroids, labels = kmeans_clustering(data, k_means)
    # Check dimensions
    @test size(centroids) == (2, k_means)
    @test length(labels) == 100
    # Now test kmeans on a vector.
    data = randn(100,)
    centroids, labels = kmeans_clustering(data, k_means)
    # Check dimensions
    @test size(centroids) == (1, k_means)
    @test length(labels) == 100
end

function test_block_tridgm()
    # Test with minimal block sizes
    super = [rand(1, 1) for i in 1:1]
    sub = [rand(1, 1) for i in 1:1]
    main = [rand(1, 1) for i in 1:2]
    A = block_tridgm(main, super, sub)
    @test size(A) == (2, 2)
    @test A[1, 1] == main[1][1, 1]
    @test A[2, 2] == main[2][1, 1]
    @test A[1, 2] == super[1][1, 1]
    @test A[2, 1] == sub[1][1, 1]

    # Test with 2x2 blocks and a larger matrix
    super = [rand(2, 2) for i in 1:9]
    sub = [rand(2, 2) for i in 1:9]
    main = [rand(2, 2) for i in 1:10]
    A = block_tridgm(main, super, sub)
    @test size(A) == (20, 20)

    # Check some blocks in the matrix
    for i in 1:10
        @test A[(2i-1):(2i), (2i-1):(2i)] == main[i]
        if i < 10
            @test A[(2i-1):(2i), (2i+1):(2i+2)] == super[i]
            @test A[(2i+1):(2i+2), (2i-1):(2i)] == sub[i]
        end
    end

    # Test with integer blocks
    super = [rand(Int, 2, 2) for i in 1:9]
    sub = [rand(Int, 2, 2) for i in 1:9]
    main = [rand(Int, 2, 2) for i in 1:10]
    A = block_tridgm(main, super, sub)
    @test size(A) == (20, 20)
    for i in 1:10
        @test A[(2i-1):(2i), (2i-1):(2i)] == main[i]
        if i < 10
            @test A[(2i-1):(2i), (2i+1):(2i+2)] == super[i]
            @test A[(2i+1):(2i+2), (2i-1):(2i)] == sub[i]
        end
    end
end

function test_interleave_reshape()
    # Test with valid data and dimensions
    data = collect(1:6)
    t = 2
    d = 3
    X = interleave_reshape(data, t, d)
    @test size(X) == (2, 3)
    @test X == [1 2 3; 4 5 6]

    # Test with another set of valid data and dimensions
    data = collect(1:12)
    t = 4
    d = 3
    X = interleave_reshape(data, t, d)
    @test size(X) == (4, 3)
    @test X == [1 2 3; 4 5 6; 7 8 9; 10 11 12]

    # Test with a longer set of data
    data = collect(1:20)
    t = 4
    d = 5
    X = interleave_reshape(data, t, d)
    @test size(X) == (4, 5)
    @test X == [1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20]

    # Test with float data
    data = collect(1.0:0.5:6.5)
    t = 4
    d = 3
    X = interleave_reshape(data, t, d)
    @test size(X) == (4, 3)
    @test X == [1.0 1.5 2.0; 2.5 3.0 3.5; 4.0 4.5 5.0; 5.5 6.0 6.5]

    # Test with mismatched dimensions (should raise an error)
    data = collect(1:11)
    t = 2
    d = 5
    @test_throws ErrorException interleave_reshape(data, t, d)
end


@testset "Utilities.jl Tests" begin
    test_euclidean_distance()
    test_kmeanspp_initialization()
    test_kmeans_clustering()
    test_block_tridgm()
    test_interleave_reshape()
end

"""
Tests for Preprocessing.jl
"""

"""
Tests for MarkovRegression.jl
"""
function test_hmmglm_properties(model::SSM.hmmglm)
    # test basic properties of the model
    @test size(model.A) == (model.K, model.K)
    @test length(model.B) == model.K
    @test length(model.πₖ) == model.K
    @test sum(model.πₖ) ≈ 1.0
    @test sum(model.A, dims=2) ≈ ones(model.K)
end

function test_HMMGLM_initialization()
    # initialize models
    K = 3
    gaussian_model = SwitchingGaussianRegression(K=K)
    bernoulli_model = SwitchingBernoulliRegression(K=K)
    poisson_model = SwitchingPoissonRegression(K=K)
    # test properties
    test_hmmglm_properties(gaussian_model)
    test_hmmglm_properties(bernoulli_model)
    test_hmmglm_properties(poisson_model)
end

@testset "SwitchingRegression Tests" begin
    test_HMMGLM_initialization()
end