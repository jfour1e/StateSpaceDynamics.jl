# Test general properties of GaussianMixtureModel
function test_GaussianMixtureModel_properties(gmm::GaussianMixtureModel, k::Int, data_dim::Int)
    @test gmm.k == k
    @test length(gmm.emissions) == k

    for emission in gmm.emissions
        @test length(emission.μ) == data_dim
        @test size(emission.Σ) == (data_dim, data_dim)
        @test ishermitian(emission.Σ)

        # Check positive definiteness of covariance matrix
        eigvals_Σ = eigvals(emission.Σ)
        @test minimum(eigvals_Σ) > 0  # Positive definite check
        @test all(isfinite, eigvals_Σ)  # Ensure finite eigenvalues
    end

    @test length(gmm.πₖ) == k
    @test sum(gmm.πₖ) ≈ 1.0
    @test all(p -> p > 0, gmm.πₖ)
end

# Test E-Step
function testGaussianMixtureModel_EStep(gmm::GaussianMixtureModel, data::Matrix{<:Real})
    k = gmm.k
    data_dim = size(data, 1) 

    # Run E_Step
    class_probabilities = StateSpaceDynamics.E_Step(gmm, data)

    # Check dimensions
    @test size(class_probabilities) == (k, size(data, 2))

    # Check if the column sums are close to 1 
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(class_probabilities, dims=1))

    @test all(p -> p >= 0.0 && p <= 1.0, class_probabilities)
    
    test_GaussianMixtureModel_properties(gmm, k, data_dim)
end

# Test M-Step 
function testGaussianMixtureModel_MStep(gmm::GaussianMixtureModel, data::Matrix{<:Real})
    k = gmm.k
    data_dim = size(data, 1) 

    class_probabilities = StateSpaceDynamics.E_Step(gmm, data)

    # Run M-Step
    StateSpaceDynamics.M_Step!(gmm, data, class_probabilities)

    test_GaussianMixtureModel_properties(gmm, k, data_dim)
end

# Test the fitting process
function testGaussianMixtureModel_fit(gmm::GaussianMixtureModel, data::Matrix{<:Real})
    k = gmm.k
    data_dim = size(data, 1)  # Adjust to column-based

    # Run fit!
    fit!(gmm, data; maxiter=10, tol=1e-6)

    test_GaussianMixtureModel_properties(gmm, k, data_dim)
end

# Test log-likelihood
function test_log_likelihood(gmm::GaussianMixtureModel, data::Matrix{<:Real})
    # Calculate log-likelihood
    ll = log_likelihood(gmm, data)

    # Check if log-likelihood is a finite scalar
    @test ll isa Float64
    @test isfinite(ll)
    @test ll < 0.0

    # Log-likelihood should monotonically increase with iterations
    log_likelihoods = fit!(gmm, data; maxiter=10, tol=1e-3, initialize_kmeans=false)

    @test all(log_likelihoods[i] <= log_likelihoods[i + 1] for i in 1:length(log_likelihoods) - 1)

    # Ensure each log-likelihood in the sequence matches expectations
    for i in 1:length(log_likelihoods)
        ll = log_likelihood(gmm, data)
        @test ll isa Float64
        @test isfinite(ll)
        @test ll < 0.0
    end
end

# Test monotonicity of GMM fit 
function test_GaussianMixtureModel_fit_monotonicity(gmm::GaussianMixtureModel, data::Matrix{<:Real})
    # Run fit! with initial k-means initialization
    log_likelihoods = fit!(gmm, data; maxiter=100, tol=1e-3, initialize_kmeans=true)

    # Check if log-likelihoods are monotonically increasing
    @test all(log_likelihoods[i] <= log_likelihoods[i + 1] for i in 1:length(log_likelihoods) - 1)

    # Check if the final log-likelihood change is below tolerance
    if length(log_likelihoods) > 1
        @test abs(log_likelihoods[end] - log_likelihoods[end-1]) < 1e-3
    end
end
