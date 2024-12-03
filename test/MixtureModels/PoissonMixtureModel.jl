# Test general properties of PoissonMixtureModel (compatible with Univariate and Multivariate PMMs)
function test_PoissonMixtureModel_properties(pmm::PoissonMixtureModel, k::Int, dim::Int)
    @test pmm.k == k
    @test length(pmm.πₖ) == k
    @test isapprox(sum(pmm.πₖ), 1.0, atol=1e-6)  # Ensure mixing coefficients sum to 1
    for i in 1:k
        @test length(pmm.emissions[i].λ) == dim  # Check the dimension of each emission's parameter vector
    end
end

# Test E-Step 
function testPoissonMixtureModel_EStep(pmm::PoissonMixtureModel, data::Matrix{Int})
    k = pmm.k
    n_samples = size(data, 2)  # Use columns for data samples

    # Run E-Step
    class_probabilities = StateSpaceDynamics.E_Step(pmm, data)

    @test size(class_probabilities) == (k, n_samples) 

    # Check if column sums are close to 1
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(class_probabilities, dims=1))

    # Test general properties of the model
    dim = size(data, 1) 
    test_PoissonMixtureModel_properties(pmm, k, dim)
end

# Test M-Step
function testPoissonMixtureModel_MStep(pmm::PoissonMixtureModel, data::Matrix{Int})
    k = pmm.k
    dim = size(data, 1) 

    # Run E-Step to obtain class probabilities
    class_probabilities = StateSpaceDynamics.E_Step(pmm, data)

    # Run M-Step
    StateSpaceDynamics.M_Step!(pmm, data, class_probabilities)

    # Ensure the model properties hold after M-Step
    test_PoissonMixtureModel_properties(pmm, k, dim)
end

# Test fit! function
function testPoissonMixtureModel_fit(pmm::PoissonMixtureModel, data::Matrix{Int})
    k = pmm.k
    dim = size(data, 1)  

    # Run fit! with convergence check
    log_likelihoods = fit!(pmm, data; maxiter=50, tol=1e-3, initialize_kmeans=true)

    # Check if log-likelihoods are monotonically increasing
    @test all(diff(log_likelihoods) .≥ 0)

    # Ensure model properties hold after fitting
    test_PoissonMixtureModel_properties(pmm, k, dim)

    final_ll = last(log_likelihoods)
    initial_ll = log_likelihood(pmm, data)
    @test final_ll ≥ initial_ll
end

# Test log-likelihood function 
function test_log_likelihood(pmm::PoissonMixtureModel, data::Matrix{Int})
    # Calculate initial log-likelihood
    initial_ll = log_likelihood(pmm, data)

    @test initial_ll isa Float64 # Ensure log-likelihood is a scalar

    # Initialize λₖ_matrix
    λₖ_matrix = kmeanspp_initialization(Float64.(data), pmm.k)  # Returns a matrix of shape (d, k)

    for i in 1:pmm.k
        pmm.emissions[i].λ = λₖ_matrix[:, i] 
    end

    # Perform log-likelihood testing after fitting the model
    prev_ll = -Inf
    for i in 1:10
        fit!(pmm, data; maxiter=1, tol=1e-3, initialize_kmeans=false)
        current_ll = log_likelihood(pmm, data)
        @test current_ll > prev_ll || isapprox(current_ll, prev_ll; atol=1e-6)
        prev_ll = current_ll
    end
end
