# Test general properties of MultinomialMixtureModel
function test_MultinomialMixtureModel_properties(
    mmm::MultinomialMixtureModel, k::Int, n_trials::Int, n_categories::Int
)
    @test mmm.k == k  # num of clusters
    @test mmm.n == n_trials  # num of trials
    @test length(mmm.πₖ) == k  
    @test all(π -> π >= 0.0 && π <= 1.0, mmm.πₖ)  # Ensure all mixing coefficients are valid probabilities
    @test isapprox(sum(mmm.πₖ), 1.0, atol=1e-6)  # Ensure mixing coefficients sum to 1
    @test all(size(mmm.emissions[i].p) == (n_categories, 1) for i in 1:k)  
end

# Test E-Step 
function test_MultinomialMixtureModel_EStep(mmm::MultinomialMixtureModel, data::Matrix{Int})
    k = mmm.k
    n_samples = size(data, 2) 

    # Run E-Step
    class_probabilities = E_Step(mmm, data)

    @test size(class_probabilities) == (k, n_samples)  # Check class probabilities dimensions

    # Check if column sums are close to 1 
    @test all(x -> isapprox(x, 1.0, atol=1e-6), sum(class_probabilities, dims=1))

    # Verify general properties of the model
    n_categories = size(data, 1)
    test_MultinomialMixtureModel_properties(mmm, k, mmm.n, n_categories)
end

# Test M-Step
function test_MultinomialMixtureModel_MStep(mmm::MultinomialMixtureModel, data::Matrix{Int})
    k = mmm.k

    # Run E-Step to calculate class probabilities
    class_probabilities = E_Step(mmm, data)

    # Run M-Step to update model parameters
    M_Step!(mmm, data, class_probabilities)

    # Verify general properties of the updated model
    n_categories = size(data, 1)
    test_MultinomialMixtureModel_properties(mmm, k, mmm.n, n_categories)
end

# Test the fitting process for column-based MultinomialMixtureModel
function test_MultinomialMixtureModel_fit!(mmm::MultinomialMixtureModel, data::Matrix{Int})
    k = mmm.k
    n_categories = size(data, 1)

    log_likelihoods = fit!(mmm, data; maxiter=50, tol=1e-3)

    # Check if log-likelihoods are monotonically increasing
    @test all(diff(log_likelihoods) .≥ 0)

    # Ensure model properties hold after fitting
    test_MultinomialMixtureModel_properties(mmm, k, mmm.n, n_categories)

    # Ensure that the sum of mixing coefficients is approximately 1
    @test isapprox(sum(mmm.πₖ), 1.0, atol=1e-6)

    # Ensure that the final log-likelihood is greater than or equal to the initial log-likelihood
    final_ll = last(log_likelihoods)
    initial_ll = log_likelihood(mmm, data)
    @test final_ll ≥ initial_ll
end

# Test log-likelihood function
function test_log_likelihood(mmm::MultinomialMixtureModel, data::Matrix{Int})
    initial_ll = log_likelihood(mmm, data)

    # Ensure log-likelihood is a finite scalar
    @test initial_ll isa Float64  
    @test isfinite(initial_ll)  

    # Initialize probabilities for MMM components with normalized values
    for i in 1:mmm.k
        mmm.emissions[i].p .= reshape(normalize(rand(length(mmm.emissions[i].p)), 1), length(mmm.emissions[i].p), 1)
    end
    mmm.πₖ .= fill(1 / mmm.k, mmm.k)  # Uniform initialization for mixing coefficients

    # Test if log-likelihood increases or remains stable during fitting
    ll_prev = -Inf
    for i in 1:10
        fit!(mmm, data; maxiter=1, tol=1e-3, initialize_kmeans=false)
        ll = log_likelihood(mmm, data)
        @test ll > ll_prev || isapprox(ll, ll_prev, atol=1e-6)  # Ensure log-likelihood does not decrease
        ll_prev = ll
    end
end
