using StateSpaceDynamics
using Distributions
using ForwardDiff
using LinearAlgebra
using Random
using StatsFuns
using SpecialFunctions
using Test

Random.seed!(1234)

"""
Tests for MixtureModels.jl
"""

include("MixtureModels/GaussianMixtureModel.jl")
include("MixtureModels/PoissonMixtureModel.jl")
include("MixtureModels/MultinomialMixtureModel.jl")

@testset "MixtureModels.jl Tests" begin

    # --- Test GaussianMixtureModel ---
    @testset "GaussianMixtureModel Tests" begin
        # Initialize test models
        k = 3  # Number of clusters
        data_dim = 2  # Dimension of data points
        
        # Construct GMM
        standard_gmm = GaussianMixtureModel(k, data_dim)
        
        # Generate sample data
        n_samples = 100
        standard_data = StateSpaceDynamics.sample(standard_gmm, n_samples) 

        single_dimension_data = reshape(randn(1000), 1, 1000)

        # Test constructor method of GaussianMixtureModel
        test_GaussianMixtureModel_properties(standard_gmm, k, data_dim)

        # Test EM methods of the GaussianMixtureModel

        tester_set = [
            (standard_gmm, standard_data),                   # 2D data
            (GaussianMixtureModel(2, 1), single_dimension_data)  # 1D data 
        ]

        for (gmm, data) in tester_set
            testGaussianMixtureModel_EStep(gmm, data)
            testGaussianMixtureModel_MStep(gmm, data)
            testGaussianMixtureModel_fit(gmm, data)
            test_log_likelihood(gmm, data)
        end
    end

 
    # --- Test PoissonMixtureModel ---
    @testset "PoissonMixtureModel Tests" begin
        k = 3  # Number of clusters
        d = 2  # Dimension of Poisson emissions
        
        # Initialize PoissonMixtureModel with the new constructor format
        standard_pmm = PoissonMixtureModel(k, d)
        
        # Generate Poisson-distributed sample data
        temp_pmm = PoissonMixtureModel(k, d)  # Temporary model for sampling
        temp_pmm.emissions[1].λ .= 5.0  # Assign specific λ values to generate data
        temp_pmm.emissions[2].λ .= 10.0
        temp_pmm.emissions[3].λ .= 15.0
        temp_pmm.πₖ .= [1/3, 1/3, 1/3]  # Uniform mixing coefficients
        data = StateSpaceDynamics.sample(temp_pmm, 300)  # Generate sample data with 300 points
        
        # Conduct tests for general properties
        test_PoissonMixtureModel_properties(standard_pmm, k, d)
        
        # Run EM and log-likelihood tests
        tester_set = [(standard_pmm, data)]
        
        for (pmm, data) in tester_set
            testPoissonMixtureModel_EStep(pmm, data)
            testPoissonMixtureModel_MStep(pmm, data)
            testPoissonMixtureModel_fit(pmm, data)
            test_log_likelihood(pmm, data)
        end
    end

    # --- Test MultinomialMixtureModel ---
    @testset "MultinomialMixtureModel Tests" begin
        k, n_trials, n_categories = 3, 5, 4  # 3 components, 5 trials, 4 categories
    
        # Initialize a MultinomialMixtureModel
        mmm = MultinomialMixtureModel(k, n_trials, n_categories)
        
        # Generate sample data
        n_samples = 100
        sample_data = StateSpaceDynamics.sample(mmm, n_samples)
    
        # Verify the shape of the sample data
        @test size(sample_data) == (n_categories, n_samples)
    
        # Verify the shape of each emission's probability vector `p`
        @test all(size(mmm.emissions[i].p) == (n_categories, 1) for i in 1:k)
        
        # Run all tests for E-Step, M-Step, log-likelihood, and fitting process
        test_MultinomialMixtureModel_EStep(mmm, sample_data)
        test_MultinomialMixtureModel_MStep(mmm, sample_data)
        test_MultinomialMixtureModel_fit!(mmm, sample_data)
        test_log_likelihood(mmm, sample_data)
    end
    

    # --- Test for Mixture Models General Properties ---
    @testset "General Properties of Mixture Models" begin
        # Validate general properties for GMM, PMM, and MMM
        # GMM
        k, data_dim = 3, 2
        gmm = GaussianMixtureModel(k, data_dim)
        test_GaussianMixtureModel_properties(gmm, k, data_dim)

        # PMM
        k, d = 3, 2
        pmm = PoissonMixtureModel(k, d)
        test_PoissonMixtureModel_properties(pmm, k, d)

        # MMM
        n_trials, n_categories = 5, 4
        mmm = MultinomialMixtureModel(k, n_trials, n_categories)
        test_MultinomialMixtureModel_properties(mmm, k, n_trials, n_categories)
    end

end

"""
Tests for HiddenMarkovModels.jl
"""

include("HiddenMarkovModels/HiddenMarkovModels.jl")

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

include("LDS/LDS.jl")

@testset "LDS.jl Tests" begin
    test_LDS_with_params()
    test_LDS_without_params()
    test_LDS_E_Step()
    test_LDS_M_Step!()
    test_LDS_EM()
    test_direct_smoother()
    test_LDS_gradient()
    test_LDS_Hessian()
end

"""
Tests for Regression.jl
""" 

include("Regression/GaussianRegression.jl")

@testset "GaussianRegression Tests" begin
    test_GaussianRegression_fit()
    test_GaussianRegression_loglikelihood()
    test_GaussianRegression_default_model()
    test_GaussianRegression_intercept()
    test_Gaussian_ll_gradient()
end

include("Regression/BernoulliRegression.jl")

@testset "BernoulliRegression Tests" begin
    test_BernoulliRegression_fit()
    test_BernoulliRegression_loglikelihood()
    test_BernoulliRegression_empty_model()
    test_BernoulliRegression_intercept()
    test_Bernoulli_ll_gradient()
end

include("Regression/PoissonRegression.jl")

@testset "PoissonRegression Tests" begin
    test_PoissonRegression_fit()
    test_PoissonRegression_loglikelihood()
    test_PoissonRegression_empty_model()
    test_PoissonRegression_intercept()
    test_Poisson_ll_gradient()
end

"""
Tests for Emissions.jl
"""

include("Emissions/Emissions.jl")

@testset "Emissions.jl Tests" begin
    test_GaussianEmission()
    test_regression_emissions()
end

"""
Tests for Utilities.jl
"""

include("Utilities/Utilities.jl")

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

include("Preprocessing/Preprocessing.jl")

@testset "PPCA Tests" begin
    test_PPCA_with_params()
    test_PPCA_without_params()
    test_PPCA_E_and_M_Step()
    test_PPCA_fit()
end

"""
Tests for MarkovRegression.jl
"""

include("MarkovRegression/MarkovRegression.jl")

@testset "SwitchingRegression Tests" begin
    test_HMMGLM_initialization()
end
