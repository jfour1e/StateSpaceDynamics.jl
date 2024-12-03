using StateSpaceDynamics
using Distributions
using ForwardDiff
using LinearAlgebra
using Optim
using Random
using StatsFuns
using SpecialFunctions
using Test
using Aqua
using CSV
using DataFrames
using MAT

"""
Package Wide Tests
"""

@testset "Package Wide Tests" begin
    Aqua.test_all(StateSpaceDynamics; ambiguities=false)
    @test isempty(Test.detect_ambiguities(StateSpaceDynamics))
end

include("helper_functions.jl")

"""
Tests for LDS.jl
"""

include("LinearDynamicalSystems//GaussianLDS.jl")

@testset "GaussianLDS Tests" begin
    @testset "Constructor Tests" begin
        test_lds_with_params()
        test_lds_without_params()
    end
    @testset "Smoother tests" begin
        test_Gradient()
        test_Hessian()
        test_smooth()
    end
    @testset "EM tests" begin
        test_estep()
        # test when ntrials=1
        test_initial_observation_parameter_updates()
        test_state_model_parameter_updates()
        test_obs_model_params_updates()
        # test when ntrials>1
        test_initial_observation_parameter_updates(3)
        test_state_model_parameter_updates(3)
        test_obs_model_params_updates(3)
        # test fit method using n=1 and n=3
        test_EM()
        test_EM(3)
    end
end

"""
Tests for PoissonLDS.jl
"""

include("LinearDynamicalSystems//PoissonLDS.jl")

@testset "PoissonLDS Tests" begin
    @testset "Constructor Tests" begin
        test_PoissonLDS_with_params()
        test_poisson_lds_without_params()
    end
    @testset "Smoother Tests" begin
        test_Gradient()
        test_Hessian()
        test_smooth()
    end
    @testset "EM Tests" begin
        test_parameter_gradient()
        # test when ntrials=1
        test_initial_observation_parameter_updates()
        test_state_model_parameter_updates()
        # test when n_trials>1
        test_initial_observation_parameter_updates(3)
        test_state_model_parameter_updates(3)
        # test fit method using 1 trial and three trials
        test_EM()
        test_EM(3)
        # test resutlts are same as matlab code
        test_EM_matlab()
    end
end

"""
Tests for Switching Regression Models
"""

include("HiddenMarkovModels/GaussianHMM.jl")

@testset "GaussianHMM Tests" begin
    test_SwitchingGaussian_fit()
    test_SwitchingGaussian_SingleState_fit()
    test_kmeans_init()
    test_trialized_GaussianHMM()
end

include("HiddenMarkovModels/SwitchingGaussianRegression.jl")

@testset "Switching Gaussian Regression Tests" begin
    test_SwitchingGaussianRegression_fit()
    test_SwitchingGaussianRegression_SingleState_fit()
    test_trialized_SwitchingGaussianRegression()
end

include("HiddenMarkovModels/SwitchingPoissonRegression.jl")

@testset "Switching Poisson Regression Tests" begin
    test_SwitchingPoissonRegression_fit()
    test_trialized_SwitchingPoissonRegression()
end

include("HiddenMarkovModels/SwitchingBernoulliRegression.jl")

@testset "Switching Bernoulli Regression Tests" begin
    test_SwitchingBernoulliRegression()
    test_trialized_SwitchingBernoulliRegression()
end

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

end

"""
Tests for RegressionModels.jl
"""

include("RegressionModels/GaussianRegression.jl")

@testset "GaussianRegression Tests" begin
    test_GaussianRegression_initialization()
    test_GaussianRegression_loglikelihood()
    test_GaussianRegression_fit()
    test_GaussianRegression_sample()
    test_GaussianRegression_optimization()
    test_GaussianRegression_sklearn()
end

include("RegressionModels/BernoulliRegression.jl")

@testset "BernoulliRegression Tests" begin
    test_BernoulliRegression_initialization()
    test_BernoulliRegression_loglikelihood()
    test_BernoulliRegression_fit()
    test_BernoulliRegression_sample()
    test_BernoulliRegression_optimization()
    test_BernoulliRegression_sklearn()
end

include("RegressionModels/PoissonRegression.jl")

@testset "PoissonRegression Tests" begin
    test_PoissonRegression_initialization()
    test_PoissonRegression_loglikelihood()
    test_PoissonRegression_fit()
    test_PoissonRegression_sample()
    test_PoissonRegression_optimization()
    test_PoissonRegression_sklearn()
end

include("RegressionModels/AutoRegression.jl")

# @testset "AutoRegression Tests" begin
#     test_AutoRegression_loglikelihood()
#     # test_AutoRegression_Σ()
#     # test_AutoRegression_constructor()
#     test_AutoRegression_standard_fit()
#     test_AutoRegression_regularized_fit()
# end

"""
Tests for Utilities.jl
"""

include("Utilities/Utilities.jl")

@testset "Utilities.jl Tests" begin
    test_euclidean_distance()
    test_kmeanspp_initialization()
    test_kmeans_clustering()
    test_block_tridgm()
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