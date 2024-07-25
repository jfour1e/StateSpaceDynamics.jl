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
    gaussian_model = SwitchingGaussianRegression(num_features=2, num_targets=1, K=K)
    bernoulli_model = SwitchingBernoulliRegression(K=K)
    poisson_model = SwitchingPoissonRegression(K=K)
    # test properties
    test_hmmglm_properties(gaussian_model)
    test_hmmglm_properties(bernoulli_model)
    test_hmmglm_properties(poisson_model)
end