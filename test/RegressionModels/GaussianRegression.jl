function GaussianRegression_simulation(; include_intercept::Bool=true)
    # Generate synthetic data
    n = 1000
    X = randn(n, 2)  # Remove intercept from data generation
    true_β = [-1.2, 2.3]
    if include_intercept
        true_β = vcat(0.5, true_β)
    end
    true_β = reshape(true_β, :, 1)
    true_covariance = reshape([0.25], 1, 1)
    
    # Generate y with or without intercept
    X_with_intercept = include_intercept ? hcat(ones(n), X) : X
    y = X_with_intercept * true_β + rand(MvNormal(zeros(1), true_covariance), n)'
    
    return X, y, true_β, true_covariance, n
end

function test_GaussianRegression_initialization()
    # Test with default parameters
    model = GaussianRegressionEmission(input_dim=2, output_dim=1)
    
    @test model.input_dim == 2
    @test model.output_dim == 1
    @test model.include_intercept == true
    @test size(model.β) == (3, 1)  # input_dim + 1 (intercept) × output_dim
    @test size(model.Σ) == (1, 1)
    @test model.λ == 0.0
    
    # Test without intercept
    model_no_intercept = GaussianRegressionEmission(
        input_dim=2,
        output_dim=1,
        include_intercept=false
    )
    @test size(model_no_intercept.β) == (2, 1)
end

function test_GaussianRegression_fit()
    X, y, true_β, true_covariance, n = GaussianRegression_simulation()
    
    model = GaussianRegressionEmission(input_dim=2, output_dim=1)
    fit!(model, X, y)
    
    # Check if the fitted coefficients are close to the true coefficients
    @test isapprox(model.β, true_β, atol=0.5)
    @test isposdef(model.Σ)
    @test isapprox(model.Σ, true_covariance, atol=0.1)
    
    # Test with weights
    w = ones(n)
    fit!(model, X, y, w)
    @test isapprox(model.β, true_β, atol=0.5)
end

function test_GaussianRegression_loglikelihood()
    X, y, true_β, true_covariance, n = GaussianRegression_simulation()
    
    model = GaussianRegressionEmission(input_dim=2, output_dim=1)
    fit!(model, X, y)
    
    # Test full dataset loglikelihood
    ll = StateSpaceDynamics.loglikelihood(model, X, y)
    @test length(ll) == n
    @test all(isfinite.(ll))
    
    # Test single observation
    single_ll = StateSpaceDynamics.loglikelihood(model, X[1:1,:], y[1:1,:])
    @test length(single_ll) == 1
    @test isfinite(single_ll[1])
    
    # Test with weights
    w = ones(n)
    weighted_ll = StateSpaceDynamics.loglikelihood(model, X, y, w)
    @test length(weighted_ll) == n
    @test all(isfinite.(weighted_ll))
end

function test_GaussianRegression_sample()
    model = GaussianRegressionEmission(input_dim=2, output_dim=1)
    
    # Test single sample
    X_test = randn(1, 2)
    sample_single = StateSpaceDynamics.sample(model, X_test)
    @test size(sample_single) == (1, 1)
    
    # Test multiple samples
    X_test = randn(10, 2)
    samples = StateSpaceDynamics.sample(model, X_test)
    @test size(samples) == (10, 1)
end

function test_GaussianRegression_optimization()
    X, y, true_β, true_covariance, n = GaussianRegression_simulation()
    
    model = GaussianRegressionEmission(
        input_dim=2,
        output_dim=1,
        λ=0.1  # Add regularization for testing
    )
    
    # Test optimization problem creation
    opt_problem = StateSpaceDynamics.create_optimization(model, X, y)
    
    # Test objective function
    β_vec = vec(model.β)
    obj_val = StateSpaceDynamics.objective(opt_problem, β_vec)
    @test isfinite(obj_val)
    
    # Test gradient calculation
    G = similar(β_vec)
    StateSpaceDynamics.objective_gradient!(G, opt_problem, β_vec)
    @test length(G) == length(β_vec)
    @test all(isfinite.(G))
    
    # Compare with ForwardDiff
    grad_fd = ForwardDiff.gradient(β -> StateSpaceDynamics.objective(opt_problem, β), β_vec)
    @test isapprox(G, grad_fd, rtol=1e-5)
end

function test_GaussianRegression_regularization()
    X, y, true_β, true_covariance, n = GaussianRegression_simulation()
    
    # Test model with different regularization values
    λ_values = [0.0, 0.1, 1.0]
    
    for λ in λ_values
        model = GaussianRegressionEmission(
            input_dim=2,
            output_dim=1,
            λ=λ
        )
        
        fit!(model, X, y)
        
        # Higher regularization should result in smaller coefficients
        if λ > 0
            @test norm(model.β) < norm(true_β)
        end
    end
end