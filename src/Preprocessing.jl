export ProbabilisticPCA, PoissonPCA, E_Step, M_Step!, loglikelihood, fit!

"""
    mutable struct ProbabilisticPCA 

Probabilistic PCA model from Bishop's Pattern Recognition and Machine Learning.

# Fields:
    W: Weight matrix that maps from latent space to data space.
    σ²: Noise variance
    μ: Mean of the data
    k: Number of latent dimensions
    D: Number of features
    z: Latent variables
"""
mutable struct ProbabilisticPCA
    W::Matrix{<:AbstractFloat} # weight matrix
    σ²::AbstractFloat # noise variance
    μ::Matrix{<:AbstractFloat} # mean of the data
    k::Int # number of latent dimensions
    D::Int # dimension of the data
    z::Matrix{<:AbstractFloat} # latent variables
end

"""
#     ProbabilisticPCA(;W::Matrix{<:AbstractFloat}, σ²:: <: AbstractFloat, μ::Matrix{<:AbstractFloat}, k::Int, D::Int)

# Constructor for ProbabilisticPCA model.

# # Args:
# - `W::Matrix{<:AbstractFloat}`: Weight matrix that maps from latent space to data space.
# - `σ²:: <: AbstractFloat`: Noise variance
# - `μ::Matrix{<:AbstractFloat}`: Mean of the data
# - `k::Int`: Number of latent dimensions
# - `D::Int`: Number of features

# # Example:
# ```julia
# # PPCA with unknown parameters
# ppca = ProbabilisticPCA(k=1, D=2)
# # PPCA with known parameters
# ppca = ProbabilisticPCA(W=rand(2, 1), σ²=0.1, μ=rand(2), k=1, D=2)
# ```
# """
function ProbabilisticPCA(;
    W::Matrix{<:AbstractFloat}=Matrix{Float64}(undef, 0, 0),
    μ::Matrix{<:AbstractFloat}=Matrix{Float64}(undef, 0, 0),
    σ²::AbstractFloat=0.0,
    k::Int,
    D::Int,
)
    # if W is not provided, initialize it randomly
    W = isempty(W) ? rand(D, k) / sqrt(k) : W
    # if σ² is not provided, initialize it randomly
    σ² = σ² === 0.0 ? abs(rand()) : σ²
    # add empty z
    z = Matrix{Float64}(undef, 0, 0)
    return ProbabilisticPCA(W, σ², μ, k, D, z)
end

"""
    E_Step(ppca::ProbabilisticPCA, X::Matrix{<:AbstractFloat})

Expectation step of the EM algorithm for PPCA. See Bishop's Pattern Recognition and Machine Learning for more details.

# Args:
- `ppca::ProbabilisticPCA`: PPCA model
- `X::Matrix{<:AbstractFloat}`: Data matrix

# Examples:
```julia
ppca = ProbabilisticPCA(K=1, D=2)
E_Step(ppca, rand(10, 2))
```
"""
function E_Step(ppca::ProbabilisticPCA, X::Matrix{<:Real})
    # get dims
    N, _ = size(X)
    # preallocate E_zz and E_zz
    E_z = zeros(N, ppca.k)
    E_zz = zeros(N, ppca.k, ppca.k)
    # calculate M
    M = ppca.W' * ppca.W + (ppca.σ² * I(ppca.k))
    M_inv = cholesky(M).U \ (cholesky(M).L \ I)
    # calculate E_z and E_zz
    for i in 1:N
        E_z[i, :] = M_inv * ppca.W' * (X[i, :] - ppca.μ')
        E_zz[i, :, :] = (ppca.σ² * M_inv) + (E_z[i, :] * E_z[i, :]')
    end
    return E_z, E_zz
end

"""
    M_Step!(model::ProbabilisticPCA, X::Matrix{<:AbstractFloat}, E_z::Matrix{<:AbstractFloat}, E_zz::Array{<:AbstractFloat, 3}
Maximization step of the EM algorithm for PPCA. See Bishop's Pattern Recognition and Machine Learning for more details.

# Args:
- `model::ProbabilisticPCA`: PPCA model
- `X::Matrix{<:AbstractFloat}`: Data matrix
- `E_z::Matrix{<:AbstractFloat}`: E[z]
- `E_zz::Matrix{<:AbstractFloat}`: E[zz']

# Examples:
```julia
ppca = ProbabilisticPCA(K=1, D=2)
E_z, E_zz = E_Step(ppca, rand(10, 2))
M_Step!(ppca, rand(10, 2), E_z, E_zzᵀ)
```
"""
function M_Step!(
    ppca::ProbabilisticPCA, X::Matrix{<:Real}, E_z::AbstractArray, E_zz::AbstractArray
)
    # get dims
    N, D = size(X)
    # update W and σ²
    running_sum_W = zeros(D, ppca.k)
    running_sum_σ² = 0.0
    WW = ppca.W' * ppca.W
    for i in 1:N
        running_sum_W += (X[i, :] - ppca.μ') * E_z[i, :]'
        running_sum_σ² +=
            sum((X[i, :] - ppca.μ') .^ 2) -
            sum((2 * E_z[i, :]' * ppca.W' * (X[i, :] - ppca.μ'))) + tr(E_zz[i, :, :] * WW)
    end
    ppca.z = E_z
    ppca.W = running_sum_W * pinv(sum(E_zz; dims=1)[1, :, :])
    return ppca.σ² = running_sum_σ² / (N * D)
end

"""
    loglikelihood(model::ProbabilisticPCA, X::Matrix{<:AbstractFloat})
    
Calculate the log-likelihood of the data given the PPCA model.

# Args:
- `model::ProbabilisticPCA`: PPCA model
- `X::Matrix{<:AbstractFloat}`: Data matrix

# Examples:
```julia
ppca = ProbabilisticPCA(K=1, D=2)
loglikelihood(ppca, rand(10, 2))
```
"""
function loglikelihood(ppca::ProbabilisticPCA, X::Matrix{<:Real})
    # get dims
    N, D = size(X)
    # calculate C and S
    C = ppca.W * ppca.W' + (ppca.σ² * I(D))
    # center the data
    X = X .- ppca.μ
    S = sum([X[i, :] * X[i, :]' for i in axes(X, 1)]) / N
    # calculate log-likelihood
    ll = -(N / 2) * (D * log(2 * π) + logdet(C) + tr(pinv(C) * S))
    return ll
end

"""
    fit!(model::ProbabilisticPCA, X::Matrix{<:AbstractFloat}, max_iter::Int=100, tol::AbstractFloat=1e-6)

Fit the PPCA model to the data using the EM algorithm.

# Args:
- `model::ProbabilisticPCA`: PPCA model
- `X::Matrix{<:AbstractFloat}`: Data matrix
- `max_iter::Int`: Maximum number of iterations
- `tol::AbstractFloat`: Tolerance for convergence

# Examples:
```julia
ppca = ProbabilisticPCA(K=1, D=2)
fit!(ppca, rand(10, 2))
```
"""
function fit!(
    ppca::ProbabilisticPCA, X::Matrix{Float64}, max_iters::Int=100, tol::Float64=1e-6
)
    # initialize the μ if not done 
    ppca.μ = mean(X; dims=1)
    # initiliaze the log-likelihood
    lls = []
    prev_ll = -Inf
    prog = Progress(max_iters; desc="Fitting Probabilistic PCA...")

    for i in 1:max_iters
        E_z, E_zz = E_Step(ppca, X)
        M_Step!(ppca, X, E_z, E_zz)

        ll = loglikelihood(ppca, X)
        push!(lls, ll)
        next!(prog)

        if abs(ll - prev_ll) < tol
            finish!(prog)
            return lls
        end

        prev_ll = ll  # Update prev_ll for the next iteration
    end

    finish!(prog)
    return lls
end


"""
    mutable struct PoissonPCA

# Fields:
    W: Weight matrix that maps from latent space to data space.
    μ: Mean parameter for Poisson distribution.
    σ²: Variance of Gaussian prior.
    k: Number of latent dimensions.
    D: Number of features.
    z: Latent variables.
"""

mutable struct PoissonPCA
    W::Matrix{<:AbstractFloat} # Weight matrix
    μ::Matrix{<:AbstractFloat} 
    σ²::AbstractFloat 
    k::Int #latend dimension 
    D::Int # data dimension 
    z::Matrix{<:AbstractFloat} # Latent variables
end


"""
    PoissonPCA(; W::Matrix{<:AbstractFloat}, μ::Matrix{<:AbstractFloat}, σ²::AbstractFloat, k::Int, D::Int)

Constructor for PoissonPCA model.

# Args:
- `W`: Weight matrix that maps latent space to data space.
- `μ`: Mean parameter for the Poisson distribution.
- `σ²`: Variance of Gaussian prior.
- `k`: Number of latent dimensions.
- `D`: Number of features.
"""

function PoissonPCA(; 
    W::Matrix{<:AbstractFloat}=Matrix{Float64}(undef, 0, 0), 
    μ::Matrix{<:AbstractFloat}=Matrix{Float64}(undef, 0, 0), 
    σ²::AbstractFloat=1.0, 
    k::Int, 
    D::Int)
    
    # Initialize W if not provided
    W = isempty(W) ? rand(D, k) / sqrt(k) : W
    # Initialize μ if not provided
    μ = isempty(μ) ? rand(D) : μ
    # Initialize z as an empty matrix
    z = Matrix{Float64}(undef, 0, 0)
    return PoissonPCA(W, μ, σ², k, D, z)
end


"""
    E_Step(poisson_pca::PoissonPCA, X::Matrix{<:AbstractFloat})

Expectation step of the EM algorithm for PoissonPCA.

# Args:
- `poisson_pca`: PoissonPCA model.
- `X`: Data matrix.
"""

function E_Step(poisson_pca::PoissonPCA, X::Matrix{<:Real})
    M, N = poisson_pca.k, size(X, 2)
    Z_hat = zeros(M, N)
    posterior_covariances = Matrix{<:Real}[]

    for n in 1:N
        Z_n = zeros(M)
        function objective(Z)
            λ = exp.(poisson_pca.W * Z .+ poisson_pca.μ)
            poisson_log_likelihood = sum(X[:, n] .* log.(λ) .- λ .- logfactorial.(X[:, n]))
            log_prior = -0.5 * sum(Z .^ 2) / poisson_pca.σ²
            return -(poisson_log_likelihood + log_prior)
        end

        function gradient!(g, Z)
            λ = exp.(poisson_pca.W * Z .+ poisson_pca.μ)
            grad_likelihood = poisson_pca.W' * (X[:, n] .- λ)
            grad_prior = -Z / poisson_pca.σ²
            g .= -(grad_likelihood + grad_prior)
        end

        function hessian!(h, Z)
            λ = exp.(poisson_pca.W * Z .+ poisson_pca.μ)
            hessian_likelihood = -poisson_pca.W' * Diagonal(λ) * poisson_pca.W
            hessian_prior = -(1 / poisson_pca.σ²) * I(M)
            h .= -(hessian_likelihood + hessian_prior + 1e-5 * I(M))
        end

        result = optimize(objective, gradient!, hessian!, Z_n, Newton(), Optim.Options(g_tol=1e-6, iterations=10))
        Z_hat[:, n] = Optim.minimizer(result)

        hessian = hessian!(zeros(M, M), Z_hat[:, n])
        posterior_cov = inv(hessian .- 1e-5 * I(M))
        push!(posterior_covariances, posterior_cov)
    end

    poisson_pca.z = Z_hat
    return Z_hat, posterior_covariances
end
 

"""
    M_Step!(poisson_pca::PoissonPCA, X::Matrix{<:AbstractFloat}, Z_hat::Matrix{<:AbstractFloat}, posterior_covariances::Vector{<:Matrix{<:AbstractFloat}})

Maximization step of the EM algorithm for PoissonPCA.

# Args:
- `poisson_pca`: PoissonPCA model.
- `X`: Data matrix.
- `Z_hat`: Latent variable estimates.
- `posterior_covariances`: Covariances from E-step.
"""
function M_Step!(poisson_pca::PoissonPCA, X::Matrix{<:Real}, Z_hat::Matrix{<:Real}, posterior_covariances::Vector{<:Matrix{<:Real}})
    D, M = size(poisson_pca.W)
    N = size(X, 2)
    λ = exp.(poisson_pca.W * Z_hat .+ poisson_pca.μ)
    
    # Calculate gradients
    grad_W = (X - λ) * Z_hat' .+ poisson_pca.W * sum(posterior_covariances) / N
    grad_μ = sum(X - λ, dims=2)

    # Update W and μ
    poisson_pca.W .+= 0.001 * grad_W
    poisson_pca.μ .+= 0.001 * grad_μ
end


"""
    loglikelihood(poisson_pca::PoissonPCA, X::Matrix{<:AbstractFloat}, Z_hat::Matrix{<:AbstractFloat}, posterior_covariances::Vector{<:AbstractFloat})

Calculate the log-likelihood for PoissonPCA.
"""

function loglikelihood(poisson_pca::PoissonPCA, X::Matrix{<:Real}, Z_hat::Matrix{<:Real}, posterior_covariances::Vector{<:Matrix{<:Real}})
    M, N = size(Z_hat)
    μ = reshape(poisson_pca.μ, :)
    ll = 0.0

    for n in 1:N
        λ_n = exp.(poisson_pca.W * Z_hat[:, n] .+ μ)
        ll += sum(X[:, n] .* log.(λ_n) .- λ_n .- logfactorial.(X[:, n]))
        ll += -0.5 * sum(Z_hat[:, n] .^ 2) / poisson_pca.σ²


        hessian = hessian_log_joint(X[:, n], Z_hat[:, n], poisson_pca.W, μ, poisson_pca.σ²)
        ll -= 0.5 * logdet(hessian - 1e-5 * I(M))
    end

    return ll
end


"""
    fit!(poisson_pca::PoissonPCA, X::Matrix{Float64}, max_iter::Int=100, tol::Float64=1e-3)

Fit PoissonPCA to the data using EM algorithm.
"""

function fit!(poisson_pca::PoissonPCA, X::Matrix{Float64}, max_iter::Int=200, tol::Float64=1e-6)
    # initiliaze the log-likelihood and z_hats
    lls = []
    prev_ll = -Inf

    Z_hat = zeros(Float64, M, N) 
    
    prog = Progress(max_iters; desc="Fitting Poisson PCA...")

    for iter in 1:max_iter
        Z_hat, posterior_covariances = E_Step(X, W, μ, σ2)
        M_Step!(poisson_pca, X, Z_hat, posterior_covariances)

        ll = loglikelihood(poisson_pca, X, Z_hat, posterior_covariances)

        push!(lls, ll)
        next!(prog)

        # Check for convergence
        if abs(ll - prev_ll) < tol
            println("Convergence reached at iteration $iter")
            break
        end
        prev_ll = ll
    end

    finish!(prog)
    return lls
end
