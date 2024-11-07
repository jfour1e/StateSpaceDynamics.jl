export GaussianMixtureModel, PoissonMixtureModel, MultinomialMixtureModel, 
    fit!, log_likelihood, sample, E_Step, M_Step!, MixtureModel
    
"""
    sample(model::MixtureModel, n::Int)

Draw `n` samples from the given mixture model.

# Returns
- A matrix or vector of samples (with shape 'n' by 'data dimension'), depending on mixture model type. 

# Examples
```julia
# Gaussian Mixture Model example
gmm = GaussianMixtureModel(3, 2)  # Create a Gaussian Mixture Model with 3 clusters and 2-dimensional data
gmm_samples = sample(gmm, 100)  # Draw 100 samples from the Gaussian Mixture Model
```
"""
function sample(mixture_model::MixtureModel, n::Int) end



"""
    fit!(model::MixtureModel, data::AbstractMatrix; <keyword arguments>)

Fit the given mixture model to the data using the Expectation-Maximization (EM) algorithm. 

# Arguments
- `model::MixtureModel`: The mixture model to fit.
- `data::AbstractMatrix`: The data matrix where rows correspond to data points.
- `maxiter::Int=50`: The maximum number of iterations for the EM algorithm (default is 50).
- `tol::Float64=1e-3`: The convergence tolerance for the log-likelihood (default is 1e-3).
- `initialize_kmeans::Bool=true`: Whether to initialize the model parameters using KMeans (default is true).

# Returns
- The class probabilities (responsibilities) for each data point.

# Examples
```julia
gmm = GaussianMixtureModel(3, 2)  # Create a Gaussian Mixture Model with 3 clusters and 2-dimensional data
fit!(gmm, data)  # Fit the model to the data
```
"""
function fit!(gmm::MixtureModel, data::AbstractMatrix; maxiter::Int=50, tol::Float64=1e-3, initialize_kmeans::Bool=true) end




    """
        GaussianMixtureModel

    A Gaussian Mixture Model for clustering and density estimation.

    # Fields
    - `k::Int`: Number of clusters.
    - `emissions::Vector{GaussianEmission}``: vector of emission objects for each cluster (dimensions: k) Each emission object contains: 
        - `μₖ`: Mean of each cluster. 
        - `Σₖ`: Covariance matrix of each cluster.
    - `πₖ::Vector{Float64}`: Mixing coefficients for each cluster.


    # Examples
    ```julia
    gmm = GaussianMixtureModel(3, 2) # Create a Gaussian Mixture Model with 3 clusters and 2-dimensional data
    fit!(gmm, data)
    ```
    """

mutable struct GaussianMixtureModel <: MixtureModel
    k::Int  # Number of clusters
    emissions::Vector{GaussianEmission} #instantiate a vector of gaussian emissions
    πₖ::Vector{<:Real}  # Mixing coefficients
end


"""
    GaussianMixtureModel(k::Int, data_dim::Int)
    
Constructor for GaussianMixtureModel. Initializes Σₖ's covariance matrices to the 
identity, πₖ to a uniform distribution, and μₖ's means to zeros.

"""
#Changed function to initialize emissions instead of redundant sigmas and mu's
function GaussianMixtureModel(k::Int, data_dim::Int)
    # Initialize GaussianEmission objects for each cluster
    emissions = [GaussianEmission(zeros(Float64, data_dim), Matrix{Float64}(I, data_dim, data_dim)) for _ = 1:k]
    πₖ = ones(k) ./ k
    return GaussianMixtureModel(k, emissions, πₖ)
end

"""
Draw 'n' samples from gmm. Returns a Matrix{<:Real}, where each row is a data point.
"""
function sample(gmm::GaussianMixtureModel, n::Int)
    # Sample using the Categorical distribution for component assignments
    component_assignments = rand(Categorical(gmm.πₖ), n)
    
    # Initialize a matrix to store the samples, with each sample as a column
    data_dim = length(gmm.emissions[1].μ)  # Dimension of the data (assumed same for all components)
    samples = Matrix{Float64}(undef, data_dim, n)  # Shape (2, 100) for 2D data and 100 samples

    for i in 1:gmm.k
        # Find indices where samples are assigned to the i-th component
        component_indices = findall(x -> x == i, component_assignments)
        num_samples = length(component_indices)

        if num_samples > 0
            emission = gmm.emissions[i]  # Access the i-th GaussianEmission

            if data_dim == 1
                # Univariate case
                samples[:, component_indices] .= rand(Normal(emission.μ[1], sqrt(emission.Σ[1, 1])), 1, num_samples)
            else
                # Multivariate case
                samples[:, component_indices] .= rand(MvNormal(emission.μ, emission.Σ), num_samples)
            end
        end
    end
    
    return samples
end


# E-Step for GMM
"""
    E_Step(gmm::GaussianMixtureModel, data::Matrix{<:Real})
Performs the Expectation (E) step in the EM algorithm, calculating the responsibilities for each component of the GMM.

# Arguments
- `gmm::GaussianMixtureModel`: The Gaussian Mixture Model to be fitted.
- `data::Matrix{<:Real}`: The dataset on which the model will be fitted, where each row represents a data point.

Returns 
 - `class_probabilities::Matrix{<:Real}`: Probability of each class for each data point.

# Example
```julia
data = rand(2, 100)  # Generate some random data
gmm = GaussianMixtureModel(k=3, d=2)  # Initialize a GMM with 3 components and 2-dimensional data
E_Step(gmm::GaussianMixtureModel, data)
``` 
"""
function E_Step(gmm::GaussianMixtureModel, data::Matrix{<:Real})
    _, N = size(data)  # Get number of data points (columns)
    K = gmm.k
    log_γ = zeros(K, N)  # Responsibilities in log space

    for n in 1:N
        log_likelihoods = zeros(K)
        #calculate weighted likelihood for each emission 
        for k in 1:K
            log_likelihood_nk = loglikelihood(gmm.emissions[k], data[:, n])
            log_likelihoods[k] = log(gmm.πₖ[k]) + log_likelihood_nk
        end
        #normalize 
        log_sum_exp = logsumexp(log_likelihoods)
        for k in 1:K
            log_γ[k, n] = log_likelihoods[k] - log_sum_exp
        end
    end

    γ = exp.(log_γ)
    return γ
end


"""
    logsumexp(log_vals::Vector{Float64})
Computes the log-sum-exp of a vector of log values for numerical stability.

#Arguments
- log_vals::Vector{Float64}: Vector of log values.

Returns
- Float64: The log-sum-exp of the input values.
"""
function logsumexp(log_vals::Vector{Float64})
    max_log = maximum(log_vals)
    return max_log + log(sum(exp.(log_vals .- max_log))) # subtract max for stability
end


"""
    M_Step!(gmm::GaussianMixtureModel, data::Matrix{<:Real}, γ::Matrix{<:Real})
Performs the Maximization (M) step in the EM algorithm, updating the model parameters.

#Arguments
- gmm::GaussianMixtureModel: The GMM being updated.
- data::Matrix{<:Real}: The dataset, with each column representing a data point.
- γ::Matrix{<:Real}: The responsibilities from the E-step.

Returns
- Nothing, but updates the GMM parameters in place.

"""
function M_Step!(gmm::GaussianMixtureModel, data::Matrix{<:Real}, γ::Matrix{<:Real})
    _, N = size(data)
    K = gmm.k
    N_k = sum(γ, dims=2)

    for k in 1:K
        γ_k = γ[k, :]
        N_k_k = N_k[k]

        if N_k_k > 0
            normalized_γ_k = γ_k / N_k_k
        else
            normalized_γ_k = γ_k
        end

        gmm.emissions[k] = updateEmissionModel!(gmm.emissions[k], data, normalized_γ_k)
        gmm.πₖ[k] = max(N_k_k / N, 1e-6)
    end

    gmm.πₖ .= gmm.πₖ ./ sum(gmm.πₖ)
end


"""
    log_likelihood(gmm::GaussianMixtureModel, data::Matrix{<:Real})

Compute the log-likelihood of the data given the Gaussian Mixture Model (GMM). The data matrix should be of shape (# observations, # features).

# Returns
- `Float64`: The log-likelihood of the data given the model.
"""
function log_likelihood(gmm::GaussianMixtureModel, data::Matrix{<:Real})
    N = size(data, 2)
    K = gmm.k
    ll = 0.0

    #loop over data (column wise) for accululated log likelihood
    for n in 1:N
        log_probabilities = [
            log(gmm.πₖ[k]) + loglikelihood(gmm.emissions[k], data[:, n]) for k in 1:K
        ]
        max_log_prob = maximum(log_probabilities)
        ll_n = max_log_prob + log(sum(exp(log_prob - max_log_prob) for log_prob in log_probabilities))
        ll += ll_n
    end

    return ll
end

"""
Revised the kmeansapp_initialization to work for column based mixture model functions
"""
function kmeanspp_initialization(data::Matrix{Float64}, k::Int)
    T, N = size(data)
    centroids = Matrix{Float64}(undef, T, k)
    centroids[:, 1] = data[:, rand(1:N)]

    weights = zeros(N)

    for i in 2:k
        for j in 1:N
            weights[j] = minimum(sum((data[:, j] .- centroids[:, c]).^2) for c in 1:i-1)
        end
        weights .= weights ./ sum(weights)

        next_centroid_index = rand(Categorical(weights))
        centroids[:, i] = data[:, next_centroid_index]
    end

    return centroids
end

"""
    fit!(gmm::GaussianMixtureModel, data::Matrix{<:Real}; <keyword arguments>)
Fits a Gaussian Mixture Model (GMM) to the given data using the Expectation-Maximization (EM) algorithm.

# Arguments
- `gmm::GaussianMixtureModel`: The Gaussian Mixture Model to be fitted.
- `data::Matrix{<:Real}`: The dataset on which the model will be fitted, where each row represents a data point.
- `maxiter::Int=50`: The maximum number of iterations for the EM algorithm (default: 50).
- `tol::Float64=1e-3`: The tolerance for convergence. The algorithm stops if the change in log-likelihood between iterations is less than this value (default: 1e-3).
- `initialize_kmeans::Bool=false`: If true, initializes the means of the GMM using K-means++ initialization (default: false).

# Returns
- `class_probabilities`: A matrix where each entry (i, k) represents the probability of the i-th data point belonging to the k-th component of the mixture model.

# Example
```julia
data = rand(2, 100)  # Generate some random data
gmm = GaussianMixtureModel(k=3, d=2)  # Initialize a GMM with 3 components and 2-dimensional data
class_probabilities = fit!(gmm, data, maxiter=100, tol=1e-4, initialize_kmeans=true)
```
"""

function fit!(
    gmm::GaussianMixtureModel,
    data::Matrix{<:Real};
    maxiter::Int=100,
    tol::Float64=1e-3,
    initialize_kmeans::Bool=false
)
    prev_ll = -Inf
    log_likelihoods = Float64[] 

    if initialize_kmeans
        kmeans_centroids = permutedims(kmeanspp_initialization(data, gmm.k))
        println("Initialize means via k-means++")

        for k in 1:gmm.k
            gmm.emissions[k].μ = kmeans_centroids[k, :]  # Assign centroid to the k-th Gaussian component
        end
    end

    for i in 1:maxiter
        # E-Step: Calculate responsibilities (γ)
        class_probabilities = E_Step(gmm, data)

        # M-Step: Update parameters
        M_Step!(gmm, data, class_probabilities)
 
        # Calculate log-likelihood and check for convergence
        curr_ll = log_likelihood(gmm, data)

        println("Iteration: $i, Log-likelihood: $curr_ll")
        push!(log_likelihoods, curr_ll)  # Store the log-likelihood

        if abs(curr_ll - prev_ll) < tol
            println("Converged at iteration $i")
            break
        end
        prev_ll = curr_ll
    end
    return log_likelihoods
end

# Handle vector data by reshaping it into a 2D matrix with a single column
function E_Step(gmm::GaussianMixtureModel, data::Vector{Float64})
    E_Step(gmm, reshape(data, :, 1))
end

function M_Step!(gmm::GaussianMixtureModel, data::Vector{Float64}, class_probabilities::Matrix{<:Real})
    M_Step!(gmm, reshape(data, :, 1), class_probabilities::Matrix{<:Real})
end

function log_likelihood(gmm::GaussianMixtureModel, data::Vector{Float64})
    log_likelihood(gmm, reshape(data, :, 1))
end

function fit!(gmm::GaussianMixtureModel, data::Vector{Float64}; maxiter::Int=50, tol::Float64=1e-3, initialize_kmeans::Bool=true)
    fit!(gmm, reshape(data, :, 1); maxiter=maxiter, tol=tol, initialize_kmeans=initialize_kmeans)
end

"""
    PoissonMixtureModel

A Poisson Mixture Model for clustering and density estimation.

## Fields
- `k::Int`: Number of poisson-distributed clusters.
- `λₖ::Vector{Float64}`: Means of each cluster.
- `πₖ::Vector{Float64}`: Mixing coefficients for each cluster.

## Examples
```julia
pmm = PoissonMixtureModel(3) # 3 clusters, 2-dimensional data
fit!(pmm, data)```
"""
mutable struct PoissonMixtureModel <: MixtureModel
    k::Int  # Number of clusters
    emissions::Vector{PoissonEmissions}  # Emission models for each cluster
    πₖ::Vector{Float64}  # Mixing coefficients
end


"""
    PoissonMixtureModel(k::Int, d::Int)

Initializes a Poisson Mixture Model (PMM) with a specified number of components and dimensionality.

# Arguments
- `k::Int`: Number of clusters or components in the mixture model.
- `d::Int`: Dimensionality of the data.

# Returns
- `PoissonMixtureModel`: A PMM with `k` components, each having a Poisson emission model with `d` dimensions.

# Example
```julia
pmm = PoissonMixtureModel(3, 2)  # Initialize a PMM with 3 components and 2-dimensional data
"""
function PoissonMixtureModel(k::Int, d::Int)
    emissions = [PoissonEmissions(0.5 .+ 0.5 * rand(d)) for _ in 1:k]  # Initialize λ with small random values
    πₖ = ones(k) ./ k  # Uniform mixing coefficients
    return PoissonMixtureModel(k, emissions, πₖ)
end


"""
sample(pmm::PoissonMixtureModel, n::Int)

Generates synthetic samples from a fitted Poisson Mixture Model, where each sample is assigned to a component based on the model’s mixing coefficients.

# Arguments
- `pmm::PoissonMixtureModel`: The Poisson Mixture Model from which samples are drawn.
- `n::Int`: The number of data points (samples) to generate.

# Returns
- `samples::Matrix{Int}`: A matrix where each column represents a sampled data point in the `d`-dimensional space defined by the model.

# Example
```julia
pmm = PoissonMixtureModel(3, 2)
samples = sample(pmm, 100)
"""
function sample(pmm::PoissonMixtureModel, n::Int)

    component_samples = rand(Multinomial(n, pmm.πₖ))  # Vector of sample counts per component

    # Initialize a container for all samples, with n rows (samples) and d columns (dimensions)
    d = length(pmm.emissions[1].λ)
    samples = Matrix{Int}(undef, d, n)  

    start_idx = 1

    for i in 1:pmm.k
        num_samples = component_samples[i]
        
        if num_samples > 0
            # Generate samples for each dimension based on the Poisson rate parameter λ for component i
            λ = pmm.emissions[i].λ  # Rate parameters for the i-th component

            # Generate a matrix where each row is a Poisson-sampled dimension for `num_samples` points
            generated_samples = [rand(Poisson(λ[dim]), num_samples) for dim in 1:d]
            generated_samples = hcat(generated_samples...)'  # Transpose to get (d, num_samples)

            samples[:, start_idx:(start_idx + num_samples - 1)] .= generated_samples
            start_idx += num_samples
        end
    end

    return samples
end

"""
    E_Step(pmm::PoissonMixtureModel, data::Matrix{<:Real})

Performs the Expectation step in the EM algorithm, calculating the responsibility matrix for each data point.

# Arguments
- `pmm::PoissonMixtureModel`: The Poisson Mixture Model for which responsibilities are being calculated.
- `data::Matrix{<:Real}`: The dataset, where each column is a data point.

# Returns
- `γ::Matrix{Float64}`: A matrix where each entry (k, n) represents the probability of the `n`-th data point belonging to the `k`-th component.

# Example
```julia
data = rand(1:10, 2, 100)
pmm = PoissonMixtureModel(3, 2)
responsibilities = E_Step(pmm, data)
"""
function E_Step(pmm::PoissonMixtureModel, data::Matrix{<:Real})
    d, N = size(data)
    K = pmm.k

    γ = zeros(K, N)  # Each column is a data point, with K rows representing responsibilities per cluster

    log_πₖ = log.(pmm.πₖ)

    for n in 1:N
        # Calculate unnormalized log responsibilities for each component
        for k in 1:K
            log_likelihood = loglikelihood(pmm.emissions[k], data[:, n])  # Use data[:, n] for column-based access
            γ[k, n] = log_πₖ[k] + log_likelihood  # Log of the posterior responsibility
        end

        # Normalize the responsibilities
        logsum = logsumexp(γ[:, n])
        γ[:, n] .= exp.(γ[:, n] .- logsum)  # Exponentiate and normalize
    end
    
    return γ  # Return the responsibility matrix
end

    
"""
    M_Step!(pmm::PoissonMixtureModel, data::Matrix{<:Real}, γ::Matrix{<:Real})

Performs the Maximization step in the EM algorithm, updating the parameters of each component in the Poisson Mixture Model.

# Arguments
- `pmm::PoissonMixtureModel`: The Poisson Mixture Model being updated.
- `data::Matrix{<:Real}`: The dataset, where each column is a data point.
- `γ::Matrix{<:Real}`: The responsibility matrix from the E-step.

# Returns
- None. The function updates the parameters of `pmm` in place.

"""
function M_Step!(pmm::PoissonMixtureModel, data::Matrix{<:Real}, γ::Matrix{<:Real})
    d, N = size(data)

    for k in 1:pmm.k
        γ_k = γ[k, :]  # Responsibilities for component k as a row vector
        Nk = sum(γ_k)  # Effective number of points assigned to cluster k

        # Update emission model directly
        pmm.emissions[k] = updateEmissionModel!(pmm.emissions[k], data, γ_k)

        # Update the mixing coefficients
        pmm.πₖ[k] = Nk / N
    end
end

"""
    fit!(pmm::PoissonMixtureModel, data::Matrix{Int}; <keyword arguments>)

Fits a Poisson Mixture Model (PMM) to the given data using the Expectation-Maximization (EM) algorithm.

# Arguments
- `pmm::PoissonMixtureModel`: The Poisson Mixture Model to be fitted.
- `data::Matrix{Int}`: The dataset on which the model will be fitted, where each row represents a data point.
- `maxiter::Int=50`: The maximum number of iterations for the EM algorithm (default: 50).
- `tol::Float64=1e-3`: The tolerance for convergence. The algorithm stops if the change in log-likelihood between iterations is less than this value (default: 1e-3).
- `initialize_kmeans::Bool=false`: If true, initializes the means of the PMM using K-means++ initialization (default: false).

# Returns
- `class_probabilities`: A matrix where each entry (i, k) represents the probability of the i-th data point belonging to the k-th component of the mixture model.

# Example
```julia
data = rand(1:10, 100, 1)  # Generate some random integer data
pmm = PoissonMixtureModel(k=3)  # Initialize a PMM with 3 components
class_probabilities = fit!(pmm, data, maxiter=100, tol=1e-4, initialize_kmeans=true)
```
"""
function fit!(
    pmm::PoissonMixtureModel,
    data::Matrix{<:Real};
    maxiter::Int=50,
    tol::Float64=1e-3,
    initialize_kmeans::Bool=false,
)
    prev_ll = -Inf  # Initialize previous log likelihood to negative infinity
    N, d = size(data) 

    γ = zeros(N, pmm.k)

    if initialize_kmeans
        λₖ_matrix = permutedims(kmeanspp_initialization(Float64.(data), pmm.k))

        # Assign the centroids as initial values for Poisson rate parameters (λ)
        for k in 1:pmm.k
            pmm.emissions[k].λ = max.(λₖ_matrix[k, :], 1e-6)
        end
        println("Initialized Poisson rates via k-means++")
    end

    log_likelihoods = Float64[]

    for iter in 1:maxiter
        γ = E_Step(pmm, data)  # E-Step
        M_Step!(pmm, data, γ)  # M-Step
        curr_ll = log_likelihood(pmm, data)  # Current log likelihood

        println("Iteration: $iter, Log-likelihood: $curr_ll")
        push!(log_likelihoods, curr_ll)  # Store the log-likelihood

        if abs(curr_ll - prev_ll) < tol  # Check for convergence
            println("Convergence reached at iteration $iter")
            break
        end
        prev_ll = curr_ll  

    end
    return log_likelihoods
end

"""
    log_likelihood(pmm::PoissonMixtureModel, data::Matrix{Int})

Compute the log-likelihood of the data given the Poisson Mixture Model (PMM). The data matrix should be of shape (# observations, # features).

# Returns
- `Float64`: The log-likelihood of the data given the model.
"""
function log_likelihood(pmm::PoissonMixtureModel, data::Matrix{<:Real})
    d, N = size(data)
    ll = 0.0

    for n in 1:N
        # Calculate the log-sum-exp of cluster likelihoods for numerical stability
        cluster_log_likelihoods = [
            log(pmm.πₖ[k]) + loglikelihood(pmm.emissions[k], data[:, n]) for k in 1:pmm.k
        ]
        ll += logsumexp(cluster_log_likelihoods)

        if any(isnan.(cluster_log_likelihoods))
            println("Warning: NaN encountered in cluster_log_likelihoods at data point $n")
        end
    end
    
    return ll
end



# Handle vector data by reshaping it into a 2D matrix with a single column
function E_Step(pmm::PoissonMixtureModel, data::Vector{Int})
    E_Step(pmm, reshape(data, :, 1))
end

function M_Step!(pmm::PoissonMixtureModel, data::Vector{Int}, class_probabilities::Matrix{<:Real})
    M_Step!(pmm, reshape(data, :, 1), class_probabilities)
end

function log_likelihood(pmm::PoissonMixtureModel, data::Vector{Int})
    log_likelihood(pmm, reshape(data, :, 1))
end

function fit!(pmm::PoissonMixtureModel, data::Vector{Int}; maxiter::Int=50, tol::Float64=1e-3, initialize_kmeans::Bool=true)
    fit!(pmm, reshape(data, :, 1); maxiter=maxiter, tol=tol, initialize_kmeans=initialize_kmeans)
end


"""
    MultinomialMixtureModel(k::Int, n::Int)

A Multinomial Mixture Model for clustering and density estimation.

# Fields
- `k::Int`: Number of clusters.
- `n::Int`: Number of trials for each multinomial distribution (same for all clusters).
- `emissions::Vector{MultinomialEmissions}`: Emission models containing category probabilities for each cluster.
- `πₖ::Vector{Float64}`: Mixing coefficients representing the probability of a data point belonging to each cluster.

# Methods
- `fit!`: Fits the model to the given data using the Expectation-Maximization (EM) algorithm.
- `log_likelihood`: Computes the log-likelihood of the data given the model.
- `sample`: Generates samples from the mixture model.

# Examples
```julia
mmm = MultinomialMixtureModel(3, 10)  # 3 clusters, 10 trials per multinomial
fit!(mmm, data)

"""
mutable struct MultinomialMixtureModel <: MixtureModel
    k::Int  # Number of clusters
    n::Int  # Number of trials per multinomial
    emissions::Vector{MultinomialEmissions}  # Emission models for each cluster
    πₖ::Vector{Float64}  # Mixing coefficients
end

"""
    MultinomialMixtureModel(k::Int, n_trials::Int, p_dims::Int)

Constructor for MultinomialMixtureModel. Initializes the emission models 
with random probabilities for each category and uniform mixing coefficients.

# Arguments
- `k`: Number of clusters.
- `n_trials`: Number of trials for the multinomial distribution.
- `p_dims`: Number of categories in the multinomial distribution.

# Returns
- A new instance of MultinomialMixtureModel.
"""

function MultinomialMixtureModel(k::Int, n_trials::Int, p_dims::Int)
    # If p_dims == 2, treat it as binomial
    emissions = [MultinomialEmissions(n_trials, reshape(normalize(rand(p_dims), 1), p_dims, 1)) for _ in 1:k]
    πₖ = ones(k) ./ k  # Uniform mixing coefficients
    return MultinomialMixtureModel(k, n_trials, emissions, πₖ)
end

"""
    E_Step(mmm::MultinomialMixtureModel, data::Matrix{Int})

Performs the E-step of the Expectation-Maximization (EM) algorithm for the 
Multinomial Mixture Model. Calculates the responsibility matrix γ.

# Arguments
- `mmm`: A MultinomialMixtureModel instance.
- `data`: A matrix of observed data points, where each row is a data point.

# Returns
- `γ`: A matrix of responsibilities, where γ[n, k] represents the responsibility 
  of cluster `k` for data point `n`.
"""
function E_Step(mmm::MultinomialMixtureModel, data::Matrix{<:Real})
    d, N = size(data)  # Get dimensions (d categories, N data points)
    K = mmm.k           # Number of clusters
    γ = zeros(K, N)     # Responsibility matrix 

    log_πₖ = log.(mmm.πₖ) 

    for n in 1:N
        for k in 1:K
            log_likelihood = loglikelihood(mmm.emissions[k], data[:, n])  # Column-wise access
            γ[k, n] = log_πₖ[k] + log_likelihood  # Avoid exp/log instability here
        end
        γ[:, n] .= exp.(γ[:, n] .- logsumexp(γ[:, n])) # Normalize each column
    end

    return γ  # Return the responsibility matrix
end

"""
    M_Step for Multinomial Mixture Model

Performs the M-step of the Expectation-Maximization (EM) algorithm 
for the Multinomial Mixture Model. Updates the category probabilities 
and mixing coefficients.
"""
function M_Step!(mmm::MultinomialMixtureModel, data::Matrix{<:Real}, γ::Matrix{Float64})
    d, N = size(data)  # Number of categories and data points
    K = mmm.k          

    for k in 1:K
        γ_k = γ[k, :]   # Responsibilities for component k
        Nk = sum(γ_k)   # Effective number of points assigned to cluster k

        normalized_γ_k = γ_k / Nk

        mmm.emissions[k] = updateEmissionModel!(mmm.emissions[k], data, normalized_γ_k)

        mmm.πₖ[k] = Nk / N  # Update mixing coefficient
    end
end

"""
    fit!(mmm::MultinomialMixtureModel, data::Matrix{Int};
         maxiter::Int=50, tol::Float64=1e-3, initialize_kmeans::Bool=false)

Fit the Multinomial Mixture Model to the data using the Expectation-Maximization (EM) algorithm.

# Arguments
- `mmm::MultinomialMixtureModel`: The multinomial mixture model to be fitted.
- `data::Matrix{Int}`: Observed data points, where each row represents a data point.
- `maxiter::Int`: Maximum number of iterations (default: 50).
- `tol::Float64`: Convergence tolerance for the log-likelihood (default: 1e-3).
- `initialize_kmeans::Bool`: Whether to initialize probabilities using kmeans++ (default: false).

# Updates
- `mmm.emissions`: Updated category probabilities for each cluster.
- `mmm.πₖ`: Updated mixing coefficients for each cluster.
"""
function fit!(
    mmm::MultinomialMixtureModel,
    data::Matrix{Int};
    maxiter::Int = 50,
    tol::Float64 = 1e-3,
    initialize_kmeans::Bool = false,
)
    prev_ll = -Inf  # Initialize previous log-likelihood
    d, N = size(data)  # Number of categories and data points

    if initialize_kmeans
        # Initialize category probabilities using kmeans++
        pₖ_matrix = permutedims(kmeanspp_initialization(Float64.(data), mmm.k))
        for k in 1:mmm.k
            mmm.emissions[k].p .= pₖ_matrix[:, k]
        end
    end

    log_likelihoods = Float64[] 

    for iter in 1:maxiter
        γ = E_Step(mmm, data)  # E-Step
        
        M_Step!(mmm, data, γ) # M-Step: Update parameters based on responsibilities

        # Compute current log-likelihood
        curr_ll = log_likelihood(mmm, data)
        push!(log_likelihoods, curr_ll)
        println("Iteration: $iter, Log-likelihood: $curr_ll")

        # Check for convergence
        if abs(curr_ll - prev_ll) < tol
            println("Convergence reached at iteration $iter")
            break
        end
        prev_ll = curr_ll  # Update previous log-likelihood
    end
    return log_likelihoods
end

"""
    log_likelihood(mmm::MultinomiallMixtureModel, data::Vector{Int})

Computes the log-likelihood of the data under the Multinomial Mixture Model (MMM).

# Returns
- `Float64`: The log-likelihood of the data given the model.
"""
function log_likelihood(mmm::MultinomialMixtureModel, data::Matrix{<:Real})
    d, N = size(data)  # Number of categories and data points
    ll = 0.0  # Initialize log-likelihood

    for n in 1:N
        # Compute log-likelihood for each cluster, summed over all clusters
        cluster_likelihoods = [
            mmm.πₖ[k] * exp(loglikelihood(mmm.emissions[k], data[:, n]))
            for k in 1:mmm.k
        ]
        ll_n = log(sum(cluster_likelihoods))  # use Log-sum-exp for stability
        ll += ll_n  # Accumulate total log-likelihood
    end

    return ll
end

"""
sample(mmm::MultinomialMixtureModel, n::Int)

Draw 'n' samples from the fitted Multinomial Mixture Model (MMM).

# Returns
- A matrix of size `(n, d)`, where each row is a multinomial observation. 
  `d` is the number of categories in the multinomial distribution.
"""
function sample(mmm::MultinomialMixtureModel, n::Int)
    # Determine which component each data point is assigned to based on mixture weights
    component_assignments = rand(Categorical(mmm.πₖ), n)
    d = length(mmm.emissions[1].p)  # Number of categories
    samples = Matrix{Int}(undef, d, n)  # `d × n` structure for the samples

    for i in 1:mmm.k
        # Get indices of samples assigned to the i-th component
        component_indices = findall(x -> x == i, component_assignments)
        num_samples = length(component_indices)

        if num_samples > 0
            # Sample from the i-th Multinomial component for all selected indices
            p = vec(mmm.emissions[i].p)
            generated_samples = rand(Multinomial(mmm.emissions[i].n, p), num_samples)

            # Assign generated samples directly into columns for each component
            samples[:, component_indices] = generated_samples
        end
    end

    return samples
end