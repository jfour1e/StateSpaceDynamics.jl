export GaussianEmission, PoissonEmissions, MultinomialEmissions, RegressionEmissions, loglikelihood, sample_emission, update_emission_model!

"""
GaussianEmission: Struct representing a Gaussian emission model.
"""
mutable struct GaussianEmission <: EmissionsModel
    μ::Vector{Float64}  # State-dependent mean
    Σ::Matrix{<:Real}  # State-dependent covariance
end


# Loglikelihood function
function loglikelihood(emission::GaussianEmission, observation::Vector{<:Real})
    if length(emission.μ) == 1
        # Univariate case: Use Normal distribution
        return logpdf(Normal(emission.μ[1], sqrt(emission.Σ[1, 1])), observation[1])
    else
        # Multivariate case: Ensure dimensions match
        return logpdf(MvNormal(emission.μ, emission.Σ), observation)
    end
end

# Likelihood function
function likelihood(emission::GaussianEmission, observation::Vector{Float64})
    return pdf(MvNormal(emission.μ, emission.Σ), observation)
end

# Sampling function 
function sample_emission(emission::GaussianEmission)
    return rand(MvNormal(emission.μ, emission.Σ))
end

# Update emissions model for Gaussian model
function updateEmissionModel!( emission::GaussianEmission, data::Matrix{<:Real}, γ::Vector{Float64} )
    T, D = size(data)
    # Update mean
    new_mean = sum(data .* γ', dims=2) ./ sum(γ)
    new_mean = vec(new_mean)
    # Update covariance
    centered_data = data .- new_mean
    weighted_centered = centered_data .* γ'
    new_covariance = (weighted_centered * centered_data') / sum(γ)
    # check if the covariance is symmetric
    if !ishermitian(new_covariance)
        new_covariance = (new_covariance + new_covariance') * 0.5
    end
    # check if matrix is posdef
    if !isposdef(new_covariance)
        new_covariance = new_covariance + 1e-12 * I
    end
    # update the emission model
    emission.μ = new_mean
    emission.Σ = new_covariance
    return emission
end

"""
PoissonEmissions: Struct representing a Poisson emission model. This assumes a Poisson distribution for each 
dimension of the observation. This is referred to as a compound Poisson distribution. This is used in HMM-Poisson models,
though, Multivariate Poisson distributions have been derived to capture correlations between dimensions.
"""
mutable struct PoissonEmissions <: EmissionsModel
    λ::Vector{Float64} # rate of events per unit time 
end

# loglikelihood of the poisson model.
function loglikelihood(emission::PoissonEmissions, observation::Vector{<:Real})
    D = length(emission.λ)
    ll = 0.0
    for d in 1:D
        ll += logpdf(Poisson(emission.λ[d]), observation[d])  # Calculate log likelihood for each dimension
    end
    return ll
end

function updateEmissionModel!(
    emission::PoissonEmissions, data::Matrix{<:Real}, γ::Vector{Float64}
)
    d, N = size(data)

    # Calculate weighted sum of data for each dimension
    weighted_sum = sum(data .* reshape(γ, 1, :); dims=2)
    total_responsibility = sum(γ) + 1e-6  # Ensure a small regularization term to avoid division by zero
    new_λ = weighted_sum[:] ./ total_responsibility  # Normalize by the total responsibility sum

    # Enforce a lower bound on λ to avoid issues with the Poisson distribution
    new_λ[new_λ .< 1e-6] .= 1e-6

    # Update the Poisson emission model
    emission.λ = new_λ

    return emission
end

"""
MultinomialEmissions: Struct representing a Multinomial emission model.
"""
mutable struct MultinomialEmissions <: EmissionsModel
    n::Int64 # number of trials
    p::Matrix{<:Real} # probability of each category
end


# loglikelihood of the multinomial model.
function loglikelihood(emission::MultinomialEmissions, observation::Vector{<:Real})
    if length(emission.p) == 1
        # Binomial case
        return logpdf(Binomial(emission.n, emission.p[1]), observation[1])
    else
        # Multinomial case 
        return logpdf(Multinomial(emission.n, vec(emission.p)), observation)  # Convert p to vector
    end
end

function updateEmissionModel!(emission::MultinomialEmissions, data::Matrix{<:Real}, γ::Vector{Float64})
    d, N = size(data)  

    Nk = sum(γ)  # Sum of responsibilities 

    # Update category probabilities pₖ 
    p_new = sum(data .* γ', dims=2) / (Nk * emission.n)  # Sum across data points

    emission.p = reshape(p_new, size(emission.p))  # Reshape for multivariate case

    return emission 
end

"""
RegressionEmissions: A struct representing a regression model for emissions. This is used in HMM-GLM models.
"""
mutable struct RegressionEmissions <: EmissionsModel
    regression::Regression
end

# loglikelihood of the regression model.
function loglikelihood(emission::RegressionEmissions, X::Vector{Float64}, y::Float64)
    loglikelihood(emission.regression, X, y)
end

# loglikelihood of the regression model.
function loglikelihood(emission::RegressionEmissions, X::Matrix{<:Real}, y::Matrix{<:Real})
    loglikelihood(emission.regression, X, y)
end


# Update the parameters of the regression model, e.g. the betas.
function update_emissions_model!(emission::RegressionEmissions, X::Matrix{<:Real}, y::Vector{Float64}, w::Vector{Float64}=ones(length(y)))
    fit!(emission.regression, X, y, w)
end




# Update the parameters of the regression model, e.g. the betas.
function update_emissions_model!(emission::RegressionEmissions, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))

    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."

    # confirm the size of w is correct
    @assert length(w) == size(y, 1) "Length of w must be equal to the number of observations in y."


    fit!(emission.regression, X, y, w)
end
