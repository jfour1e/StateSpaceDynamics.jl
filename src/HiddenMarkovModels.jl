export HMM, baumWelch!, viterbi
# HMM Definition
struct HMM{EM <: EmissionsModel}
    A::Matrix{Float64}  # State Transition Matrix
    B::Vector{EM}       # Emission Model
    πₖ ::Vector{Float64} # Initial State Distribution
    D::Int              # Observation Dimension
end

function HMM(data::Matrix{Float64}, k_states::Int=2, emissions::String="Gaussian")
    N, D = size(data)
    # Initialize A
    A = rand(k_states, k_states)
    A = A ./ sum(A, dims=2)  # normalize rows to ensure they are valid probabilities
    # Initialize π
    πₖ = rand(k_states)
    πₖ = πₖ ./ sum(πₖ)          # normalize to ensure it's a valid probability vector
    # Initialize Emission Model
    if emissions == "Gaussian"
        # Randomly sample k_states observations from data
        sample_means = data[sample(1:N, k_states, replace=false), :]
        # Using identity matrix for covariance
        sample_covs = [Matrix(I, D, D) for _ in 1:k_states]
        B = [GaussianEmission(sample_means[i, :], sample_covs[i]) for i in 1:k_states]
    else
        throw(ErrorException("$emissions is not a supported emissions model, please choose one of the supported models."))
    end
    return HMM(A, B, πₖ, D)
end


function forward(hmm::HMM, data::Matrix{Float64})
    T, _ = size(data)
    K = size(hmm.A, 1)  # Number of states

    # Initialize an α-matrix 
    α = zeros(Float64, K, T)

    # Calculate α₁
    for k in 1:K
        α[k, 1] = log(hmm.πₖ[k]) + loglikelihood(hmm.B[k], data[1, :])
    end
    
    # Now perform the rest of the forward algorithm for t=2 to T
    for t in 2:T
        for j in 1:K
            values_to_sum = Float64[]
            for i in 1:K
                push!(values_to_sum, log(hmm.A[i, j]) + α[i, t-1])
            end
            log_sum_alpha_a = logsumexp(values_to_sum)
            α[j, t] = log_sum_alpha_a + loglikelihood(hmm.B[j], data[t, :])
        end
    end
    return α
end

function backward(hmm::HMM, data::Matrix{Float64})
    T, _ = size(data)
    K = size(hmm.A, 1)  # Number of states

    # Initialize a β matrix
    β = zeros(Float64, K, T)
    
    # Set last β values. In log-space, 0 corresponds to a value of 1 in the original space.
    β[:, T] .= 0  # log(1) = 0

    # Calculate β, starting from T-1 and going backward to 1
    for t in T-1:-1:1
        for i in 1:K
            values_to_sum = Float64[]
            for j in 1:K
                push!(values_to_sum, log(hmm.A[i, j]) + loglikelihood(hmm.B[j], data[t+1, :]) + β[j, t+1])
            end
            β[i, t] = logsumexp(values_to_sum)
        end
    end
    return β
end

# function calculate_gamma(α::Vector{Float64}, β::Vector{Float64})
#     T, _ = size(α)
#     γ = α .+ β
#     # Normalize γ values
#     for t in 1:T
#         max_gamma = maximum(γ[:, t])
#         log_sum = max_gamma + log(sum(exp.(γ[:, t] .- max_gamma)))
#         γ[:, t] .-= log_sum
#     end
# end


function baumWelch!(hmm::HMM,  data::Matrix{Float64}, max_iters::Int=100, tol::Float64=1e-6)
    T, _ = size(data)
    K = size(hmm.A, 1)
    log_likelihood = -Inf
    # Initialize progress bar
    p = Progress(max_iters; dt=1, desc="Computing Baum-Welch...",)
    for iter in 1:max_iters
        # Update the progress bar
        next!(p; showvalues = [(:iteration, iter), (:log_likelihood, log_likelihood)])
        α = forward(hmm, data)
        β = backward(hmm, data)
        # Compute and update the log-likelihood
        log_likelihood_current = logsumexp(α[:, T])
        if abs(log_likelihood_current - log_likelihood) < tol
            finish!(p)
            break
        else
            log_likelihood = log_likelihood_current
        end
        # Calculate proabilities according to Bayes rule, i.e. E-Step
        γ = α .+ β
        # Normalize γ values
    for t in 1:T
        max_gamma = maximum(γ[:, t])
        log_sum = max_gamma + log(sum(exp.(γ[:, t] .- max_gamma)))
        γ[:, t] .-= log_sum
    end
        # Now we calculate ξ values
        ξ = zeros(Float64, K, K, T-1)
        for t in 1:T-1
            # Array to store the unnormalized ξ values
            log_ξ_unnormalized = zeros(Float64, K, K)
            for i in 1:K
                for j in 1:K
                    log_ξ_unnormalized[i, j] = α[i, t] + log(hmm.A[i, j]) + loglikelihood(hmm.B[j], data[t+1, :]) + β[j, t+1]
                end
            end
            # Normalize the ξ values using log-sum-exp operation
            max_ξ = maximum(log_ξ_unnormalized)
            denominator = max_ξ + log(sum(exp.(log_ξ_unnormalized .- max_ξ)))
            
            ξ[:, :, t] .= log_ξ_unnormalized .- denominator
        end
        # M-Step; update our parameters based on E-Step
        # Update initial state probabilities
        hmm.πₖ .= exp.(γ[:, 1])
        # Update transition probabilities
        for i in 1:K
            for j in 1:K
                hmm.A[i,j] = exp(log(sum(exp.(ξ[i,j,:]))) - log(sum(exp.(γ[i,1:T-1]))))
            end
        end
        for k in 1:K
            hmm.B[k] = updateEmissionModel!(hmm.B[k], data, exp.(γ[k,:]))
        end
    end
end

function viterbi(hmm::HMM, data::Matrix{Float64})
    T, _ = size(data)
    K = size(hmm.A, 1)  # Number of states

    # Step 1: Initialization
    viterbi = zeros(Float64, K, T)
    backpointer = zeros(Int, K, T)
    
    for i in 1:K
        viterbi[i, 1] = log(hmm.πₖ[i]) + loglikelihood(hmm.B[i], data[1, :])
        backpointer[i, 1] = 0
    end

    # Step 2: Recursion
    for t in 2:T
        for j in 1:K
            max_prob, max_state = -Inf, 0
            for i in 1:K
                prob = viterbi[i, t-1] + log(hmm.A[i,j]) + loglikelihood(hmm.B[j], data[t, :])
                if prob > max_prob
                    max_prob = prob
                    max_state = i
                end
            end
            viterbi[j, t] = max_prob
            backpointer[j, t] = max_state
        end
    end

    # Step 3: Termination
    best_path_prob, best_last_state = findmax(viterbi[:, T])
    best_path = [best_last_state]
    for t in T:-1:2
        push!(best_path, backpointer[best_path[end], t])
    end

    return reverse(best_path)
end

function fit!()
    #TODO Implement the viterbi algorithm.
end