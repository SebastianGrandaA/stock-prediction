module Evaluation

using Statistics: std, var, mean
using Plots: plot

using Main.Utils: fill!

export evaluate, evaluate!, sharpe_ratio, weighted_returns, mean_skip

"""
    sharpe_ratio(returns::Array)::Float64

Calculated as the expected return divided by the standard deviation of the returns.
"""
function sharpe_ratio(returns::Array)
    deviation = std(returns)

    return mean(returns ./ deviation)
end

function evaluate!(metrics::Dict{Symbol, Any}, predictions::Array; digits::Int = 3)::Nothing
    sharpe = round(sharpe_ratio(predictions), digits = digits)
    push!(metrics, :predictions => predictions, :sharpe_ratio => sharpe)

    return nothing
end

function evaluate(predictions::Array)::Dict{Symbol, Any}
    metrics = Dict{Symbol, Any}()
    evaluate!(metrics, predictions)

    return metrics
end

function weighted_returns(weights::Array, returns::Array, mask::Array)::Array
    base = deepcopy(mask) # T periods x N firms

    for firm in eachindex(returns)
        factor = weights[firm] .* returns[firm]
        fill!(base, factor, fixed_idx = firm, start_idx = 1)
    end

    return base
end

function mean_skip(vector::Vector)::Float64
    return mean(filter(x -> !isnan(x), vector))
end

function mean_skip(matrix::Matrix; dims::Int64)::Vector
    return vcat([mean_skip(matrix[:, i]) for i in 1:size(matrix, dims)]...)
end

function plot_metrics!(metrics::Dict{Symbol, Any})::Nothing
    plot(
        metrics[:train_info].epoch,
        metrics[:train_info].discriminator_loss,
        label = "Discriminator Loss",
    )
    plot(
        metrics[:train_info].epoch,
        metrics[:train_info].generator_loss,
        label = "Generator Loss",
    )
    plot(
        metrics[:train_info].epoch,
        metrics[:train_info].fixed_discriminator_loss,
        label = "Fixed Discriminator Loss",
    )

    return nothing
end

end # module Evaluation