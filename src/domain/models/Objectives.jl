module Objectives

using Flux: params, gradient, update!

using Main.GANetwork: GAN
using Main.Instances: Data, firms
using Main.Evaluation: weighted_returns, mean_skip

export compute_loss

"""
    compute_loss(gan::GAN, data::Data; network_type::Symbol, loss_type::Symbol, sense::Symbol)::Float64

Compute the loss function for a given GAN model and data set.
"""
function compute_loss(
    gan::GAN,
    data::Data;
    network_type::Symbol,
    loss_type::Symbol,
    sense::Symbol,
)::Float64
    @assert loss_type in [:unconditional, :conditional]
    @assert sense in [:min, :max]
    sign = sense == :min ? 1 : -1
    discount_factor =
        stochastic_discount_factor(gan.discriminator(data), data.returns, data.mask)
    weights = if loss_type == :conditional
        gan.generator(data)
    else
        [ones(length(data.returns[firm])) for firm in 1:firms(data)]
    end
    loss_value = sign * loss(data.returns, weights, discount_factor, data.mask)

    @assert network_type in [:generator, :discriminator]
    network = getproperty(gan, network_type)
    grads = gradient(() -> loss_value, params(network))
    update!(network.optimizer, params(network), grads)

    return loss_value
end

function loss(returns::Array, weights::Array, discount_factor::Array, mask::Array)::Float64
    weighted = weighted_returns(weights, returns, mask)
    discounted = discount_factor .* weighted
    loss_per_firm = mean_skip(discounted, dims = 2) .^ 2

    return mean_skip(loss_per_firm)
end

"""
    stochastic_discount_factor(weights::Array, returns::Array, mask::Array)::Array

The stochastic discount factor in a given period equals 1 - the sum of the weights times returns for all firms.
"""
function stochastic_discount_factor(weights::Array, returns::Array, mask::Array)::Array
    weighted = weighted_returns(weights, returns, mask)

    return [1 - sum(firm[.!isnan.(firm)]) for firm in eachrow(weighted)]
end

end # module Objectives
