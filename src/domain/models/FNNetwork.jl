"""
    FFNetwork

This module implements a Feed Forward Neural Network (FFN) to stablish relationships between the macroeconomic features and the firm characteristics.
"""
module FNNetwork

using Flux: Chain, Dense, relu, tanh, gpu

export FFN

mutable struct FFN
    layers::Chain
end

"""
    FFN(dims::Vector{Int})::FFN

Create a FFN with Dense layers.

# Arguments

  - `dims::Vector{Int}` : dimensions of the FFN ; where `dims[1]` is the input dimension, `dims[end]` is the output dimension and `dims[2:end-1]` are the hidden dimensions.
"""
function FFN(
    dims::Vector{Int};
    hidden_activation::Function = relu,
    output_activation::Function = tanh,
)::FFN
    @assert length(dims) >= 3
    hidden_layers =
        [Dense(dims[i], dims[i + 1], hidden_activation) for i in 1:(length(dims) - 2)]
    output_activation = dims[end] == 1 ? identity : output_activation
    output_layer = Dense(dims[end - 1], dims[end], output_activation)

    return FFN(gpu(Chain(hidden_layers..., output_layer)))
end

function (network::FFN)(x::Array)::Array
    return network.layers(x)
end

end # module FNNetwork