"""
    RNNetwork

This module implements a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) cells to summarize the macroeconomic variables into lower dimensional features.
"""
module RNNetwork

using Flux: Chain, LSTM, Dense, gpu

export RNN

mutable struct RNN
    cell::Chain
end

"""
    RNN(dims::Vector{Int})::RNN

Create a RNN with LSTM cells.

# Arguments

  - `dims::Vector{Int}` : dimensions of the RNN ; where `dims[1]` is the input dimension, `dims[end]` is the output dimension and `dims[2:end-1]` are the hidden dimensions.
"""
function RNN(dims::Vector{Int})::RNN
    @assert length(dims) >= 3
    lstm_layers = [LSTM(dims[i], dims[i + 1]) for i in 1:(length(dims) - 2)]
    output_layer = Dense(dims[end - 1], dims[end])

    return RNN(gpu(Chain(lstm_layers..., output_layer)))
end

function (network::RNN)(x::Array)::Array
    return network.cell(x)
end

end # module RNNetwork