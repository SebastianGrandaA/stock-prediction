"""
    Networks

This module implements two networks:

 1. Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) cells to summarize the macroeconomic variables into lower dimensional features,
 2. Feed Forward Neural Network (FFN) to stablish relationships between the macroeconomic features and the firm characteristics.
"""
module Networks

import Flux.trainable
using Flux: gpu, ADAM
using Flux.Optimise: AbstractOptimiser
using LinearAlgebra: normalize!

using Main.Instances: Data
using Main.RNNetwork: RNN
using Main.FNNetwork: FFN

mutable struct Network
    ID::Symbol
    rnn::RNN
    ffn::FFN
    optimizer::AbstractOptimiser
end

function Network(
    ID::Symbol,
    dims_rnn::Vector{Int},
    dims_ffn::Vector{Int};
    kwargs...,
)::Network
    learning_rate = get(kwargs, :learning_rate, 0.001)
    optimizer = eval(get(kwargs, :optimizer, :ADAM))(learning_rate)

    return Network(ID, RNN(dims_rnn), FFN(dims_ffn), optimizer)
end

"""
    (network::Network)(data::Data; to_normalize::Bool=true)::Array

Given a network and a data set, it returns the weights for each firm in each period.
"""
function (network::Network)(data::Data; to_normalize::Bool = true)::Array
    macro_states = network.rnn(Array(data.macroeconomic.features'))' # T periods x M macro states.
    weights = []

    for (firm, individual_features) in enumerate(data.individual.features)
        valid_macro_states = macro_states[.!isnan.(data.mask[:, firm]), :] # T valid periods x M macro states
        firm_features = Array(hcat(valid_macro_states, individual_features)') # T periods x (M macro states + I individual features)
        firm_weights = collect(vec(network.ffn(firm_features)')) # T periods x 1 weights
        to_normalize && normalize!(firm_weights)

        push!(weights, firm_weights)
    end

    @debug "Network $(network.ID) weights: $(size(weights))"
    return weights
end

trainable(network::Network) = (rnn = network.rnn.cell, ffn = network.ffn.layers)

end # module Networks
