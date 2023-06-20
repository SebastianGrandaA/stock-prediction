"""
    GAN

This module aims to identify portfolio strategies with the most pricing information (the hardest to explain) to enforce no-arbitrage conditions.
This model implements two networks: the discriminator (aka SDF network) and the generator (aka Conditional network).
"""
module GANetwork

import Flux.trainable

using Main.Networks: Network
using Main.Instances: Data
using Main.Utils: log!
using Main.Constants: DEFAULT_DIMS, DEFAULT_OUTPUT_DIM

export GAN

mutable struct GAN
    generator::Network
    discriminator::Network
end

function GAN(dims_generator::Dict, dims_discriminator::Dict)::GAN
    generator = Network(:generator, dims_generator[:rnn], dims_generator[:ffn])
    discriminator =
        Network(:discriminator, dims_discriminator[:rnn], dims_discriminator[:ffn])

    return GAN(generator, discriminator)
end

function GAN(data::Data; kwargs...)::GAN
    n_individual_features = length(data.individual.variables)
    n_macro_features = length(data.macroeconomic.variables)
    output_rnn_gen = get(kwargs, :output_dim_rnn_gen, DEFAULT_OUTPUT_DIM)
    output_rnn_disc = get(kwargs, :output_dim_rnn_disc, DEFAULT_OUTPUT_DIM)

    dims_generator = Dict(
        :rnn => [
            n_macro_features,
            get(kwargs, :hidden_dim_rnn_gen, DEFAULT_DIMS)...,
            output_rnn_gen,
        ],
        :ffn => [
            n_individual_features + output_rnn_gen,
            get(kwargs, :hidden_dim_ffn_gen, DEFAULT_DIMS)...,
            get(kwargs, :output_dim_ffn_gen, DEFAULT_OUTPUT_DIM),
        ],
    )
    dims_discriminator = Dict(
        :rnn => [
            n_macro_features,
            get(kwargs, :hidden_dim_rnn_disc, DEFAULT_DIMS)...,
            output_rnn_disc,
        ],
        :ffn => [
            n_individual_features + output_rnn_disc,
            get(kwargs, :hidden_dim_ffn_disc, DEFAULT_DIMS)...,
            get(kwargs, :output_dim_ffn_disc, DEFAULT_OUTPUT_DIM),
        ],
    )

    return GAN(dims_generator, dims_discriminator)
end

"""
    (gan::GAN)(data::Data; loss::Function, epochs::Int, log_epoch::Int)::NamedTuple

Train the GAN model based on a given data set.
"""
function (gan::GAN)(data::Data; loss::Function, epochs::Int, log_epoch::Int)::NamedTuple
    discriminator_loss, generator_loss, fixed_discriminator_loss = NaN, NaN, NaN

    # 1. SDF network: unconditional loss
    for epoch in 1:epochs
        discriminator_loss = loss(
            gan,
            data,
            network_type = :discriminator,
            loss_type = :unconditional,
            sense = :min,
        )
        log!("(discriminator_loss = $(discriminator_loss))", epoch, log_epoch = log_epoch)
    end

    # 2. Conditional network
    for epoch in 1:epochs
        generator_loss = loss(
            gan,
            data,
            network_type = :generator,
            loss_type = :conditional,
            sense = :max,
        )
        log!("(generator_loss = $(generator_loss))", epoch, log_epoch = log_epoch)
    end

    # 3. SDF network: conditional loss
    for epoch in 1:epochs
        fixed_discriminator_loss = loss(
            gan,
            data,
            network_type = :discriminator,
            loss_type = :conditional,
            sense = :min,
        )
        log!(
            "(fixed_discriminator_loss = $(fixed_discriminator_loss))",
            epoch,
            log_epoch = log_epoch,
        )
    end

    return (
        discriminator_loss = discriminator_loss,
        generator_loss = generator_loss,
        fixed_discriminator_loss = fixed_discriminator_loss,
    )
end

function trainable(gan::GAN)
    return (
        generator = trainable(gan.generator),
        discriminator = trainable(gan.discriminator),
    )
end

end # module GANetwork
