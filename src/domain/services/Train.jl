module Train

using DataFrames: DataFrame
using CSV: write

using Main.Constants:
    BASE_TRAIN_INFO,
    DEFAULT_EPOCHS,
    DEFAULT_SUB_EPOCHS,
    DEFAULT_LOG_EPOCH,
    BASE_METRICS,
    DEFAULT_OUTPUT_PATH
using Main.Instances: Instance, Data, iterate, firms
using Main.GANetwork: GAN
using Main.Utils: log!
using Main.Evaluation: weighted_returns, evaluate!, mean_skip
using Main.Objectives: compute_loss

export train!

"""
    train(network::GAN, data::Data; epochs::Int, sub_epochs::Int, log_epoch::Int)::Dict{Symbol, Any}

Train the GAN model based on a given data set.
"""
function train(
    network::GAN,
    data::Data;
    epochs::Int,
    sub_epochs::Int,
    log_epoch::Int
)::Dict{Symbol,Any}
    train_info = DataFrame(BASE_TRAIN_INFO)

    for epoch in 1:epochs
        start = time()
        step =
            network(data, loss=compute_loss, epochs=sub_epochs, log_epoch=log_epoch)
        push!(train_info, (epoch=epoch, step...))

        log!(step, epoch, start_time=start, log_epoch=log_epoch)
    end

    metrics = Dict{Symbol,Any}(:train_info => train_info)

    weighted = weighted_returns(network.generator(data), data.returns, data.mask)
    predictions = mean_skip(weighted, dims=2)
    evaluate!(metrics, predictions)

    return metrics
end

function train(data::Data; epochs::Int, sub_epochs::Int, log_epoch::Int)::Dict{Symbol,Any}
    gan = GAN(data)
    metrics =
        train(gan, data, epochs=epochs, sub_epochs=sub_epochs, log_epoch=log_epoch)

    return metrics
end

"""
    train(instance::Instance; kwargs...)::DataFrame

Train the GAN model based on the instance. We split the instance into train, validation and test data sets.
"""
function train!(instance::Instance; kwargs...)::Nothing
    epochs = get(kwargs, :epochs, DEFAULT_EPOCHS)
    sub_epochs = get(kwargs, :sub_epochs, DEFAULT_SUB_EPOCHS)
    log_epoch = get(kwargs, :log_epoch, DEFAULT_LOG_EPOCH)
    output_path = get(kwargs, :output_path, DEFAULT_OUTPUT_PATH)
    save = get(kwargs, :save, true)
    result = DataFrame(BASE_METRICS)

    for data in iterate(instance)
        @info "Start training on $(data.ID) | $(firms(data)) firms"
        gan = GAN(data)
        metrics = train(
            gan,
            data,
            epochs=epochs,
            sub_epochs=sub_epochs,
            log_epoch=log_epoch,
        )

        push!(result, (data.ID, metrics[:predictions], metrics[:sharpe_ratio]...))
        save && write("$(output_path)/$(data.ID)_metrics.csv", metrics)

        @info "Finish training on $(data.ID) | Sharpe Ratio: $(metrics[:sharpe_ratio])"
    end

    @info "Finish training on all instances: $(result)"
    save && write("$(output_path)/result.csv", result)

    return nothing
end

end # module Train