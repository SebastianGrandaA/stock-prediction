module Instances

import Base.size
using Base: @kwdef
using PyCall: pyimport, @py_str

using Main.Constants:
    UNKNOWN, DEFAULT_DATA_SPLIT, DEFAULT_INDIVIDUAL_FILES, DEFAULT_MACROECONOMIC_FILES
using Main.Utils: to_string, filter_periods

export Data, Instance, periods, firms, load, iterate

mutable struct Feature
    variables::Vector{String}
    features::Array
end

mutable struct Data
    ID::String
    macroeconomic::Feature
    individual::Feature
    returns::Array
    mask::Array
end

mutable struct Instance
    train::Data
    test::Data
    validate::Data
end

periods(data::Data)::Int64 = size(data.mask, 1)

firms(data::Data)::Int64 = size(data.mask, 2)

function size(feature::Feature)::Dict{Symbol, Tuple}
    return Dict{Symbol, Tuple}(
        :variables => size(feature.variables),
        :features => size(feature.features),
    )
end

function size(data::Data)::Dict{Symbol, Dict}
    return Dict{Symbol, Dict}(
        :macroeconomic => size(data.macroeconomic),
        :individual => size(data.individual),
        :returns => Dict(:historical => size(data.returns)),
    )
end

function validate!(returns::Array, individual_features::Array)::Nothing
    # Number of firms validation
    @assert length(returns) == length(individual_features)

    for firm in eachindex(returns)
        # Number of periods validation
        # for each firm, the number of valid periods for return is equal to the number of valid periods for each individual characteristic
        @assert length(returns[firm]) == size(individual_features[firm], 1)
    end

    return nothing
end

function process(
    ID::String,
    individual_path::String,
    macroeconomic_path::String;
    kwargs...,
)::Data
    returns_type = get(kwargs, :returns_type, :regular_returns)
    cols_map = Dict{Symbol, Int64}(
        :regular_returns => 1,
        :net_returns => 2,
        :individual_features => 4,
    )

    # Individual features and returns
    individual_file = load(individual_path)
    individual_data = individual_file["data"][:, :, 1:end]
    individual_idx = cols_map[:individual_features]
    individual_variables =
        [to_string(var) for var in individual_file["variable"][individual_idx:end]]

    individual_features = individual_data[:, :, individual_idx:end]
    valid_individual_features = filter_periods(individual_features)

    returns = individual_data[:, :, cols_map[returns_type]]
    valid_returns = filter_periods(returns)

    validate!(valid_returns, valid_individual_features)

    individual = Feature(individual_variables, valid_individual_features)

    # Macroeconomic features
    macroeconomic_file = load(macroeconomic_path)
    macro_variables = [to_string(var) for var in macroeconomic_file["variable"][2:end]]
    macroeconomic_data = macroeconomic_file["data"][:, 1:size(macro_variables, 1)]
    macroeconomic = Feature(macro_variables, macroeconomic_data)
    macro_mask = zeros(size(returns))
    macro_mask[returns .== UNKNOWN] .= NaN # periods in which there are no valid returns per firm
    @assert length(macroeconomic.variables) == size(macroeconomic.features, 2)
    @assert size(macroeconomic.features, 1) == size(macro_mask, 1)

    return Data(ID, macroeconomic, individual, valid_returns, macro_mask)
end

function iterate(instance::Instance)::Vector{Data}
    return [instance.train, instance.test, instance.validate]
end

function load(; kwargs...)::Instance
    data_split = get(kwargs, :data_split, DEFAULT_DATA_SPLIT)
    individual_files = get(kwargs, :individual_files, DEFAULT_INDIVIDUAL_FILES)
    macroeconomic_files = get(kwargs, :macroeconomic_files, DEFAULT_MACROECONOMIC_FILES)
    @assert length(data_split) == length(individual_files) == length(macroeconomic_files)

    return Instance(
        [
            process(name, individual_file, macroeconomic_file) for
            (name, individual_file, macroeconomic_file) in
            zip(data_split, individual_files, macroeconomic_files)
        ]...,
    )
end

function load(filename::String; kwargs...)::Any
    try
        file = nothing

        if occursin("np", filename)
            np = pyimport("numpy")

            if endswith(filename, ".npz")
                file = py"dict($np.load($filename, allow_pickle=True))"
            else
                file = py"$np.load($filename, allow_pickle=True)"
            end
        else
            @warn "File extension not supported"
        end

        isnothing(file) && error("File $(filename) not found")
        @info "Loaded data from $(filename)"

        return file

    catch error
        @warn "Error loading file $(filename): $(error)"
    end
end

end # module Instances