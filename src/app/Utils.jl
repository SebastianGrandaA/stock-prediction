module Utils

using PyCall: PyObject, pystring

using Main.Constants: UNKNOWN

function to_string(object::PyObject)::String
    return strip(pystring(object), ['"', '\''])
end

"""
    filter_periods(x::Matrix)::Array

For each firm, filter periods in which there are no valid return.
"""
function filter_periods(x::Matrix)::Array
    n_firms = size(x, 2)
    return [filter(period -> period != UNKNOWN, x[:, firm]) for firm in 1:n_firms]
end

"""
    filter_periods(x::Array{Float64, 3})::Array

For each firm and feature, filter periods in which there are no valid feature.
"""
function filter_periods(x::Array{Float64, 3})::Array
    _, n_firms, n_features = size(x)

    return [
        hcat(
            [
                filter(period -> period != UNKNOWN, x[:, firm, feature])
                for feature in 1:n_features
            ]...,
        ) for firm in 1:n_firms
    ]
end

function fill!(
    base_values::Matrix,
    new_values::Vector;
    fixed_idx::Int64,
    start_idx::Int,
)::Nothing
    for period in 1:size(base_values, 1)
        if !isnan(base_values[period, fixed_idx])
            base_values[period, fixed_idx] = new_values[start_idx]
            start_idx += 1
        end
    end

    return nothing
end

function validate_adjacent_items(array::Array, operation::Function)::Bool
    return all([operation(array[i], array[i + 1]) for i in 1:(length(array) - 1)])
end

function log!(
    info::Any,
    epoch::Int64;
    start_time::Float64 = 0.0,
    log_epoch::Int64 = 1,
)::Nothing
    if epoch % log_epoch == 0
        log_info = "Epoch $(epoch) | $(info)"

        if !(iszero(start_time))
            elapsed_time = round(time() - start_time, digits = 2)
            log_info *= " | Elapsed time $(elapsed_time)"
        end

        @info log_info
    end
end

end # module Utils