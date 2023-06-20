module Settings

using Main.Constants:
    DEFAULT_SETTINGS_FILE, DEFAULT_EPOCHS, DEFAULT_SUB_EPOCHS, DEFAULT_LOG_EPOCH
using JSON: parsefile

export get_settings

function get_settings(; kwargs...)::Dict{Symbol, Any}
    path = get(kwargs, :settings, DEFAULT_SETTINGS_FILE)
    result = nothing
    try
        if endswith(path, ".json")
            result = parsefile(path, dicttype = Dict{Symbol, Any})
        else
            @warn "File extension not supported."
        end

        isnothing(result) && error("File $(path) not found.")
        @info "Settings loaded."

    catch error
        @warn "Error loading file $(path): $(error)"
    end

    merge!(result, kwargs)
    epochs = get(result, :epochs, DEFAULT_EPOCHS)
    sub_epochs = get(result, :sub_epochs, DEFAULT_SUB_EPOCHS)
    log_epoch = get(result, :log_epoch, DEFAULT_LOG_EPOCH)
    push!(result, :epochs => epochs, :sub_epochs => sub_epochs, :log_epoch => log_epoch)

    return result
end

end # module Settings