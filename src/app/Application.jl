SRC_PATH = ".."
include("$(SRC_PATH)/repositories/Constants.jl")
include("$(SRC_PATH)/repositories/Settings.jl")
include("$(SRC_PATH)/app/Utils.jl")
include("$(SRC_PATH)/repositories/Instances.jl")
include("$(SRC_PATH)/domain/models/RNNetwork.jl")
include("$(SRC_PATH)/domain/models/FNNetwork.jl")
include("$(SRC_PATH)/domain/models/Networks.jl")
include("$(SRC_PATH)/domain/models/GANetwork.jl")
include("$(SRC_PATH)/domain/services/Evaluation.jl")
include("$(SRC_PATH)/domain/models/Objectives.jl")
include("$(SRC_PATH)/domain/services/Train.jl")

using Main.Utils
using Main.Settings: get_settings
using Main.Instances: load
using Main.Train: train!

"""
    main(; kwargs...)

Entrypoint for the application.
"""
function main(; kwargs...)::Nothing
    @info "Starting application..."

    settings = get_settings(; kwargs...)
    instance = load(; settings...)
    train!(instance; settings...)
end

@time main()
