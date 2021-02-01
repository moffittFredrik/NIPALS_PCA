module NIPALS_PCA

using DataFrames, JLD2, FileIO

include("structs.jl")
include("pca.jl")
include("pls.jl")
include("pls_norm.jl")

function loadIrisData()::DataFrame
    moduleisdefined = @isdefined NIPALS_PCA

    irisdir = @pipe pathof(NIPALS_PCA) |> splitpath |> _[1:end-2] |> joinpath(_...,"test","data","iris")

    CSV.File(joinpath(irisdir,"IRIS.csv"), header=false) |> DataFrame!
end

function loadDataFrame(file::String,valuekey::String="df")
    df=load(file,valuekey);    
end

export loadIrisData

#structs
export Dataset, PCA, PLS

export calcPCA, calcPLS, Dataset, parseDataFrame, parseMatrix, normalize,savemodel,loadmodel,predictY,crossvalidate, selectNumerical, selectColumns,onehot,predictLevel,calcVariances

# pls norm
export correct,calibrate_model,predict_xres

end # module
