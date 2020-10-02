module NIPALS_PCA

include("structs.jl")
include("pca.jl")
include("pls.jl")

export calcPCA, calcPLS, Dataset, parseDataFrame,normalize,savemodel,loadmodel,predictY

end # module
