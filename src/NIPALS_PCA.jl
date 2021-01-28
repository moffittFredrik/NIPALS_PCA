module NIPALS_PCA

include("structs.jl")
include("pca.jl")
include("pls.jl")

export calcPCA, calcPLS, Dataset, parseDataFrame, parseMatrix, normalize,savemodel,loadmodel,predictY,crossvalidate, selectNumerical, selectColumns,onehot,predictLevel,calcVariances

end # module
