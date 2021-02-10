using Pipe
using DataFrames
using Statistics
using JLD2
using CategoricalArrays
using DocStringExtensions

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct Dataset
    X::Array{Union{Missing,Float64},2}
    means::Array{Float64,1}
    stdevs::Array{Float64,1}
    value_columns::Array{String,1}
    xmask::BitArray{2}
    mv::Bool
    mvs::Array{Float64,1}
    varvalues::Array{Int64,1}
    ranges::Array{Float64,1}
end

abstract type MultivariateModel end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct PCA <: MultivariateModel
    T::DataFrame
    P::DataFrame
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct PLS <: MultivariateModel
    T::DataFrame
    P::DataFrame
    C::DataFrame
    W::DataFrame
    U::DataFrame
end

function copydataset(dataset)::Dataset
    Dataset(
        copy(dataset.X),
        copy(dataset.means),
        copy(dataset.stdevs),
        copy(dataset.value_columns),
        copy(dataset.xmask),
        dataset.mv,
        copy(dataset.mvs),
        copy(dataset.varvalues),
        copy(dataset.ranges)
    )
end

function copydataset(dataset,obsmask::BitArray{1}, varmask::BitArray{1})::Dataset
    Dataset(
        dataset.X[obsmask,varmask],
        dataset.means[varmask],
        dataset.stdevs[varmask],
        dataset.value_columns[varmask],
        dataset.xmask[obsmask,varmask],
        dataset.mv,
        dataset.mvs[varmask],
        dataset.varvalues[varmask],
        dataset.ranges[varmask]
    )
end

function fillmissings(dataset::Dataset; fillvalue = 0.0)
    dataset.X[.~dataset.xmask] .= fillvalue
end

function hasVariation(x)
    !isnan(x) && x > 0
end 

function norm(v)
    sqrt(sum(v.^2))
end

function valueRange(array)
    valrng = NaN

    try
        valrng = maximum(skipmissing(array))-minimum(skipmissing(array)) 
    catch
        valrng = NaN
    end

    valrng
end

isnumcol(type)= type <: Union{Missing,Number}

function parseMatrix(X::Array{Union{Missing, Float64},2},value_columns::Array{String,1})

    ranges = [valueRange(skipmissing(col)) for col in eachcol(X) ]

    var_means::Array{Float64,1} = [mean(skipmissing(col)) for col in eachcol(X) ]
    mean_mask = (!isnan).(var_means)

    var_stdevs::Array{Float64,1} = [std(skipmissing(col)) for col in eachcol(X) ]
    std_mask = hasVariation.(var_stdevs)

    inc_mask = mean_mask .& std_mask

    #Xtr = X[:,inc_mask]
    Xtr = X[:,:]

    xmask = (!ismissing).(Xtr)

    varvalues = sum(xmask,dims=1)[:]

    mvs = 1 .- (varvalues ./ size(X,1))

    Dataset(Xtr, var_means, var_stdevs, value_columns, xmask, sum(.~xmask) > 0, mvs, varvalues,ranges)
end

function parseDataFrame(df::AbstractDataFrame; filters=[dataset-> .~isnan.(dataset.means)])::Dataset

    value_columns = names(df[:,eltype.(eachcol(df)) |> coltypes -> isnumcol.(coltypes)])

    X = convert(Array{Union{Missing,Float64},2}, df[:,value_columns])

    return parseMatrix(X,value_columns) |> ds -> filterDataset(ds,filters=filters)
end

function calcObsMissingValues(dataset::Dataset)
    calcMissingValues(dataset.xmask,2)
end

function calcObsMissingValues(mask::BitArray{2})
    calcMissingValues(mask,2)
end

function calcVarMissingValues(mask::BitArray{2})
    calcMissingValues(mask,1)
end

function calcMissingValues(mask::BitArray{2}, dim)
    sum(.~mask,dims=dim) ./ size(mask,dim)
end    

function filterDataset(dataset::Dataset; filters=[], obsfilters=[])::Dataset
    mask = filterVariables(dataset,filters=filters)

    obsmask = filterObservations(dataset,filters=obsfilters)

    Dataset(
        dataset.X[obsmask,mask],
        dataset.means[mask],
        dataset.stdevs[mask],
        dataset.value_columns[mask],
        dataset.xmask[obsmask,mask],
        sum(.~(dataset.xmask[obsmask,mask])) > 0,
        dataset.mvs[mask],
        dataset.varvalues[mask],
        dataset.ranges[mask]
    )
end

function filterVariables(dataset::Dataset; filters=[])

    allvariables = allvars(dataset)

    varmask = @pipe [filter(dataset) for filter in filters] |> [allvariables,_...] |> tomatrix |> m->all(m,dims=2) |> vec
end

function filterObservations(dataset::Dataset; filters=[])

    allobservations = allobs(dataset)

    varmask = @pipe [filter(dataset) for filter in filters] |> [allobservations,_...] |> tomatrix |> m->all(m,dims=2) |> vec
end

function nobs(dataset::Dataset)
    size(dataset.X,1)
end

function nvars(dataset::Dataset)
    size(dataset.X,2)
end

function tomatrix(arrayOfArrays)
    hcat(arrayOfArrays...)
end

function allvars(dataset::Dataset)::BitArray{1}
    BitArray{1}(ones(nvars(dataset)))
end

function allobs(dataset::Dataset)::BitArray{1}
    BitArray{1}(ones(nobs(dataset)))
end

function normalize!(dataset::Dataset; doscale::Bool=false, stdevs=dataset.stdevs, means=dataset.means, transformations=[])

    #TODO:double check that it works 
    mean_mask = (!isnan).(means)
    std_mask = hasVariation.(stdevs)

    inc_mask = mean_mask .& std_mask

    # has variables to be excluded
    excludevars = sum(inc_mask) < nvars(dataset)

    if !isempty(transformations) 
        [transform(dataset) transform in transformations]
    else
        dataset.X[:,mean_mask] .-= dataset.means[mean_mask]'

        if doscale
            dataset.X[:,inc_mask] ./= dataset.stdevs[inc_mask]'
        end
    end

    dataset.X[.~dataset.xmask] .= 0

    dataset
end

function normalize(dataset::Dataset; doscale::Bool=false, stdevs=dataset.stdevs, means=dataset.means)

    dscopy = copydataset(dataset)
 
    normalize!(dscopy,doscale=doscale)
end

function normalizedata(X::Array{Union{Missing,Float64},2};normalize::Bool=false)
    var_means::Array{Float64,1} = [mean(skipmissing(col)) for col in eachcol(X) ]
    mean_mask = (!isnan).(var_means)

    var_stdevs::Array{Float64,1} = [std(skipmissing(col)) for col in eachcol(X) ]
    std_mask = hasVariation.(var_stdevs)

    inc_mask = mean_mask .& std_mask

    Xtr = X[:,inc_mask]

    xmask = (!ismissing).(Xtr)

    Xtr .-= var_means[inc_mask]'

    if normalize
        Xtr ./= var_stdevs[inc_mask]'
    end

    Xtr[.~xmask] .= 0
end


"""
$(FUNCTIONNAME)(model::T, dataset::Dataset, name::String) where T <: MultivariateModel

    Save PCA or PLS model as JLD2 file
"""
function savemodel(model::T, dataset::Dataset, path::String, transformations::Array{String,1}) where T <: MultivariateModel

    values = fieldnames(T) |> fns -> getfield.(Ref(model), fns)

    zipped = zip(string.(fieldnames(T)), values)    

    jldopen(path, "w") do file

        file["modeltype"] = string(T)

        foreach(fv -> file[fv[1]] = fv[2], zipped)

        file["means"] = DataFrame(zip(dataset.value_columns, dataset.means)) |> df -> rename!(df, [:var,:mean])

        file["stdevs"] = DataFrame(zip(dataset.value_columns, dataset.stdevs)) |> df -> rename!(df, [:var,:stdev])

        file["transformations"] = transformations

        close(file)
    end
end    

"""
$(FUNCTIONNAME)(path::String)::Tuple{MultivariateModel,Array{Float64,1},Array{Float64,1}}

    Load PCA or PLS model from JLD2 file into a tuple containing the model, variable standard deviations from variable means from model calibration


"""
function loadmodel(modelfile::String)::Tuple{MultivariateModel,Array{Float64,1},Array{Float64,1},Array{String,1},Array{String,1}}

    jldfile = jldopen(modelfile, "r")

    modeltype = jldfile["modeltype"]
    stdevs = jldfile["stdevs"] |> df -> convert(Array{Float64,2}, df[:,[:stdev]])[:]
    means = jldfile["means"] |> df -> convert(Array{Float64,2}, df[:,[:mean]])[:]

    transformations = jldfile["transformations"]

    variables = jldfile["means"][:,:var]

    type::DataType =  modeltype == "PCA" ? PCA : PLS

    values = map(fname -> jldfile[string(fname)], fieldnames(type))

    close(jldfile)

    type(values...),stdevs,means,variables,transformations
end

function selectNumerical(df)

    isnumcol(type) = type <: Union{Missing,Number}

    select(df, eltype.(eachcol(df)) |> (coltypes -> isnumcol.(coltypes)) |> findall)
end

function selectColumns(df,typesel)

    select(df, eltype.(eachcol(df)) |> typesel |> findall)
end

```
Get a onehot representation of a CategoricalArray
```
function onehot(carray::CategoricalArray)    
    lvls = carray |> levels

    hots = [(carray .== lv) |> bvals -> convert(Array{Union{Missing,Float64},1},bvals) for lv in lvls]

    DataFrame(hots) |> v -> rename!(v,Symbol.(lvls))
end

function predictLevel(carray::CategoricalArray,predMatrix::Array{Union{Missing, Float64},2})::Array{String,1}
    predicted_classes = [findmax(row) |> last for row in eachrow(predMatrix)] |> maxidxs -> getindex(levels(carray),maxidxs)
end

function getTransformations(model::T) where T <: MultivariateModel
    return []
end