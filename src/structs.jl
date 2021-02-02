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
        copy(dataset.ranges)
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
    valrng = missing

    try
        valrng = maximum(skipmissing(array))-minimum(skipmissing(array)) 
    catch
        valrng = missing
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

    Xtr = X[:,inc_mask]

    xmask = (!ismissing).(Xtr)

    mvs = sum(.~xmask,dims=1)[:] |> values -> convert(Array{Float64},values) |> vs -> vs ./= size(X)[1]

    Dataset(Xtr, var_means, var_stdevs, value_columns, xmask, sum(.~xmask) > 0, mvs,ranges)
end

function parseDataFrame(df::AbstractDataFrame)

    value_columns = names(df[:,eltype.(eachcol(df)) |> coltypes -> isnumcol.(coltypes)])

    X = convert(Array{Union{Missing,Float64},2}, df[:,value_columns])

    return parseMatrix(X,value_columns)
end

function filterDataset(dataset::Dataset; filters=[])::Dataset
    mask = filterVariables(dataset,filters=filters)

    Dataset(
        dataset.X[:,mask],
        dataset.means[mask],
        dataset.stdevs[mask],
        dataset.value_columns[mask],
        dataset.xmask[:,mask],
        sum(.~(dataset.xmask[:,mask])) > 0,
        dataset.mvs[mask],
        dataset.ranges[mask]
    )
end

function filterVariables(dataset::Dataset; filters=[])

    allvars = createDefaultVariableMask(dataset)

    varmask = @pipe [filter(dataset) for filter in filters] |> [allvars,_...] |> tomatrix |> m->all(m,dims=2) |> vec
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

function createDefaultVariableMask(dataset::Dataset)::BitArray{1}
    BitArray{1}(ones(nvars(dataset)))
end

function normalize!(dataset::Dataset; doscale::Bool=false, stdevs=dataset.stdevs, means=dataset.means)

    mean_mask = (!isnan).(dataset.means)
    std_mask = hasVariation.(dataset.stdevs)

    inc_mask = mean_mask .& std_mask

    dataset.X .-= dataset.means[inc_mask]'

    if doscale
        dataset.X ./= dataset.stdevs[inc_mask]'
    end

    dataset.X[.~dataset.xmask] .= 0

    dataset
end

function normalize(dataset::Dataset; doscale::Bool=false, stdevs=dataset.stdevs, means=dataset.means)

    dscopy = copydataset(dataset)

    normalize!(dscopy)
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
function savemodel(model::T, dataset::Dataset, path::String) where T <: MultivariateModel

    values = fieldnames(T) |> fns -> getfield.(Ref(model), fns)

    zipped = zip(string.(fieldnames(T)), values)    

    jldopen(path, "w") do file

        file["modeltype"] = string(T)

        foreach(fv -> file[fv[1]] = fv[2], zipped)

        file["means"] = DataFrame(zip(dataset.value_columns, dataset.stdevs)) |> df -> rename!(df, [:var,:stdev])

        file["stdevs"] = DataFrame(zip(dataset.value_columns, dataset.means)) |> df -> rename!(df, [:var,:mean])

        close(file)
    end
end    

"""
$(FUNCTIONNAME)(path::String)::Tuple{MultivariateModel,Array{Float64,1},Array{Float64,1}}

    Load PCA or PLS model from JLD2 file into a tuple containing the model, variable standard deviations from variable means from model calibration


"""
function loadmodel(path::String)::Tuple{MultivariateModel,Array{Float64,1},Array{Float64,1}}

    jldfile = jldopen(path, "r")

    modeltype = jldfile["modeltype"]
    stdevs = jldfile["stdevs"] |> df -> convert(Array{Float64,2}, df[:,[:mean]])[:]
    means = jldfile["means"] |> df -> convert(Array{Float64,2}, df[:,[:stdev]])[:]

    type::DataType =  modeltype == "PCA" ? PCA : PLS

    values = map(fname -> jldfile[string(fname)], fieldnames(type))

    close(jldfile)

    type(values...),stdevs,means
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

    hots = [(carray .== lv) |> bvals -> convert(Array{Float64,1},bvals) for lv in lvls]

    DataFrame(hots) |> v -> rename!(v,Symbol.(lvls))
end

function predictLevel(carray::CategoricalArray,predMatrix::Array{Union{Missing, Float64},2})::Array{String,1}
    predicted_classes = [findmax(row) |> last for row in eachrow(predMatrix)] |> maxidxs -> getindex(levels(carray),maxidxs)
end