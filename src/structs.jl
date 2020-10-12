using DataFrames
using Statistics
using JLD2
using CategoricalArrays

struct Dataset
    X::Array{Union{Missing,Float64},2}
    means::Array{Float64,1}
    stdevs::Array{Float64,1}
    value_columns::Array{String,1}
    xmask::BitArray{2}
    mv::Bool
end

abstract type MultivariateModel end

struct PCA <: MultivariateModel
    T::DataFrame
    P::DataFrame
end

struct PLS <: MultivariateModel
    T::DataFrame
    P::DataFrame
    C::DataFrame
    W::DataFrame
    U::DataFrame
end

function hasVariation(x)
    !isnan(x) && x > 0
end 

function norm(v)
    sqrt(sum(v.^2))
end

isnumcol(type)= type <: Union{Missing,Number}

function parseDataFrame(df::AbstractDataFrame)

    value_columns = names(df[:,eltype.(eachcol(df)) |> coltypes -> isnumcol.(coltypes)])
    #value_columns = names(df[eltypes(df) .<: Union{Missing,Number}])

    X = convert(Array{Union{Missing,Float64},2}, df[:,value_columns])

    var_means::Array{Float64,1} = [mean(skipmissing(col)) for col in eachcol(X) ]
    mean_mask = (!isnan).(var_means)

    var_stdevs::Array{Float64,1} = [std(skipmissing(col)) for col in eachcol(X) ]
    std_mask = hasVariation.(var_stdevs)

    inc_mask = mean_mask .& std_mask

    Xtr = X[:,inc_mask]

    xmask = (!ismissing).(Xtr)

    Dataset(Xtr, var_means, var_stdevs, value_columns, xmask, sum(xmask) > 0)
end

function normalize(dataset::Dataset; doscale::Bool=false, stdevs=dataset.stdevs, means=dataset.means)

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

function savemodel(model::T, dataset::Dataset, name::String) where T <: MultivariateModel

    values = fieldnames(T) |> fns -> getfield.(Ref(model), fns)

    zipped = zip(string.(fieldnames(T)), values)    

    jldopen(name, "w") do file

        file["modeltype"] = string(T)

        foreach(fv -> file[fv[1]] = fv[2], zipped)

        file["means"] = DataFrame(zip(dataset.value_columns, dataset.stdevs)) |> df -> names!(df, [:var,:stdev])

        file["stdevs"] = DataFrame(zip(dataset.value_columns, dataset.means)) |> df -> names!(df, [:var,:mean])

        close(file)
    end
end    

function loadmodel(path::String)::Tuple{MultivariateModel,Array{Float64,1},Array{Float64,1}}

    jldfile = jldopen(path, "r")

    modeltype = jldfile["modeltype"]
    stdevs = jldfile["stdevs"] |> df -> convert(Array{Float64,2}, df[[:mean]])[:]
    means = jldfile["means"] |> df -> convert(Array{Float64,2}, df[[:stdev]])[:]

    type::DataType =  modeltype == "PCA" ? PCA : PLS

    values = map(fname -> jldfile[string(fname)], fieldnames(type))

    close(jldfile)

    type(values...),stdevs,means
end

function selectNumerical(df)

    isnumcol(type) = type <: Union{Missing,Number}

    select(df, eltype.(eachcol(df)) |> coltypes -> isnumcol.(coltypes) |> findall)
end

function onehot(carray::CategoricalArray)    
    lvls = carray |> levels

    hots = [(carray .== lv) |> bvals -> convert(Array{Float64,1},bvals) for lv in lvls]

    DataFrame((;zip(Symbol.(lvls), hots)...))
end

function predictLevel(carray::CategoricalArray,predMatrix::Array{Union{Missing, Float64},2})::Array{String,1}
    predicted_classes = [findmax(row) |> last for row in eachrow(predMatrix)] |> maxidxs -> getindex(levels(carray),maxidxs)
end