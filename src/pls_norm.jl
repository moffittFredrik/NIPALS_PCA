using ArgParse,Match
using DataFrames,CSV,Pipe,CategoricalArrays

struct Yvariable
    label::String
    valtype::Type
end

function load_data(xfile, yfile, yvars::Array{Yvariable,1};idpair::Pair=1 => :id) where T <: String
    xrawdf = CSV.File(xfile) |> DataFrame |> df -> rename(df, idpair)
    yrawdf = @pipe CSV.File(yfile) |> DataFrame |> df -> rename(df, idpair) |> df -> select(df, ["id",getfield.(yvars, :label)...]) 

    merged_df = innerjoin(xrawdf, yrawdf, on=:id)
    
    noncatlabels::Array{String,1} = @pipe yvars |> filter(yc -> yc.valtype != CategoricalArray, _) |> getfield.(_, :label)
    
    xdf = select(merged_df,names(xrawdf)) |> selectNumerical
    yrawdf = select(merged_df,names(yrawdf))
    
    categorical!(yrawdf, Not(["id",noncatlabels...]))

    continousNames = selectNumerical(yrawdf) |> names
    ycontinous = select(yrawdf, ["id",continousNames...])

    categoricalNames = select(yrawdf, eltype.(eachcol(yrawdf)) .<: Union{Missing, CategoricalValue}) |> names

    ycategoricals::Array{DataFrame,1} = @pipe select(yrawdf, categoricalNames) |> # create a new dataframe only containing categorical columns
        onehot.(eachcol(_)) |>  # transform each column to a dataframe with onehot values 
        insertcols!.(_, 1, :id => yrawdf[:,:id]) # insert column with row identifierto each dataframe

    ydf = undef
    if hasCategorical(yvars) && hasContinous(yvars)
        ydf = innerjoin(ycontinous, ycategoricals..., on=:id)
    elseif hasCategorical(yvars)
        ydf = ycategoricals |> first
    else
        ydf = ycontinous
    end

    (xdf = xdf, ydf = ydf)
end

function hasContinous(ycols)
    filter(ycol -> ycol.valtype <: Number, ycols) |> !isempty
end

function hasCategorical(ycols)
    filter(ycol -> ycol.valtype == CategoricalArray, ycols) |> !isempty
end

function todaframe(df, colname)
    @pipe onehot(df[:,colname]) |> insertcols!(_, 1, :id => df[:,:id]) 
end

"""
$(FUNCTIONNAME)(modelfile::String,xfile::String, outfile::String)

    Loads model from jld2 file, predicts using xfile and exports residual matrix into .csv file
"""
function predict_xres(modelfile::String, xfile::String, outfile::String)::DataFrame
    pls, stdevs, means, modelvariables, transformations = loadmodel(modelfile)

    #----------------------------------------------------
    #TODO: implement proper handling of transformations
    #= transformations = getTransformations(pls)
    push!(transformations,docenter(means))
    push!(transformations,douvscale(stdevs))

    [transform(preddataset) for transform in transformations] =#
    #----------------------------------------------------

    douv = "uv" in transformations

    outfileext = splitext(outfile)|> last 
    delim = outfileext == ".csv" ? "," : "\t"

    xdf = CSV.File(xfile) |> DataFrame
    preddataset = @pipe xdf |>
        select(_,Symbol.(modelvariables))|>
        parseDataFrame(_) |> 
        normalize!(_, doscale=douv, stdevs=stdevs, means=means)

    YpredVar, Xres = predictY(preddataset, pls)

    outdf::DataFrame = @pipe DataFrame(Xres) |> rename!(_, Symbol.(preddataset.value_columns)) |> hcat(xdf[:,[1]],_) 

    CSV.write(outfile, outdf, delim = delim)

    outdf
end

function docenter(means)
    dataset->dataset.X .-= means'    
end

function douvscale(stdevs)
    dataset->dataset.X ./= stdevs'    
end

function splitArrayArgs(parsed_args, field)
    if !isnothing(parsed_args[field])
        return split(parsed_args[field], ";") |> a -> convert(Array{String,1}, a) |> strarr -> filter(str -> length(str) > 1, strarr)
    end

    return []
end

"""
$(FUNCTIONNAME)(xdf::DataFrame, ydf::DataFrame, A::Int64, modelfile::String; filters = [dataset -> dataset.mvs .< 0.25])

    Calibrates PLS model based on datatypes in DataFrame for y

    Columns of type CategoricalArray is handled by one-hot precedure

    The calibrated model is saved to specified locations

    Variables with more than 25% missing values are excluded by default
"""
function calibrate_model(xdf::DataFrame, ydf::DataFrame, A::Int64, modelfile::String; filters = [dataset -> dataset.mvs .< 0.25], transformations=["center","uv"])
    
    uvscale = "uv" in transformations

    # allow some missingness in x data
    xobsfilters = [ds -> calcObsMissingValues(ds) .<= 0.5]

    # do not allow any missing values observations in y dataset
    yobsfilters = [ds -> calcObsMissingValues(ds) .== 0]

    # parse datasets to dataframes
    xdataset = xdf |> parseDataFrame
    ydataset = ydf |> parseDataFrame

    # filter datasets for sample missingness
    xobsmask = filterObservations(xdataset,filters=xobsfilters)
    yobsmask = filterObservations(ydataset,filters=yobsfilters)

    # check overlap between datasets for inclusion of observations
    obsmask = xobsmask .& yobsmask

    # filter variables for both datasets using the same filter list
    xvarmask = filterVariables(xdataset,filters=filters)
    yvarmask = filterVariables(ydataset,filters=filters)

    # copy datasets from filter masks 
    xdataset = copydataset(xdataset,obsmask,xvarmask) |> ds -> normalize(ds,doscale=uvscale)
    ydataset = copydataset(ydataset,obsmask,yvarmask) |> ds -> normalize(ds,doscale=true)

    # calculate PLS model
    pls = calcPLS(xdataset, ydataset, A)

    # save PLS model to disk
    savemodel(pls, xdataset, modelfile,transformations)

    pls
end

"""
$(FUNCTIONNAME)(parsed_args::Dict{String,Any})

    Calibrates PLS model based on datatypes in DataFrame for y

    Columns of type CategoricalArray is handled by one-hot precedure

    The calibrated model is saved to specified locations
"""
function calibrate_model(parsed_args::Dict{String,Any})::MultivariateModel

    mvcutoff = parsed_args["mvcutoff"]
    minvalues = parsed_args["minvalues"]
    uvscale = parsed_args["uvscale"]

    transformations = ["center"]
    if uvscale
        push!(transformations,"uv")
    end

    varfilters = [dataset -> dataset.mvs .<= mvcutoff, dataset -> dataset.varvalues .>= minvalues]

    yvars::Array{Yvariable,1} = Array{Yvariable,1}()

    @pipe splitArrayArgs(parsed_args, "ycategorical") |> Yvariable.(_, CategoricalArray) |> append!(yvars, _)

    @pipe splitArrayArgs(parsed_args, "ycontinous") |> Yvariable.(_, Float64) |> append!(yvars, _)

    yvars |> println

    if !isempty(yvars)
        xfile = parsed_args["xfile"]
        yfile = parsed_args["yfile"]
        modelfile = parsed_args["modelfile"]

        A = parsed_args["components"]

        xdf, ydf = load_data(xfile, yfile, yvars)

        calibrate_model(xdf, ydf, A, modelfile,filters=varfilters, transformations = transformations)
    end
end

"""
$(FUNCTIONNAME)(model::T, dataset::Dataset, name::String) where T <: MultivariateModel

    Save PCA or PLS model as JLD2 file
"""
function correct(parsed_args)
    xfile = parsed_args["xfile"]
    modelfile = parsed_args["modelfile"]
    outfile = parsed_args["outfile"]

    predict_xres(modelfile, xfile, outfile)    
end
