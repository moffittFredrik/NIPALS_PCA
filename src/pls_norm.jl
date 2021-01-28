using ArgParse,Match
using DataFrames,CSV,Pipe,CategoricalArrays,NIPALS_PCA

struct Yvariable
    label::String
    valtype::Type
end

function load_data(xfile,yfile,ycols::Array{Yvariable,1};idpair::Pair = 1=>:id) where T <: String
    xrawdf = CSV.File(xfile) |> DataFrame! |> df -> rename(df,idpair)
    yrawdf = @pipe CSV.File(yfile) |> DataFrame! |> df -> rename(df,idpair) |> df -> select(df,["id",getfield.(ycols,:label)...]) 
    
    noncatlabels::Array{String,1} = @pipe ycols |> filter(yc->yc.valtype != CategoricalArray,_) |> getfield.(_,:label)
    categorical!(yrawdf,Not(["id",noncatlabels...]))

    merged_df = innerjoin(xrawdf,yrawdf,on=:id)

    xdf = merged_df |> df -> select(df,Not(getfield.(ycols,:label)))|> selectNumerical

    continousNames = selectColumns(yrawdf,coltypes -> coltypes .== Float64) |> names

    ycontinous = select(yrawdf,["id",continousNames...])
    ycategoricals = @pipe selectColumns(yrawdf,coltypes -> coltypes .<: CategoricalValue) |> names |> todaframe.(Ref(yrawdf),_)

    ydf=undef
    if hasCategorical(ycols) && hasContinous(ycols)
        ydf = innerjoin(ycontinous,ycategoricals...,on=:id)
    elseif hasCategorical(ycols)
        ydf = ycategoricals |> first
    else
        ydf = ycontinous
    end

    (xdf=xdf,ydf=ydf)
end

function hasContinous(ycols)
    filter(ycol->ycol.valtype <: Number,ycols) |> !isempty
end

function hasCategorical(ycols)
    filter(ycol->ycol.valtype == CategoricalArray,ycols) |> !isempty
end

function todaframe(df,colname)
    @pipe onehot(df[:,colname]) |> insert!(_, 1, df.id, :id) 
end

function predict_xres(modelfile::String,xfile::String, outfile::String)
    pls, stdevs, means = loadmodel(modelfile)

    xdf = CSV.File(xfile) |> DataFrame!
    preddataset = @pipe parseDataFrame(xdf) |> normalize(_,doscale=true,stdevs=_.stdevs,means=_.means)

    YpredVar,Xres = predictY(preddataset,pls)

    @pipe DataFrame(Xres)|> rename!(_,Symbol.(preddataset.value_columns)) |> CSV.write(outfile,_)

end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin

        "--xfile"
            help = "x data"  
            required = true  

        "--yfile"
            help = "metadata.txt"

        "--ycontinous"
            help = "y variable names"

        "--ycategorical"
            help = "y variable names"

        "--mode"
            help = "calibrate or correct"
            required = true

        "--modelfile"
            help = "name of model file to read or write"
            required = true

        "--components", "-A"
            help = "specify the number of components to use"
            arg_type = Int
            default = 3

         "--outfile", "-o"
            help = "output corrected data file name.txt"
    end

    return parse_args(s)
end

function splitArrayArgs(parsed_args,field)
    if !isnothing(parsed_args[field])
        return split(parsed_args[field],";") |> a->convert(Array{String,1},a) |> strarr -> filter(str->length(str) >1 ,strarr)
    end

    return []
end

function calibrate_model(x::DataFrame,y::DataFrame,A::Int64, modelfile::String)
    xdataset = x |> parseDataFrame |> normalize
    ydataset = y |> selectNumerical |>  parseDataFrame |> normalize

    pls = calcPLS(xdataset,ydataset,A)

    savemodel(pls,xdataset,modelfile)

    pls
end

function calibrate_model(parsed_args)

    yvars::Array{Yvariable,1} = []

    @pipe splitArrayArgs(parsed_args,"ycategorical") |> Yvariable.(_,CategoricalArray) |> append!(yvars,_)

    @pipe splitArrayArgs(parsed_args,"ycontinous") |> Yvariable.(_,Float64) |> append!(yvars,_)

    yvars |> println

    if !isempty(yvars)
        xfile = parsed_args["xfile"]
        yfile = parsed_args["yfile"]
        modelfile = parsed_args["modelfile"]

        A = parsed_args["components"]

        xdf,ydf = load_data(xfile,yfile,yvars)

        calibrate_model(xdf,ydf,A,modelfile)
    end
end

function correct(parsed_args)
    xfile = parsed_args["xfile"]
    modelfile = parsed_args["modelfile"]
    outfile = parsed_args["outfile"]


    predict_xres(modelfile,xfile,outfile)    
end

function main()
    parsed_args = parse_commandline()
    parsed_args |> println

    xfile = parsed_args["xfile"]

    @match parsed_args["mode"] begin
        "calibrate" => calibrate_model(parsed_args)
        "correct" => correct(parsed_args)
    end
end

main()
