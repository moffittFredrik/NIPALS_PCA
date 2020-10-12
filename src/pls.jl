using DataFrames,CSV,Pipe
using DocStringExtensions

#include("structs.jl")
#= x_df = CSV.File("/Users/petterhf/projects/bioinformatics/normalization/x.txt") |> DataFrame!
y_df = CSV.File("/Users/petterhf/projects/bioinformatics/normalization/y.txt") |> DataFrame!

yda_df = CSV.File("/Users/petterhf/projects/bioinformatics/normalization/input_matrix_for_plsda_training_transposed.txt") |> DataFrame!

meta_df = CSV.File("/Users/petterhf/projects/bioinformatics/normalization/avatar_myeloma_debatched_v041_metadata.txt") |> DataFrame!

xdataset = @pipe parseDataFrame(x_df) |> normalize(_)
ydataset = @pipe parseDataFrame(y_df) |> normalize(_)

pls = calcPLS(xdataset,ydataset,3)

savemodel(pls,xdataset,"testing.jld2")

loadedmodel, stdevs, means = loadmodel("testing.jld2")

preddataset = @pipe parseDataFrame(x_df) |> normalize(_,doscale=true,stdevs=_.stdevs,means=_.means)

predictY(preddataset,loadedmodel) =#


"""
$(FUNCTIONNAME)(xdataset::Dataset,ydataset::Dataset,comps::Int64,incsamples::Array{Int64,1} = collect(1:size(xdataset.X)[1]))

    Calculates a PLS model
"""
function calcPLS(xdataset::Dataset,ydataset::Dataset,comps::Int64,incsamples::Array{Int64,1} = collect(1:size(xdataset.X)[1]))

    X::Array{Union{Missing, Float64},2} = xdataset.X[incsamples,:]
    Y::Array{Union{Missing, Float64},2} = ydataset.X[incsamples,:]

    xmask::BitArray{2} = xdataset.xmask[incsamples,:]
    ymask::BitArray{2} = ydataset.xmask[incsamples,:]

    hasmv = sum(.~xdataset.xmask) > 0

    n,k = size(X)
    T = Array{Float64,1}[]
    P = Array{Float64,1}[]
    C = Array{Float64,1}[]
    W = Array{Float64,1}[]
    U = Array{Float64,1}[]

    # initialize by setting zeroes in missing values
    X[.~xmask] .= 0
    Y[.~ymask] .= 0

    dCrit = 1e-12
    maxIter = 1000

    ssx_orig = sum(X.^2)

    for i in range(1,stop=comps)

        iter = 0
        diff = 1

        t = []
        p = []
        c = []
        w = []
        u = convert(Array{Float64,1},X[:,1])

        while diff > dCrit && iter < maxIter

            t,w,u,c,diff = nipals(X, Y, xdataset, u, xmask,hasmv)

            iter+=1
        end

        #println("Iter:$(iter) ConvValue:$(diff)")

        p = X't

        # if X has missing values
        if hasmv
            pcorr = ((t.^2)'*xmask)'
            p ./= pcorr

            p[pcorr.==0].=0
        else
            p /= t't    
        end    

        Xcomp = t*p'
        Xcomp[.~xmask] .= 0
        X -= Xcomp
        X[.~xmask] .= 0

        Ycomp = t*c'
        Ycomp[.~ymask] .= 0
        Y -= Ycomp
        Y[.~ymask] .= 0

        matrixType = Array{Float64,1}

        push!(T,convert(Array{Float64,1},t))
        push!(P,convert(Array{Float64,1},p))
        push!(C,convert(Array{Float64,1},c))
        push!(W,convert(Array{Float64,1},w))
        push!(U,convert(Array{Float64,1},u))

    end

    tocompindices = string.(range(1,stop=comps))

    PLS(
        DataFrame(hcat(T...),"t".*tocompindices),
        DataFrame(hcat(P...),"p".*tocompindices),
        DataFrame(hcat(C...),"c".*tocompindices),
        DataFrame(hcat(W...),"w".*tocompindices),
        DataFrame(hcat(U...),"u".*tocompindices)
    )
end

function numcomps(model::MultivariateModel)
    size(model.P,2)
end

function predictY(xdataset::Dataset, model::PLS, incsamples::Array{Int64,1} = collect(1:size(xdataset.X,1)); comps::Int64 = numcomps(model))
    
    P = convert(Array{Float64,2},model.W)
    W = convert(Array{Float64,2},model.P)
    C = convert(Array{Float64,2},model.C)
    
    predictY(xdataset,W,P,C,incsamples,comps=comps)
end

function predictY(xdataset::Dataset, W::Array{Float64,2}, P::Array{Float64,2}, C::Array{Float64,2}, incsamples::Array{Int64,1} = collect(1:size(xdataset.X,1));comps::Int64=numcomps(model))

    X::Array{Union{Missing, Float64},2} = xdataset.X[incsamples,:]
    xmask::BitArray{2} = xdataset.xmask[incsamples,:]

    Ycvs::Array{Array{Union{Missing, Float64},2}} = []

    for a in 1:comps
        tpred = X*W[:,a]

        # if X has missing values
        tpred ./= xmask*W[:,a].^2

        Ycv = tpred*C[:,a]'

        push!(Ycvs,Ycv)
        
        Xpred = tpred * P[:,a]'

        X -= Xpred

        X[.~xmask] .= 0 
    end

    YpredVar = cat(Ycvs...,dims=3)
    Xres = X

    return (;YpredVar,Xres)
end

function crossvalidate(xdataset::Dataset, ydataset::Dataset, comps::Int64, cvgroups::Int64=7)

    cvgroups = createCVgroups(xdataset.X,cvgroups);

    Ycv = Array{Union{Missing, Float64}}(missing, (size(ydataset.X)...,comps))

    for (train,test) in cvgroups
        cvmodel = calcPLS(xdataset,ydataset,comps, train)      

        Ycv[test,:,:] = predictY(xdataset,cvmodel,test,comps=comps).YpredVar
    end

    return calcQ2(Ycv,ydataset.X)
end

function calcQ2(Ycvs,y)
    Ycv_cum = cumsum(Ycvs,dims=3)

    R2cv_var = sum((y.-Ycv_cum).^2,dims=1)
    R2y_var = sum(y.^2,dims=1)

    Q2y_var = (R2y_var .- R2cv_var) ./ R2y_var

    return Q2y_var
end    

function createCVgroups(df, numgroups::Int64)

    n,k = size(df)

    testsets = [collect(range(step,step=numgroups,stop=n)) for step in 1:numgroups]
    cvgroups = []

    for testset in testsets
        trainset = trues(n)
        trainset[testset] .= false
        push!(cvgroups,(findall(trainset),testset)) 
    end
    
    cvgroups
end

function nipals(X::Array{Union{Missing, Float64},2}, Y::Array{Union{Missing, Float64},2}, dataset::Dataset, u::Array{Float64,1}, xmask::BitArray{2},mv::Bool)

    w=X'*u

    # if X has missing values
    # valueCols=vec(sum(xmask,dims=1).>0)
    # novalueCols=.~valueCols
    if mv
        wcorr = ((u.^2)'*xmask)'
        w ./= wcorr

        w[wcorr.==0].=0
    end

    w/=norm(w)

    t =X*w
    
    # if X has missing values
    if mv
        t ./= xmask*w.^2
    end

    c = Y't/(t'*t)

    unew = Y*c / (c'*c)

    diff =  rootsq(u .- unew)/rootsq(u)

    t,w,unew,c,diff

end

function rootsq(array)
    array .^2 |> sum |>sqrt
end