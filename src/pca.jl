using Statistics
using DataFrames
using CSV
using Random
using DocStringExtensions
using JLD2
using FileIO
using Pipe

function nipals(X::Array{Union{Missing, Float64},2}, dataset::Dataset, t::Array{Union{Missing, Float64},1})

    p=X'*t

    # if X has missing values
    if dataset.mv
        pcorr = ((t.^2)'*dataset.xmask)'
        p ./= pcorr
        p[pcorr.==0].=0
    end

    p/=norm(p)

    tnew = X*p

    # if X has missing values
    if dataset.mv
        tnew ./= dataset.xmask*p.^2
    end

    diff = sum((t .- tnew) .^2)

    tnew,p,diff
end

export calcPCA

"""
$(FUNCTIONNAME)(dataset::Dataset, comps::Int64=3; dCrit = 1e-23, maxIter = 1000)::PCA

    Calculates a PCA model

# Examples
```julia
julia> calcPCA(datset,3)
```
"""
function calcPCA(dataset::Dataset, comps::Int64=3; dCrit = 1e-23, maxIter = 1000, minimal=false)::PCA

    X::Array{Union{Missing, Float64},2} = dataset.X

    r2x_cum = Float64[]

    T = Array{Float64,1}[]
    #T = zeros(n,comps)
    P = Array{Float64,1}[]
    #P = zeros(k,comps)

    # initialize by setting zeroes in missing values
    X[.~dataset.xmask] .= 0

    #dCrit = 1e-23
    #maxIter = 1000
    if !minimal
        ssx_orig = sum(X.^2)
    end

    for i in range(1,stop=comps)

        iter = 0
        diff = 1

        t = X[:,1]
        p = []

        while diff > dCrit && iter < maxIter

            t,p,diff = nipals(X, dataset, t)

            iter+=1
        end

        #convert to disallow missing values in components
        push!(T,convert(Array{Float64,1},t))
        push!(P,convert(Array{Float64,1},p))

        X -= t*p'
        
        if !minimal
            println("Iter:$(iter) ConvValue:$(diff)")
           
            # calculate sum of squares
            ssx = sum(X.^2)

            # Calculate cumulative explained variation
            expl_var = (ssx_orig - ssx) / ssx_orig;

            push!(r2x_cum,expl_var)
        end    
        
        # after calculating each component, set zeroes in missing values
        X[.~dataset.xmask] .= 0

    end

    r2x = minimal ? [] : r2x_cum .- [0,r2x_cum[1:end-1]...]

    eigenvalues = minimal ? [] : r2x .* min(size(dataset.X)...)

    # convert Array of Array to 2D array
    PCA(
        DataFrame(hcat(T...),[Symbol("t$(a)") for a in range(1,stop=comps)]),
        DataFrame(hcat(P...),[Symbol("p$(a)") for a in range(1,stop=comps)]),
        r2x_cum,
        r2x,
        eigenvalues)
end       

export norm

function tryparsem(T, str)
    something(tryparse(T, str), missing)
end   

function createTestMatrix(irisFile)
    df = CSV.File(irisFile,header=false) |> DataFrame
    values=convert(Array{Union{Missing, Float64},2},df[:,1:end-1])

    mask = convert(BitArray,[rand()<0.2 for i in 1:nrow(df), j in 1:(ncol(df)-1)])

    values[mask] .= missing

    mv_df = DataFrame(values)

    insertcols!(mv_df, 1, :type=>df[:,end])

    return mv_df

end

function xpred(t::Array{Float64,1},p::Array{Float64,1})
    xpred = t*p'
end

function fill!(matrix,dataset::Dataset;value::Float64=0.0)
    matrix[.~dataset.xmask] .= 0

    matrix
end

function toarray(matrix::Array{Float64,3})
    A = size(matrix)[3]

    reshape(matrix,(A,1,1))[:]   
end

"""
$(FUNCTIONNAME)(dataset::Dataset, model::PCA)::NamedTuple

    Calculates r2x, r2x_cum and eigenvalues for all components in PCA model

# Examples
```julia-repl
julia> r2x,r2x_cum,eigenvalues = calcPCA(datset,model)
```
"""
function calcVariances(dataset::Dataset, model::PCA)::NamedTuple

    bycomp = [1,2]

    fillmissings(dataset)

    T = model.T |> T-> convert(Matrix,T)
    P = model.P |> P-> convert(Matrix,P)       

    A = numcomps(model)

    ssx_orig = sum(dataset.X.^2)

    r2x_cum = Float64[]

    Xres = dataset.X |> copy

    for a in 1:A
        Xres .-= T[:,a]*P[:,a]' 
        
        fill!(Xres,dataset)

        push!(r2x_cum,((ssx_orig - sum(Xres.^2)) / ssx_orig))
    end

    r2x = r2x_cum .- [0,r2x_cum[1:end-1]...]

    eigenvalues = r2x .* min(size(dataset.X)...)

    (;r2x,r2x_cum,eigenvalues)

end
