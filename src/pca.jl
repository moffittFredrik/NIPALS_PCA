using Statistics
using DataFrames
using CSV
using Random
using DocStringExtensions
using JLD2
using FileIO

#include("structs.jl")

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
$(FUNCTIONNAME)(dataset::Dataset, comps::Int64; normalize::Bool = false)

    Calculates a PCA model

# Examples
```julia
julia> calcPCA(datset,3,normalize=true)
```
"""
function calcPCA(dataset::Dataset, comps::Int64; normalize::Bool = false)::PCA

    X::Array{Union{Missing, Float64},2} = dataset.X

    n,k = size(X)

    T = Array{Float64,1}[]
    #T = zeros(n,comps)
    P = Array{Float64,1}[]
    #P = zeros(k,comps)

    # initialize by setting zeroes in missing values
    X[.~dataset.xmask] .= 0

    dCrit = 1e-23
    maxIter = 1000

    ssx_orig = sum(X.^2)

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

        println("Iter:$(iter) ConvValue:$(diff)")

        X -= t*p'
        
        # after calculating each component, set zeroes in missing values
        X[.~dataset.xmask] .= 0
    end

    # convert Array of Array to 2D array
    PCA(DataFrame(hcat(T...),[Symbol("t$(a)") for a in range(1,stop=comps)]),DataFrame(hcat(P...),[Symbol("p$(a)") for a in range(1,stop=comps)]))
end    

export norm

function tryparsem(T, str)
    something(tryparse(T, str), missing)
end   

function createTestMatrix(irisFile)
    df = CSV.File(irisFile,header=false) |> DataFrame!
    values=convert(Array{Union{Missing, Float64},2},df[1:end-1])

    mask = convert(BitArray,[rand()<0.2 for i in 1:nrow(df), j in 1:(ncol(df)-1)])

    #mask = bitrand(rng,nrow(df),ncol(df)-1) |> (x->reshape(x,nrow(df),ncol(df)-1))

    values[mask] .= missing

    mv_df = DataFrame(values)

    insert!(mv_df, 1, df[end], :type)
    #mv_df[:type] = df[end]

    return mv_df

end
