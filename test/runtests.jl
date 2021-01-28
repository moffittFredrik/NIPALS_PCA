using NIPALS_PCA
using Test
using CSV
using DataFrames
using Pipe
# include("../src/structs.jl")

@testset "PCA" begin

    isnumcol(type) = type <: Union{Missing,Number}

    irisdir = @pipe pathof(NIPALS_PCA) |> splitpath |> _[1:end-2] |> joinpath(_...,"test","data","iris")

    x_df = CSV.File(joinpath(irisdir,"iris.csv"), header=false) |> DataFrame!
    select!(x_df, eltype.(eachcol(x_df)) |> coltypes -> isnumcol.(coltypes) |> findall)
    xdataset = parseDataFrame(x_df) |> normalize

    pca = calcPCA(xdataset, 3)

    calcVariances(xdataset,pca)
    @test true
end

@testset "PLS" begin
    @test true
end