using NIPALS_PCA
using Test
using CSV
using DataFrames
# include("../src/structs.jl")

@testset "PCA" begin

    isnumcol(type) = type <: Union{Missing,Number}

    x_df = CSV.File("test/data/iris/iris.csv", header=false) |> DataFrame!
    select!(x_df, eltype.(eachcol(x_df)) |> coltypes -> isnumcol.(coltypes) |> findall)
    xdataset = parseDataFrame(x_df) |> dataset -> normalize(dataset)

    calcPCA(xdataset, 3)
    @test true
end

@testset "PLS" begin
    @test true
end