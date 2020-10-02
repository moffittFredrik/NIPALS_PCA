using NIPALS_PCA
using Test
using CSV
using DataFrames
#include("../src/structs.jl")

@testset "PCA" begin

    x_df = CSV.File("./data/iris/iris.csv", header=false) |> DataFrame!
    select!(x_df,eltypes(x_df) .<: Number )
    xdataset = parseDataFrame(x_df) |> dataset -> normalize(dataset)

    calcPCA(xdataset,3)
    @test true
end

@testset "PLS" begin
    @test true
end