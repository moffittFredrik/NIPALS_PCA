using NIPALS_PCA
using Test
using CSV
using DataFrames
using Pipe
# include("../src/structs.jl")

@testset "PCA" begin

    isnumcol(type) = type <: Union{Missing,Number}

    pwd()|> println
    pathof(NIPALS_PCA) |> println

    irisdir = @pipe pathof(NIPALS_PCA) |> splitpath |> _[1:end - 2] |> joinpath(_..., "test", "data", "iris")

    erange = range(1,length=4,step=2)

    x_df = CSV.File(joinpath(irisdir, "Dataset_01.txt"), header=false) |> DataFrame
    #x_df = CSV.File(joinpath(irisdir, "iris.csv"), header=false) |> DataFrame

    #x_df = x_df[:,erange]
    select!(x_df, eltype.(eachcol(x_df)) |> coltypes -> isnumcol.(coltypes) |> findall)
    #xdataset = x_df |> parseDataFrame |> dataset -> filterDataset(dataset, filters=[dataset -> dataset.mvs .< 0.25]) |> normalize
    xdataset = x_df |> parseDataFrame |> ds -> normalize(ds,doscale=true)

    pca = calcPCA(xdataset, 4)

    aeb_r2x = [0.7277,0.2303,0.0368,0.0052]

    @test isapprox(pca.r2x,aeb_r2x,atol=0.0001)
end

@testset "PLS" begin
    @test true
end