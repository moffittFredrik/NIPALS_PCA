using PackageCompiler
using Pipe

function sysimagegen(sysimage_path::String)

    execfile = @pipe pathof(NIPALS_PCA) |> splitpath |> _[1:end - 1] |> joinpath(_..., "precompiler", "precompile_package.jl")

    create_sysimage([:NIPALS_PCA,:Pipe], sysimage_path=sysimage_path, precompile_execution_file=execfile)
end