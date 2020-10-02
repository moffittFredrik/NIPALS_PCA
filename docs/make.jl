using Documenter, NIPALS_PCA

makedocs(
    modules = [NIPALS_PCA],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Fredrik Pettersson",
    sitename = "NIPALS_PCA.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/Fredrikp-ume/NIPALS_PCA.jl.git",
    push_preview = true
)
