# NIPALS_PCA
A Julia package for calculating PCA and PLS using the NIPALS implementation. Both models handles missing values

<b>For more information open [documentation](./docs/NIPALS_PCA.jl.pdf) (CI/CD is currently failing due to SSL issue)</b><br> 

The package contains data structures for models and datasets

## Installation
In Julia add https://gitlab.moffitt.usf.edu:8000/Bios2Projects/NIPALS_PCA as [unregistered package](https://julialang.github.io/Pkg.jl/v1.1/managing-packages/)
```julia-repl
using Pkg
Pkg.add("https://gitlab.moffitt.usf.edu:8000/Bios2Projects/NIPALS_PCA")
```

## Running Julia REPL
Julia can be started using 
1. The base installation
2. From Singularity container (in progress, details pending)
3. Utilizing the downloaded folder as a local environment.

To activate NIPALS_PCA as local environment
```bash
cd path/to/cloned/NIPALS_PCA
julia --project=.
```
## Loading package
```julia
using NIPALS_PCA
``` 

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/Fredrikp-ume/NIPALS_PCA.jl.svg?branch=master)](https://travis-ci.com/Fredrikp-ume/NIPALS_PCA.jl)
[![codecov.io](http://codecov.io/github/Fredrikp-ume/NIPALS_PCA.jl/coverage.svg?branch=master)](http://codecov.io/github/Fredrikp-ume/NIPALS_PCA.jl?branch=master)
<!--
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://Fredrikp-ume.github.io/NIPALS_PCA.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://Fredrikp-ume.github.io/NIPALS_PCA.jl/dev)
-->
