# NIPALS_PCA
A package for calculating PCA and PLS using the NIPALS implementation. Both models handles missing values

The package contains data structures for models and datasets

## Installation
In Julia add https://gitlab.moffitt.usf.edu:8000/Bios2Projects/NIPALS_PCA as [unregistered package](https://julialang.github.io/Pkg.jl/v1.1/managing-packages/)

## Loading package
```julia
using NIPALS_PCA
``` 

## Tutorial
### PCA modelling
1. Load package
```julia
using NIPALS_PCA
```
2. Load dataset from .csv file to DataFrame
```julia
x_df = loadIrisData()
```
3. Create dataset and apply normalize to mean center data
```julia
xdataset = parseDataFrame(x_df) |> normalize
``` 
4. Calculate PCA model
```julia
pca = calcPCA(xdataset, 3)
```
5. Calculate variances for model
```julia
calcVariances(xdataset,pca)
```

### PLS normalization
The PLS normalization work flow can either be run from a script or from an interactive Julia session

#### run from script
```bash
julia plsnorm.jl \
--xfile /path/to/xmatrix.txt \
--yfile /path/to/ymatrix.txt \
--ycategorical "colname" \
--ycontinous "colname1;colname2" \
--mode calibrate \
--modelfile model.jld2 \
--outfile output_file.csv
```

#### run from interactive session
```julia
using NIPALS_PCA

parsed_args=Dict{String,Any}("xfile" => "/path/to/xmatrix.txt","ycategorical" => "colname","yfile" => "/path/to/ymatrix.txt","modelfile" => "model.jld2","ycontinous" => "colname1;colname2","mode" => "calibrate","outfile" => "output_file.csv","components" => 3)

#to calibrate
calibrate_model(parsed_args)

#to correct
correct(parsed_args)
```

To get help
```bash
julia plsnorm.jl --help
```
#### Calibrate and save model

#### Normalize data from saved model

### Structures
```@docs
Dataset
PCA
PLS
```

### Functions
#### General functionality
```@docs
calcPCA

calcPLS

calcVariances

loadmodel

savemodel
```

#### PLS normalization
```@docs
correct

calibrate_model

predict_xres
```
