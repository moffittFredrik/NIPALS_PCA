# NIPALS_PCA
A Julia package for calculating PCA and PLS using the NIPALS implementation. Both models handles missing values

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

## Example
![getstarted](../img/gettingStarted.png)

## On cluster using Singularity
A prebundled singularity container for Julia can be accessed at /share/data2/applications/singularity_images/bbsrTools.sif

-C, is required to utilize packages installed in the container. If this flag is not used, Julia will load packages from the users home directory 


-B, binds folders to be accessible to the container. Repeated -B flags can be entered for multiple binds


```bash
module load singularity/3.10
singularity run -C -B ../data:/data  path/to/bbsrJuliaTools2.sif
```

or if running script

```bash
module load singularity/3.10
singularity run -C -B ../data:/data  path/to/bbsrJuliaTools2.sif myScript.jl args
```
## Tutorial
### PCA modelling
From Julia REPL

1)Load package
```julia
using NIPALS_PCA
```
2)Load dataset from .csv file to DataFrame
```julia
x_df = loadIrisData()
```
3)Create dataset and apply normalize to mean center data
```julia
xdataset = parseDataFrame(x_df) |> normalize
``` 
4)Calculate PCA model
```julia
pca = calcPCA(xdataset, 3)
```
5)Calculate variances for model
```julia
calcVariances(xdataset,pca)
```

### PLS normalization
The PLS normalization workflow can either be run from a script or from an interactive Julia session. The default script can be find in src/scripts/plsnorm.jl

#### Run from script
```bash
julia src/scripts/plsnorm.jl \
--xfile /path/to/xmatrix.txt \
--yfile /path/to/ymatrix.txt \
--ycategorical "colname" \
--ycontinous "colname1;colname2" \
--mode calibrate \
--modelfile model.jld2 \
--outfile output_file.csv
```

#### Run from interactive session
```julia
using NIPALS_PCA

parsed_args=Dict{String,Any}("xfile" => "/path/to/xmatrix.txt","ycategorical" => "colname","yfile" => "/path/to/ymatrix.txt","modelfile" => "model.jld2","ycontinous" => "colname1;colname2","mode" => "calibrate","outfile" => "output_file.csv","components" => 3)

#to calibrate
calibrate_model(parsed_args)

#to correct
correct(parsed_args)
```

#### Run on cluster using Singularity image
The folder(s) used for reading and writing needs a Bind. In the example below<br> 
<b>-B ../data/:/data</b> 
<br>
data will be accessible from the root level within the container. Multiple folders can be bound. 
```bash
module load singularity/3.10

singularity run --app plsnorm  -C -B ../data/:/data bbsrJuliaTools2.sif \
--xfile /data/xmatrix.txt \
--yfile /data/ymatrix.txt \
--ycategorical "colname" \
--ycontinous "colname1;colname2" \
--mode calibrate \
--modelfile model.jld2 \
--outfile output_file.csv
```

Get help
```bash
julia plsnorm.jl --help
```
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
calibrate_model

correct

predict_xres
```
