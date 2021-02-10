using ArgParse,Match
using NIPALS_PCA

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin

        "--xfile"
            help = "x data in delimited file with the first row containing variable names and the first column containing observation names"  
            required = true  

        "--yfile"
            help = "y data in delimited file with the first row containing variable names and the first column containing observation names"

        "--ycontinous"
            help = "y variable names to include for model calibration. Multiple entries are separated by ;, example:\"DV200;RIN\""

        "--ycategorical"
            help = "y variable names to include for model calibration"

        "--mode"
            help = "calibrate or correct"
            required = true

        "--uvscale"
            help = "scale variables for unit variance"
            arg_type = Bool
            default = true    

        "--modelfile"
            help = "name of model file to read or write"
            required = true

        "--components", "-A"
            help = "specify the number of components to use"
            arg_type = Int
            default = 3

        "--mvcutoff"
            help = "cutoff for excluding variables from modelling. Values with missingness <= mvcutoff is gets included"
            arg_type = Float64
            default = 0.25

        "--minvalues"
            help = "the number of values that a variable needs to have to not be excluded, numvalues >= minvalues gets included"
            arg_type = Int
            default = 5

         "--outfile", "-o"
            help = "output corrected data filename"
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    parsed_args |> println

    xfile = parsed_args["xfile"]

    @match parsed_args["mode"] begin
        "calibrate" => calibrate_model(parsed_args)
        "correct" => correct(parsed_args)
    end
end

main()