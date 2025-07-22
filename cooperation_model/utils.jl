module utils

using Memoization
using SparseArrays
using ArnoldiMethod
using Dates
using Serialization
using LinearAlgebra
using JSON
using Colors
using Distributions

using StatsBase

# --------------------------
# Strat labels
# --------------------------

const STRAT_ALLC::Int = 0
const STRAT_ALLD::Int = 1
const STRAT_DISC::Int = 2

# --------------------------
# Default norms
# --------------------------

# Common social norms
const snIS::Vector{Float64} = [1.0, 0.0, 1.0, 0.0]
const snSJ::Vector{Float64} = [1.0, 0.0, 0.0, 1.0]
const snSH::Vector{Float64} = [1.0, 0.0, 0.0, 0.0]
const snSS::Vector{Float64} = [1.0, 0.0, 1.0, 1.0]
const snAG::Vector{Float64} = [1.0, 1.0, 1.0, 1.0]
const snAB::Vector{Float64} = [0.0, 0.0, 0.0, 0.0]

# --------------------------
# Memoization
# --------------------------

function clear_memoization_list(functions)
    for f in functions
        Memoization.empty_cache!(f)
    end
end

# --------------------------
# Norm loading from LLM result json
# --------------------------

function extract_norms(json_path::String, model_names::Vector{String})
    """
    Extract all entries for specified models from a JSON file.

    Args:
        json_path::String: Path to the JSON file with results.
        model_names::Vector{String}: List of model names to include.

    Returns:
        Dict{String, Dict{String, Any}}: Dictionary containing all entries for each model.
    """
    # Parse the JSON file
    results = JSON.parsefile(json_path)

    # Initialize the dictionary for storing norms
    aa_norms = Dict{String, Dict{String, Any}}()

    # Iterate through the results and extract data for the specified models
    for entry in results
        model_name = entry["model_name"]
        if model_name in model_names
            # Add all key-value pairs from the entry, excluding the "model_name"
            aa_norms[model_name] = Dict(k => v for (k, v) in entry if k != "model_name" && k != "standard_dev")
        end
    end

    return aa_norms
end

function filter_norms(data::Dict, keys::Vector{String})
    filtered_data = Dict("norm" => data["norm"])  # Always include "norm"
    
    for key in keys
        if haskey(data, key)
            filtered_data[key] = data[key]
        end
    end
    
    return filtered_data
end

function extract_norm_standard_dev(json_path::String, model_names::Vector{String})
    # Parse the JSON file
    results = JSON.parsefile(json_path)

    # Initialize the dictionary for storing norms
    aa_std_dev = Dict{String, Dict{String, Any}}()

    # Iterate through the results and extract data for the specified models
    for entry in results
        model_name = entry["model_name"]
        if model_name in model_names
            # Add all key-value pairs from the entry, excluding the "model_name"
            aa_std_dev[model_name] = Dict(k => v for (k, v) in entry["standard_dev"])
            aa_std_dev[model_name]["mean_vector"] = Vector{Float64}(aa_std_dev[model_name]["mean_vector"])
            aa_std_dev[model_name]["cov_matrix"] = hcat(aa_std_dev[model_name]["cov_matrix"]...)
            aa_std_dev[model_name]["std_vector"] = Vector{Float64}(aa_std_dev[model_name]["std_vector"])
        end
    end


    return aa_std_dev
end

# --------------------------
# Make color map for norms
# --------------------------

function make_color_map_norms()::Dict{String, Any}
    # Initialize color_map as an empty dictionary
    color_map = Dict{String, Any}()

    color_map["norm_M_to_M"] = :red
    color_map["norm_M_to_F"] = :blue
    color_map["norm_F_to_M"] = :green
    color_map["norm_F_to_F"] = :orange


    tag_shades = [RGB(0.5, 0.5, 0.5), RGB(0.6, 0.6, 0.6), RGB(0.7, 0.7, 0.7), RGB(0.8, 0.8, 0.8), RGB(1.0, 1.0, 1.0)]
    tags = ["no-topic", "neutral", "non-neutral", "explicit-neutral", "explicit-non-neutral"]

    for (i, tag) in enumerate(tags)
        color_map["norm_tag_$(tag)"] = tag_shades[i]
    end

    region_colors = Dict(
        "WEST" => RGBA(0.2, 0.2, 1.0, 0.5),
        "EASTASIA" => RGBA(0.2, 1.0, 0.2, 0.5),
        "SUBSAHARA" => RGBA(1.0, 0.5, 0.0, 0.5),
        "MENA" => RGBA(1.0, 0.2, 0.2, 0.5)
    )
    for donor in keys(region_colors)
        for recipient in keys(region_colors)
            color_map["norm_$(donor)_to_$(recipient)"] = region_colors[donor]
        end
    end

    color_map["norm"] = :black
    return color_map
end

# --------------------------
# Plot folder making and parameter and results input
# --------------------------

function make_plot_folder(foldername::String = "")::String
    # Create the path for the folder
    folder_path = joinpath("results_cooperation", foldername)
    
    # Create the folder if it doesn't exist
    isdir(folder_path) || mkdir(folder_path)

    return folder_path
end

#folder_path, snI, snH, snA, AAinfluence ,popsize, execError, assessError, b, mutChance, gossipRounds, strengthOfSelection,parameterVaried, rangeOfValues
function write_parameters(folder_path::String, snH::Vector, snA::Vector, AAinfluence::Float64, normInfluenceFunctionName::String, popsize::Int, 
            execError::Float64, assessError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int, strengthOfSelection::Float64, parameterVaried::String, rangeOfValues::Vector)
    parameters = (
        "Population Size" => popsize,
        "Human Norm" => snH,
        "AA Norm" => snA,
        "AA Influence Function Name" => normInfluenceFunctionName,
        "AA Influence" => AAinfluence,
        "errorExecut" => execError,
        "errorAssess" => assessError,
        "strengthOfSel" => strengthOfSelection,
        "mutationChance" => mutChance,
        "b/c" => b,
        "Gossip Rounds" => gossipRounds,
        "----------" => "",
        "Parameter Varied" => parameterVaried,
        "Range of Values" => rangeOfValues
    )
    write_parameters_txt(joinpath(folder_path, "parameters.txt"),parameters)
end

function write_parameters_txt(path::AbstractString, parameters)
    try
        open(path, "w") do file
            for (param, value) in parameters
                println(file, "$param = $value")
            end
        end
        println("Parameters written to $path\n")
    catch e
        println("Error writing parameters: $e\n")
    end
end

# Abstract method to write any result in the format A -> B in a file
function write_result_txt(path::String, tag::String, result)
    try
        open(path, "a") do file
            println(file, "$tag -> $result")
        end
    catch e
        println("Error writing data: $e\n")
    end
end

# Write all results straight from an array of strategy.get_all_data() to the respective files
# Since this is to be called for each time the results are calculated, it appends to the existing files
function write_all_results(path::String, tag::String, allResults::Vector, save_txt::Bool=false)

    # Create the path for the folder and create the folder if it doesn't exist
    resultsPath = joinpath(path, "Results")
    isdir(resultsPath) || mkdir(resultsPath)

    # Serialize results vector into a folder
    savedVarPath = joinpath(resultsPath, "ResultsBackup")
    isdir(savedVarPath) || mkdir(savedVarPath)
    fileBackup =  open(joinpath(savedVarPath, tag*"_results.jls"), "w")
    serialize(fileBackup, allResults)
    close(fileBackup)

        if (save_txt)
        # Go over each of the vars to make txts with all data
        valuesPath = joinpath(resultsPath, "CoopIndex")
        isdir(valuesPath) || mkdir(valuesPath)
        val = [data[1][1] for data in allResults]
        write_result_txt(joinpath(valuesPath, "CoopIndex_Avg.txt"), tag, val)
        val = [data[1][2] for data in allResults]
        write_result_txt(joinpath(valuesPath, "CoopIndex_All.txt"), tag, val)

        valuesPath = joinpath(resultsPath, "Reputation")
        isdir(valuesPath) || mkdir(valuesPath)
        val = [data[2][1] for data in allResults]
        write_result_txt(joinpath(valuesPath, "Reputation_Avg.txt"), tag, val)
        val = [data[2][2] for data in allResults]
        write_result_txt(joinpath(valuesPath, "Reputation_All.txt"), tag, val)

        valuesPath = joinpath(resultsPath, "StatDist")
        isdir(valuesPath) || mkdir(valuesPath)
        val = [data[3] for data in allResults]
        write_result_txt(joinpath(valuesPath, "StatDist.txt"), tag, val)
        val = [data[4] for data in allResults]
        write_result_txt(joinpath(valuesPath, "State_Avg.txt"), tag, val)

        valuesPath = joinpath(resultsPath, "Agreement")
        isdir(valuesPath) || mkdir(valuesPath)
        val = [data[5][1] for data in allResults]
        write_result_txt(joinpath(valuesPath, "Agreement_Avg.txt"), tag, val)
        val = [data[5][2] for data in allResults]
        write_result_txt(joinpath(valuesPath, "Agreement_All.txt"), tag, val)

        valuesPath = joinpath(resultsPath, "GradientOfSelection")
        isdir(valuesPath) || mkdir(valuesPath)
        val = [data[6] for data in allResults]
        write_result_txt(joinpath(valuesPath, "Gradients.txt"), tag, val)
    end
end

function read_parameters(file_path::AbstractString)
    params = Dict{AbstractString, Any}()

    open(file_path, "r") do file
        for line in eachline(file)
            parts = split(line, "=")
            key = strip(parts[1])
            value = try
                parse(Float64, strip(parts[2]))
            catch
                parse.(Float64, split(strip(parts[2]), ","))
            end
            params[key] = value
        end
    end

    return params
end

# --------------------------
# Result extraction from txt
# --------------------------

# Converts a coopIndex.txt to a vector of vectors with the cooperation index of each
function process_results(filepath::String, population::String)::Vector{Vector{Float64}}
    result_arrays = Vector{Vector{Float32}}()
    open(filepath) do file
        for line in eachline(file)
            if startswith(line, population)
                # Extract the array part from the line
                array_part = split(line, " -> ")[2]
                # Remove "Float32[" and "]" from the array part
                array_string = replace(array_part, r"Float64\[|\]" => "")
                # Convert the comma-separated string into an array of Float64
                array = parse.(Float32, split(array_string, ", "))
                push!(result_arrays, array)
            end
        end
    end
    return result_arrays
end

function deserialize_file(file_path::AbstractString)
    # Open the file for reading
    file = open(file_path, "r")
    
    # Deserialize the content of the file
    data = deserialize(file)
    
    # Close the file
    close(file)
    
    return data
end

# --------------------------
# Processing results
# --------------------------

function get_average_rep_states(results, popsize)
    # Receives the result straight from get_all_data
    # Returns the average human reputation for each of the states given the proportion of each strategy
    # returns [avgRepState1, avgRepState2, ...]

    function get_states_strat(popsize::Int)::Vector{Tuple{Int, Int, Int}}
        return [(nAllC, nAllD, popsize - nAllC - nAllD) for nAllC in 0:popsize for nAllD in 0:(popsize - nAllC) if nAllC + nAllD <= popsize]
    end

    states = get_states_strat(popsize)
    avg_reps = []

    for i in eachindex(states)
        rep_state = results[2][2][i]
        arep = (rep_state[1]*states[i][1] + rep_state[2]*states[i][2] + rep_state[3]*states[i][3]) / popsize
        push!(avg_reps, arep)
    end

    return avg_reps
end

# --------------------------
# Parameter extraction from txt
# --------------------------

# Get parameter value from parameter.txt
function get_parameter_value(filepath::String, parameter::String)
    value = ""
    open(joinpath(filepath, "parameters.txt")) do file
        for line in eachline(file)
            if startswith(line, parameter)
                # Extract the parameter value
                value = split(line, " = ")[2]
                break
            end
        end
    end
    return value
end

# Parse parameter value that is an array of float64
function parse_float64_array(array_string::String)::Vector{Float64}
    # Remove "Float64[" and "]" from the array string
    #array_string = replace(array_string, r"Float64\[|\]" => "")
    # Evaluate the string as Julia code
    array_expr = Meta.parse(array_string)
    # Convert the expression to a tuple of Float64
    array_tuple = eval(array_expr)
    # Convert the tuple to a vector
    array = collect(array_tuple)
    return array
end

# --------------------------
# Plot functions
# --------------------------

function generate_log_spaced_values(start_val::Float64, end_val::Float64, num_samples::Int)::Vector
    if start_val <= 0
        start_val = 1e-10  # Set a small positive value instead of 0
    end
    log_start = log10(start_val)
    log_end = log10(end_val)
    log_spaced_vals = Float32(10) .^ LinRange(log_start, log_end, num_samples)
    return log_spaced_vals
end

# --------------------------
# State and transition matrix functions
# --------------------------

function find_pos_of_state(states::Vector, state::Tuple{Int,Int,Int})::Int
    # Find the position without using the lookup table
    return findfirst(x -> x == state, states)
end

function pos_of_state(table::Dict{Tuple{Int, Int, Int}, Int}, state::Tuple{Int, Int, Int})::Int
    get(table, state, -1)  
end

function create_lookup_table(states::Vector)::Dict{Tuple{Int, Int, Int}, Int}
    lookup_table = Dict{Tuple{Int, Int, Int}, Int}()

    for (index, state) in enumerate(states)
        lookup_table[state] = index
    end

    return lookup_table
end

function get_transition_matrix_statdist(transition_matrix::SparseMatrixCSC)::Vector{Float32}
    transition_matrix_transp = transpose(transition_matrix)

    decomp, _ = partialschur(transition_matrix_transp, nev=1, which=:LR, tol=1e-15);
    stat_dist = vec(real(decomp.Q))
    stat_dist /= sum(stat_dist)

    return stat_dist
end

# --------------------------
# List of all norms
# --------------------------

# Gender-related norms
const norm_gender = ["norm_M_to_M","norm_M_to_F","norm_F_to_M","norm_F_to_F"]

# Region-related norms
const norm_region = ["norm_WEST_to_WEST","norm_WEST_to_EASTASIA","norm_WEST_to_SUBSAHARA","norm_WEST_to_MENA","norm_EASTASIA_to_WEST","norm_EASTASIA_to_EASTASIA","norm_EASTASIA_to_SUBSAHARA","norm_EASTASIA_to_MENA","norm_SUBSAHARA_to_WEST","norm_SUBSAHARA_to_EASTASIA","norm_SUBSAHARA_to_SUBSAHARA","norm_SUBSAHARA_to_MENA","norm_MENA_to_WEST","norm_MENA_to_EASTASIA","norm_MENA_to_SUBSAHARA","norm_MENA_to_MENA"]

# Tag-related norms
const norm_tag = ["norm_tag_no-topic","norm_tag_neutral","norm_tag_non-neutral","norm_tag_explicit-neutral","norm_tag_explicit-non-neutral"]

function complete_family_set(norm_names, families)
    for norm in norm_names
        isInFamily = false
        for family in families 
            if norm in family 
                isInFamily=true 
                break 
            end
        end 
        if !isInFamily
            push!(families,[norm])
        end
    end
    return families
end

# --------------------------
# Gossip value exponential function
# --------------------------

function generate_gossip_range(max_val::Int, num_points::Int)
    return round.(Int, max_val * ((range(0, 1; length=num_points)) .^ 2))
end

# --------------------------
# Multivariate gaussian 
# --------------------------
function compute_norm_deviation_norms(norm_names, norms, all_norms_std_dev, std_devs::Float64=1.0)
    norm_points = Dict()

    for (i, norm) in enumerate(norms)
        if any(isnothing, norm) || !haskey(all_norms_std_dev, norm_names[i])
            continue
        end

        mean_vec = all_norms_std_dev[norm_names[i]]["mean_vector"]
        cov_mat = all_norms_std_dev[norm_names[i]]["cov_matrix"]

        # Extract 4D covariance submatrix
        cov_4d = cov_mat[1:4, 1:4]
        μ = mean_vec[1:4]

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = eigen(cov_4d)

        # Ensure all eigenvalues are non-negative (avoid numerical issues)
        eigvals = max.(eigvals, 0)

        # Compute scaling factor using Chi-squared distribution for 4 degrees of freedom
        confidence_level = cdf(Chisq(4), std_devs^2)
        scaling_factor = sqrt(quantile(Chisq(4), confidence_level))

        # Compute principal axes
        axes_lengths = sqrt.(eigvals) .* scaling_factor
        principal_axes = [axes_lengths[j] * eigvecs[:, j] for j in 1:4]

        # Generate the 9 points: original mean and 8 extreme points
        extreme_points = [μ]  # Start with the center

        for j in 1:4
            pos_point = μ .+ principal_axes[j]
            neg_point = μ .- principal_axes[j]

            # Store without clamping first
            push!(extreme_points, pos_point)
            push!(extreme_points, neg_point)
        end

        # Optional: Apply clamping only when needed
        extreme_points = [clamp.(p, 0.0, 1.0) for p in extreme_points]

        norm_points[norm_names[i]] = extreme_points
    end

    return norm_points
end

function ellipse_extremes(norm_name, all_norms_std_dev, std_devs::Float64=1.0)
    if isnothing(all_norms_std_dev) || !haskey(all_norms_std_dev, norm_name)
        return nothing
    end

    mean_vec = all_norms_std_dev[norm_name]["mean_vector"]
    cov_mat = all_norms_std_dev[norm_name]["cov_matrix"]

    # Extract the relevant 2x2 covariance submatrix
    cov_2d = cov_mat[3:4, 3:4]  # Last two entries of the norm
    μ_x, μ_y = mean_vec[3], mean_vec[4]

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = eigen(cov_2d)

    # Scale the semi-axes
    scaling_factor = sqrt(quantile(Chisq(2), cdf(Chisq(2), std_devs^2)))
    a = sqrt(eigvals[2] * scaling_factor)  # Major axis
    b = sqrt(eigvals[1] * scaling_factor)  # Minor axis

    # Eigenvectors determine directions
    major_axis = a * eigvecs[:,2]
    minor_axis = b * eigvecs[:,1]

    # Compute the four extremities
    top = (μ_x + major_axis[1], μ_y + major_axis[2])
    bottom = (μ_x - major_axis[1], μ_y - major_axis[2])
    right = (μ_x + minor_axis[1], μ_y + minor_axis[2])
    left = (μ_x - minor_axis[1], μ_y - minor_axis[2])

    return (center=(μ_x, μ_y), top=top, bottom=bottom, right=right, left=left)
end


# --------------------------
# Variation calculation 
# -------------------------

# Function to compute KL divergence
function kl_divergence(P, Q)
    P = P .+ eps()  # Avoid log(0) issues
    Q = Q .+ eps()
    return sum(P .* log.(P ./ Q))
end

# Function to compute JSD between two distributions
function jensen_shannon_divergence(P, Q)
    M = 0.5 .* (P .+ Q)
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)
end

# Function to compute JSD for a set of norms
function jsd_for_norms(norms)
    # Convert norms to probability distributions (normalize)
    P = normalize(norms, 1)  # Normalizing over sum to ensure sum(P) = 1
    num_norms = length(P)

    # Compute pairwise JSD for all norm pairs and average
    total_jsd = 0.0
    count = 0

    for i in 1:num_norms
        for j in (i+1):num_norms
            total_jsd += jensen_shannon_divergence(P[i], P[j])
            count += 1
        end
    end

    return count > 0 ? total_jsd / count : 0.0
end

end