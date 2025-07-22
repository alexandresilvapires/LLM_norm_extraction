include("./utils.jl")
include("./reputation.jl")
include("./strategy.jl")
include("./plotter.jl")
include("./norm_influence.jl")

using LaTeXStrings

# Space to run statistics. 
# Define all the parameters below and in the end of the file choose the desired function to run

# Population Settings
const popsize::Int = 100
const b::Float64 = 5.0      # c = 1 always. Some functions override this, such as b_study

const mutChance::Float64 = 1.0 / popsize
const gossipRounds::Int = 10000000000                         # Frequency of gossip based on "A mechanistic model of gossip, reputations, and cooperation". 0 = private reputations, infinite (or 1000000000) = public reputations
const strenghtOfSelection::Float64 = 1.0

# Errors
const execError::Float64 = 0.01
const assessError::Float64 = 0.01

# Norm aggregations for easy indexing and labling -> can be used as references for most experiences
const H_norms = [utils.snIS, utils.snSJ, utils.snSH, utils.snSS]
const H_norm_names = ["Image Score","Stern-Judging","Shunning","Simple Standing"]

# Get AA norms from prompting results
const norm_json_path = "./prompt_dataset/all_llm_norms.json"        # Where the json with the LLM norms is.

# Select which LLMs from norm_json_path to study
const A_norm_names = ["gpt-3.5-turbo","gpt-4o", "qwen2.5-7B-Instruct","qwen2.5-14B-Instruct", "gemma-2-9b-it", "gemma-2-27b-it" ,"gemini-1.5-pro","gemini-2.0-flash","mistral-small", "mistral-large","phi-3.5-mini-instruct","phi-4","llama-2-7b-chat-hf","llama-2-13b-chat-hf", "llama-3.1-8B-Instruct","llama-3.3-70B-Instruct","claude-3-5-haiku","claude-3-7-sonnet", "grok-2", "deepseek-v3", "deepseek-r1"]

# Lists from the same family are given the same colors, and a line connects them in the norm space plot
A_norm_families = [["gpt-3.5-turbo","gpt-4o"], ["qwen2.5-7B-Instruct","qwen2.5-14B-Instruct"],["gemma-2-9b-it", "gemma-2-27b-it"], ["gemini-1.5-pro","gemini-2.0-flash"], ["mistral-small", "mistral-large"], ["phi-3.5-mini-instruct","phi-4"],["llama-2-7b-chat-hf","llama-2-13b-chat-hf","llama-3.1-8B-Instruct","llama-3.3-70B-Instruct"],["claude-3-5-haiku","claude-3-7-sonnet"],["grok-2"],["deepseek-v3", "deepseek-r1"]]   

# For LLM norm maniputation - to keep colors on plot we replace the models we manipulate with gpt4o (or another unmanipulated model)
#const A_norm_names = ["gpt-3.5-turbo","gpt-4o", "gpt-4o","qwen2.5-14B-Instruct", "gemma-2-9b-it", "gemma-2-27b-it", "gpt-4o","gemini-2.0-flash","mistral-small", "mistral-large","phi-3.5-mini-instruct","gpt-4o","llama-2-7b-chat-hf","llama-2-13b-chat-hf", "gpt-4o","llama-3.3-70B-Instruct","claude-3-5-haiku","claude-3-7-sonnet", "grok-2", "deepseek-v3", "deepseek-r1"]
#append!(A_norm_names,["qwen2.5-7B-Instruct","qwen2.5-7B-Instruct-universalization","qwen2.5-7B-Instruct-empathizing","qwen2.5-7B-Instruct-signaling","qwen2.5-7B-Instruct-motivation","gemini-1.5-pro","gemini-1.5-pro-universalization","gemini-1.5-pro-empathizing","gemini-1.5-pro-signaling","gemini-1.5-pro-motivation","phi-4","phi-4-universalization","phi-4-empathizing","phi-4-signaling","phi-4-motivation", "llama-3.1-8B-Instruct", "llama-3.1-8B-Instruct-universalization", "llama-3.1-8B-Instruct-empathizing", "llama-3.1-8B-Instruct-signaling", "llama-3.1-8B-Instruct-motivation"])
#A_norm_families = [["gpt-3.5-turbo","gpt-4o"], ["qwen2.5-7B-Instruct","qwen2.5-7B-Instruct-universalization","qwen2.5-7B-Instruct-empathizing","qwen2.5-7B-Instruct-signaling","qwen2.5-7B-Instruct-motivation","qwen2.5-14B-Instruct"],["gemma-2-9b-it", "gemma-2-27b-it"], ["gemini-1.5-pro","gemini-1.5-pro-universalization","gemini-1.5-pro-empathizing","gemini-1.5-pro-signaling","gemini-1.5-pro-motivation","gemini-2.0-flash"], ["mistral-small", "mistral-large"], ["phi-3.5-mini-instruct","phi-4","phi-4-universalization","phi-4-empathizing","phi-4-signaling","phi-4-motivation"],["llama-2-7b-chat-hf","llama-2-13b-chat-hf","llama-3.1-8B-Instruct", "llama-3.1-8B-Instruct-universalization", "llama-3.1-8B-Instruct-empathizing", "llama-3.1-8B-Instruct-signaling", "llama-3.1-8B-Instruct-motivation","llama-3.3-70B-Instruct"],["claude-3-5-haiku","claude-3-7-sonnet"],["grok-2"],["deepseek-v3", "deepseek-r1"]]

# For cooperation study plot
#const A_norm_names = ["gpt-4o","llama-3.3-70B-Instruct","claude-3-5-haiku","grok-2"]
#const A_norm_names = ["qwen2.5-14B-Instruct","gemma-2-27b-it","llama-3.1-8B-Instruct","deepseek-v3"]
#const A_norm_names = ["gemini-2.0-flash","mistral-small","llama-2-7b-chat-hf","phi-4"]
#const A_norm_names = ["qwen2.5-7B-Instruct","gemma-2-9b-it","gemini-1.5-pro","claude-3-7-sonnet"]
#const A_norm_names = ["gpt-3.5-turbo","mistral-large","phi-3.5-mini-instruct","llama-2-13b-chat-hf","deepseek-r1"]
#A_norm_families = []

const normInfluenceFunction::Function = linear
const normInfluenceFunctionName::String = "Linear" 

# Study for influence of AAs
const AAinfluence::Float64 = 0.0
const valsAAinfluence::Vector{Float64} = collect(0.0:0.02:1.0)

# Study of gossip values -> assumes influence = 1
const valsGossip::Vector{Int} = utils.generate_gossip_range(popsize*10, 101)

# Study of bc values -> assumes influence = 1 and uses gossip = gossipRounds
const valsbc::Vector{Float64} = collect(1.0:0.1:8.0)

# Additional settings
const justGenPlots::Bool = false         # if true, instead of generating data and plotting it, it just accesses the folder to generate the plots

foldername::String = "norm_analysis"     # change folder name for each experiment

# Premade functions for studies below

# Study behaviour with varying levels of influence and a fixed b/c
function influence_study(norm_names_H::Vector{String}, norm_names_A::Vector{String})
    filenames::Vector{String} = []

    # We do a file for each pair aa_norm/human_norm
    for norm_a in eachindex(norm_names_A)
        for norm_h in eachindex(norm_names_H)
            filename = norm_names_A[norm_a]*norm_names_H[norm_h]
            !justGenPlots && vary_variable([H_norms[norm_h], A_norms[norm_a], 0.0, normInfluenceFunction, normInfluenceFunctionName, popsize, 
                execError, assessError, b, mutChance, gossipRounds, strenghtOfSelection],"AAinfluence", 3, valsAAinfluence, filename, foldername)

            push!(filenames, filename)
        end
    end

    #plot_grad(stratAA, norms, [0,0.05,0.1,0.15,0.2], popsize, execError, b, gossipRounds, mutChance, imitAAs, plotpath, filenames)

    run_all_plots(plotpath, filenames, norm_names_A, norm_names_H)
end

# Study cooperation and disagreement when interpolating between social norms social norms
function intermediate_norms_study(simplify_plot::Bool=false)
    # If simplify_plot is true, simplifies the plot to remove numbers and expand norm names
    
    # make a matrix where we interpolate between (1, 0, a, b), a,b in [0, 1]
    bcs_to_test = [b]

    # Run once for all-equal, then run 4 times varying only H-A-H
    for bc in bcs_to_test
        println("----running intermediate plot with bc="*string(bc))
        
        # Run when all norms are the same
        all_sets_of_norms = vec([[1, 0, a, b] for (a, b) in Iterators.product(0:0.1:1.0, 0:0.1:1.0)])

        # TODO: currently not considering the norm of the AA
        fname = "intermediatenorms_bc="*string(bc)
        !justGenPlots && vary_variable([[1.0,1.0,1.0,1.0], [1.0,1.0,1.0,1.0], 0.0, normInfluenceFunction, normInfluenceFunctionName, popsize, 
        execError, assessError, bc, mutChance, gossipRounds, strenghtOfSelection],"H_sn", 1, all_sets_of_norms, fname, foldername)

    
        intermediate_norms_plot(plotpath, fname, L"\text{Cooperation under Social Norm = }(1, 0, d_{BC}, d_{BD}) \text{ in public and private reputations}", fname,all_sets_of_norms, simplify_plot)
        println("intermediate norm done")
    end
end

# Analyses the norms of each LLM for biases (each scenario) and analyses the same scenario in multiple LLMs 
function norm_analysis(norm_names::Vector, all_norms, main_norms::Vector, all_norms_std_dev; plot_ellipse::Bool=false, plot_detailed_norms::Bool=true)
    # First plots: All main norms, for bad part and good part
    println("Plotting average norms of all models")
    norm_analysis_plot(plotpath,norm_names,main_norms,L"\text{LLM Social Norms = }(-,-, d_{BC}, d_{BD})","llm_norm_plot_bad", "Models", special_labels = false,color_map=Dict{String, Any}(), plot_ellipse=plot_ellipse, all_norms_std_dev=all_norms_std_dev, std_devs=1.0, families=A_norm_families, clarify_left_edge=true)
    norm_analysis_plot(plotpath,norm_names,main_norms,L"\text{LLM Social Norms = }(d_{GC}, d_{GD},-,-)","llm_norm_plot_good", "Models", special_labels = false,color_map=Dict{String, Any}(), for_good_norm=true, plot_ellipse=plot_ellipse, all_norms_std_dev=all_norms_std_dev, std_devs=1.0, families=A_norm_families)

    # Second plots: Per LLM detailed norms, separated by part of focus
    if plot_detailed_norms
        for name in norm_names
            if haskey(all_norms, name)
                println("Plotting detailed norms for model: $name...")
                for (parts, part_name) in zip((utils.norm_region, utils.norm_gender, utils.norm_tag), ("region", "gender", "tag"))
                    norms_dict = all_norms[name]
                    filtered_norms = utils.filter_norms(norms_dict, parts)
                    norms_list = [v for v in values(filtered_norms)]
                    norm_labels = [k for k in keys(filtered_norms)]

                    # Custom color mapping
                    color_map = utils.make_color_map_norms()

                    # Define order of points
                    ordered_labels = parts

                    norm_analysis_plot(plotpath,norm_labels,norms_list,"Social Norm = (_, _, α¹, α²) for $name","norm_analysis_bad_$(part_name)_$name", "Norms",special_labels = true,color_map = color_map, order=ordered_labels)
                    norm_analysis_plot(plotpath,norm_labels,norms_list,"Social Norm = (β¹, β², _, _) for $name","norm_analysis_good_$(part_name)_$name", "Norms",special_labels = true,color_map = color_map, for_good_norm=true, order=ordered_labels)
                end
            end
        end
    end
end

# Study behaviour with varying levels of gossip and a fixed b/c, assuming norms are fixed and not influenced (either just LLM norms or just human norms)
function gossip_study(use_human_norms::Bool=false)
    # If use_human_norms is true, instead of using LLM norms, it uses human norms
    filenames::Vector{String} = []

    norm_names = A_norm_names
    norms = A_norms

    if use_human_norms
        norm_names = H_norm_names
        norms = H_norms
    end

    # We do a file for each pair aa_norm/human_norm
    for norm_a in eachindex(norm_names)
        filename = norm_names[norm_a]
        !justGenPlots && vary_variable([norms[norm_a], norms[norm_a], 1.0, normInfluenceFunction, normInfluenceFunctionName, popsize, 
            execError, assessError, b, mutChance, gossipRounds, strenghtOfSelection],"gossipRounds", 11, valsGossip, filename, foldername)

        push!(filenames, filename)
    end

    plot_coop_indexes_varying_gossip(valsGossip, plotpath, norm_names, norm_names, "Social Norms") 
end

function gossip_study_with_uncertainty()
    filenames::Vector{String} = []
    norm_names = A_norm_names
    norms = A_norms
    # Get extremity points for each norm
    norm_extreme_points = utils.compute_norm_deviation_norms(norm_names, norms, all_A_norms_std_dev)

    all_results = Dict()
    
    for norm_a in eachindex(norm_names)
        extreme_norms = norm_extreme_points[norm_names[norm_a]]
        
        coop_results = []
        for norm_i in eachindex(extreme_norms)  # Iterate through center + extremities
            norm = extreme_norms[norm_i]
            temp_filename = norm_names[norm_a] * "_extreme_"*string(norm_i)
            !justGenPlots && vary_variable([norm, norm, b, normInfluenceFunction, normInfluenceFunctionName, popsize, 
                execError, assessError, b, mutChance, gossipRounds, strenghtOfSelection], "gossipRounds", 11, valsGossip, temp_filename, foldername)
            
            coop_data = utils.deserialize_file(joinpath("./results_cooperation/" * foldername * "/Results/ResultsBackup", temp_filename * "_results.jls"))
            push!(coop_results, [coop_data[k][1][1][1] for k in eachindex(valsGossip)])
        end
        
        all_results[norm_names[norm_a]] = coop_results
        push!(filenames, norm_names[norm_a])
    end
    
    # Plot with shaded regions
    plot_coop_indexes_varying_gossip_with_uncertainty(valsGossip, plotpath, norm_names, all_results, "Social Norms")
end

# Study behaviour with varying levels of b/c and a fixed gossip, assuming norms are fixed and not influenced (either just LLM norms or just human norms)
function b_study(use_human_norms::Bool=false)
    # If use_human_norms is true, instead of using LLM norms, it uses human norms
    filenames::Vector{String} = []

    norm_names = A_norm_names
    norms = A_norms

    if use_human_norms
        norm_names = H_norm_names
        norms = H_norms
    end

    # We do a file for each pair aa_norm/human_norm
    for norm_a in eachindex(norm_names)
        filename = norm_names[norm_a]
        !justGenPlots && vary_variable([norms[norm_a], norms[norm_a], 1.0, normInfluenceFunction, normInfluenceFunctionName, popsize, 
            execError, assessError, b, mutChance, gossipRounds, strenghtOfSelection],"bc", 9, valsbc, filename, foldername)

        push!(filenames, filename)
    end

    plot_coop_indexes_varying_bc(valsbc, plotpath, norm_names, norm_names, "Social Norms") 
end

function b_study_with_uncertainty()
    filenames::Vector{String} = []
    norm_names = A_norm_names
    norms = A_norms
    # Get extremity points for each norm
    norm_extreme_points = utils.compute_norm_deviation_norms(norm_names, norms, all_A_norms_std_dev)

    all_results = Dict()
    
    for norm_a in eachindex(norm_names)
        extreme_norms = norm_extreme_points[norm_names[norm_a]]
        
        coop_results = []
        for norm_i in eachindex(extreme_norms)  # Iterate through center + extremities
            norm = extreme_norms[norm_i]
            temp_filename = norm_names[norm_a] * "_extreme_"*string(norm_i)
            !justGenPlots && vary_variable([norm, norm, 1.0, normInfluenceFunction, normInfluenceFunctionName, popsize, 
                execError, assessError, b, mutChance, gossipRounds, strenghtOfSelection], "bc", 9, valsbc, temp_filename, foldername)
            
            coop_data = utils.deserialize_file(joinpath("./results_cooperation/" * foldername * "/Results/ResultsBackup", temp_filename * "_results.jls"))
            push!(coop_results, [coop_data[k][1][1][1] for k in eachindex(valsbc)])
        end
        
        all_results[norm_names[norm_a]] = coop_results
        push!(filenames, norm_names[norm_a])
    end
    
    # Plot with shaded regions
    plot_coop_indexes_varying_bc_with_uncertainty(valsbc, plotpath, norm_names, all_results, "Social Norms")
end

# Study behaviour with varying levels of b/c and both public and private reputations, assuming norms are fixed and not influenced (just LLM norms)
function b_study_with_uncertainty_public_and_private(foldername_public, foldername_private)
    filenames::Vector{String} = []
    norm_names = A_norm_names
    norms = A_norms
    # Get extremity points for each norm
    norm_extreme_points = utils.compute_norm_deviation_norms(norm_names, norms, all_A_norms_std_dev)

    all_results_public, all_results_private = Dict(), Dict()
    
    for (all_results, folder_name, gossip) in zip([all_results_public, all_results_private], [foldername_public, foldername_private], [1000000000, 0])
        for norm_a in eachindex(norm_names)
            extreme_norms = norm_extreme_points[norm_names[norm_a]]
            coop_results = []
            for norm_i in eachindex(extreme_norms)  # Iterate through center + extremities
                norm = extreme_norms[norm_i]
                temp_filename = norm_names[norm_a] * "_extreme_"*string(norm_i)
                !justGenPlots && vary_variable([norm, norm, 1.0, normInfluenceFunction, normInfluenceFunctionName, popsize, 
                    execError, assessError, b, mutChance, gossip, strenghtOfSelection], "bc", 9, valsbc, temp_filename, folder_name,true, true)
                
                coop_data = utils.deserialize_file(joinpath("./results_cooperation/" * folder_name * "/Results/ResultsBackup", temp_filename * "_results.jls"))
                push!(coop_results, [coop_data[k][1][1][1] for k in eachindex(valsbc)])
            end
            
            all_results[norm_names[norm_a]] = coop_results
            push!(filenames, norm_names[norm_a])
        end
    end
        
    # Plot with shaded regions
    plot_public_private_coop_indexes_varying_bc_with_uncertainty(valsbc, plotpath, A_norm_names, all_results_public, all_results_private, "")
end

# Parameters that are automatically set following the parameters above 
const all_A_norms = utils.extract_norms(norm_json_path, A_norm_names)
const A_norms = [all_A_norms[name]["norm"] for name in A_norm_names if haskey(all_A_norms, name)] #extract the main norm from every llm
const all_A_norms_std_dev = utils.extract_norm_standard_dev(norm_json_path, A_norm_names)
A_norm_families = utils.complete_family_set(A_norm_names, A_norm_families)

plotpath::String = "./results_cooperation/"*foldername

# Leave the desired function uncommented.

# The figures from the main paper are done via the following functions:
# Fig 2: norm_analysis, then merged via image editing software
# Fig 3: llm_intermediate_norms_plot, which requires running intermediate_norms_study to run and store results in seperate folders for public and private reputations (gossip >>>> popsize and gossip = 0, respectively)
# Fig 4: b_study_with_uncertainty_public_and_private, which first requires b_study_with_uncertainty to run and store results in seperate folders for public and private reputations (gossip >>>> popsize and gossip = 0, respectively)
# Fig 5: norm_analysis_intervention_plot

# The figures for the supplementary material are done through the following functions:
# Fig S1: plot_error_heatmap
# Fig S2: llm_intermediate_norms_plot
# Fig S3-S6: b_study_with_uncertainty_public_and_private
# Fig S7-S9: plot_bias_analysis
# Fig S10: norm_analysis_intervention_plot



norm_analysis(A_norm_names, all_A_norms, A_norms, all_A_norms_std_dev; plot_ellipse=true, plot_detailed_norms=false)
#plot_bias_analysis(plotpath, A_norm_names, all_A_norms)
#variance_plotter(all_A_norms, A_norm_names, plotpath)
#b_study_with_uncertainty()
#b_study(true)
#gossip_study_with_uncertainty()
#gossip_study(false)
#influence_study(H_norm_names, A_norm_names)
#intermediate_norms_study(false)

# Compound plots from article

#b_study_with_uncertainty_public_and_private("bc_study_public_uncertainty", "bc_study_private_uncertainty") # must first do results for both public and private individually

#public_private_intermediate_norms_plot(plotpath, "./results_cooperation/intermediate_norms_public/", "./results_cooperation/intermediate_norms_private/", "intermediatenorms_bc="*string(b), "intermediatenorms_bc="*string(b), "intermediatenorms_bc="*string(4),vec([[1, 0, a, b] for (a, b) in Iterators.product(0:0.1:1.0, 0:0.1:1.0)]), true, 
#    models_to_show=["gpt-3.5-turbo","gpt-4o","llama-2-13b-chat-hf", "llama-3.1-8B-Instruct","llama-3.3-70B-Instruct","claude-3-5-haiku","claude-3-7-sonnet","grok-2"], norm_names_LLMs=A_norm_names, norms_LLMs=A_norms, color_map_LLMs=Dict{String, Any}(), plot_ellipse_LLMs=false, all_norms_std_dev=all_A_norms_std_dev, families_LLMs=A_norm_families)

#norm_analysis_intervention_plot(plotpath,A_norm_names,A_norms,L"\text{LLM Social Norms = }(-,-, d_{BC}, d_{BD})","llm_norm_plot_bad-intervention", "Models", special_labels = false, plot_ellipse=true, all_norms_std_dev=all_A_norms_std_dev, std_devs=1.0, families=A_norm_families, for_good_norm=false)
#norm_analysis_intervention_plot(plotpath,A_norm_names,A_norms,L"\text{LLM Social Norms = }(-,-, d_{BC}, d_{BD})","llm_norm_plot_good-intervention", "Models", special_labels = false, plot_ellipse=true, all_norms_std_dev=all_A_norms_std_dev, std_devs=1.0, families=A_norm_families, for_good_norm=true)
#llm_intermediate_norms_plot(plotpath, "./results_cooperation/intermediate_norms_public/", "intermediatenorms_bc="*string(b), "intermediatenorms_bc="*string(5),vec([[1, 0, a, b] for (a, b) in Iterators.product(0:0.1:1.0, 0:0.1:1.0)]), true, 
#    models_to_show=["gpt-3.5-turbo","gpt-4o","llama-2-13b-chat-hf", "llama-3.1-8B-Instruct","llama-3.3-70B-Instruct","claude-3-5-haiku","claude-3-7-sonnet","grok-2"], norm_names_LLMs=A_norm_names, norms_LLMs=A_norms, color_map_LLMs=Dict{String, Any}(), plot_ellipse_LLMs=false, all_norms_std_dev=all_A_norms_std_dev, families_LLMs=A_norm_families)

#plot_error_heatmap("./prompt_dataset/all_llm_errors.json", ["gpt-3.5-turbo","gpt-4o", "qwen2.5-7B-Instruct","qwen2.5-14B-Instruct", "gemma-2-9b-it", "gemma-2-27b-it" ,"gemini-1.5-pro","gemini-2.0-flash","mistral-small", "mistral-large","phi-3.5-mini-instruct","phi-4","llama-2-7b-chat-hf","llama-2-13b-chat-hf", "llama-3.1-8B-Instruct","llama-3.3-70B-Instruct","claude-3-5-haiku","claude-3-7-sonnet", "grok-2", "deepseek-v3", "deepseek-r1"], "./results_cooperation/parse_error_analysis")