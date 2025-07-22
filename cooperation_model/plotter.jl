include("./utils.jl")
include("./strategy.jl")
using LaTeXStrings
using LinearAlgebra
using Statistics
using CairoMakie
using Colors
using Distributions
using JSON

const line_colors = (RGBA(0.1176, 0.5647, 1.0), RGBA(1.0, 0.4980, 0.0549), RGBA(0.1961, 0.8039, 0.1961), RGBA(1.0, 0.8431, 0.0), RGBA(0.8, 0.1784, 0.5765), RGBA(0.0, 0.8078, 0.8196), RGBA(1.0, 0.0, 0.0), RGBA(0.5804, 0.0, 0.8275), RGBA(0.5020, 0.7020, 0.0), RGBA(0.4118, 0.4118, 0.4118), RGBA(1.0, 0.4980, 0.0549))
const markerTypes = (:circle, :rect, :diamond, :cross, :utriangle, :xcross, :star4, :dtriangle, :star8, :rtriangle, :ltriangle)
const lineTypes = (:solid, :dash, :dashdot, :dot)
const letter_labels = ["a)", "b)", "c)", "d)", "e)", "f)","g)","h)", "i)","j)","k)","l)"]

# Utility functions

function get_index_k(lis, k) return [res[k] for res in lis] end # aux function to do cross dimentional array indexing

function transform_points(points)
    # transforms points to the new triangle coordinates
    d = π / 3
    tmatrix = [cos(-2d) sin(-2d); cos(-d) sin(-d)]
    tpoints = points * tmatrix
    return tpoints
end
function closest_index(sample_states::Matrix{Float64}, point::Vector{Int64})
    distances = [sqrt((point[1] - state[1])^2 + (point[2] - state[2])^2) for state in eachrow(sample_states)]
    return argmin(distances)
end
function extract_data_vector_sample(all_stts::Matrix, data::Vector, sample_stts::Matrix, sample_size::Int)
    # Initialize dataSample matrix
    dataSample = zeros(length(get_states_strat(sample_size)))
    # Iterate over each point in sample_stts and find its corresponding data value
    for (i, point) in enumerate(eachrow(sample_stts))
        # Find the index of the point in all_stts
        index_in_all_stts = findfirst(all_stt -> all_stt == point, eachrow(all_stts))
        # Copy the data value from data to dataSample
        dataSample[i] = data[index_in_all_stts]
    end
    return dataSample
end
function extract_data_sample(all_stts::Matrix, data::Matrix, sample_stts::Matrix)
    # Initialize dataSample matrix
    dataSample = zeros(size(sample_stts, 1), size(data, 2))
    # Iterate over each point in sample_stts and find its corresponding data value
    for (i, point) in enumerate(eachrow(sample_stts))
        # Find the index of the point in all_stts
        index_in_all_stts = findfirst(all_stt -> all_stt == point, eachrow(all_stts))
        # Copy the data value from data to dataSample
        dataSample[i, :] = data[index_in_all_stts, :]
    end
    return dataSample
end
function rescale_norm(x::Vector{<:Real}, y::Vector{<:Real}, minNorm::Real, maxNorm::Real)
    # Combine x and y into vectors of 2D points
    points = [(x[i], y[i]) for i in 1:length(x)]
    
    # Calculate norms
    norms = [norm(point) for point in points]
    
    # Find min and max norms
    min_norm = minimum(norms)
    max_norm = maximum(norms)
    
    # Rescale norms to range [minNorm, maxNorm]
    scaled_norms = [(norm - min_norm) / (max_norm - min_norm) * (maxNorm - minNorm) + minNorm for norm in norms]
    
    # Rescale vectors
    scaled_points = [(point[1] * (scaled_norms[i] / norms[i]), point[2] * (scaled_norms[i] / norms[i])) for (i, point) in enumerate(points)]
    
    # Separate x and y back
    scaled_x = [point[1] for point in scaled_points]
    scaled_y = [point[2] for point in scaled_points]

    return scaled_x, scaled_y
end

function compute_scaling(k_sigma)
    confidence_level = cdf(Chisq(2), k_sigma^2)  # Convert k-sigma to confidence level
    scaling_factor = sqrt(quantile(Chisq(2), confidence_level))  # Convert confidence to scaling
    return scaling_factor
end

# Plotting functions

function plot_coop_indexes(results,x_axis_scale, folder_path, sn_labels_a::Vector{String}, sn_labels_h::Vector{String}) 
    
    # Make plot folder
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)
    
    for i in range(1,2) # First make plots fixating AA norm and varying H (so H norms are lines, A norms are figures), then making other way

        line_labels, figure_labels = (i == 1 ? (sn_labels_h, sn_labels_a) : (sn_labels_a, sn_labels_h)) 

        f = Figure(backgroundcolor = :white, size = (1700*length(figure_labels)/4, 350))

        for subfig in eachindex(figure_labels) 
            # Define when plot parts appear
            XAxisTitle = (subfig == 1 ? "AA Influence, α" : "")
            YAxisTitle = (subfig == 1 ? "Cooperation Index, I" : "")

            subplot_title = figure_labels[subfig]

            ax = Axis(f[1, subfig], titlesize=24, xlabelsize=20, ylabelsize=20, 
                xticklabelsize=16, yticklabelsize=16,
                title=subplot_title, xlabel=XAxisTitle, ylabel=YAxisTitle,
                xticks=0:0.1:1.01, yticks=(0:0.2:1.0),xautolimitmargin=(0.0,0.01),
                yautolimitmargin=(0.06,0.06), yminorticksvisible=true)
            ylims!(ax, (-0.05,1.051))

            res = []
            if i == 1 # Iterate by col
                startindex = (subfig-1)*length(sn_labels_h) + 1

                for indexRes in startindex:startindex+(length(sn_labels_h)-1)
                    push!(res,results[indexRes])
                end
            else # Iterate by row
                for indexRes in eachindex(sn_labels_a)
                    push!(res,results[subfig+(indexRes-1)*length(sn_labels_h)])
                end
            end

            for v in eachindex(res)
                # Add lines incrementally for each norm
                if i == 1   # Norm lines maintain colors of each norm
                    l = scatterlines!(ax, x_axis_scale, res[v], label=line_labels[v], color=line_colors[v], linestyle=lineTypes[subfig], marker=markerTypes[v])
                else        # Norm lines of each LLM maintain the linestyle of each norm
                    l = lines!(ax, x_axis_scale, res[v], label=line_labels[v], color=line_colors[subfig], linestyle=lineTypes[v], linewidth=3.5)
                end
            end

            #if (keepLegend) axislegend(plotLinesTitle,orientation = :horizontal, nbanks=2, position = :lt) end

            text!(ax, 1, 1, text = letter_labels[i], font = :bold, align = (:left, :top), offset = (-430, -2),
                space = :relative, fontsize = 24
            )

            plotLinesTitle = (i == 1 ? "Human Norm" : "LLM Norm")
            Legend(f[1, length(figure_labels)+1], ax, plotLinesTitle)

        end

        # Save the plot inside the folder
        final_path = (i == 1 ? joinpath(plot_path, "coop_index_per_llm.pdf") : joinpath(plot_path, "coop_index_per_h_norm.pdf")) 
        save(final_path, f)
        display(f)
    end
end

function plot_coop_indexes_varying_gossip(gossip_vals, folder_path, filenames, sn_labels_a::Vector{String}, legend_label::String) 

    res_path::String = joinpath(folder_path, "Results/ResultsBackup")

    results = []
    for name in filenames
        push!(results, utils.deserialize_file(joinpath(res_path,name*"_results.jls")))
    end

    results_coop = []
    for i in eachindex(filenames)
        push!(results_coop, [results[i][k][1][1][1] for k in eachindex(gossip_vals)])
    end
    
    # Make plot folder
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)
    
    f = Figure(backgroundcolor = :white, size = (800, 550))

    # Define when plot parts appear
    XAxisTitle = "Gossip, τ"
    YAxisTitle = "Cooperation Index, I"

    subplot_title = "Cooperation Index per LLM"

    ax = Axis(f[1, 1], titlesize=24, xlabelsize=20, ylabelsize=20, 
        xticklabelsize=16, yticklabelsize=16, xlabel=XAxisTitle, ylabel=YAxisTitle,
        xticks=0:0.1:1.00, yticks=(0:0.2:1.0),xautolimitmargin=(0.0,0.01),
        yautolimitmargin=(0.06,0.06), yminorticksvisible=true)
    ylims!(ax, (-0.05,1.051))

    for v in eachindex(results)
        # Add lines incrementally for each norm
        scatterlines!(ax, 0:0.01:1.0, results_coop[v], label=sn_labels_a[v], color=line_colors[v], marker=markerTypes[v])
    end

    #if (keepLegend) axislegend(plotLinesTitle,orientation = :horizontal, nbanks=2, position = :lt) end
    Legend(f[1, 2], ax, legend_label)

    # Save the plot inside the folder
    final_path = joinpath(plot_path, "coop_index_per_a_norm_gossip.pdf")
    save(final_path, f)
    display(f)
end

function plot_coop_indexes_varying_gossip_with_uncertainty(gossip_vals, folder_path, norm_names, all_results, legend_label)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)
    
    f = Figure(backgroundcolor=:white, size=(800, 550))
    ax = Axis(f[1,1], titlesize=24, xlabelsize=20, ylabelsize=20, 
            xticklabelsize=16, yticklabelsize=16, xlabel="Gossip, τ", ylabel="Cooperation Index, I",
            xticks=gossip_vals, yticks=(0:0.2:1.0), xautolimitmargin=(0.0,0.01),
            yautolimitmargin=(0.06,0.06), yminorticksvisible=true)
    ylims!(ax, (-0.05,1.051))

    for v in eachindex(norm_names)
        norm_name = norm_names[v]
        coop_vals = all_results[norm_name]  # This should be a list of vectors
    
        coop_matrix = hcat(coop_vals...)  # Convert list of vectors into a matrix where each column is a different norm
    
        min_vals = minimum(coop_matrix, dims=2)[:]
        max_vals = maximum(coop_matrix, dims=2)[:]
        mean_vals = coop_matrix[:,1]  # Use the center norm (first column)
    
        band!(ax, gossip_vals, min_vals, max_vals, color=(line_colors[v], 0.3))  # Shaded region
        scatterlines!(ax, gossip_vals, mean_vals, label=norm_name, color=line_colors[v], marker=markerTypes[v])
    end
    
    Legend(f[1,2], ax, legend_label)
    save(joinpath(plot_path, "coop_index_per_norm_gossip_uncertainty.pdf"), f)
    display(f)
end

function plot_coop_indexes_varying_bc(bc_vals, folder_path, filenames, sn_labels_a::Vector{String}, legend_label::String) 

    res_path::String = joinpath(folder_path, "Results/ResultsBackup")

    results = []
    for name in filenames
        push!(results, utils.deserialize_file(joinpath(res_path,name*"_results.jls")))
    end

    results_coop = []
    for i in eachindex(filenames)
        push!(results_coop, [results[i][k][1][1][1] for k in eachindex(bc_vals)])
    end
    
    # Make plot folder
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)
    
    f = Figure(backgroundcolor = :white, size = (800, 550))

    # Define when plot parts appear
    XAxisTitle = "Benefit-to-cost ratio, b/c"
    YAxisTitle = "Cooperation Index, I"

    subplot_title = "Cooperation Index per LLM"

    ax = Axis(f[1, 1], titlesize=24, xlabelsize=20, ylabelsize=20, 
        xticklabelsize=16, yticklabelsize=16, xlabel=XAxisTitle, ylabel=YAxisTitle,
        xticks=(1.0:1.0:8.0), yticks=(0:0.2:1.0),xautolimitmargin=(0.0,0.01),
        yautolimitmargin=(0.06,0.06), yminorticksvisible=true)
    ylims!(ax, (-0.05,1.051))

    for v in eachindex(results)
        # Add lines incrementally for each norm
        scatterlines!(ax, bc_vals, results_coop[v], label=sn_labels_a[v], color=line_colors[v], marker=markerTypes[v])
    end

    #if (keepLegend) axislegend(plotLinesTitle,orientation = :horizontal, nbanks=2, position = :lt) end
    Legend(f[1, 2], ax, legend_label)

    # Save the plot inside the folder
    final_path = joinpath(plot_path, "coop_index_per_a_norm_bc.pdf")
    save(final_path, f)
    display(f)
end

function plot_coop_indexes_varying_bc_with_uncertainty(bc_vals, folder_path, norm_names, all_results, legend_label)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)
    
    f = Figure(backgroundcolor=:white, size=(800, 550))
    ax = Axis(f[1,1], titlesize=24, xlabelsize=20, ylabelsize=20, 
            xticklabelsize=16, yticklabelsize=16, xlabel="Benefit-to-cost ratio, b/c", ylabel="Cooperation Index, I",
            xticks=(1.0:1.0:8.0), yticks=(0:0.2:1.0), xautolimitmargin=(0.0,0.01),
            yautolimitmargin=(0.06,0.06), yminorticksvisible=true)
    ylims!(ax, (-0.05,1.051))

    line_colors_bc =  (RGBA(0.3686, 0.5059, 0.7098),RGBA(0.9216, 0.3843, 0.2078), RGBA(0.635, 0.475, 0.725), RGBA(0.5608,0.6902,0.1961), RGBA(0.5, 0.5, 0.5), RGBA(0.2333, 0.2333, 0.2333),RGBA(0.5569, 0.1765, 0.6745),  RGBA(0.2039, 0.7059, 0.6824),RGBA(0.9765, 0.4510, 0.8824),RGBA(0.9373, 0.8118, 0.2353),  RGBA(0.1294, 0.1294, 0.7882))

    for v in eachindex(norm_names)
        norm_name = norm_names[v]
        coop_vals = all_results[norm_name]  # This should be a list of vectors
    
        coop_matrix = hcat(coop_vals...)  # Convert list of vectors into a matrix where each column is a different norm
    
        min_vals = minimum(coop_matrix, dims=2)[:]
        max_vals = maximum(coop_matrix, dims=2)[:]
        mean_vals = coop_matrix[:,1]  # Use the center norm (first column)
    
        band!(ax, bc_vals, min_vals, max_vals, color=(line_colors_bc[v], 0.3))  # Shaded region
        scatterlines!(ax, bc_vals, mean_vals, label=norm_name, color=line_colors_bc[v], marker=markerTypes[v])
    end
    
    Legend(f[1,2], ax, legend_label)
    save(joinpath(plot_path, "coop_index_per_norm_bc_uncertainty.pdf"), f)
    display(f)
end

function plot_reputations(results, x_axis_scale, line_titles, folder_path, sn_labels_a::Vector{String}, sn_labels_h::Vector{String}) 
    # Make plot folder
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    rep_line_colors = (RGB(9/255, 150/255, 11/255), RGB(180/255, 50/255, 25/255), RGB(13/255, 140/255, 227/255))

    f = Figure(backgroundcolor = :white, size = (1700*length(sn_labels_h)/4, 1070*length(length(sn_labels_a))))

    for llm in eachindex(sn_labels_a)
        for human_norm in eachindex(sn_labels_h) # for each SN we make a norm
            currentx = human_norm
            currenty = llm

            # Define when plot parts appear.
            keepLegend = (currentx == length(sn_labels_h) ? true : false) # Legend = name of each line
            XAxisTitle = (currentx == 1 ? "AA Influence, α" : "")
            YAxisTitle = (currentx == 1 ? "Reputation, rᴴ" : "")

            titleaxis = (currenty == 1 ? sn_labels_h[human_norm] : "")

            ax = Axis(f[currenty*2, currentx], titlesize=24, xlabelsize=20, ylabelsize=20, 
                xticklabelsize=16, yticklabelsize=16,
                title=titleaxis, xlabel=XAxisTitle, ylabel=YAxisTitle,
                xticks=0:0.1:1.01, yticks=(0:0.2:1.0),xautolimitmargin=(0.0,0.01),
                yautolimitmargin=(0.06,0.06), yminorticksvisible=true)

                ylims!(ax, (-0.05,1.051))
                xlims!(ax, (-0.05,1.051))
            
            res = collect(eachrow(reduce(hcat, results[(llm-1)*length(sn_labels_h) + human_norm])))

            for v in eachindex(res)
                scatterlines!(ax, x_axis_scale, res[v], label=line_titles[v], color=rep_line_colors[v],marker=markerTypes[v])
            end

            if (keepLegend) axislegend("Strategy",orientation = :vertical, nbanks=1, position = (1, 0.5)) end

            #text!(ax, 1, 1, text = letter_labels[k], font = :bold, align = (:left, :top), offset = (-350, -20),
            #    space = :relative, fontsize = 24
            #)

            #if (k == length(results)) Legend(f[1, length(results) + 1], ax, "Reputation")
            #elseif (k == length(results) * 2) Legend(f[2, length(results) + 1], ax, "Reputation") end
        end
        Label(f[llm*2-1, :], sn_labels_a[llm],justification = :center,font = :bold, fontsize = 24)
    end

    # Save the plot inside the folder
    final_path = joinpath(plot_path, "reputation.pdf")
    save(final_path, f)
    display(f)
end

function plot_simplex(gradients::Vector, reputation::Vector, cooperation::Vector, stationary::Vector, disagreement::Vector, popsize::Int, sample_size::Int, folder_path::String, path_extension::String, file_extension::String)
    # Make plot folder
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)
    plot_path = joinpath(folder_path, "Plots/Simplex/")
    isdir(plot_path) || mkdir(plot_path)
    plot_path = joinpath(folder_path, "Plots/Simplex/"*path_extension)
    isdir(plot_path) || mkdir(plot_path)

    sample_states = hcat([x[1] for x in get_states_strat(sample_size)], [x[2] for x in get_states_strat(sample_size)])
    sample_states = round.((sample_states .* (popsize / sample_size)))

    # Calculate sum of stationary distributions for the sample states (necessary since sample states < all_states)
    all_states = hcat([x[1] for x in get_states_strat(popsize)], [x[2] for x in get_states_strat(popsize)])
    stationary_sum = zeros(length(get_states_strat(sample_size)))

    for i in 1:length(get_states_strat(popsize))
        closest_indx = closest_index(sample_states, all_states[i, :])
        stationary_sum[closest_indx] += stationary[i]
    end

    # Transform points
    all_transformed_points = transform_points(sample_states)
    gradients_as_points = hcat([x[1] for x in gradients], [x[2] for x in gradients])
    sample_grads = extract_data_sample(all_states, gradients_as_points, sample_states) * sample_size * 1.3
    transformed_gradients = transform_points(sample_grads)
    vector_field = hcat(all_transformed_points, transformed_gradients)

    # only get reputations and coop for specific points
    sample_rep = extract_data_vector_sample(all_states, reputation, sample_states, sample_size)
    sample_coop = extract_data_vector_sample(all_states, cooperation, sample_states, sample_size)
    sample_disagreement = extract_data_vector_sample(all_states, disagreement, sample_states, sample_size)

    # Plotting

    x_axis_lims, y_axis_lims = (-popsize/1.5, popsize/1.5), (-popsize, popsize/4.9)
    
    f_combined = Figure(backgroundcolor = :white, size = (1050, 700))
    ax_combined = Axis(f_combined[1,1], titlesize=24)

    # add indicators for each pop 
    labelFontSize = 24
    colorbarTickSize = 20
    point_scale = popsize*0.41

    y_axis_increase = popsize/10 # pulls points up
    #annotate!(baseplot,[(0, popsize*0.075, text("Disc", Plots.font("Arial", pointsize=labelFontSize))),(cos(-deg * 2) * popsize * 1.1, sin(-deg * 2) * popsize * 1.1, text("AllC", Plots.font("Arial", pointsize=labelFontSize), halign=:center)),(cos(-deg) * popsize * 1.1, sin(-deg) * popsize * 1.1, text("AllD", Plots.font("Arial", pointsize=labelFontSize), halign=:center))])
    minNorm, maxNorm = 1, 5
    vector_field[:, 3],vector_field[:, 4] = rescale_norm(vector_field[:, 3],vector_field[:, 4], minNorm, maxNorm)

    #rep = scatter(baseplot,all_transformed_points[:, 1], all_transformed_points[:, 2], markersize=point_scale, zcolor=sample_rep, color=cgrad(:RdYlBu_4, rev = false),markerstrokewidth=0, shape = :h)
    rep_color =:RdBu_9
    coop_color = :Purples_5
    disagreement_color = Reverse(:RdYlGn_9)
    normalized_stationary = stationary_sum ./ maximum(stationary_sum)
    statdist_color = :matter

    # Combined plot
    # Repeat the same stuff for the stat dist but add the reputations in the corner
    combinedpoint_scale = popsize*0.46
    minipoint_scale = popsize*0.16
    
    scatter!(ax_combined, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, marker=:hexagon, markersize=combinedpoint_scale*1.1, color=:black)
    scatter!(ax_combined, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, marker=:hexagon, markersize=combinedpoint_scale, color=normalized_stationary, colorrange = (0.0, 1.0), colormap=statdist_color)

    scatter!(ax_combined, (all_transformed_points[:, 1]./3).-popsize/2.5, (all_transformed_points[:, 2]./3).+popsize/8, marker=:hexagon, markersize=minipoint_scale*1.25, color=:black)
    scatter!(ax_combined, (all_transformed_points[:, 1]./3).-popsize/2.5, (all_transformed_points[:, 2]./3).+popsize/8, marker=:hexagon, markersize=minipoint_scale, color=sample_rep, colorrange = (0.0, 1.0), colormap=rep_color)

    scatter!(ax_combined, (all_transformed_points[:, 1]./3).+popsize/2.5, (all_transformed_points[:, 2]./3).+popsize/8, marker=:hexagon, markersize=minipoint_scale*1.25, color=:black)
    scatter!(ax_combined, (all_transformed_points[:, 1]./3).+popsize/2.5, (all_transformed_points[:, 2]./3).+popsize/8, marker=:hexagon, markersize=minipoint_scale, color=sample_disagreement, colorrange = (0.0, 0.5), colormap=disagreement_color)

    Colorbar(f_combined[1,2], colormap=statdist_color, limits=(0,1), size=15, ticks = 0:0.1:1, ticksize=10,tellheight=true,height = Relative(2.5/4),ticklabelsize=colorbarTickSize, label = "Stationary Distribution, σₙ", labelsize=20)
    Colorbar(f_combined[1,3], colormap=rep_color, limits=(0,1), size=15, ticks = 0:0.1:1, ticksize=10,tellheight=true,height = Relative(2.5/4),ticklabelsize=colorbarTickSize, label = "Average Reputation, rᴴ", labelsize=20)
    Colorbar(f_combined[1,4], colormap=disagreement_color, limits=(0,0.5), size=15, ticks = 0:0.1:0.5, ticksize=10,tellheight=true,height = Relative(2.5/4),ticklabelsize=colorbarTickSize, label = "Average Disagreement, qᵈ", labelsize=20)

    arrows!(ax_combined, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, vector_field[:, 3],vector_field[:, 4]);

    text!(ax_combined, 1, 1, text = "DISC", font = :bold, align = (:left, :top), offset = (-383, -8), space = :relative, fontsize = labelFontSize)
    text!(ax_combined, 1, 1, text = "ALLC", font = :bold, align = (:left, :top), offset = (-705, -550), space = :relative, fontsize = labelFontSize)
    text!(ax_combined, 1, 1, text = "ALLD", font = :bold, align = (:left, :top), offset = (-69, -550), space = :relative, fontsize = labelFontSize)
    # change plot lims and axis decorations
    xlims!(ax_combined, x_axis_lims)
    ylims!(ax_combined, y_axis_lims)
    hidedecorations!(ax_combined)  
    hidespines!(ax_combined)

    # Save all plots
    save(joinpath(plot_path, "simplex_combined_"*file_extension*".pdf"), f_combined)    
    #display(f_combined)
end

function plot_disagreement(results,x_axis_scale, folder_path, filenames::Vector{String}, plotLinesTitle::String)
    # Plot difference between ALLC and ALLD rep, and DISC and ALLD rep.
    # Make plot folder
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    f = Figure(backgroundcolor = :white, size = (600, 350))

    XAxisTitle = "Interactions with AA, τ"
    YAxisTitle = "Average Disagreement, qᵈ"

    ax = Axis(f[1,1], titlesize=24, xlabelsize=20, ylabelsize=20, 
        xticklabelsize=16, yticklabelsize=16, xlabel=XAxisTitle, ylabel=YAxisTitle,
        xticks=0:0.1:1.01, yticks=(0:0.1:0.5),xautolimitmargin=(0.0,0.01),
        yautolimitmargin=(0.06,0.06), yminorticksvisible=true)
    ylims!(ax, (-0.05,0.526))

    for v in eachindex(results)
        # Add lines incrementally for each norm
        scatterlines!(ax, x_axis_scale, results[v], label=filenames[v], color=line_colors[v],marker=markerTypes[v])
    end

    Legend(f[1, 2], ax, plotLinesTitle)

    # Save the plot inside the folder
    final_path = joinpath(plot_path, "disagreement.pdf")
    save(final_path, f)
    display(f)
end

function intermediate_norms_plot(folder_path::String, filename::String, title, filetag::String, norms::Vector, simplify::Bool=false)
    # Make plot folder
    isdir(folder_path) || mkdir(folder_path)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    # Make cooperation and disagreement folder
    coop_path = joinpath(plot_path, "Cooperation")
    isdir(coop_path) || mkdir(coop_path)
    disagreement_path = joinpath(plot_path, "Disagreement")
    isdir(disagreement_path) || mkdir(disagreement_path)
    reputation_path = joinpath(plot_path, "Average Reputation")
    isdir(reputation_path) || mkdir(reputation_path)

    res_path::String = joinpath(folder_path, "Results/ResultsBackup")

    results = utils.deserialize_file(joinpath(res_path,filename*"_results.jls"))
    x_range = 0:0.1:1.0

    for type in [1,2,3]   # type 1 = coop, type 2 = disagreement, type 3 = reputation
        
        res = [results[k][1][1][1] for k in eachindex(norms)]
        if type == 2
            res = [results[k][5][1][2] for k in eachindex(norms)]
        elseif type == 3
            res = [results[k][2][1][1] for k in eachindex(norms)]
        end
        results_matrix = reshape(res, length(x_range), length(x_range))
        
        label = type == 1 ? "Cooperation Index, I" : (type == 2 ? "Disagreement, qᵈ" : "Average Reputation, r")
        crange = (type == 1 ? (0,1) : (type == 2 ? (0, 0.5) : (0,1)))

        colormap = (type == 1 ? :viridis : (type == 2 ? :matter : :RdBu_6))
        # Plot the heatmap
        f = Figure(backgroundcolor = :white, size = (600, 600))
        xlabel, ylabel = (L"\text{Prob. assign good after cooperating with B}, d_{BC}", L"\text{Prob. assign good after defecting with B}, d_{BD}")
        ax = Axis(f[1, 1], xlabel=xlabel, ylabel=ylabel, titlesize=24, xlabelsize=20, 
        ylabelsize=20, xticklabelsize=16, yticklabelsize=16,xticks=0:0.2:1.01, yticks=(0:0.2:1.0))
        heatmap!(ax, x_range, x_range, results_matrix, colorrange=crange, colormap=colormap)
        Colorbar(f[1, 2], label=label, limits=crange, ticks=0.0:0.25:1.0, colormap=colormap)

        if (!simplify)

            # Add value labels on top of each cell
            for i in eachindex(x_range), j in eachindex(x_range)
                txtcolor = results_matrix[i, j] < 0.3 / type ? :white : :black
                text!(ax, "$(round(results_matrix[i,j], digits = 2))", position = (x_range[i], x_range[j]),
                    color = txtcolor, align = (:center, :center), fontsize=6)
            end
            text!(ax, "SH", position=(0.04, 0.04), color=:white)
            text!(ax, "IS", position=(0.945, 0.045), color=:white)
            text!(ax, "SJ", position=(0.04, 0.945), color=:white)
            text!(ax, "SS", position=(0.94, 0.94), color=:white)
        else
            text!(ax, "Shunning", fontsize=22, position=(0.01, 0.0), color=:white)
            text!(ax, "Image\nScore", fontsize=22, position=(0.84, -0.02), color=:white)
            text!(ax, "Stern-Judging", fontsize=22, position=(0.01, 0.96), color=:white)
            text!(ax, "Simple\nStanding", fontsize=22, position=(0.80, 0.92), color=:white)
        end


        # Save the plot inside the folder
        tag = (type == 1 ? "_coop" : (type == 2 ? "_disagreement" : "_reputation"))
        final_path = (type == 1 ? coop_path : (type == 2 ? disagreement_path : reputation_path))
        plot_path = joinpath(final_path, filetag*tag*".pdf")
        save(plot_path, f)
        display(f)
    end
end

function run_all_plots(folder_path::String, filenames::Vector{String}, sn_labels_a::Vector{String}, sn_labels_h::Vector{String}, includeSimplex::Bool=true)
    
    AAinfluence_vals::Vector = utils.parse_float64_array(String(utils.get_parameter_value(folder_path,"Range of Values")))
    popsize = parse(Int,utils.get_parameter_value(folder_path,"Population Size"))
    
    # Change what part of the results structure we want to extract
    # NOTE: All the files and results will be processed in the order sn_labels_a[1]*sn_labels_h[1], sn_labels_a[1]*sn_labels_h[2], ... sn_labels_a[N]*sn_labels_h[1], .. sn_labels_a[N]*sn_labels_h[M]
    # So the results must then be processed considering this order
    
    res_path::String = joinpath(folder_path, "Results/ResultsBackup")
    results = []
    for name in filenames
        push!(results, utils.deserialize_file(joinpath(res_path,name*"_results.jls")))
    end
    
    results_coop = []
    for i in eachindex(filenames)
        push!(results_coop, [results[i][k][1][1] for k in eachindex(AAinfluence_vals)])
    end

    titles_rep::Vector = ["ALLC", "ALLD", "DISC"]
    
    results_rep = []
    for i in eachindex(filenames)
        push!(results_rep, [results[i][k][2][1] for k in eachindex(AAinfluence_vals)])
    end

    results_disagreement = []
    for i in eachindex(filenames)
        push!(results_disagreement, [results[i][k][5][1][2] for k in eachindex(AAinfluence_vals)])
    end

    #results_coop = reshape(results_coop, length(sn_labels_a), length(sn_labels_h))
    #results_rep = reshape(results_rep, length(sn_labels_a), length(sn_labels_h))
    #results_disagreement = reshape(results_disagreement, length(sn_labels_a), length(sn_labels_h))

    plot_coop_indexes(results_coop, AAinfluence_vals, folder_path, sn_labels_a, sn_labels_h)
    plot_reputations(results_rep, AAinfluence_vals, titles_rep, folder_path, sn_labels_a, sn_labels_h)
    #plot_disagreement(results_disagreement, AAinfluence_vals, folder_path, sn_labels_a, sn_labels_h)

    # Make simples with all the social norms, for a given number of AAs
    sample_size = 20 # how many points to include in each simplex side
    results_reshaped = reshape(results, length(sn_labels_a), length(sn_labels_h))
    if (includeSimplex)
        for interVal in 0:0.1:1.0
            interAA_index = findfirst(x -> x == interVal, AAinfluence_vals)
            for i in eachindex(sn_labels_a)
                for j in eachindex(sn_labels_h)
                    gradients_i::Vector = results_reshaped[i,j][interAA_index][6]
                    reputation_i::Vector = utils.get_average_rep_states(results_reshaped[i,j][interAA_index], popsize)
                    cooperation_i::Vector = get_index_k(results_reshaped[i,j][interAA_index][1][2], 1)
                    stationary_i::Vector = results_reshaped[i,j][interAA_index][3]
                    disagreement_i = get_index_k(results_reshaped[i,j][interAA_index][5][2], 2)

                    plot_simplex(gradients_i, reputation_i, cooperation_i, stationary_i, disagreement_i, popsize, sample_size, folder_path, "AAinfluence="*string(interVal), sn_labels_a[i]*sn_labels_h[j]*"_"*string(interVal))
                end
            end
        end
    end
end

function norm_analysis_plot(folder_path::String,norm_names,norms,title,filename::String, legendtitle; special_labels::Bool=false,color_map::Dict{String, Any} = Dict(), simplify::Bool=true, for_good_norm::Bool=false, plot_ellipse::Bool=false, all_norms_std_dev=nothing, std_devs::Float64=1.0, families::Vector=Vector(), order=nothing, clarify_left_edge::Bool=false)
    # If clarify_left_edge is true, all points with dBC > 0.9 won't have an ellipse, and a second axis is generated to better plot these points

    # Make plot folder
    isdir(folder_path) || mkdir(folder_path)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)
    plot_path = joinpath(plot_path, "Norm Analysis")
    isdir(plot_path) || mkdir(plot_path)
    # Define colors and markers
    num_colors = length(line_colors)

    #xlabel, ylabel = for_good_norm ? ("Prob. assign good after cooperating with G, β¹", "Prob. assign good after defecting with G, β²") : ("Prob. assign good after cooperating with B, α¹", "Prob. assign good after defecting with B, α²")
    xlabel, ylabel = for_good_norm ? (L"\text{Prob. assign good after cooperating with G}, d_{GC}", L"\text{Prob. assign good after defecting with G}, d_{GD}") : (L"\text{Prob. assign good after cooperating with B}, d_{BC}", L"\text{Prob. assign good after defecting with B}, d_{BD}")

    #fig = Figure(size = (1200, 800))
    fig = clarify_left_edge ? Figure(size = (1100, 600)) : Figure(size = (900, 600))
    ax = Axis(fig[1, 1], xlabel = xlabel, ylabel = ylabel, titlesize=24, xlabelsize=20, 
        ylabelsize=20, xticklabelsize=16, yticklabelsize=16,xticks=0:0.2:1.01, yticks=(0:0.2:1.0),xautolimitmargin=(0.0,0.01))

    ylims!(ax, (-0.01,1.01))
    xlims!(ax, (-0.01,1.01))

    family_colors = Dict()
    color_index = 1

    # Used for keeping the colors yet changing the models that are displayed
    #models_to_hide = ["qwen2.5-7B-Instruct","qwen2.5-14B-Instruct", "gemma-2-9b-it", "gemma-2-27b-it"]
    models_to_hide = []

    # Shade background for x > 0.95
    edgecolor = RGBAf(0.0, 0.0, 0.0, 0.05)
    if clarify_left_edge poly!(ax, Point2f[(0.95, -0.01),(1.01, -0.01),(1.01, 1.01),(0.95, 1.01)], color = edgecolor, strokewidth = 0, inspectable = false) end 

    for family in families
        family_colors[family] = line_colors[(color_index - 1) % num_colors + 1]
        color_index += 1
    end

    if !isnothing(order)
        # Define desired order: "norm" first, then part norms
        ordered_norm_names = vcat(["norm"], order)
        ordered_norms = [norms[findfirst(x -> x == n, norm_names)] for n in ordered_norm_names if n in norm_names]

        # Reassign names to match the new order
        norm_names = ordered_norm_names
        norms = ordered_norms
    end

    # storing values for clarify_left_edge
    norms_on_edge = [norm for norm in norms if norm[3] > 0.95] # all valid norms
    norm_names_on_edge = [norm_names[k] for k in eachindex(norms) if norms[k] in norms_on_edge] # all norms in edge 
    norms_on_edge_y_bounds = Dict{String, Tuple{Float64, Float64}}()

    for (i, norm) in enumerate(norms)   # Draw elipses first so they never overlap points
        if isnothing(norm[3]) || isnothing(norm[4]) || norm_names[i] in models_to_hide
            continue
        end
        index_norms_x, index_norms_y = for_good_norm ? (1,2) : (3, 4)
        base_color = get(color_map, norm_names[i], get(family_colors, norm_names[i], line_colors[(i - 1) % num_colors + 1]))
        for fam in families
            if norm_names[i] in fam
                base_color = family_colors[fam]
                break
            end
        end
        if plot_ellipse && !isnothing(all_norms_std_dev) && haskey(all_norms_std_dev, norm_names[i])
            mean_vec = all_norms_std_dev[norm_names[i]]["mean_vector"]
            cov_mat = all_norms_std_dev[norm_names[i]]["cov_matrix"]
        
            # Extract the relevant 2x2 covariance submatrix
            cov_2d = cov_mat[index_norms_x:index_norms_y, index_norms_x:index_norms_y]
            μ_x, μ_y = mean_vec[index_norms_x], mean_vec[index_norms_y]
        
            # Compute eigenvalues and eigenvectors
            eigvals, eigvecs = eigen(cov_2d)
        
            # Generate ellipse points
            #     θ = range(0, 2π, length=100)
            #     ellipse = [sqrt(eigvals[1]) * cos(t) * eigvecs[:,1] + sqrt(eigvals[2]) * sin(t) * eigvecs[:,2] for t in θ]
            # Eigenvalues define the length of the semi-axes
            scaling_factor = compute_scaling(std_devs)
            a = sqrt(eigvals[2] * scaling_factor)  # Major axis
            b = sqrt(eigvals[1] * scaling_factor)  # Minor axis
        
            # Eigenvectors define the orientation (rotation)
            θ = range(0, 2π, length=100)
            ellipse = [a * cos(t) * eigvecs[:,2] + b * sin(t) * eigvecs[:,1] for t in θ]
        
            # Extract x and y coordinates
            ellipse_x = μ_x .+ [e[1] for e in ellipse]
            ellipse_y = μ_y .+ [e[2] for e in ellipse]
            norms_on_edge_y_bounds[norm_names[i]] = (minimum(ellipse_y), maximum(ellipse_y))
        
            # Plot the ellipse
            if !clarify_left_edge || !(norm in norms_on_edge)
                lines!(ax, ellipse_x, ellipse_y, color=(base_color, 0.4), linewidth=2, linestyle=:dash)
            end
        end
    end

    for family in families
        if !isempty(intersect(family, models_to_hide)) continue end
        for j in 1:(length(family) - 1)
            idx1 = findfirst(x -> x == family[j], norm_names)
            idx2 = findfirst(x -> x == family[j+1], norm_names)
            if !isnothing(idx1) && !isnothing(idx2)
                x_vals = for_good_norm ? [norms[idx1][1], norms[idx2][1]] : [norms[idx1][3], norms[idx2][3]]
                y_vals = for_good_norm ? [norms[idx1][2], norms[idx2][2]] : [norms[idx1][4], norms[idx2][4]]
                lines!(ax, x_vals, y_vals, color=family_colors[family], linewidth=2, linestyle=:dash, alpha=0.6)
            end
        end
    end

    for (i, norm) in enumerate(norms)
        if isnothing(norm[3]) || isnothing(norm[4]) || norm_names[i] in models_to_hide
            continue
        end

        # Determine color and marker
        base_color = get(color_map, norm_names[i], get(family_colors, norm_names[i], line_colors[(i - 1) % num_colors + 1]))
        for fam in families
            if norm_names[i] in fam
                base_color = family_colors[fam]
                break
            end
        end
        marker = markerTypes[(i-1) % length(markerTypes) + 1]

        # Add to plot
        index_norms_x, index_norms_y = for_good_norm ? (1,2) : (3, 4)
        scatter!(ax, (norm[index_norms_x], norm[index_norms_y]), color = base_color, markersize = ifelse(special_labels && norm_names[i] == "norm", 30, 25),
            strokewidth = ifelse(special_labels, 1.5, 1.5), marker = marker, label = norm_names[i])
    end

    if clarify_left_edge

        ax_edge = Axis(fig[1, 2], xlabel = L"\text{Models with d_{BC} > 0.95}", ylabel = "", titlesize=24, xlabelsize=20, 
            ylabelsize=20, xticklabelsize=16, yticklabelsize=0,xticks=(Float64[], String[]), yticks=(0:0.2:1.0),xautolimitmargin=(0.04,0.04), backgroundcolor=edgecolor)
        ax_edge.xticklabelrotation = 45 #xticks=(1:length(norm_names_on_edge), norm_names_on_edge)
        ylims!(ax_edge, (-0.01,1.01))

        name_to_xpos = Dict(name => i for (i, name) in enumerate(norm_names_on_edge))

        for (i, norm) in enumerate(norms) # Draw second axis where we clarify norms on the SS-IS axis 
            if isnothing(norm[3]) || isnothing(norm[4]) || norm_names[i] in models_to_hide || !(norm in norms_on_edge)
                continue
            end
            base_color = get(color_map, norm_names[i], get(family_colors, norm_names[i], line_colors[(i - 1) % num_colors + 1]))
            for fam in families
                if norm_names[i] in fam
                    base_color = family_colors[fam]
                    break
                end
            end

            # Plot the uncertainty line
            min_y, max_y = norms_on_edge_y_bounds[norm_names[i]]
            x = name_to_xpos[norm_names[i]]
            lines!(ax_edge, [x,x], [min_y, max_y], color = base_color, linewidth = 3)

            # Plot the point
            marker = markerTypes[(i - 1) % length(markerTypes) + 1]
            scatter!(ax_edge, [x], [for_good_norm ? norm[2] : norm[4]], color = base_color,markersize = ifelse(special_labels && norm_names[i] == "norm", 30, 25),strokewidth = ifelse(special_labels, 1.5, 1.5),marker = marker,label = norm_names[i])
        end
    end

    #axislegend(ax, position = :lc)
    legend = Legend(fig, ax, legendtitle)
    if clarify_left_edge fig[1, 3] = legend else fig[1, 2] = legend end

    if (!clarify_left_edge)
        colsize!(fig.layout, 1, Relative(11/16))
    else
        colsize!(fig.layout, 1, Relative(9/16))
        #colsize!(fig.layout, 2, Relative(2/11))
    end
    #colsize!(fig.layout, 1, Fixed(700)) 
    if (!for_good_norm)
        if (!simplify)
            text!(ax, "SH", fontsize=25, position=(0.02, 0.01), color=:black)
            text!(ax, "IS", fontsize=25, position=(0.93, 0.01), color=:black)
            text!(ax, "SJ", fontsize=25, position=(0.02, 0.94), color=:black)
            text!(ax, "SS", fontsize=25, position=(0.92, 0.94), color=:black)
        else
            text!(ax, "Shunning", fontsize=22, position=(0.1, 0.125), color=:black, align=(:center, :top))
            text!(ax, "Image\nScore", fontsize=22, position=(0.9, 0.13), color=:black, align=(:center, :top))
            text!(ax, "Stern\nJudging", fontsize=22, position=(0.1, 0.95), color=:black, align=(:center, :top))
            text!(ax, "Simple\nStanding", fontsize=22, position=(0.9, 0.95), color=:black, align=(:center, :top))
        end
    end

    save(joinpath(plot_path, filename * ".pdf"), fig)
    if !special_labels
        display(fig)
    end
    #println("Plot saved to $(joinpath(plot_path, filename * ".pdf"))")
end

function variance_plotter(data, model_names, folder_path)
    # Reorder data based on model_names
    ordered_data = []
    for model_name in model_names
        if haskey(data, model_name)
            push!(ordered_data, (model_name, data[model_name]))
        end
    end

    # Extract norms for each category (gender, region, tag) for each model
    function extract_norms_for_models(dat, category_list)
        model_norms = []
        for (model_name, model_data) in dat
            norms_for_model = []
            for category in category_list
                if haskey(model_data, category)
                    push!(norms_for_model, model_data[category])  # Extract the norm data
                end
            end
            push!(model_norms, norms_for_model)
        end
        return model_norms
    end
    
    # Extract norms for each category
    gender_norms_per_model = extract_norms_for_models(ordered_data, utils.norm_gender)
    region_norms_per_model = extract_norms_for_models(ordered_data, utils.norm_region)
    tag_norms_per_model = extract_norms_for_models(ordered_data, utils.norm_tag)
    
    # Calculate variance for each model and category
    function calculate_variance(norms_per_model)
        model_variances = []
        for norms_for_model in norms_per_model
            println(norms_for_model)
            push!(model_variances, var(norms_for_model))  # Compute variance for each norm group
            #push!(model_variances, utils.jsd_for_norms(norms_for_model))
        end
        return model_variances
    end
    
    # Calculate variances for each category per model
    gender_variances_per_model = calculate_variance(gender_norms_per_model)
    region_variances_per_model = calculate_variance(region_norms_per_model)
    tag_variances_per_model = calculate_variance(tag_norms_per_model)

    # Prepare the data for plotting
    labels = ["Gender", "Region", "Tag"]
    
    # For each model, calculate the variance per category
    model_variances = []
    for i in 1:length(ordered_data)
        model_variances_for_model = [
            maximum(gender_variances_per_model[i]), 
            maximum(region_variances_per_model[i]), 
            maximum(tag_variances_per_model[i])
        ]
        push!(model_variances, model_variances_for_model)
    end

    # Prepare the plot
    fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1], xlabel="Model", ylabel="Variance")
    ax.xticks = (1:length(model_names), model_names)  # Set model names as x-ticks
    ax.xticklabelrotation = 45  # Angle the labels by 45 degrees
    #ax.yticks = [0.0, 0.25, 0.5, 0.75, 1.0]

    limits!(ax, 0, length(model_names) + 1, 0, 1.0)
    ylims!(ax, 0, 0.25)
    ax.yticks = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


    # For each model, plot the variance bars
    for (i, model_variance) in enumerate(model_variances)
        barplot!(ax, [i-0.2, i, i+0.2], model_variance, color=[:blue, :orange, :green], width=0.2)
    end

    # Create a legend manually
    leg_labels = ["Gender", "Region", "Tag"]
    leg_colors = [:blue, :orange, :green]
    Legend(fig[1, 2], [PolyElement(color=color) for color in leg_colors], leg_labels, "Categories", framevisible=false)

    # Show the plot
    display(fig)

    # Make plot folder
    isdir(folder_path) || mkdir(folder_path)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)
    plot_path = joinpath(plot_path, "Norm Analysis")
    isdir(plot_path) || mkdir(plot_path)

    save(joinpath(plot_path, "norm_variance_analysis.pdf"), fig)
end

# Specific functions for article plots

function public_private_intermediate_norms_plot(folder_path::String,folder_path_public::String,folder_path_private::String, filename_public::String, filename_private::String, filetag::String, norms::Vector, simplify::Bool=false; models_to_show=[], norm_names_LLMs=[], norms_LLMs=[], color_map_LLMs::Dict{String, Any} = Dict(), plot_ellipse_LLMs::Bool=false, all_norms_std_dev=nothing, std_devs_LLMs::Float64=1.0, families_LLMs::Vector=Vector(), order_LLMs=nothing)
    # Make plot folder
    isdir(folder_path) || mkdir(folder_path)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    # Make cooperation and disagreement folder
    coop_path = joinpath(plot_path, "Cooperation")
    isdir(coop_path) || mkdir(coop_path)
    disagreement_path = joinpath(plot_path, "Disagreement")
    isdir(disagreement_path) || mkdir(disagreement_path)
    reputation_path = joinpath(plot_path, "Average Reputation")
    isdir(reputation_path) || mkdir(reputation_path)

    res_path_public::String = joinpath(folder_path_public, "Results/ResultsBackup")
    results_public = utils.deserialize_file(joinpath(res_path_public,filename_public*"_results.jls"))
    res_path_private::String = joinpath(folder_path_private, "Results/ResultsBackup")
    results_private = utils.deserialize_file(joinpath(res_path_private,filename_private*"_results.jls"))
    x_range = 0:0.1:1.0

    for type in [1,2,3]   # type 1 = coop, type 2 = disagreement, type 3 = reputation
        
        res_public = [results_public[k][1][1][1] for k in eachindex(norms)]
        res_private = [results_private[k][1][1][1] for k in eachindex(norms)]
        if type == 2
            res_public = [results_public[k][5][1][2] for k in eachindex(norms)]
            res_private = [results_private[k][5][1][2] for k in eachindex(norms)]
        elseif type == 3
            res_public = [results_public[k][2][1][1] for k in eachindex(norms)]
            res_private = [results_private[k][2][1][1] for k in eachindex(norms)]
        end
        results_matrix_public = reshape(res_public, length(x_range), length(x_range))
        results_matrix_private = reshape(res_private, length(x_range), length(x_range))
        
        label = type == 1 ? L"\text{Cooperation Index, I}" : (type == 2 ? L"\text{Disagreement, qᵈ}" : L"\text{Average Reputation, r}")
        crange = (type == 1 ? (0,1) : (type == 2 ? (0, 0.5) : (0,1)))

        colormap = (type == 1 ? :viridis : (type == 2 ? :matter : :RdBu_6))

        # Plot the heatmap
        #title_plot = (type == 1 ? L"\text{Cooperation under social norm = }(1, 0, d_{BC}, d_{BD})" : (type == 2 ? L"\text{Disagreement under social norm = }(1, 0, d_{BC}, d_{BD})" : L"\text{Average reputation under social norm = }(1, 0, d_{BC}, d_{BD})"))
        f = Figure(backgroundcolor = :white, size = (1300, 600))

        #Label(f[0, 1:2], title_plot, fontsize=25, halign=:center)

        title_public = L"\text{Public reputations}"
        title_private = L"\text{Private reputations}"
        xlabel, ylabel = (L"\text{Prob. assign good after cooperating with B}, d_{BC}", L"\text{Prob. assign good after defecting with B}, d_{BD}")

        ax_public = Axis(f[1, 1], xlabel=xlabel, ylabel=ylabel, title=title_public, titlesize=24, xlabelsize=20, 
        ylabelsize=20, xticklabelsize=16, yticklabelsize=16,xticks=0:0.2:1.01, yticks=(0:0.2:1.0))
        ax_private = Axis(f[1, 2], title=title_private, titlesize=24, xlabelsize=20, 
        ylabelsize=20, xticklabelsize=16, yticklabelsize=16,xticks=0:0.2:1.01, yticks=(0:0.2:1.0))
        heatmap!(ax_public, x_range, x_range, results_matrix_public, colorrange=crange, colormap=colormap)
        heatmap!(ax_private, x_range, x_range, results_matrix_private, colorrange=crange, colormap=colormap)
        Colorbar(f[1, 3], label=label, limits=crange, ticks=0.0:0.25:1.0, colormap=colormap, labelsize=20)

        for ax in [ax_public, ax_private]
            # Add LLM norm markers
            family_colors = Dict()
            num_colors = length(line_colors)

            color_index = 1
            for family in families_LLMs
                family_colors[family] = line_colors[(color_index - 1) % num_colors + 1]
                color_index += 1
            end
            for (i, norm) in enumerate(norms_LLMs)
                if isnothing(norm[3]) || isnothing(norm[4]) || !(norm_names_LLMs[i] in models_to_show) continue end
                # Determine color and marker
                base_color = get(color_map_LLMs, norm_names_LLMs[i], get(family_colors, norm_names_LLMs[i], line_colors[(i - 1) % num_colors + 1]))
                for fam in families_LLMs
                    if norm_names_LLMs[i] in fam
                        base_color = family_colors[fam]
                        break
                    end
                end
                marker = markerTypes[(i-1) % length(markerTypes) + 1]
                # Add to plot
                index_norms_x, index_norms_y = (3, 4)
                scatter!(ax, (norm[index_norms_x], norm[index_norms_y]), color = base_color, markersize = 20,
                    strokewidth = 1.5, marker = marker, label = norm_names_LLMs[i])
    
                if plot_ellipse_LLMs && !isnothing(all_norms_std_dev) && haskey(all_norms_std_dev, norm_names_LLMs[i])
                    mean_vec = all_norms_std_dev[norm_names_LLMs[i]]["mean_vector"]
                    cov_mat = all_norms_std_dev[norm_names_LLMs[i]]["cov_matrix"]
    
                    cov_2d = cov_mat[index_norms_x:index_norms_y, index_norms_x:index_norms_y]
                    μ_x, μ_y = mean_vec[index_norms_x], mean_vec[index_norms_y]
                
                    eigvals, eigvecs = eigen(cov_2d)
                
                    # Generate ellipse points
                    scaling_factor = compute_scaling(std_devs_LLMs)
                    a = sqrt(eigvals[2] * scaling_factor)  # Major axis
                    b = sqrt(eigvals[1] * scaling_factor)  # Minor axis
                
                    # Eigenvectors define the orientation (rotation)
                    θ = range(0, 2π, length=100)
                    ellipse = [a * cos(t) * eigvecs[:,2] + b * sin(t) * eigvecs[:,1] for t in θ]
                
                    # Extract x and y coordinates
                    ellipse_x = μ_x .+ [e[1] for e in ellipse]
                    ellipse_y = μ_y .+ [e[2] for e in ellipse]
                
                    # Plot the ellipse
                    lines!(ax, ellipse_x, ellipse_y, color=(base_color, 0.4), linewidth=2, linestyle=:dash)
                end
            end
    
            for family in families_LLMs
                if isempty(intersect(family, models_to_show)) continue end
                for j in 1:(length(family) - 1)
                    idx1 = findfirst(x -> x == family[j], norm_names_LLMs)
                    idx2 = findfirst(x -> x == family[j+1], norm_names_LLMs)
                    if norm_names_LLMs[idx1] in models_to_show && norm_names_LLMs[idx2] in models_to_show
                        if !isnothing(idx1) && !isnothing(idx2)
                            x_vals = [norms_LLMs[idx1][3], norms_LLMs[idx2][3]]
                            y_vals = [norms_LLMs[idx1][4], norms_LLMs[idx2][4]]
                            lines!(ax, x_vals, y_vals, color=family_colors[family], linewidth=2, linestyle=:dash, alpha=0.6)
                        end
                    end
                end
            end

            # Add corner text
            if (!simplify)
                # Add value labels on top of each cell
                results_matrix = ax == ax_public ? results_matrix_public : results_matrix_private
                for i in eachindex(x_range), j in eachindex(x_range)
                    txtcolor = results_matrix[i, j] < 0.3 / type ? :white : :black
                    text!(ax, "$(round(results_matrix[i,j], digits = 2))", position = (x_range[i], x_range[j]),
                        color = txtcolor, align = (:center, :center), fontsize=6)
                end
                text!(ax, "SH", fontsize=22, position=(0.04, 0.04), color=:white)
                text!(ax, "IS", fontsize=22, position=(0.90, 0.04), color=:white)
                text!(ax, "SJ", fontsize=22, position=(0.04, 0.90), color=:white)
                text!(ax, "SS", fontsize=22, position=(0.90, 0.90), color=:white)
            else
                text!(ax, "Shunning", fontsize=22, position=(0.010, 0.05), color=:white)
                text!(ax, "Image\nScore", fontsize=22, position=(0.8, 0.03), color=:white)
                text!(ax, "Stern\nJudging", fontsize=22, position=(0.035, 0.86), color=:white)
                text!(ax, "Simple\nStanding", fontsize=22, position=(0.78, 0.86), color=:white)
            end
        end

        legend = Legend(f, ax_private, "Models")
        f[1, 4] = legend
        #colsize!(f.layout, 4, Relative(1/16))

        # Save the plot inside the folder
        tag = (type == 1 ? "_coop" : (type == 2 ? "_disagreement" : "_reputation"))
        final_path = (type == 1 ? coop_path : (type == 2 ? disagreement_path : reputation_path))
        plot_path = joinpath(final_path, filetag*tag*".pdf")
        save(plot_path, f)
        if type == 1 display(f) end
    end
end

function llm_intermediate_norms_plot(folder_path::String,folder_path_results::String,filename_results::String, filetag::String, norms::Vector, simplify::Bool=false; models_to_show=[], norm_names_LLMs=[], norms_LLMs=[], color_map_LLMs::Dict{String, Any} = Dict(), plot_ellipse_LLMs::Bool=false, all_norms_std_dev=nothing, std_devs_LLMs::Float64=1.0, families_LLMs::Vector=Vector(), order_LLMs=nothing)
    # Make plot folder
    isdir(folder_path) || mkdir(folder_path)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    # Make cooperation and disagreement folder
    coop_path = joinpath(plot_path, "Cooperation")
    isdir(coop_path) || mkdir(coop_path)
    disagreement_path = joinpath(plot_path, "Disagreement")
    isdir(disagreement_path) || mkdir(disagreement_path)
    reputation_path = joinpath(plot_path, "Average Reputation")
    isdir(reputation_path) || mkdir(reputation_path)

    res_path_public::String = joinpath(folder_path_results, "Results/ResultsBackup")
    results_public = utils.deserialize_file(joinpath(res_path_public,filename_results*"_results.jls"))
    x_range = 0:0.1:1.0

    for type in [1,2,3]   # type 1 = coop, type 2 = disagreement, type 3 = reputation
        
        res_public = [results_public[k][1][1][1] for k in eachindex(norms)]
        if type == 2
            res_public = [results_public[k][5][1][2] for k in eachindex(norms)]
        elseif type == 3
            res_public = [results_public[k][2][1][1] for k in eachindex(norms)]
        end
        results_matrix_public = reshape(res_public, length(x_range), length(x_range))
        
        label = type == 1 ? L"\text{Cooperation Index, I}" : (type == 2 ? L"\text{Disagreement, qᵈ}" : L"\text{Average Reputation, r}")
        crange = (type == 1 ? (0,1) : (type == 2 ? (0, 0.5) : (0,1)))

        colormap = (type == 1 ? :viridis : (type == 2 ? :matter : :RdBu_6))

        # Plot the heatmap
        f = Figure(backgroundcolor = :white, size = (900, 600))
        xlabel, ylabel = (L"\text{Prob. assign good after cooperating with B}, d_{BC}", L"\text{Prob. assign good after defecting with B}, d_{BD}")

        ax = Axis(f[1, 1], xlabel=xlabel, ylabel=ylabel, titlesize=24, xlabelsize=20, 
            ylabelsize=20, xticklabelsize=16, yticklabelsize=16,xticks=0:0.2:1.01, yticks=(0:0.2:1.0))
        heatmap!(ax, x_range, x_range, results_matrix_public, colorrange=crange, colormap=colormap)
        #contour!(ax, x_range, x_range, results_matrix_public, labels=true, color=:black, levels = [0.4, 0.7, 0.9])
        Colorbar(f[1, 2], label=label, limits=crange, ticks=0.0:0.25:1.0, colormap=colormap, labelsize=20)

        # Add LLM norm markers
        family_colors = Dict()
        num_colors = length(line_colors)

        color_index = 1
        for family in families_LLMs
            family_colors[family] = line_colors[(color_index - 1) % num_colors + 1]
            color_index += 1
        end
        for (i, norm) in enumerate(norms_LLMs)
            if isnothing(norm[3]) || isnothing(norm[4]) || !(norm_names_LLMs[i] in models_to_show) continue end
            # Determine color and marker
            base_color = get(color_map_LLMs, norm_names_LLMs[i], get(family_colors, norm_names_LLMs[i], line_colors[(i - 1) % num_colors + 1]))
            for fam in families_LLMs
                if norm_names_LLMs[i] in fam
                    base_color = family_colors[fam]
                    break
                end
            end
            marker = markerTypes[(i-1) % length(markerTypes) + 1]
            # Add to plot
            index_norms_x, index_norms_y = (3, 4)
            scatter!(ax, (norm[index_norms_x], norm[index_norms_y]), color = base_color, markersize = 20,
                strokewidth = 1.5, marker = marker, label = norm_names_LLMs[i])
    
            if plot_ellipse_LLMs && !isnothing(all_norms_std_dev) && haskey(all_norms_std_dev, norm_names_LLMs[i])
                mean_vec = all_norms_std_dev[norm_names_LLMs[i]]["mean_vector"]
                cov_mat = all_norms_std_dev[norm_names_LLMs[i]]["cov_matrix"]
    
                cov_2d = cov_mat[index_norms_x:index_norms_y, index_norms_x:index_norms_y]
                μ_x, μ_y = mean_vec[index_norms_x], mean_vec[index_norms_y]
            
                eigvals, eigvecs = eigen(cov_2d)
            
                # Generate ellipse points
                scaling_factor = compute_scaling(std_devs_LLMs)
                a = sqrt(eigvals[2] * scaling_factor)  # Major axis
                b = sqrt(eigvals[1] * scaling_factor)  # Minor axis
            
                # Eigenvectors define the orientation (rotation)
                θ = range(0, 2π, length=100)
                ellipse = [a * cos(t) * eigvecs[:,2] + b * sin(t) * eigvecs[:,1] for t in θ]
            
                # Extract x and y coordinates
                ellipse_x = μ_x .+ [e[1] for e in ellipse]
                ellipse_y = μ_y .+ [e[2] for e in ellipse]
            
                # Plot the ellipse
                lines!(ax, ellipse_x, ellipse_y, color=(base_color, 0.4), linewidth=2, linestyle=:dash)
            end
        end
    
        for family in families_LLMs
            if isempty(intersect(family, models_to_show)) continue end
            for j in 1:(length(family) - 1)
                if (family[j] in models_to_show)  
                    idx1 = findfirst(x -> x == family[j], norm_names_LLMs)
                    idx2 = findfirst(x -> x == family[j+1], norm_names_LLMs)
                    if !isnothing(idx1) && !isnothing(idx2)
                        x_vals = [norms_LLMs[idx1][3], norms_LLMs[idx2][3]]
                        y_vals = [norms_LLMs[idx1][4], norms_LLMs[idx2][4]]
                        lines!(ax, x_vals, y_vals, color=family_colors[family], linewidth=2, linestyle=:dash, alpha=0.6)
                    end
                end
            end
        end

        # Add corner text
        if (!simplify)
            # Add value labels on top of each cell
            for i in eachindex(x_range), j in eachindex(x_range)
                txtcolor = results_matrix_public[i, j] < 0.3 / type ? :white : :black
                text!(ax, "$(round(results_matrix_public[i,j], digits = 2))", position = (x_range[i], x_range[j]),
                    color = txtcolor, align = (:center, :center), fontsize=6)
            end
            text!(ax, "SH", fontsize=22, position=(0.04, 0.04), color=:white)
            text!(ax, "IS", fontsize=22, position=(0.90, 0.04), color=:white)
            text!(ax, "SJ", fontsize=22, position=(0.04, 0.90), color=:white)
            text!(ax, "SS", fontsize=22, position=(0.90, 0.90), color=:white)
        else
            # text!(ax, "Shunning", fontsize=22, position=(0.010, 0.05), color=:black)
            # text!(ax, "Image\nScore", fontsize=22, position=(0.8, 0.03), color=:black)
            # text!(ax, "Stern\nJudging", fontsize=22, position=(0.035, 0.86), color=:black)
            # text!(ax, "Simple\nStanding", fontsize=22, position=(0.78, 0.86), color=:black)

            text!(ax, "Shunning", fontsize=22, position=(0.1, 0.125), color=:white, align=(:center, :top))
            text!(ax, "Image\nScore", fontsize=22, position=(0.9, 0.13), color=:white, align=(:center, :top))
            text!(ax, "Stern\nJudging", fontsize=22, position=(0.1, 0.95), color=:white, align=(:center, :top))
            text!(ax, "Simple\nStanding", fontsize=22, position=(0.9, 0.95), color=:white, align=(:center, :top))
        end

        legend = Legend(f, ax, "Models")
        f[1, 3] = legend
        #colsize!(f.layout, 4, Relative(1/16))

        # Save the plot inside the folder
        tag = (type == 1 ? "_coop" : (type == 2 ? "_disagreement" : "_reputation"))
        final_path = (type == 1 ? coop_path : (type == 2 ? disagreement_path : reputation_path))
        plot_path = joinpath(final_path, filetag*tag*".pdf")
        save(plot_path, f)
        if type == 1 display(f) end
    end
end

function plot_public_private_coop_indexes_varying_bc_with_uncertainty(bc_vals, folder_path, norm_names, all_results_public, all_results_private, legend_label)
    isdir(folder_path) || mkdir(folder_path)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)
    
    f = Figure(backgroundcolor=:white, size=(850, 420))
    ax_public = Axis(f[1,1], titlesize=24, xlabelsize=20, ylabelsize=20, 
            xticklabelsize=16, yticklabelsize=16,
            title=L"\text{Public reputations}", xlabel=L"\text{Benefit-cost ratio, b/c}", ylabel=L"\text{Cooperation Index, I}",
            xticks=(1.0:1.0:8.0), yticks=(0:0.2:1.0), xautolimitmargin=(0.0,0.0),
            yautolimitmargin=(0.02,0.02), yminorticksvisible=true)
    ax_private = Axis(f[1,2], titlesize=24, xlabelsize=20, ylabelsize=20, 
            xticklabelsize=16, yticklabelsize=16,
            title=L"\text{Private reputations}", xlabel="", ylabel="",
            xticks=(1.0:1.0:8.0), yticks=(0:0.2:1.0), xautolimitmargin=(0.0,0.0),
            yautolimitmargin=(0.02,0.02), yminorticksvisible=true)
    ylims!(ax_public, (-0.01,1.011))
    ylims!(ax_private, (-0.01,1.011))

    #Label(f[0, 1:2], L"\text{Cooperation under LLM social norms}", fontsize=25, halign=:center)

    full_family = [["gpt-3.5-turbo","gpt-4o"], ["qwen2.5-7B-Instruct","qwen2.5-7B-Instruct-universalization","qwen2.5-7B-Instruct-empathizing","qwen2.5-7B-Instruct-signaling","qwen2.5-7B-Instruct-motivation","qwen2.5-14B-Instruct"],["gemma-2-9b-it", "gemma-2-27b-it"], ["gemini-1.5-pro","gemini-1.5-pro-universalization","gemini-1.5-pro-empathizing","gemini-1.5-pro-signaling","gemini-1.5-pro-motivation","gemini-2.0-flash"], ["mistral-small", "mistral-large"], ["phi-3.5-mini-instruct","phi-4","phi-4-universalization","phi-4-empathizing","phi-4-signaling","phi-4-motivation"],["llama-2-7b-chat-hf","llama-2-13b-chat-hf","llama-3.1-8B-Instruct", "llama-3.1-8B-Instruct-universalization", "llama-3.1-8B-Instruct-empathizing", "llama-3.1-8B-Instruct-signaling", "llama-3.1-8B-Instruct-motivation","llama-3.3-70B-Instruct"],["claude-3-5-haiku","claude-3-7-sonnet"],["grok-2"],["deepseek-v3", "deepseek-r1"]]
    #line_colors_full = (RGBA(0.1176, 0.5647, 1.0), RGBA(1.0, 0.4980, 0.0549), RGBA(0.1961, 0.8039, 0.1961), RGBA(1.0, 0.8431, 0.0), RGBA(0.8, 0.1784, 0.5765), RGBA(0.0, 0.8078, 0.8196), RGBA(1.0, 0.0, 0.0), RGBA(0.5804, 0.0, 0.8275), RGBA(0.5020, 0.7020, 0.0), RGBA(0.4118, 0.4118, 0.4118), RGBA(1.0, 0.4980, 0.0549))
    line_colors_full = (RGBA(0.3686, 0.5059, 0.7098), RGBA(1.0, 0.4980, 0.0549), RGBA(0.1961, 0.8039, 0.1961), RGBA(1.0, 0.8431, 0.0), RGBA(0.8, 0.1784, 0.5765), RGBA(0.0, 0.8078, 0.8196), RGBA(0.9216, 0.3843, 0.2078), RGBA(0.635, 0.475, 0.725), RGBA(0.5608, 0.6902, 0.1961), RGBA(0.4118, 0.4118, 0.4118), RGBA(0.9216, 0.3843, 0.2078))
    model_color = Dict()
    for fam in eachindex(full_family)
        for model in full_family[fam]
            model_color[model] = line_colors_full[fam]
        end
    end
    #line_colors_bc =  (RGBA(0.3686, 0.5059, 0.7098),RGBA(0.9216, 0.3843, 0.2078), RGBA(0.635, 0.475, 0.725), RGBA(0.5608,0.6902,0.1961), RGBA(0.5, 0.5, 0.5), RGBA(0.2333, 0.2333, 0.2333),RGBA(0.5569, 0.1765, 0.6745),  RGBA(0.2039, 0.7059, 0.6824),RGBA(0.9765, 0.4510, 0.8824),RGBA(0.9373, 0.8118, 0.2353),  RGBA(0.1294, 0.1294, 0.7882))

    for (results, axis) in zip([all_results_public, all_results_private],[ax_public, ax_private])

        for v in eachindex(norm_names)
            norm_name = norm_names[v]
            coop_vals = results[norm_name]  # This should be a list of vectors
        
            coop_matrix = hcat(coop_vals...)  # Convert list of vectors into a matrix where each column is a different norm
        
            min_vals = minimum(coop_matrix, dims=2)[:]
            max_vals = maximum(coop_matrix, dims=2)[:]
            mean_vals = coop_matrix[:,1]  # Use the center norm (first column)
        
            band!(axis, bc_vals, min_vals, max_vals, color=(model_color[norm_name], 0.3))  # Shaded region
            println(norm_name)
            scatterlines!(axis, bc_vals, mean_vals, label=norm_name, color=model_color[norm_name], marker=markerTypes[v])

            if (results == all_results_private) axislegend(axis,legend_label,orientation = :vertical, nbanks=1, position = (0.0, 1.0), fontsize=16) end
        end
    end
        
    save(joinpath(plot_path, "coop_public_private_index_per_norm_bc_uncertainty.pdf"), f)
    display(f)
end

function parse_model_name(name::String)
    suffixes = [
        "-motivation",
        "-universalization",
        "-empathizing",
        "-signaling",
        #"-public",
        #"-private"
    ]
    for suffix in suffixes
        if endswith(name, suffix)
            return name[1:end-length(suffix)], suffix
        end
    end
    return name, ""
end

# Helper function for ellipse calculation (keep as before)
function compute_scaling(std_devs)
    # Placeholder: Needs proper statistical basis if used seriously
    return std_devs^2
end

function norm_analysis_intervention_plot(folder_path::String,norm_names::Vector{String},norms::Vector,title, filename::String,legendtitle::String; intervention_markers::Dict{String, Symbol} = Dict("" => :circle,"-universalization" => :utriangle,"-empathizing" => :dtriangle,"-signaling" => :rect,"-motivation" => :xcross),special_labels::Bool=false,simplify::Bool=true,for_good_norm::Bool=false,plot_ellipse::Bool=false,all_norms_std_dev=nothing,std_devs::Float64=1.0,families::Vector=Vector(),order=nothing)
    isdir(folder_path) || mkdir(folder_path)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)
    plot_path = joinpath(plot_path, "Norm Analysis")
    isdir(plot_path) || mkdir(plot_path)

    num_colors = length(line_colors)
    xlabel, ylabel = for_good_norm ? (L"\text{Prob. assign good after cooperating with G}, d_{GC}", L"\text{Prob. assign good after defecting with G}, d_{GD}") : (L"\text{Prob. assign good after cooperating with B}, d_{BC}", L"\text{Prob. assign good after defecting with B}, d_{BD}")

    fig = Figure(size = (900, 600)) 
    ax = Axis(fig[1, 1], xlabel = xlabel, ylabel = ylabel, titlesize=24, xlabelsize=20,
        ylabelsize=20, xticklabelsize=16, yticklabelsize=16,xticks=0:0.2:1.01, yticks=(0:0.2:1.0),xautolimitmargin=(0.0,0.01))

    ylims!(ax, (-0.01,1.01))
    xlims!(ax, (-0.01,1.01))

    family_colors = Dict()
    color_index = 1
    # Assign family colors
    for family in families
        current_color = line_colors[(color_index - 1) % num_colors + 1]
        assigned = false
        for m in family; if haskey(family_colors, m); current_color = family_colors[m]; assigned = true; break; end; end
        for model_name in family; if !haskey(family_colors, model_name); family_colors[model_name] = current_color; end; end
        color_index += 1
    end

    models_to_hide = ["gpt-3.5-turbo","gpt-4o","qwen2.5-14B-Instruct", "gemma-2-9b-it", "gemma-2-27b-it" ,"gemini-2.0-flash","mistral-small", "mistral-large","phi-3.5-mini-instruct","llama-2-7b-chat-hf","llama-2-13b-chat-hf","llama-3.3-70B-Instruct","claude-3-5-haiku","claude-3-7-sonnet", "grok-2", "deepseek-v3", "deepseek-r1"] # Add base gpt-4o? Check definition

    plotted_base_models = Dict{String, Any}()
    plotted_interventions = Set{String}()
    plotted_points_data = Dict()

    for (i, full_norm_name) in enumerate(norm_names)
        current_norm_data = norms[i]
        if isnothing(current_norm_data) || length(current_norm_data) < 4 || any(isnothing.(current_norm_data[1:4])) || full_norm_name in models_to_hide
            continue
        end

        index_norms_x, index_norms_y = for_good_norm ? (1, 2) : (3, 4)
        base_color = get(family_colors, full_norm_name, line_colors[(i - 1) % num_colors + 1])

        # Ellipse plotting
        if plot_ellipse && !isnothing(all_norms_std_dev) && haskey(all_norms_std_dev, full_norm_name)
            mean_vec = all_norms_std_dev[norm_names[i]]["mean_vector"]
            cov_mat = all_norms_std_dev[norm_names[i]]["cov_matrix"]
        
            # Extract the relevant 2x2 covariance submatrix
            cov_2d = cov_mat[index_norms_x:index_norms_y, index_norms_x:index_norms_y]
            μ_x, μ_y = mean_vec[index_norms_x], mean_vec[index_norms_y]
        
            # Compute eigenvalues and eigenvectors
            eigvals, eigvecs = eigen(cov_2d)
        
            # Generate ellipse points
            #     θ = range(0, 2π, length=100)
            #     ellipse = [sqrt(eigvals[1]) * cos(t) * eigvecs[:,1] + sqrt(eigvals[2]) * sin(t) * eigvecs[:,2] for t in θ]
            # Eigenvalues define the length of the semi-axes
            scaling_factor = compute_scaling(std_devs)
            a = sqrt(eigvals[2] * scaling_factor)  # Major axis
            b = sqrt(eigvals[1] * scaling_factor)  # Minor axis
        
            # Eigenvectors define the orientation (rotation)
            θ = range(0, 2π, length=100)
            ellipse = [a * cos(t) * eigvecs[:,2] + b * sin(t) * eigvecs[:,1] for t in θ]
        
            # Extract x and y coordinates
            ellipse_x = μ_x .+ [e[1] for e in ellipse]
            ellipse_y = μ_y .+ [e[2] for e in ellipse]
        
            # Plot the ellipse
            lines!(ax, ellipse_x, ellipse_y, color=(base_color, 0.4), linewidth=2, linestyle=:dash)
        end
    end

    # No family lines in this type of plot
    # for family in families
    #     plotted_family_members = [m for m in family if haskey(plotted_points_data, m)]
    #     if length(plotted_family_members) < 2 continue end
    #     family_color = plotted_points_data[plotted_family_members[1]].color # Get consistent color

    #     for j in 1:(length(plotted_family_members) - 1)
    #         model1_name = plotted_family_members[j]
    #         model2_name = plotted_family_members[j+1]
    #         data1 = plotted_points_data[model1_name]
    #         data2 = plotted_points_data[model2_name]
    #         lines!(ax, [data1.x, data2.x], [data1.y, data2.y],
    #                color=(family_color, 0.6), linewidth=2, linestyle=:dash)
    #     end
    # end

    for (i, full_norm_name) in enumerate(norm_names)
        current_norm_data = norms[i]
        if isnothing(current_norm_data) || length(current_norm_data) < 4 || any(isnothing.(current_norm_data[1:4])) || full_norm_name in models_to_hide
            continue
        end

        base_name, suffix = parse_model_name(full_norm_name)
        base_color = get(family_colors, full_norm_name, line_colors[(i - 1) % num_colors + 1])
        marker = get(intervention_markers, suffix, :star8)

        index_norms_x, index_norms_y = for_good_norm ? (1, 2) : (3, 4)
        x_coord = current_norm_data[index_norms_x]
        y_coord = current_norm_data[index_norms_y]

        scatter!(ax, (x_coord, y_coord), color = base_color, markersize = 20,
            strokewidth = 1.5, marker = marker)

        plotted_points_data[full_norm_name] = (x=x_coord, y=y_coord, color=base_color, marker=marker)

        if suffix == ""
            if !haskey(plotted_base_models, base_name)
                plotted_base_models[base_name] = Dict("color" => base_color, "marker" => marker)
            end
            push!(plotted_interventions, suffix)
        else
            push!(plotted_interventions, suffix)
        end
    end

    # Custom Legend to have two legends
    legend_area = fig[1, 2] = GridLayout(tellheight=false) # Assign grid to figure cell

    row_idx = 1

    # Create Legend Elements for base models
    base_elements = []
    base_labels = []
    sorted_base_names = sort(collect(keys(plotted_base_models)))
    if !isempty(sorted_base_names)
        for base_name in sorted_base_names
            info = plotted_base_models[base_name]
            push!(base_elements, MarkerElement(color=info["color"], marker=info["marker"], markersize=15, strokewidth=1))
            push!(base_labels, base_name)
        end
        # Add the models Legend to the grid
        legend_area[row_idx, 1] = Legend(fig, base_elements, base_labels, "Models", tellwidth=false, orientation=:vertical) # Use title argument of Legend
        row_idx += 1
    end

    # Create Legend Elements for Interventions
    int_elements = []
    int_labels = []
    sorted_interventions = sort(collect(plotted_interventions))
    suffix_labels = Dict(
        "" => "Default",
        #"-public" => "Public",
        #"-private" => "Private",
        "-motivation" => "Motivation",
        "-universalization" => "Universalisation",
        "-empathizing" => "Empathising",
        "-signaling" => "Signalling",
        #"-motivation-public" => "Motivation + Public",
        #"-motivation-private" => "Motivation + Private"
    )
    if !isempty(sorted_interventions)
        for suffix in sorted_interventions
            marker = get(intervention_markers, suffix, :star8)
            label = get(suffix_labels, suffix, suffix)
            push!(int_elements, MarkerElement(color=:gray, marker=marker, markersize=15, strokewidth=1))
            push!(int_labels, label)
        end
         # Add the Interventions Legend to the grid
        legend_area[row_idx, 1] = Legend(fig, int_elements, int_labels, "Interventions", tellwidth=false, orientation=:vertical) # Use title argument of Legend
        row_idx += 1
    end

    # Adjust column sizes
    #colsize!(fig.layout, 1, Relative(3/4))
    colsize!(fig.layout, 1, Relative(11/16))
    #colsize!(fig.layout, 2, Relative(1/4))

    trim!(legend_area)

    # Add corner labels
    if (!for_good_norm)
         if (!simplify)
            text!(ax, "SH", fontsize=25, position=(0.02, 0.01), color=:black)
            text!(ax, "IS", fontsize=25, position=(0.93, 0.01), color=:black)
            text!(ax, "SJ", fontsize=25, position=(0.02, 0.94), color=:black)
            text!(ax, "SS", fontsize=25, position=(0.92, 0.94), color=:black)
        else
            # text!(ax, "Shunning", fontsize=22, position=(0.015, 0.07), color=:black)
            # text!(ax, "Image\nScore", fontsize=22, position=(0.85, 0.05), color=:black)
            # text!(ax, "Stern\nJudging", fontsize=22, position=(0.04, 0.85), color=:black)
            # text!(ax, "Simple\nStanding", fontsize=22, position=(0.83, 0.85), color=:black)
            text!(ax, "Shunning", fontsize=22, position=(0.1, 0.125), color=:black, align=(:center, :top))
            text!(ax, "Image\nScore", fontsize=22, position=(0.9, 0.13), color=:black, align=(:center, :top))
            text!(ax, "Stern\nJudging", fontsize=22, position=(0.1, 0.95), color=:black, align=(:center, :top))
            text!(ax, "Simple\nStanding", fontsize=22, position=(0.9, 0.95), color=:black, align=(:center, :top))
        end
    end

    save(joinpath(plot_path, filename * ".pdf"), fig)

    if !special_labels
        display(fig)
    end
end

function plot_error_heatmap(json_path::String, selected_models::Vector{String}, folderpath::String)
    # Receives the path to the error json file, plots the figure for the error rate of each model parsing each norm
    # Load and parse the JSON data
    data = JSON.parsefile(json_path)

    norm_keys = ["norm", "norm_M_to_M", "norm_M_to_F", "norm_F_to_M", "norm_F_to_F", "norm_WEST_to_WEST", "norm_WEST_to_EASTASIA", "norm_WEST_to_SUBSAHARA", "norm_WEST_to_MENA", "norm_EASTASIA_to_WEST", "norm_EASTASIA_to_EASTASIA", "norm_EASTASIA_to_SUBSAHARA", "norm_EASTASIA_to_MENA", "norm_SUBSAHARA_to_WEST", "norm_SUBSAHARA_to_EASTASIA", "norm_SUBSAHARA_to_SUBSAHARA", "norm_SUBSAHARA_to_MENA", "norm_MENA_to_WEST", "norm_MENA_to_EASTASIA", "norm_MENA_to_SUBSAHARA", "norm_MENA_to_MENA", "norm_tag_no-topic", "norm_tag_neutral", "norm_tag_non-neutral", "norm_tag_explicit-neutral", "norm_tag_explicit-non-neutral"]
    norm_keys_names = ["Full dataset", "Male donor, male recipient", "Male donor, female recipient", "Female donor, male recipient", "Female donor, female recipient", "Western donor, western recipient", "Western donor, East Asian recipient", "Western donor, Sub-Saharan recipient", "Western donor, MENA recipient", "East Asian donor, Western recipient", "East Asian donor, East Asian recipient", "East Asian donor, Sub-Saharan recipient", "East Asian donor, MENA recipient", "Sub-Saharan donor, Western recipient", "Sub-Saharan donor, East Asian recipient", "Sub-Saharan donor, Sub-Saharan recipient", "Sub-Saharan donor, MENA recipient", "MENA donor, Western recipient", "MENA donor, East Asian recipient", "MENA donor, Sub-Saharan recipient", "MENA donor, MENA recipient", "No topic", "Neutral and non-explicit topic", "Non-neutral and non-explicit topic", "Neutral and explicit topic", "Non-neutral and explicit topic"]

    # Build matrix of values [model_index, norm_index]
    norm_matrix = zeros(length(selected_models), length(norm_keys))

    for (i, model_name) in enumerate(selected_models)
        entry = only(filter(d -> d["model_name"] == model_name, data))
        if isnothing(entry)
            error("Model $(model_name) not found in data.")
        end
        for (j, key) in enumerate(norm_keys)
            norm_matrix[i, j] = entry[key]
        end
    end
    #norm_matrix = norm_matrix[end:-1:1, :] # we want the first model in the top

    # Create the heatmap
    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1],
        xticks=(1:length(norm_keys_names), replace.(norm_keys_names, "norm_" => "")),
        yticks=(1:length(selected_models), selected_models),
        xlabel="Norm",
        ylabel="Model", xticklabelrotation = 45
    )
    heatmap!(ax, transpose(norm_matrix); colormap=:viridis, colorrange=(0, 1))
    Colorbar(fig[1, 2], label="Parsing Error Rate", ticks=[0,0.25,0.5,0.75,1.0])

    display(fig)
    isdir(folderpath) || mkdir(folderpath)
    save(joinpath(folderpath,"parse_error_analysis.pdf"), fig)
end

function plot_bias_analysis(folder_path::String, A_norm_names, A_norms)
    groups_of_keys = [["norm", "norm_M_to_M", "norm_M_to_F", "norm_F_to_M", "norm_F_to_F"], 
                    ["norm", "norm_WEST_to_WEST", "norm_WEST_to_EASTASIA", "norm_WEST_to_SUBSAHARA", "norm_WEST_to_MENA", "norm_EASTASIA_to_WEST", "norm_EASTASIA_to_EASTASIA", "norm_EASTASIA_to_SUBSAHARA", "norm_EASTASIA_to_MENA", "norm_SUBSAHARA_to_WEST", "norm_SUBSAHARA_to_EASTASIA", "norm_SUBSAHARA_to_SUBSAHARA", "norm_SUBSAHARA_to_MENA", "norm_MENA_to_WEST", "norm_MENA_to_EASTASIA", "norm_MENA_to_SUBSAHARA", "norm_MENA_to_MENA"],
                    ["norm", "norm_tag_no-topic", "norm_tag_neutral", "norm_tag_non-neutral", "norm_tag_explicit-neutral", "norm_tag_explicit-non-neutral"]]
    groups_norm_keys_names = [["Male donor, male recipient", "Male donor, female recipient", "Female donor, male recipient", "Female donor, female recipient"],
                                ["Western donor, western recipient", "Western donor, East Asian recipient", "Western donor, Sub-Saharan recipient", "Western donor, MENA recipient", "East Asian donor, Western recipient", "East Asian donor, East Asian recipient", "East Asian donor, Sub-Saharan recipient", "East Asian donor, MENA recipient", "Sub-Saharan donor, Western recipient", "Sub-Saharan donor, East Asian recipient", "Sub-Saharan donor, Sub-Saharan recipient", "Sub-Saharan donor, MENA recipient", "MENA donor, Western recipient", "MENA donor, East Asian recipient", "MENA donor, Sub-Saharan recipient", "MENA donor, MENA recipient"], 
                                ["No topic", "Neutral and non-explicit topic", "Non-neutral and non-explicit topic", "Neutral and explicit topic", "Non-neutral and explicit topic"]]


    for (norm_keys, norm_keys_names, norm_type) in zip(groups_of_keys,groups_norm_keys_names, ["gender", "region", "tag"])
        A_norms_average = Dict{String, Vector{Float64}}()
        for key in norm_keys
            A_norms_average[key] = mean([A_norms[name][key] for name in A_norm_names], dims=1)[:][1]
        end
        
        # compute deltas with respect to "norm"
        base = A_norms_average["norm"]
        delta_norms = Dict{String, Vector{Float64}}()
        for key in norm_keys
            if key != "norm"
                delta_norms[key] = A_norms_average[key] .- base
            end
        end

        marker_types = [:circle, :rect, :utriangle, :dtriangle, :diamond, :cross, :xcross, :star5, :pentagon, :hexagon]
        marker_iter = Iterators.Stateful(Iterators.cycle(marker_types))
        colors_iter = Iterators.Stateful(Iterators.cycle(line_colors))

        fig = Figure(size = (1200, 500)) 
        ticks = -1.0:0.1:1.01
        xlabel_good, ylabel_good = (L"\text{Δ Prob. assign good after cooperating with G}, d_{GC}", L"\text{Δ Prob. assign good after defecting with G}, d_{GD}")
        xlabel_bad, ylabel_bad = (L"\text{Δ Prob. assign good after cooperating with B}, d_{BC}", L"\text{Δ Prob. assign good after defecting with B}, d_{BD}")

        ax1 = Axis(fig[1, 1]; title="Observing good recipient", xlabel=xlabel_good, ylabel=ylabel_good, xticks=ticks, yticks=ticks)
        ax2 = Axis(fig[1, 2]; title="Observing bad recipient", xlabel=xlabel_bad, ylabel=ylabel_bad, xticks=ticks, yticks=ticks)
        #lim = (-0.4, 0.4)
        max_dev = ceil(maximum(abs.(reduce(vcat, values(delta_norms)))), digits=1)
        margin = 0.01
        lim = (-max_dev - margin, max_dev + margin)
        xlims!(ax1, lim); ylims!(ax1, lim)
        xlims!(ax2, lim); ylims!(ax2, lim)

        scatter_elements = []

        for key_i in eachindex(norm_keys)
            key = norm_keys[key_i]
            if key != "norm"
                delta = delta_norms[key]
                marker = popfirst!(marker_iter)
                cur_color = popfirst!(colors_iter)
                push!(scatter_elements, (MarkerElement(marker=marker, color=cur_color), key))
                
                scatter!(ax1, [delta[1]], [delta[2]], marker=marker, color=cur_color, markersize=20, strokewidth=1.5, label=norm_keys_names[key_i-1])
                scatter!(ax2, [delta[3]], [delta[4]], marker=marker, color=cur_color, markersize=20, strokewidth=1.5, label=norm_keys_names[key_i-1])
            end
        end

        Legend(fig[1, 3], ax1, "Prompt type")

        display(fig)
        isdir(folder_path) || mkdir(folder_path)
        save(joinpath(folder_path, "bias_analysis_$norm_type.pdf"), fig)
    end
end

# Space to run specific plots not dependent on stats.jl
#plot_error_heatmap("./prompt_dataset/all_llm_errors.json", ["gpt-3.5-turbo","gpt-4o", "qwen2.5-7B-Instruct","qwen2.5-14B-Instruct", "gemma-2-9b-it", "gemma-2-27b-it" ,"gemini-1.5-pro","gemini-2.0-flash","mistral-small", "mistral-large","phi-3.5-mini-instruct","phi-4","llama-2-7b-chat-hf","llama-2-13b-chat-hf", "llama-3.1-8B-Instruct","llama-3.3-70B-Instruct","claude-3-5-haiku","claude-3-7-sonnet", "grok-2", "deepseek-v3", "deepseek-r1"], "./results_cooperation/parse_error_analysis")