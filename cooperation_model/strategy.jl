using LinearAlgebra
using Memoization
using SparseArrays

include("./reputation.jl")
include("./utils.jl")

# Average donations of each strategy

function allc_donates(execError::Float64)::Float64
    return (1.0 - execError)
end

function alld_donates()::Float64
    return 0.0
end

function disc_donates(fAllC::Float64, fAllD::Float64, fDisc::Float64, execError::Float64, snI::Vector, popsize::Int, gossipRounds::Int)::Float64
    rAllC_H, rAllD_H, rDisc_H = avg_rep(fAllC, fAllD, fDisc, snI, popsize, gossipRounds)

    r = fAllC * rAllC_H + fAllD * rAllD_H + fDisc * rDisc_H      # Disc donates when it interacts with a good human
    return (1.0 - execError) * r
end

# Average receivings of each strategy

function human_receives(rep_H::Float64, fAllC::Float64, fDisc::Float64, execError::Float64)::Float64
    # a human receives a donation if there are no errors and it interacts with a human and it gets donated by all AllC + all discs that consider it good
    return (1.0 - execError) * (fAllC + fDisc * rep_H)
end

function allc_receives(fAllC::Float64, fAllD::Float64, fDisc::Float64, execError::Float64, snI::Vector, popsize::Int, gossipRounds::Int)::Float64
    rAllC_H, _, _ = avg_rep(fAllC, fAllD, fDisc, snI, popsize, gossipRounds)

    return human_receives(rAllC_H, fAllC, fDisc, execError)
end

function alld_receives(fAllC::Float64, fAllD::Float64, fDisc::Float64, execError::Float64, snI::Vector, popsize::Int, gossipRounds::Int)::Float64
    _, rAllD, _ = avg_rep(fAllC, fAllD, fDisc, snI, popsize, gossipRounds)

    return human_receives(rAllD, fAllC, fDisc, execError)
end

function disc_receives(fAllC::Float64, fAllD::Float64, fDisc::Float64, execError::Float64, snI::Vector, popsize::Int, gossipRounds::Int)::Float64
    _, _, rDisc = avg_rep(fAllC, fAllD, fDisc, snI, popsize, gossipRounds)

    return human_receives(rDisc, fAllC, fDisc, execError)
end

# Average fitness functions
# These are slightly different from the original paper due to the lack of payoff received from execution errors

function fit_allc(fAllC::Float64, fAllD::Float64, fDisc::Float64, execError::Float64, b::Float64, snI::Vector, popsize::Int, gossipRounds::Int)::Float64
    
    return b * allc_receives(fAllC, fAllD, fDisc, execError, snI, popsize, gossipRounds) - 1.0 * allc_donates(execError)   # where c = 1, thus -1
end

function fit_alld(fAllC::Float64, fAllD::Float64, fDisc::Float64, execError::Float64, b::Float64, snI::Vector, popsize::Int, gossipRounds::Int)::Float64
    
    return b * alld_receives(fAllC, fAllD, fDisc, execError, snI, popsize, gossipRounds) - 1.0 * alld_donates()   # where c = 1, thus -1
end

function fit_disc(fAllC::Float64, fAllD::Float64, fDisc::Float64, execError::Float64, b::Float64, snI::Vector, popsize::Int, gossipRounds::Int)::Float64
    
    return b * disc_receives(fAllC, fAllD, fDisc, execError, snI, popsize, gossipRounds) - 1.0 * disc_donates(fAllC, fAllD, fDisc, execError, snI, popsize, gossipRounds)   # where c = 1, thus -1
end

function p_imit(fImitator::Float64, fRoleModel::Float64, sos::Float64=1.0)::Float64
    # Fermi update function, receives fitness of imitator and role model, and strength of selection and returns prob of imitating
    return (1 + exp(-sos * (fRoleModel - fImitator))) ^ -1
end

function all_fitness(n_allC::Int, n_allD::Int, snI::Vector, popsize::Int, execError::Float64, b::Float64, gossipRounds::Int)::Tuple
    # Calculates all the fitness of the given state
    f_allc = fit_allc(n_allC/popsize, n_allD/popsize, (popsize-n_allC-n_allD)/popsize, execError, b, snI, popsize, gossipRounds)
    f_alld = fit_alld(n_allC/popsize, n_allD/popsize, (popsize-n_allC-n_allD)/popsize, execError, b, snI, popsize, gossipRounds)
    f_disc = fit_disc(n_allC/popsize, n_allD/popsize, (popsize-n_allC-n_allD)/popsize, execError, b, snI, popsize, gossipRounds)
    return (f_allc, f_alld, f_disc)
end

@memoize function transition_prob(fit_imitator::Float64, n_imitator::Int, fit_newstrat::Float64, n_newstrat::Int, mutChance::Float64, popsize::Int)::Float64
    return (1 - mutChance) * (n_imitator / popsize) * (n_newstrat / (popsize - 1)) * p_imit(fit_imitator, fit_newstrat) + mutChance * n_imitator / (2 * popsize)
end

@memoize function get_states_strat(popsize::Int)::Vector{Tuple{Int, Int, Int}}
    return [(nAllC, nAllD, popsize - nAllC - nAllD) for nAllC in 0:popsize for nAllD in 0:(popsize - nAllC) if nAllC + nAllD <= popsize]
end

@memoize function stationary_dist_strategy(snI::Vector, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int)::Vector{Float64}
    states::Vector{Tuple{Int, Int, Int}} = get_states_strat(popsize)
    lookup = utils.create_lookup_table(states)

    transition_matrix = spzeros(length(states),length(states))

    currentPos = 0
    t1, t2, t3, t4, t5, t6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for i in 0:popsize
        for j in 0:(popsize - i)
            k = popsize - j - i
            t1, t2, t3, t4, t5, t6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            if k < 0
                continue
            end

            currentPos = utils.pos_of_state(lookup, (i, j, k))

            f_allc, f_alld, f_disc, = all_fitness(i, j, snI, popsize, execError, b, gossipRounds)

            if i < popsize && j > 0
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i+1, j-1, k))] = t1 = transition_prob(f_alld, j, f_allc, i, mutChance, popsize)
            end
            if i < popsize && k > 0
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i+1, j, k-1))] = t2 = transition_prob(f_disc, k, f_allc, i, mutChance, popsize)
            end
            if i > 0 && j < popsize
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i-1, j+1, k))] = t3 = transition_prob(f_allc, i, f_alld, j, mutChance, popsize)
            end
            if i > 0 && k < popsize
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i-1, j, k+1))] = t4 = transition_prob(f_allc, i, f_disc, k, mutChance, popsize)
            end
            if k > 0 && j < popsize
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i, j+1, k-1))] = t5 = transition_prob(f_disc, k, f_alld, j, mutChance, popsize)
            end
            if j > 0 && k < popsize
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i, j-1, k+1))] = t6 = transition_prob(f_alld, j, f_disc, k, mutChance, popsize)
            end
            transition_matrix[currentPos,currentPos] = 1 - t1 - t2 - t3 - t4 - t5 - t6
        end
    end

    result = utils.get_transition_matrix_statdist(transition_matrix)

    return result
end

function stat_dist_at_point_strat(n_allC::Int, n_allD::Int, snI::Vector, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int)::Float64
    states = get_states_strat(popsize)

    if (n_allC, n_allD, popsize-n_allC-n_allD) ∉ states return 0 end

    stat_dist = stationary_dist_strategy(snI, popsize, execError, b, mutChance, gossipRounds)
    pos = utils.find_pos_of_state(states, (n_allC, n_allD, popsize-n_allC-n_allD))
    result = stat_dist[pos]

    return result
end

function gradient_of_selection_at_state(allC::Int, allD::Int, disc::Int, snI::Vector, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int)::Tuple{Float64,Float64,Float64}
    # For a given state, calculates the gradient of selection at that state
    grad = [0.0, 0.0, 0.0]

    f_allc, f_alld, f_disc = all_fitness(allC, allD, snI,popsize, execError, b, gossipRounds)
    # allC = prob(+AllC) - prob(-AllC)
    grad[1] = (transition_prob(f_alld, allD, f_allc, allC, mutChance, popsize)) -
            (transition_prob(f_allc, allC, f_alld, allD, mutChance, popsize))

    grad[2] = (transition_prob(f_allc, allC, f_alld, allD, mutChance, popsize)) -
            (transition_prob(f_alld, allD, f_allc, allC, mutChance, popsize))

    grad[3] = (transition_prob(f_allc, allC, f_disc, disc, mutChance, popsize)) -
            (transition_prob(f_disc, disc, f_allc, allC, mutChance, popsize))

    return (grad[1], grad[2], grad[3])
end

function gradient_of_selection(snI::Vector, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int)::Vector
    # returns the gradient of selection at each state
    states = get_states_strat(popsize)
    all_grads = []
    for state in states
        grad = gradient_of_selection_at_state(state[1], state[2], state[3], snI,popsize, execError, b, mutChance, gossipRounds)
        push!(all_grads, grad)
    end
    return all_grads
end

function coop_at_state(allC::Int, allD::Int, disc::Int, snI::Vector, popsize::Int, execError::Float64, gossipRounds::Int)::Float64
    # returns cooperation index in a given state
    fAllC, fAllD, fDisc = allC/popsize, allD/popsize, disc/popsize

    allcDonates = allc_donates(execError)
    alldDonates = alld_donates() 
    discDonates = disc_donates(fAllC, fAllD, fDisc,execError, snI, popsize, gossipRounds)

    return allcDonates * fAllC + alldDonates * fAllD + discDonates * fDisc
end

function coop_index(snI::Vector, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int)::Tuple
    # returns the cooperation index averaged by statDist, plus all the cooperation indexes at each state
    states = get_states_strat(popsize)
    coop_index_aggr = 0.0
    all_coop_index::Vector{Float64} = []
    stationaryDist = 0.0
    coop_index_at_state = 0.0

    for state in states
        stationaryDist = stat_dist_at_point_strat(state[1], state[2], snI,popsize, execError, b, mutChance, gossipRounds)
        coop_index_at_state = coop_at_state(state[1], state[2], state[3], snI,popsize, execError, gossipRounds)
        push!(all_coop_index, coop_index_at_state)
        coop_index_aggr += stationaryDist * coop_index_at_state
    end
    return (coop_index_aggr, all_coop_index)
end

function avg_reputations(snI::Vector, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int)::Tuple
    # returns the 3 resulting reputations averaged by statDist, plus all the reputations at each state
    states = get_states_strat(popsize)
    avg_r = [Float64(0),Float64(0),Float64(0)]
    all_r = []
    stationaryDist = 0.0

    for state in states
        stationaryDist = stat_dist_at_point_strat(state[1], state[2], snI,popsize, execError, b, mutChance, gossipRounds)
        rep = avg_rep(state[1]/popsize, state[2]/popsize, state[3]/popsize,snI, popsize, gossipRounds)
        push!(all_r, rep)
        avg_r .+= stationaryDist .* rep
    end
    return (avg_r, all_r)
end

function avg_agreement(snI::Vector, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int)::Tuple
    # returns the agreement and disagreement averaged by statDist, plus all the agreement and disagreements at each state
    states = get_states_strat(popsize)
    avg_ag = [Float64(0),Float64(0)]
    all_ag = []
    stationaryDist = 0.0

    for state in states
        stationaryDist = stat_dist_at_point_strat(state[1], state[2], snI,popsize, execError, b, mutChance, gossipRounds)
        ag = agreement_and_disagreement(state[1]/popsize, state[2]/popsize, state[3]/popsize,snI, popsize, gossipRounds)
        push!(all_ag, ag)
        avg_ag .+= stationaryDist .* ag
    end
    return (avg_ag, all_ag)
end

function avg_state(snI::Vector, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int)::Tuple{Float64,Float64,Float64}
    # returns the average state of the system
    states = get_states_strat(popsize)
    avg_s = [Float64(0),Float64(0),Float64(0)]
    stationaryDist = 0.0

    for state in states
        stationaryDist = stat_dist_at_point_strat(state[1], state[2], snI,popsize, execError, b, mutChance, gossipRounds)
        avg_s .+= stationaryDist .* state
    end
    return (avg_s[1], avg_s[2], avg_s[3])
end

function get_all_data(snI::Vector, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int)::Tuple
    # Returns all possible data from the model. coopIndex, reputations, statDist, avgState, agreement, gradOfSel.
    coop = coop_index(snI,popsize, execError, b, mutChance, gossipRounds)   # tuple (average coop, [coop at each markov chain state])
    rep = avg_reputations(snI,popsize, execError, b, mutChance, gossipRounds) # tuple ([average rep ALLC, average rep ALLD, average rep DISC], [coop at each markov chain state])
    statDist = stationary_dist_strategy(snI,popsize, execError, b, mutChance, gossipRounds)
    avgState = avg_state(snI,popsize, execError, b, mutChance, gossipRounds)
    agreement = avg_agreement(snI,popsize, execError, b, mutChance, gossipRounds)
    grad = gradient_of_selection(snI,popsize, execError, b, mutChance, gossipRounds)
    return (coop, rep, statDist, avgState, agreement, grad)
end

# --------------------------
# Run results
# --------------------------

function vary_variable(params::Vector, parameterVaried::String, indexParamToVary::Int, rangeOfValues::Vector, tag::String, foldername::String="", addErrorToNorm::Bool=true, skipExistingData::Bool=false) 
    # Given a foldername, a list of parameters, and the parameter to vary within a given interval, calculates all data and stores it

    folder_path = utils.make_plot_folder(foldername)
    
    if skipExistingData && isfile(joinpath(folder_path, "parameters.txt")) && isfile(joinpath(folder_path, "Results", "ResultsBackup", tag * "_results.jls")) 
        println("Results already exist. Skipping")
        return 
    end

    snH::Vector{Float64}, snA::Vector{Float64}, AAinfluence::Float64, normInfluenceFunction::Function, normInfluenceFunctionName::String, popsize::Int, execError::Float64, assessError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int, strengthOfSelection::Float64 = params

    
    utils.write_parameters(folder_path, snH, snA, AAinfluence, normInfluenceFunctionName, popsize, execError, assessError, b, mutChance, gossipRounds, strengthOfSelection,parameterVaried, rangeOfValues)
    
    # Create a list with all results
    array_of_any = [nothing for _ in 1:length(rangeOfValues)]
    allResults = Vector{Any}(array_of_any)
    
    for i in 1:length(rangeOfValues)
        
        # change our parameter acording to range provided
        params[indexParamToVary] = rangeOfValues[i]
        snH, snA, AAinfluence, normInfluenceFunction, normInfluenceFunctionName,popsize, execError, assessError, b, mutChance, gossipRounds, strenghtOfSelection = params

        snI = normInfluenceFunction(snH, snA, AAinfluence)
        snI = add_errors_sn(snI, execError, assessError)

        #println("Running simulations for $tag: $parameterVaried = ",  rangeOfValues[i], ". ", i,"/",length(rangeOfValues))
        allResults[i] = get_all_data(snI, popsize, execError, b, mutChance, gossipRounds)
    
        Memoization.empty_all_caches!()
    end

    utils.write_all_results(folder_path, tag, allResults)
    println("Simulation done! Saving results.")


    println("Results saved.")
end