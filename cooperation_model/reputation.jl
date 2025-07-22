using LinearAlgebra
using NonlinearSolve 
using Memoization

# Add errors (assign and execute) to social norm
function add_errors_sn(sn::Vector, execError::Float64, assessmentError::Float64)::Vector
    newsn = [0.0,0.0,0.0,0.0]

    epsi = (1 - execError) * (1 - assessmentError) + execError * assessmentError

    newsn[1] = sn[1] * (epsi - assessmentError) + sn[2] * (1 - epsi - assessmentError) + assessmentError
    newsn[2] = sn[2] * (1 - 2 * assessmentError) + assessmentError
    newsn[3] = sn[3] * (epsi - assessmentError) + sn[4] * (1 - epsi - assessmentError) + assessmentError
    newsn[4] = sn[4] * (1 - 2 * assessmentError) + assessmentError

    return newsn
end

# ODEs to determine the average reputation of each strategy
# We consider a single social norm: the human's influenced norm (after influence of the AA), snI. 
# The norm the human uses, snH, and the AA uses, snA, are mixed to create snI, and therefore not directly relevant for reputation dynamics.
# We thus have an 1 extra parameters: 
# AAinfluence, sigma (AA's norm influence on norms), where snI = (1-sigma) * snH + sigma * snA
# TODO: AAfrequency, alpha (frequency of AA's influece), how often people will just use snH, instead of snM (simulating not interacting with the AA before an interaction)

function avg_rep_ode!(dr, r, p)
    fAllC, fAllD, fDisc, snI, popsize, gossipRounds = p

    # The average reputation in an interaction for humans with humans
    rall = fAllC * r[1] + fAllD * r[2] + fDisc * r[3]

    # Pre gossip reputation alignment metrics for humans interacting with humans
    # These now have to account for the fraction of time you interact with the machine too
    g2init = fAllC * r[1]^2             + fAllD * r[2]^2            + fDisc * r[3]^2   # Fraction of agreement that a focal individual is good
    b2init = fAllC * (1-r[1])^2         + fAllD * (1-r[2])^2        + fDisc * (1-r[3])^2   # Fraction of agreement that a focal individual is bad
    d2init = fAllC * r[1] * (1-r[1])    + fAllD * r[2] * (1-r[2])   + fDisc * r[3] * (1-r[3])   # Fraction of disagreement about a focal individual

    # Post gossip reputation alignment metrics for humans interacting with humans-> using peer-to-peer
    T = gossipRounds / popsize
    g2 = g2init + d2init * (1 - ℯ^(-T))
    b2 = b2init + d2init * (1 - ℯ^(-T))
    d2 = d2init * ℯ^(-T)

    # ODE problem for reputations of humans by humans
    # These now have to account for the fraction of time you interact with the machine too
    dr[1] = (rall * snI[1] + (1 - rall) * snI[3]) - r[1]

    dr[2] = (rall * snI[2] + (1 - rall) * snI[4]) - r[2]

    dr[3] = (g2 * snI[1] + d2 * (snI[3] + snI[2]) + b2 * snI[4]) - r[3]
end

@memoize function avg_rep(fAllC::Float64, fAllD::Float64, fDisc::Float64, snI::Vector, popsize::Int, gossipRounds::Int)::Tuple{Float64,Float64,Float64}
    average_rep_strat0 = [0.5,0.5,0.5]      # initial reputation state (rAllC, rAllD, rDisc)

    nl_prob = NonlinearProblem(avg_rep_ode!, average_rep_strat0, (fAllC, fAllD, fDisc, snI, popsize, gossipRounds))
    sol = solve(nl_prob,NewtonRaphson())

    return Tuple(clamp(x, 0, 1) for x in sol.u)
end

# Agreement level of private reputations

function agreement(fAllC::Float64, fAllD::Float64, fDisc::Float64, snI::Vector, popsize::Int, gossipRounds::Int)::Float64
    rAllC, rAllD, rDisc = avg_rep(fAllC, fAllD, fDisc, snI, popsize, gossipRounds)

    return fAllC * rAllC^2 + fAllD * rAllD^2 + fDisc * rDisc^2 + fAllC * (1 - rAllC)^2 + fAllD * (1 - rAllD)^2 + fDisc * (1 - rDisc)^2
end

function disagreement(fAllC::Float64, fAllD::Float64, fDisc::Float64, snI::Vector, popsize::Int, gossipRounds::Int)::Float64
    rAllC, rAllD, rDisc = avg_rep(fAllC, fAllD, fDisc, snI, popsize, gossipRounds)

    return 2 * ( fAllC * rAllC * (1-rAllC) + fAllD * rAllD * (1-rAllD)   + fDisc * rDisc * (1-rDisc))
end

function agreement_and_disagreement(fAllC::Float64, fAllD::Float64, fDisc::Float64, snI::Vector, popsize::Int, gossipRounds::Int)::Tuple{Float64, Float64}
    #returns all the agreements and disagreements
    return (agreement(fAllC, fAllD, fDisc, snI, popsize, gossipRounds),disagreement(fAllC, fAllD, fDisc, snI, popsize, gossipRounds))
end