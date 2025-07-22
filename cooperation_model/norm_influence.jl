# Space to declare different functions of the influence AAs have on human norms

function linear(snH::Vector{Float64}, snA::Vector{Float64}, AAinfluence::Float64)::Vector{Float64}
    # Performs linear interpolation
    return (1 - AAinfluence) .* snH .+ AAinfluence .* snA
end