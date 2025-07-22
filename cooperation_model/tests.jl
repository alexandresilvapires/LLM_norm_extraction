include("./reputation.jl")
include("./strategy.jl")
include("./norm_influence.jl")
using Test


tolerance = 1e-4
snIS = [1.0, 0.0, 1.0, 0.0] 
snSJ = [1.0, 0.0, 0.0, 1.0] 
snSS = [1.0, 0.0, 1.0, 1.0] 
snSH = [1.0, 0.0, 0.0, 0.0] 
snAG = [1.0, 1.0, 1.0, 1.0]
snAB = [0.0, 0.0, 0.0, 0.0]

@testset "Reputation Tests" begin

    begin # If everyone is AllC, in IS, everyone is good
        fallC, fallD, fDisc = 1.0, 0.0, 0.0
        sn = snIS
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 0
        result = avg_rep(fallC, fallD, fDisc, sn, pop, gossip)[1]
        @test isapprox(result, 1.0, atol=tolerance)
    end

    begin # If everyone is AllD, in IS, everyone is bad
        fallC, fallD, fDisc = 0.0, 1.0, 0.0
        sn = snIS
        pop = 50
        gossip = 0

        result = avg_rep(fallC, fallD, fDisc, sn, pop, gossip)[2]
        @test isapprox(result, 0.0, atol=tolerance) 
    end

    begin # In a mixed population, in IS, every AllC is good and every AllD is bad
        fallC, fallD, fDisc = 0.2, 0.2, 0.6
        sn = snIS
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 0

        result = avg_rep(fallC, fallD, fDisc, sn, pop, gossip)[1]
        @test isapprox(result, 1.0, atol=tolerance)

        result = avg_rep(fallC, fallD, fDisc, sn, pop, gossip)[2]
        @test isapprox(result, 0.0, atol=tolerance)
    end

    begin # In a Disc population, in SJ with errors, reputations without gossip should be around 0.5
        fallC, fallD, fDisc = 0.0, 0.0, 1.0
        sn = add_errors_sn(snSJ, 0.02, 0.02)
        pop = 50
        gossip = 0

        result = avg_rep(fallC, fallD, fDisc, sn, pop, gossip)[3]

        @test 0.4 < result < 0.6
    end

    begin # In a Disc population, in SJ with errors, reputations with some gossip should follow the values from the article
        fallC, fallD, fDisc = 0.0, 0.0, 1.0
        sn = add_errors_sn(snSJ, 0.02, 0.02)
        pop = 50
        gossip = 20

        result = avg_rep(fallC, fallD, fDisc, sn, pop, gossip)[3]

        @test 0.65 < result < 0.75
    end

    begin # In a Disc population, in SJ with errors, agreement without gossip should be around 0.5
        fallC, fallD, fDisc = 0.0, 0.0, 1.0
        sn = add_errors_sn(snSJ, 0.02, 0.02)
        pop = 50
        gossip = 0

        result = agreement(fallC, fallD, fDisc, sn, pop, gossip)

        @test 0.4 < result < 0.6
    end

    begin # In mixed population, agreement + disagreement = 1
        fallC, fallD, fDisc = 0.25, 0.35, 0.4
        sn = add_errors_sn(snSJ, 0.02, 0.02)
        pop = 50
        gossip = 2

        result1 = agreement(fallC, fallD, fDisc, sn, pop, gossip)
        result2 = disagreement(fallC, fallD, fDisc, sn, pop, gossip)

        @test isapprox(result1 + result2, 1.0, atol=tolerance) 
    end

    begin # In a Disc population, in SJ with errors, if everyone gossips a lot, the agreement is very high
        fallC, fallD, fDisc = 0.0, 0.0, 1.0
        sn = add_errors_sn(snSJ, 0.01, 0.01)
        pop = 50
        gossip = 5000000

        result = agreement(fallC, fallD, fDisc, sn, pop, gossip)
        @test result > 0.95
    end

    begin # If everyone is AllD, in IS, and the assessment and execution error is 0.5, half should be good
        fallC, fallD, fDisc = 0.0, 1.0, 0.0
        sn = add_errors_sn(snIS, 0.5, 0.5)
        pop = 50
        gossip = 0

        result = avg_rep(fallC, fallD, fDisc, sn, pop, gossip)[2]
        @test isapprox(result, 0.5, atol=tolerance) 
    end

    begin # No matter the population, in SJ, and the assessment and execution error is 0.5, half should be good
        fallC, fallD, fDisc = 0.3, 0.2, 0.5
        sn = add_errors_sn(snSJ, 0.5, 0.5)
        pop = 50
        gossip = 0

        result = avg_rep(fallC, fallD, fDisc, sn, pop, gossip)[1]
        @test isapprox(result, 0.5, atol=tolerance) 

        result = avg_rep(fallC, fallD, fDisc, sn, pop, gossip)[2]
        @test isapprox(result, 0.5, atol=tolerance) 

        result = avg_rep(fallC, fallD, fDisc, sn, pop, gossip)[3]
        @test isapprox(result, 0.5, atol=tolerance) 
    end

end

@testset "Payoff Tests" begin

    begin # If everyone is AllC, in IS, the average payoff of AllC should be b - c
        fallC, fallD, fDisc = 1.0, 0.0, 0.0
        sn = snIS
        pop = 50
        gossip = 0
        execError = 0.0
        b = 5.0
        result = fit_allc(fallC, fallD, fDisc, execError, b, sn, pop, gossip)
        @test isapprox(result, b - 1, atol=tolerance)
    end

    begin # If everyone is AllD, in IS, the average payoff of AllD should be 0
        fallC, fallD, fDisc = 0.0, 1.0, 0.0
        sn = snIS
        pop = 50
        gossip = 0
        execError = 0.0
        b = 5.0

        result = fit_alld(fallC, fallD, fDisc, execError, b, sn, pop, gossip)
        @test isapprox(result, 0.0, atol=tolerance)
    end

    begin # If everyone is Disc, in All Good, the average payoff is b - c
        fallC, fallD, fDisc = 0.0, 0.0, 1.0
        sn = snAG
        pop = 50
        gossip = 0
        execError = 0.0
        b = 5.0

        result = fit_disc(fallC, fallD, fDisc, execError, b, sn, pop, gossip)
        @test isapprox(result, b - 1 , atol=tolerance)
    end

end

@testset "Evolutionary Tests" begin

    begin # In SJ, without gossip, the cooperation index should be very low
        sn = snSJ
        pop = 30
        gossip = 0
        execError = 0.00
        b = 5.0

        result = coop_index(sn, pop, execError, b, 0.01, gossip)[1][1]
        @test result < 0.05
    end

    begin # In SJ, without gossip, the average state should be of full defection
        sn = snSJ
        pop = 30
        gossip = 0
        execError = 0.00
        b = 5.0

        result = avg_state(sn, pop, execError, b, 0.01, gossip)
        @test result[2] > pop * 0.95
    end

    begin # In SJ, with gossip, the average state should no longer be full of defectors
        sn = snSJ
        pop = 30
        gossip = 30
        execError = 0.00
        b = 5.0

        result = avg_state(sn, pop, execError, b, 0.01, gossip)
        @test result[2] < 0.6 * pop
    end

end


@testset "Norm Interpolation Tests" begin
    # We can linearly interpolate between norms, and the results should be as expected. 

    begin # Full transition from the inverse of SJ to SJ should yield the results of SJ
        sn_H = [0.0, 1.0, 1.0, 0.0]
        sn_A = snSJ
        pop = 30
        gossip = 0
        execError = 0.00
        b = 5.0

        influence = 1.0

        sn = linear(sn_H, sn_A, influence)

        result = coop_index(sn, pop, execError, b, 0.01, gossip)[1][1]
        @test result < 0.05
    end

    begin # Midway transition from the inverse of SJ to SJ should yield [0.5, 0.5, 0.5, 0.5]
        sn_H = [0.0, 1.0, 1.0, 0.0]
        sn_A = snSJ
        pop = 30
        gossip = 0
        execError = 0.00
        b = 5.0

        influence = 0.5

        sn = linear(sn_H, sn_A, influence)

        @test isapprox([0.5, 0.5, 0.5, 0.5], sn)
    end

    begin # Full transition from AG to AB should yield AB
        sn_H = [1.0, 1.0, 1.0, 1.0]
        sn_A = [0.0, 0.0, 0.0, 0.0]
        pop = 30
        gossip = 0
        execError = 0.00
        b = 5.0

        influence = 1.0

        sn = linear(sn_H, sn_A, influence)

        @test isapprox(sn_A, sn)
    end

    begin # 3/4th transition from AG to AB should yield [0.25, 0.25, 0.25, 0.25]
        sn_H = [1.0, 1.0, 1.0, 1.0]
        sn_A = [0.0, 0.0, 0.0, 0.0]
        pop = 30
        gossip = 0
        execError = 0.00
        b = 5.0

        influence = 0.75

        sn = linear(sn_H, sn_A, influence)

        @test isapprox([0.25, 0.25, 0.25, 0.25], sn)
    end
end