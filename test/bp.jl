using BPGauge
using Graphs, LinearAlgebra, OMEinsum
using Test

@testset "bp" begin
    g = random_regular_graph(30, 3)

    tn = zero_state(g, d_virtual = 2)
    bp_state = BPState(tn)
    @test length(bp_state.messages) == ne(g) * 2

    bp_path = BPPath(tn)
    @test length(bp_path.bp_steps) == ne(g) * 2
    for step in bp_path.bp_steps
        @test step.output[1] == step.Tid
        @test step.output[2] ∈ neighbors(g, step.Tid)
        for input in step.inputs
            @test input[2] == step.Tid
            @test input[1] ∈ neighbors(g, step.Tid)
        end
    end
    
    @test isnothing(bp!(bp_state, bp_path, tn))
end

@testset "1d chain" begin
    function chain(n::Int)
        g = SimpleGraph(n)
        for i in 1:n - 1
            add_edge!(g, i, i + 1)
        end
        return g
    end

    g = chain(10)
    for d in [2, 3, 10]
        tn = random_state(g, d_virtual = 3)
        normalize_state!(tn)
        bp_state = BPState(tn)
        bp_path = BPPath(tn)

        bp!(bp_state, bp_path, tn)

        T1 = tn.site_tensors[1]
        T2 = tn.site_tensors[2]
        T3 = tn.site_tensors[3]

        M12 = ein"li, ni -> ln"(T1, conj(T1))
        M23 = ein"li, ni, lmj, noj -> mo"(T1, conj(T1), T2, conj(T2))

        r12 = M12[1] / bp_state.messages[(1, 2)][1]
        @test isapprox(M12, r12 * bp_state.messages[(1, 2)], rtol = 1e-6)

        r23 = M23[1] / bp_state.messages[(2, 3)][1]
        @test isapprox(M23, r23 * bp_state.messages[(2, 3)], rtol = 1e-6)
    end
end

@testset "tree" begin
    g = SimpleGraph(6)
    add_edge!(g, 1, 2)
    add_edge!(g, 2, 3)
    add_edge!(g, 2, 4)
    add_edge!(g, 3, 5)
    add_edge!(g, 3, 6)

    for d in [2, 3, 10]
        tn = random_state(g, d_virtual = d)
        normalize_state!(tn)
        bp_state = BPState(tn)
        bp_path = BPPath(tn)

        bp!(bp_state, bp_path, tn)
        # check the message 2 -> 3
        T1 = tn.site_tensors[1]
        T2 = tn.site_tensors[2]
        T4 = tn.site_tensors[4]

        M23 = ein"li, oi, lnmj, oqpj, mk, pk -> nq"(T1, conj(T1), T2, conj(T2), T4, conj(T4))
        r23 = M23[1] / bp_state.messages[(2, 3)][1]
        @test isapprox(M23, r23 * bp_state.messages[(2, 3)], rtol = 1e-6)        
    end
end