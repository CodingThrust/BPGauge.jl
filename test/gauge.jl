using BPGauge
using Graphs, LinearAlgebra, OMEinsum
using Test

using Random
Random.seed!(1234)

using BPGauge: square_root

@testset "absorb" begin
    for g in [random_regular_graph(30, 3), BPGauge.square_lattice(10, 10, 0.8)]
        tn = random_state(g, d_virtual = 2)
        for gt in tn.gauge_tensors
            gt .= diagm(rand(size(gt, 1)))
        end
        normalize_state!(tn)
        absorb!(tn)
        @test inner_product(tn, adjoint(tn)) ≈ 1.0

        tn = zero_state(g, d_virtual = 2)
        for gt in tn.gauge_tensors
            gt .*= 2.0
        end
        absorb!(tn)
        for i in 1:nv(g)
            @test tn.site_tensors[i][1] ≈ 2^(0.5 * degree(g, i))
        end
    end
end

@testset "svd" begin
    A = rand(ComplexF64, 10, 10)
    res = svd(A)
    @test A ≈ ein"ij, jk, kl -> il"(res.U, diagm(res.S), res.Vt)
end

@testset "gauge" begin
    # gauge by message tensor
    g = SimpleGraph(5)
    add_edge!(g, 1, 3)
    add_edge!(g, 2, 3)
    add_edge!(g, 3, 4)
    add_edge!(g, 4, 5)
    tn = random_state(g, d_virtual = 10)
    normalize_state!(tn)
    origin_34 = ein"ijkm, kln -> ijmnl"(tn.site_tensors[3], tn.site_tensors[4])

    t = rand(ComplexF64, 10, 10)
    M_34 = t * adjoint(t)
    t = rand(ComplexF64, 10, 10)
    M_43 = t * adjoint(t)

    apply_gauge!(tn, 3, 4, M_34, M_43)

    gauged_34 = ein"ijom, pln, op -> ijmnl"(tn.site_tensors[3], tn.site_tensors[4], tn.gauge_tensors[tn.gauge_tensors_map[(3, 4)]])
    @test isapprox(gauged_34, origin_34, atol = 1e-8)
    @test inner_product(tn, adjoint(tn)) ≈ 1.0
end

@testset "gauge by bp" begin
    g = random_regular_graph(10, 3)
    tn = random_state(g, d_virtual = 3)
    normalize_state!(tn)
    bp_state = BPState(tn)
    bp_path = BPPath(tn)
    bp!(bp_state, bp_path, tn, verbose = true)

    gauge!(tn, bp_state)
    @test inner_product(tn, adjoint(tn)) ≈ 1.0
end

@testset "Vidal gauge mps" begin
    g = path_graph(100)
    tn = random_state(g, d_virtual = 8)
    normalize_state!(tn)
    
    bp_state = BPState(tn)
    bp_path = BPPath(tn)
    bp!(bp_state, bp_path, tn, atol = 1e-14)

    gauge!(tn, bp_state)
    @test inner_product(tn, adjoint(tn)) ≈ 1.0

    for i in 2:nv(g) - 2
        G = tn.gauge_tensors[tn.gauge_tensors_map[(i, i + 1)]]
        T = tn.site_tensors[i + 1]
        L = ein"ij, jk, jln, kmn -> lm"(G, G, T, conj(T))
        @test maximum(abs.(L ./ L[1, 1] - I(size(L, 1)))) < 1e-6
    end
end

@testset "cycle graph gauge" begin
    g = path_graph(100)
    add_edge!(g, 1, 100)

    tn = random_state(g, d_virtual = 4)
    normalize_state!(tn)
    @test inner_product(tn, adjoint(tn)) ≈ 1.0

    bp_state = BPState(tn)
    bp_path = BPPath(tn)
    bp!(bp_state, bp_path, tn, atol = 1e-8)

    gauge!(tn, bp_state)
    @test inner_product(tn, adjoint(tn)) ≈ 1.0

    @test dist_to_vidal(tn, bp_path) < 1e-5
end

@testset "rr3 graph gauge" begin
    g = random_regular_graph(20, 3)
    tn = random_state(g, d_virtual = 4)
    normalize_state!(tn)
    @test inner_product(tn, adjoint(tn)) ≈ 1.0

    bp_state = BPState(tn)
    bp_path = BPPath(tn)
    bp!(bp_state, bp_path, tn, atol = 1e-8)

    gauge!(tn, bp_state)
    @test inner_product(tn, adjoint(tn)) ≈ 1.0

    @test dist_to_vidal(tn, bp_path) < 1e-5
end