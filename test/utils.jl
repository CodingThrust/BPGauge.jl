using BPGauge
using Graphs, LinearAlgebra, OMEinsum
using Test

@testset "graphs" begin
    g = BPGauge.square_lattice(10, 10, 0.8)
    @test nv(g) == 80
end

@testset "eins" begin
    g = random_regular_graph(30, 3)
    ixs, count = BPGauge.all_eins(g, 31)
    @test count == 31 + 2 * ne(g)
    @test length(ixs) == nv(g) + ne(g)

    vec_ixs = ixs[1:nv(g)]
    edge_ixs = ixs[nv(g) + 1:end]

    for (i, e) in enumerate(edges(g))
        src, dst = minmax(e.src, e.dst)
        @test edge_ixs[i][1] == vec_ixs[src][findfirst(x -> x == dst, neighbors(g, src))]
        @test edge_ixs[i][2] == vec_ixs[dst][findfirst(x -> x == src, neighbors(g, dst))]
    end
end

@testset "bp eins" begin
    nebis = [2, 3, 4]
    inputs = [(2, 1), (4, 1)]
    output = (1, 3)
    eincode_generated = BPGauge.bp_eins(nebis, inputs, output)
    eincode_manul = ein"ijkl, mnol, im, ko -> jn"
    T1 = rand(3, 4, 5, 2)
    T2 = rand(3, 4, 5, 2)
    M1 = rand(3, 3)
    M3 = rand(5, 5)
    M2_generated = eincode_generated(T1, T2, M1, M3)
    M2_manul = eincode_manul(T1, T2, M1, M3)
    @test M2_generated ≈ M2_manul
end

@testset "absorb_eins" begin
    g = SimpleGraph(5)
    add_edge!(g, 1, 3)
    add_edge!(g, 2, 3)
    add_edge!(g, 3, 4)
    add_edge!(g, 4, 5)
    s,d = 3, 4
    eincode_s, eincode_d = BPGauge.absorb_eins(s, d, neighbors(g, s), neighbors(g, d))
    eincode_s_man = ein"ijkm, ko -> ijom"
    eincode_d_man = ein"kln, ok -> oln"

    T3 = rand(3, 4, 5, 2)
    T4 = rand(5, 6, 2)
    G = diagm(rand(5))

    @test eincode_s(T3, G) ≈ eincode_s_man(T3, G)
    @test eincode_d(T4, G) ≈ eincode_d_man(T4, G)
end

@testset "square_root" begin
    # generate a random positive definite matrix
    A = rand(ComplexF64, 10, 10)
    A = A * adjoint(A)
    sqrt_A, sqrt_A_inv = BPGauge.square_root(A)

    # sqrt
    @test sqrt_A * adjoint(sqrt_A) ≈ A

    # inverse
    @test sqrt_A * sqrt_A_inv ≈ I(10)
    @test sqrt_A_inv * sqrt_A ≈ I(10)
end