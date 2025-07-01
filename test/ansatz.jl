using BPGauge
using Graphs, LinearAlgebra, OMEinsum
using Test

@testset "states" begin
    for g in [random_regular_graph(30, 3), BPGauge.square_lattice(10, 10, 0.8)]
        phi = BPGauge.zero_state(g, d_virtual = 2)
        @test inner_product(phi, adjoint(phi)) ≈ 1.0

        phi = BPGauge.random_state(g, d_virtual = 2)
        normalize_state!(phi)
        @test inner_product(phi, adjoint(phi)) ≈ 1.0
    end
end