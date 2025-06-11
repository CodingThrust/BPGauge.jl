using BPGauge
using Test
using ITensorNetworks
using ITensors
using BPGauge:generate_zero_itensor

@testset "generate_zero_itensor" begin
    inds = [Index(2),Index(1),Index(1)]
    elt = ComplexF64
    bd = 1
    it = generate_zero_itensor(inds,elt,bd)
    @test it isa ITensor
    @test it.tensor.storage.data == elt[1.0; 0.0]
end

@testset "create_zero_state" begin
    atoms = [(1.0,1.0),(0.0,1.0),(1.0,0.0),(0.0,0.0)]
    r = 1.2
    tn,ind_vec,g = create_zero_state(atoms,r, bd)

    @test tn.data_graph.underlying_graph.position_graph == g
end

@testset "build_adiabatic_sweep" begin 
    Ω_max = 2π * 4
    Δ_max = 3 * Ω_max
    t_max = 2.0
    t = 0.0
    Ω, Δ = build_adiabatic_sweep(Ω_max, Δ_max, t_max,t)
    @test Ω ≈ 0.0 atol = 1e-10
    @test Δ ≈ -Δ_max atol = 1e-10

    t = 1.0
    Ω, Δ = build_adiabatic_sweep(Ω_max, Δ_max, t_max,t)
    @test Ω ≈ Ω_max atol = 1e-10
    @test Δ ≈ 0.0 atol = 1e-10

    t = 2.0
    Ω, Δ = build_adiabatic_sweep(Ω_max, Δ_max, t_max,t)
    @test Ω ≈ 0.0 atol = 1e-10
    @test Δ ≈ Δ_max atol = 1e-10
end

@testset "BPGauge" begin
    unit = 3.0
    r = 3.5
    bd = 1
    Ω_max = 2π * 4
    Δ_max = 3 * Ω_max
    t_max = 1.5
    U = Ω_max * 20
    dt = 0.01
    maxdim = 10
    lattice_size = (3,3)

    lattice = SquareLattice()
    atoms = map(x -> unit .* x, generate_sites(lattice, lattice_size...));
    tn,ind_vec,g = create_zero_state(atoms,r, bd)
    for t in dt:dt:t_max
        Ω, Δ = build_adiabatic_sweep(Ω_max, Δ_max, t_max,t)
        @show t Ω Δ 
        tn = apply_rydberg_hamiltonian(Ω,Δ,U,dt,ind_vec,g,tn;maxdim)
    end
    expc = ITensorNetworks.expect(tn, "Sz"; alg="bp")
    @show real(expc).> 0.0
    @show expc
end
