using BPGauge
using Test
using ITensorNetworks

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
    @show  real(expc).> 0.0
    @show expc
end
