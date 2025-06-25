using LinearAlgebra
using LaTeXStrings
using Plots

⊗(a, b) = kron(a, b)
Id = [1 0; 0 1]
σ_x = [0 1; 1 0]
σ_z = [1 0; 0 -1]
#V/(j-i)^6 set to a rigid number, nearest-neighbor interaction, here we set to 1

number=[0 0; 0 1] #or could ~~~~ r*r.dag()(qt.qeye(2)-qt.sigmaz())/2

function n(i)
    cup=fill(Id,N)
    cup[i]= number
    number_operator=foldl(⊗, cup)
    return number_operator
end 

function nn(i1, i2)
    cup=fill(Id,N)
    cup[i1],cup[i2]=number, number
    number_operator=foldl(⊗, cup)
    return number_operator
end

function sigma_x(i) 
    cup=fill(Id,N)
    cup[i]= [0 1; 1 0] #sigma_x
    sigma_x=foldl(⊗, cup)
    return sigma_x
end

function min_distance_pbc(i, j, n_atoms)
    direct_dist = abs(j - i) 
    wrap_dist = (n_atoms - abs(j - i)) 
    return min(direct_dist, wrap_dist)
end

function Rydberg_Ham1D(timelist::Vector{Float64}, n::Int64, d::Float64, Ω::Vector{Float64}, Δ::Vector{Float64}, V::Float64, pbc::Bool = true)
    # n: number of atoms
    # d: distance between atoms
    # Ω: Rabi frequency 
    # Δ: detuning
    # V: interaction strength
    @assert length(Ω) == length(timelist) && length(Δ) == length(timelist) "Ω and Δ must be vectors of length n"

    H_timelis = Vector{Matrix{Float64}}(zeros(Float64, 2^n, 2^n))

    H= zeros(Float64, 2^n, 2^n)
    
    for (idx, t) in enumerate(timelist)
        Ω_t = Ω[idx]
        Δ_t = Δ[idx]
        for i in 1:n
            H-=Δ_t*n(i)+Ω_t/2*sigma_x(i)
        end
        if pbc
            for i in 1:n
                for j in i+1:n
                    H+= V/(min_distance_pbc(i,j,n)*d)^6*nn(i,j)
                end
            end
        else
            for i in 1:n
                for j in i+1:n
                    H+= V/((j-i)*d)^6*nn(i,j)
                end
            end
        end
        H_timelis[idx] = H
    end

    return H
    
end

function Ising_Ham1D(N::Int64, pbc::Bool=true) 
    zz=σ_z ⊗ σ_z
    H = zeros(2^N, 2^N)

    
    if pbc
        if N==1
            H=-σ_x
        else
            for i in 1:N-1
                H-=I(2^(i-1)) ⊗ zz ⊗ I(2^(N-i-1))
            end
            for i in 1:N
                H-=I(2^(i-1)) ⊗ σ_x ⊗ I(2^(N-i))
            end
            H-=I(2^(N-2)) ⊗ σ_z ⊗ I(2^(N-2))
        end
    else
        if N==1
            H=-σ_x
        else
            for i in 1:N-1
                H-=I(2^(i-1)) ⊗ zz ⊗ I(2^(N-i-1))
            end
            for i in 1:N
                H-=I(2^(i-1)) ⊗ σ_x ⊗ I(2^(N-i))
            end
        end
    end
    return H
end

# Function for generating the Ising Hamiltonian in 2D
function Ising_Ham2D(N::Int64, J::Float64, h::Float64, pbc::Bool=true)
    H = zeros(Float64, 2^(N^2), 2^(N^2))  # Hamiltonian matrix

    # Iterate over all spins to build the Hamiltonian
    for i in 1:N
        for j in 1:N
            # Index of the spin in the flattened array
            index = (i - 1) * N + j
            
            # Interaction with neighboring spins
            if j < N  # Right neighbor
                neighbor_index = index + 1 
                @show (index, neighbor_index)
                H -= J * (I(2^(index-1))⊗σ_z ⊗ σ_z ⊗ I(2^(N^2-index-1)))
            end
            
            
            if i < N  # Upper neighbor
                neighbor_index = index + N
                H -= J * (I(2^(index-1))⊗σ_z ⊗ I(2^(N-1))) ⊗ σ_z ⊗ I(2^(N^2 -N- index))
                @show (index, neighbor_index)
            end
            
            
            # Transverse field term
            H -= h * I(2^(index-1)) ⊗ σ_x ⊗ I(2^(N^2 - index))
        end
    end

    # # Handle periodic boundary conditions
    if pbc
        # Add periodic boundary interactions in x-direction
        for i in 1:N
            H -= J * (I(2^((i - 1) * N)) ⊗ σ_z ⊗ I(2^(N - 1))) ⊗ (I(2^(N * i - N)) ⊗ σ_z)
        end
        
        # Add periodic boundary interactions in y-direction
        for j in 1:N
            H -= J * (I(2^((j - 1))) ⊗ σ_z ⊗ I(2^(N - 1))) ⊗ (I(2^(N^2 - j)) ⊗ σ_z)
        end
    end
    
    return H
end

function wf_time_evolution(psi0::Vector{T}, times::Vector{Float64}, energy::Vector{Float64},states::Matrix{Float64}) where {T <: Real}
    wflis=Vector{Vector{ComplexF64}}(undef,length(times))
    c = states'*psi0
    exp_factors = [exp.(-1im * t * energy) for t in times]
    
    # Use multi-threading for parallel computation
    Threads.@threads for i in eachindex(times)
        wflis[i] = states * (c .* exp_factors[i])
    end
    return wflis
end

function build_adiabatic_sweep(Ω_max::Float64, Δ_max::Float64, t_max::Float64,t)
    Ω = Ω_max * sin(pi * t / t_max)^2
    Δ = (Δ_max * (2 * t / t_max - 1))
    return Ω, Δ
end

# N=6
# timelis = collect(0:0.1:1.5)
# unit = 3.0
# r = 3.5
# Ω_max = 2π * 4
# Δ_max = 3 * Ω_max
# t_max = 1.5
# V= Ω_max * 20

# Ω, Δ = build_adiabatic_sweep(Ω_max, Δ_max, t_max, timelis)
# initial_state = zeros(Float64, 2^N)
# initial_state[1] = 1.0  # Set the first state to 1 (ground state)

N=8
initial_state = zeros(Float64, 2^N)
initial_state[1] = 1.0  # Set the first state to 1 (ground state)
timelis = collect(0:0.05:1.5)

energy, states = eigen(Ising_Ham1D(N))
wflis = wf_time_evolution(initial_state, timelis, energy, states)

magnetization = zeros(Float64, length(timelis))
for (i,state) in enumerate(wflis)
    mag=0.0
    for j in 1:N
        cup = fill(Id, N)
        cup[j] = sigmaz
        sigmaz_operator = foldl(⊗, cup)
        mag+=state'*sigmaz_operator*state  # Apply the sig
    end
    magnetization[i]=mag  # Calculate magnetization
end 

plot(timelis, magnetization ./N, label=false, xlabel=L"t", ylabel=L"M(t)=\sum_i Z_i /N", marker=:circle)

N =3 
timelis = collect(0:0.05:10)
energy, states = eigen(Ising_Ham2D(N, 1.0, 0.5, false))
initial_state = zeros(Float64, 2^(N^2))
initial_state[1] = 1.0  # Set the first state to 1
wflis = wf_time_evolution(initial_state, timelis, energy, states)
magnetization = zeros(Float64, length(timelis))
for (i,state) in enumerate(wflis)
    mag=0.0
    for j in 1:N^2          
        cup = fill(Id, N^2)
        cup[j] = σ_z
        sigmaz_operator = foldl(⊗, cup)
        mag+=state'*sigmaz_operator*state  # Apply the sigmaz operator
    end
    magnetization[i]=mag  # Calculate magnetization
end

plot(timelis, magnetization ./N^2, label=false, xlabel=L"t", ylabel=L"M(t)=\sum_i Z_i /N^2", marker=:circle)