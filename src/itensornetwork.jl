
function generate_zero_itensor(inds,elt,bd)
    l = length(inds)
    it = zeros(elt,2,fill(bd,l-1)...)
    it[1] = one(elt)
    return ITensor(it,inds...)
end


function create_zero_state(atoms, r, bd;elt=ComplexF64)
    g = unitdisk_graph(atoms,r)
    
    s = siteinds("S=1/2", g)
    x = ITensorNetwork(elt, s; link_space=bd)
    ind_vec = Index{Int64}[]
    for i in 1:length(x.data_graph.vertex_data.values)
        v = x.data_graph.vertex_data.values[i]
        x.data_graph.vertex_data.values[i] = generate_zero_itensor(v.tensor.inds,elt,bd)
        push!(ind_vec,v.tensor.inds[1])
    end
     
    return x, ind_vec,g
end


function apply_rydberg_hamiltonian(omega,delta,U,dt,ind_vec,g,tn;elt=ComplexF64,maxdim = 20)
    xop = [0 1;1 0]
    xit = exp(-im*dt*xop*omega/2)

    nop = [0 0;0 1]
    nit = exp(-im*dt*nop*(-delta))

    nnop = zeros(elt,4,4)
    nnop[end] = one(elt)
    nnit = reshape(exp(-im*dt*nnop*U),2,2,2,2)

    site_num = length(ind_vec)

    it_vec = ITensor[]
    for i in 1:site_num
        it = ITensor(xit,ind_vec[i],prime(ind_vec[i]))
        push!(it_vec,it)
    end

    for i in 1:site_num
        it = ITensor(nit,ind_vec[i],prime(ind_vec[i]))
        push!(it_vec,it)
    end
    tn = ITensorNetworks.apply(it_vec,tn;maxdim)

    ψψ = norm_sqr_network(tn)
    #Simple Belief Propagation Grouping
    bp_cache = BeliefPropagationCache(ψψ, group(v -> v[1], vertices(ψψ)))
    bp_cache = update(bp_cache; maxiter=20)

    for edge in edges(g)
        # @show tn.data_graph.vertex_data.values
        it = ITensor(nnit,prime(ind_vec[edge.src]),prime(ind_vec[edge.dst]),ind_vec[edge.src],ind_vec[edge.dst])
        envsSBP = environment(bp_cache, [(edge.src, "bra"), (edge.src, "ket"), (edge.dst, "bra"), (edge.dst, "ket")])

        tn = apply(it,tn;maxdim,envs = envsSBP,normalize=true,print_fidelity_loss=true,envisposdef=true,)
    end
    # tn = ITensorNetworks.apply(it_vec,tn;maxdim = 10)
    return tn
end
function build_adiabatic_sweep(Ω_max::Float64, Δ_max::Float64, t_max::Float64,t)
    Ω = Ω_max * sin(pi * t / t_max)^2
    Δ = (Δ_max * (2 * t / t_max - 1))
    return Ω, Δ
end


function apply_ising_hamiltonian(dt,ind_vec,g,tn;elt=ComplexF64,maxdim = 20,omega = 1.0)
    xop = [0 1;1 0]
    xit = exp(im*dt*xop*omega)

    zzop = zeros(elt,4,4)
    zzop[1,1] = one(elt)
    zzop[2,2] = -one(elt)
    zzop[3,3] = -one(elt)
    zzop[4,4] = one(elt)

    zzit = reshape(exp(im*dt*zzop),2,2,2,2)

    site_num = length(ind_vec)

    it_vec = ITensor[]
    for i in 1:site_num
        it = ITensor(xit,ind_vec[i],prime(ind_vec[i]))
        push!(it_vec,it)
    end
    tn = ITensorNetworks.apply(it_vec,tn;maxdim)

    ψψ = norm_sqr_network(tn)
    #Simple Belief Propagation Grouping
    bp_cache = BeliefPropagationCache(ψψ, group(v -> v[1], vertices(ψψ)))
    bp_cache = update(bp_cache; maxiter=20)

    for edge in edges(g)
        # @show tn.data_graph.vertex_data.values
        it = ITensor(zzit,prime(ind_vec[edge.src]),prime(ind_vec[edge.dst]),ind_vec[edge.src],ind_vec[edge.dst])
        envsSBP = environment(bp_cache, [(edge.src, "bra"), (edge.src, "ket"), (edge.dst, "bra"), (edge.dst, "ket")])

        tn = apply(it,tn;maxdim,envs = envsSBP,normalize=true,print_fidelity_loss=true,envisposdef=true,)
    end
    # tn = ITensorNetworks.apply(it_vec,tn;maxdim = 10)
    return tn
end