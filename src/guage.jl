# absorb the gauge tensors (diagonal matrices) into the site tensors, for bp to work
# - Ts --- Gamma --- Td - 
# - Ts_new - = - Ts -d- sqrt(Gamma) -dd-
# - Td_new - = -ss- sqrt(Gamma) -s- Td -
function absorb!(ansatz::TensorNetworkAnsatz{TA, TB}) where {TA, TB}
    for e in edges(ansatz.g)
        s, d = minmax(e.src, e.dst)
        sqrt_Gamma = sqrt.(ansatz.gauge_tensors[ansatz.gauge_tensors_map[(s, d)]])
        
        id_open = nv(ansatz.g) + 1
        dd = id_open + 1
        ss = id_open + 2
        # symmetric gauge, s and d tensor both absorb sqrt(Gamma), where Gamma is the gauge tensor on edge (s, d)
        # site s: [[n1, n2, ..., d, ..., nds, id_open], [d, dd]] -> [n1, n2, ..., dd, ..., nds, id_open]
        s_ixs = [[neighbors(ansatz.g, s)..., id_open], [d, dd]]
        s_iy = [neighbors(ansatz.g, s)..., id_open]
        s_iy[findfirst(==(d), s_iy)] = dd
        s_xs = (ansatz.site_tensors[s], sqrt_Gamma)
        size_dict = OMEinsum.get_size_dict(s_ixs, s_xs)
        einsum!(s_ixs, s_iy, s_xs, ansatz.site_tensors[s], 1, 0, size_dict)

        # site d: [[n1, n2, ..., s, ..., ndd], [ss, s]] -> [n1, n2, ..., ss, ..., ndd]
        d_ixs = [[neighbors(ansatz.g, d)..., id_open], [ss, s]]
        d_iy = [neighbors(ansatz.g, d)..., id_open]
        d_iy[findfirst(==(s), d_iy)] = ss
        d_xs = (ansatz.site_tensors[d], sqrt_Gamma)
        size_dict = OMEinsum.get_size_dict(d_ixs, d_xs)
        einsum!(d_ixs, d_iy, d_xs, ansatz.site_tensors[d], 1, 0, size_dict)
    end
    nothing
end

# assume that the existing gauge tensors are already absorbed into the site tensors
# the bp converge, using the bp messages to generate the new gauge tensors, which are real valued diagonal matrices
# according to Eqs.12~17 of https://scipost.org/SciPostPhys.15.6.222
# site s and site d
# Ts -i- sqrt_Msd_inv (j, i) -j- sqrt_Msd (k, j) -k- sqrt_Mds (k, l) -l- sqrt_Mds_inv (l, m) -m- Td
# Ts -i- sqrt_Msd_inv (j, i) -j- U(j, n) -n- S(n, o) -o- Vt(o, l) -l- sqrt_Mds_inv (l, m) -m- Td
# Ts -i- As (i, n) -n- Gamma -o- Ad (m, o) -m- Td
# Ts -i- Gamma -m- Td
function gauge!(ansatz::TensorNetworkAnsatz{TA, TB}, state::BPState{TA}) where {TA, TB}
    for e in edges(ansatz.g)
        s, d = minmax(e.src, e.dst)
        M_sd = state.messages[(s, d)]
        M_ds = state.messages[(d, s)]
        sqrt_Msd, sqrt_Msd_inv = square_root(M_sd)
        sqrt_Mds, sqrt_Mds_inv = square_root(M_ds)

        M_mid = ein"kj, kl -> jl"(sqrt_Msd, sqrt_Mds)
        res = svd(M_mid)
        U = res.U
        S = res.S
        Vt = res.Vt

        ansatz.gauge_tensors[ansatz.gauge_tensors_map[(s, d)]] .= diagm(S)
        
        # the tensors have to be absorbed into the site tensors s and d
        absorb_s = ein"ji, jn -> in"(sqrt_Msd_inv, U)
        absorb_d = ein"lm, ol -> mo"(sqrt_Mds_inv, Vt)

        id_open = nv(ansatz.g) + 1
        i = d
        n = id_open + 1
        m = s
        o = id_open + 2

        # site s
        s_ixs = [[neighbors(ansatz.g, s)..., id_open], [i, n]]
        s_iy = [neighbors(ansatz.g, s)..., id_open]
        s_iy[findfirst(==(i), s_iy)] = n
        s_xs = (ansatz.site_tensors[s], absorb_s)
        size_dict = OMEinsum.get_size_dict(s_ixs, s_xs)
        einsum!(s_ixs, s_iy, s_xs, ansatz.site_tensors[s], 1, 0, size_dict)

        # site d
        d_ixs = [[neighbors(ansatz.g, d)..., id_open], [m, o]]
        d_iy = [neighbors(ansatz.g, d)..., id_open]
        d_iy[findfirst(==(m), d_iy)] = o
        d_xs = (ansatz.site_tensors[d], absorb_d)
        size_dict = OMEinsum.get_size_dict(d_ixs, d_xs)
        einsum!(d_ixs, d_iy, d_xs, ansatz.site_tensors[d], 1, 0, size_dict)
    end
    nothing
end