# absorb the gauge tensors (diagonal matrices) into the site tensors, for bp to work
# after the gauge is absorbed, set them to identity
function absorb!(ansatz::TensorNetworkAnsatz{TA, TB}) where {TA, TB}
    for e in edges(ansatz.g)
        s, d = minmax(e.src, e.dst)
        Gamma = ansatz.gauge_tensors[ansatz.gauge_tensors_map[(s, d)]]
        sqrt_Gamma = sqrt.(Gamma)

        eincode_s, eincode_d = absorb_eins(s, d, neighbors(ansatz.g, s), neighbors(ansatz.g, d))

        s_xs = (ansatz.site_tensors[s], sqrt_Gamma)
        sds = OMEinsum.get_size_dict(eincode_s.ixs, s_xs)
        einsum!(eincode_s.ixs, eincode_s.iy, s_xs, ansatz.site_tensors[s], 1, 0, sds)

        d_xs = (ansatz.site_tensors[d], sqrt_Gamma)
        sdd = OMEinsum.get_size_dict(eincode_d.ixs, d_xs)
        einsum!(eincode_d.ixs, eincode_d.iy, d_xs, ansatz.site_tensors[d], 1, 0, sdd)

        # reset the gauge tensor to identity
        Gamma .= I(size(Gamma, 1))
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

        apply_gauge!(ansatz, s, d, M_sd, M_ds)
    end
    nothing
end

# given the message matrix M_sd and M_ds, generate the new gauge tensor Gamma, update T_s and T_d
function apply_gauge!(ansatz::TensorNetworkAnsatz{TA, TB}, s::Int, d::Int, M_sd::Matrix{T}, M_ds::Matrix{T}) where {TA, TB, T}
    # square root the message matrix
    sqrt_Msd, sqrt_Msd_inv = square_root(M_sd)
    sqrt_Mds, sqrt_Mds_inv = square_root(M_ds)

    # the eigen decomposition is only correct when the message matrix is positive definite
    # @show ein"ji, kj -> ik"(sqrt_Msd_inv, sqrt_Msd) ≈ I(size(M_sd, 1))
    # @show ein"kl, lm -> km"(sqrt_Mds, sqrt_Mds_inv) ≈ I(size(M_sd, 1))
    # @show ein"ji, kj, kl, lm -> im"(sqrt_Msd_inv, sqrt_Msd, sqrt_Mds, sqrt_Mds_inv) ≈ I(size(M_sd, 1))

    # generate the new gauge tensor
    M_mid = ein"kj, kl -> jl"(sqrt_Msd, sqrt_Mds)

    # @show ein"ji, jl, lm -> im"(sqrt_Msd_inv, M_mid, sqrt_Mds_inv) ≈ I(size(M_mid, 1))

    res = svd(M_mid)
    U = res.U # jn
    S = diagm(res.S) # no
    Vt = res.Vt # ol

    # @show ein"jn, no, ol -> jl"(U, S, Vt) ≈ M_mid
    # @show maximum(abs.(ein"ji, jn, no, ol, lm -> im"(sqrt_Msd_inv, U, S, Vt, sqrt_Mds_inv) - I(size(M_mid, 1))))

    As = ein"ji, jn -> in"(sqrt_Msd_inv, U)
    Ad = ein"lm, ol -> mo"(sqrt_Mds_inv, Vt)

    # @show maximum(abs.(ein"in, no, mo -> im"(As, S, Ad) - I(size(As, 1))))

    #update the gauge tensor
    ansatz.gauge_tensors[ansatz.gauge_tensors_map[(s, d)]] .= S

    # absorb As and Ad into the site tensors
    eincode_s, eincode_d = absorb_eins(s, d, neighbors(ansatz.g, s), neighbors(ansatz.g, d))
    s_xs = (ansatz.site_tensors[s], As)
    sds = OMEinsum.get_size_dict(eincode_s.ixs, s_xs)
    einsum!(eincode_s.ixs, eincode_s.iy, s_xs, ansatz.site_tensors[s], 1, 0, sds)

    d_xs = (ansatz.site_tensors[d], Ad)
    sdd = OMEinsum.get_size_dict(eincode_d.ixs, d_xs)
    einsum!(eincode_d.ixs, eincode_d.iy, d_xs, ansatz.site_tensors[d], 1, 0, sdd)

    nothing
end

# check the distance from a gauged state to the vidal gauge
function dist_to_vidal(tn::TensorNetworkAnsatz, bp_path::BPPath)
    Ts = tn.site_tensors
    Tcs = conj.(Ts)

    err = 0.0
    for step in bp_path.bp_steps
        eincode = step.eincode
        Tid = step.Tid
        inputs = step.inputs
        output = step.output

        T = Ts[Tid]
        Tc = Tcs[Tid]
        Λs2 = [tn.gauge_tensors[tn.gauge_tensors_map[(min(i, j), max(i, j))]] for (i, j) in inputs].^2

        output_tensor = eincode(T, Tc, Λs2...)
        err = max(err, maximum(abs.(output_tensor - output_tensor[1, 1] * I(size(output_tensor, 1)))))
    end
    return err
end