# generating graphs
function square_lattice(mx::Int, my::Int, p::Float64; seed::Int = 1234)
    Random.seed!(seed)
    n = Int(ceil(mx * my * p))
    g = SimpleGraph(n)
    positions = [(i, j) for i in 1:mx, j in 1:my]
    shuffle!(positions)
    positions = positions[1:n]
    sort!(positions)
    for (i, pos) in enumerate(positions)
        mxi, myi = pos
        for (dx, dy) in [(0, 1), (1, 0)]
            mxj, myj = mxi + dx, myi + dy
            if (mxj, myj) in positions
                j = findfirst(x -> x == (mxj, myj), positions)
                add_edge!(g, i, j)
            end
        end
    end
    return g
end

function chain(n::Int)
    g = SimpleGraph(n)
    for i in 1:n-1
        add_edge!(g, i, i+1)
    end
    return g
end

# codes about constructing the einsum expression

function all_eins(g::SimpleGraph{Int}, count::Int)
    indices_dict = Dict{Tuple{Int, Int}, Int}()
    ixs = Vector{Vector{Int}}()

    # site tensors
    for v in vertices(g)
        ix = Int[]
        for n in neighbors(g, v)
            push!(ix, count)
            indices_dict[(v, n)] = count
            count += 1
        end
        push!(ix, v)
        push!(ixs, ix)
    end

    # gauge tensors
    for e in edges(g)
        src, dst = minmax(e.src, e.dst)
        push!(ixs, [indices_dict[(src, dst)], indices_dict[(dst, src)]])
    end

    return ixs, count
end

function bp_eins(nebis::Vector{Int}, inputs::Vector{Tuple{Int, Int}}, output::Tuple{Int, Int})
    ixs = Vector{Vector{Int}}()
    iy = Int[]

    # 1 is indices of the open indices
    # 2:d + 1 are indices of the virtual indices of T
    # d + 2:2d + 1 are indices of the virtual indices of T*
    d = length(nebis)
    T_ids = [2:d+1..., 1]
    Tc_ids = [d+2:2d+1..., 1]
    push!(ixs, T_ids)
    push!(ixs, Tc_ids)

    output_leg = findfirst(x -> x == output[2], nebis)
    iy = [output_leg + 1, output_leg + 1 + d]

    for (i, j) in inputs
        input_leg = findfirst(x -> x == i, nebis)
        push!(ixs, [input_leg + 1, input_leg + 1 + d])
    end

    raw_code = EinCode(ixs, iy)
    return optimize_code(raw_code, uniformsize(raw_code, 2), GreedyMethod())
end


# the eincode for absorbing the square root of the gauge tensor into the site
function absorb_eins(s::Int, d::Int, nebis_s::Vector{Int}, nebis_d::Vector{Int})
    id_open = 0

    iy_s = [nebis_s..., id_open]
    iy_d = [nebis_d..., id_open]

    ix_s = [nebis_s..., id_open]
    ix_s[findfirst(==(d), ix_s)] = s

    ix_d = [nebis_d..., id_open]
    ix_d[findfirst(==(s), ix_d)] = d

    return EinCode([ix_s, [s, d]], iy_s), EinCode([ix_d, [d, s]], iy_d)
end

# a very simple strategy to uniform the message tensor
function uniform!(t::AbstractArray)
    t ./= abs(sum(t))
end

# square_root of the message matrix M_12 via eigen decomposition
# note that the definition here is a little bit different from the one in the paper https://scipost.org/SciPostPhys.15.6.222
# with eigen decomposition, M_12 = U * diagm(D) * adjoint(U) = SM_13 * SM_32
# SM_13 = M_12^0.5 = U * sqrt(diagm(D)), SM_32 = M_12^(-0.5) = 1/sqrt(diagm(D)) * adjoint(U)
# it is assumed that the message matrix is positive definte, thus all elements of D should be positive
function square_root(M::Matrix{Complex{T}}) where T
    @assert M ≈ adjoint(M) # the message matrix is Hermitian
    M = LinearAlgebra.hermitian(M)
    res = eigen(M)
    D = res.values
    U = res.vectors

    # it seems that eigen decomposition some time leads to small negative values, fix them to 10 times the machine precision
    # see https://github.com/ITensor/ITensorNetworks.jl/blob/2d7db49c2f003c6588a1632fd6f0fd802c0ac9ca/src/gauging.jl#L70
    # this operation is extremely ill-conditioned, leading to huge drop of accuracy
    for i in eachindex(D)
        (abs(D[i]) < 10 * eps(T)) && (D[i] = 10 * eps(T))
    end

    # need to consider the pseudoinverse if D contains 0
    sqrt_D = sqrt.(D)
    sqrt_M = U * diagm(sqrt_D)
    sqrt_M_inv = diagm([!iszero(d) ? inv(d) : zero(typeof(d)) for d in sqrt_D]) * adjoint(U)
    return sqrt_M, sqrt_M_inv
end