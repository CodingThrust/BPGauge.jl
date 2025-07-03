# the tensor network ansatz, |ket> 
struct TensorNetworkAnsatz{TA <: AbstractArray, TB <: AbstractArray}
    g::SimpleGraph{Int} # g is used to store the connections between the tensors
    site_tensors::Vector{TA} # the tensors are stored as vector of arrays, corresponding to vertices of the graph. The virtual bounds are in order of its neighbors, the last dimension is the open dimension
    gauge_tensors::Vector{TB} # corresponding to edges of the graph, listed in the order of edges(g), and enforce e.src < e.dst, real valued diagonal matrices
    gauge_tensors_map::Dict{Tuple{Int, Int}, Int} # maps between the edge and the index of the gauge tensor

    function TensorNetworkAnsatz(g::SimpleGraph{Int}, d_virtual::Int, d_open::Int; type::Type = Float64)
        TA = Array{Complex{type}}
        TB = Array{type}

        site_tensors = Vector{TA}()
        gauge_tensors = Vector{TB}()

        for v in vertices(g)
            t = zeros(Complex{type}, [d_virtual for _ in 1:length(neighbors(g, v))]..., d_open)
            push!(site_tensors, t)
        end

        for e in edges(g)
            t = (one(type) * I)(d_virtual)
            push!(gauge_tensors, t)
        end

        map = Dict{Tuple{Int, Int}, Int}()
        for (i, e) in enumerate(edges(g))
            src, dst = minmax(e.src, e.dst)
            map[(src, dst)] = i
        end

        new{TA, TB}(g, site_tensors, gauge_tensors, map)
    end

    function TensorNetworkAnsatz(g::SimpleGraph{Int}, site_tensors::Vector{TA}, gauge_tensors::Vector{TB}, map::Dict{Tuple{Int, Int}, Int}) where {TA, TB}
        @assert length(site_tensors) == nv(g)
        @assert length(gauge_tensors) == ne(g)
        new{TA, TB}(g, site_tensors, gauge_tensors, map)
    end
end

Base.show(io::IO, ansatz::TensorNetworkAnsatz{TA, TB}) where {TA, TB} = print(io, "TensorNetworkAnsatz{$(TA), $(TB)} with $(nv(ansatz.g)) sites and $(ne(ansatz.g)) edges")

Base.adjoint(ansatz::TensorNetworkAnsatz{TA, TB}) where {TA, TB} = TensorNetworkAnsatz(ansatz.g, TA[conj(t) for t in ansatz.site_tensors], ansatz.gauge_tensors, ansatz.gauge_tensors_map)

function inner_product(bra::TensorNetworkAnsatz{TA, TB}, ket::TensorNetworkAnsatz{TA}; optimizer = GreedyMethod()) where {TA, TB}
    @assert bra.g == ket.g

    raw_code = inner_product_eins(bra.g)

    xs = [bra.site_tensors..., bra.gauge_tensors..., ket.site_tensors..., ket.gauge_tensors...]
    size_dict = OMEinsum.get_size_dict!(raw_code.ixs, xs, Dict{Int, Int}())

    opt_code = optimize_code(raw_code, size_dict, optimizer)
    # @info "contraction complexity: $(contraction_complexity(opt_code, size_dict))"

    res = opt_code(xs...)
    return res[]
end

# initialize the state for rydberg system, all sites are in the |0> state
function zero_state(g::SimpleGraph{Int}; d_virtual::Int = 1)
    phi = TensorNetworkAnsatz(g, d_virtual, 2)
    # set all to the |0> state
    for tensor in phi.site_tensors
        tensor[1] = ComplexF64(1.0)
    end
    return phi
end

# generate non-normalized random state
function random_state(g::SimpleGraph{Int}; d_virtual::Int = 1)
    phi = TensorNetworkAnsatz(g, d_virtual, 2)
    for tensor in phi.site_tensors
        tensor .= rand(ComplexF64, size(tensor))
    end
    return phi
end

function normalize_state!(phi::TensorNetworkAnsatz{TA, TB}) where {TA, TB}
    n = inner_product(phi, adjoint(phi))
    phi.site_tensors .= phi.site_tensors ./ (sqrt(real(n))^(1 / nv(phi.g)))
    return phi
end