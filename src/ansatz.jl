# the tensor network ansatz, |ket> 
struct TensorNetworkAnsatz{TA}
    g::SimpleGraph{Int} # g is used to store the connections between the tensors
    site_tensors::Vector{TA} # the tensors are stored as vector of arrays, corresponding to vertices of the graph. The virtual bounds are in order of its neighbors, the last dimension is the open dimension
    gauge_tensors::Vector{TA} # corresponding to edges of the graph, listed in the order of edges(g), and enforce e.src < e.dst
    gauge_tensors_map::Dict{Tuple{Int, Int}, Int} # maps between the edge and the index of the gauge tensor

    function TensorNetworkAnsatz(g::SimpleGraph{Int}, d_virtual::Int, d_open::Int)
        TA = Array{ComplexF64}

        site_tensors = Vector{TA}()
        gauge_tensors = Vector{TA}()

        for v in vertices(g)
            t = zeros(ComplexF64, [d_virtual for _ in 1:length(neighbors(g, v))]..., d_open)
            push!(site_tensors, t)
        end

        for e in edges(g)
            t = (one(ComplexF64) * I)(d_virtual)
            push!(gauge_tensors, t)
        end

        map = Dict{Tuple{Int, Int}, Int}()
        for (i, e) in enumerate(edges(g))
            src, dst = minmax(e.src, e.dst)
            map[(src, dst)] = i
        end

        new{Array{ComplexF64}}(g, site_tensors, gauge_tensors, map)
    end
end

Base.show(io::IO, ansatz::TensorNetworkAnsatz{TA}) where {TA} = print(io, "TensorNetworkAnsatz{$(TA)} with $(nv(ansatz.g)) sites and $(ne(ansatz.g)) edges")

function inner_product(bra::TensorNetworkAnsatz{TA}, ket::TensorNetworkAnsatz{TA}; optimizer = GreedyMethod()) where {TA}
    @assert bra.g == ket.g

    # 1:nv(g) are the indices of open indices
    count = nv(bra.g) + 1
    ixs_bra, count = all_eins(bra.g, count)
    ixs_ket, count = all_eins(ket.g, count)

    ixs = vcat(ixs_bra, ixs_ket)
    raw_code = EinCode(ixs, Int[])

    xs = [bra.site_tensors..., bra.gauge_tensors..., ket.site_tensors..., ket.gauge_tensors...]
    size_dict = OMEinsum.get_size_dict!(ixs, xs, Dict{Int, Int}())

    opt_code = optimize_code(raw_code, size_dict, optimizer)
    @info "contraction complexity: $(contraction_complexity(opt_code, size_dict))"

    res = opt_code(xs...)
    return res[]
end