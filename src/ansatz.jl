# the tensor network ansatz, |ket> 
struct TensorNetworkAnsatz{TA, TB}
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
end

Base.show(io::IO, ansatz::TensorNetworkAnsatz{TA, TB}) where {TA, TB} = print(io, "TensorNetworkAnsatz{$(TA), $(TB)} with $(nv(ansatz.g)) sites and $(ne(ansatz.g)) edges")

function inner_product(bra::TensorNetworkAnsatz{TA, TB}, ket::TensorNetworkAnsatz{TA}; optimizer = GreedyMethod()) where {TA, TB}
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