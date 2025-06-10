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