# bp on the TensorNetworkAnsatz

# each message tensor have two legs, as follows
# --Ta --   |--1--Tb --
#          Mab
# --Ta*--   |--2--Tb*--
struct BPState{TA}
    messages::Dict{Tuple{Int, Int}, TA}
    function BPState(ansatz::TensorNetworkAnsatz{TA, TB}) where {TA, TB}
        messages = Dict{Tuple{Int, Int}, TA}()
        TE = eltype(TA)

        # initialize the messages from neighbors to sites
        for dst in vertices(ansatz.g)
            for (i, src) in enumerate(neighbors(ansatz.g, dst))
                d_virtual = size(ansatz.site_tensors[dst])[i]
                messages[(src, dst)] = ones(TE, d_virtual, d_virtual)
                normalize_message!(messages[(src, dst)])
            end
        end

        new{TA}(messages)
    end
end

# a single bp update step: \sum_{T_neighbors} T * T* * M_inputs... -> M_output
struct BPStep
    eincode::AbstractEinsum # Eincode([T..., T*..., M_inputs...], [M_output])
    Tid::Int # the id of the tensor to interact with
    inputs::Vector{Tuple{Int, Int}} # input messages 
    output::Tuple{Int, Int} # output message
end

struct BPPath
    bp_steps::Vector{BPStep} # steps of bp update, each step is a single bp update
    function BPPath(ansatz::TensorNetworkAnsatz{TA, TB}; random_order::Bool = true, seed::Int = 1234) where {TA, TB}
        g = ansatz.g
        Random.seed!(seed)
        bp_steps = BPStep[]
        # the different step are taken in random order
        outputs = Tuple{Int, Int}[]
        for e in edges(g)
            push!(outputs, (src(e), dst(e)))
            push!(outputs, (dst(e), src(e)))
        end

        # the steps are taken in random order
        random_order && shuffle!(outputs)

        for (s, d) in outputs
            inputs = Tuple{Int, Int}[]
            Tid = s
            for v in neighbors(g, s)
                if v != d
                    push!(inputs, (v, s))
                end
            end

            eincode = bp_eins(neighbors(g, s), inputs, (s, d))
            push!(bp_steps, BPStep(eincode, Tid, inputs, (s, d)))
        end

        new(bp_steps)
    end
end

function bp!(state::BPState, path::BPPath, ansatz::TensorNetworkAnsatz; max_iter::Int = 1000, atol::Float64 = 1e-6, damping::Float64 = 0.2, verbose::Bool = false)
    Ts = ansatz.site_tensors
    Tcs = conj.(ansatz.site_tensors)
    for iter in 1:max_iter
        error = bp_update!(state, path, Ts, Tcs, damping)
        verbose && @info "BP iteration $iter, error: $error"
        error < atol && break
    end
    nothing
end

function bp_update!(state::BPState, path::BPPath, Ts::Vector{TA}, Tcs::Vector{TB}, damping::Float64) where {TA, TB}

    error = 0.0
    for step in path.bp_steps
        eincode = step.eincode
        Tid = step.Tid
        inputs = step.inputs
        output = step.output

        T = Ts[Tid]
        Tc = Tcs[Tid]

        M_inputs = [state.messages[input] for input in inputs]

        # update the message tensor
        new_message = normalize_message!(eincode(T, Tc, M_inputs...))
        error = max(error, maximum(abs.(state.messages[output] - new_message)))
        state.messages[output] .*= damping
        state.messages[output] .+= (1 - damping) * new_message
    end
    return error
end