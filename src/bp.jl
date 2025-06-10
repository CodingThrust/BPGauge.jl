# bp on the TensorNetworkAnsatz

struct BPState{TA}
    message_e2v::Dict{Tuple{Int, Int}, TA}
    message_v2e::Dict{Tuple{Int, Int}, TA}
end