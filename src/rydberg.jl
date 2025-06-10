# initialize the state for rydberg system, all sites are in the |0> state
function zero_state(g::SimpleGraph{Int}; d_virtual::Int = 1)
    phi = TensorNetworkAnsatz(g, d_virtual, 2)
    # set all to the |0> state
    for tensor in phi.site_tensors
        tensor[1] = ComplexF64(1.0)
    end
    return phi
end