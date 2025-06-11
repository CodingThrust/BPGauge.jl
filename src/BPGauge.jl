module BPGauge

using LinearAlgebra, Graphs
using OMEinsum
using Yao
using GenericMessagePassing # provide the `bp` backend

export TensorNetworkAnsatz

include("eins.jl")

# define and construct the network
include("ansatz.jl")
include("rydberg.jl")

# bp on the tensor network ansatz
include("bp.jl")
include("gauge.jl")

end
