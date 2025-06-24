module BPGauge

using LinearAlgebra, Graphs, Random
using OMEinsum
using Yao
using GenericMessagePassing # provide the `bp` backend

export TensorNetworkAnsatz
export BPState, BPPath, BPStep

export zero_state
export bp!, bp_update!
export absorb!

include("utils.jl")

# define and construct the network
include("ansatz.jl")
include("rydberg.jl")

# bp on the tensor network ansatz
include("bp.jl")
include("gauge.jl")

# operations about gauging``
include("guage.jl")

end
