using BPGauge
using Test

@testset "BPGauge.jl" begin
    # Write your tests here.
    include("./gauge.jl")
    include("./bp.jl")
end
