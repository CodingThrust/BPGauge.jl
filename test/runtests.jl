using BPGauge
using Test

@testset "utils" begin
    include("utils.jl")
end

@testset "ansatz" begin
    include("ansatz.jl")
end

@testset "bp" begin
    include("bp.jl")
end

@testset "gauge" begin
    include("gauge.jl")
end
