using BPGauge
using Test

@testset "bp_iterate" begin
    # Test the basic functionality of the bp_iterate function
    ansatz = TensorNetworkAnsatz()
    state = BPState{Float64}(Dict(), Dict())
    
    # Run a single iteration of BP
    new_state = bp_iterate(ansatz, state)
    
    @test new_state isa BPState{Float64}
    
    # Check if messages are updated correctly
    @test length(new_state.message_e2v) > 0
    @test length(new_state.message_v2e) > 0
end