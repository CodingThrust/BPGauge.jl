using BPGauge
using Test  

@testset "BPGauge.jl" begin
    # Test the basic functionality of the BPGauge module
    @testset "TensorNetworkAnsatz" begin
        # Test the construction of the TensorNetworkAnsatz
        ansatz = TensorNetworkAnsatz()
        @test ansatz isa TensorNetworkAnsatz
    end

    @testset "Rydberg Hamiltonian" begin
        # Test the Rydberg Hamiltonian construction
        hamiltonian = RydbergHamiltonian()
        @test hamiltonian isa RydbergHamiltonian
    end

    @testset "BP Algorithm" begin
        # Test the BP algorithm on the tensor network ansatz
        result = run_bp_algorithm(TensorNetworkAnsatz())
        @test result isa SomeExpectedType  # Replace with actual expected type
    end

    @testset "Gauge Transformations" begin
        # Test gauge transformations on the tensor network ansatz
        transformed_ansatz = apply_gauge_transformation(TensorNetworkAnsatz())
        @test transformed_ansatz isa TensorNetworkAnsatz  # Replace with actual expected type if different
    end
end