# BPGauge

[![Build Status](https://github.com/CodingThrust/BPGauge.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/CodingThrust/BPGauge.jl/actions/workflows/CI.yml?query=branch%3Amain)
<!-- [![Coverage](https://codecov.io/gh/CodingThrust/BPGauge.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/CodingThrust/BPGauge.jl) -->

## Introduction

This package provides a gauge transformation for graph tensor network states.

A simple example is to consider a cycle graph:
```julia
using BPGauge, Graphs

g = path_graph(100)
add_edge!(g, 1, 100)

# generate a random state
tn = random_state(g, d_virtual = 4)
normalize_state!(tn)

# generate a BP state, and do BP until convergence
bp_state = BPState(tn)
bp_path = BPPath(tn)
bp!(bp_state, bp_path, tn, atol = 1e-8)

# gauge transform the state
gauge!(tn, bp_state)
```

After gauge transformation, the state should satisfy Eq. 18 of "Gauging tensor networks with belief propagation", which is checked as follows:
```julia
julia> using OMEinsum

# the guage tensor between site 1 and 2
julia> G = tn.gauge_tensors[tn.gauge_tensors_map[(1, 2)]]
4×4 Matrix{Float64}:
 0.24286  0.0         0.0          0.0
 0.0      0.00378541  0.0          0.0
 0.0      0.0         0.000261187  0.0
 0.0      0.0         0.0          9.42455e-5

# the site tensor at site 2
julia> T = tn.site_tensors[2]
4×4×2 Array{ComplexF64, 3}:
[:, :, 1] =
   -1.99291+1.65377im    -2.06772+1.22581im  -0.945118+0.217569im  0.134128-0.306926im
    2.04773+1.5168im      14.9128-10.1478im    16.8693+45.1754im    146.295-116.843im
    -0.5367+0.153043im   -117.962+2.32997im    -16.307+111.833im   -593.359+194.269im
 -0.0990699-0.0472216im   234.743+85.4164im    -1593.4-578.339im   -3710.62+940.987im

[:, :, 2] =
  -2.07123+1.60435im     2.10162-1.14093im  0.914902-0.210208im  -0.152577+0.25503im
  -1.97444-1.57188im     2.86458-86.7691im  -69.2749+182.169im    -20.1493-26.9031im
  0.522798-0.147941im   -259.707-229.664im   481.794+969.353im    -387.999+839.232im
 0.0697514+0.0437448im  -114.465-47.7015im  -1001.09-181.03im     -2431.68+548.58im

# the left-hand side of Eq. 18
julia> L = ein"ij, jk, jln, kmn -> lm"(G, G, T, conj(T));

# the right-hand side of Eq. 18, approximately equal to the identity matrix
julia> L ./ L[1, 1]
4×4 Matrix{ComplexF64}:
         1.0+0.0im         1.26014e-8+1.43958e-8im  -2.10249e-8-4.83406e-8im    7.7485e-8-1.30709e-8im
  1.26014e-8-1.43958e-8im         1.0+3.4123e-17im    1.1739e-8+6.43834e-9im   5.49462e-8-1.45059e-8im
 -2.10249e-8+4.83406e-8im   1.1739e-8-6.43834e-9im          1.0+3.49397e-17im  2.19632e-8+5.61849e-8im
   7.7485e-8+1.30709e-8im  5.49462e-8+1.45059e-8im   2.19632e-8-5.61849e-8im          1.0+4.00859e-17im
```

## TODO

1. Simple update based on the gauge tensor network, including applying two qubit gates and truncation.
2. Ill-conditioned operations: https://github.com/CodingThrust/BPGauge.jl/issues/15.
