Such report is summary targeted for the Belief Propagation and Single Update algorithm in [quantum evolution dynamics training camp](https://github.com/CodingThrust/QuantumEvolutionTrainingCamp), which is a new proposed algorithm for 2D quantum system dynamics. This report is structured as 2D system dynamics and its difficulties, 2D tensor network algorithm (especially PEPS), system analyzing for Rydberg atoms array and kicked Ising dynamics, finally our BP algorithm and technical details.

# 2D system dynamics

When we deal with the dynamics of two dimensional or higher dimension, the algorithms have to overcome the obstacles such as: **degrees of freedom increasing exponentially**, **more quick growth of entanglement entropy**, and **computational accuracy and efficiency**. Several algorithms are listed below for comparing.

## *Exact Diagonalization, ED* $\sim O(D^3)$
   
The principle is basically $|\psi(t)\rangle=e^{−iHt}|\psi(0)\rangle$, no matter directly diagonalizing the Hamiltonian, or do Krylov dynamics, the Hilbert space dimension scales exponentially $dim(H)\sim 2^N$. So the maximal matirx size is fixed due to the classical capacity. But the former is the most accurate numerical methods and you can obtain all information of system. As long as you can obtain eigenstates despite its $O(D^3)$ complexity, you can do **exact dynamics for any long time**. So it can be used as benchmark for other outcome.

More complex geometry and property will reduce more the dimension of system by introducing symmetry (translation, rotation, U(1) symmetry, etc.). For example in [^QMBSPRB] we can completely diagonalize 1D PXP model at size $L=32, D=77436$, then enlarging to $L=36, D=467160$ to extract a subset of eigenstates by shift invert (then do Krylov dynamics). In 2D, people can easily compute $6 \times 6$ square lattice, then nearly enlarge by sparse matrix to $7 \times 7$. For other models, people have been scales up to[^Leshouches] 40 spins square lattice, 39 sites triangular, 42 sites Honeycomb lattice, 48 sites kagome lattice[^LauchliKagome] and 64 spins or more in elevated magnetization sectors. This can be visualize below:

$D=2^N \xrightarrow{\text{Constraint}} \alpha^N  \xrightarrow{\text{Particle Number}}\alpha^{N-1}/k\xrightarrow{\text{translation}}\alpha^{N-1}/kN\xrightarrow{\text{spin flip/inversion}}\alpha^{N-2}/kN$.

where $\alpha \sim 1.618, 1.502$ for Fibonacci path_graph, hard core square lattice, $k \sim 5$. So for fully ED, $D \sim 10^6$. When doing dynamics, Krylov subspace $K_m​=\text{span}\{v,Hv,H^2v,⋯,H^{m−1}v\}$ is usually incorporated, which can scale up to $N=32$ even without anyon symmetry ($D \sim 10^{10}, m \sim 10^4$,  as long as the locality of interaction ensuring sparse matrix). In such basis, the Its complexity decreases to $O(m^2D)$


## *Tensor Networks, TN*

(a) In one dimension, MPS (or to see)
(b) Projected entangled pairs state（PEPS）Unlike ED prefer PBC, loop structure will fail due to circle.

## Others 
### *Quantum Monte Carlo, QMC* [^Dowling]
utilizing $⟨O⟩=Tr[e^{−βH}O]$， Path integral Monte Carlo: for finite temperature dynamics, inchworm monte carlo. However, when it comes to real-time calculations (i.e. dynamics), path integral QMC methods become difficult because of sign or phase
problems. The stochastic-gauge representation is a method of mapping the equation of motion for the quantum mechanical density operator onto a set of equivalent stochastic
differential equations. One of the stochastic variables is termed the “weight”, and
its magnitude is related to the importance of the stochastic trajectory. We investigate the use of Monte Carlo algorithms to improve the sampling of the weighted
trajectories and thus reduce sampling error in a simulation of quantum dynamics. The method can be applied to calculations in real time, as well as imaginary
time for which Monte Carlo algorithms are more-commonly used. The method is
applicable when the weight is guaranteed to be real, and we demonstrate how to
ensure this is the case. Examples are given for the anharmonic oscillator, where
large improvements over stochastic sampling are observe

An alternative approach is provided by phase-space representations [8–10],
which can be used to map quantum dynamics to a set of equivalent stochastic
differential equations. The number of phase-phase equations scale polynomially with the number of modes, allowing computationally tractable simulations. Phase-space methods have proved useful in the past for simulating quantum dynamics, particularly in the field of quantum optics [11–13] A natural
extension of these techniques is to the field of degenerate quantum gases, where
the interacting particles are atoms or molecules rather than photons [14,15].
Recently it has been discovered that Fermi gases can be treated with related
techniques [16,17].
The mapping of a quantum problem to phase-space equations is far from
unique. This nonuniqueness can be exploited to tailor the form the stochastic
equation without affecting the physical, ensemble result. The different choices
correspond to different “stochastic gauges”. In this paper, we use this freedom to generate stochastic equations with real weights, which we then sample
with Monte Carlo techniques. The real weights avoid the sign or phase problem encountered in other QMC approaches to quantum dynamics. Since the
stochastic gauge method is a relatively new technique, we choose to focus here
on an especially simple case with known exact solutions, in order to clarify
the problems and advantages of this real weight approach.

### Neural network quantum state（NQS）：
variational optimizing the wavefunction parameters by NN $\psi_{\theta}​(s_1​, \cdots,s_N​)$ to $min_{\theta}​||i\partial_t ​\psi−H\psi||$.


# 2D tensor network algorithm (especially PEPS)

# Rydberg atoms array and kicked Ising dynamics
The articles on 2D Rydberg atom arrays related to our algorithm: (Because thermal density matrix is effective to time evolution's TN's imaginary rotation(But exp(-Real) will always coverge, while exp(i H \delta t) may not, that's why it's difficult.))

- [Exploring the Finite-Temperature Behavior of Rydberg Atom Arrays: A Tensor
Network Approach](https://arxiv.org/pdf/2503.18413), using "Because the Hamiltonian involves long-range interactions, the imaginary time evolution with a simple update method [31] is not suitable for optimization. Instead, we optimize the wave functions using a stochastic reconfiguration (SR) method", thermal density matrix.
- [Finite-temperature Rydberg arrays: quantum phases and entanglement characterization](https://arxiv.org/html/2405.18477v1#S1), using MPS-MPO contraction method, Tree Tensor Operator density
matrix, imaginary time evolution to obtain thermal density matrix
- [Entanglement in the quantum phases of an unfrustrated Rydberg atom array](https://www.nature.com/articles/s41467-023-41166-0), "The second is a representation of long-range interactions[32](https://www.nature.com/articles/s41467-023-41166-0#ref-CR32) compatible with projected entangled pair states (PEPS)[33](https://www.nature.com/articles/s41467-023-41166-0#ref-CR33),[34](https://www.nature.com/articles/s41467-023-41166-0#ref-CR34),[35](https://www.nature.com/articles/s41467-023-41166-0#ref-CR35),[36](https://www.nature.com/articles/s41467-023-41166-0#ref-CR36). With this, we use PEPS to find the ground states of a Hamiltonian with long-range interactions for the first time"
  
## Rydberg atoms array 
In 2D, people usually choose[^Hsiehscar20]

$$
H = \sum_{i=1}^N X_i (\Pi_{|i-j|=1} P_j)
$$
## kicked Ising dynamics

# BP algorithm 

$$|\Psi_{\mathrm{GS}}\rangle=\lim_{\tau\to\infty}\frac{e^{-\tau H}|\Psi_0\rangle}{||e^{-\tau H}|\Psi_0\rangle||}.$$

the Hamiltonian is a translationally invariant su st-neighbour terms, $H=\sum_{\langle i,j\rangle}H_{i,j}$, one can app the ITE operator for infnitesimal time steps $\delta$ ng a Suzuki-Trotter decomposition, i.e.,


$$e^{-\delta\tau H}\approx\prod_{\langle i,j\rangle}U_{i,j}=\prod_{\langle i,j\rangle}e^{-\delta\tau H_{i,j}}.$$


The method in ITensorNetwork.jl only support two site gate, how to promote to PXP like model, three or four body term? (Or is it neccesary to do this?)

![ITensorNetwork.jl](https://github.com/user-attachments/assets/833d5c22-4274-4003-b223-34117857bd25)

Here has a algorithm I take from [Fine-Grained Tensor Network Methods
Supplemental Material](https://arxiv.org/pdf/1911.04882), which maybe helpful?

- [Tensor networks contraction and the belief propagation algorithm](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.023073), seeming the first paper put up with BP and SU.

- [Universal tensor-network algorithm for any infinite lattice](https://journals.aps.org/prb/pdf/10.1103/PhysRevB.99.195105), using SU, which has many fruitful outcome as benchmark, can also see its appendix, talking about the gauge fixing.

<img width="872" alt="Image" src="https://github.com/user-attachments/assets/df2cf6ed-da28-4320-9f28-335b383a5cbb" />

[^Leshouches]: Les houch notes for Exact Diagonalizaton. https://indico.ictp.it/event/a14246/session/31/contribution/51/material/0/0.pdf
[^QMBSPRB]: Quantum scarred eigenstates in a Rydberg atom path_graph: Entanglement, breakdown of thermalization, and stability to perturbations. https://journals.aps.org/prb/pdf/10.1103/PhysRevB.98.155134
[^LauchliKagome]: S = 1/2 kagome Heisenberg antiferromagnet revisited https://journals.aps.org/prb/pdf/10.1103/PhysRevB.100.155142
[^Dowling]: Monte Carlo techniques for real-time
quantum dynamics https://arxiv.org/pdf/quant-ph/0507003
[^Hsiehscar20]: Quantum many-body scar states in two-dimensional Rydberg atom arrays https://journals.aps.org/prb/pdf/10.1103/PhysRevB.101.220304