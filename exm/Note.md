
Such report is summary targeted for the Belief Propagation and Single Update algorithm in [quantum evolution dynamics training camp](https://github.com/CodingThrust/QuantumEvolutionTrainingCamp), which is a new proposed algorithm for 2D quantum system dynamics. This report is structured as 2D system dynamics and its difficulties, 2D tensor network algorithm (especially PEPS), system analyzing for Rydberg atoms array and kicked Ising dynamics, finally our BP algorithm and technical details.

# 2D system dynamics

Usually in quantum many body physics, people tends to discuss observables such as energy, magnetization these static (equilibrium) quantity, rather than their dynamics or how they change with time (the non-equilibrium quantity). However, the development of quantum simulation and numerical methods makes people realize that in dynamics, quantum matter could exhibit more phases during the evolution, such as Eigenstates Thermalization Hypothesis (ETH), Many body localization (MBL), discrete time crystal quantum many body scar and various dynamical phase transition, in which dynamics is deeply related to how statistics mechanics emerge from quantum mechanics. On the other hand, in the context of quantum algorithm and quantum circuit which is equivalent to many body dynamics. We could analyze the complexity of circuit and algorithm to be whether polynomial or not to determine the difficulty to do dynamics. Theoretical methods to do dynamics contain random circuits, dual-unitary, clifford circuits (basically free fermions). While turning to numerical methods, especially dealing with the dynamics of two dimensional or higher dimension, the algorithms have to overcome the obstacles such as: **degrees of freedom increasing exponentially**, **complex geometry and property**, **more quick growth of entanglement entropy**, and **computational accuracy and efficiency**. The main reason of dynamics difficulty lies at that the initial state will nearly visit each part of Hilbert space (real time evolution, imaginary part at exp), in contrast ground property or finite temperature property (imaginary time evolution, real part at exp). Several algorithms are listed below for comparing. 

## *Exact Diagonalization, ED* $\sim O(D^3)$
   
The principle is basically $|\psi(t)\rangle=e^{−iHt}|\psi(0)\rangle$, no matter directly diagonalizing the Hamiltonian, or do Krylov dynamics, the generic Hilbert space dimension scales exponentially $dim(H)\sim 2^N$. More complex geometry and property will reduce more the dimension of system by choosing symmetry sector (translation, rotation, U(1) symmetry, etc.) or ruling out the constrained basis (Fibonacci chain, hard core square lattice). For example in [^QMBSPRB] we can completely diagonalize 1D PXP model at size $L=32, D=77436$, then enlarging to $L=36, D=467160$ to extract high energy eigenstates by shift invert. In 2D, people can easily compute $6 \times 6$[^Hsiehscar20][^Stabscar] square lattice, then nearly enlarge by sparse matrix to $7 \times 7$. For other models, people have been scales up to[^Leshouches] 40 spins square lattice, 39 sites triangular, 42 sites Honeycomb lattice, 48 sites kagome lattice[^LauchliKagome] and 64 spins or more in elevated magnetization sectors. This procedure can be visualized below:

$D=2^N \xrightarrow{\text{Constraint}} \alpha^N  \xrightarrow{\text{Particle Number}}\alpha^{N-1}/k\xrightarrow{\text{translation}}\alpha^{N-1}/kN\xrightarrow{\text{spin flip/inversion}}\alpha^{N-2}/kN$.

where $\alpha \sim 1.618, 1.502$. For Fibonacci chain, hard core square lattice, $k \sim 7$. 

So the maximal system size we can compute depends on the maximal matirx size, which is fixed on the classical devices capacity. Fully ED, $D \sim 10^6$, is the most accurate numerical methods and you can obtain all information of system. As long as you can obtain eigenstates despite its $O(D^3)$ complexity, you can do **exact dynamics for any long time**. So it can be used as benchmark for other methods.When doing dynamics, Krylov subspace $K_m​=\text{span}\{v,Hv,H^2v,⋯,H^{m−1}v\}$ is usually incorporated, which can scale up to $N=32$ even without anyon symmetry ($D \sim 10^{10}, m \sim 10^4$,  as long as the locality of interaction ensuring sparse matrix). In such basis, the Its complexity decreases to $O(m^2D)$

## *Tensor Networks, TN*

Usual tensor network structure such as
- In one dimension, MPS
- In 2D, Projected entangled pairs state（PEPS）Unlike ED prefer PBC, loop structure will fail due to gauging problem.

Here we focus on PEPS, whether we want to find the best (ground) state:
1. iterative optimization of individual tensors (energy minimization)
2. imaginary time evolution

or do dynamics (real time evolution), we will always encounter:

**Problems:** contracting the PEPS, no matter how we contract, we will get intermediate tensors with $O(L)$ legs number of coefficients $D^{2L}$. Exponentially increasing with $L$! 

## Controlled approximate contraction scheme
Exact contraction of an PEPS is exponentially hard! So here are some schemes:

1. MPS-MPO-based approaches
   - SVD
   - variational optimization
   - zip-up algorithm
2. Corner transfer matrix method (CTM) 
   ‣Environment tensors account for infinite system around a bulk site
   ‣ CTM: Compute environment in an iterative way
   ‣ Accuracy can be systematically controlled with $\chi$
3. TRG:
   ★ Contract PEPS with periodic boundary conditions
   ★ Finite or infinite systems
   ★ Related schemes: SRG, HOTRG, HOSRG.

## Single update:
simple update (SVD)
★ “local” update like in TEBD
★ Cheap, but not optimal
(e.g. overestimates magnetization
in S=1/2 Heisenberg model)

## full update
Jordan et al, PRL 101 (2008)
★ Take the full wave function into
account for truncation
★ optimal, but computationally more
expensive
★ Fast-full update [Phien et al, PRB 92 (2015)]

## Cluster update Wang, Verstraete, arXiv:1110.4362 (2011)

### System size
Can enlarge to $N=300$
### Time scale
But only precise to time $t \sim v/L$
## Others 
### *Quantum Monte Carlo, QMC* [^Dowling]
The main ideas is utilizing $⟨O⟩=Tr[e^{−βH}O], ⟨O(t)⟩=Tr[e^{−iHt}O]$, usual containing Path integral Monte Carlo (PIMC), inchworm monte carlo. They are powerful for finite temperature dynamics (i.e. imaginary time dynamics). However, when it comes to real-time dynamics, path integral QMC methods become difficult because of the dynamical sign or phase problems, with sign problem of fermions. There's a phase-space method called the stochastic-gauge representation, which maps the equation of motion for the quantum mechanical density operator onto a set of equivalent stochastic differential equations. QMC 

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
## Kicked Ising dynamics

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
[^QMBSPRB]: Quantum scarred eigenstates in a Rydberg atom chain: Entanglement, breakdown of thermalization, and stability to perturbations. https://journals.aps.org/prb/pdf/10.1103/PhysRevB.98.155134
[^LauchliKagome]: S = 1/2 kagome Heisenberg antiferromagnet revisited https://journals.aps.org/prb/pdf/10.1103/PhysRevB.100.155142
[^Dowling]: Monte Carlo techniques for real-time
quantum dynamics https://arxiv.org/pdf/quant-ph/0507003
[^Hsiehscar20]: Quantum many-body scar states in two-dimensional Rydberg atom arrays https://journals.aps.org/prb/pdf/10.1103/PhysRevB.101.220304
[^Stabscar]: Stabilizing two-dimensional quantum scars by deformation and synchronization https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.022065