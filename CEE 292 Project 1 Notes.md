---
type: hw
course: CEE 292
authors:
  - Wonuola Abiodun
---
# 0 To-Do
- Create a sample generator
- Train the two MLPs
- Figure out better ways of sampling. Think about lower-frequency noise, more heavily weighting uniform loads, etc.

# 1 Notes from paper
We consider a pin-jointed bar system with $N$ nodes connected by $M$ bars. Each node experiences an external load vector. Some nodes are constrained, thus forming the boundary conditions $\vec b\in \mathbb R^L.$ After choosing stiffnesses $\vec q\in \mathbb R^m,$ form-finding has the goal to identify nodal positions that ensure equilibrium (no net residual force) without bending or torsion.

The residual of a single node can be calculated
$$
r_i(X; q) = \sum_{j\in \mathcal N(i)} [K(q)]_{ij}(x_j - x_i) - p_i(x_i)
$$
where $\mathcal N(i)$ are the neighbors of node $i,$ $K(q)\in \mathbb R^{N\times N}$ is the stiffness matrix as a function of $q.$ It can be solved using form-finding methods, but the map from stiffness to geometry, $X(q),$ is implicit and nonlinear—difficult to reason about directly.

A designer may have a target shape $\hat X.$ They want to find some $q^*$ that generates an equilibrium form that approximates $\hat X$ as closely as possible. The optimization problem formalized is
$$
q^* = \arg\min_{q\in \mathbb R^M} \mathcal L_{shape}(q),\qquad \mathcal L_{shape}(q) = \sum_{i = 1}^N\sum_{d = 1}^3\ || X(q)_{i, d} - \hat X_{i, d}||^p
$$
with $p > 0.$ Solving this is slow and computationally costly.

Model architecture:
- A neural encoder $\mathcal{E}_\phi$ maps the target shape $\hat X$ into $q$
- The FDM solver decodes the associated shape $X(q)$ subject to the boundary conditions $\vec b$

Since they use FDM, the resulting shape will be mechanically sound even if the neural network is inaccurate.

**Encoder:** a neural network with learnable parameters $\phi$that ingests the targets and projects them into a stiffness space—$\mathcal E_\phi: \mathbb{R}^{N\times 3}\to \mathbb{R}^M.$
	- They cast the problem as a pointwise matching task and use an MLP as the encoder.
	- Zero force densities cause problems, so they used strictly positive force densities, meaning no mixed-mode structures. They use a strictly positive activation on the output.
	- They apply a bias (so there’s a minimum force density) and $\pm$ (for compression vs tension) to the activated output of the final layer
# 2 Goals
- Train a PINN to predict force densities and geometry in parallel
	- OR train one geometry predictor and one force density predictor
	- OR use the paper’s method

- Figure out a way to address multiple topologies: differing numbers of edges, arrangements, support conditions

- Support more types of structures:
	- Mixed tension-compression structures
	- Structures with ribs

- Formulate a shape loss dependent on proximity rather than associating nodes by index
	- For example, it takes distances from all the nodes in the structure, but the distance to the closest node is weighted the most. The closer the closest node(s) is/are, the less the other distances matter.
	- Or, the distance from the surface of the structure but this might be more difficult

# 3 Ideas
>[!Important] Step 1: train a PINN to predict shape from force densities for a fixed topology (maybe supports can change). This sets a realistic baseline for the project because I think I can do this lmao


- Use tributary area iteration to tackle geometric nonlinearities

- Use my existing grammar to generate a bunch of shells. Get their data and do a PCA to find out about how many dimensions I should use for the latent space.
- Use a discontinuous activation function on the output, thus setting a minimum magnitude force density but allowing both positive and negative values

- For variable-length input, will probably need to use a **graph neural network:**
	- Nodes carry target position, support flag, load
	- Edges come from the connectivity
	- Output is an edge-level readout. After message passing, each edge’s embedding (formed from its two endpoint embeddings plus edge features) goes through a small MLP to predict q
	- Since equilibrium is global and plain message passing only sees about $k$ hops after $k$ layers, I’ll need eitehr many layers or a global context mechanism such as a virtual node connected to all nodes, a pooled global embedding added back to each node, or attention layers like a graph transformer

- PyTorch seems like my best bet for library since it can support variable inputs more easily and has the most mature graph neural network library.

## 3.1 Implementation
- Use `float64` for the solver, at least while validating, because FDM systems can be poorly conditioned when force densities span orders of magnitude
- Check gradients with `torch.autograd.gradcheck` on a tiny network before training anything
- Plan for sparsity later: dense solves are fine for a few hundred nodes. For bigger meshes, write a custom `torch.autograd.Function` that does a sparse factorization in the forward pass and uses the adjoint in the backward pass
- Batching: in PyG, a batch is one large disconnected graph. Its FDM matrix is block-diagonal, so either solve as one sparse system or loop over graphs in the batch. But don’t build a dense block-diagonal matrix
# 4 Other references
PINNs:
- Raissi et al 2019b
- Karniadakis et all 2021
- Lu et al 2021
- Bastek and Kochman 2023

Loss balancing for PINNs:
- Bischof and Kraus 2021
- Wang et al 2022