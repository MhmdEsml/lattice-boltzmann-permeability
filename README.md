# lattice-boltzmann-permeability
A JAX-based Lattice Boltzmann solver for computing absolute permeability of binary digital rock samples. Implements the D3Q19 BGK scheme with Guo body-force forcing and halfway bounce-back boundaries. Supports single-sample, batched, and multi-device workflows via JAX's vmap and pmap, running on CPU, GPU, and TPU with no code changes.
