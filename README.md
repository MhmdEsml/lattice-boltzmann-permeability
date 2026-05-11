# LBM Permeability Solver

A GPU/TPU-accelerated Lattice Boltzmann Method (LBM) solver for computing permeability of binary digital rock samples, built with [JAX](https://github.com/google/jax).

## Features

- **D3Q19 BGK** scheme with Guo et al. (2002) body-force forcing
- **Halfway link bounce-back** no-slip boundary conditions, periodic elsewhere
- **Darcy permeability** computed via `k = μ · U_Darcy / (ΔP/L)`
- **Single-sample**, **batched (vmap)**, and **multi-device (pmap + vmap)** APIs — runs on GPU and TPU
- Synthetic rock generators: straight square-duct channel and random sphere pack
- Analytical validation against the square-duct solution (Shah & London, 1978)

## Installation

```bash
git clone https://github.com/your-username/lbm-permeability.git
cd lbm-permeability
pip install -r requirements.txt
```

> **GPU users:** install the appropriate `jaxlib` CUDA wheel from the [JAX installation guide](https://github.com/google/jax#installation) before running `pip install -r requirements.txt`.
>
> **TPU users:** install the TPU-specific `jaxlib` wheel — see the [JAX TPU installation guide](https://github.com/google/jax#installation). No other changes are needed; JAX's `pmap` and `vmap` work identically on TPU pods.

## Quick Start

### Single sample

```python
import numpy as np
from lbm import compute_permeability
from lbm_utils import make_sphere_pack

rock = make_sphere_pack(shape=(64, 64, 64), n_spheres=20, r_mean=8.0)
print(f"Porosity: {np.mean(~rock):.3f}")

result = compute_permeability(rock, direction="x", max_iter=20_000, tol=1e-6)
print(f"Permeability : {result['permeability']:.4e} lu²")
print(f"Converged    : {result['converged']}  in {result['iterations']} iters")
print(f"Mach number  : {result['mach']:.4f}")
```

### Batch (single GPU/TPU)

```python
import numpy as np
from lbm import compute_permeability_batch
from lbm_utils import make_sphere_pack

rocks = np.stack([make_sphere_pack((64, 64, 64), seed=s) for s in range(8)])
results = compute_permeability_batch(rocks, direction="x", batch_size=4)
print(results["permeability"])
```

### Multi-device batch (GPU/TPU)

```python
from lbm import compute_permeability_batch_multi

results = compute_permeability_batch_multi(
    rocks,
    direction="x",
    batch_size_per_device=4,   # tune to fit per-device memory
)
```

### Analytical validation

```python
from lbm_utils import test_square_duct

passed = test_square_duct(shape=(32, 32, 32), wall_thickness=4, tol_pct=5.0)
```

## API Reference

### `lbm.py`

| Function | Description |
|---|---|
| `compute_permeability(rock, ...)` | Single sample, single device |
| `compute_permeability_batch(rocks, ...)` | Batch via `vmap` on one device |
| `compute_permeability_batch_multi(rocks, ...)` | Batch via `pmap` across GPU/TPU devices + `vmap` within |

All three return a dict with keys: `permeability`, `mean_velocity`, `porosity`, `converged`, `iterations`, `mach`, `u_field`, `rho_field`.

### `lbm_utils.py`

| Function | Description |
|---|---|
| `make_channel(shape, wall_thickness)` | Straight square-duct channel |
| `make_sphere_pack(shape, n_spheres, ...)` | Random overlapping sphere pack |
| `test_square_duct(shape, wall_thickness, ...)` | Validate against analytical solution |

## Device Support

JAX automatically detects available hardware. No code changes are needed when switching between CPU, GPU, and TPU — the same API works across all backends.

| Backend | Single sample | Batched (`vmap`) | Multi-device (`pmap`) |
|---|---|---|---|
| CPU | ✓ | ✓ | — |
| GPU (CUDA) | ✓ | ✓ | ✓ multi-GPU |
| TPU | ✓ | ✓ | ✓ TPU pod slices |

`compute_permeability_batch_multi` uses `jax.device_count()` to detect all available devices automatically. On a TPU pod, each TPU core is treated as a separate device.

## References


- Guo et al. (2002) *Phys. Rev. E* **65**, 046308
- Succi (2001) *The Lattice Boltzmann Equation*, Oxford UP
- Qian et al. (1992) *Europhys. Lett.* **17**(6), 479
- Shah & London (1978) *Laminar Flow Forced Convection in Ducts*
