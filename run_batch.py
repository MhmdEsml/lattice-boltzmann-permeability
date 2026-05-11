"""
examples/run_batch.py
=====================
Batch example: compute permeability for multiple sphere packs.
Uses vmap (single GPU) or pmap+vmap (multi-GPU).
"""

import numpy as np
from lbm import compute_permeability_batch, compute_permeability_batch_multi
from lbm_utils import make_sphere_pack

# ── Build a small batch of rocks ───────────────────────────────────────────────
B = 8
rocks = np.stack([
    make_sphere_pack(shape=(64, 64, 64), n_spheres=20, r_mean=8.0, seed=s)
    for s in range(B)
])
print(f"Batch shape : {rocks.shape}  ({B} samples)")

# ── Single-device batch (vmap) ─────────────────────────────────────────────────
print("\n── Single-device batch ──────────────────────────────────")
results = compute_permeability_batch(
    rocks,
    direction="x",
    batch_size=4,       # tune to GPU memory
    max_iter=20_000,
    tol=1e-6,
    verbose=True,
)

print("\nPermeabilities (lu²):")
for i, k in enumerate(results["permeability"]):
    print(f"  Sample {i}: {k:.4e}  phi={results['porosity'][i]:.3f}  "
          f"converged={results['converged'][i]}")

# ── Multi-device batch (pmap + vmap) — uncomment if multiple GPUs available ────
# print("\n── Multi-device batch ──────────────────────────────────")
# results_multi = compute_permeability_batch_multi(
#     rocks,
#     direction="x",
#     batch_size_per_device=2,
#     max_iter=20_000,
#     tol=1e-6,
#     verbose=True,
# )
