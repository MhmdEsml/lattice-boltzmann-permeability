"""
examples/run_single.py
======================
Minimal example: compute permeability of a random sphere pack.
"""

import numpy as np
from lbm import compute_permeability
from lbm_utils import make_sphere_pack, test_square_duct

# ── 1. Build a synthetic rock ──────────────────────────────────────────────────
rock = make_sphere_pack(shape=(64, 64, 64), n_spheres=20, r_mean=8.0, seed=42)
print(f"Porosity : {np.mean(~rock):.3f}")

# ── 2. Compute permeability ────────────────────────────────────────────────────
result = compute_permeability(
    rock,
    direction="x",
    nu=1.0 / 6.0,
    force_mag=1e-4,
    max_iter=20_000,
    tol=1e-6,
    verbose=True,
)

print(f"\nPermeability : {result['permeability']:.4e} lu²")
print(f"Converged    : {result['converged']}  in {result['iterations']} iters")
print(f"Mach number  : {result['mach']:.4f}")

# ── 3. Analytical validation (square duct) ─────────────────────────────────────
print("\nRunning square-duct validation …")
passed = test_square_duct(shape=(32, 32, 32), wall_thickness=4, tol_pct=5.0)
print("Validation PASSED" if passed else "Validation FAILED")
