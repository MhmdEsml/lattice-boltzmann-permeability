"""
lbm_utils.py
============
Synthetic rock generators and validation utilities for lbm.py.

Nothing in this file belongs in the core solver.  Import from here when
building datasets, running tests, or doing benchmarks.

Functions
---------
make_channel(shape, wall_thickness)        -- straight square-duct channel
make_sphere_pack(shape, n_spheres, ...)    -- random overlapping sphere pack
test_square_duct(shape, wall_thickness, tol_pct, verbose)  -- analytical validation
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ── Synthetic rock generators ──────────────────────────────────────────────────

def make_channel(
    shape: Tuple[int, int, int] = (64, 32, 32),
    wall_thickness: int = 4,
) -> np.ndarray:
    """
    Straight square-duct channel with solid walls in the y and z directions.

    Analytical permeability (superficial-velocity / Darcy convention):
        k = φ · h² / 28.45
    where h = shape[1] - 2*wall_thickness  (open side length)
          φ = (h/shape[1]) · (h/shape[2])  (porosity)

    Reference: Shah & London (1978) Laminar Flow Forced Convection in Ducts.

    Parameters
    ----------
    shape          : (Nx, Ny, Nz) grid dimensions
    wall_thickness : solid wall thickness in lattice units (each side)

    Returns
    -------
    solid : (Nx, Ny, Nz) bool array  — True = solid
    """
    solid = np.ones(shape, dtype=bool)
    solid[:, wall_thickness:-wall_thickness,
             wall_thickness:-wall_thickness] = False
    return solid


def make_sphere_pack(
    shape: Tuple[int, int, int] = (64, 64, 64),
    n_spheres: int  = 20,
    r_mean: float   = 8.0,
    r_std: float    = 1.5,
    seed: int       = 42,
) -> np.ndarray:
    """
    Random sphere-pack binary rock with periodic wrapping and overlaps allowed.

    The actual porosity depends on sphere count, size, and overlaps.
    Call  np.mean(~result)  to check it before running the solver.

    Parameters
    ----------
    shape     : (Nx, Ny, Nz) grid dimensions
    n_spheres : number of spheres to place
    r_mean    : mean sphere radius in lattice units
    r_std     : standard deviation of sphere radius
    seed      : RNG seed for reproducibility

    Returns
    -------
    solid : (Nx, Ny, Nz) bool array  — True = solid grain
    """
    rng  = np.random.default_rng(seed)
    Nx, Ny, Nz = shape
    solid = np.zeros(shape, dtype=bool)

    x = np.arange(Nx)[:, None, None]
    y = np.arange(Ny)[None, :, None]
    z = np.arange(Nz)[None, None, :]

    for _ in range(n_spheres):
        cx = rng.uniform(0, Nx)
        cy = rng.uniform(0, Ny)
        cz = rng.uniform(0, Nz)
        r  = max(1.0, rng.normal(r_mean, r_std))

        # Periodic distance
        dx = np.minimum(np.abs(x - cx), Nx - np.abs(x - cx))
        dy = np.minimum(np.abs(y - cy), Ny - np.abs(y - cy))
        dz = np.minimum(np.abs(z - cz), Nz - np.abs(z - cz))
        solid |= (dx**2 + dy**2 + dz**2) <= r**2

    return solid


# ── Analytical validation ──────────────────────────────────────────────────────

def test_square_duct(
    shape: Tuple[int, int, int] = (32, 32, 32),
    wall_thickness: int = 4,
    tol_pct: float = 5.0,
    verbose: bool  = True,
) -> bool:
    """
    Validate lbm.compute_permeability against the square-duct analytical solution.

    The solver uses the Darcy (superficial) velocity convention:
        U_Darcy = φ · U_pore
    so the correct reference is:
        k_ref = φ · h² / 28.45

    A result within tol_pct % of k_ref is considered a pass.

    Parameters
    ----------
    shape          : (Nx, Ny, Nz)
    wall_thickness : solid wall thickness in lattice units
    tol_pct        : pass/fail threshold in percent (default 5 %)
    verbose        : print comparison table

    Returns
    -------
    passed : bool
    """
    # Import here to avoid circular dependency if lbm imports lbm_utils
    from lbm import compute_permeability

    rock = make_channel(shape=shape, wall_thickness=wall_thickness)
    h    = shape[1] - 2 * wall_thickness
    phi  = (h / shape[1]) * (h / shape[2])
    k_ref = phi * h**2 / 28.45

    result  = compute_permeability(rock, direction="x",
                                   max_iter=50_000, tol=1e-7, verbose=verbose)
    k_sim   = result["permeability"]
    rel_err = abs(k_sim - k_ref) / k_ref * 100.0

    if verbose:
        print(f"\n── Square-duct validation ──────────────────────────")
        print(f"  Grid                    : {shape}")
        print(f"  Open side h             : {h} lu")
        print(f"  Porosity φ              : {phi:.4f}")
        print(f"  k_ref  (φ · h²/28.45)  : {k_ref:.4e} lu²")
        print(f"  k_sim                   : {k_sim:.4e} lu²")
        print(f"  Relative error          : {rel_err:.2f} %  "
              f"(tolerance {tol_pct:.1f} %)")

    passed = rel_err < tol_pct
    if verbose:
        print(f"  Result : {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed