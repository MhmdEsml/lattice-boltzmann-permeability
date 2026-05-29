"""
lbm.py
======
Lattice Boltzmann Method for permeability calculation of binary digital rocks.

Scheme  : D3Q19, BGK single-relaxation-time
Forcing : Guo et al. (2002) body-force scheme
BC      : Halfway link bounce-back (no-slip), periodic elsewhere
Perm.   : Darcy's law  k = μ · U_Darcy / (ΔP/L)

Public API
----------
compute_permeability(rock, ...)               -- single sample, single device
compute_permeability_batch(rocks, ...)        -- vmap within one device
compute_permeability_batch_multi(rocks, ...)  -- pmap across devices + vmap within

Device strategy
---------------
- 1 device  : compute_permeability_batch  (vmap only)
- N devices : compute_permeability_batch_multi  (pmap over devices, vmap within)

  compute_permeability_batch_multi automatically detects available devices.
  batch_size_per_device controls how many samples run in parallel on each device.
  Total parallelism = n_devices × batch_size_per_device.

  Example — 4 devices, batch_size_per_device=8, 100 samples:
    Round 1: devices 0-3 each run 8 samples (32 total) via pmap+vmap
    Round 2: same
    Round 3: same
    Round 4: remaining 4 samples (padded to 32, padding stripped after)

References
----------
Guo et al. (2002)  Phys. Rev. E 65, 046308
Succi (2001)       The Lattice Boltzmann Equation, Oxford UP
Qian et al. (1992) Europhys. Lett. 17(6), 479
"""

from __future__ import annotations

import functools
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

# ── D3Q19 lattice constants ────────────────────────────────────────────────────

_E_np = np.array([
    [ 0,  0,  0],
    [ 1,  0,  0], [-1,  0,  0],
    [ 0,  1,  0], [ 0, -1,  0],
    [ 0,  0,  1], [ 0,  0, -1],
    [ 1,  1,  0], [-1, -1,  0],
    [ 1, -1,  0], [-1,  1,  0],
    [ 1,  0,  1], [-1,  0, -1],
    [ 1,  0, -1], [-1,  0,  1],
    [ 0,  1,  1], [ 0, -1, -1],
    [ 0,  1, -1], [ 0, -1,  1],
], dtype=np.int32)                          # (19, 3)

_E_f32     = _E_np.astype(np.float32)
E          = jnp.array(_E_f32)              # (19, 3)  device constant

W = jnp.array([
    1/3,
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36,
], dtype=jnp.float32)                       # (19,)

Q   = 19
cs2 = jnp.float32(1.0 / 3.0)

_OPPOSITE_np = np.array(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17],
    dtype=np.int32)
OPPOSITE = jnp.array(_OPPOSITE_np)         # (19,)

# ── Neighbour table ────────────────────────────────────────────────────────────

def build_neighbour_table(Nx: int, Ny: int, Nz: int) -> jnp.ndarray:
    """
    Precompute pull-streaming neighbour indices (CPU, transferred once).

    nb[n, q] = flat index of the upstream node for direction q at node n.
    i.e.  nb[i,j,k, q] = flat( (i-ex)%Nx, (j-ey)%Ny, (k-ez)%Nz )

    Returns
    -------
    jnp.ndarray  shape (Nx*Ny*Nz, 19)  dtype int32
    """
    ix, iy, iz = (np.arange(n, dtype=np.int32) for n in (Nx, Ny, Nz))
    II, JJ, KK = np.meshgrid(ix, iy, iz, indexing='ij')   # (Nx,Ny,Nz) each

    nb = np.empty((Nx * Ny * Nz, Q), dtype=np.int32)
    for q in range(Q):
        ex, ey, ez = int(_E_np[q, 0]), int(_E_np[q, 1]), int(_E_np[q, 2])
        nb[:, q] = (
            ((II - ex) % Nx) * Ny * Nz +
            ((JJ - ey) % Ny) * Nz +
            ((KK - ez) % Nz)
        ).reshape(-1)

    return jnp.array(nb)   # (N, 19) on device

# ── Fused collision (macroscopic + equilibrium + Guo + BGK) ───────────────────

def _collide(f: jnp.ndarray, tau: float,
             F: jnp.ndarray, solid: jnp.ndarray) -> jnp.ndarray:
    """
    Parameters  (all flat over spatial nodes)
    ----------
    f      : (N, 19)
    tau    : scalar float
    F      : (3,)
    solid  : (N,)  bool

    Returns
    -------
    f_post : (N, 19)
    """
    rho      = f.sum(axis=-1)                                    # (N,)
    rho_safe = jnp.where(rho > 0.0, rho, jnp.float32(1.0))
    j        = f @ E                                             # (N, 3)
    u        = (j + jnp.float32(0.5) * F) / rho_safe[:, None]  # (N, 3)

    eu  = u @ E.T                                                # (N, 19)
    u2  = (u * u).sum(axis=-1, keepdims=True)                   # (N, 1)
    feq = W * rho_safe[:, None] * (
        jnp.float32(1.0)
        + eu / cs2
        + eu * eu / (jnp.float32(2.0) * cs2 * cs2)
        - u2 / (jnp.float32(2.0) * cs2)
    )

    eF    = E @ F                                                # (19,)
    eu_eF = eu * eF / cs2                                       # (N, 19)
    uF    = (u * F).sum(axis=-1, keepdims=True)                 # (N, 1)
    S = (jnp.float32(1.0) - jnp.float32(0.5) / tau) * W * (
        eF / cs2 + eu_eF / cs2 - uF / cs2
    )

    f_post = f - (f - feq) / tau + S
    return jnp.where(solid[:, None], f, f_post)

# ── Pull streaming + halfway bounce-back ──────────────────────────────────────

def _stream(f: jnp.ndarray, solid: jnp.ndarray,
            Nx: int, Ny: int, Nz: int) -> jnp.ndarray:
    f4    = f.reshape(Nx, Ny, Nz, Q)        # (Nx, Ny, Nz, 19)
    s4    = solid.reshape(Nx, Ny, Nz)       # (Nx, Ny, Nz)
    f_new = jnp.zeros_like(f4)
    for q in range(Q):
        ex, ey, ez = int(_E_np[q, 0]), int(_E_np[q, 1]), int(_E_np[q, 2])
        f_shifted  = jnp.roll(f4[..., q], shift=(ex, ey, ez), axis=(0, 1, 2))
        src_solid  = jnp.roll(s4, shift=(ex, ey, ez), axis=(0, 1, 2))
        bounce     = src_solid & ~s4
        f_q        = jnp.where(bounce, f4[..., OPPOSITE[q]], f_shifted)
        f_new      = f_new.at[..., q].set(f_q)
    # Zero out solid nodes
    f_new = jnp.where(s4[..., None], jnp.float32(0.0), f_new)
    return f_new.reshape(-1, Q)             # back to (N, 19)

# ── Step function factory ──────────────────────────────────────────────────────

def make_step_fn(tau: float, solid: jnp.ndarray,
                 F: jnp.ndarray,
                 Nx: int, Ny: int, Nz: int):
    """
    Return a JIT-compiled step function.

    All static data (solid, F, tau) are closed over so XLA treats them
    as compile-time constants — no per-step transfer overhead.

    Parameters
    ----------
    tau        : relaxation time (Python float, baked in at compile time)
    solid      : (N,)    bool device array
    F          : (3,)    float32 device array
    Nx, Ny, Nz : grid dims

    Returns
    -------
    step : f (N,19) -> f (N,19)
    """
    @jax.jit
    def step(f: jnp.ndarray) -> jnp.ndarray:
        f = _collide(f, tau, F, solid)
        f = _stream(f, solid, Nx, Ny, Nz)
        return f
    return step

# ── On-device while_loop ───────────────────────────────────────────────────────

def _run_on_device(
    f_init: jnp.ndarray,
    step_fn,
    solid: jnp.ndarray,
    d_idx: int,
    check_every: int,
    tol: float,
    max_iter: int,
    ema_alpha: float = 0.1,
    n_hits_req: int  = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Run LBM time-stepping entirely on-device via jax.lax.while_loop.
    Host is synced only once, when the loop exits.

    Returns
    -------
    f_final   : (N, 19)
    it_final  : scalar int32
    converged : scalar bool
    """
    N          = solid.shape[0]
    fluid_mask = (~solid).astype(jnp.float32)   # (N,)

    def cond(state):
        _, it, _, n_hits, converged = state
        return (~converged) & (it < max_iter)

    def body(state):
        f, it, u_ema, n_hits, converged = state

        # Skip stepping if already converged — avoids wasted compute in vmap
        f = jax.lax.cond(
            converged,
            lambda f: f,
            lambda f: jax.lax.fori_loop(
                0, check_every, lambda _, fi: step_fn(fi), f),
            f
        )
        it = it + check_every

        u_d    = (f @ E[:, d_idx]) * fluid_mask
        u_mean = u_d.sum() / jnp.float32(N)

        is_first  = (it == check_every)
        u_ema_new = jnp.where(
            is_first,
            u_mean,
            jnp.float32(ema_alpha) * u_mean + jnp.float32(1.0 - ema_alpha) * u_ema
        )
        rel_err = jnp.where(
            is_first,
            jnp.float32(jnp.inf),
            jnp.abs(u_ema_new - u_ema) / (jnp.abs(u_ema) + jnp.float32(1e-30))
        )

        n_hits_new    = jnp.where(rel_err < tol, n_hits + 1, jnp.int32(0))
        # Freeze it at the step where convergence is first reached
        converged_new = n_hits_new >= n_hits_req
        it_frozen     = jnp.where(
            (~converged) & converged_new,   # first time converging this step
            it,
            jnp.where(converged, it - check_every, it)  # already converged: keep frozen value
        )

        return (f, it_frozen, u_ema_new, n_hits_new, converged_new)

    init = (
        f_init,
        jnp.int32(0),
        jnp.float32(0.0),
        jnp.int32(0),
        jnp.bool_(False),
    )

    f_final, it_final, _, _, converged = jax.lax.while_loop(cond, body, init)
    return f_final, it_final, converged

# ── Post-processing helper ─────────────────────────────────────────────────────

def _post_process(
    f_final: jnp.ndarray,
    solid_np: np.ndarray,
    F_np: np.ndarray,
    d_idx: int,
    nu: float,
    force_mag: float,
) -> dict:
    """
    Extract permeability and velocity/density fields from final f.
    Runs on host (numpy) after device sync.

    Parameters
    ----------
    f_final  : (N, 19)  numpy array (already transferred)
    solid_np : (N,)     bool numpy
    F_np     : (3,)     float32 numpy
    d_idx    : int      flow direction index
    nu       : float
    force_mag: float

    Returns
    -------
    dict: permeability, mean_velocity, rho_mean, mach, u_flat, rho_flat
    """
    N          = solid_np.shape[0]
    fluid_mask = (~solid_np).astype(np.float32)

    rho_np     = f_final.sum(axis=-1)
    rho_safe   = np.where(rho_np > 0.0, rho_np, 1.0)
    u_np       = (f_final @ _E_f32 + 0.5 * F_np) / rho_safe[:, None]

    u_np   *= fluid_mask[:, None]
    rho_np *= fluid_mask

    mean_u   = float(u_np[:, d_idx].sum()) / N
    rho_mean = float(rho_np.sum()) / max(float(fluid_mask.sum()), 1.0)
    u_max    = float(np.linalg.norm(u_np, axis=-1).max())
    mach     = u_max * float(np.sqrt(3.0))

    k = (rho_mean * nu * mean_u) / force_mag

    return dict(permeability=k, mean_velocity=mean_u,
                rho_mean=rho_mean, mach=mach,
                u_flat=u_np, rho_flat=rho_np)

# ── Single-sample solver ───────────────────────────────────────────────────────

def compute_permeability(
    rock: np.ndarray,
    *,
    direction: str   = "x",
    nu: float        = 1.0 / 6.0,
    force_mag: float = 1e-4,
    max_iter: int    = 20_000,
    tol: float       = 1e-6,
    check_every: int = 500,
    verbose: bool    = True,
) -> dict:
    """
    Compute permeability of a single binary rock sample via LBM.

    Parameters
    ----------
    rock        : (Nx, Ny, Nz) bool/int array — True/1 = solid
    direction   : 'x', 'y', or 'z'
    nu          : kinematic viscosity in lattice units (default 1/6 → τ=1)
    force_mag   : body-force magnitude (keep << 1 for low Mach)
    max_iter    : hard iteration cap
    tol         : EMA convergence tolerance
    check_every : steps between convergence checks
    verbose     : print summary

    Returns
    -------
    dict
        permeability  : float   lattice units²
        mean_velocity : float   lattice units / step
        porosity      : float
        converged     : bool
        iterations    : int
        mach          : float
        u_field       : np.ndarray (Nx, Ny, Nz, 3)
        rho_field     : np.ndarray (Nx, Ny, Nz)
    """
    assert rock.ndim == 3, "rock must be 3-D"
    Nx, Ny, Nz = rock.shape
    N          = Nx * Ny * Nz
    solid_np   = rock.astype(bool).reshape(-1)   # (N,)

    porosity = float((~solid_np).mean())
    if porosity == 0.0:
        raise ValueError("Zero porosity — no pore space found.")

    # Optional connectivity check
    try:
        from scipy.ndimage import label as nd_label
        ax = {"x": 0, "y": 1, "z": 2}[direction.lower()]
        lbl, _ = nd_label(~solid_np.reshape(Nx, Ny, Nz))
        sl = (slice(None),) * ax
        if not (set(np.unique(lbl[sl + (0,)])) - {0}) & \
               (set(np.unique(lbl[sl + (-1,)])) - {0}):
            print(f"WARNING: pore space does not percolate in {direction}.")
    except ImportError:
        pass

    tau = float(nu) / float(cs2) + 0.5
    if not (0.51 < tau < 2.0):
        print(f"WARNING: τ={tau:.4f} outside stable range (0.51, 2.0).")

    d_idx         = {"x": 0, "y": 1, "z": 2}[direction.lower()]
    F_np          = np.zeros(3, dtype=np.float32)
    F_np[d_idx]   = force_mag
    F             = jnp.array(F_np)
    solid_dev     = jnp.array(solid_np)
    step          = make_step_fn(tau, solid_dev, F, Nx, Ny, Nz)

    f_init = jnp.where(solid_dev[:, None],
                       jnp.float32(0.0),
                       W[None, :])   # ρ=1, u=0 equilibrium

    if verbose:
        print(f"Grid : {Nx}×{Ny}×{Nz}  |  φ={porosity:.4f}  |  "
              f"dir={direction}  |  τ={tau:.4f}  |  F={force_mag:.1e}")

    f_final, it_final, converged = _run_on_device(
        f_init, step, solid_dev, d_idx,
        check_every, tol, max_iter)

    iterations = int(it_final)
    converged  = bool(converged)
    pp = _post_process(np.asarray(f_final), solid_np,
                       F_np, d_idx, nu, force_mag)

    if pp["mach"] > 0.1:
        print(f"WARNING: Mach={pp['mach']:.3f} > 0.1. Reduce force_mag.")

    if verbose:
        status = "converged" if converged else "NOT converged"
        print(f"{status} in {iterations} iters  |  "
              f"k={pp['permeability']:.4e} lu²  |  Ma={pp['mach']:.4f}")

    return dict(
        permeability  = pp["permeability"],
        mean_velocity = pp["mean_velocity"],
        porosity      = porosity,
        converged     = converged,
        iterations    = iterations,
        mach          = pp["mach"],
        u_field       = pp["u_flat"].reshape(Nx, Ny, Nz, 3),
        rho_field     = pp["rho_flat"].reshape(Nx, Ny, Nz),
    )

# ── Batch solver ───────────────────────────────────────────────────────────────

def _step_chunk(
    f: jnp.ndarray,
    solids: jnp.ndarray,
    F: jnp.ndarray,
    tau: float,
    d_idx: int,
    check_every: int,
    Nx: int,
    Ny: int,
    Nz: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Advance all slots by check_every LBM steps and return mean velocities.

    Parameters
    ----------
    f       : (C, N, 19)
    solids  : (C, N) bool
    F       : (3,)
    tau, d_idx, check_every : scalars
    Nx, Ny, Nz : grid dims

    Returns
    -------
    f_new    : (C, N, 19)
    u_means  : (C,)
    """
    def _run_one(f_i: jnp.ndarray, solid_i: jnp.ndarray):
        def step(f):
            f = _collide(f, tau, F, solid_i)
            f = _stream(f, solid_i, Nx, Ny, Nz)
            return f
        f_i = jax.lax.fori_loop(0, check_every, lambda _, fi: step(fi), f_i)
        fluid_mask = (~solid_i).astype(jnp.float32)
        u_mean = ((f_i @ E[:, d_idx]) * fluid_mask).sum() / jnp.float32(solid_i.shape[0])
        return f_i, u_mean

    return jax.vmap(_run_one, in_axes=(0, 0))(f, solids)

def compute_permeability_batch(
    rocks: np.ndarray,
    *,
    direction: str    = "x",
    nu: float         = 1.0 / 6.0,
    force_mag: float  = 1e-4,
    max_iter: int     = 20_000,
    tol: float        = 1e-6,
    check_every: int  = 500,
    batch_size: int   = None,
    verbose: bool     = True,
) -> dict:
    """
    Compute permeability for a collection of same-shape samples.

    Runs batch_size samples in parallel via jax.vmap. Each sample is tracked
    independently: as soon as a slot converges it is post-processed and
    replaced with the next queued sample, so the device stays fully occupied
    until all samples are done.

    Parameters
    ----------
    rocks       : (B, Nx, Ny, Nz) bool/int array — True/1 = solid
    direction   : 'x', 'y', or 'z'
    nu          : kinematic viscosity in lattice units (default 1/6 → τ=1)
    force_mag   : body-force magnitude (keep << 1 for low Mach)
    max_iter    : hard iteration cap per sample
    tol         : EMA convergence tolerance
    check_every : steps between convergence checks
    batch_size  : number of slots run in parallel.
                  None → min(B, 8).
                  Tune to fit device memory:
                    memory ≈ batch_size × N × 19 × 4 bytes
    verbose     : print per-sample summary

    Returns
    -------
    dict  (all arrays shape (B,) or (B, Nx, Ny, Nz, ...))
        permeability  : np.ndarray (B,)
        mean_velocity : np.ndarray (B,)
        porosity      : np.ndarray (B,)
        converged     : np.ndarray (B,)  bool
        iterations    : np.ndarray (B,)  int
        mach          : np.ndarray (B,)
        u_field       : np.ndarray (B, Nx, Ny, Nz, 3)
        rho_field     : np.ndarray (B, Nx, Ny, Nz)
    """
    assert rocks.ndim == 4, "rocks must be (B, Nx, Ny, Nz)"
    B, Nx, Ny, Nz = rocks.shape
    N = Nx * Ny * Nz

    if batch_size is None:
        batch_size = min(B, 8)
    batch_size = min(batch_size, B)

    tau = float(nu) / float(cs2) + 0.5
    if not (0.51 < tau < 2.0):
        print(f"WARNING: τ={tau:.4f} outside stable range (0.51, 2.0).")

    d_idx       = {"x": 0, "y": 1, "z": 2}[direction.lower()]
    F_np        = np.zeros(3, dtype=np.float32)
    F_np[d_idx] = force_mag
    F           = jnp.array(F_np)

    solids_np = rocks.astype(bool).reshape(B, N)

    permeabilities  = np.empty(B, dtype=np.float64)
    mean_velocities = np.empty(B, dtype=np.float64)
    porosities      = np.empty(B, dtype=np.float64)
    machs           = np.empty(B, dtype=np.float64)
    iterations_out  = np.empty(B, dtype=np.int32)
    convergeds_out  = np.empty(B, dtype=bool)
    u_fields        = np.empty((B, Nx, Ny, Nz, 3), dtype=np.float32)
    rho_fields      = np.empty((B, Nx, Ny, Nz),    dtype=np.float32)

    if verbose:
        mem_mb = batch_size * N * 19 * 4 / 1024**2
        print(f"Total samples : {B}  |  slots : {batch_size}")
        print(f"Grid          : {Nx}×{Ny}×{Nz}  |  dir={direction}  "
              f"|  τ={tau:.4f}  |  F={force_mag:.1e}")
        print(f"Memory        : ~{mem_mb:.0f} MB  (f array only)")
        print("-" * 60)

    # Initialise slots with the first batch_size samples
    next_sample = batch_size
    slot_sample = list(range(batch_size))          # slot j → global sample index

    solids_slots = np.array(solids_np[:batch_size])        # (C, N)
    f_slots      = np.where(
        solids_slots[:, :, None],
        np.float32(0.0),
        np.broadcast_to(np.asarray(W)[None, None, :], (batch_size, N, Q)).copy(),
    ).astype(np.float32)

    u_ema  = np.full(batch_size, np.nan, dtype=np.float64)
    n_hits = np.zeros(batch_size, dtype=np.int32)
    iters  = np.zeros(batch_size, dtype=np.int32)

    n_done    = 0
    n_checks  = int(np.ceil(max_iter / check_every))

    for check_idx in range(n_checks):
        if n_done == B:
            break

        f_dev, u_means_dev = _step_chunk(
            jnp.array(f_slots),
            jnp.array(solids_slots),
            F, tau, d_idx, check_every, Nx, Ny, Nz,
        )
        f_slots   = np.asarray(f_dev)
        u_means   = np.asarray(u_means_dev)
        it        = (check_idx + 1) * check_every

        for j in range(batch_size):
            i = slot_sample[j]
            if i == -1:
                continue

            um = float(u_means[j])
            if np.isnan(u_ema[j]):
                u_ema[j] = um
            else:
                u_new     = 0.1 * um + 0.9 * u_ema[j]
                rel_err   = abs(u_new - u_ema[j]) / (abs(u_ema[j]) + 1e-30)
                u_ema[j]  = u_new

                if rel_err < tol:
                    n_hits[j] += 1
                else:
                    n_hits[j] = 0

                if n_hits[j] >= 3:
                    iters[j] = it
                    pp = _post_process(f_slots[j], solids_np[i], F_np,
                                       d_idx, nu, force_mag)

                    permeabilities[i]  = pp["permeability"]
                    mean_velocities[i] = pp["mean_velocity"]
                    porosities[i]      = float((~solids_np[i]).mean())
                    machs[i]           = pp["mach"]
                    iterations_out[i]  = it
                    convergeds_out[i]  = True
                    u_fields[i]        = pp["u_flat"].reshape(Nx, Ny, Nz, 3)
                    rho_fields[i]      = pp["rho_flat"].reshape(Nx, Ny, Nz)

                    if pp["mach"] > 0.1:
                        print(f"  WARNING sample {i}: Mach={pp['mach']:.3f}. "
                              f"Reduce force_mag.")
                    if verbose:
                        print(f"  [{i:3d}] ✓  iters={it:6d}  "
                              f"φ={porosities[i]:.3f}  "
                              f"k={permeabilities[i]:.4e} lu²  "
                              f"Ma={machs[i]:.4f}  "
                              f"[slot {j} → ", end="")

                    n_done += 1

                    if next_sample < B:
                        ni = next_sample
                        next_sample += 1
                        slot_sample[j] = ni
                        solids_slots[j] = solids_np[ni]
                        f_slots[j]      = np.where(
                            solids_np[ni, :, None],
                            np.float32(0.0),
                            np.asarray(W),
                        ).astype(np.float32)
                        u_ema[j]  = np.nan
                        n_hits[j] = 0
                        iters[j]  = 0
                        if verbose:
                            print(f"sample {ni}]", flush=True)
                    else:
                        slot_sample[j] = -1
                        if verbose:
                            print("idle]", flush=True)

        # Handle samples that hit max_iter without converging
        if it >= max_iter:
            for j in range(batch_size):
                i = slot_sample[j]
                if i == -1:
                    continue
                pp = _post_process(f_slots[j], solids_np[i], F_np,
                                   d_idx, nu, force_mag)
                permeabilities[i]  = pp["permeability"]
                mean_velocities[i] = pp["mean_velocity"]
                porosities[i]      = float((~solids_np[i]).mean())
                machs[i]           = pp["mach"]
                iterations_out[i]  = it
                convergeds_out[i]  = False
                u_fields[i]        = pp["u_flat"].reshape(Nx, Ny, Nz, 3)
                rho_fields[i]      = pp["rho_flat"].reshape(Nx, Ny, Nz)
                if verbose:
                    print(f"  [{i:3d}] ✗  iters={it:6d}  "
                          f"φ={porosities[i]:.3f}  "
                          f"k={permeabilities[i]:.4e} lu²  "
                          f"Ma={machs[i]:.4f}")
            break

    return dict(
        permeability  = permeabilities,
        mean_velocity = mean_velocities,
        porosity      = porosities,
        converged     = convergeds_out,
        iterations    = iterations_out,
        mach          = machs,
        u_field       = u_fields,
        rho_field     = rho_fields,
    )

# ── Multi-device batch solver (pmap over devices, vmap within) ─────────────────

@functools.lru_cache(maxsize=None)
def _make_step_pmap(tau: float, d_idx: int, check_every: int,
                    Nx: int = 0, Ny: int = 0, Nz: int = 0):
    """
    Build and cache a pmap+vmap function that advances f by check_every
    LBM steps and returns (f, u_means).

    Cached on (tau, d_idx, check_every) — compiles once, reused every
    iteration of the Python convergence loop.

    Signature of returned function:
        step_pmap(f, solids, F)
            f      : (D, C, N, 19)
            solids : (D, C, N)
            F      : (D, 3)
        returns:
            f_new  : (D, C, N, 19)
            u_means: (D, C)
    """
    def _one_device(f_dev, solids_dev, F_dev):
        def _run_one(f_i, solid_i):
            def step(f):
                f = _collide(f, tau, F_dev, solid_i)
                f = _stream(f, solid_i, Nx, Ny, Nz)
                return f
            f_i = jax.lax.fori_loop(0, check_every, lambda _, fi: step(fi), f_i)
            fluid_mask = (~solid_i).astype(jnp.float32)
            u_mean = ((f_i @ E[:, d_idx]) * fluid_mask).sum() / jnp.float32(solid_i.shape[0])
            return f_i, u_mean
        return jax.vmap(_run_one, in_axes=(0, 0))(f_dev, solids_dev)

    return jax.pmap(_one_device, in_axes=(0, 0, 0))

def compute_permeability_batch_multi(
    rocks: np.ndarray,
    *,
    direction: str         = "x",
    nu: float              = 1.0 / 6.0,
    force_mag: float       = 1e-4,
    max_iter: int          = 20_000,
    tol: float             = 1e-6,
    check_every: int       = 500,
    batch_size_per_device: int = 4,
    verbose: bool          = True,
    progress: bool         = False,
    debug_convergence: bool = False,
) -> dict:
    """
    Compute permeability across all available devices.

    Architecture
    ------------
    pmap  : distributes slots across devices.
    vmap  : runs batch_size_per_device slots in parallel within each device.

    Each slot is tracked independently. When a slot converges it is
    post-processed and immediately replaced with the next queued sample,
    keeping all devices occupied until every sample is done.

    Parameters
    ----------
    rocks                 : (B, Nx, Ny, Nz) bool/int — True/1 = solid
    direction             : 'x', 'y', or 'z'
    nu                    : kinematic viscosity in lattice units
    force_mag             : body-force magnitude
    max_iter              : hard iteration cap per sample
    tol                   : EMA convergence tolerance
    check_every           : steps between convergence checks
    batch_size_per_device : slots per device.
                            Total slots = n_devices × batch_size_per_device.
                            Tune to fit device memory:
                              mem ≈ batch_size_per_device × N × 19 × 4 bytes
    verbose               : print per-sample summary
    progress              : show a tqdm progress bar

    Returns
    -------
    dict  (arrays shape (B,) or (B, Nx, Ny, Nz, ...))
        permeability  : np.ndarray (B,)
        mean_velocity : np.ndarray (B,)
        porosity      : np.ndarray (B,)
        converged     : np.ndarray (B,)  bool
        iterations    : np.ndarray (B,)  int
        mach          : np.ndarray (B,)
        u_field       : np.ndarray (B, Nx, Ny, Nz, 3)
        rho_field     : np.ndarray (B, Nx, Ny, Nz)
    """
    assert rocks.ndim == 4, "rocks must be (B, Nx, Ny, Nz)"
    B, Nx, Ny, Nz = rocks.shape
    N = Nx * Ny * Nz

    devices   = jax.devices()
    n_devices = len(devices)

    tau = float(nu) / float(cs2) + 0.5
    if not (0.51 < tau < 2.0):
        print(f"WARNING: τ={tau:.4f} outside stable range (0.51, 2.0).")

    d_idx       = {"x": 0, "y": 1, "z": 2}[direction.lower()]
    F_np        = np.zeros(3, dtype=np.float32)
    F_np[d_idx] = force_mag

    F_rep = jnp.array(np.stack([F_np] * n_devices))

    solids_np  = rocks.astype(bool).reshape(B, N)

    total_slots = n_devices * batch_size_per_device

    permeabilities  = np.empty(B, dtype=np.float64)
    mean_velocities = np.empty(B, dtype=np.float64)
    porosities      = np.empty(B, dtype=np.float64)
    machs           = np.empty(B, dtype=np.float64)
    iterations_out  = np.empty(B, dtype=np.int32)
    convergeds_out  = np.empty(B, dtype=bool)
    u_fields        = np.empty((B, Nx, Ny, Nz, 3), dtype=np.float32)
    rho_fields      = np.empty((B, Nx, Ny, Nz),    dtype=np.float32)

    if verbose:
        mem_mb = batch_size_per_device * N * 19 * 4 / 1024**2
        print(f"Devices       : {n_devices}  "
              f"({', '.join(str(d) for d in devices)})")
        print(f"Slots         : {total_slots}  "
              f"({n_devices} devices × {batch_size_per_device})")
        print(f"Total samples : {B}")
        print(f"Grid          : {Nx}×{Ny}×{Nz}  |  dir={direction}  "
              f"|  τ={tau:.4f}  |  F={force_mag:.1e}")
        print(f"Memory/device : ~{mem_mb:.0f} MB  (f array only)")
        print("-" * 60)

    step_pmap = _make_step_pmap(tau, d_idx, check_every, Nx, Ny, Nz)

    # Initialise slots
    next_sample = min(total_slots, B)
    slot_sample = list(range(next_sample)) + [-1] * (total_slots - next_sample)

    W_np = np.asarray(W)

    def _init_f(solid: np.ndarray) -> np.ndarray:
        return np.where(solid[:, None], np.float32(0.0), W_np).astype(np.float32)

    f_slots      = np.stack([_init_f(solids_np[i]) if i != -1
                             else np.zeros((N, Q), np.float32)
                             for i in slot_sample])           # (S, N, 19)
    solids_slots = np.stack([solids_np[i] if i != -1
                             else np.zeros(N, bool)
                             for i in slot_sample])           # (S, N)

    u_ema  = np.full(total_slots, np.nan, dtype=np.float64)
    n_hits = np.zeros(total_slots, dtype=np.int32)

    n_done   = 0
    n_checks = int(np.ceil(max_iter / check_every))

    if progress:
        from tqdm import tqdm
        pbar = tqdm(total=B, unit="sample", desc="LBM")

    for check_idx in range(n_checks):
        if n_done == B:
            break

        # Reshape to (D, C, N, 19) / (D, C, N) for pmap
        f_dc      = jnp.array(f_slots.reshape(n_devices, batch_size_per_device, N, Q))
        solids_dc = jnp.array(solids_slots.reshape(n_devices, batch_size_per_device, N))

        f_dc, u_means = step_pmap(f_dc, solids_dc, F_rep)

        f_slots   = np.asarray(f_dc).reshape(total_slots, N, Q)
        u_means_f = np.asarray(u_means).reshape(total_slots)
        it        = (check_idx + 1) * check_every

        if debug_convergence:
            parts = []
            for j in range(total_slots):
                i = slot_sample[j]
                if i == -1:
                    parts.append(f"s{j}=idle")
                    continue
                um    = float(u_means_f[j])
                u_new = 0.1 * um + 0.9 * (u_ema[j] if not np.isnan(u_ema[j]) else um)
                err   = abs(u_new - (u_ema[j] if not np.isnan(u_ema[j]) else u_new)) / (abs(u_ema[j]) + 1e-30) if not np.isnan(u_ema[j]) else float("inf")
                parts.append(f"s{j}={err:.2e}({err/tol:.1f}x)")
            print(f"  [iter {it:6d}] " + "  ".join(parts), flush=True)

        for j in range(total_slots):
            i = slot_sample[j]
            if i == -1:
                continue

            um = float(u_means_f[j])
            if np.isnan(u_ema[j]):
                u_ema[j] = um
                continue

            u_new    = 0.1 * um + 0.9 * u_ema[j]
            rel_err  = abs(u_new - u_ema[j]) / (abs(u_ema[j]) + 1e-30)
            u_ema[j] = u_new

            if rel_err < tol:
                n_hits[j] += 1
            else:
                n_hits[j] = 0

            if n_hits[j] >= 3:
                pp = _post_process(f_slots[j], solids_np[i], F_np,
                                   d_idx, nu, force_mag)

                permeabilities[i]  = pp["permeability"]
                mean_velocities[i] = pp["mean_velocity"]
                porosities[i]      = float((~solids_np[i]).mean())
                machs[i]           = pp["mach"]
                iterations_out[i]  = it
                convergeds_out[i]  = True
                u_fields[i]        = pp["u_flat"].reshape(Nx, Ny, Nz, 3)
                rho_fields[i]      = pp["rho_flat"].reshape(Nx, Ny, Nz)

                if pp["mach"] > 0.1:
                    print(f"  WARNING sample {i}: Mach={pp['mach']:.3f}. "
                          f"Reduce force_mag.")

                dev_id = j // batch_size_per_device
                if verbose:
                    print(f"  [{i:3d}] ✓  dev={dev_id}  iters={it:6d}  "
                          f"φ={porosities[i]:.3f}  "
                          f"k={permeabilities[i]:.4e} lu²  "
                          f"Ma={machs[i]:.4f}  "
                          f"[slot {j} → ", end="")

                if progress:
                    pbar.update(1)

                n_done += 1

                if next_sample < B:
                    ni = next_sample
                    next_sample += 1
                    slot_sample[j]  = ni
                    solids_slots[j] = solids_np[ni]
                    f_slots[j]      = _init_f(solids_np[ni])
                    u_ema[j]        = np.nan
                    n_hits[j]       = 0
                    if verbose:
                        print(f"sample {ni}]", flush=True)
                else:
                    slot_sample[j] = -1
                    if verbose:
                        print("idle]", flush=True)

        if it >= max_iter:
            for j in range(total_slots):
                i = slot_sample[j]
                if i == -1:
                    continue
                pp = _post_process(f_slots[j], solids_np[i], F_np,
                                   d_idx, nu, force_mag)
                permeabilities[i]  = pp["permeability"]
                mean_velocities[i] = pp["mean_velocity"]
                porosities[i]      = float((~solids_np[i]).mean())
                machs[i]           = pp["mach"]
                iterations_out[i]  = it
                convergeds_out[i]  = False
                u_fields[i]        = pp["u_flat"].reshape(Nx, Ny, Nz, 3)
                rho_fields[i]      = pp["rho_flat"].reshape(Nx, Ny, Nz)
                if verbose:
                    dev_id = j // batch_size_per_device
                    print(f"  [{i:3d}] ✗  dev={dev_id}  iters={it:6d}  "
                          f"φ={porosities[i]:.3f}  "
                          f"k={permeabilities[i]:.4e} lu²  "
                          f"Ma={machs[i]:.4f}")
            break

    if progress:
        pbar.close()

    return dict(
        permeability  = permeabilities,
        mean_velocity = mean_velocities,
        porosity      = porosities,
        converged     = convergeds_out,
        iterations    = iterations_out,
        mach          = machs,
        u_field       = u_fields,
        rho_field     = rho_fields,
    )
