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

  Example — 4 GPUs, batch_size_per_device=8, 100 samples:
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
            nb: jnp.ndarray) -> jnp.ndarray:
    """
    Parameters
    ----------
    f     : (N, 19)
    solid : (N,)   bool
    nb    : (N, 19) int32  neighbour table

    Returns
    -------
    f_new : (N, 19)
    """
    # Pull each population from its upstream neighbour
    f_pull = f[nb, jnp.arange(Q)]          # (N, 19)

    # Halfway bounce-back: upstream is solid AND current node is fluid
    src_solid = solid[nb]                  # (N, 19)
    bounce    = src_solid & ~solid[:, None]
    f_new     = jnp.where(bounce, f[:, OPPOSITE], f_pull)

    # Solid nodes carry no populations
    return jnp.where(solid[:, None], jnp.float32(0.0), f_new)


# ── Step function factory ──────────────────────────────────────────────────────

def make_step_fn(tau: float, solid: jnp.ndarray,
                 nb: jnp.ndarray, F: jnp.ndarray):
    """
    Return a JIT-compiled step function.

    All static data (solid, nb, F, tau) are closed over so XLA treats them
    as compile-time constants — no per-step transfer overhead.

    Parameters
    ----------
    tau   : relaxation time (Python float, baked in at compile time)
    solid : (N,)    bool device array
    nb    : (N, 19) int32 device array
    F     : (3,)    float32 device array

    Returns
    -------
    step : f (N,19) -> f (N,19)
    """
    @jax.jit
    def step(f: jnp.ndarray) -> jnp.ndarray:
        f = _collide(f, tau, F, solid)
        f = _stream(f, solid, nb)
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
    nb            = build_neighbour_table(Nx, Ny, Nz)
    step          = make_step_fn(tau, solid_dev, nb, F)

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

def _run_chunk(
    solids_chunk: jnp.ndarray,
    nb: jnp.ndarray,
    F: jnp.ndarray,
    tau: float,
    d_idx: int,
    check_every: int,
    tol: float,
    max_iter: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    vmap a single chunk of samples on-device.

    Parameters
    ----------
    solids_chunk : (C, N) bool   — C = chunk size
    nb           : (N, 19) int32
    F            : (3,)
    tau, d_idx, check_every, tol, max_iter : scalars

    Returns
    -------
    f_finals   : (C, N, 19)
    it_finals  : (C,)
    convergeds : (C,)
    """
    C = solids_chunk.shape[0]
    N = nb.shape[0]

    f_init_chunk = jnp.where(
        solids_chunk[:, :, None],
        jnp.float32(0.0),
        W[None, None, :]
    )   # (C, N, 19)

    def _run_one(f_init: jnp.ndarray, solid: jnp.ndarray):
        @jax.jit
        def step(f):
            f = _collide(f, tau, F, solid)
            f = _stream(f, solid, nb)
            return f
        return _run_on_device(f_init, step, solid, d_idx,
                              check_every, tol, max_iter)

    return jax.vmap(_run_one, in_axes=(0, 0))(f_init_chunk, solids_chunk)


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

    Internally splits the collection into chunks of `batch_size` and runs
    each chunk with jax.vmap (all samples in a chunk run in parallel on-device).
    The neighbour table and compiled step function are built once and reused
    across all chunks.

    Parameters
    ----------
    rocks       : (B, Nx, Ny, Nz) bool/int array — True/1 = solid
    direction   : 'x', 'y', or 'z'
    nu          : kinematic viscosity in lattice units (default 1/6 → τ=1)
    force_mag   : body-force magnitude (keep << 1 for low Mach)
    max_iter    : hard iteration cap — all samples in a chunk run this many
                  steps (vmap lockstep); set generously for high-porosity rocks
    tol         : EMA convergence tolerance
    check_every : steps between convergence checks
    batch_size  : number of samples per vmap chunk.
                  None  → run all B samples in one chunk (original behaviour).
                  Tune this to fit GPU memory:
                    memory per chunk ≈ batch_size × N × 19 × 4 bytes
                  Example: 64³ grid, batch_size=8 → ~120 MB per chunk.
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

    # Default: one chunk = whole batch
    if batch_size is None:
        batch_size = B
    batch_size = min(batch_size, B)

    tau = float(nu) / float(cs2) + 0.5
    if not (0.51 < tau < 2.0):
        print(f"WARNING: τ={tau:.4f} outside stable range (0.51, 2.0).")

    d_idx       = {"x": 0, "y": 1, "z": 2}[direction.lower()]
    F_np        = np.zeros(3, dtype=np.float32)
    F_np[d_idx] = force_mag
    F           = jnp.array(F_np)

    # Build neighbour table once — shared across all chunks
    nb         = build_neighbour_table(Nx, Ny, Nz)          # (N, 19)
    solids_np  = rocks.astype(bool).reshape(B, N)           # (B, N)

    # Output arrays — filled chunk by chunk
    permeabilities  = np.empty(B, dtype=np.float64)
    mean_velocities = np.empty(B, dtype=np.float64)
    porosities      = np.empty(B, dtype=np.float64)
    machs           = np.empty(B, dtype=np.float64)
    iterations_out  = np.empty(B, dtype=np.int32)
    convergeds_out  = np.empty(B, dtype=bool)
    u_fields        = np.empty((B, Nx, Ny, Nz, 3), dtype=np.float32)
    rho_fields      = np.empty((B, Nx, Ny, Nz),    dtype=np.float32)

    n_chunks = int(np.ceil(B / batch_size))

    if verbose:
        mem_mb = batch_size * N * 19 * 4 / 1024**2
        print(f"Total samples : {B}  |  chunk size : {batch_size}  "
              f"|  chunks : {n_chunks}")
        print(f"Grid          : {Nx}×{Ny}×{Nz}  |  dir={direction}  "
              f"|  τ={tau:.4f}  |  F={force_mag:.1e}")
        print(f"Memory/chunk  : ~{mem_mb:.0f} MB  (f array only)")
        print("-" * 60)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * batch_size
        end   = min(start + batch_size, B)
        C     = end - start   # actual chunk size (last chunk may be smaller)

        if verbose:
            print(f"Chunk {chunk_idx + 1}/{n_chunks}  "
                  f"(samples {start}–{end - 1}) …", flush=True)

        solids_chunk = jnp.array(solids_np[start:end])      # (C, N)

        f_finals, it_finals, convergeds = _run_chunk(
            solids_chunk, nb, F, tau, d_idx,
            check_every, tol, max_iter)

        # Transfer to host
        f_finals_np   = np.asarray(f_finals)    # (C, N, 19)
        iterations_np = np.asarray(it_finals)   # (C,)
        convergeds_np = np.asarray(convergeds)  # (C,)

        # Post-process each sample in the chunk
        for j in range(C):
            i       = start + j
            solid_i = solids_np[i]
            pp = _post_process(f_finals_np[j], solid_i, F_np,
                               d_idx, nu, force_mag)

            permeabilities[i]  = pp["permeability"]
            mean_velocities[i] = pp["mean_velocity"]
            porosities[i]      = float((~solid_i).mean())
            machs[i]           = pp["mach"]
            iterations_out[i]  = iterations_np[j]
            convergeds_out[i]  = convergeds_np[j]
            u_fields[i]        = pp["u_flat"].reshape(Nx, Ny, Nz, 3)
            rho_fields[i]      = pp["rho_flat"].reshape(Nx, Ny, Nz)

            if pp["mach"] > 0.1:
                print(f"  WARNING sample {i}: Mach={pp['mach']:.3f}. "
                      f"Reduce force_mag.")

            if verbose:
                status = "✓" if convergeds_np[j] else "✗"
                print(f"  [{i:3d}] {status}  iters={iterations_np[j]:6d}  "
                      f"φ={porosities[i]:.3f}  "
                      f"k={permeabilities[i]:.4e} lu²  "
                      f"Ma={machs[i]:.4f}")

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
def _make_step_pmap(tau: float, d_idx: int, check_every: int):
    """
    Build and cache a pmap+vmap function that advances f by check_every
    LBM steps and returns (f, u_means).

    Cached on (tau, d_idx, check_every) — compiles once, reused every
    iteration of the Python convergence loop.

    Signature of returned function:
        step_pmap(f, solids, nb, F)
            f      : (D, C, N, 19)
            solids : (D, C, N)
            nb     : (D, N, 19)
            F      : (D, 3)
        returns:
            f_new  : (D, C, N, 19)
            u_means: (D, C)        mean velocity per sample
    """
    def _one_device(f_dev, solids_dev, nb_dev, F_dev):
        def _run_one(f_i, solid_i):
            def step(f):
                f = _collide(f, tau, F_dev, solid_i)
                f = _stream(f, solid_i, nb_dev)
                return f
            f_i = jax.lax.fori_loop(0, check_every, lambda _, fi: step(fi), f_i)
            fluid_mask = (~solid_i).astype(jnp.float32)
            u_mean = ((f_i @ E[:, d_idx]) * fluid_mask).sum() / jnp.float32(solid_i.shape[0])
            return f_i, u_mean
        return jax.vmap(_run_one, in_axes=(0, 0))(f_dev, solids_dev)

    return jax.pmap(_one_device, in_axes=(0, 0, 0, 0))


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
) -> dict:
    """
    Compute permeability across all available devices (GPUs/TPUs).

    Architecture
    ------------
    pmap  : distributes samples across devices — each device is independent,
            no lockstep between devices.
    vmap  : runs batch_size_per_device samples in parallel within each device —
            lockstep within one device.

    Devices are detected automatically via jax.device_count().
    If only 1 device is available this falls back to vmap-only behaviour
    (equivalent to compute_permeability_batch).

    Parameters
    ----------
    rocks                 : (B, Nx, Ny, Nz) bool/int — True/1 = solid
    direction             : 'x', 'y', or 'z'
    nu                    : kinematic viscosity in lattice units
    force_mag             : body-force magnitude
    max_iter              : hard iteration cap per sample
    tol                   : EMA convergence tolerance
    check_every           : steps between convergence checks
    batch_size_per_device : samples processed in parallel on each device.
                            Total parallelism = n_devices × batch_size_per_device.
                            Tune to fit per-device memory:
                              mem ≈ batch_size_per_device × N × 19 × 4 bytes
    verbose               : print per-sample and per-round summary
    progress              : show a tqdm progress bar over samples

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

    # ── Device detection ──────────────────────────────────────────────────────
    devices   = jax.devices()
    n_devices = len(devices)

    tau = float(nu) / float(cs2) + 0.5
    if not (0.51 < tau < 2.0):
        print(f"WARNING: τ={tau:.4f} outside stable range (0.51, 2.0).")

    d_idx       = {"x": 0, "y": 1, "z": 2}[direction.lower()]
    F_np        = np.zeros(3, dtype=np.float32)
    F_np[d_idx] = force_mag

    nb_np  = np.asarray(build_neighbour_table(Nx, Ny, Nz))
    nb_rep = jnp.array(np.stack([nb_np] * n_devices))
    F_rep  = jnp.array(np.stack([F_np]  * n_devices))

    solids_np  = rocks.astype(bool).reshape(B, N)   # (B, N)

    # Round size = all devices × per-device batch
    round_size = n_devices * batch_size_per_device
    n_rounds   = int(np.ceil(B / round_size))

    # Output arrays
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
        print(f"Per-device    : {batch_size_per_device} samples  "
              f"(~{mem_mb:.0f} MB f-array per device)")
        print(f"Round size    : {round_size} samples  "
              f"({n_devices} devices × {batch_size_per_device})")
        print(f"Total samples : {B}  |  rounds : {n_rounds}")
        print(f"Grid          : {Nx}×{Ny}×{Nz}  |  dir={direction}  "
              f"|  τ={tau:.4f}  |  F={force_mag:.1e}")
        print("-" * 60)

    if progress:
        from tqdm import tqdm
        pbar = tqdm(total=max_iter, unit="step", desc="LBM")

    step_pmap = _make_step_pmap(tau, d_idx, check_every)

    for round_idx in range(n_rounds):
        r_start = round_idx * round_size
        r_end   = min(r_start + round_size, B)
        n_real  = r_end - r_start

        if verbose:
            print(f"Round {round_idx + 1}/{n_rounds}  "
                  f"(samples {r_start}–{r_end - 1}) …", flush=True)

        # Pad last round to exactly round_size
        if n_real < round_size:
            pad = round_size - n_real
            solids_round = np.concatenate([
                solids_np[r_start:r_end],
                np.tile(solids_np[r_end - 1], (pad, 1))
            ], axis=0)
        else:
            solids_round = solids_np[r_start:r_end]

        # (D, C, N)
        solids_dc = jnp.array(
            solids_round.reshape(n_devices, batch_size_per_device, N)
        )

        # Initialise f for all samples in this round
        f_dc = jnp.where(
            solids_dc[:, :, :, None],
            jnp.float32(0.0),
            W[None, None, None, :]
        )   # (D, C, N, 19)

        # Per-sample convergence state (flat over round_size)
        u_ema      = np.full(round_size, np.nan, dtype=np.float64)
        n_hits     = np.zeros(round_size, dtype=np.int32)
        convergeds_np = np.zeros(round_size, dtype=bool)
        iterations_np = np.zeros(round_size, dtype=np.int32)

        ema_alpha = 0.1
        n_hits_req = 3
        n_checks = int(np.ceil(max_iter / check_every))

        if progress:
            pbar.reset(total=max_iter)

        # Python convergence loop — syncs every check_every steps
        for check_idx in range(n_checks):
            if convergeds_np[:n_real].all():
                break

            f_dc, u_means = step_pmap(f_dc, solids_dc, nb_rep, F_rep)
            u_means_flat = np.asarray(u_means).reshape(round_size)

            it = (check_idx + 1) * check_every

            for j in range(round_size):
                if convergeds_np[j]:
                    continue
                um = float(u_means_flat[j])
                if np.isnan(u_ema[j]):
                    u_ema[j] = um
                    rel_err  = np.inf
                else:
                    u_ema_new = ema_alpha * um + (1.0 - ema_alpha) * u_ema[j]
                    rel_err   = abs(u_ema_new - u_ema[j]) / (abs(u_ema[j]) + 1e-30)
                    u_ema[j]  = u_ema_new

                if rel_err < tol:
                    n_hits[j] += 1
                else:
                    n_hits[j] = 0

                if n_hits[j] >= n_hits_req:
                    convergeds_np[j] = True
                    iterations_np[j] = it

            if progress:
                n_converged = int(convergeds_np[:n_real].sum())
                pbar.n = min(it, max_iter)
                pbar.set_postfix(converged=f"{n_converged}/{n_real}")
                pbar.refresh()

        # For samples that never converged, record max_iter
        for j in range(round_size):
            if not convergeds_np[j]:
                iterations_np[j] = max_iter

        f_finals_np = np.asarray(f_dc).reshape(round_size, N, 19)

        # Post-process real samples
        for j in range(n_real):
            i       = r_start + j
            solid_i = solids_np[i]
            pp = _post_process(f_finals_np[j], solid_i, F_np,
                               d_idx, nu, force_mag)

            permeabilities[i]  = pp["permeability"]
            mean_velocities[i] = pp["mean_velocity"]
            porosities[i]      = float((~solid_i).mean())
            machs[i]           = pp["mach"]
            iterations_out[i]  = iterations_np[j]
            convergeds_out[i]  = convergeds_np[j]
            u_fields[i]        = pp["u_flat"].reshape(Nx, Ny, Nz, 3)
            rho_fields[i]      = pp["rho_flat"].reshape(Nx, Ny, Nz)

            if pp["mach"] > 0.1:
                print(f"  WARNING sample {i}: Mach={pp['mach']:.3f}. "
                      f"Reduce force_mag.")

            if verbose:
                dev_id = j // batch_size_per_device
                status = "✓" if convergeds_np[j] else "✗"
                print(f"  [{i:3d}] {status}  dev={dev_id}  "
                      f"iters={iterations_np[j]:6d}  "
                      f"φ={porosities[i]:.3f}  "
                      f"k={permeabilities[i]:.4e} lu²  "
                      f"Ma={machs[i]:.4f}")

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
