"""Mixed-precision iterative refinement (NEW in driftjax).

Idea
----
On accelerator hardware FP32/BF16 tensor cores are ~2–8× faster than the
FP64 path.  The DDP Newton solve spends most of its time in the linear
solve J·dx = −F; we can run that solve in low precision and *recover*
near-FP64 accuracy with a few correction steps (Higham-style iterative
refinement):

    x₀  = solve_{low}(J, b)
    r_k = b − J·x_k            (residual computed in FP64  ← the key)
    d_k = solve_{low}(J, r_k)
    x_{k+1} = x_k + d_k

Convergence theory: the iteration contracts with factor ρ ≈ c·uₗₒ·cond(J);
it reaches the FP64 accuracy floor (~cond(J)·u₆₄) provided
cond(J) ≲ 1/uₗₒ ≈ 8×10⁶ (with residuals in FP64).  The DDP Jacobian at
bias is row-equilibrated to κ ≲ 1e4–1e5, so refinement converges in 2–4
steps; near-equilibrium degenerate cases (κ ≫ 1/uₗₒ) are *detected* by a
non-converging refinement and routed to the FP64 backend.  This converts
an unconditioned speed/accuracy trade-off into a gated one.

The FP32 dense core is used because it is vmappable and benefits from
accelerator tensor cores; the same refinement wraps the batched sweep.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from driftjax._util import is_tracer as _is_tracer_mp  # M8: single shared helper
from driftjax.numerics.linalg import row_equilibrate

MAX_REFINE = 8
REFINE_TOL = 1e-13  # relative residual target (≈ FP64 floor for κ ~ 1e4)
COND_ROUGH = 5e6  # rough κ threshold where FP32 refinement becomes dubious


def _dense_solve_f32(Je, be):
    """Dense FP32 solve for iterative refinement core."""
    J32 = Je.astype(jnp.float32)
    b32 = be.astype(jnp.float32)
    x32 = jnp.linalg.solve(J32, b32)
    return x32.astype(jnp.float64)


def solve_refined(J, b, tol: float = REFINE_TOL, max_refine: int = MAX_REFINE):
    """J·x = b with FP32 dense core + FP64 residual refinement.

    Returns (x, rel_resid, n_refine, converged). Trace-safe: the
    refinement loop is a ``lax.while_loop`` so it compiles under
    ``jit``/``grad``/``vmap`` and the FP32 core can use accelerator
    tensor cores. ``converged`` is a JAX bool array when traced,
    else a Python bool.
    """
    Je, be, _ = row_equilibrate(J, b)
    nrm_b = jnp.linalg.norm(be)
    safe_nrm = jnp.where(nrm_b == 0, jnp.asarray(1.0, dtype=nrm_b.dtype), nrm_b)
    x0 = _dense_solve_f32(Je, be)
    rel0 = jnp.linalg.norm(be - Je @ x0) / safe_nrm

    def _cond(state):
        _, rel, i = state
        return (i < max_refine) & (rel >= tol)

    def _body(state):
        x, _, i = state
        r = be - Je @ x
        d = _dense_solve_f32(Je, r)
        x_new = x + d
        rel_new = jnp.linalg.norm(be - Je @ x_new) / safe_nrm
        return (x_new, rel_new, i + 1)

    x, rel, n_ref = jax.lax.while_loop(_cond, _body, (x0, rel0, jnp.asarray(0, dtype=jnp.int32)))
    conv = rel < tol
    if _is_tracer_mp(conv):
        return x, rel, n_ref, conv
    return x, rel, int(n_ref), bool(conv)


def mixed_dense_solve(A, b, tol: float = 1e-10, max_refine: int = 2):
    """Dense ``A x = b`` via FP32 LU (factored once) + FP64 refinement.

    GPU-oriented: FP32 LU halves memory traffic vs FP64 on accelerators
    (on CPU, measured FP32 == FP64 at N=1500, so this is a wash there).
    Row-equilibrates first (mirrors :func:`solve_refined`), refines a
    fixed count (trace-safe), then gates: pass → refined ``x``; fail →
    full FP64 dense solve. Returns ``(x, rel, used_fp64_fallback)``.
    Wired into the adjoint dense path via :func:`adjoint_dense_solve`
    (GPU-gated); T4-measured 2.3x solve speedup, FD-validated.
    """
    from jax.scipy.linalg import lu_factor, lu_solve

    Ae, be, _ = row_equilibrate(A, b)
    nrm = jnp.linalg.norm(be)
    safe_nrm = jnp.where(nrm == 0, jnp.asarray(1.0, dtype=nrm.dtype), nrm)
    Af = Ae.astype(jnp.float32)
    lu_piv = lu_factor(Af)
    x0 = lu_solve(lu_piv, be.astype(jnp.float32)).astype(jnp.float64)

    def _refine_cond(state):
        _, _, i = state
        return i < int(max_refine)

    def _refine_body(state):
        x, _, i = state
        r = be - Ae @ x
        x_new = x + lu_solve(lu_piv, r.astype(jnp.float32)).astype(jnp.float64)
        return (x_new, r, i + 1)

    x, _, _ = jax.lax.while_loop(
        _refine_cond, _refine_body, (x0, be, jnp.array(0, dtype=jnp.int32))
    )
    rel = jnp.linalg.norm(be - Ae @ x) / safe_nrm
    ok = jnp.all(jnp.isfinite(x)) & jnp.isfinite(rel) & (rel < tol)
    x_dense = jnp.linalg.solve(A, b)
    out = jax.lax.cond(ok, lambda _: x, lambda _: x_dense, None)
    return out, rel, jnp.logical_not(ok)


def _mixed_adjoint_enabled() -> bool:
    """Whether adjoint dense solves may use the FP32 mixed path.

    Auto-selects by platform (any non-CPU device, e.g. GPU where FP32 LU
    is ~2x faster — T4-measured 2.29x); override with
    ``DRIFTJAX_MIXED_ADJOINT=1`` (force) or ``=0`` (disable). Evaluated
    at trace time from static context, so only the taken path compiles.
    """
    import os

    env = os.environ.get("DRIFTJAX_MIXED_ADJOINT", "auto")
    if env == "1":
        return True
    if env == "0":
        return False
    try:
        return any(d.platform != "cpu" for d in jax.devices())
    except Exception:
        return False


def adjoint_dense_solve(A, b, tol: float = 1e-10) -> jax.Array:
    """Dense solve for adjoint systems with GPU-gated mixed precision.

    Drop-in replacement for ``jnp.linalg.solve`` on the hot path
    (``_sweep_bwd.per_bias``, ``_eq_bwd``): FP32 LU + FP64 refinement
    where beneficial, bit-identical FP64 dense elsewhere. The gate
    inside :func:`mixed_dense_solve` keeps it correct on
    ill-conditioned systems (falls back automatically).
    """
    if _mixed_adjoint_enabled():
        x, _, _ = mixed_dense_solve(A, b, tol=tol)
        return x
    return jnp.linalg.solve(A, b)


def solve_refined_batched(Jb, bb, tol: float = REFINE_TOL, max_refine: int = MAX_REFINE):
    """Batched (B, 3n, 3n) solve with FP32 core + FP64 refinement (vmap-free).

    Trace-safe via ``lax.while_loop``; ``converged`` is a JAX bool array
    when traced, else a Python bool.
    """
    Je, be, _ = row_equilibrate(Jb, bb)

    def dense32(Je_b, be_b):
        return jnp.linalg.solve(Je_b.astype(jnp.float32), be_b.astype(jnp.float32)).astype(
            jnp.float64
        )

    x0 = jax.vmap(lambda j, b: dense32(j, b))(Je, be)
    nrm_b = jnp.linalg.norm(be, axis=-1, keepdims=True)
    safe_nrm = jnp.where(nrm_b == 0, jnp.asarray(1.0, dtype=nrm_b.dtype), nrm_b)

    def _resid(v):
        return be - (Je @ v[..., None])[..., 0]  # per-batch matvec

    rel0 = jnp.linalg.norm(_resid(x0), axis=-1, keepdims=True) / safe_nrm

    def _cond(state):
        _, rel, i = state
        return (i < max_refine) & (jnp.max(rel) >= tol)

    def _body(state):
        x, _, i = state
        r = _resid(x)
        d = jax.vmap(lambda j, b: dense32(j, b))(Je, r)
        x_new = x + d
        rel_new = jnp.linalg.norm(_resid(x_new), axis=-1, keepdims=True) / safe_nrm
        return (x_new, rel_new, i + 1)

    x, rel, n_ref = jax.lax.while_loop(_cond, _body, (x0, rel0, jnp.asarray(0, dtype=jnp.int32)))
    conv = jnp.all(rel < tol)
    if _is_tracer_mp(conv):
        return x, rel, n_ref, conv
    return x, rel, int(n_ref), bool(conv)
