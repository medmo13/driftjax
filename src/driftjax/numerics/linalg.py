"""Sparse / direct linear algebra for the DDP Jacobian.

Backends (single entry point ``linsolve``):
  "csr"     row-equilibrated banded→CSR spsolve (JAX CPU sparse-direct)
  "banded"  exact O(N) block-Thomas on the interleaved 3×3 blocks (pure jnp,
            vmappable — the batched-sweep workhorse)
  "dense"   jnp.linalg.solve (small n / gradient A-B reference)
  "auto"    CSR with dense fallback on bad linear residual

The scipy CSR compatibility workaround is isolated in this module so the
solver core does not depend on SciPy's matrix-construction details.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from jax import lax, vmap

from driftjax._util import is_tracer as _is_tracer_linalg  # M8: single shared helper
from driftjax.numerics import block_thomas as _bt


def _spsolve_host(data, indices, indptr, b):
    """Host-side sparse direct solve.

    Inputs are copied first: JAX hands host callbacks numpy views of device
    buffers flagged WRITEBACKIFCOPY/read-only, and scipy's ``spsolve``
    mutates them internally (``sum_duplicates``).  This replaces
    ``jax.experimental.sparse.linalg.spsolve``, whose internal callback hit
    exactly that failure and motivated an earlier process-wide monkey-patch.
    """
    n = int(b.shape[0])
    A = scipy.sparse.csr_matrix(
        (
            np.array(data, dtype=np.float64, copy=True),
            np.array(indices, dtype=np.int32, copy=True),
            np.array(indptr, dtype=np.int32, copy=True),
        ),
        shape=(n, n),
    )
    return scipy.sparse.linalg.spsolve(A, np.array(b, dtype=np.float64, copy=True))


def spsolve(data, indices, indptr, b, tol: float = 1e-6):
    """Sparse direct solve J·x = b from CSR triplets (host callback).

    Drop-in replacement for ``jax.experimental.sparse.linalg.spsolve`` kept
    local to this module.  Not differentiable — under AD use the analytic
    Block-Thomas or dense backends instead.
    """
    del tol  # scipy's splu-based spsolve has no drop tolerance knob
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
    return jax.pure_callback(_spsolve_host, result_shape, data, indices, indptr, b)

_W = 13  # banded width for the nearest-neighbour DDP Jacobian (3 unknowns/node)

# NOTE: an earlier revision monkey-patched ``scipy.sparse.csr_matrix.__init__``
# globally to coerce JAX's read-only WRITEBACKIFCOPY buffers into writable
# numpy arrays.  That patch was process-wide (it affected every scipy user in
# the host application) and has been removed; writability is guaranteed
# locally at the only spsolve call sites in ``linsolve`` below via explicit
# ``np.array(jax.device_get(...), copy=True)`` conversions.

# _W banded width for the nearest-neighbour DDP Jacobian (3 unknowns/node) already defined

# ---------------------------------------------------------------------------
# Banded storage helpers
# ---------------------------------------------------------------------------


def dense_to_banded(dense_mat: jax.Array, width: int = _W) -> jax.Array:
    """Extract the (n, W) banded view of a dense nearest-neighbour Jacobian."""
    n = dense_mat.shape[0]
    half_w = width // 2
    rows = jnp.arange(n)[:, None]
    cols = rows + jnp.arange(width)[None, :] - half_w
    cols_clip = jnp.clip(cols, 0, n - 1)
    vals = dense_mat[rows, cols_clip]
    return jnp.where((cols >= 0) & (cols < n), vals, 0.0)


def banded_to_csr(banded: jax.Array) -> tuple:
    """(data, indices, indptr) CSR of a banded (n, W) matrix."""
    n = banded.shape[0]
    half_w = _W // 2
    rows = jnp.repeat(jnp.arange(n), _W)
    cols_raw = rows + jnp.tile(jnp.arange(_W), n) - half_w
    cols = jnp.clip(cols_raw, 0, n - 1).astype(jnp.int32)
    vals = jnp.where((cols_raw >= 0) & (cols_raw < n), banded.flatten(), 0.0)
    order = jnp.lexsort((cols, rows))
    indptr = jnp.arange(0, n * _W + 1, _W, dtype=jnp.int32)
    return vals[order], cols[order], indptr


def dense_to_csr(dense_J: jax.Array) -> tuple:
    """CSR of a dense Jacobian via its banded envelope (O(n·W))."""
    return banded_to_csr(dense_to_banded(dense_J))


def blocks_to_csr(A, B, C) -> tuple:
    """CSR of the block-tridiagonal Jacobian from its 3×3 blocks (O(N)).

    Assembles the banded (3N,13) window directly from the 3×3 blocks
    without dense (3N,3N) materialization. Prefer ``block_thomas``; use
    this only where CSR is strictly required (legacy ``linsolve(csr)``).
    """
    N = A.shape[0]
    n = 3 * N
    W = _W
    half = W // 2  # 6
    banded = jnp.zeros((n, W), dtype=jnp.float64)
    # Each dense row has at most 9 nonzeros inside the 13-wide band
    for dr in range(3):
        for dc in range(3):
            # diagonal A[i]
            rows_diag = jnp.arange(N, dtype=jnp.int32) * 3 + dr
            bcol_diag = half + dc - dr
            banded = banded.at[rows_diag, bcol_diag].set(A[:, dr, dc])
            if N > 1:
                # super B[i] at (3*i+dr, 3*(i+1)+dc)
                rows_sup = jnp.arange(N - 1, dtype=jnp.int32) * 3 + dr
                bcol_sup = half + 3 + dc - dr
                banded = banded.at[rows_sup, bcol_sup].set(B[:, dr, dc])
                # sub C[i] at (3*(i+1)+dr, 3*i+dc)
                rows_sub = (jnp.arange(N - 1, dtype=jnp.int32) + 1) * 3 + dr
                bcol_sub = half - 3 + dc - dr
                banded = banded.at[rows_sub, bcol_sub].set(C[:, dr, dc])
    return banded_to_csr(banded)


def _banded_matvec(banded_J: jax.Array, x: jax.Array) -> jax.Array:  # deprecated alias for test compatibility
    """banded[i, d] · x[i + d − W/2] (zero-padded). Prefer banded direct solves."""
    n = banded_J.shape[0]
    half_w = _W // 2
    x_pad = jnp.pad(x, half_w)
    def onerow(i):
        return jnp.dot(banded_J[i], lax.dynamic_slice(x_pad, [i], [_W]))
    return vmap(onerow)(jnp.arange(n))


def row_equilibrate(J: jax.Array, b: jax.Array):
    """Row-scale J·x = b to unit max-abs rows (x unchanged; much better
    conditioning for the stiff DDP system)."""
    row_max = jnp.maximum(jnp.max(jnp.abs(J), axis=1), 1e-30)
    D = 1.0 / row_max
    return D[:, None] * J, D * b, D


# ---------------------------------------------------------------------------
# Unified solve
# ---------------------------------------------------------------------------


def linsolve(J, rhs, backend: str = "auto", tol: float = 1e-6):
    """Solve J·x = rhs.  Returns (x, rel_resid, tag)."""
    # Host spsolve is not jit/trace-safe (needs device_get on tracer)
    if backend in ("csr", "auto") and _is_tracer_linalg(J):
        x = jnp.linalg.solve(J, rhs)
        return x, jnp.linalg.norm(J @ x - rhs) / (jnp.linalg.norm(rhs) + 1e-30), "dense-tracer-fallback"
    if backend == "dense":
        x = jnp.linalg.solve(J, rhs)
        # AUDIT: +1e-30 floor (was missing) — at an exact root rhs=0 the
        # residual was 0/0=NaN, poisoning stats and solver-selection gates.
        return x, jnp.linalg.norm(J @ x - rhs) / (jnp.linalg.norm(rhs) + 1e-30), "dense"
    if backend == "banded":
        x = _bt.solve_block_tridiagonal(J, rhs)
        resid = jnp.linalg.norm(J @ x - rhs) / (jnp.linalg.norm(rhs) + 1e-30)
        # H4: mirror the auto-path dense fallback — an ill-conditioned
        # block-Thomas solve previously returned NaN silently.
        if jnp.isnan(resid) or resid > 1e-4:
            x = jnp.linalg.solve(J, rhs)
            resid = jnp.linalg.norm(J @ x - rhs) / (jnp.linalg.norm(rhs) + 1e-30)
            return x, resid, "dense"
        return x, resid, "banded"
    if backend == "csr":
        Je, be, _ = row_equilibrate(J, rhs)
        data, indices, indptr = dense_to_csr(Je)
        # JAX DeviceArray -> numpy is read-only WRITEBACKIFCOPY; scipy needs writable
        data = np.array(jax.device_get(data), dtype=np.float64, copy=True)
        indices = np.array(jax.device_get(indices), dtype=np.int32, copy=True)
        indptr = np.array(jax.device_get(indptr), dtype=np.int32, copy=True)
        x = spsolve(data, indices, indptr, be, tol=tol)
        return x, jnp.linalg.norm(J @ x - rhs) / (jnp.linalg.norm(rhs) + 1e-30), "csr"
    if backend != "auto":
        # AUDIT: unknown backend strings previously fell through silently
        # into the auto-CSR path; fail loudly instead.
        raise ValueError(f"unknown linsolve backend {backend!r} (use 'auto'|'csr'|'banded'|'dense')")
    # auto: equilibrated CSR; dense fallback if the linear residual is bad
    Je, be, _ = row_equilibrate(J, rhs)
    data, indices, indptr = dense_to_csr(Je)
    data = np.array(jax.device_get(data), dtype=np.float64, copy=True)
    indices = np.array(jax.device_get(indices), dtype=np.int32, copy=True)
    indptr = np.array(jax.device_get(indptr), dtype=np.int32, copy=True)
    x = spsolve(data, indices, indptr, be, tol=tol)
    resid = jnp.linalg.norm(J @ x - rhs) / (jnp.linalg.norm(rhs) + 1e-30)
    if jnp.isnan(resid) or resid > 1e-4:
        x = jnp.linalg.solve(J, rhs)
        resid = jnp.linalg.norm(J @ x - rhs) / (jnp.linalg.norm(rhs) + 1e-30)
        return x, resid, "dense"
    return x, resid, "csr"
