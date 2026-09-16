"""JAX-native banded solver (dense reconstruction, no host callback)."""
from __future__ import annotations
import jax
import jax.numpy as jnp
KL, KU = 5, 5


def _dense_from_blocks(A, B, C):
    """(A,B,C) 3x3 blocks (n,3,3) -> full (3n,3n) dense banded matrix."""
    n = A.shape[0]; N = 3 * n
    M = jnp.zeros((N, N), dtype=A.dtype)
    idx = jnp.arange(n); off = jnp.arange(3)
    rio = (3*idx)[:,None,None] + off[None,:,None]
    cjo = (3*idx)[:,None,None] + off[None,None,:]
    rr = jnp.broadcast_to(rio, (n,3,3)).reshape(-1)
    cc = jnp.broadcast_to(cjo, (n,3,3)).reshape(-1)
    M = M.at[rr, cc].set(A.reshape(-1))
    if n > 1:
        ip = jnp.arange(n-1); si = 3*ip
        rioB = si[:,None,None] + off[None,:,None]
        cjoB = (3*(ip+1))[:,None,None] + off[None,None,:]
        M = M.at[jnp.broadcast_to(rioB,(n-1,3,3)).reshape(-1), jnp.broadcast_to(cjoB,(n-1,3,3)).reshape(-1)].set(B.reshape(-1))
        rioC = (3*(ip+1))[:,None,None] + off[None,:,None]
        cjoC = si[:,None,None] + off[None,None,:]
        M = M.at[jnp.broadcast_to(rioC,(n-1,3,3)).reshape(-1), jnp.broadcast_to(cjoC,(n-1,3,3)).reshape(-1)].set(C.reshape(-1))
    return M


def native_banded_solve(A, B, C, b, *, tol=1e-12):
    """Native (no-callback) solve of J x = b via dense reconstruction +
    jnp.linalg.lstsq (SVD), which returns the exact solve for nonsingular M
    and the min-norm least-squares solution for singular M -- identical in
    method-selection to scipy.linalg.solve_banded -> numpy.linalg.lstsq
    fallback. Stays inside XLA (differentiable, no host round-trip).

    Returns (x, {ok, singular, method, rank}). See tests/unit/test_banded_native.py.
    """
    n = A.shape[0]
    M = _dense_from_blocks(A, B, C)
    b_flat = b.reshape(-1)
    x_ls, res, rank, sv = jnp.linalg.lstsq(M, b_flat[:, None])
    x = x_ls[:, 0].reshape(n, 3)
    ok = (rank >= 3 * n)
    return x, {"ok": ok, "singular": ~ok, "method": "native_lstsq", "rank": rank}


def native_banded_transpose(A, B, C, g):
    """J^T lam = g via dense reconstruction of J^T + native lstsq."""
    M = _dense_from_blocks(A, B, C).T
    g_flat = g.reshape(-1)[:, None]
    x_ls, res, rank, sv = jnp.linalg.lstsq(M, g_flat)
    n = A.shape[0]
    lam = x_ls[:, 0].reshape(n, 3)
    ok = (rank >= 3 * n)
    return lam, {"ok": ok, "singular": ~ok, "method": "native_lstsq", "rank": rank}
