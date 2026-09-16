"""Pure-JAX pivoted banded GE via ``lax.scan`` (block Thomas algorithm).

Replaces the O(N^3) SVD native solve with O(N * bw^2) GE that stays
inside XLA (no host callback) and is differentiable.

Algorithm: block-tridiagonal Gaussian elimination with 3x3 partial
pivoting (jnp.linalg.solve at each step) in lax.scan. Singularity is
detected via per-block determinant threshold during forward elimination —
no SVD computation needed for well-conditioned systems.

See: tests/unit/test_banded_ge.py, tests/unit/test_banded_native.py
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def _block_thomas(A, B, C, b):
    """Block Thomas elimination + back-substitution, all in lax.scan.

    Forward scan (blocks 0..n-2):
        M[i] = A_prime_i^{-1} @ B[i]     (3x3 pivoted solve)
        z[i] = A_prime_i^{-1} @ b_prime_i  (3x3 pivoted solve)
        A_prime_{i+1} = A[i+1] - C[i] @ M[i]   (Schur complement)
        b_prime_{i+1} = b[i+1] - C[i] @ z[i]

    Backward scan (blocks n-2..0):
        x[i] = A_prime_i^{-1} @ (b_prime_i - B[i] @ x[i+1])

    A: (n,3,3)  B: (n,3,3)  C: (n-1,3,3)  b: (n,3)
    Returns (x_ge, A_mod_full, b_mod_full, block_dets).
    block_dets: (n,) per-block det for singularity check.
    """
    n = A.shape[0]

    def _forward(carry, i):
        Ai, bi = carry
        Bi = B[i]
        Ci = C[i]

        Mi = jnp.linalg.solve(Ai, Bi)
        zi = jnp.linalg.solve(Ai, bi)

        Ai_next = A[i + 1] - Ci @ Mi
        bi_next = b[i + 1] - Ci @ zi

        det_next = jnp.linalg.det(Ai_next)
        return (Ai_next, bi_next), (Ai, bi, det_next)

    (A_last, b_last), (A_mod, b_mod, dets) = lax.scan(
        _forward, (A[0], b[0]), jnp.arange(n - 1)
    )

    A_mod_full = jnp.concatenate([A_mod, A_last[None, :, :]], axis=0)
    b_mod_full = jnp.concatenate([b_mod, b_last[None, :]], axis=0)

    # dets already includes det(A'_1..A'_{n-1}) from the forward scan.
    # Prepend det(A'_0) = det(A[0]) (first block is unmodified).
    all_dets = jnp.concatenate([jnp.linalg.det(A[0:1]), dets], axis=0)

    def _backward(carry, i):
        x_next = carry
        Ai = A_mod_full[i]
        Bi = B[i]
        rhs = b_mod_full[i] - Bi @ x_next
        xi = jnp.linalg.solve(Ai, rhs)
        return xi, xi

    x_last = jnp.linalg.solve(A_mod_full[n - 1], b_mod_full[n - 1])
    indices = jnp.arange(n - 2, -1, -1)
    _, x_stacked = lax.scan(_backward, x_last, indices)
    x_rest_rev = x_stacked[::-1]
    x_ge = jnp.concatenate([x_rest_rev, x_last[None, :]], axis=0)

    return x_ge, A_mod_full, b_mod_full, all_dets


def banded_ge_solve(A, B, C, b, *, tol=1e-12):
    """Pure-JAX pivoted block-banded GE via lax.scan.

    No host callbacks. Differentiable via jax.grad/jax.jvp.

    Singularity is detected via per-block determinant threshold during
    forward elimination -- no SVD fallback path is needed for the common
    case. When a block is found singular (det near zero), ok=False
    and the result may be non-finite; the caller routes to dense LU.

    Args:
        A: (n, 3, 3) diagonal blocks
        B: (n, 3, 3) super-diagonal blocks (B[-1] unused)
        C: (n-1, 3, 3) sub-diagonal blocks
        b: (n, 3) RHS
        tol: determinant magnitude threshold for singularity

    Returns:
        (x, info) where info = {"ok", "singular", "method", "rank"}
    """
    n = A.shape[0]
    N = 3 * n

    x_ge, A_mod_full, b_mod_full, block_dets = _block_thomas(A, B, C, b)
    ge_finite = (
        jnp.all(jnp.isfinite(x_ge))
        & jnp.all(jnp.isfinite(A_mod_full))
        & jnp.all(jnp.isfinite(b_mod_full))
    )

    det_min = jnp.min(jnp.abs(block_dets))
    det_scale = jnp.max(jnp.abs(block_dets)) + 1e-30
    nonsingular_blocks = det_min / det_scale > tol
    ok = ge_finite & nonsingular_blocks

    info = {
        "ok": ok,
        "singular": ~ok,
        "method": jnp.where(ok, jnp.int32(0), jnp.int32(1)),  # 0=GE, 1=failed
        "rank": jnp.where(ok, jnp.int32(N), jnp.int32(0)),
    }
    return x_ge, info


def _dense_from_blocks(A, B, C):
    """(A,B,C) 3x3 blocks (n,3,3) -> full (3n,3n) dense banded matrix."""
    n = A.shape[0]
    N = 3 * n
    M = jnp.zeros((N, N), dtype=A.dtype)
    idx = jnp.arange(n)
    off = jnp.arange(3)
    rio = (3 * idx)[:, None, None] + off[None, :, None]
    cjo = (3 * idx)[:, None, None] + off[None, None, :]
    rr = jnp.broadcast_to(rio, (n, 3, 3)).reshape(-1)
    cc = jnp.broadcast_to(cjo, (n, 3, 3)).reshape(-1)
    M = M.at[rr, cc].set(A.reshape(-1))
    if n > 1:
        ip = jnp.arange(n - 1)
        si = 3 * ip
        rioB = si[:, None, None] + off[None, :, None]
        cjoB = (3 * (ip + 1))[:, None, None] + off[None, None, :]
        M = M.at[jnp.broadcast_to(rioB, (n - 1, 3, 3)).reshape(-1),
                 jnp.broadcast_to(cjoB, (n - 1, 3, 3)).reshape(-1)].set(B.reshape(-1))
        rioC = (3 * (ip + 1))[:, None, None] + off[None, :, None]
        cjoC = si[:, None, None] + off[None, None, :]
        M = M.at[jnp.broadcast_to(rioC, (n - 1, 3, 3)).reshape(-1),
                 jnp.broadcast_to(cjoC, (n - 1, 3, 3)).reshape(-1)].set(C.reshape(-1))
    return M
