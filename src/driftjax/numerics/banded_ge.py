'''Pure-JAX row-equilibrated block Thomas GE via ``lax.scan`` (local 3x3 pivoted solves).

Replaces the O(N^3) SVD native solve with O(N * bw^2) GE that stays
inside XLA (no host callback) and is differentiable.

Algorithm: block-tridiagonal Gaussian elimination with local 3x3
pivoted solves (jnp.linalg.solve at each step) in lax.scan. Singularity is
detected via per-block determinant ratio threshold during forward elimination
— no SVD computation needed. The rank_flag field is binary (N or 0),
NOT a numerically estimated rank (block-factorization singularity heuristic).

Row equilibration (Δ — NEW in v0.1.18b): per-scalar-row diag(D) from block
entries tames the ~30-order-of-magnitude SRV/carrier imbalance at ohmic
contacts, cutting stress-device Jacobian floor from κ~4e13 to κ~1 without
leaving XLA. The scaled system D·J·x = D·b has identical solution x (since
D is diagonal and non-singular); the GE operates on the scaled blocks and
the solution is returned in original units.

See: tests/unit/test_banded_ge.py, tests/unit/test_banded_native.py
'''

from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def _scalar_row_scales(A, B, C):
    '''Per-SCALAR-row row-equilibration from blocks alone (O(N·bw), no dense J).

    Returns (N,) diagonal D where D[i] = 1 / max|row_i of [A,B,C]|.
    One scale per scalar row (not per 3-row block) — the finer granularity
    is needed for the ~30-order SRV/contact row-scaling imbalance.

    Block-row i spans rows 3i, 3i+1, 3i+2 of the dense Jacobian, which
    receive contributions from A[i,:], B[i,:] (i < n-1), and C[i-1,:] (i > 0).
    '''
    n = A.shape[0]
    # Per-scalar-row max-abs from A blocks: A is (n, 3, 3) → max over cols (axis=2)
    rmax = jnp.max(jnp.abs(A), axis=2)  # (n, 3)
    if n > 1:
        bmax = jnp.max(jnp.abs(B), axis=2)  # (n-1, 3)
        cmax = jnp.max(jnp.abs(C), axis=2)  # (n-1, 3)
        rmax = rmax.at[:-1].set(jnp.maximum(rmax[:-1], bmax))  # block-row i gets B[i]
        rmax = rmax.at[1:].set(jnp.maximum(rmax[1:], cmax))  # block-row i gets C[i-1]
    rmax_flat = rmax.reshape(-1)  # (N,)
    safe = jnp.where(
        jnp.isfinite(rmax_flat) & (rmax_flat > 0),
        rmax_flat,
        jnp.asarray(1.0, dtype=rmax_flat.dtype),
    )
    return 1.0 / safe


def _apply_row_equil(A, B, C, b, dr):
    '''Apply (N,) row scale D to block arrays. D reshapes to (n, 3, 3) → (n,3,1).'''
    n = A.shape[0]
    dr_blk = dr.reshape(n, 3)  # (n, 3) per-scalar-row within each block
    Ae = A * dr_blk[:, :, None]  # (n,3,3) * (n,3,1)
    Be = B * dr_blk[:-1, :, None] if n > 1 else B
    Ce = C * dr_blk[1:, :, None] if n > 1 else C
    be = b * dr_blk
    return Ae, Be, Ce, be


def _block_thomas(A, B, C, b):
    '''Block Thomas elimination + back-substitution, all in lax.scan.

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
    '''
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

    (A_last, b_last), (A_mod, b_mod, dets) = lax.scan(_forward, (A[0], b[0]), jnp.arange(n - 1))

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


def _block_thomas_transpose(A, B, C, g):
    '''Transpose (adjoint) block Thomas elimination via lax.scan.

    Solves J^T y = g where J is block-tridiagonal with blocks A (diag),
    B (super-diagonal), C (sub-diagonal). The transpose J^T has the same
    block structure but with transposed blocks and swapped B/C roles:
        J^T diagonal blocks: A^T (at positions i)
        J^T super-diagonal: C^T (C was sub-diagonal, transposed becomes super)
        J^T sub-diagonal: B^T (B was super-diagonal, transposed becomes sub)

    All in lax.scan, no host callbacks. Differentiable.

    A: (n,3,3)  B: (n,3,3)  C: (n-1,3,3)  g: (n,3)
    Returns (y, A_mod_full, g_mod_full, block_dets)
    '''
    import jax.numpy as jnp
    from jax import lax

    n = A.shape[0]
    # Transposed blocks
    At = jnp.transpose(A, (0, 2, 1))  # A^T on diagonal
    Ct = jnp.transpose(C, (0, 2, 1))  # C^T becomes super-diagonal
    Bt = jnp.transpose(B, (0, 2, 1))  # B^T becomes sub-diagonal

    def _forward(carry, i):
        Ai, bi = carry
        Bi = Bt[i]  # sub-diagonal of J^T (to be eliminated)
        Ci = Ct[i]  # super-diagonal of J^T

        # Forward elimination of sub-diagonal B^T:
        # A_{i+1}^T' = A_{i+1}^T - B_i^T @ (A_i^T)^{-1} @ C_i^T
        Mi = jnp.linalg.solve(Ai, Ci)  # (A_i^T)^{-1} @ C_i^T
        zi = jnp.linalg.solve(Ai, bi)  # (A_i^T)^{-1} @ g_i'

        Ai_next = At[i + 1] - Bi @ Mi  # A_{i+1}^T - B_i^T @ (A_i^T)^{-1} @ C_i^T
        bi_next = g[i + 1] - Bi @ zi    # g_{i+1} - B_i^T @ (A_i^T)^{-1} @ g_i' 

        det_next = jnp.linalg.det(Ai_next)
        return (Ai_next, bi_next), (Ai, bi, det_next)

    (A_last, b_last), (A_mod, b_mod, dets) = lax.scan(_forward, (At[0], g[0]), jnp.arange(n - 1))

    A_mod_full = jnp.concatenate([A_mod, A_last[None, :, :]], axis=0)
    b_mod_full = jnp.concatenate([b_mod, b_last[None, :]], axis=0)
    all_dets = jnp.concatenate([jnp.linalg.det(At[0:1]), dets], axis=0)

    def _backward(carry, i):
        x_next = carry
        Ai = A_mod_full[i]
        Bi = Ct[i]  # super-diagonal of J^T (C^T) for back-substitution
        rhs = b_mod_full[i] - Bi @ x_next
        xi = jnp.linalg.solve(Ai, rhs)
        return xi, xi

    x_last = jnp.linalg.solve(A_mod_full[n - 1], b_mod_full[n - 1])
    indices = jnp.arange(n - 2, -1, -1)
    _, x_stacked = lax.scan(_backward, x_last, indices)
    x_rest_rev = x_stacked[::-1]
    y = jnp.concatenate([x_rest_rev, x_last[None, :]], axis=0)

    return y, A_mod_full, b_mod_full, all_dets


def banded_ge_solve_transpose(A, B, C, g, *, tol=1e-12, equilibrate=True):
    '''Native GE transpose solve for the adjoint: J^T y = g.

    Same algorithm as ``banded_ge_solve`` but on the transposed system.
    Row equilibration is applied to the FORWARD blocks (dr from A,B,C),
    then D^T = D (diagonal), so the scaled adjoint system is:
        (D J)^T y = g  =>  J^T y = g (solution-preserving).

    The row scale D from the forward system becomes a column scale of J^T,
    which is a valid (right) equilibration for the transpose solve.

    No host callbacks. Differentiable via jax.grad/jax.jvp.

    Returns (y, info) where info = {'ok', 'singular', 'method', 'rank_flag'}
    '''
    n = A.shape[0]
    N = 3 * n

    if equilibrate:
        dr = _scalar_row_scales(A, B, C)  # (N,) per-scalar-row
        dr_blk = dr.reshape(n, 3)
        # Apply row equilibration D to the matrix blocks: (D*J) has same null
        # space as J but better-conditioned.  For the transpose we solve
        # (D*J)^T z = g  ==>  J^T D z = g  ==>  y = D z  is the solution of J^T y = g.
        Ae = A * dr_blk[:, :, None]
        Be = B * dr_blk[:-1, :, None] if n > 1 else B
        Ce = C * dr_blk[1:, :, None] if n > 1 else C
    else:
        dr = jnp.ones(N)
        dr_blk = jnp.ones((n, 3))
        Ae, Be, Ce = A, B, C

    # Reshape g from flat (3n,) to block (n,3) for block-structured arithmetic.
    g_blk = g.reshape(n, 3)

    # Solve the EQUILIBRATED transposed system (D*J)^T z = g (original RHS).
    y_ge, A_mod_full, g_mod_full, block_dets = _block_thomas_transpose(Ae, Be, Ce, g_blk)

    # Recover solution of original system: y = D * z  (since J^T D z = g).
    lam = y_ge * dr_blk  # (n,3)
    lam = lam.reshape(N)  # back to flat (3n,)

    ge_finite = (
        jnp.all(jnp.isfinite(y_ge))
        & jnp.all(jnp.isfinite(A_mod_full))
        & jnp.all(jnp.isfinite(g_mod_full))
    )

    det_min = jnp.min(jnp.abs(block_dets))
    det_scale = jnp.max(jnp.abs(block_dets)) + 1e-30
    nonsingular_blocks = det_min / det_scale > tol
    ok = ge_finite & nonsingular_blocks

    info = {
        'ok': ok,
        'singular': ~ok,
        'method': jnp.where(ok, jnp.int32(0), jnp.int32(1)),
        'rank_flag': jnp.where(ok, jnp.int32(N), jnp.int32(0)),  # binary: N if ok, 0 if singular (NOT a rank estimate)
    }
    return lam, info


def banded_ge_solve(A, B, C, b, *, tol=1e-12, equilibrate=True):
    '''Pure-JAX pivoted block-banded GE via lax.scan, with optional row equilibration.

    No host callbacks. Differentiable via jax.grad/jax.jvp.

    Row equilibration (Δ — v0.1.18b): when ``equilibrate=True``, per-scalar-row
    diagonal scaling is applied from the block entries alone (O(N·bw), no dense
    Jacobian). This tames the SRV/contact row-scaling imbalance that causes
    κ~1e13-1e45 on stiff devices (ohmic contacts, perovskite stress tests),
    cutting the effective condition number to κ~1-100 without leaving XLA.

    The scaled system D·J·x = D·b has the identical solution x (D is diagonal
    and non-singular), so equilibration is solution-preserving by construction.

    Singularity is detected via per-block determinant ratio threshold during
    forward elimination — no SVD fallback path is needed for the common
    case. The rank_flag field in info is binary (N if ok, 0 if singular),
    NOT a numerically estimated rank (block-factorization singularity heuristic).
    When a block is found singular, ok=False and the result may be
    non-finite; the caller routes to dense LU.

    Args:
        A: (n, 3, 3) diagonal blocks
        B: (n, 3, 3) super-diagonal blocks (B[-1] unused)
        C: (n-1, 3, 3) sub-diagonal blocks
        b: (n, 3) RHS
        tol: per-block determinant ratio threshold for singularity
        equilibrate: whether to apply row equilibration (default True)

    Returns:
        (x, info) where info = {'ok', 'singular', 'method', 'rank_flag'}
    '''
    n = A.shape[0]
    N = 3 * n

    if equilibrate:
        dr = _scalar_row_scales(A, B, C)  # (N,) per-scalar-row
        Ae, Be, Ce, be = _apply_row_equil(A, B, C, b, dr)
    else:
        Ae, Be, Ce, be = A, B, C, b

    x_ge, A_mod_full, b_mod_full, block_dets = _block_thomas(Ae, Be, Ce, be)

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
        'ok': ok,
        'singular': ~ok,
        'method': jnp.where(ok, jnp.int32(0), jnp.int32(1)),  # 0=GE, 1=failed
        'rank_flag': jnp.where(ok, jnp.int32(N), jnp.int32(0)),  # binary: N if ok, 0 if singular (NOT a rank estimate)
    }
    return x_ge, info


def _dense_from_blocks(A, B, C):
    '''(A,B,C) 3x3 blocks (n,3,3) -> full (3n,3n) dense banded matrix.'''
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
        M = M.at[
            jnp.broadcast_to(rioB, (n - 1, 3, 3)).reshape(-1),
            jnp.broadcast_to(cjoB, (n - 1, 3, 3)).reshape(-1),
        ].set(B.reshape(-1))
        rioC = (3 * (ip + 1))[:, None, None] + off[None, :, None]
        cjoC = si[:, None, None] + off[None, None, :]
        M = M.at[
            jnp.broadcast_to(rioC, (n - 1, 3, 3)).reshape(-1),
            jnp.broadcast_to(cjoC, (n - 1, 3, 3)).reshape(-1),
        ].set(C.reshape(-1))
    return M
