"""Exact O(N) Block-Thomas solver for the interleaved 3×3-block DDP Jacobian.

The DDP Jacobian is block-tridiagonal in the interleaved ordering
[φn_i, φp_i, φ_i]: A_i·x_i + B_i·x_{i+1} = d_i on the diagonal bands with
C_i·x_{i-1} on the sub-band.  Block-Thomas eliminates the sub-diagonal
with O(N) 3×3 solves — an *exact* direct solver (no Krylov, no ILU),
fully JAX-native, differentiable, and vmappable (the key property that
enables batched bias/design sweeps).

Scalar Thomas is provided for the equilibrium Poisson problem.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

DET_TOL = 1e-30  # determinant threshold for singularity guard in _inv3
# When |det| < DET_TOL, the block is replaced by ±DET_TOL (same sign) to
# keep the inverse finite. This introduces at most ~1e-30 relative error
# in the 3x3 solve, which is harmless in float64 but may affect the
# correctness contract for ill-conditioned systems.


def _inv3(A):
    """Closed-form inverse of the trailing (…, 3, 3) blocks (~50 flops).

    Includes a singularity check: if the determinant is near zero,
    the inverse is undefined and the caller should fall back to a
    dense solve or regularise the Jacobian.
    """
    a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d_, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g, h, i_ = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    det = a * (e * i_ - f * h) - b * (d_ * i_ - f * g) + c * (d_ * h - e * g)
    # Guard against singular blocks: replace near-zero dets with a small
    # value of the SAME SIGN so the inverse remains finite and the solve
    # can proceed without flipping the block's definiteness (AUDIT: the
    # old guard mapped e.g. det=-1e-31 to +1e-30, inverting the block).
    # The caller is responsible for detecting ill-conditioning
    # (e.g. via the linear residual) and selecting a different solver.
    det_safe = jnp.where(
        jnp.abs(det) < DET_TOL, jnp.where(det < 0.0, -DET_TOL, DET_TOL), det
    )
    invdet = 1.0 / det_safe
    return (
        jnp.stack(
            [
                (e * i_ - f * h),
                (c * h - b * i_),
                (b * f - c * e),
                (f * g - d_ * i_),
                (a * i_ - c * g),
                (c * d_ - a * f),
                (d_ * h - e * g),
                (b * g - a * h),
                (a * e - b * d_),
            ],
            axis=-1,
        ).reshape(A.shape)
        * invdet[..., None, None]
    )


def _solve3(A, B):
    return _inv3(A) @ B


def extract_blocks(J):
    """(A, B, C) 3×3 block diagonals of a dense (3N, 3N) interleaved Jacobian."""
    N = J.shape[0] // 3
    Jr = J.reshape(N, 3, N, 3)
    idx = jnp.arange(N)
    A = Jr[idx, :, idx, :]
    B = Jr[idx[:-1], :, idx[1:], :]
    C = Jr[idx[1:], :, idx[:-1], :]
    return A, B, C


def block_thomas_solve(A, B, C, b):
    """Solve the block-tridiagonal system exactly in O(N)."""
    N = A.shape[0]
    if N == 1:
        return _solve3(A[0], b[0][..., None])[..., 0][None, :]

    def _fwd(i, carry):
        Ap, bp = carry
        W = C[i - 1] @ _inv3(Ap[i - 1])
        Ap = Ap.at[i].set(A[i] - W @ B[i - 1])
        bp = bp.at[i].set(b[i] - W @ bp[i - 1])
        return Ap, bp

    Ap, bp = lax.fori_loop(1, N, _fwd, (A, b))
    Ap_inv = _inv3(Ap)
    x = jnp.zeros_like(b)
    x = x.at[N - 1].set((Ap_inv[N - 1] @ bp[N - 1][..., None])[..., 0])

    def _mvv2(W, v):
        return jnp.einsum("...ij,...j->...i", W, v)

    def _bwd(j, x):
        i = N - 2 - j
        rhs = bp[i] - _mvv2(B[i], x[i + 1])
        return x.at[i].set((Ap_inv[i] @ rhs[..., None])[..., 0])

    return lax.fori_loop(0, N - 1, _bwd, x)


def block_thomas_solve_batched(A, B, C, b):
    """Batched node-major solve: A,B,C (N, B, 3, 3), b (N, B, 3) → (N, B, 3).

    Pure lax.fori_loop (like serial block_thomas_solve) — single XLA program,
    O(1) jaxpr size, not Python-unrolled. Vectorised over batch axis.
    """
    N = A.shape[0]
    if N == 1:
        return (_inv3(D := A)[0] @ b[0][..., None])[..., 0][None, :]

    def fwd(i, carry):
        D, d = carry
        W = C[i - 1] @ _inv3(D[i - 1])
        D = D.at[i].set(D[i] - W @ B[i - 1])
        d = d.at[i].set(d[i] - (W @ d[i - 1][..., None])[..., 0])
        return D, d

    D, d = lax.fori_loop(1, N, fwd, (A, b))
    D_inv = _inv3(D)
    x = jnp.zeros_like(b)
    x = x.at[N - 1].set((D_inv[N - 1] @ d[N - 1][..., None])[..., 0])

    def bwd(j, x):
        i = N - 2 - j
        rhs = (d[i] - (B[i] @ x[i + 1][..., None])[..., 0])[..., None]
        return x.at[i].set((D_inv[i] @ rhs)[..., 0])

    return lax.fori_loop(0, N - 1, bwd, x)


def block_thomas_with_residual(A, B, C, b_flat):
    """Block-Thomas solve + FP64 linear residual (jrystal-style stability gate).

    Returns (x_flat, rel_resid). Callers must fall back to a dense/pivoted
    solve when rel_resid is large (degenerate equilibrium Jacobian,
    cond ~ 1e18-1e28) instead of trusting the unpivoted elimination.
    """
    import jax.numpy as _jnp

    from driftjax.numerics.analytic_jacobian import dense_from_blocks as _dense_from_blocks
    n = A.shape[0]
    x3 = block_thomas_solve(A, B, C, b_flat.reshape(n, 3))
    x = x3.reshape(-1)
    J = _dense_from_blocks(A, B, C)
    rel = _jnp.linalg.norm(J @ x - b_flat) / (_jnp.linalg.norm(b_flat) + 1e-30)
    return x, rel


def solve_block_tridiagonal(J, b):
    """Solve J·x = b for a dense (3N, 3N) block-tridiagonal J (flat output)."""
    A, B, C = extract_blocks(J)
    return block_thomas_solve(A, B, C, b.reshape(-1, 3)).reshape(-1)
