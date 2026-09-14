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
    det_safe = jnp.where(jnp.abs(det) < DET_TOL, jnp.where(det < 0.0, -DET_TOL, DET_TOL), det)
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


# ---------------------------------------------------------------------------
# Pivoted banded solver (LAPACK dgbsv via scipy)
# ---------------------------------------------------------------------------


def _blocks_to_lapack_banded(A, B, C):
    """Convert (A, B, C) block-tridiagonal to LAPACK banded storage ab(kl+ku+1, 3N).

    For interleaved 3×3 blocks the bandwidth is kl=ku=5 (each block spans
    3 diagonals; super/sub-diagonal blocks shift by ±3 rows).
    LAPACK column-major storage: ab[ku + i - j, j] = M[i, j].
    """
    import jax.numpy as _jnp

    n = A.shape[0]
    N = 3 * n
    kl, ku = 5, 5
    ab = _jnp.zeros((kl + ku + 1, N), dtype=A.dtype)

    idx = _jnp.arange(n)

    # Diagonal blocks A[i]: block row = block col = i
    for di in range(3):
        for dj in range(3):
            row = ku + (di - dj)
            cols = 3 * idx + dj
            ab = ab.at[row, cols].set(A[idx, di, dj])

    # Super-diagonal blocks B[i]: block row = i, block col = i+1
    for di in range(3):
        for dj in range(3):
            row = ku + (di - dj - 3)
            cols = 3 * idx + 3 + dj
            mask = (3 * idx + 3 + dj) < N
            ab = ab.at[row, cols].set(jnp.where(mask, B[idx, di, dj], 0.0))

    # Sub-diagonal blocks C[i]: block row = i+1, block col = i
    for di in range(3):
        for dj in range(3):
            row = ku + (di + 3 - dj)
            cols = 3 * idx + dj
            mask = (3 * idx + 3 + di) < N
            ab = ab.at[row, cols].set(jnp.where(mask, C[idx, di, dj], 0.0))

    return ab, kl, ku


def banded_solve(A, B, C, b):
    """Solve block-tridiagonal system via pivoted LAPACK banded (dgbsv).

    Converts (A, B, C, b) → LAPACK banded storage and calls
    scipy.linalg.solve_banded with partial pivoting.  This is O(N·κ²)
    but the Fortran BLAS/LAPACK implementation is fast enough to beat
    the unpivoted O(N) Block-Thomas in practice (especially on
    ill-conditioned heterojunctions where BT fails entirely).

    Returns x with shape (n, 3) matching b.shape.  If the banded matrix
    contains non-finite values (e.g. NaN from degenerate Jacobian blocks),
    returns zeros — the caller should detect this via the residual check
    and fall back to a dense pivoted solve.
    """
    import jax

    n = A.shape[0]
    banded, kl, ku = _blocks_to_lapack_banded(A, B, C)
    b_flat = b.reshape(-1)

    def _solve(ab_flat, b_in):
        import numpy as _np

        ab_np = ab_flat.reshape(kl + ku + 1, 3 * n)
        try:
            from scipy.linalg import solve_banded as _sb

            x_np = _sb((kl, ku), ab_np, b_in)
        except Exception:
            x_np = _np.zeros_like(b_in)
        return x_np.astype(ab_flat.dtype)

    def _do_solve(_):
        return jax.pure_callback(
            _solve,
            jax.ShapeDtypeStruct(b_flat.shape, b_flat.dtype),
            banded.reshape(-1),
            b_flat,
            vmap_method="sequential",
        )

    def _zeros(_):
        return jnp.zeros_like(b_flat)

    has_nan = ~jnp.all(jnp.isfinite(banded))
    x = jax.lax.cond(has_nan, _zeros, _do_solve, None)
    return x.reshape(n, 3)


def banded_solve_batched(A, B, C, b):
    """Batched pivoted banded solve: (N, B, 3, 3) blocks, (N, B, 3) rhs.

    Vmaps banded_solve over the batch axis.  Each batch member is solved
    independently via LAPACK dgbsv with partial pivoting.
    """
    import jax

    # Transpose from (N, B, ...) to (B, N, ...) for vmap
    A_b = jax.tree.map(lambda x: x.transpose(1, 0, 2, 3), A)
    B_b = jax.tree.map(lambda x: x.transpose(1, 0, 2, 3), B)
    C_b = jax.tree.map(lambda x: x.transpose(1, 0, 2, 3), C)
    b_b = b.transpose(1, 0, 2)  # (N, B, 3) → (B, N, 3)

    x_b = jax.vmap(lambda a, bb, cc, rhs: banded_solve(a, bb, cc, rhs))(A_b, B_b, C_b, b_b)
    return x_b.transpose(1, 0, 2)  # (B, N, 3) → (N, B, 3)
