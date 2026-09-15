"""Pivoted banded linear solver for the interleaved 3×3-block DDP Jacobian.

The DDP Jacobian is block-tridiagonal in the interleaved ordering
[φn_i, φp_i, φ_i]: A_i·x_i + B_i·x_{i+1} = d_i on the diagonal bands with
C_i·x_{i-1} on the sub-band (bandwidth kl=ku=5, bw=11).

Solver: LAPACK dgbsv with partial pivoting (via scipy) — works on all
devices including ill-conditioned heterojunctions where unpivoted
elimination fails.  O(N·bw²) flop complexity, i.e. O(121·N), independent
of the condition number κ (which affects accuracy, not flop count).
"""

from __future__ import annotations

import jax.numpy as jnp


def extract_blocks(J):
    """(A, B, C) 3×3 block diagonals of a dense (3N, 3N) interleaved Jacobian."""
    N = J.shape[0] // 3
    Jr = J.reshape(N, 3, N, 3)
    idx = jnp.arange(N)
    A = Jr[idx, :, idx, :]
    B = Jr[idx[:-1], :, idx[1:], :]
    C = Jr[idx[1:], :, idx[:-1], :]
    return A, B, C


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


def _banded_to_dense(ab_np, kl, ku, n):
    """Reconstruct full dense (3N, 3N) matrix from LAPACK banded storage."""
    import numpy as _np

    N = 3 * n
    M = _np.zeros((N, N), dtype=ab_np.dtype)
    for j in range(N):
        k1 = max(0, j - ku)
        k2 = min(N - 1, j + kl)
        for k in range(k1, k2 + 1):
            M[k, j] = ab_np[ku + k - j, j]
    return M


def banded_solve(A, B, C, b):
    """Solve block-tridiagonal system via pivoted LAPACK banded (dgbsv).

    Converts (A, B, C, b) → LAPACK banded storage and calls
    scipy.linalg.solve_banded with partial pivoting.  Flop cost is
    O(N·kl·ku) with kl=ku=5 (bandwidth 11), independent of κ;
    the Fortran BLAS/LAPACK implementation is fast enough to beat
    unpivoted methods in practice (especially on ill-conditioned
    heterojunctions where unpivoted elimination fails entirely).

    Fallback: when dgbsv raises LinAlgError (singular matrix), reconstructs
    the full dense matrix and uses numpy lstsq (truncated SVD) which
    handles rank-deficient systems by returning the minimum-norm
    least-squares solution.

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
            try:
                M = _banded_to_dense(ab_np, kl, ku, n)
                x_np, *_ = _np.linalg.lstsq(M, b_in, rcond=None)
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


def banded_transpose(A, B, C, g):
    """Solve J^T·x = g for block-tridiagonal J via pivoted banded (dgbsv).

    J^T is block-tridiagonal with diagonal blocks A^T, super-diagonal C^T
    and sub-diagonal B^T; the pivoted solve applies unchanged.
    """
    import jax.numpy as _jnp

    At = _jnp.transpose(A, (0, 2, 1))
    Ct = _jnp.transpose(C, (0, 2, 1))
    Bt = _jnp.transpose(B, (0, 2, 1))
    return banded_solve(At, Ct, Bt, g.reshape(A.shape[0], 3)).reshape(-1)


def _row_equilibrate(J):
    """Row equilibration: (Je, Dr) with Je = Dr @ J, Dr = diag(1/rowmax)."""
    s = jnp.max(jnp.abs(J), axis=1)
    s = jnp.where(jnp.isfinite(s) & (s > 0), s, jnp.asarray(1.0, dtype=s.dtype))
    Dr = 1.0 / s
    return Dr[:, None] * J, Dr


def adjoint_banded_solve(J, g, tol=1e-8):
    """Solve J^T·lam = g via equilibrated pivoted banded transpose.

    Row equilibration tames the SRV-contact row-scaling imbalance before
    the banded transpose solve; the solution is unscaled (lam = Dr @ y)
    and checked against the ORIGINAL system. Returns (lam, used_fallback):
    used_fallback is True when the residual exceeds tol or is non-finite,
    in which case lam is zeros and the caller must route to dense LU.

    Caution (v0.1.12 lesson): a residual gate cannot certify adjoint
    accuracy under severe ill-conditioning (kappa amplifies small
    residuals into large solution errors). This path is opt-in
    (DRIFTJAX_BANDED_ADJOINT=1); dense LU remains the default.
    """
    import jax
    from scipy.linalg import solve_banded as _sb

    Je, Dr = _row_equilibrate(J)
    A, B, C = extract_blocks(Je)
    banded, kl, ku = _blocks_to_lapack_banded(A, B, C)
    # Transposed-band storage of the equilibrated blocks = band storage
    # of (Je^T); solve Je^T y = g then unscale lam = Dr y.
    At = jnp.transpose(A, (0, 2, 1))
    Ct = jnp.transpose(C, (0, 2, 1))
    Bt = jnp.transpose(B, (0, 2, 1))
    banded_t, _, _ = _blocks_to_lapack_banded(At, Ct, Bt)
    g_flat = g.reshape(-1)

    def _solve(ab_flat, b_in):
        import numpy as _np

        ab_np = ab_flat.reshape(kl + ku + 1, -1)
        try:
            x_np = _sb((kl, ku), ab_np, b_in)
        except Exception:
            x_np = _np.zeros_like(b_in)
        return x_np.astype(ab_flat.dtype)

    def _do_solve(_):
        return jax.pure_callback(
            _solve,
            jax.ShapeDtypeStruct(g_flat.shape, g_flat.dtype),
            banded_t.reshape(-1),
            g_flat,
            vmap_method="sequential",
        )

    def _zeros(_):
        return jnp.zeros_like(g_flat)

    has_nan = ~jnp.all(jnp.isfinite(banded_t))
    y = jax.lax.cond(has_nan, _zeros, _do_solve, None)
    lam = (Dr * y).reshape(g.shape)
    resid = jnp.linalg.norm(J.T @ lam.reshape(-1) + (-g.reshape(-1))) / (
        jnp.linalg.norm(g_flat) + 1e-30
    )
    fallback = has_nan | (~jnp.isfinite(resid)) | (resid > tol)
    return jnp.where(fallback, jnp.zeros_like(lam), lam), fallback


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
