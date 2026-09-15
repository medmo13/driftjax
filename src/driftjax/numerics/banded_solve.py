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
    # Vectorized stamp: one scatter per block family (3 dispatches total,
    # not 27). (DI, DJ) enumerate the 3x3 intra-block offsets; rows/cols
    # broadcast to (9, n) and values are the transposed blocks.
    _DI, _DJ = _jnp.meshgrid(_jnp.arange(3), _jnp.arange(3), indexing="ij")
    _P = (_DI.reshape(9, 1), _DJ.reshape(9, 1))

    # Diagonal blocks A[i]: block row = block col = i
    rows = ku + _P[0] - _P[1]
    cols = 3 * idx[None, :] + _P[1]
    ab = ab.at[(rows, cols)].set(A.transpose(1, 2, 0).reshape(9, n))

    # Super-diagonal blocks B[i]: block row = i, block col = i+1.
    # B has n-1 blocks; pad one zero block so the (9, n) stamp has matching
    # shape — the mask zeroes column n-1 exactly as the scalar loop did.
    _Z = _jnp.zeros((1, 3, 3), dtype=A.dtype)
    Bp = _jnp.concatenate([B, _Z], axis=0)
    rows_b = ku + _P[0] - _P[1] - 3
    cols_b = 3 * idx[None, :] + 3 + _P[1]
    mask_b = cols_b < N
    ab = ab.at[(rows_b, cols_b)].set(_jnp.where(mask_b, Bp.transpose(1, 2, 0).reshape(9, n), 0.0))

    # Sub-diagonal blocks C[i]: block row = i+1, block col = i (same pad).
    Cp = _jnp.concatenate([C, _Z], axis=0)
    rows_c = ku + _P[0] + 3 - _P[1]
    cols_c = 3 * idx[None, :] + _P[1]
    mask_c = (3 * idx[None, :] + 3 + _P[0]) < N
    ab = ab.at[(rows_c, cols_c)].set(_jnp.where(mask_c, Cp.transpose(1, 2, 0).reshape(9, n), 0.0))

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


def banded_solve_with_info(A, B, C, b):
    """banded_solve + provenance: (x, info) with ``used_lstsq``/``used_zeros``.

    ``used_lstsq`` is True when LAPACK dgbsv raised (singular matrix) and the
    truncated-SVD least-squares fallback produced the step; ``used_zeros`` is
    True when the banded storage was non-finite and zeros were returned. Both
    are trace-safe boolean arrays so they can ride a ``lax.while_loop`` carry
    into Newton statistics (R2 provenance: which solver actually ran).

    Opt-in native path (``DRIFTJAX_NATIVE_BANDED=1``): tries the JAX-native
    Givens QR solver first (no host callback); on ``ok=False`` falls through
    to the callback path below, preserving all flags and fallbacks. The env
    var is read at trace time (static); the default path is bit-identical
    to before.
    """
    import os

    import jax

    n = A.shape[0]
    banded, kl, ku = _blocks_to_lapack_banded(A, B, C)
    b_flat = b.reshape(-1)

    def _solve(ab_flat, b_in):
        import numpy as _np

        ab_np = ab_flat.reshape(kl + ku + 1, 3 * n)
        # flag: 0 = dgbsv, 1 = lstsq fallback, 2 = zeros fallback
        try:
            from scipy.linalg import solve_banded as _sb

            x_np = _sb((kl, ku), ab_np, b_in)
            flag_np = _np.int32(0)
        except Exception:
            try:
                M = _banded_to_dense(ab_np, kl, ku, n)
                x_np, *_ = _np.linalg.lstsq(M, b_in, rcond=None)
                flag_np = _np.int32(1)
            except Exception:
                x_np = _np.zeros_like(b_in)
                flag_np = _np.int32(2)
        out = _np.concatenate(
            [x_np.astype(ab_flat.dtype).reshape(-1), flag_np.reshape(-1).astype(ab_flat.dtype)]
        )
        return out

    def _do_solve(_):
        return jax.pure_callback(
            _solve,
            jax.ShapeDtypeStruct((b_flat.shape[0] + 1,), b_flat.dtype),
            banded.reshape(-1),
            b_flat,
            vmap_method="sequential",
        )

    def _zeros(_):
        return jnp.concatenate([jnp.zeros_like(b_flat), jnp.full((1,), 2.0, dtype=b_flat.dtype)])

    has_nan = ~jnp.all(jnp.isfinite(banded))
    out = jax.lax.cond(has_nan, _zeros, _do_solve, None)
    x = out[:-1].reshape(n, 3)
    flag = out[-1]
    info = {"used_lstsq": flag == 1, "used_zeros": (flag == 2) | has_nan}
    if os.environ.get("DRIFTJAX_NATIVE_BANDED", "0") == "1":
        from driftjax.numerics.banded_native import native_banded_solve as _nbs

        x_nat, info_nat = _nbs(A, B, C, b)
        use_native = jnp.asarray(info_nat["ok"]) & (~has_nan)
        x = jax.tree.map(lambda xn, xc: jnp.where(use_native, xn, xc), x_nat, x)
        info = {
            "used_lstsq": jnp.where(use_native, False, info["used_lstsq"]),
            "used_zeros": jnp.where(use_native, False, info["used_zeros"]),
            "used_native": use_native,
        }
    return x, info


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
    returns zeros — N2 contract: the zeros themselves must NEVER be trusted
    (a zero step satisfies any step-norm gate with ||F|| huge); the caller
    detects the failure via the residual check, and the certified detector
    is the post-hoc absolute audit in simulate() (_audit_sweep_solution),
    which marks the sweep unconverged and warns.  Prefer
    banded_solve_with_info when the caller needs the used_zeros flag.
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


def _block_row_scales(A, B, C):
    """Per-block-row scales dr[i] = 1/max|block-row i|, from blocks alone.

    Block-row i spans A[i], B[i] (i < n-1) and C[i-1] (i > 0) — the exact
    nonzero pattern of interleaved row triple (3i, 3i+1, 3i+2). Coarser
    than per-scalar-row equilibration (one scale per 3 rows) but requires
    no dense Jacobian; still tames ~1e11 SRV/transport imbalances.
    """
    import jax.numpy as _jnp

    n = A.shape[0]
    amax = _jnp.max(_jnp.abs(A), axis=(1, 2))
    s = amax
    if n > 1:
        bmax = _jnp.max(_jnp.abs(B), axis=(1, 2))
        cmax = _jnp.max(_jnp.abs(C), axis=(1, 2))
        s = s.at[:-1].set(_jnp.maximum(s[:-1], bmax))
        s = s.at[1:].set(_jnp.maximum(s[1:], cmax))
    s = _jnp.where(_jnp.isfinite(s) & (s > 0), s, _jnp.asarray(1.0, dtype=s.dtype))
    return 1.0 / s


def adjoint_banded_solve_blocks(A, B, C, g, tol=1e-8):
    """Equilibrated banded transpose solve from FORWARD blocks (no dense J).

    Phase-B API: the caller passes the analytic (A, B, C) blocks it already
    holds, so there is no dense-Jacobian construction and no dense-to-block
    extraction. Row scales come from the blocks (_block_row_scales); the
    residual is checked against the ORIGINAL system via the O(N) blockwise
    transpose matvec, never a dense matrix.

    With D = kron(diag(dr), I_3), Je = D @ J, Je^T = J^T @ D: solving
    Je^T y = g gives lam = D @ y exactly (solution-preserving). As with
    adjoint_banded_solve, the Dr here is the FORWARD row scaling (hence a
    column scaling of J^T); true row-equilibration of J^T (D from column
    maxima of J) is the open B4 experiment. Returns (lam, used_fallback);
    used_fallback True routes the caller to dense LU.
    """
    import jax

    from driftjax.numerics.analytic_jacobian import blockwise_matvec_transpose

    n = A.shape[0]
    dr = _block_row_scales(A, B, C)  # (n,)
    Ae = A * dr[:, None, None]
    Be = B * dr[:-1, None, None] if n > 1 else B
    Ce = C * dr[1:, None, None] if n > 1 else C
    At = jnp.transpose(Ae, (0, 2, 1))
    Ct = jnp.transpose(Ce, (0, 2, 1))
    Bt = jnp.transpose(Be, (0, 2, 1))
    banded_t, kl, ku = _blocks_to_lapack_banded(At, Ct, Bt)
    g_flat = g.reshape(-1)

    def _solve(ab_flat, b_in):
        import numpy as _np
        from scipy.linalg import solve_banded as _sb

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
    dr_tiled = jnp.repeat(dr, 3)
    lam = (dr_tiled * y).reshape(g.shape)
    # ORIGINAL-system residual via O(N) blockwise transpose matvec (A,B,C
    # are the raw forward blocks): r = J^T lam - g. Never the scaled system.
    r = blockwise_matvec_transpose(A, B, C, lam.reshape(-1)) - g_flat
    resid = jnp.linalg.norm(r) / (jnp.linalg.norm(g_flat) + 1e-30)
    fallback = has_nan | (~jnp.isfinite(resid)) | (resid > tol)
    return jnp.where(fallback, jnp.zeros_like(lam), lam), fallback


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
    # M1 derivation note: with Je = Dr @ J and Dr = diag(1/rowmax(J)),
    # transposing gives Je^T = J^T @ Dr.  Solving Je^T y = g yields
    # J^T (Dr y) = g, so lam = Dr @ y exactly — the unscaling below is
    # solution-preserving by construction, not an approximation.  Note the
    # asymmetry: Dr equilibrates the ROWS OF J, which appear as the COLUMNS
    # OF J^T, so the factored matrix is column-scaled (right-equilibrated).
    # True row-equilibration of J^T would instead use D = diag(1/colmax(J));
    # whether that conditions the transpose solve better is the open Tier-B
    # (B4) experiment — it is NOT claimed here.  What IS guaranteed: the
    # residual gate below checks the ORIGINAL system J^T lam = g, never the
    # scaled one, so a small scaled residual cannot certify a wrong lam —
    # any scaling-induced error shows up in resid and routes to dense LU.
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
    # NOTE (measured): the JAX-native transpose was tried here
    # (native_banded_transpose on the Je blocks) but reverted: under the
    # vmapped + checkpointed backward the scan loop costs ~440 ms flat
    # (vs ~10 ms jitted standalone) — an XLA CPU batching cliff independent
    # of scatter style. The native solver stays available for serial/jit
    # use (forward path) and as a benchmark reference; see the paper note.
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
