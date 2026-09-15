"""JAX-native banded solver: Givens QR in lax.scan primitives (no callbacks).

Why QR and not pivoted LU in scan form: unpivoted block-Thomas failed on
row-scaled heterojunctions, and hand-rolled partial-pivot bookkeeping in
scan form is where subtle index bugs live. Givens QR is unconditionally
backward-stable with NO pivot decisions at all — orthogonal
transformations cannot amplify rounding the way unpivoted elimination
does — at the same O(N*bw^2) order as band LU (~2x the flops, same
asymptotics). Stability comes from the algorithm class, not from
replicating LAPACK pivot logic.

Representation: row-band windows. Row i of an (kl, ku) band matrix spans
cols [i-kl, i+ku+fill]; with QR fill-in the R bandwidth is ku+kl, so each
row window is cols [i-kl, i+ku+kl], width Wd = 2*kl+ku+1, i.e.
W[i,k] = M[i, i-kl+k] (zeros outside the matrix). Rotations on matrix
rows (r, r+1) combine static slices of W[r], W[r+1] — no index
arithmetic under trace beyond the scan counter.

Contract: native_banded_solve(A, B, C, b) returns (x, info) with
info["ok"] = all-finite x. It does NOT do least-squares: on singular
systems R has a zero diagonal, back-substitution yields nonfinite x, and
ok=False routes the caller to the existing dgbsv-callback path (which
owns the lstsq fallback). The downstream linresid gate + post-hoc audit
remain the certifiers, exactly as with the callback path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

KL = 5
KU = 5
WD = 2 * KL + KU + 1  # 16


def blocks_to_rowband(A, B, C, kl: int = KL, ku: int = KU):
    """(A, B, C) 3x3 blocks -> row-band windows W (3n, Wd).

    W[r, k] = M[r, r-kl+k]: one row per UNKNOWN (3n rows), window width
    Wd = 2*kl+ku+1 covering cols [r-kl, r+ku+kl] (QR fill-in included).
    Vectorized (3 scatters); exact by construction — pinned against the
    dense matrix in tests/unit/test_banded_native.py.
    """
    wd = 2 * kl + ku + 1
    n = A.shape[0]
    N = 3 * n
    idx = jnp.arange(n)
    W = jnp.zeros((N, wd), dtype=A.dtype)
    # (di, dj) enumerate intra-block offsets; every stamp below builds
    # flat (9n,) index/value vectors in pair-major order via tile/repeat.
    _DI = jnp.arange(3)[:, None] * jnp.ones((1, 3), dtype=jnp.int32)
    _DJ = jnp.ones((3, 1), dtype=jnp.int32) * jnp.arange(3)[None, :]
    _Pdi = jnp.repeat(_DI.reshape(-1), n)  # (9n,) di per entry
    _Pdj = jnp.repeat(_DJ.reshape(-1), n)  # (9n,) dj per entry
    _Pidx = jnp.tile(idx, 9)  # (9n,) block index per entry
    # Diagonal blocks: M[3i+di, 3i+dj] at window kl+dj-di.
    rr = 3 * _Pidx + _Pdi
    cc = kl + _Pdj - _Pdi
    vals_d = jnp.transpose(A, (1, 2, 0)).reshape(-1)
    W = W.at[(rr, cc)].set(vals_d)
    _Z = jnp.zeros((1, 3, 3), dtype=A.dtype)
    # Super-diagonal B[i] at M[3i+di, 3i+3+dj], window kl+3+dj-di.
    # B has n-1 blocks; the zero pad lands on (valid-row, valid-window)
    # positions whose matrix entries are outside M — zeros, hence exact.
    Bp = jnp.concatenate([B, _Z], axis=0)
    rr_b = 3 * _Pidx + _Pdi
    cc_b = kl + 3 + _Pdj - _Pdi
    W = W.at[(rr_b, cc_b)].set(jnp.transpose(Bp, (1, 2, 0)).reshape(-1))
    # Sub-diagonal C[i] at M[3i+3+di, 3i+dj], window kl-3+dj-di.
    # OOB rows (i = n-1) are dropped by .at; values are pad zeros anyway.
    Cp = jnp.concatenate([C, _Z], axis=0)
    rr_c = 3 * _Pidx + 3 + _Pdi
    cc_c = kl - 3 + _Pdj - _Pdi
    W = W.at[(rr_c, cc_c)].set(jnp.transpose(Cp, (1, 2, 0)).reshape(-1))
    return W


def _givens(a, b):
    """Stable Givens (c, s) with [[c, s], [-s, c]] @ [a, b] = [hypot, 0].

    Scaled hypot avoids overflow for large Jacobian entries; the (0, 0)
    input maps to the identity rotation. Branchless (where-selected) so it
    is safe under jit/vmap/scan.
    """
    m = jnp.maximum(jnp.abs(a), jnp.abs(b))
    safe = m > 0.0
    m = jnp.where(safe, m, 1.0)
    h = jnp.sqrt((a / m) ** 2 + (b / m) ** 2)
    c = jnp.where(safe, (a / m) / h, 1.0)
    s = jnp.where(safe, (b / m) / h, 0.0)
    return c, s


def qr_solve_rowband(W, b, kl: int = KL, ku: int = KU):
    """Solve Mx=b for row-band W via Givens QR + back-substitution.

    Pure lax.fori_loop/scan, no callbacks, vmappable. All WRITES use
    one-hot masked full-array updates (never dynamic scatter): dynamic
    scatters inside a loop compile to full-buffer copies on CPU/XLA and
    explode under vmap (measured 50x for 2x work), while masked dense
    updates batch cleanly. READS may use dynamic gathers (cheap).
    Returns (x, ok) with ok False on singular/nonfinite systems.
    """
    wd = 2 * kl + ku + 1
    n = W.shape[0]
    b = b.reshape(n)
    ar = jnp.arange(n)

    def body(j, carry):
        Wc, bc, min_diag = carry
        mJ = (ar == j)[:, None]
        for t in range(1, kl + 1):
            r = j + t
            r_in = r < n
            rc = jnp.minimum(r, n - 1)
            a = Wc[j, kl]
            bb = Wc[rc, kl - t]
            c, s = _givens(a, jnp.where(r_in, bb, 0.0))
            rowJ = Wc[j]
            rowR = Wc[rc]
            # Overlap columns [r-kl, j+ku+kl]: rowJ indices [t, wd),
            # rowR indices [0, wd-t). Static concatenation, no dynamic write.
            newJseg = c * rowJ[t:wd] + s * rowR[0 : wd - t]
            newRseg = -s * rowJ[t:wd] + c * rowR[0 : wd - t]
            newJrow = jnp.concatenate([rowJ[:t], jnp.where(r_in, newJseg, rowJ[t:])])
            newRrow = jnp.concatenate([jnp.where(r_in, newRseg, rowR[: wd - t]), rowR[wd - t :]])
            mR = ((ar == rc) & r_in)[:, None]
            Wc = Wc + mJ * (newJrow - rowJ) + mR * (newRrow - rowR)
            bj = bc[j]
            br = bc[rc]
            newbj = jnp.where(r_in, c * bj + s * br, bj)
            newbr = jnp.where(r_in, -s * bj + c * br, br)
            bc = bc + (ar == j) * (newbj - bj) + ((ar == rc) & r_in) * (newbr - br)
        # R[j,j] is final after this column's eliminations (row j is never
        # touched again): track the minimum for singularity detection.
        min_diag = jnp.minimum(min_diag, jnp.abs(Wc[j, kl]))
        return Wc, bc, min_diag

    Wq, bq, min_diag = jax.lax.fori_loop(0, n, body, (W, b, jnp.asarray(jnp.inf, dtype=W.dtype)))

    def back(i, x):
        j = n - 1 - i
        row = Wq[j]
        # R[j,j] at window kl; upper entries pair with x[j+1 .. j+nu-1]
        # via dynamic gather (reads are cheap; only writes avoid dynamics).
        nu = wd - kl - 1
        kvals = jnp.arange(nu)
        xidx = jnp.minimum(j + 1 + kvals, n - 1)
        valid = (j + 1 + kvals) < n
        s = bq[j] - jnp.sum(jnp.where(valid, row[kl + 1 : wd] * x[xidx], 0.0))
        d = row[kl]
        xj = s / jnp.where(d == 0.0, 1.0, d)
        return x + (ar == j) * (xj - x[j])

    x = jax.lax.fori_loop(0, n, back, jnp.zeros((n,), dtype=W.dtype))
    # Singular (zero R diagonal) or nonfinite -> FAILED: the caller routes
    # to the dgbsv-callback path, which owns the lstsq fallback. Accuracy
    # certification stays downstream (linresid gate + post-hoc audit).
    ok = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(Wq)) & (min_diag > 0.0)
    return x, ok


def native_banded_solve(A, B, C, b):
    """Structured forward solve from blocks, JAX-native (no callbacks).

    Returns (x (n,3), info) with info["ok"] False on singular/nonfinite
    systems (caller routes to the dgbsv-callback path, which owns lstsq).
    info["used_lstsq"] is always False here by construction.
    """
    n = A.shape[0]
    W = blocks_to_rowband(A, B, C)
    x, ok = qr_solve_rowband(W, b.reshape(-1))
    return x.reshape(n, 3), {"ok": ok, "used_lstsq": jnp.array(False), "used_zeros": ~ok}


def native_banded_transpose(A, B, C, g):
    """J^T x = g from forward blocks via the native QR path.

    J^T is block-tridiagonal with diagonal A^T, super C^T, sub B^T.
    """
    At = jnp.transpose(A, (0, 2, 1))
    Ct = jnp.transpose(C, (0, 2, 1))
    Bt = jnp.transpose(B, (0, 2, 1))
    return native_banded_solve(At, Ct, Bt, g.reshape(A.shape[0], 3))
