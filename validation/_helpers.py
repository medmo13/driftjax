"""Shared utilities for validation scripts — solver comparisons and diagnostics."""

import jax.numpy as jnp
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from driftjax.numerics.block_thomas import block_thomas_solve


def bt_transpose(A, B, C, g):
    """Solve J^T x = g via transposed Block-Thomas (O(N))."""
    At = jnp.transpose(A, (0, 2, 1))
    Ct = jnp.transpose(C, (0, 2, 1))
    Bt = jnp.transpose(B, (0, 2, 1))
    return block_thomas_solve(At, Ct, Bt, g.reshape(A.shape[0], 3)).reshape(-1)


def solve_sparse_lu(JT, g):
    """Solve J^T x = g via SuperLU sparse factorisation."""
    try:
        JT_sp = scipy.sparse.csc_matrix(np.array(JT, dtype=np.float64))
        lu = scipy.sparse.linalg.splu(JT_sp)
        x = lu.solve(np.array(g, dtype=np.float64))
        r = np.linalg.norm(JT_sp @ x - np.array(g)) / (np.linalg.norm(g) + 1e-30)
        return jnp.array(x), float(r)
    except Exception:
        return None, float("inf")


def equilibrate(J):
    """Row + column scaling to unit max-norm (for iterative refinement)."""
    J_np = np.array(J, dtype=np.float64)
    row_scale = np.max(np.abs(J_np), axis=1)
    row_scale[row_scale == 0] = 1
    Dr = np.diag(1.0 / row_scale)
    col_scale = np.max(np.abs(Dr @ J_np), axis=0)
    col_scale[col_scale == 0] = 1
    Dc = np.diag(1.0 / col_scale)
    return Dr @ J_np @ Dc, Dr, Dc


def iterative_refinement(JT, g, x0, iters=2):
    """Dense iterative refinement (FP64) around an initial guess."""
    x = np.array(x0, dtype=np.float64)
    JT_np = np.array(JT, dtype=np.float64)
    g_np = np.array(g, dtype=np.float64)
    for _ in range(iters):
        r = g_np - JT_np @ x
        try:
            x = x + np.linalg.solve(JT_np, r)
        except np.linalg.LinAlgError:
            break
    r_final = np.linalg.norm(JT_np @ x - g_np) / (np.linalg.norm(g_np) + 1e-30)
    return jnp.array(x), float(r_final)
