"""JAX-native banded solver (no host callback).

Primary: pivoted block-tridiagonal GE via lax.scan (O(N*bw^2), pure XLA).
Singular systems detected via per-block determinant threshold.
Both differentiable end-to-end via jax.grad/jax.jvp.
"""

from __future__ import annotations

import jax.numpy as jnp

from driftjax.numerics.banded_ge import (
    _dense_from_blocks,  # noqa: F401 (re-exported for test compatibility)
    banded_ge_solve,
)


def native_banded_solve(A, B, C, b, *, tol=1e-12, equilibrate=True):
    """Native (no-callback) solve of J x = b via pivoted banded GE.

    Uses the block Thomas algorithm in lax.scan (O(N*bw^2)).
    Singular systems are detected via per-block determinant threshold.
    Stays inside XLA (differentiable, no host round-trip).

    When ``equilibrate=True`` (default, v0.1.18b), per-scalar-row diagonal
    scaling tames SRV/contact row-scaling imbalances without leaving XLA.

    Returns (x, {ok, singular, method, rank}).
    """
    return banded_ge_solve(A, B, C, b, tol=tol, equilibrate=equilibrate)


def native_banded_transpose(A, B, C, g):
    """J^T lam = g via banded GE on the transposed blocks.

    J^T is block-tridiagonal with A^T on diagonal, C^T super-diagonal,
    B^T sub-diagonal. The block Thomas algorithm applies unchanged.
    Row equilibration from the forward blocks is applied to tame
    conditioning in the transpose direction.
    """
    At = jnp.transpose(A, (0, 2, 1))
    Ct = jnp.transpose(C, (0, 2, 1))
    Bt = jnp.transpose(B, (0, 2, 1))
    return banded_ge_solve(At, Ct, Bt, g.reshape(A.shape[0], 3), tol=1e-12)
