"""JAX-native Givens band QR solver: construction exactness + solve accuracy."""

import jax
import jax.numpy as jnp
import numpy as np

from driftjax.numerics.analytic_jacobian import dense_from_blocks
from driftjax.numerics.banded_native import (
    KL,
    KU,
    WD,
    blocks_to_rowband,
    native_banded_solve,
    native_banded_transpose,
    qr_solve_rowband,
)


def _synthetic_blocks(key_seed=0, n=8):
    rng = np.random.default_rng(key_seed)
    A = jnp.asarray(rng.normal(size=(n, 3, 3)))
    B = jnp.asarray(rng.normal(size=(n - 1, 3, 3)) * 0.1)
    C = jnp.asarray(rng.normal(size=(n - 1, 3, 3)) * 0.1)
    A = A + 6.0 * jnp.eye(3)[None, :, :]
    b = jnp.asarray(rng.normal(size=(n, 3)))
    return A, B, C, b


def test_rowband_matches_dense():
    A, B, C, _ = _synthetic_blocks(n=7)
    W = blocks_to_rowband(A, B, C)
    J = np.asarray(dense_from_blocks(A, B, C))
    N = J.shape[0]
    Wn = np.asarray(W)
    for i in range(N):
        for dj in range(-KL, KU + KL + 1):
            j = i + dj
            if 0 <= j < N:
                assert Wn[i, dj + KL] == J[i, j], (i, j)


def test_native_matches_dense():
    A, B, C, b = _synthetic_blocks(n=12)
    x, info = native_banded_solve(A, B, C, b)
    assert bool(info["ok"]) is True
    J = dense_from_blocks(A, B, C)
    xd = jnp.linalg.solve(J, b.reshape(-1))
    rel = float(jnp.linalg.norm(x.reshape(-1) - xd) / (jnp.linalg.norm(xd) + 1e-30))
    assert rel < 1e-10, rel


def test_native_transpose_matches_dense():
    A, B, C, b = _synthetic_blocks(key_seed=3, n=12)
    x, info = native_banded_transpose(A, B, C, b.reshape(-1))
    assert bool(info["ok"]) is True
    J = dense_from_blocks(A, B, C)
    xd = jnp.linalg.solve(J.T, b.reshape(-1))
    rel = float(jnp.linalg.norm(x.reshape(-1) - xd) / (jnp.linalg.norm(xd) + 1e-30))
    assert rel < 1e-10, rel


def test_native_flags_singular():
    A, B, C, b = _synthetic_blocks(n=6)
    # An entire zero block ROW (A[2], B[2], C[1] all zero) makes the matrix
    # exactly singular (verified rank-deficient below): must report FAILED
    # (caller routes to the dgbsv-callback path), never a plausible vector.
    A = A.at[2].set(jnp.zeros((3, 3)))
    B = B.at[2].set(jnp.zeros((3, 3)))
    C = C.at[1].set(jnp.zeros((3, 3)))
    J = np.asarray(dense_from_blocks(A, B, C))
    assert np.linalg.matrix_rank(J) < J.shape[0]
    _, info = native_banded_solve(A, B, C, b)
    assert bool(info["ok"]) is False


def test_native_vmap_batch():
    A, B, C, b = _synthetic_blocks(n=6)
    Ab = jnp.stack([A, A + 0.5 * jnp.eye(3)[None, :, :]])
    Bb = jnp.stack([B, B])
    Cb = jnp.stack([C, C])
    bb = jnp.stack([b, b])

    def one(a, bb_, cc, rhs):
        x, info = native_banded_solve(a, bb_, cc, rhs)
        return x, info["ok"]

    x, ok = jax.vmap(one)(Ab, Bb, Cb, bb)
    assert bool(jnp.all(ok)) is True
    J = dense_from_blocks(A, B, C)
    xd = jnp.linalg.solve(J, b.reshape(-1))
    assert float(jnp.max(jnp.abs(x[0].reshape(-1) - xd))) < 1e-9
