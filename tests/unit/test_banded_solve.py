"""Pivoted banded solver (LAPACK dgbsv) == dense solve on synthetic systems."""

import jax.numpy as jnp
import numpy as np

from driftjax.numerics.banded_solve import _blocks_to_lapack_banded, banded_solve


def _synthetic_blocks(key_seed=0, n=8):
    rng = np.random.default_rng(key_seed)
    A = jnp.asarray(rng.normal(size=(n, 3, 3)))
    B = jnp.asarray(rng.normal(size=(n - 1, 3, 3)) * 0.1)
    C = jnp.asarray(rng.normal(size=(n - 1, 3, 3)) * 0.1)
    # Diagonally dominant => nonsingular.
    A = A + 6.0 * jnp.eye(3)[None, :, :]
    b = jnp.asarray(rng.normal(size=(n, 3)))
    return A, B, C, b


def _dense_from(A, B, C):
    from driftjax.numerics.analytic_jacobian import dense_from_blocks

    return dense_from_blocks(A, B, C)


def test_banded_matches_dense():
    A, B, C, b = _synthetic_blocks()
    x = banded_solve(A, B, C, b).reshape(-1)
    J = _dense_from(A, B, C)
    xd = jnp.linalg.solve(J, b.reshape(-1))
    assert float(jnp.max(jnp.abs(x - xd))) < 1e-10


def test_banded_residual_small():
    A, B, C, b = _synthetic_blocks(key_seed=1, n=30)
    x = banded_solve(A, B, C, b).reshape(-1)
    J = _dense_from(A, B, C)
    rel = float(jnp.linalg.norm(J @ x - b.reshape(-1)) / (jnp.linalg.norm(b) + 1e-30))
    assert rel < 1e-12


def test_jvp_through_banded_solve_raises_loudly():
    """Forward Newton steps run through a host callback and are opaque to AD.

    Gradients must flow via the custom_vjp implicit-adjoint path, never by
    differentiating through banded_solve.  This test pins the loud failure
    mode: jvp must raise, never silently return wrong tangents.
    """
    import jax

    A, B, C, b = _synthetic_blocks(n=4)
    with __import__("pytest").raises(ValueError, match="[Pp]ure callback"):
        jax.jvp(lambda A_: banded_solve(A_, B, C, b), (A,), (jnp.ones_like(A),))


def test_lapack_banded_layout_matches_dense():
    """_blocks_to_lapack_banded must reproduce the dense matrix entries."""
    A, B, C, _ = _synthetic_blocks(key_seed=2, n=5)
    ab, kl, ku = _blocks_to_lapack_banded(A, B, C)
    J = np.asarray(_dense_from(A, B, C))
    ab_np = np.asarray(ab)
    N = J.shape[0]
    for j in range(N):
        for i in range(max(0, j - ku), min(N, j + kl + 1)):
            assert ab_np[ku + i - j, j] == J[i, j]
