"""Mixed-precision iterative refinement (NEW): FP32 core + FP64 residual."""

import jax
import jax.numpy as jnp
import pytest

from driftjax.numerics.linalg import row_equilibrate
from driftjax.numerics.mixed_precision import solve_refined, solve_refined_batched

pytestmark = pytest.mark.smoke


def _banded_spd(n_blocks, key, cond=10000.0):
    """SPD block-tridiagonal (3*N)² matrix (DDP-like sparsity)."""
    N = 3 * n_blocks
    A = jax.random.normal(key, (N, N))
    A = jnp.tril(jnp.triu(A, -3), 3) + cond * jnp.eye(N)
    return A @ A.T


def test_solve_refined_matches_dense():
    for n in (12, 60):
        A, b = (
            _banded_spd(n, jax.random.PRNGKey(n)),
            jax.random.normal(jax.random.PRNGKey(n + 9), (3 * n,)),
        )
        x_ref, rel, n_ref, conv = solve_refined(A, b, tol=1e-11)
        x_dense = jnp.linalg.solve(A, b)
        assert float(jnp.max(jnp.abs(x_ref - x_dense))) < 1e-07, n_ref
        assert float(rel) < 1e-06


def test_solve_refined_batched():
    A = _banded_spd(30, jax.random.PRNGKey(7))
    b = jax.random.normal(jax.random.PRNGKey(8), (90,))
    B = jnp.stack([b, 2 * b, -b], axis=0)
    Ab = jnp.broadcast_to(A, (3, 90, 90))
    X, rel, n_ref, conv = solve_refined_batched(Ab, B, tol=1e-10)
    assert X.shape == (3, 90)
    assert float(jnp.max(jnp.abs(X[0] - jnp.linalg.solve(A, b)))) < 1e-07
    assert float(jnp.max(jnp.abs(X[1] - 2 * jnp.linalg.solve(A, b)))) < 1e-07


def test_row_equilibrate_improves_cond():
    A = _banded_spd(40, jax.random.PRNGKey(3), cond=10000.0)
    b = jax.random.normal(jax.random.PRNGKey(4), (120,))
    scale = jnp.exp(jnp.linspace(0, 10, 120))
    A = scale[:, None] * A * scale[None, :]
    Ae, _, _ = row_equilibrate(A, b)
    assert float(jnp.linalg.cond(Ae)) < float(jnp.linalg.cond(A))
