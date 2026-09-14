"""Legacy Block-Thomas solver == dense solve; scalar Thomas check."""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.smoke  # no solves: pure linear algebra, milliseconds

from driftjax.numerics.banded_solve import _legacy_solve_block_tridiagonal as solve_block_tridiagonal


def _random_block_tridiag(n, key):
    key1, key2 = jax.random.split(key)
    J = jax.random.normal(key1, (3 * n, 3 * n))
    Jr = J.reshape(n, 3, n, 3)
    idx = jnp.arange(n)
    A = Jr[idx, :, idx, :] + 20.0 * jnp.eye(3)
    Jb = jnp.zeros((3 * n, 3 * n))
    for i in range(n):
        Jb = Jb.at[3 * i : 3 * i + 3, 3 * i : 3 * i + 3].set(A[i])
    for i in range(n - 1):
        Jb = Jb.at[3 * i : 3 * i + 3, 3 * (i + 1) : 3 * (i + 1) + 3].set(Jr[i, :, i + 1, :])
        Jb = Jb.at[3 * (i + 1) : 3 * (i + 1) + 3, 3 * i : 3 * i + 3].set(Jr[i + 1, :, i, :])
    return Jb


def test_block_thomas_matches_dense():
    for n in (3, 10, 40):
        Jb = _random_block_tridiag(n, jax.random.PRNGKey(n))
        b = jax.random.normal(jax.random.PRNGKey(n + 1), (3 * n,))
        x_dense = jnp.linalg.solve(Jb, b)
        x_bt = solve_block_tridiagonal(Jb, b)
        assert float(jnp.max(jnp.abs(x_bt - x_dense))) < 1e-09, n


def test_block_thomas_scalar_tridiag():
    """3x3 blocks with D = diag(d_i, 1, 2): component 0 is scalar Thomas."""
    n = 20
    d = 4.0 * jnp.ones(n)
    a = -1.0 * jnp.ones(n - 1)
    bvec = jnp.arange(1.0, n + 1.0)
    Jb = jnp.zeros((3 * n, 3 * n))
    for i in range(n):
        Jb = Jb.at[3 * i : 3 * i + 3, 3 * i : 3 * i + 3].set(jnp.diag(jnp.array([d[i], 1.0, 2.0])))
    for i in range(n - 1):
        Jb = Jb.at[3 * i, 3 * (i + 1)].set(a[i])
        Jb = Jb.at[3 * (i + 1), 3 * i].set(a[i])
    rhs = jnp.zeros(3 * n)
    rhs = rhs.at[0::3].set(bvec)
    rhs = rhs.at[1::3].set(7.0)
    rhs = rhs.at[2::3].set(11.0)
    x = solve_block_tridiagonal(Jb, rhs)
    x_dense = jnp.linalg.solve(Jb, rhs)
    assert float(jnp.max(jnp.abs(x - x_dense))) < 1e-10
    dd = d.copy()
    bb = bvec.copy()
    for i in range(1, n):
        w = a[i - 1] / dd[i - 1]
        dd = dd.at[i].set(dd[i] - w * a[i - 1])
        bb = bb.at[i].set(bb[i] - w * bb[i - 1])
    y = jnp.zeros(n)
    y = y.at[n - 1].set(bb[n - 1] / dd[n - 1])
    for i in range(n - 2, -1, -1):
        y = y.at[i].set((bb[i] - a[i] * y[i + 1]) / dd[i])
    assert float(jnp.max(jnp.abs(x[0::3] - y))) < 1e-10
