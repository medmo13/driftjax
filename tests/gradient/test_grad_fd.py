"""AD-vs-FD gradient checks on kernels and solvers."""

import jax
import jax.numpy as jnp
import pytest

from driftjax.numerics.block_thomas import solve_block_tridiagonal
from driftjax.numerics.scharfetter_gummel import bernoulli

pytestmark = pytest.mark.smoke


def test_grad_bernoulli_fd():
    for x in (0.3, -2.0, 5.0):
        g = float(jax.grad(lambda s: jnp.sum(bernoulli(s)))(jnp.float64(x)))
        h = 1e-06
        fd = (float(bernoulli(x + h)) - float(bernoulli(x - h))) / (2 * h)
        assert abs(g - fd) < 1e-06


def test_grad_block_thomas_linear_solve():
    """d/db of x = solve(A(b), b) via AD == FD (linear in b)."""
    n = 12
    A = jnp.tril(
        jnp.triu(jax.random.normal(jax.random.key(0), (3 * n, 3 * n)), -3), 3
    ) + 40.0 * jnp.eye(3 * n)

    def f(bb):
        return jnp.sum(solve_block_tridiagonal(A, bb) ** 2)

    b0 = jax.random.normal(jax.random.PRNGKey(1), (3 * n,))
    g_ad = jax.grad(f)(b0)
    h = 1e-07
    g_fd = (f(b0 + h * jnp.eye(3 * n)[0]) - f(b0 - h * jnp.eye(3 * n)[0])) / (2 * h)
    assert abs(float(g_ad[0] - g_fd)) < 0.0001
