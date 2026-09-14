"""AD-vs-FD gradient checks on kernels and solvers."""

import jax
import jax.numpy as jnp
import pytest

from driftjax.numerics.banded_solve import banded_solve, extract_blocks
from driftjax.numerics.scharfetter_gummel import bernoulli

pytestmark = pytest.mark.smoke


def test_grad_bernoulli_fd():
    for x in (0.3, -2.0, 5.0):
        g = float(jax.grad(lambda s: jnp.sum(bernoulli(s)))(jnp.float64(x)))
        h = 1e-06
        fd = (float(bernoulli(x + h)) - float(bernoulli(x - h))) / (2 * h)
        assert abs(g - fd) < 1e-06


def test_grad_banded_linear_solve_raises_loudly():
    """The forward banded solve runs through a host callback (opaque to AD).

    Differentiating through it must raise loudly, never silently return
    wrong tangents; device gradients flow only via the custom_vjp
    implicit-adjoint path.
    """
    n = 12
    J = jnp.tril(
        jnp.triu(jax.random.normal(jax.random.key(0), (3 * n, 3 * n)), -3), 3
    ) + 40.0 * jnp.eye(3 * n)
    A, B, C = extract_blocks(J)

    def f(bb):
        return jnp.sum(banded_solve(A, B, C, bb.reshape(n, 3)) ** 2)

    b0 = jax.random.normal(jax.random.PRNGKey(1), (3 * n,))
    with pytest.raises(ValueError, match="[Pp]ure callback"):
        jax.grad(f)(b0)
