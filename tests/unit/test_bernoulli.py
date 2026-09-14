"""Bernoulli kernel: values + AD-vs-FD (NaN-free derivatives)."""

import jax
import jax.numpy as jnp
import pytest

from driftjax.numerics.scharfetter_gummel import bernoulli

pytestmark = pytest.mark.smoke  # no solves: pure kernel math, milliseconds


def test_bernoulli_values():
    assert float(bernoulli(jnp.float64(0.0))) == 1.0
    assert float(bernoulli(jnp.float64(1e-12))) == 1.0 - 5e-13
    b20 = float(bernoulli(jnp.float64(20.0)))
    assert abs(b20 - 20.0 / (jnp.exp(20.0) - 1.0)) / b20 < 1e-12
    bm20 = float(bernoulli(jnp.float64(-20.0)))
    assert abs(bm20 - 20.0) / 20.0 < 1e-07


def test_bernoulli_jvp_nan_free():
    xs = jnp.array([-30.0, -3.0, -0.1, 0.0, 1e-12, 0.3, 12.0, 30.0])
    J = jax.jacrev(lambda x: jnp.sum(bernoulli(x)))(xs)
    assert not bool(jnp.isnan(J).any())
    for x in xs:
        g = float(jax.grad(lambda s: jnp.sum(bernoulli(s)))(x))
        h = 1e-06 if abs(float(x)) > 1.0 else 0.0001
        fd = (float(bernoulli(x + h)) - float(bernoulli(x - h))) / (2 * h)
        assert abs(g - fd) < 0.0001 * max(1.0, abs(fd))
