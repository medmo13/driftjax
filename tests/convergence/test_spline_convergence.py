"""PCHIP interpolation order on smooth data."""

import jax.numpy as jnp
import pytest

from driftjax.numerics.spline import pchip_coefs

pytestmark = pytest.mark.smoke


def _pchip_interp(x, y, xs):
    a, b, c, d = pchip_coefs(x, y)
    idx = jnp.clip(jnp.searchsorted(x, xs) - 1, 0, len(x) - 2)
    dx = xs - x[idx]
    return a[idx] * dx**3 + b[idx] * dx**2 + c[idx] * dx + d[idx]


def test_pchip_convergence_order():

    def f(z):
        return jnp.exp(-3.0 * z)

    errs = []
    for n in (11, 21, 41, 81, 161):
        x = jnp.linspace(0.0, 1.0, n)
        xs = jnp.linspace(0.0, 1.0, 5001)
        errs.append(float(jnp.max(jnp.abs(_pchip_interp(x, f(x), xs) - f(xs)))))
    p = jnp.array([jnp.log(errs[i] / errs[i + 1]) / jnp.log(2.0) for i in range(4)])
    assert float(jnp.min(p)) > 2.5, p
