"""PCHIP / qspline kernels + MPP extraction (vs scipy ground truth)."""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.smoke  # no solves: pure spline math, milliseconds

from driftjax.numerics.spline import _qspline_coefs, calcPmax_cubic, calcPmax_quadratic, pchip_coefs


def test_pchip_matches_scipy():
    from scipy.interpolate import PchipInterpolator

    x = np.linspace(0.0, 1.0, 9)
    y = np.exp(-3.0 * x)
    a, b, c, d = pchip_coefs(jnp.asarray(x), jnp.asarray(y))
    xs = np.linspace(0.0, 1.0, 201)
    idx = np.clip(np.searchsorted(x, xs) - 1, 0, len(x) - 2)
    dx = xs - x[idx]
    ys = (
        np.asarray(a)[idx] * dx**3
        + np.asarray(b)[idx] * dx**2
        + np.asarray(c)[idx] * dx
        + np.asarray(d)[idx]
    )
    sp = PchipInterpolator(x, y)(xs)
    assert np.max(np.abs(ys - sp)) < 1e-12


def test_pchip_no_overshoot_data_range():
    x = jnp.linspace(0.0, 1.0, 9)
    y = jnp.exp(-3.0 * x)
    a, b, c, d = pchip_coefs(x, y)
    xs = jnp.linspace(0.0, 1.0, 401)
    idx = jnp.clip(jnp.searchsorted(x, xs) - 1, 0, len(x) - 2)
    dx = xs - x[idx]
    ys = a[idx] * dx**3 + b[idx] * dx**2 + c[idx] * dx + d[idx]
    assert float(jnp.min(ys)) >= float(jnp.min(y)) - 1e-09
    assert float(jnp.max(ys)) <= float(jnp.max(y)) + 1e-09


def test_pchip_interpolates_nodes():
    x = jnp.linspace(0.0, 2.0, 7)
    y = jnp.sin(x)
    a, b, c, d = pchip_coefs(x, y)
    assert float(jnp.max(jnp.abs(d - y[:-1]))) < 1e-12


def test_calcPmax_parabolic():
    v = jnp.linspace(0.0, 1.0, 21)
    jv = 0.06 - 0.05 * v**2
    pmax, vmax = calcPmax_cubic(v, jv)
    assert abs(float(pmax) - 0.0253) < 0.001, float(pmax)
    assert abs(float(vmax) - 0.6325) < 0.03, float(vmax)


def test_calcPmax_matches_quadratic_on_smooth():
    v = jnp.linspace(0.05, 0.95, 20)
    jv = 0.06 - 0.04 * v**2
    p1, driftjax = calcPmax_cubic(v, jv)
    p2, driftjax = calcPmax_quadratic(v, jv)
    assert abs(float(p1) - float(p2)) / float(p1) < 0.02
    assert abs(float(driftjax) - float(driftjax)) < 0.05


def test_qspline_roundtrip():
    x = jnp.linspace(0.0, 1.0, 6)
    y = jnp.cos(2.0 * x)
    a, b, c = _qspline_coefs(x, y)
    for i in range(len(x) - 1):
        v = a[i] * x[i] ** 2 + b[i] * x[i] + c[i]
        assert abs(float(v) - float(y[i])) < 1e-08
    last = a[-1] * x[-1] ** 2 + b[-1] * x[-1] + c[-1]
    assert abs(float(last) - float(y[-1])) < 1e-08
