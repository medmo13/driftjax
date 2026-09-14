"""Design objectives: efficiency, polar IV-curve distance, MSE.

NOTE (pinned convention): ``iv_curve_distance`` swaps the arctan2
argument order (θ = atan2(x, y)) so θ increases with x; the quadratic
spline interpolation inside it relies on that.  Do not "fix" it — the
metric is defined by that convention.
"""

from __future__ import annotations

import jax.numpy as jnp

from driftjax.numerics.spline import _qspline_coefs


def _polar(x, y):
    theta = jnp.arctan2(x, y)  # swapped by design (package quirk)
    r = jnp.sqrt(x**2 + y**2)
    return theta, r


def iv_curve_distance(j, j0, norm: float = 2.0) -> jnp.ndarray:
    """Polar-radius distance between two IV curves (same voltage grid)."""
    y1 = 10.0 * jnp.asarray(j)
    y2 = 10.0 * jnp.asarray(j0)
    x1 = jnp.arange(y1.size) * 0.05
    x2 = jnp.arange(y2.size) * 0.05
    theta1, r1 = _polar(x1, y1)
    theta2, r2 = _polar(x2, y2)
    thetaint = jnp.linspace(0.0, jnp.pi / 2, 100)
    rint1 = _interp(thetaint, theta1, r1)
    rint2 = _interp(thetaint, theta2, r2)
    return jnp.sum(jnp.abs(rint1 - rint2) ** norm)


def _interp(xq, xp, yp):
    """Clamped piecewise-quadratic interpolation (reference semantics for iv_curve_distance)."""
    if xq.ndim == 0:
        xq = xq[None]
        sq = True
    else:
        sq = False
    a, b, c = _qspline_coefs(xp, yp)
    n_segments = xp.size - 1
    idx = jnp.clip(jnp.searchsorted(xp, xq) - 1, 0, n_segments - 1)
    y = a[idx] * xq**2 + b[idx] * xq + c[idx]
    return y[0] if sq else y


def iv_mse(j, j0) -> jnp.ndarray:
    """Normalised mean-squared-error on the current curve (smooth metric)."""
    j = jnp.asarray(j)
    j0 = jnp.asarray(j0)
    return jnp.mean((j - j0) ** 2) / (jnp.mean(j0**2) + 1e-30)


def efficiency(design, ls=None) -> jnp.ndarray:
    """PCE in percent under AM1.5G (or `ls`)."""
    from driftjax.problems import Sweep
    from driftjax.simulate import simulate

    return 100.0 * simulate(design, Sweep(), ls=ls).efficiency
