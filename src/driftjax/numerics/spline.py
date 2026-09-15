"""Maximum-power-point extraction: quadratic and shape-preserving cubic (PCHIP).

PCHIP (Fritsch–Carlson 1980) cubic Hermite coefficients are C¹ and never
overshoot the data range — critical near the sharp exponential turn-on of
the P(V) curve, where the natural cubic spline oscillates and can fabricate
a spurious maximum.  Maxima are found analytically per segment (candidates:
endpoints + roots of the derivative quadratic); no gradient ascent (the historical
package's ascent diverged with lr=1 and flat extrapolation).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import vmap


@jax.jit
def pchip_coefs(x, y):
    """Per-segment (a, b, c, d): y = a·dx³ + b·dx² + c·dx + d, dx = x − x_i."""
    h = jnp.diff(x)
    dlt = jnp.diff(y) / h
    d0, d1 = dlt[:-1], dlt[1:]
    w1, w2 = 2.0 * h[1:] + h[:-1], h[1:] + 2.0 * h[:-1]
    d0s = jnp.where(d0 == 0.0, 1e-300, d0)
    d1s = jnp.where(d1 == 0.0, 1e-300, d1)
    mint = jnp.where(d0 * d1 <= 0.0, 0.0, (w1 + w2) / (w1 / d0s + w2 / d1s))
    m0 = jnp.clip(
        ((2.0 * h[0] + h[1]) * dlt[0] - h[0] * dlt[1]) / (h[0] + h[1]),
        -3.0 * jnp.abs(dlt[0]),
        3.0 * jnp.abs(dlt[0]),
    )
    m0 = jnp.where(m0 * dlt[0] < 0.0, 0.0, m0)
    mn = jnp.clip(
        ((2.0 * h[-1] + h[-2]) * dlt[-1] - h[-1] * dlt[-2]) / (h[-1] + h[-2]),
        -3.0 * jnp.abs(dlt[-1]),
        3.0 * jnp.abs(dlt[-1]),
    )
    mn = jnp.where(mn * dlt[-1] < 0.0, 0.0, mn)
    m = jnp.concatenate([m0[None], mint, mn[None]])
    y0 = y[:-1]
    dH = y0
    cH = m[:-1]
    bH = 3.0 * (y[1:] - y0) / h**2 - (2.0 * m[:-1] + m[1:]) / h
    aH = 2.0 * (y0 - y[1:]) / h**3 + (m[:-1] + m[1:]) / h**2
    return aH, bH, cH, dH


@jax.jit
def calcPmax_cubic(v, j):
    """(pmax, vmax) of P = v·j via PCHIP cubic, analytic per-segment max."""
    p = v * j
    a, b, c, d = pchip_coefs(v, p)
    h = jnp.diff(v)

    def _cands(a_, b_, c_, h_):
        aa = jnp.where(a_ == 0, 1.0, a_)
        disc = 4.0 * b_**2 - 4.0 * (3.0 * aa) * c_
        sqrt = jnp.sqrt(jnp.maximum(disc, 1e-24))  # eps avoids inf·0=nan in grad at disc==0
        r1 = (-2.0 * b_ + sqrt) / (6.0 * aa)
        r2 = (-2.0 * b_ - sqrt) / (6.0 * aa)
        q2 = (disc > 0.0) & (a_ != 0.0)
        r1 = jnp.where(q2 & (r1 > 0.0) & (r1 < h_), r1, -1.0)
        r2 = jnp.where(q2 & (r2 > 0.0) & (r2 < h_), r2, -1.0)
        r3 = jnp.where(b_ == 0.0, -1.0, -c_ / (2.0 * b_ + 1e-24))  # eps avoids inf·0=nan in grad
        r3 = jnp.where((a_ == 0.0) & (r3 > 0.0) & (r3 < h_), r3, -1.0)
        return jnp.stack([jnp.zeros_like(h_), h_, r1, r2, r3])

    dx = vmap(_cands)(a, b, c, h)
    y = a[:, None] * dx**3 + b[:, None] * dx**2 + c[:, None] * dx + d[:, None]
    valid = (dx >= 0.0) & (dx <= h[:, None])
    y = jnp.where(valid, y, -jnp.inf)
    idx = jnp.argmax(y)
    seg = idx // 5
    off = dx.flatten()[idx]
    return y.flatten()[idx], v[seg] + off


def calcPmax_smooth(v, j, tau):
    """Soft-maximum MPP: smooth in (v, j), no segment-selection jump.

    Same PCHIP candidate values as calcPmax_cubic, but the maximum over
    candidates is replaced by log-sum-exp with temperature tau (power
    units): pmax_soft = tau*log(sum(exp(y/tau))), Vmpp_soft =
    softmax-weighted candidate voltages. Satisfies
    0 <= pmax_soft - max(y) <= tau*log(#candidates); -inf-masked invalid
    candidates contribute exactly zero weight. The default hard selection
    is unchanged; opt in via Sweep(mpp_tau=...) when a globally smooth
    efficiency objective is needed for differentiation. tau is a physical
    modeling choice (document it): smaller tau tracks the hard MPP more
    closely but concentrates weight (stiffer gradients).
    """
    p = v * j
    a, b, c, d = pchip_coefs(v, p)
    h = jnp.diff(v)

    def _cands(a_, b_, c_, h_):
        aa = jnp.where(a_ == 0, 1.0, a_)
        disc = 4.0 * b_**2 - 4.0 * (3.0 * aa) * c_
        sqrt = jnp.sqrt(jnp.maximum(disc, 1e-24))
        r1 = (-2.0 * b_ + sqrt) / (6.0 * aa)
        r2 = (-2.0 * b_ - sqrt) / (6.0 * aa)
        q2 = (disc > 0.0) & (a_ != 0.0)
        r1 = jnp.where(q2 & (r1 > 0.0) & (r1 < h_), r1, -1.0)
        r2 = jnp.where(q2 & (r2 > 0.0) & (r2 < h_), r2, -1.0)
        r3 = jnp.where(b_ == 0.0, -1.0, -c_ / (2.0 * b_ + 1e-24))
        r3 = jnp.where((a_ == 0.0) & (r3 > 0.0) & (r3 < h_), r3, -1.0)
        return jnp.stack([jnp.zeros_like(h_), h_, r1, r2, r3])

    dx = vmap(_cands)(a, b, c, h)
    y = a[:, None] * dx**3 + b[:, None] * dx**2 + c[:, None] * dx + d[:, None]
    valid = (dx >= 0.0) & (dx <= h[:, None])
    y = jnp.where(valid, y, -jnp.inf)
    # Candidate values + voltages: segment base v[:-1] plus offsets.
    yf = y.flatten()
    vf = (v[:-1, None] + dx).flatten()
    # Shift by max for exp-range safety (all-tracer, no concretization).
    m = jnp.max(jnp.where(jnp.isfinite(yf), yf, -jnp.inf))
    e = jnp.exp(jnp.where(jnp.isfinite(yf), (yf - m) / tau, -jnp.inf))
    Z = jnp.sum(e) + 1e-300
    w = e / Z
    pmax_soft = m + tau * jnp.log(Z)
    return pmax_soft, jnp.sum(w * vf)


@jax.jit
def _qspline_coefs(x, y):
    """Quadratic spline coefficients (a, b, c per segment) solved as one
    dense system (C0 through knots, derivative continuity at interior
    knots, first knot anchored). Vectorized — no Python loops."""
    n = x.size
    ns = n - 1  # number of segments
    dim = 3 * ns
    M = jnp.zeros((dim, dim))
    z = jnp.zeros(dim)

    # Row 0: anchor (a0 = 0)
    M = M.at[0, 0].set(1)

    # C0 at knots: row 3i+1 uses left endpoint, row 3i+2 uses right endpoint
    i_seg = jnp.arange(ns)
    row_left = 3 * i_seg + 1
    row_right = 3 * i_seg + 2
    col_base = 3 * i_seg

    # Left endpoint: [x[i]^2, x[i], 1]
    M = M.at[row_left, col_base].set(x[:-1] ** 2)
    M = M.at[row_left, col_base + 1].set(x[:-1])
    M = M.at[row_left, col_base + 2].set(1.0)
    z = z.at[row_left].set(y[:-1])

    # Right endpoint: [x[i+1]^2, x[i+1], 1]
    M = M.at[row_right, col_base].set(x[1:] ** 2)
    M = M.at[row_right, col_base + 1].set(x[1:])
    M = M.at[row_right, col_base + 2].set(1.0)
    z = z.at[row_right].set(y[1:])

    # Derivative continuity at interior knots: row 3i+3
    if ns > 1:
        i_int = jnp.arange(ns - 1)
        row_d = 3 * (i_int + 1)
        col_d = 3 * i_int
        xm = x[1:-1]
        M = M.at[row_d, col_d].set(2 * xm)
        M = M.at[row_d, col_d + 1].set(1.0)
        M = M.at[row_d, col_d + 2].set(0.0)
        M = M.at[row_d, col_d + 3].set(-2 * xm)
        M = M.at[row_d, col_d + 4].set(-1.0)
        M = M.at[row_d, col_d + 5].set(0.0)

    coef = jnp.linalg.solve(M, z)
    return coef[::3], coef[1::3], coef[2::3]


@jax.jit
def calcPmax_quadratic(v, j):
    """(pmax, vmax) via the quadratic spline reference."""
    a, b, c = _qspline_coefs(v, v * j)
    xl, xu = v[:-1], v[1:]
    filla = jnp.where(a != 0, a, 1.0)
    xm = jnp.clip(-b / (2 * filla), xl, xu)

    def quad(X, a_, b_, c_):
        return a_ * X**2 + b_ * X + c_

    yl = quad(xl, a, b, c)
    yu = quad(xu, a, b, c)
    ym = quad(xm, a, b, c)
    idx = jnp.argmax(jnp.concatenate([yl, yu, ym]))
    xall = jnp.concatenate([xl, xu, xm])
    yall = jnp.concatenate([yl, yu, ym])
    return yall[idx], xall[idx]
