"""Scharfetter–Gummel current discretisation with a stable Bernoulli kernel.

The Bernoulli function B(z) = z/(e^z − 1) is the exact finite-volume flux
for the drift–diffusion operator on a uniform cell (Scharfetter & Gummel
1969).  Its stable evaluation is the numerical crux:

  B(z) = 1 − z/2 + z²/12 + O(z⁴)   for |z| ≪ 1  (Taylor, avoids 0/0)
       = z/(e^z − 1)                otherwise    (clipped exponent)

A *forward-only* formulation: the Jacobian comes from jax.jacrev of the
residual assembly (hand-coded derivatives).
The custom JVP on Bernoulli protects the derivative at z ≈ 0 (naive AD of
the `where`-selected branch can emit 0/0 when the discarded branch is
differentiated).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import custom_jvp

from driftjax.fields import Potentials, PVCell

# Threshold for clamping small Dpsi values to avoid 0/0 in Bernoulli evaluation.
DPSI_CLAMP = 1e-5
# Threshold for the Taylor series branch of the Bernoulli function.
BERN_TAYLOR_THRESH = 1e-6
# Threshold for the JVP Taylor series branch.
BERN_JVP_TAYLOR_THRESH = 1e-4


@custom_jvp
def bernoulli(x: jax.Array) -> jax.Array:
    """x/(e^x − 1), numerically stable and smooth for all real x."""
    taylor = 1 - x / 2 + x**2 / 12
    expm = jnp.exp(jnp.clip(x, -700, 700))
    exact = jnp.where(jnp.abs(expm - 1) < 1e-15, taylor, x / (expm - 1))
    return jnp.where(jnp.abs(x) < BERN_TAYLOR_THRESH, taylor, exact)


@bernoulli.defjvp
def _bernoulli_jvp(primals, tangents):
    """Exact derivative: B' = B·(1 − x − B)/x, series near 0 (stable JVP)."""
    (x,) = primals
    (dx,) = tangents
    b = bernoulli(x)
    series = -0.5 + x * (1.0 / 6.0) - x**3 / 180.0
    d = jnp.where(jnp.abs(x) < BERN_JVP_TAYLOR_THRESH, series, b * (1 - x - b) / x)
    return b, d * dx


def _Jn_impl(cell: PVCell, pot: Potentials) -> jax.Array:
    """Electron current density (dimensionless), shape (n−1,) — impl."""
    phi = pot.phi
    phi_n = pot.phi_n
    fm = jnp.exp(phi_n[1:]) - jnp.exp(phi_n[:-1])
    psi = cell.Chi + jnp.log(cell.Nc) + phi
    psip = psi[:-1]
    psim = psi[1:]
    Dpsi = psip - psim
    Dpsi_exact = jnp.where(jnp.abs(Dpsi) < DPSI_CLAMP, DPSI_CLAMP, Dpsi)
    dpsi_s = jnp.clip(Dpsi, -DPSI_CLAMP, DPSI_CLAMP)
    exp_psip = jnp.exp(psip)
    bern = jnp.where(
        jnp.abs(Dpsi) < DPSI_CLAMP,
        exp_psip / (1 + dpsi_s / 2 + dpsi_s**2 / 6),
        exp_psip * Dpsi_exact / (jnp.exp(Dpsi_exact) - 1),
    )
    return cell.mn[:-1] * bern * fm / cell.dgrid


Jn = custom_jvp(_Jn_impl)


@Jn.defjvp
def _Jn_jvp(primals, tangents):
    cell, pot = primals
    dcell, dpot = tangents
    y = _Jn_impl(cell, pot)
    # Analytic pot part (hot path for g_x) via _Jn_deriv-like formulas
    # Use analytic derivatives for phi_n and phi; cell part via jvp (less hot)
    try:
        # Lazy import to avoid circular at import time
        from driftjax.numerics.analytic_jacobian import _Jn_deriv as _Jn_deriv_analytic

        DJnDphi_n0, DJnDphi_n1, DJnDphi0, DJnDphi1 = _Jn_deriv_analytic(cell, pot)
        dJn_pot = (
            DJnDphi_n0 * dpot.phi_n[:-1]
            + DJnDphi_n1 * dpot.phi_n[1:]
            + DJnDphi0 * dpot.phi[:-1]
            + DJnDphi1 * dpot.phi[1:]
        )
    except Exception:
        # Fallback to autodiff for pot if analytic not available
        _, dJn_pot = jax.jvp(lambda p: _Jn_impl(cell, p), (pot,), (dpot,))
        # dJn_pot is already the full pot contribution, return directly with cell part
        _, dJn_cell = jax.jvp(lambda c: _Jn_impl(c, pot), (cell,), (dcell,))
        return y, dJn_pot + dJn_cell
    # Cell part via jvp (covers mn, Chi, Nc etc.)
    _, dJn_cell = jax.jvp(lambda c: _Jn_impl(c, pot), (cell,), (dcell,))
    return y, dJn_pot + dJn_cell


def _Jp_impl(cell: PVCell, pot: Potentials) -> jax.Array:
    """Hole current density (dimensionless), shape (n−1,) — impl."""
    phi = pot.phi
    phi_p = pot.phi_p
    fm = jnp.exp(-phi_p[1:]) - jnp.exp(-phi_p[:-1])
    psi = cell.Chi + cell.Eg - jnp.log(cell.Nv) + phi
    psip = psi[:-1]
    psim = psi[1:]
    Dpsi = psip - psim
    Dpsi_exact = jnp.where(jnp.abs(Dpsi) < DPSI_CLAMP, DPSI_CLAMP, Dpsi)
    dpsi_s = jnp.clip(Dpsi, -DPSI_CLAMP, DPSI_CLAMP)
    exp_psi = jnp.exp(-psip)
    bern = jnp.where(
        jnp.abs(Dpsi) < DPSI_CLAMP,
        exp_psi / (-1 + dpsi_s / 2 - dpsi_s**2 / 6),
        exp_psi * Dpsi_exact / (jnp.exp(-Dpsi_exact) - 1),
    )
    return cell.mp[:-1] * bern * fm / cell.dgrid


Jp = custom_jvp(_Jp_impl)


@Jp.defjvp
def _Jp_jvp(primals, tangents):
    cell, pot = primals
    dcell, dpot = tangents
    y = _Jp_impl(cell, pot)
    try:
        from driftjax.numerics.analytic_jacobian import _Jp_deriv as _Jp_deriv_analytic

        DJpDphi_p0, DJpDphi_p1, DJpDphi0, DJpDphi1 = _Jp_deriv_analytic(cell, pot)
        dJp_pot = (
            DJpDphi_p0 * dpot.phi_p[:-1]
            + DJpDphi_p1 * dpot.phi_p[1:]
            + DJpDphi0 * dpot.phi[:-1]
            + DJpDphi1 * dpot.phi[1:]
        )
    except Exception:
        _, dJp_pot = jax.jvp(lambda p: _Jp_impl(cell, p), (pot,), (dpot,))
        _, dJp_cell = jax.jvp(lambda c: _Jp_impl(c, pot), (cell,), (dcell,))
        return y, dJp_pot + dJp_cell
    _, dJp_cell = jax.jvp(lambda c: _Jp_impl(c, pot), (cell,), (dcell,))
    return y, dJp_pot + dJp_cell
