"""Residual assembly F(x) for the coupled drift–diffusion–Poisson system.

Assembly order (interleaved, the single layout):
    [φn₀, φp₀, φ₀, φn₁, φp₁, φ₁, …]  (see types.pot2vec)

Two Jacobian paths exist:
  * ``F_jacobian`` — the canonical reference, forward-mode AD (``jax.jacfwd``)
    of the flat residual (identical values to reverse-mode to ~1e-16, but
    ~2x faster to trace and ~35% faster to execute);
  * ``numerics.analytic_jacobian.banded_jacobian`` — the hand-coded analytic
    block-tridiagonal fast backend used by the Newton loop by default,
    pinned against ``F_jacobian`` by tests/unit/test_analytic_jacobian.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev

from driftjax.fields import BoundaryConditions, Potentials, PVCell, pot2vec, vec2pot
from driftjax.numerics.drift_diffusion import ddn, ddn_precomputed, ddp, ddp_precomputed
from driftjax.numerics.poisson import poisson
from driftjax.science.contacts import contact_phi, contact_phin, contact_phip
from driftjax.science.carrier_statistics import n as _n
from driftjax.science.carrier_statistics import ni as _ni
from driftjax.science.carrier_statistics import p as _p
from driftjax.science.recombination import total_precomputed


def comp_F(cell: PVCell, bound: BoundaryConditions, pot: Potentials) -> jax.Array:
    """Full 3n-dimensional residual (interleaved order)."""
    n = pot.n
    dd_n = ddn(cell, pot)
    dd_p = ddp(cell, pot)
    pois = poisson(cell, pot)
    ct_phin0, ct_phinL = contact_phin(cell, bound, pot)
    ct_phip0, ct_phipL = contact_phip(cell, bound, pot)
    ct_phi0, ct_phiL = contact_phi(cell, bound, pot)
    return jnp.concatenate(
        [
            jnp.expand_dims(ct_phin0, 0),
            jnp.expand_dims(ct_phip0, 0),
            jnp.expand_dims(ct_phi0, 0),
            jnp.stack([dd_n, dd_p, pois], axis=1).reshape(3 * (n - 2)),
            jnp.expand_dims(ct_phinL, 0),
            jnp.expand_dims(ct_phipL, 0),
            jnp.expand_dims(ct_phiL, 0),
        ]
    )


def comp_F_precomputed(cell: PVCell, bound: BoundaryConditions, pot: Potentials) -> tuple:
    """Residual with carrier stats (n_v, p_v, ni_v, R) computed once.

    Returns (F, n_v, p_v, ni_v) so callers can reuse the carrier stats.
    """
    n_arr = pot.n
    n_v = _n(cell, pot)
    p_v = _p(cell, pot)
    ni_v = _ni(cell)
    R = total_precomputed(cell, n_v, p_v, ni_v)
    dd_n = ddn_precomputed(cell, pot, R)
    dd_p = ddp_precomputed(cell, pot, R)
    pois = poisson(cell, pot)
    ct_phin0, ct_phinL = contact_phin(cell, bound, pot)
    ct_phip0, ct_phipL = contact_phip(cell, bound, pot)
    ct_phi0, ct_phiL = contact_phi(cell, bound, pot)
    F = jnp.concatenate(
        [
            jnp.expand_dims(ct_phin0, 0),
            jnp.expand_dims(ct_phip0, 0),
            jnp.expand_dims(ct_phi0, 0),
            jnp.stack([dd_n, dd_p, pois], axis=1).reshape(3 * (n_arr - 2)),
            jnp.expand_dims(ct_phinL, 0),
            jnp.expand_dims(ct_phipL, 0),
            jnp.expand_dims(ct_phiL, 0),
        ]
    )
    return F, n_v, p_v, ni_v


def F_jacobian(cell: PVCell, bound: BoundaryConditions, pot: Potentials) -> jax.Array:
    """Dense (3n, 3n) Jacobian of comp_F via forward AD (the only source)."""

    def f(v):
        return comp_F(cell, bound, vec2pot(v))

    return jacfwd(f)(pot2vec(pot))


def F_eq(cell: PVCell, bound: BoundaryConditions, pot: Potentials) -> jax.Array:
    """Equilibrium residual: Poisson only (φn = φp = 0) + Dirichlet φ BCs."""
    pois = poisson(cell, pot)
    ct_phi0, ct_phiL = contact_phi(cell, bound, pot)
    return jnp.concatenate([jnp.expand_dims(ct_phi0, 0), pois, jnp.expand_dims(ct_phiL, 0)])


def F_eq_jacobian(cell: PVCell, bound: BoundaryConditions, pot: Potentials) -> jax.Array:
    """(n, n) Jacobian of F_eq wrt the flat φ vector."""

    def eq_from_phi(phi):
        return F_eq(cell, bound, Potentials(jnp.zeros_like(phi), jnp.zeros_like(phi), phi))

    return jacrev(eq_from_phi)(pot.phi)
