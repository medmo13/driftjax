"""Recombination: SRH, radiative, Auger — all ∝ (np − ni²) (detailed balance)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from driftjax.fields import Potentials, PVCell
from driftjax.science.carrier_statistics import n, ni, p


def srh(cell: PVCell, pot: Potentials) -> jax.Array:
    """Shockley–Read–Hall rate: (np−ni²) / (τp(n + n1) + τn(p + p1)) with
    n1 = ni·e^{Et}, p1 = ni·e^{−Et}, i.e. Et = (E_trap − E_i)/kT in Vt
    units (0 = midgap trap, the most recombination-active position).
    Reference: H. Föll / SRH theory — the auxiliary densities are measured
    from the intrinsic (midgap) level, n1·p1 = ni², and recombination peaks
    at E_trap = E_i. (AUDIT: ``Material.Et`` was documented as "trap level
    above Ev"; the implementation — and the physical default Et=0 —
    is the Ei-relative convention. Docs corrected, math unchanged.)"""
    ni_v = ni(cell)
    n_v = n(cell, pot)
    p_v = p(cell, pot)
    return (n_v * p_v - ni_v**2) / (
        cell.tp * (ni_v * jnp.exp(cell.Et) + n_v) + cell.tn * (ni_v * jnp.exp(-cell.Et) + p_v)
    )


def radiative(cell: PVCell, pot: Potentials) -> jax.Array:
    """Radiative (bimolecular) rate: Br·(np − ni²)."""
    ni_v = ni(cell)
    return cell.Br * (n(cell, pot) * p(cell, pot) - ni_v**2)


def auger(cell: PVCell, pot: Potentials) -> jax.Array:
    """Auger rate: (Cn·n + Cp·p)·(np − ni²)."""
    ni_v = ni(cell)
    n_v = n(cell, pot)
    p_v = p(cell, pot)
    return (cell.Cn * n_v + cell.Cp * p_v) * (n_v * p_v - ni_v**2)


def total(cell: PVCell, pot: Potentials) -> jax.Array:
    """Total recombination R = R_SRH + R_rad + R_Auger (dimensionless)."""
    return srh(cell, pot) + radiative(cell, pot) + auger(cell, pot)


def _srh_precomputed(cell, n_v, p_v, ni_v):
    return (n_v * p_v - ni_v**2) / (
        cell.tp * (ni_v * jnp.exp(cell.Et) + n_v) + cell.tn * (ni_v * jnp.exp(-cell.Et) + p_v)
    )


def _radiative_precomputed(cell, n_v, p_v, ni_v):
    return cell.Br * (n_v * p_v - ni_v**2)


def _auger_precomputed(cell, n_v, p_v, ni_v):
    return (cell.Cn * n_v + cell.Cp * p_v) * (n_v * p_v - ni_v**2)


def total_precomputed(cell, n_v, p_v, ni_v):
    """Total recombination with pre-computed carrier stats (avoids redundant n/p/ni)."""
    return _srh_precomputed(cell, n_v, p_v, ni_v) + _radiative_precomputed(cell, n_v, p_v, ni_v) + _auger_precomputed(cell, n_v, p_v, ni_v)
