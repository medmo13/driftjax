"""Poisson equation discretisation (finite volume, harmonic-mean ε).

∇·(ε∇φ) = −(p − n + Ndop)

Family fix carried forward (F-02): the permittivity on each face is the
harmonic mean 2ab/(a+b) — the only average preserving D = εE continuity
across a jump in ε.  Verified by MMS with grid-refinement order 2
(tests/convergence/test_mesh_order.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from driftjax.fields import Potentials, PVCell
from driftjax.science.carrier_statistics import n, p


def harmonic_ave_eps(cell: PVCell) -> jax.Array:
    """Face-centred permittivity (n−1,)."""
    a, b = cell.eps[1:], cell.eps[:-1]
    safe = jnp.where(jnp.abs(a + b) < 1e-30, 1e-30, a + b)
    return 2 * a * b / safe


def poisson(cell: PVCell, pot: Potentials) -> jax.Array:
    """Poisson residual at interior nodes 1..n−2 (dimensionless)."""
    dgrid = cell.dgrid
    ave = (dgrid[:-1] + dgrid[1:]) / 2
    eps_face = harmonic_ave_eps(cell)
    flux = eps_face * jnp.diff(pot.phi) / dgrid
    div_flux = (flux[:-1] - flux[1:]) / ave
    charge = p(cell, pot) - n(cell, pot) + cell.Ndop
    return div_flux - charge[1:-1]
