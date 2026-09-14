"""Electron/hole continuity residuals (finite volume, interior nodes)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from driftjax.fields import Potentials, PVCell
from driftjax.numerics.scharfetter_gummel import Jn, Jp
from driftjax.science.recombination import total as recomb_total


def ddn(cell: PVCell, pot: Potentials) -> jax.Array:
    """Electron continuity residual: −R + G + ∂x Jn = 0."""
    R = recomb_total(cell, pot)
    ave = (cell.dgrid[:-1] + cell.dgrid[1:]) / 2
    return -R[1:-1] + cell.G[1:-1] + jnp.diff(Jn(cell, pot)) / ave


def ddp(cell: PVCell, pot: Potentials) -> jax.Array:
    """Hole continuity residual: R − G + ∂x Jp = 0."""
    R = recomb_total(cell, pot)
    ave = (cell.dgrid[:-1] + cell.dgrid[1:]) / 2
    return R[1:-1] - cell.G[1:-1] + jnp.diff(Jp(cell, pot)) / ave


def ddn_precomputed(cell: PVCell, pot: Potentials, R: jax.Array) -> jax.Array:
    """Electron continuity with pre-computed recombination R (avoids redundant n/p/ni)."""
    ave = (cell.dgrid[:-1] + cell.dgrid[1:]) / 2
    return -R[1:-1] + cell.G[1:-1] + jnp.diff(Jn(cell, pot)) / ave


def ddp_precomputed(cell: PVCell, pot: Potentials, R: jax.Array) -> jax.Array:
    """Hole continuity with pre-computed recombination R (avoids redundant n/p/ni)."""
    ave = (cell.dgrid[:-1] + cell.dgrid[1:]) / 2
    return R[1:-1] - cell.G[1:-1] + jnp.diff(Jp(cell, pot)) / ave
