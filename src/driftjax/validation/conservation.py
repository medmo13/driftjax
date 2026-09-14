"""Conservation / identity residuals on a converged solution (test pyramid L5)."""

from __future__ import annotations

import jax.numpy as jnp

from driftjax.numerics.scharfetter_gummel import Jn, Jp
from driftjax.science.carrier_statistics import n as _n
from driftjax.science.carrier_statistics import ni
from driftjax.science.carrier_statistics import p as _p
from driftjax.science.recombination import total as recomb_total


def terminal_current_variation(cell, pot) -> float:
    """Max edge-to-edge relative variation of Jn+Jp (uniformity)."""
    j = Jn(cell, pot) + Jp(cell, pot)
    return float(jnp.max(jnp.abs(j[1:] - j[:-1])) / (jnp.max(jnp.abs(j)) + 1e-30))


def mass_action_residual(cell, pot) -> float:
    """max |n·p − ni²| / ni² (equilibrium only)."""
    ni2 = ni(cell) ** 2
    return float(jnp.max(jnp.abs(_n(cell, pot) * _p(cell, pot) - ni2) / ni2))


def gr_balance_residual(cell, pot) -> float:
    """Summed electron continuity on interior nodes: |Σ(R−G)·ave − ΔJn| / scale.

    Matches the exact finite-volume identity used by the solver residual
    (drift_diffusion.ddn): (Jn[i+1]−Jn[i]) = (R−G)ᵢ·aveᵢ at every interior
    node, so the summed identity holds to solver tolerance.  Scale: mean
    |Jn+Jp| (the device current scale; Jn alone is ~0 at a blocking contact)."""
    jn = Jn(cell, pot)
    j = jn + Jp(cell, pot)
    # ddn pairs node i (1..n-2) with edge i−1:  Jn[i]−Jn[i−1] = (R−G)ᵢ·ave_{i−1}
    ave = 0.5 * (cell.dgrid[:-1] + cell.dgrid[1:])
    r = recomb_total(cell, pot)[1:-1]
    g = cell.G[1:-1]
    delta = float(jn[-1] - jn[0])          # telescopes over interior nodes
    integral = float(jnp.sum((r - g) * ave))
    scale = float(jnp.mean(jnp.abs(j)) + 1e-30)
    return float(jnp.abs(delta - integral) / scale)
