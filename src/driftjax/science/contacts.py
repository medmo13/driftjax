"""Contact boundary conditions: ohmic/Schottky φ, physical equilibrium densities,
SRV (Robin) conditions for the carrier continuity equations.

Known package bug fixed here (pvx BUG-1): ``boundary_eq`` must return the
*physical* equilibrium carrier densities (identical to ``boundary_bias`` at
v=0).  Zeroing them gave FF = 2.33 under bias — nonsense.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from driftjax.fields import BoundaryConditions, Potentials, PVCell
from driftjax.science.carrier_statistics import n, p


def boundary_phi(cell: PVCell) -> tuple:
    """Built-in electrostatic potential at each contact (dimensionless).

    Ohmic contact (metal work function Φ ≤ 0): the thermal-equilibrium
    potential set by the doping.  Schottky (Φ > 0): φ_bi = −Φ_M.
    Handles intrinsic (Ndop==0) via midgap + 0.5*log(Nv/Nc).
    """
    N0, NL = cell.Ndop[0], cell.Ndop[-1]
    # Intrinsic branch: -Chi -Eg/2 +0.5*log(Nv/Nc)
    ohm0_intr = -cell.Chi[0] - cell.Eg[0] / 2 + 0.5 * jnp.log(cell.Nv[0] / cell.Nc[0])
    ohmL_intr = -cell.Chi[-1] - cell.Eg[-1] / 2 + 0.5 * jnp.log(cell.Nv[-1] / cell.Nc[-1])
    ohm0 = jnp.where(
        N0 > 0,
        -cell.Chi[0] + jnp.log(jnp.abs(N0 / cell.Nc[0])),
        jnp.where(
            N0 < 0,
            -cell.Chi[0] - cell.Eg[0] - jnp.log(jnp.abs(-N0 / cell.Nv[0])),
            ohm0_intr,
        ),
    )
    ohmL = jnp.where(
        NL > 0,
        -cell.Chi[-1] + jnp.log(jnp.abs(NL / cell.Nc[-1])),
        jnp.where(
            NL < 0,
            -cell.Chi[-1] - cell.Eg[-1] - jnp.log(jnp.abs(-NL / cell.Nv[-1])),
            ohmL_intr,
        ),
    )
    phi0 = jnp.where(cell.PhiMl > 0, -cell.PhiMl, ohm0)
    phiL = jnp.where(cell.PhiMr > 0, -cell.PhiMr, ohmL)
    return phi0, phiL


def eq_carrier_dens(cell: PVCell) -> tuple:
    """Equilibrium carrier densities at the contacts (n₀ scale)."""
    phi0, phiL = boundary_phi(cell)
    return (
        cell.Nc[0] * jnp.exp(cell.Chi[0] + phi0),
        cell.Nc[-1] * jnp.exp(cell.Chi[-1] + phiL),
        cell.Nv[0] * jnp.exp(-cell.Chi[0] - cell.Eg[0] - phi0),
        cell.Nv[-1] * jnp.exp(-cell.Chi[-1] - cell.Eg[-1] - phiL),
    )


def boundary_eq(cell: PVCell) -> BoundaryConditions:
    phi0, phiL = boundary_phi(cell)
    neq0, neqL, peq0, peqL = eq_carrier_dens(cell)
    return BoundaryConditions(phi0=phi0, phiL=phiL, neq0=neq0, neqL=neqL, peq0=peq0, peqL=peqL)


def boundary_bias(cell: PVCell, v: float | jax.Array) -> BoundaryConditions:
    """Boundary conditions with applied bias v (dimensionless)."""
    phi0, phiL = boundary_phi(cell)
    neq0, neqL, peq0, peqL = eq_carrier_dens(cell)
    return BoundaryConditions(phi0=phi0, phiL=phiL + v, neq0=neq0, neqL=neqL, peq0=peq0, peqL=peqL)


def contact_phin(cell: PVCell, bound: BoundaryConditions, pot: Potentials) -> tuple:
    """Electron-current SRV residuals at x=0 and x=L: Jn ∓ S·(n − n_eq)."""
    from driftjax.numerics.scharfetter_gummel import Jn

    n_v = n(cell, pot)
    Jn_v = Jn(cell, pot)
    return (
        Jn_v[0] - cell.Snl * (n_v[0] - bound.neq0),
        Jn_v[-1] + cell.Snr * (n_v[-1] - bound.neqL),
    )


def contact_phip(cell: PVCell, bound: BoundaryConditions, pot: Potentials) -> tuple:
    """Hole-current SRV residuals at x=0 and x=L: ±Jp + S·(p − p_eq)."""
    from driftjax.numerics.scharfetter_gummel import Jp

    p_v = p(cell, pot)
    Jp_v = Jp(cell, pot)
    return (
        Jp_v[0] + cell.Spl * (p_v[0] - bound.peq0),
        Jp_v[-1] - cell.Spr * (p_v[-1] - bound.peqL),
    )


def contact_phi(cell: PVCell, bound: BoundaryConditions, pot: Potentials) -> tuple:
    """Electrostatic Dirichlet residuals at the contacts."""
    return pot.phi[0] - bound.phi0, pot.phi[-1] - bound.phiL
