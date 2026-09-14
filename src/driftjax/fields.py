"""Core pytree types for driftjax.

Frozen, JIT-safe dataclasses built on equinox.  Every physical quantity is
dimensionless unless explicitly documented (energies in units of Vt,
densities in units of 1e19 cm⁻³, lengths in units of L0).

Single DOF ordering (invariant, used everywhere):
    interleaved  [φn₀, φp₀, φ₀, φn₁, φp₁, φ₁, ...]
``pot2vec`` / ``vec2pot`` are the only place the layout is defined.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

F64 = Float[Array, "..."]


class Potentials(eqx.Module):
    """The three unknown fields of the drift-diffusion–Poisson system."""

    phi_n: Float[Array, n]  # electron quasi-Fermi potential (Vt units)
    phi_p: Float[Array, n]  # hole quasi-Fermi potential (Vt units)
    phi: Float[Array, n]  # electrostatic potential (Vt units)

    def __init__(self, phi_n: Any, phi_p: Any, phi: Any) -> None:
        self.phi_n = jnp.asarray(phi_n, dtype=jnp.float64)
        self.phi_p = jnp.asarray(phi_p, dtype=jnp.float64)
        self.phi = jnp.asarray(phi, dtype=jnp.float64)

    @property
    def n(self) -> int:
        return int(self.phi.shape[0])


class BoundaryConditions(eqx.Module):
    """Contact boundary conditions: potentials + equilibrium carrier densities."""

    phi0: Float[Array, ""]
    phiL: Float[Array, ""]
    neq0: Float[Array, ""]
    neqL: Float[Array, ""]
    peq0: Float[Array, ""]
    peqL: Float[Array, ""]

    def __init__(
        self,
        phi0: Any = 0.0,
        phiL: Any = 0.0,
        neq0: Any = 0.0,
        neqL: Any = 0.0,
        peq0: Any = 0.0,
        peqL: Any = 0.0,
    ) -> None:
        self.phi0 = jnp.asarray(phi0, dtype=jnp.float64)
        self.phiL = jnp.asarray(phiL, dtype=jnp.float64)
        self.neq0 = jnp.asarray(neq0, dtype=jnp.float64)
        self.neqL = jnp.asarray(neqL, dtype=jnp.float64)
        self.peq0 = jnp.asarray(peq0, dtype=jnp.float64)
        self.peqL = jnp.asarray(peqL, dtype=jnp.float64)


class Material(eqx.Module):
    """Bulk semiconductor material (physical units at construction; scaled on bake)."""

    Chi: Float[Array, ""]  # electron affinity (eV)
    Eg: Float[Array, ""]  # band gap (eV)
    eps: Float[Array, ""]  # relative permittivity
    Nc: Float[Array, ""]  # conduction-band DOS (cm⁻³)
    Nv: Float[Array, ""]  # valence-band DOS (cm⁻³)
    mn: Float[Array, ""]  # electron mobility (cm²/V·s)
    mp: Float[Array, ""]  # hole mobility
    tn: Float[Array, ""]  # electron lifetime (s)
    tp: Float[Array, ""]  # hole lifetime
    Et: Float[Array, ""]  # SRH trap level relative to the intrinsic level Ei, in eV (0 = midgap; AUDIT: was documented as "above Ev", which is inconsistent with the n1/p1 = ni·e^{±Et} implementation — see science/recombination.py)
    Br: Float[Array, ""]  # radiative coefficient (cm³/s)
    Cn: Float[Array, ""]  # electron Auger coefficient (cm⁶/s)
    Cp: Float[Array, ""]  # hole Auger coefficient
    A: Float[Array, ""]  # Tauc prefactor (cm⁻¹·eV⁻¹ᐟ²)
    alpha: Float[Array, nlam]  # tabulated absorption (m⁻¹)
    Lambda: Float[Array, nlam]  # tabulated wavelengths (m)

    def __init__(
        self,
        Chi: Any,
        Eg: Any,
        eps: Any,
        Nc: Any,
        Nv: Any,
        mn: Any,
        mp: Any,
        tn: Any,
        tp: Any,
        Et: Any = 0.0,
        Br: Any = 0.0,
        Cn: Any = 0.0,
        Cp: Any = 0.0,
        A: Any = 0.0,
        alpha: Any = None,
        Lambda: Any = None,
    ) -> None:
        f64 = jnp.float64
        self.Chi = jnp.asarray(Chi, dtype=f64)
        self.Eg = jnp.asarray(Eg, dtype=f64)
        self.eps = jnp.asarray(eps, dtype=f64)
        self.Nc = jnp.asarray(Nc, dtype=f64)
        self.Nv = jnp.asarray(Nv, dtype=f64)
        self.mn = jnp.asarray(mn, dtype=f64)
        self.mp = jnp.asarray(mp, dtype=f64)
        self.tn = jnp.asarray(tn, dtype=f64)
        self.tp = jnp.asarray(tp, dtype=f64)
        self.Et = jnp.asarray(Et, dtype=f64)
        self.Br = jnp.asarray(Br, dtype=f64)
        self.Cn = jnp.asarray(Cn, dtype=f64)
        self.Cp = jnp.asarray(Cp, dtype=f64)
        self.A = jnp.asarray(A, dtype=f64)
        self.alpha = jnp.asarray(alpha if alpha is not None else jnp.array([]), dtype=f64)
        self.Lambda = jnp.asarray(Lambda if Lambda is not None else jnp.array([]), dtype=f64)


class LightSource(eqx.Module):
    """Incident spectrum."""

    Lambda: Float[Array, nlam]  # wavelengths (nm — package convention; optics multiplies by 1e-9)
    P_in: Float[Array, nlam]  # power density (W/m²)
    kind: str = eqx.field(static=True, default="custom")

    def __init__(self, Lambda: Any, P_in: Any, kind: str = "custom") -> None:
        self.Lambda = jnp.asarray(Lambda, dtype=jnp.float64)
        self.P_in = jnp.asarray(P_in, dtype=jnp.float64)
        self.kind = kind


class DeviceDesign(eqx.Module):
    """1-D device specification: mesh + per-node material/doping/contact data."""

    dgrid: Float[Array, n - 1]  # node spacings (dimensionless)
    x: Float[Array, n]  # node positions (dimensionless)

    eps: Float[Array, n]
    Chi: Float[Array, n]
    Eg: Float[Array, n]
    Nc: Float[Array, n]
    Nv: Float[Array, n]
    mn: Float[Array, n]
    mp: Float[Array, n]
    tn: Float[Array, n]
    tp: Float[Array, n]
    Et: Float[Array, n]
    Br: Float[Array, n]
    Cn: Float[Array, n]
    Cp: Float[Array, n]
    A: Float[Array, n]
    alpha: Float[Array, "n_tab n"]  # α(λ_tab, x) in m⁻¹
    Ndop: Float[Array, n]  # net doping (cm⁻³, scaled)
    Snl: Float[Array, ""]
    Snr: Float[Array, ""]
    Spl: Float[Array, ""]
    Spr: Float[Array, ""]
    PhiMl: Float[Array, ""]  # metal work function (eV); 0 → ohmic
    PhiMr: Float[Array, ""]
    T: Float[Array, ""]  # temperature (K); design is built at this T

    def __init__(self, x: Any, dgrid: Any, **kwargs: Any) -> None:
        f64 = jnp.float64
        self.x = jnp.asarray(x, dtype=f64)
        self.dgrid = jnp.asarray(dgrid, dtype=f64)
        if self.x.shape[0] < 3:
            raise ValueError(f"DeviceDesign requires n_points >= 3, got {self.x.shape[0]}")
        if self.dgrid.shape[0] != self.x.shape[0] - 1:
            raise ValueError(f"dgrid shape {self.dgrid.shape} must be x.shape[0]-1")
        for k in [
            "eps",
            "Chi",
            "Eg",
            "Nc",
            "Nv",
            "mn",
            "mp",
            "tn",
            "tp",
            "Et",
            "Br",
            "Cn",
            "Cp",
            "A",
            "alpha",
            "Ndop",
        ]:
            setattr(self, k, jnp.asarray(kwargs[k], dtype=f64))
        for k in ["Snl", "Snr", "Spl", "Spr", "PhiMl", "PhiMr"]:
            setattr(self, k, jnp.asarray(kwargs[k], dtype=f64))
        self.T = jnp.asarray(kwargs.get("T", 300.0), dtype=f64)

    def with_temperature(self, new_t: float) -> DeviceDesign:
        """I5: re-scale a design built at T=self.T to a new temperature.

        The dimensionless fields embed the thermal scales of the build
        temperature (potentials in Vt, lifetimes in units of the dielectric
        time, mesh in Debye lengths); the physical inputs are recovered by
        multiplying by the previous scales and re-dividing by the new ones.
        """
        from driftjax.units import thermal_scales

        new_T_arr = jnp.asarray(new_t, dtype=jnp.float64)
        so, sn = thermal_scales(self.T), thermal_scales(new_T_arr)
        rE = so["energy"] / sn["energy"]
        rV = so["velocity"] / sn["velocity"]
        rT = so["time"] / sn["time"]
        rL = so["length"] / sn["length"]
        repl = {}
        for k in ("Chi", "Eg", "Et", "PhiMl", "PhiMr"):
            repl[k] = getattr(self, k) * rE
        for k in ("Snl", "Snr", "Spl", "Spr"):
            repl[k] = getattr(self, k) * rV
        for k in ("tn", "tp"):
            repl[k] = getattr(self, k) * rT
        for k in ("Br", "Cn", "Cp"):
            repl[k] = getattr(self, k) / rT
        repl["x"] = self.x * rL
        repl["dgrid"] = self.dgrid * rL
        repl["T"] = new_T_arr
        return eqx.tree_at(lambda m: tuple(getattr(m, k) for k in repl), self, tuple(repl.values()))


class PVCell(eqx.Module):
    """A bake-time cell: design + generation profile + statistics mode.

    This is the object the Newton solve differentiates through.
    """

    dgrid: Float[Array, n - 1]
    x: Float[Array, n]
    G: Float[Array, n]  # generation profile (dimensionless)

    eps: Float[Array, n]
    Chi: Float[Array, n]
    Eg: Float[Array, n]
    Nc: Float[Array, n]
    Nv: Float[Array, n]
    mn: Float[Array, n]
    mp: Float[Array, n]
    tn: Float[Array, n]
    tp: Float[Array, n]
    Et: Float[Array, n]
    Br: Float[Array, n]
    Cn: Float[Array, n]
    Cp: Float[Array, n]
    Ndop: Float[Array, n]
    Snl: Float[Array, ""]
    Snr: Float[Array, ""]
    Spl: Float[Array, ""]
    Spr: Float[Array, ""]
    PhiMl: Float[Array, ""]
    PhiMr: Float[Array, ""]

    statistics: str = eqx.field(static=True, default="boltzmann")
    T: Float[Array, ""] = eqx.field(
        default=300.0
    )  # K (I5); always passed by init_cell/manufactured

    def __init__(self, **kwargs: Any) -> None:
        for field_name in [
            "dgrid",
            "x",
            "G",
            "eps",
            "Chi",
            "Eg",
            "Nc",
            "Nv",
            "mn",
            "mp",
            "tn",
            "tp",
            "Et",
            "Br",
            "Cn",
            "Cp",
            "Ndop",
            "Snl",
            "Snr",
            "Spl",
            "Spr",
            "PhiMl",
            "PhiMr",
        ]:
            if field_name not in kwargs:
                raise ValueError(f"Missing required field: {field_name}")
            setattr(self, field_name, jnp.asarray(kwargs[field_name], dtype=jnp.float64))
        self.statistics = kwargs.get("statistics", "boltzmann")
        self.T = jnp.asarray(kwargs.get("T", 300.0), dtype=jnp.float64)


# ---------------------------------------------------------------------------
# DOF layout (single source of truth)
# ---------------------------------------------------------------------------


def pot2vec(pot: Potentials) -> jax.Array:
    """Interleaved flat vector [φn₀, φp₀, φ₀, φn₁, φp₁, φ₁, ...]."""
    return jnp.stack([pot.phi_n, pot.phi_p, pot.phi], axis=1).reshape(-1)


def vec2pot(v: jax.Array) -> Potentials:
    """Inverse of ``pot2vec`` (layout invariant lives here)."""
    v_3d = v.reshape(-1, 3)
    return Potentials(v_3d[:, 0], v_3d[:, 1], v_3d[:, 2])
