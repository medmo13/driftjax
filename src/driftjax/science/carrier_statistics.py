"""Carrier statistics: Boltzmann (default, validated) and approximate Fermi–Dirac.

Dimensionless formulation; energies are eV scaled by Vt (thermal voltage).

Fermi–Dirac mode ("exact" alias: historical name, NOT exact)
------------------------------------------------------------
The Fermi integral of order 1/2,

    F_{1/2}(η) = (2/√π) ∫₀^∞ t^{1/2}/(1 + e^{t−η}) dt
               = (4/√π) ∫₀¹ s(z)² σ(η − s(z)²)/(1−z)² dz,   s = z/(1−z),

is evaluated by 128-point Gauss–Legendre quadrature on the compactified
domain (accurate ~1e-6 for η ≲ 20) blended with the 3-term Sommerfeld
expansion for η ≥ 10 (relative error ~2e-4 at the switch, dropping fast).
Ground truth: F_{1/2}(0) = 0.7651, F_{1/2}(2.53) = 3.657, F_{1/2}(20) = 67.49
(scipy quad; pinned in tests/unit/test_statistics.py).

The derivatives of n and p carry custom JVPs: differentiating through the
Boltzmann chain naively produces exp-tangents that overflow in the Newton
Jacobian far from equilibrium; the manual JVP keeps every intermediate in
float range.  In FD mode the tangent uses the analytic F_{1/2} derivative
(F_{−1/2}/2) on the same quadrature/Sommerfeld machinery.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as _np
from jax import custom_jvp

from driftjax.fields import Potentials, PVCell

_GL_T, _GL_WT = _np.polynomial.legendre.leggauss(128)
_GL_N = jnp.asarray((_GL_T + 1.0) / 2.0, dtype=jnp.float64)  # [0, 1]
_GL_W = jnp.asarray(_GL_WT / 2.0, dtype=jnp.float64)

_SOMMERFELD_CUT = jnp.float64(10.0)


def _f_half_quad(eta: jax.Array) -> jax.Array:
    """F_{1/2} by 128-pt GL quadrature (η ≲ 20)."""
    s = _GL_N / (1.0 - _GL_N)
    dsdz = 1.0 / (1.0 - _GL_N) ** 2
    integrand = s**2 * jax.nn.sigmoid(eta - s**2) * dsdz
    return (4.0 / jnp.sqrt(jnp.pi)) * jnp.dot(_GL_W, integrand)


def _f_half_sommerfeld(eta: jax.Array) -> jax.Array:
    """3-term Sommerfeld expansion of F_{1/2} (η ≥ 10, err ~1e-6)."""
    e = jnp.asarray(eta, dtype=jnp.float64)
    return (2.0 / jnp.sqrt(jnp.pi)) * (
        (2.0 / 3.0) * e**1.5
        + (jnp.pi**2 / 12.0) * e**-0.5
        + (7.0 * jnp.pi**4 / 360.0) * (3.0 / 8.0) * e**-2.5
    )


def _f_half_exact(eta: jax.Array) -> jax.Array:
    """Quadrature for small η, Sommerfeld for large η, soft-switched."""
    w = 1.0 / (1.0 + jnp.exp(_SOMMERFELD_CUT - eta))
    som = _f_half_sommerfeld(jnp.maximum(eta, 1.0))
    return (1.0 - w) * _f_half_quad(eta) + w * som


def F_half(eta: jax.Array, mode: str = "exact") -> jax.Array:
    """Fermi–Dirac integral of order 1/2 (approximate evaluator).

    mode: "exact" (historical alias for the GL-quadrature + Sommerfeld
    blend below — approximate to ~1e-6 in F and ~2e-4 at the eta=10 blend,
    NOT exact), "boltzmann" (exp(η)), "blakemore" (the
    historical closed form — inaccurate, kept for provenance/cross-checking).
    """
    eta = jnp.asarray(eta, dtype=jnp.float64)
    if mode == "boltzmann":
        return jnp.exp(eta)
    if mode == "blakemore":
        # Historical closed form (Blakemore, "Semiconductor Statistics",
        # 1982): kept ONLY for provenance/A-B checks — it is NOT accurate
        # (≈60% error at η=1, 67% at η=5, 13% at η=20; envelope pinned in
        # tests/unit/test_statistics.py). Use "exact" for real degenerate
        # physics. Note the branch is discontinuous at η=0 by construction
        # (Boltzmann e^{η} for η≤0, raw formula →0 as η→0⁺); degenerate
        # Newton solves should use mode="exact".
        # For η ≤ 0 the e^{−η} term must dominate (Boltzmann branch) — the
        # raw formula's η^{3/2} is undefined there; driftjax fixes the previous
        # clamp (η→0 ⇒ F→0, which depopulated non-degenerate regions).
        # AD-safety floors: t**(-2.7) overflows past t ~ 1e-114 and
        # exp(-eta) past eta ~ -709; reverse-mode then forms 0 * inf = NaN
        # in the unselected branch, poisoning the Newton Jacobian. The floors
        # keep every intermediate finite (with finite tangents) for all
        # physical eta; where() still selects exp(eta) for eta <= 0, so the
        # primal is unchanged to machine precision there.
        t_raw = (3.0 * jnp.sqrt(jnp.pi) / 8.0) * jnp.maximum(eta, 0.0) ** 1.5
        t = jnp.maximum(t_raw, 1e-80)
        f = (jnp.exp(jnp.minimum(-eta, 700.0)) + t ** (-2.7)) ** (-1.0 / 2.7)
        return jnp.where(eta <= 0.0, jnp.exp(eta), f)
    core = _f_half_exact
    return jax.vmap(core)(eta) if eta.ndim else core(eta)


def _df_half_quad(eta: jax.Array) -> jax.Array:
    """F_{−1/2}/2 by quadrature (no s² weight — cheap and accurate)."""
    s = _GL_N / (1.0 - _GL_N)
    dsdz = 1.0 / (1.0 - _GL_N) ** 2
    integrand = jax.nn.sigmoid(eta - s**2) * dsdz
    return (2.0 / jnp.sqrt(jnp.pi)) * jnp.dot(_GL_W, integrand)


def _df_half_sommerfeld(eta: jax.Array) -> jax.Array:
    """Term-by-term derivative of the 3-term F_{1/2} expansion."""
    e = jnp.asarray(eta, dtype=jnp.float64)
    return (2.0 / jnp.sqrt(jnp.pi)) * (
        e**0.5 - (jnp.pi**2 / 24.0) * e**-1.5 - (7.0 * jnp.pi**4 / 384.0) * e**-3.5
    )


def _df_half_exact(eta: jax.Array) -> jax.Array:
    """Exact derivative of the implemented _f_half_exact blend (P0-1 fix).

    Primal: F = (1-w)*Q(eta) + w*S(c(eta)) with w = sigmoid(eta-10) and
    c(eta) = maximum(eta, 1).  Differentiating term by term,
        dF = (1-w)*Q' + w*S'*c' + w'*(S-Q),   w' = w*(1-w),
    where c' = 1{eta > 1} is the derivative of the clamp.  The previous
    implementation omitted the w'*(S-Q) blend-weight term, biasing dF
    around the eta ~= 10 transition by ~w'(S-Q) (~5e-5 relative).  At
    eta = 1 exactly the right-derivative is used (measure-zero kink).
    """
    e = jnp.asarray(eta, dtype=jnp.float64)
    w = 1.0 / (1.0 + jnp.exp(_SOMMERFELD_CUT - e))
    dw = w * (1.0 - w)
    ec = jnp.maximum(e, 1.0)
    cp = jnp.where(e > 1.0, 1.0, 0.0)
    S = _f_half_sommerfeld(ec)
    Q = _f_half_quad(e)
    return (1.0 - w) * _df_half_quad(e) + w * _df_half_sommerfeld(ec) * cp + dw * (S - Q)


def _blakemore_grad_single(eta: jax.Array) -> jax.Array:
    """Cached jitted grad for Blakemore to avoid re-tracing per call."""
    return jax.grad(lambda x: F_half(x, "blakemore"))(eta)


_blakemore_grad_jit = jax.jit(_blakemore_grad_single)


def dF_half(eta: jax.Array, mode: str = "exact") -> jax.Array:
    """dF_{1/2}/dη (analytic)."""
    eta = jnp.asarray(eta, dtype=jnp.float64)
    if mode == "boltzmann":
        return jnp.exp(eta)
    core: Callable[[Any], Any]
    if mode == "blakemore":
        core = _blakemore_grad_jit
    else:
        core = _df_half_exact
    return jax.vmap(core)(eta) if eta.ndim else core(eta)


def resolve_statistics(statistics: str) -> str:
    """Map public API names to internal F_{1/2} modes."""
    s = str(statistics).lower()
    if s in ("boltzmann", "classical", "non-degenerate"):
        return "boltzmann"
    if s in ("exact", "fermi-dirac", "fermi_dirac", "fd", "nilsson"):
        return "exact"
    if s in ("blakemore",):
        return "blakemore"
    raise ValueError(f"Unknown carrier statistics {statistics!r}")


@custom_jvp
def n(cell: PVCell, pot: Potentials) -> jax.Array:
    """Electron density n = Nc·F(η_n),  η_n = Chi + φn + φ."""
    mode = getattr(cell, "statistics", "boltzmann")
    eta = cell.Chi + pot.phi_n + pot.phi
    if mode == "boltzmann":
        return cell.Nc * jnp.exp(eta)
    return cell.Nc * F_half(eta, mode)


@n.defjvp
def n_jvp(primals, tangents):
    cell, pot = primals
    dcell, dpot = tangents
    mode = getattr(cell, "statistics", "boltzmann")
    eta = cell.Chi + pot.phi_n + pot.phi
    if mode == "boltzmann":
        e = jnp.exp(eta)
        return cell.Nc * e, e * (dcell.Nc + cell.Nc * (dcell.Chi + dpot.phi_n + dpot.phi))
    F = F_half(eta, mode)
    dF = dF_half(eta, mode)
    return cell.Nc * F, F * dcell.Nc + cell.Nc * dF * (dcell.Chi + dpot.phi_n + dpot.phi)


@custom_jvp
def p(cell: PVCell, pot: Potentials) -> jax.Array:
    """Hole density p = Nv·F(η_p),  η_p = −(Chi + Eg + φp + φ)."""
    mode = getattr(cell, "statistics", "boltzmann")
    eta = -(cell.Chi + cell.Eg + pot.phi_p + pot.phi)
    if mode == "boltzmann":
        return cell.Nv * jnp.exp(eta)
    return cell.Nv * F_half(eta, mode)


@p.defjvp
def p_jvp(primals, tangents):
    cell, pot = primals
    dcell, dpot = tangents
    mode = getattr(cell, "statistics", "boltzmann")
    eta = -(cell.Chi + cell.Eg + pot.phi_p + pot.phi)
    if mode == "boltzmann":
        e = jnp.exp(eta)
        return cell.Nv * e, e * (
            dcell.Nv - cell.Nv * (dcell.Chi + dcell.Eg + dpot.phi_p + dpot.phi)
        )
    F = F_half(eta, mode)
    dF = dF_half(eta, mode)
    return cell.Nv * F, F * dcell.Nv - cell.Nv * dF * (dcell.Chi + dcell.Eg + dpot.phi_p + dpot.phi)


def ni(cell: PVCell) -> jax.Array:
    """Intrinsic density √(Nc·Nv·exp(−Eg)).  Dimensionless."""
    return jnp.sqrt(cell.Nc * cell.Nv) * jnp.exp(-cell.Eg / 2)


def Ec(cell: PVCell) -> jax.Array:
    return -cell.Chi


def Ev(cell: PVCell) -> jax.Array:
    return -cell.Chi - cell.Eg


def EF_zero(cell: PVCell) -> jax.Array:
    """Equilibrium φ guess (Newton initial guess only, not the BC)."""
    _ni = ni(cell)
    Ndop = jnp.where(cell.Ndop != 0, cell.Ndop, _ni)
    dEF = jnp.where(Ndop > 0, jnp.log(jnp.abs(Ndop) / _ni), -jnp.log(jnp.abs(Ndop) / _ni))
    # Midgap reference includes the DOS asymmetry: Ei = (Ec+Ev)/2 + 0.5*log(Nv/Nc).
    return Ec(cell) - cell.Eg / 2 + 0.5 * jnp.log(cell.Nv / cell.Nc) + dEF
