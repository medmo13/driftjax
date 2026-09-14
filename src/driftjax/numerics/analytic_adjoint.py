"""Analytic adjoint cell-cotangents: single-kernel, zero-AD per-bias backward.

Combines the ``J^T lam`` solve inputs and the cell cotangent
``u_cell - (dF/dcell)^T lam`` into one jitted kernel with **no**
``jax.grad`` / ``jax.vjp`` / ``jax.jacrev`` calls inside:

* :func:`analytic_g_x` — ``d mean(Jn+Jp) / d potvec`` from ``_Jn_deriv`` /
  ``_Jp_deriv`` (same edge partials the Newton Jacobian uses).
* :func:`analytic_u_cell` — ``d mean(Jn+Jp) / d cell`` (direct channel).
* :func:`analytic_lamF_c` — ``(dF/dcell)^T lam`` (implicit channel),
  including the ``boundary_bias(cell, vb)`` path (contact ``phi0/phiL``,
  ``neq/peq`` depend on doping/work-function).

Boltzmann statistics only (matches ``banded_jacobian``); non-Boltzmann
callers must fall back to the AD reference path.

Conventions (match ``residual.comp_F`` / ``fields.pot2vec``):
  F = [ct_n0, ct_p0, ct_f0, ddn_1, ddp_1, pois_1, ..., ct_nL, ct_pL, ct_fL]
  vec = [phin_0, phip_0, phi_0, phin_1, ...]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from driftjax.fields import Potentials, PVCell
from driftjax.numerics.analytic_jacobian import _Jn_deriv, _Jp_deriv
from driftjax.numerics.scharfetter_gummel import _Jn_impl as _Jn_vals
from driftjax.numerics.scharfetter_gummel import _Jp_impl as _Jp_vals
from driftjax.science.contacts import boundary_bias


def _bern_Qn(cell: PVCell, pot: Potentials):
    """Stable Bernoulli Q factor for Jn edges (mirrors ``_Jn_deriv``)."""
    psi_n = cell.Chi + jnp.log(cell.Nc) + pot.phi
    psi0, psi1 = psi_n[:-1], psi_n[1:]
    D = psi0 - psi1
    Dn = jnp.where(jnp.abs(D) < 1e-5, 1e-5, D)
    Dt = jnp.clip(D, -1e-5, 1e-5)
    Q = jnp.where(
        jnp.abs(D) < 1e-5,
        jnp.exp(psi0) / (1 + Dt / 2 + Dt**2 / 6),
        jnp.exp(psi0) * Dn / (jnp.exp(Dn) - 1),
    )
    fm = jnp.exp(pot.phi_n[1:]) - jnp.exp(pot.phi_n[:-1])
    return Q, fm


def _bern_Qp(cell: PVCell, pot: Potentials):
    """Stable Bernoulli Q factor for Jp edges (mirrors ``_Jp_deriv``)."""
    psi_p = cell.Chi + cell.Eg - jnp.log(cell.Nv) + pot.phi
    psi0, psi1 = psi_p[:-1], psi_p[1:]
    D = psi0 - psi1
    Dn = jnp.where(jnp.abs(D) < 1e-5, 1e-5, D)
    Dt = jnp.clip(D, -1e-5, 1e-5)
    epsi0 = jnp.exp(-psi0)
    Q = jnp.where(
        jnp.abs(D) < 1e-5,
        epsi0 / (-1 + Dt / 2 - Dt**2 / 6),
        epsi0 * Dn / (jnp.exp(-Dn) - 1),
    )
    fm = jnp.exp(-pot.phi_p[1:]) - jnp.exp(-pot.phi_p[:-1])
    return Q, fm


def analytic_g_x(cell: PVCell, pot: Potentials) -> jax.Array:
    """d mean(Jn+Jp) / d potvec, shape (3n,) — no AD."""
    n = pot.n
    ne = n - 1
    DJn_n0, DJn_n1, DJn_0, DJn_1 = _Jn_deriv(cell, pot)
    DJp_p0, DJp_p1, DJp_0, DJp_1 = _Jp_deriv(cell, pot)
    g_phin = jnp.zeros(n, dtype=jnp.float64)
    g_phin = g_phin.at[:-1].add(DJn_n0 / ne)
    g_phin = g_phin.at[1:].add(DJn_n1 / ne)
    g_phip = jnp.zeros(n, dtype=jnp.float64)
    g_phip = g_phip.at[:-1].add(DJp_p0 / ne)
    g_phip = g_phip.at[1:].add(DJp_p1 / ne)
    g_phi = jnp.zeros(n, dtype=jnp.float64)
    g_phi = g_phi.at[:-1].add((DJn_0 + DJp_0) / ne)
    g_phi = g_phi.at[1:].add((DJn_1 + DJp_1) / ne)
    return jnp.stack([g_phin, g_phip, g_phi], axis=1).reshape(-1)


def _zero_cell(cell: PVCell) -> dict:
    """Zero accumulator dict keyed by PVCell array-field name."""
    z = {}
    for k in (
        "dgrid", "x", "G", "eps", "Chi", "Eg", "Nc", "Nv", "mn", "mp",
        "tn", "tp", "Et", "Br", "Cn", "Cp", "Ndop",
        "Snl", "Snr", "Spl", "Spr", "PhiMl", "PhiMr", "T",
    ):
        v = getattr(cell, k)
        z[k] = jnp.zeros_like(v)
    return z


def _finish_cell(cell: PVCell, z: dict) -> PVCell:
    return PVCell(
        dgrid=z["dgrid"], x=z["x"], G=z["G"], eps=z["eps"], Chi=z["Chi"],
        Eg=z["Eg"], Nc=z["Nc"], Nv=z["Nv"], mn=z["mn"], mp=z["mp"],
        tn=z["tn"], tp=z["tp"], Et=z["Et"], Br=z["Br"], Cn=z["Cn"],
        Cp=z["Cp"], Ndop=z["Ndop"], Snl=z["Snl"], Snr=z["Snr"],
        Spl=z["Spl"], Spr=z["Spr"], PhiMl=z["PhiMl"], PhiMr=z["PhiMr"],
        statistics=cell.statistics, T=z["T"],
    )


def analytic_u_cell(cell: PVCell, pot: Potentials) -> PVCell:
    """d mean(Jn+Jp) / d cell as PVCell — no AD (Boltzmann-safe)."""
    n = pot.n
    ne = n - 1
    Jn = _Jn_vals(cell, pot)
    Jp = _Jp_vals(cell, pot)
    DJn_n0, DJn_n1, DJn_0, DJn_1 = _Jn_deriv(cell, pot)
    DJp_p0, DJp_p1, DJp_0, DJp_1 = _Jp_deriv(cell, pot)
    Qn, fm_n = _bern_Qn(cell, pot)
    Qp, fm_p = _bern_Qp(cell, pot)
    z = _zero_cell(cell)
    # mn / mp (linear coeffs; Q-form exact even at mn=0)
    z["mn"] = z["mn"].at[:-1].add(Qn * fm_n / cell.dgrid / ne)
    z["mp"] = z["mp"].at[:-1].add(Qp * fm_p / cell.dgrid / ne)
    # dgrid
    z["dgrid"] = z["dgrid"] + (-(Jn + Jp) / cell.dgrid / ne)
    # Chi / Eg via psi-chain (dQ/dChi == dQ/dphi)
    z["Chi"] = z["Chi"].at[:-1].add((DJn_0 + DJp_0) / ne)
    z["Chi"] = z["Chi"].at[1:].add((DJn_1 + DJp_1) / ne)
    z["Eg"] = z["Eg"].at[:-1].add(DJp_0 / ne)
    z["Eg"] = z["Eg"].at[1:].add(DJp_1 / ne)
    # Nc / Nv (d psi/d log Nc = 1, d psi_p/d log Nv = -1)
    safe_Nc = jnp.where(cell.Nc == 0, jnp.ones_like(cell.Nc), cell.Nc)
    safe_Nv = jnp.where(cell.Nv == 0, jnp.ones_like(cell.Nv), cell.Nv)
    z["Nc"] = z["Nc"].at[:-1].add(DJn_0 / safe_Nc[:-1] / ne)
    z["Nc"] = z["Nc"].at[1:].add(DJn_1 / safe_Nc[1:] / ne)
    z["Nv"] = z["Nv"].at[:-1].add(-DJp_0 / safe_Nv[:-1] / ne)
    z["Nv"] = z["Nv"].at[1:].add(-DJp_1 / safe_Nv[1:] / ne)
    return _finish_cell(cell, z)


def _bound_phi_derivs(PhiM, Ndop_c, Nc_c, Nv_c):
    """d phi_b / d (PhiM, Ndop, Chi, Eg, Nc, Nv) matching jnp.where AD.

    dChi/dEg depend only on the branch weights, not on the Chi/Eg values,
    so those arguments are not taken.
    """
    w_sch = PhiM > 0
    w_n = (~w_sch) & (Ndop_c > 0)
    w_p = (~w_sch) & (Ndop_c < 0)
    w_i = (~w_sch) & ~(Ndop_c > 0) & ~(Ndop_c < 0)
    w_sch_f = w_sch.astype(jnp.float64)
    w_n_f = w_n.astype(jnp.float64)
    w_p_f = w_p.astype(jnp.float64)
    w_i_f = w_i.astype(jnp.float64)
    safe_N = jnp.where((w_n_f + w_p_f) > 0, Ndop_c, jnp.ones_like(Ndop_c))
    dPhiM = -w_sch_f
    # n-branch: d/dN[+log|N|] = +1/N; p-branch: d/dN[-log|N|] = -1/N
    dNdop = jnp.where((w_n_f + w_p_f) > 0, (w_n_f - w_p_f) / safe_N, jnp.zeros_like(safe_N))
    dChi = -(w_n_f + w_p_f + w_i_f)
    dEg = -(w_p_f + 0.5 * w_i_f)
    safe_Nc = jnp.where(Nc_c == 0, jnp.ones_like(Nc_c), Nc_c)
    safe_Nv = jnp.where(Nv_c == 0, jnp.ones_like(Nv_c), Nv_c)
    dNc = (-w_n_f + 0.5 * w_i_f) / safe_Nc
    dNv = (w_p_f - 0.5 * w_i_f) / safe_Nv
    return dPhiM, dNdop, dChi, dEg, dNc, dNv


def analytic_lamF_c(cell: PVCell, pot: Potentials, vb: jax.Array, lam: jax.Array) -> PVCell:
    """(dF/dcell)^T lam as PVCell — fully analytic, Boltzmann only.

    Includes the ``boundary_bias(cell, vb)`` path (contact ``phi0/phiL``,
    ``neq/peq`` depend on doping/work-function). No grad/vjp/jacrev inside.
    """
    from driftjax.science.carrier_statistics import n as _nfun
    from driftjax.science.carrier_statistics import ni as _nifun
    from driftjax.science.carrier_statistics import p as _pfun

    n = pot.n
    ne = n - 1
    nI = n - 2
    phi = pot.phi
    dgrid = cell.dgrid
    ave = (dgrid[:-1] + dgrid[1:]) / 2

    n_v = _nfun(cell, pot)
    p_v = _pfun(cell, pot)
    ni_v = _nifun(cell)
    Jn = _Jn_vals(cell, pot)
    Jp = _Jp_vals(cell, pot)
    DJn_n0, DJn_n1, DJn_0, DJn_1 = _Jn_deriv(cell, pot)
    DJp_p0, DJp_p1, DJp_0, DJp_1 = _Jp_deriv(cell, pot)
    Qn, fm_n = _bern_Qn(cell, pot)
    Qp, fm_p = _bern_Qp(cell, pot)

    # --- lam split ---
    ln0, lp0, lf0 = lam[0], lam[1], lam[2]
    lam_mid = lam[3:-3].reshape(nI, 3)
    ln, lp, lf = lam_mid[:, 0], lam_mid[:, 1], lam_mid[:, 2]
    lnL, lpL, lfL = lam[-3], lam[-2], lam[-1]

    # --- shared recombination intermediates (interior k=1..n-2) ---
    nc, pc, nic = n_v[1:-1], p_v[1:-1], ni_v[1:-1]
    num = nc * pc - nic**2
    tnc, tpc = cell.tn[1:-1], cell.tp[1:-1]
    Etc = cell.Et[1:-1]
    eEt, eMet = jnp.exp(Etc), jnp.exp(-Etc)
    nR = nic * eEt + nc
    pR = nic * eMet + pc
    den = tpc * nR + tnc * pR
    den2 = den**2
    Brc, Cnc, Cpc = cell.Br[1:-1], cell.Cn[1:-1], cell.Cp[1:-1]
    Cnp = Cnc * nc + Cpc * pc
    # dR/dn, dR/dp, dR/dni
    Rn = (pc * den - num * tpc) / den2 + Brc * pc + (Cnc * num + Cnp * pc)
    Rp = (nc * den - num * tnc) / den2 + Brc * nc + (Cpc * num + Cnp * nc)
    Ri = ((-2 * nic * den - num * (tpc * eEt + tnc * eMet)) / den2
          - 2 * Brc * nic - 2 * Cnp * nic)
    aR = -ln + lp  # dF-weight of R[k]

    # --- Poisson shared ---
    a_eps, b_eps = cell.eps[1:], cell.eps[:-1]
    S = a_eps + b_eps
    safe_S = jnp.where(jnp.abs(S) < 1e-30, jnp.full_like(S, 1e-30), S)
    ef = 2 * a_eps * b_eps / safe_S
    dphi_pot = phi[1:] - phi[:-1]
    flux = ef * dphi_pot / dgrid

    z = _zero_cell(cell)
    G = z["G"].at[1:-1].add(ln - lp)
    z["G"] = G

    # --- R-chain to Br/Cn/Cp/tn/tp/Et ---
    z["Br"] = z["Br"].at[1:-1].add(aR * num)
    z["Cn"] = z["Cn"].at[1:-1].add(aR * nc * num)
    z["Cp"] = z["Cp"].at[1:-1].add(aR * pc * num)
    z["tn"] = z["tn"].at[1:-1].add(aR * (-num * pR / den2))
    z["tp"] = z["tp"].at[1:-1].add(aR * (-num * nR / den2))
    z["Et"] = z["Et"].at[1:-1].add(aR * (-num * (tpc * nic * eEt - tnc * nic * eMet) / den2))

    # --- R-chain + Poisson-charge chain to Nc/Nv/Chi/Eg ---
    Nc, Nv = cell.Nc, cell.Nv
    sNc = jnp.where(Nc == 0, jnp.ones_like(Nc), Nc)
    sNv = jnp.where(Nv == 0, jnp.ones_like(Nv), Nv)
    NcI, NvI = Nc[1:-1], Nv[1:-1]
    sNcI = jnp.where(NcI == 0, jnp.ones_like(NcI), NcI)
    sNvI = jnp.where(NvI == 0, jnp.ones_like(NvI), NvI)
    # R part
    gNc_R = aR * (Rn * nc / sNcI + Ri * nic / (2 * sNcI))
    gNv_R = aR * (Rp * pc / sNvI + Ri * nic / (2 * sNvI))
    gChi_R = aR * (Rn * nc - Rp * pc)
    gEg_R = aR * (-Rp * pc - Ri * nic / 2)
    # Poisson charge part: dpois/dn=+1, dpois/dp=-1
    gNc_P = lf * nc / sNcI
    gNv_P = -lf * pc / sNvI
    gChi_P = lf * (nc + pc)
    gEg_P = lf * pc
    z["Nc"] = z["Nc"].at[1:-1].add(gNc_R + gNc_P)
    z["Nv"] = z["Nv"].at[1:-1].add(gNv_R + gNv_P)
    z["Chi"] = z["Chi"].at[1:-1].add(gChi_R + gChi_P)
    z["Eg"] = z["Eg"].at[1:-1].add(gEg_R + gEg_P)
    # Ndop (Poisson charge only, interior)
    z["Ndop"] = z["Ndop"].at[1:-1].add(-lf)

    # --- Jn/Jp flux weights ---
    WJn = jnp.zeros(ne, dtype=jnp.float64)
    WJn = WJn.at[:-1].add(-ln / ave)
    WJn = WJn.at[1:].add(ln / ave)
    WJp = jnp.zeros(ne, dtype=jnp.float64)
    WJp = WJp.at[:-1].add(-lp / ave)
    WJp = WJp.at[1:].add(lp / ave)
    # ave-denominator terms
    dJn_d = Jn[1:] - Jn[:-1]
    dJp_d = Jp[1:] - Jp[:-1]
    t_ave = (ln * (-dJn_d / ave**2) + lp * (-dJp_d / ave**2)
             + lf * (-((flux[:-1] - flux[1:]) / ave**2)))
    # mn / mp via Q-form
    z["mn"] = z["mn"].at[:-1].add(WJn * Qn * fm_n / dgrid)
    z["mp"] = z["mp"].at[:-1].add(WJp * Qp * fm_p / dgrid)
    # Chi / Eg / Nc / Nv via Jn/Jp psi-chain
    z["Chi"] = z["Chi"].at[:-1].add(WJn * DJn_0 + WJp * DJp_0)
    z["Chi"] = z["Chi"].at[1:].add(WJn * DJn_1 + WJp * DJp_1)
    z["Eg"] = z["Eg"].at[:-1].add(WJp * DJp_0)
    z["Eg"] = z["Eg"].at[1:].add(WJp * DJp_1)
    z["Nc"] = z["Nc"].at[:-1].add(WJn * DJn_0 / sNc[:-1])
    z["Nc"] = z["Nc"].at[1:].add(WJn * DJn_1 / sNc[1:])
    z["Nv"] = z["Nv"].at[:-1].add(WJp * (-DJp_0) / sNv[:-1])
    z["Nv"] = z["Nv"].at[1:].add(WJp * (-DJp_1) / sNv[1:])
    # dgrid: Jn/Jp denominators + ave + flux denominators
    z["dgrid"] = z["dgrid"] + (WJn * (-Jn / dgrid) + WJp * (-Jp / dgrid))
    z["dgrid"] = z["dgrid"].at[:-1].add(t_ave / 2)
    z["dgrid"] = z["dgrid"].at[1:].add(t_ave / 2)

    # --- Poisson flux eps-chain ---
    Wflx = jnp.zeros(ne, dtype=jnp.float64)
    # pois[i] = (flux[i]-flux[i+1])/ave[i]: edge j in pois[j] (+) and pois[j-1] (-)
    Wflx = Wflx.at[:-1].add(lf / ave)
    Wflx = Wflx.at[1:].add(-lf / ave)
    coef = Wflx * dphi_pot / dgrid
    S2 = safe_S**2
    A_b = coef * 2 * a_eps**2 / S2  # -> node j
    B_a = coef * 2 * b_eps**2 / S2  # -> node j+1
    z["eps"] = z["eps"].at[:-1].add(A_b)
    z["eps"] = z["eps"].at[1:].add(B_a)
    z["dgrid"] = z["dgrid"] + Wflx * (-flux / dgrid)

    # --- contacts (with bound path) ---
    bound = boundary_bias(cell, vb)
    n0, nL = n_v[0], n_v[-1]
    p0, pL = p_v[0], p_v[-1]
    # left phi0b derivs
    dPm0, dN0, dC0, dE0, dNc0, dNv0 = _bound_phi_derivs(
        cell.PhiMl, cell.Ndop[0], cell.Nc[0], cell.Nv[0])
    dPmL, dNL, dCL, dEL, dNcL, dNvL = _bound_phi_derivs(
        cell.PhiMr, cell.Ndop[-1], cell.Nc[-1], cell.Nv[-1])
    neq0, neqL = bound.neq0, bound.neqL
    peq0, peqL = bound.peq0, bound.peqL
    # neq/peq derivs
    dneq0 = {"Nc": neq0 / sNc[0] + neq0 * dNc0, "Chi": neq0 * (1 + dC0),
             "Eg": neq0 * dE0, "Nv": neq0 * dNv0,
             "Ndop": neq0 * dN0, "PhiM": neq0 * dPm0}
    dneqL = {"Nc": neqL / sNc[-1] + neqL * dNcL, "Chi": neqL * (1 + dCL),
             "Eg": neqL * dEL, "Nv": neqL * dNvL,
             "Ndop": neqL * dNL, "PhiM": neqL * dPmL}
    dpeq0 = {"Nv": peq0 / sNv[0] - peq0 * dNv0, "Chi": peq0 * (-1 - dC0),
             "Eg": peq0 * (-1 - dE0), "Nc": peq0 * (-dNc0),
             "Ndop": peq0 * (-dN0), "PhiM": peq0 * (-dPm0)}
    dpeqL = {"Nv": peqL / sNv[-1] - peqL * dNvL, "Chi": peqL * (-1 - dCL),
             "Eg": peqL * (-1 - dEL), "Nc": peqL * (-dNcL),
             "Ndop": peqL * (-dNL), "PhiM": peqL * (-dPmL)}

    # ct_phin0 = Jn0 - Snl*(n0-neq0), weight ln0
    z["Snl"] = z["Snl"] + ln0 * (-(n0 - neq0))
    z["mn"] = z["mn"].at[0].add(ln0 * Qn[0] * fm_n[0] / dgrid[0])
    z["Chi"] = z["Chi"].at[0].add(ln0 * DJn_0[0] + ln0 * (-cell.Snl) * n0 + ln0 * cell.Snl * dneq0["Chi"])
    z["Chi"] = z["Chi"].at[1].add(ln0 * DJn_1[0])
    z["Nc"] = z["Nc"].at[0].add(ln0 * DJn_0[0] / sNc[0] + ln0 * (-cell.Snl) * n0 / sNc[0] + ln0 * cell.Snl * dneq0["Nc"])
    z["Nc"] = z["Nc"].at[1].add(ln0 * DJn_1[0] / sNc[1])
    z["dgrid"] = z["dgrid"].at[0].add(ln0 * (-Jn[0] / dgrid[0]))
    z["Eg"] = z["Eg"].at[0].add(ln0 * cell.Snl * dneq0["Eg"])
    z["Nv"] = z["Nv"].at[0].add(ln0 * cell.Snl * dneq0["Nv"])
    z["Ndop"] = z["Ndop"].at[0].add(ln0 * cell.Snl * dneq0["Ndop"])
    z["PhiMl"] = z["PhiMl"] + ln0 * cell.Snl * dneq0["PhiM"]
    # ct_phinL = Jn[-1] + Snr*(nL-neqL), weight lnL
    z["Snr"] = z["Snr"] + lnL * (nL - neqL)
    z["mn"] = z["mn"].at[-1].add(lnL * Qn[-1] * fm_n[-1] / dgrid[-1])
    z["Chi"] = z["Chi"].at[-2].add(lnL * DJn_0[-1])
    z["Chi"] = z["Chi"].at[-1].add(lnL * DJn_1[-1] + lnL * cell.Snr * nL - lnL * cell.Snr * dneqL["Chi"])
    z["Nc"] = z["Nc"].at[-2].add(lnL * DJn_0[-1] / sNc[-2])
    z["Nc"] = z["Nc"].at[-1].add(lnL * DJn_1[-1] / sNc[-1] + lnL * cell.Snr * nL / sNc[-1] - lnL * cell.Snr * dneqL["Nc"])
    z["dgrid"] = z["dgrid"].at[-1].add(lnL * (-Jn[-1] / dgrid[-1]))
    z["Eg"] = z["Eg"].at[-1].add(-lnL * cell.Snr * dneqL["Eg"])
    z["Nv"] = z["Nv"].at[-1].add(-lnL * cell.Snr * dneqL["Nv"])
    z["Ndop"] = z["Ndop"].at[-1].add(-lnL * cell.Snr * dneqL["Ndop"])
    z["PhiMr"] = z["PhiMr"] + (-lnL * cell.Snr * dneqL["PhiM"])
    # ct_phip0 = Jp0 + Spl*(p0-peq0), weight lp0
    z["Spl"] = z["Spl"] + lp0 * (p0 - peq0)
    z["mp"] = z["mp"].at[0].add(lp0 * Qp[0] * fm_p[0] / dgrid[0])
    z["Chi"] = z["Chi"].at[0].add(lp0 * DJp_0[0] + lp0 * cell.Spl * (-p0) - lp0 * cell.Spl * dpeq0["Chi"])
    z["Chi"] = z["Chi"].at[1].add(lp0 * DJp_1[0])
    z["Eg"] = z["Eg"].at[0].add(lp0 * DJp_0[0] + lp0 * cell.Spl * (-p0) - lp0 * cell.Spl * dpeq0["Eg"])
    z["Eg"] = z["Eg"].at[1].add(lp0 * DJp_1[0])
    z["Nv"] = z["Nv"].at[0].add(lp0 * (-DJp_0[0]) / sNv[0] + lp0 * cell.Spl * p0 / sNv[0] - lp0 * cell.Spl * dpeq0["Nv"])
    z["Nv"] = z["Nv"].at[1].add(lp0 * (-DJp_1[0]) / sNv[1])
    z["Nc"] = z["Nc"].at[0].add(-lp0 * cell.Spl * dpeq0["Nc"])
    z["Ndop"] = z["Ndop"].at[0].add(-lp0 * cell.Spl * dpeq0["Ndop"])
    z["PhiMl"] = z["PhiMl"] + (-lp0 * cell.Spl * dpeq0["PhiM"])
    z["dgrid"] = z["dgrid"].at[0].add(lp0 * (-Jp[0] / dgrid[0]))
    # ct_phipL = Jp[-1] - Spr*(pL-peqL), weight lpL
    z["Spr"] = z["Spr"] + lpL * (-(pL - peqL))
    z["mp"] = z["mp"].at[-1].add(lpL * Qp[-1] * fm_p[-1] / dgrid[-1])
    z["Chi"] = z["Chi"].at[-2].add(lpL * DJp_0[-1])
    z["Chi"] = z["Chi"].at[-1].add(lpL * DJp_1[-1] + lpL * (-cell.Spr) * (-pL) + lpL * cell.Spr * dpeqL["Chi"])
    z["Eg"] = z["Eg"].at[-2].add(lpL * DJp_0[-1])
    z["Eg"] = z["Eg"].at[-1].add(lpL * DJp_1[-1] + lpL * (-cell.Spr) * (-pL) + lpL * cell.Spr * dpeqL["Eg"])
    z["Nv"] = z["Nv"].at[-2].add(lpL * (-DJp_0[-1]) / sNv[-2])
    z["Nv"] = z["Nv"].at[-1].add(lpL * (-DJp_1[-1]) / sNv[-1] + lpL * (-cell.Spr) * pL / sNv[-1] + lpL * cell.Spr * dpeqL["Nv"])
    z["Nc"] = z["Nc"].at[-1].add(lpL * cell.Spr * dpeqL["Nc"])
    z["Ndop"] = z["Ndop"].at[-1].add(lpL * cell.Spr * dpeqL["Ndop"])
    z["PhiMr"] = z["PhiMr"] + lpL * cell.Spr * dpeqL["PhiM"]
    z["dgrid"] = z["dgrid"].at[-1].add(lpL * (-Jp[-1] / dgrid[-1]))
    # ct_phi0 = phi[0]-phi0b, weight lf0; ct_phiL, weight lfL
    z["PhiMl"] = z["PhiMl"] + lf0 * (-dPm0)
    z["Ndop"] = z["Ndop"].at[0].add(lf0 * (-dN0))
    z["Chi"] = z["Chi"].at[0].add(lf0 * (-dC0))
    z["Eg"] = z["Eg"].at[0].add(lf0 * (-dE0))
    z["Nc"] = z["Nc"].at[0].add(lf0 * (-dNc0))
    z["Nv"] = z["Nv"].at[0].add(lf0 * (-dNv0))
    z["PhiMr"] = z["PhiMr"] + lfL * (-dPmL)
    z["Ndop"] = z["Ndop"].at[-1].add(lfL * (-dNL))
    z["Chi"] = z["Chi"].at[-1].add(lfL * (-dCL))
    z["Eg"] = z["Eg"].at[-1].add(lfL * (-dEL))
    z["Nc"] = z["Nc"].at[-1].add(lfL * (-dNcL))
    z["Nv"] = z["Nv"].at[-1].add(lfL * (-dNvL))
    return _finish_cell(cell, z)


def analytic_per_bias(
    cell: PVCell, pot: Potentials, vb: jax.Array, lam: jax.Array, gcb: jax.Array
) -> PVCell:
    """Single-kernel per-bias cell cotangent: (u_cell - lamF_c) * gcb.

    No grad/vjp/jacrev inside (Boltzmann only). Replaces the
    ``u_cell = grad(total_current)`` + ``vjp(comp_F)`` pair.
    """
    u = analytic_u_cell(cell, pot)
    w = analytic_lamF_c(cell, pot, vb, lam)
    return jax.tree.map(lambda a, b: (a - b) * gcb, u, w)
