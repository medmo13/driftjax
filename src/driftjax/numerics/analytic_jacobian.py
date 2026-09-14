"""Analytic block-tridiagonal Jacobian for the coupled DDP system (fast backend).

The interleaved residual F = [φn₀, φp₀, φ₀, …, φn_{n−1}, φp_{n−1}, φ_{n−1}]
is locally coupled: node i's equations touch only nodes i−1, i, i+1, so the
Jacobian is block-tridiagonal in 3×3 blocks.

Correctness contract (package lesson F-01: hand-coded derivatives drifted in
the original codebase): the canonical reference Jacobian remains
jax.jacfwd(comp_F) (``numerics.residual.F_jacobian``); this module is a
pinned fast backend validated against it by the unit suite —

  * tests/unit/test_analytic_jacobian.py asserts
    dense_from_blocks(banded_jacobian(...)) matches jacfwd(comp_F)
    to 1e-9 on p-n, heterojunction, and biased states.

(An earlier revision described a per-solve first-iteration jacrev tripwire
with dense fallback; that runtime check no longer exists — the pinning is
done by the unit tests above, not inside the Newton loop. An earlier
revision also used reverse-mode for the reference; forward-mode gives
identical values ~2x faster to trace.)

Block conventions (match numerics.block_thomas):
  A[i] = J[row i, col i]        (diagonal)
  B[i] = J[row i, col i+1]      (super-diagonal, B[n−1] unused)
  C[i] = J[row i+1, col i]      (sub-diagonal, C[n−2] last used)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from driftjax.fields import BoundaryConditions, Potentials, PVCell
from driftjax.numerics.poisson import harmonic_ave_eps
from driftjax.science.carrier_statistics import n as _n
from driftjax.science.carrier_statistics import ni as _ni
from driftjax.science.carrier_statistics import p as _p


def _banded_jacobian_fused(cell: PVCell, pot: Potentials):
    """Fused computation of all Jn and Jp derivatives in a single vectorized kernel.

    Computes all 8 edge derivatives (4 for Jn, 4 for Jp) simultaneously,
    avoiding redundant exponential computations. Returns the same 8 values
    as calling _Jn_deriv and _Jp_deriv separately.
    """
    phi_n0, phi_n1 = pot.phi_n[:-1], pot.phi_n[1:]
    phi_p0, phi_p1 = pot.phi_p[:-1], pot.phi_p[1:]
    mn0 = cell.mn[:-1]
    mp0 = cell.mp[:-1]
    dgl = cell.dgrid

    # --- Precompute all exponentials once ---
    exp_phi_n0 = jnp.exp(phi_n0)
    exp_phi_n1 = jnp.exp(phi_n1)
    exp_phi_p0 = jnp.exp(-phi_p0)
    exp_phi_p1 = jnp.exp(-phi_p1)

    # --- Jn derivatives ---
    fm_n = exp_phi_n1 - exp_phi_n0
    psi_n = cell.Chi + jnp.log(cell.Nc) + pot.phi
    psi_n0, psi_n1 = psi_n[:-1], psi_n[1:]
    Dpsin = psi_n0 - psi_n1
    Dpsin_norm = jnp.where(jnp.abs(Dpsin) < 1e-5, 1e-5, Dpsin)
    Dpsin_taylor = jnp.clip(Dpsin, -1e-5, 1e-5)
    expDpsin = jnp.exp(Dpsin)
    exppsi_n0 = jnp.exp(psi_n0)

    Q_n = jnp.where(
        jnp.abs(Dpsin) < 1e-5,
        exppsi_n0 / (1 + Dpsin_taylor / 2 + Dpsin_taylor**2 / 6),
        exppsi_n0 * Dpsin_norm / (jnp.exp(Dpsin_norm) - 1),
    )

    DQDphi0_norm = exppsi_n0 / (expDpsin - 1) * (Dpsin + 1 - Dpsin * expDpsin / (expDpsin - 1))
    DQDphi1_norm = exppsi_n0 / (expDpsin - 1) * (-1 + Dpsin * expDpsin / (expDpsin - 1))
    DQDphi0_taylor = (
        6
        * exppsi_n0
        * (3 + psi_n0 + psi_n0**2 - psi_n1 - 2 * psi_n0 * psi_n1 + psi_n1**2)
        / (6 + 3 * psi_n0 + psi_n0**2 - 3 * psi_n1 - 2 * psi_n0 * psi_n1 + psi_n1**2) ** 2
    )
    DQDphi1_taylor = -exppsi_n0 * (-1 / 2 - Dpsin / 3) / (1 + Dpsin / 2 + Dpsin**2 / 6) ** 2

    DfmDphi_n0 = -exp_phi_n0
    DfmDphi_n1 = exp_phi_n1
    DJnDphi0 = mn0 * fm_n / dgl * jnp.where(jnp.abs(Dpsin) < 1e-5, DQDphi0_taylor, DQDphi0_norm)
    DJnDphi1 = mn0 * fm_n / dgl * jnp.where(jnp.abs(Dpsin) < 1e-5, DQDphi1_taylor, DQDphi1_norm)
    DJnDphi_n0 = mn0 * Q_n / dgl * DfmDphi_n0
    DJnDphi_n1 = mn0 * Q_n / dgl * DfmDphi_n1

    # --- Jp derivatives ---
    fm_p = exp_phi_p1 - exp_phi_p0
    psi_p = cell.Chi + cell.Eg - jnp.log(cell.Nv) + pot.phi
    psi_p0, psi_p1 = psi_p[:-1], psi_p[1:]
    Dpsip = psi_p0 - psi_p1
    Dpsip_norm = jnp.where(jnp.abs(Dpsip) < 1e-5, 1e-5, Dpsip)
    Dpsip_taylor = jnp.clip(Dpsip, -1e-5, 1e-5)
    expmpsi_p0 = jnp.exp(-psi_p0)

    Q_p = jnp.where(
        jnp.abs(Dpsip) < 1e-5,
        expmpsi_p0 / (-1 + Dpsip_taylor / 2 - Dpsip_taylor**2 / 6),
        expmpsi_p0 * Dpsip_norm / (jnp.exp(-Dpsip_norm) - 1),
    )

    DQDphi0_norm_p = (jnp.exp(psi_p1) - jnp.exp(psi_p0) * (1 + psi_p1 - psi_p0)) / (
        jnp.exp(psi_p0) - jnp.exp(psi_p1)
    ) ** 2
    DQDphi1_norm_p = (jnp.exp(psi_p0) - jnp.exp(psi_p1) * (1 + psi_p0 - psi_p1)) / (
        jnp.exp(psi_p0) - jnp.exp(psi_p1)
    ) ** 2
    DQDphi0_taylor_p = (
        -expmpsi_p0 / (-1 + Dpsip / 2 - Dpsip**2 / 6)
        - expmpsi_p0 * (1 / 2 - Dpsip / 3) / (-1 + Dpsip / 2 - Dpsip**2 / 6) ** 2
    )
    DQDphi1_taylor_p = -expmpsi_p0 * (-1 / 2 + Dpsip / 3) / (-1 + Dpsip / 2 - Dpsip**2 / 6) ** 2

    DfmDphi_p0 = exp_phi_p0
    DfmDphi_p1 = -exp_phi_p1
    DJpDphi0 = mp0 * fm_p / dgl * jnp.where(jnp.abs(Dpsip) < 1e-5, DQDphi0_taylor_p, DQDphi0_norm_p)
    DJpDphi1 = mp0 * fm_p / dgl * jnp.where(jnp.abs(Dpsip) < 1e-5, DQDphi1_taylor_p, DQDphi1_norm_p)
    DJpDphi_p0 = mp0 * Q_p / dgl * DfmDphi_p0
    DJpDphi_p1 = mp0 * Q_p / dgl * DfmDphi_p1

    return DJnDphi_n0, DJnDphi_n1, DJnDphi0, DJnDphi1, DJpDphi_p0, DJpDphi_p1, DJpDphi0, DJpDphi1


def _Jn_deriv(cell: PVCell, pot: Potentials):
    """Per-edge partials of Jn: (∂φn_j, ∂φn_{j+1}, ∂φ_j, ∂φ_{j+1})."""
    phi_n0, phi_n1 = pot.phi_n[:-1], pot.phi_n[1:]
    fm = jnp.exp(phi_n1) - jnp.exp(phi_n0)
    mn0 = cell.mn[:-1]
    psi_n = cell.Chi + jnp.log(cell.Nc) + pot.phi
    psi_n0, psi_n1 = psi_n[:-1], psi_n[1:]
    Dpsin = psi_n0 - psi_n1
    Dpsin_norm = jnp.where(jnp.abs(Dpsin) < 1e-5, 1e-5, Dpsin)
    Dpsin_taylor = jnp.clip(Dpsin, -1e-5, 1e-5)
    expDpsin = jnp.exp(Dpsin)
    exppsi_n0 = jnp.exp(psi_n0)

    Q = jnp.where(
        jnp.abs(Dpsin) < 1e-5,
        jnp.exp(psi_n0) / (1 + Dpsin_taylor / 2 + Dpsin_taylor**2 / 6),
        jnp.exp(psi_n0) * Dpsin_norm / (jnp.exp(Dpsin_norm) - 1),
    )

    DQDphi0_norm = exppsi_n0 / (expDpsin - 1) * (Dpsin + 1 - Dpsin * expDpsin / (expDpsin - 1))
    DQDphi1_norm = exppsi_n0 / (expDpsin - 1) * (-1 + Dpsin * expDpsin / (expDpsin - 1))
    DQDphi0_taylor = (
        6
        * exppsi_n0
        * (3 + psi_n0 + psi_n0**2 - psi_n1 - 2 * psi_n0 * psi_n1 + psi_n1**2)
        / (6 + 3 * psi_n0 + psi_n0**2 - 3 * psi_n1 - 2 * psi_n0 * psi_n1 + psi_n1**2) ** 2
    )
    DQDphi1_taylor = -exppsi_n0 * (-1 / 2 - Dpsin / 3) / (1 + Dpsin / 2 + Dpsin**2 / 6) ** 2

    DfmDphi_n0 = -jnp.exp(phi_n0)
    DfmDphi_n1 = jnp.exp(phi_n1)
    DJnDphi0 = (
        mn0 * fm / cell.dgrid * jnp.where(jnp.abs(Dpsin) < 1e-5, DQDphi0_taylor, DQDphi0_norm)
    )
    DJnDphi1 = (
        mn0 * fm / cell.dgrid * jnp.where(jnp.abs(Dpsin) < 1e-5, DQDphi1_taylor, DQDphi1_norm)
    )
    DJnDphi_n0 = mn0 * Q / cell.dgrid * DfmDphi_n0
    DJnDphi_n1 = mn0 * Q / cell.dgrid * DfmDphi_n1
    return DJnDphi_n0, DJnDphi_n1, DJnDphi0, DJnDphi1


def _Jp_deriv(cell: PVCell, pot: Potentials):
    """Per-edge partials of Jp: (∂φp_j, ∂φp_{j+1}, ∂φ_j, ∂φ_{j+1})."""
    phi_p0, phi_p1 = pot.phi_p[:-1], pot.phi_p[1:]
    fm = jnp.exp(-phi_p1) - jnp.exp(-phi_p0)
    mp0 = cell.mp[:-1]
    psi_p = cell.Chi + cell.Eg - jnp.log(cell.Nv) + pot.phi
    psi_p0, psi_p1 = psi_p[:-1], psi_p[1:]
    Dpsip = psi_p0 - psi_p1
    Dpsip_norm = jnp.where(jnp.abs(Dpsip) < 1e-5, 1e-5, Dpsip)
    Dpsip_taylor = jnp.clip(Dpsip, -1e-5, 1e-5)
    expmpsi_p0 = jnp.exp(-psi_p0)

    Q = jnp.where(
        jnp.abs(Dpsip) < 1e-5,
        expmpsi_p0 / (-1 + Dpsip_taylor / 2 - Dpsip_taylor**2 / 6),
        expmpsi_p0 * Dpsip_norm / (jnp.exp(-Dpsip_norm) - 1),
    )

    DQDphi0_norm = (jnp.exp(psi_p1) - jnp.exp(psi_p0) * (1 + psi_p1 - psi_p0)) / (
        jnp.exp(psi_p0) - jnp.exp(psi_p1)
    ) ** 2
    DQDphi1_norm = (jnp.exp(psi_p0) - jnp.exp(psi_p1) * (1 + psi_p0 - psi_p1)) / (
        jnp.exp(psi_p0) - jnp.exp(psi_p1)
    ) ** 2
    DQDphi0_taylor = (
        -expmpsi_p0 / (-1 + Dpsip / 2 - Dpsip**2 / 6)
        - expmpsi_p0 * (1 / 2 - Dpsip / 3) / (-1 + Dpsip / 2 - Dpsip**2 / 6) ** 2
    )
    DQDphi1_taylor = -expmpsi_p0 * (-1 / 2 + Dpsip / 3) / (-1 + Dpsip / 2 - Dpsip**2 / 6) ** 2

    DfmDphi_p0 = jnp.exp(-phi_p0)
    DfmDphi_p1 = -jnp.exp(-phi_p1)
    DJpDphi0 = (
        mp0 * fm / cell.dgrid * jnp.where(jnp.abs(Dpsip) < 1e-5, DQDphi0_taylor, DQDphi0_norm)
    )
    DJpDphi1 = (
        mp0 * fm / cell.dgrid * jnp.where(jnp.abs(Dpsip) < 1e-5, DQDphi1_taylor, DQDphi1_norm)
    )
    DJpDphi_p0 = mp0 * Q / cell.dgrid * DfmDphi_p0
    DJpDphi_p1 = mp0 * Q / cell.dgrid * DfmDphi_p1
    return DJpDphi_p0, DJpDphi_p1, DJpDphi0, DJpDphi1


def _recomb_deriv(cell: PVCell, pot: Potentials, n_v, p_v, ni_v=None):
    """∂R/∂φn, ∂R/∂φp, ∂R/∂φ at interior nodes (n−2,) — Boltzmann algebra.

    R = R_srh + R_rad + R_aug, each ∝ (np − ni²); ∂n/∂φn = n,
    ∂p/∂φp = −p, ∂n/∂φ = n, ∂p/∂φ = −p.
    """
    if ni_v is None:
        ni_v = _ni(cell)
    num = n_v * p_v - ni_v**2
    tn, tp = cell.tn, cell.tp
    nR = ni_v * jnp.exp(cell.Et) + n_v
    pR = ni_v * jnp.exp(-cell.Et) + p_v
    den = tp * nR + tn * pR
    srh_phin = (n_v * p_v * den - num * tp * n_v) / den**2
    # dR/dphip = [(-np)*den - num*(-tn*p)]/den^2  (quotient rule; dp/dphip = -p)
    #           = (-np*den + num*tn*p)/den^2
    # The `num*tn*p` term must carry a PLUS sign: in quasi-neutral regions
    # (num ~ np, den ~ tn*p) it cancels the leading term so that dR/dphip -> 0
    # (there R ~ n/tn is independent of phip). The previous minus sign made
    # the analytic Newton Jacobian wrong by 2*num*tn*p/den^2 (relative O(1)
    # in the SRH block, absolute ~1e-13 — below the old pinning tolerance).
    srh_phip = (-n_v * p_v * den + num * tn * p_v) / den**2
    srh_phi = -num * (tp * n_v - tn * p_v) / den**2
    Br = cell.Br
    rad_phin = Br * n_v * p_v
    rad_phip = -Br * n_v * p_v
    rad_phi = jnp.zeros_like(n_v)
    Cnp = cell.Cn * n_v + cell.Cp * p_v
    aug_phin = cell.Cn * n_v * num + Cnp * n_v * p_v
    aug_phip = -cell.Cp * p_v * num - Cnp * n_v * p_v
    aug_phi = (cell.Cn * n_v - cell.Cp * p_v) * num
    DR_phin = srh_phin + rad_phin + aug_phin
    DR_phip = srh_phip + rad_phip + aug_phip
    DR_phi = srh_phi + rad_phi + aug_phi
    return DR_phin[1:-1], DR_phip[1:-1], DR_phi[1:-1]


def banded_jacobian(cell: PVCell, bound: BoundaryConditions, pot: Potentials,
                    n_v=None, p_v=None, ni_v=None) -> tuple:
    """(A, B, C) 3×3 blocks of the DDP Jacobian (Boltzmann statistics).

    Fully vectorised (no per-node Python loop): compiles to compact XLA.
    Optional n_v/p_v/ni_v: pre-computed carrier stats to avoid redundant eval.
    """
    if getattr(cell, "statistics", "boltzmann") != "boltzmann":
        raise ValueError(
            f"banded_jacobian is Boltzmann-only; got statistics={getattr(cell, 'statistics', None)!r}. "
            "Use F_jacobian (dense jacfwd) for fermi-dirac/exact."
        )
    n = pot.n
    if n < 3:
        raise ValueError("banded_jacobian needs n ≥ 3 nodes")
    dgl = cell.dgrid
    eps_face = harmonic_ave_eps(cell)
    ave = (dgl[:-1] + dgl[1:]) / 2  # (n−2,) interior means
    oa = 1.0 / ave
    m_n, u_n, m_phi, u_phi, m_p, u_p, mp_phi, up_phi = _banded_jacobian_fused(cell, pot)
    if n_v is None:
        n_v = _n(cell, pot)
    if p_v is None:
        p_v = _p(cell, pot)
    DR_phin, DR_phip, DR_phi = _recomb_deriv(cell, pot, n_v, p_v, ni_v)

    Snl, Snr, Spl, Spr = cell.Snl, cell.Snr, cell.Spl, cell.Spr
    n0, nL, p0, pL = n_v[0], n_v[-1], p_v[0], p_v[-1]

    # ---- interior blocks (k = 1..n−2 ⇔ array slot i = k−1) ----
    # sub-diagonal coupling to node k−1
    An = jnp.stack([-m_n[:-1] * oa, jnp.zeros_like(oa), -m_phi[:-1] * oa], axis=1)
    Ap = jnp.stack([jnp.zeros_like(oa), -m_p[:-1] * oa, -mp_phi[:-1] * oa], axis=1)
    Aphi = jnp.stack(
        [jnp.zeros_like(oa), jnp.zeros_like(oa), -eps_face[:-1] / dgl[:-1] * oa], axis=1
    )
    # diagonal
    Bn = jnp.stack(
        [(m_n[1:] - u_n[:-1]) * oa - DR_phin, -DR_phip, (m_phi[1:] - u_phi[:-1]) * oa - DR_phi],
        axis=1,
    )
    Bp = jnp.stack(
        [DR_phin, (m_p[1:] - u_p[:-1]) * oa + DR_phip, (mp_phi[1:] - up_phi[:-1]) * oa + DR_phi],
        axis=1,
    )
    Bphi = jnp.stack(
        [
            n_v[1:-1],
            p_v[1:-1],
            (eps_face[:-1] / dgl[:-1] + eps_face[1:] / dgl[1:]) * oa + n_v[1:-1] + p_v[1:-1],
        ],
        axis=1,
    )
    # super-diagonal coupling to node k+1
    Cn = jnp.stack([u_n[1:] * oa, jnp.zeros_like(oa), u_phi[1:] * oa], axis=1)
    Cp = jnp.stack([jnp.zeros_like(oa), u_p[1:] * oa, up_phi[1:] * oa], axis=1)
    Cphi = jnp.stack([jnp.zeros_like(oa), jnp.zeros_like(oa), -eps_face[1:] / dgl[1:] * oa], axis=1)

    A_int = jnp.stack([Bn, Bp, Bphi], axis=1)  # (n−2, 3, 3)
    C_int = jnp.stack([An, Ap, Aphi], axis=1)  # (n−2, 3, 3) rows k
    B_int = jnp.stack([Cn, Cp, Cphi], axis=1)  # (n−2, 3, 3)

    # ---- contact blocks ----
    A0 = jnp.array(
        [
            [m_n[0] - Snl * n0, 0.0, m_phi[0] - Snl * n0],
            [0.0, m_p[0] - Spl * p0, mp_phi[0] - Spl * p0],
            [0.0, 0.0, 1.0],
        ]
    )
    B0 = jnp.array([[u_n[0], 0.0, u_phi[0]], [0.0, u_p[0], up_phi[0]], [0.0, 0.0, 0.0]])
    AL = jnp.array(
        [
            [u_n[-1] + Snr * nL, 0.0, u_phi[-1] + Snr * nL],
            [0.0, u_p[-1] + Spr * pL, up_phi[-1] + Spr * pL],
            [0.0, 0.0, 1.0],
        ]
    )
    CR = jnp.array([[m_n[-1], 0.0, m_phi[-1]], [0.0, m_p[-1], mp_phi[-1]], [0.0, 0.0, 0.0]])

    A = jnp.concatenate([A0[None], A_int, AL[None]], axis=0)  # (n, 3, 3)
    B = jnp.concatenate([B0[None], B_int], axis=0)  # (n−1, 3, 3)
    C = jnp.concatenate([C_int, CR[None]], axis=0)  # (n−1, 3, 3)
    return A, B, C


def dense_from_blocks(A, B, C):
    """Expand (A, B, C) into the dense (3n, 3n) Jacobian (testing/fallback).

    Fully vectorised — no Python loop over nodes.
    """
    n = A.shape[0]
    ri = jnp.arange(3)
    rk = jnp.arange(n)
    J = jnp.zeros((3 * n, 3 * n), dtype=A.dtype)
    # Diagonal: A[k,i,j] → J[3k+i, 3k+j]
    k9 = jnp.repeat(rk, 9)
    i9 = jnp.tile(jnp.repeat(ri, 3), n)
    j9 = jnp.tile(ri, 3 * n)
    J = J.at[3 * k9 + i9, 3 * k9 + j9].add(A.reshape(-1))
    # Super-diagonal: B[k,i,j] → J[3k+i, 3(k+1)+j]  for k in 0..n-2
    if n > 1:
        k9s = jnp.repeat(rk[:-1], 9)
        i9s = jnp.tile(jnp.repeat(ri, 3), n - 1)
        j9s = jnp.tile(ri, 3 * (n - 1))
        J = J.at[3 * k9s + i9s, 3 * k9s + 3 + j9s].add(B.reshape(-1))
    # Sub-diagonal: C[k,i,j] → J[3(k+1)+i, 3k+j]  for k in 0..n-2
    if n > 1:
        J = J.at[3 * k9s + 3 + i9s, 3 * k9s + j9s].add(C.reshape(-1))
    return J


def blockwise_matvec(A, B, C, x):
    """Compute J @ x using block-tridiagonal structure (O(N), no dense matrix).

    A: (n, 3, 3) diagonal blocks
    B: (n-1, 3, 3) super-diagonal blocks (B[k] couples row k to row k+1)
    C: (n-1, 3, 3) sub-diagonal blocks (C[k] couples row k+1 to row k)
    """
    n = A.shape[0]
    x3 = x.reshape(n, 3)
    out = jnp.einsum("kij,kj->ki", A, x3)
    out = out.at[:-1].add(jnp.einsum("kij,kj->ki", B, x3[1:]))
    out = out.at[1:].add(jnp.einsum("kij,kj->ki", C, x3[:-1]))
    return out.reshape(-1)


def blockwise_matvec_transpose(A, B, C, x):
    """Compute J^T @ x using block-tridiagonal structure (O(N), no dense matrix).

    J^T has blocks: diagonal=A^T, super=C^T, sub=B^T.
    """
    n = A.shape[0]
    x3 = x.reshape(n, 3)
    out = jnp.einsum("kij,kj->ki", A.transpose(0, 2, 1), x3)
    out = out.at[:-1].add(jnp.einsum("kij,kj->ki", C.transpose(0, 2, 1), x3[1:]))
    out = out.at[1:].add(jnp.einsum("kij,kj->ki", B.transpose(0, 2, 1), x3[:-1]))
    return out.reshape(-1)


def blockwise_residual(A, B, C, F, p):
    """Compute J @ p + F in O(N) without forming the dense Jacobian."""
    return blockwise_matvec(A, B, C, p) + F



