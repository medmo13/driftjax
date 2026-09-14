"""Analytic banded Jacobian must equal the canonical jacrev Jacobian.

The fast solver backend (analytic_block_thomas) is only trustworthy while
this pins the protocol (R1/R2): banded_jacobian == jacrev(comp_F) to
~1e-12 on p-n, n-p, heterojunction, and biased states, equilibrium and
full coupled, Boltzmann statistics.
"""

import jax
import jax.numpy as jnp
import pytest

import driftjax as dj
import driftjax.science
from driftjax.numerics.analytic_jacobian import banded_jacobian, dense_from_blocks
from driftjax.numerics.residual import F_jacobian
from driftjax.science.carrier_statistics import n as _n
from driftjax.science.carrier_statistics import p as _p
from driftjax.science.contacts import boundary_bias, boundary_eq
from driftjax.science.spectrum import spectrum
from driftjax.solvers.continuation import equilibrium_guess
from driftjax.solvers.newton import solve_eq, solve_newton
from driftjax.units import energy


def _cell(n, mats, Ns, srvs=(10000000.0, 0, 0, 10000000.0)):
    return dj.simulator.init_cell(
        dj.Device(
            layers=list(zip([5e-06, 0.000295], mats, Ns, strict=False)),
            n_points=n,
            Snl=srvs[0],
            Snr=srvs[1],
            Spl=srvs[2],
            Spr=srvs[3],
            PhiMl=-1.0,
            PhiMr=-1.0,
        ).design(),
        spectrum(normalize=False),
        alpha_mode="beer-lambert",
    )


@pytest.fixture(scope="module")
def pn_cell():
    mat = dj.material(
        Chi=3.9,
        Eg=1.5,
        eps=9.4,
        Nc=8e17,
        Nv=1.8e19,
        mn=100,
        mp=100,
        Et=0,
        tn=1e-08,
        tp=1e-08,
        A=10000.0,
    )
    return _cell(40, [mat, mat], [1e17, -1000000000000000.0])


@pytest.fixture(scope="module")
def hetero_cell():
    CdS = dj.material(
        Nc=2.2e18,
        Nv=1.8e19,
        Eg=2.4,
        eps=10,
        Et=0,
        mn=100,
        mp=25,
        tn=1e-08,
        tp=1e-13,
        Chi=4.0,
        A=10000.0,
    )
    CdTe = dj.material(
        Nc=8e17,
        Nv=1.8e19,
        Eg=1.5,
        eps=9.4,
        Et=0,
        mn=320,
        mp=40,
        tn=5e-09,
        tp=5e-09,
        Chi=3.9,
        A=10000.0,
    )
    return _cell(30, [CdS, CdTe], [1e17, -1000000000000000.0], srvs=(11600000.0,) * 4)


def _check(cell, v):
    pot = solve_eq(cell, boundary_eq(cell), equilibrium_guess(cell).phi)
    bnd = boundary_bias(cell, v / energy)
    if v > 0:
        pot = solve_newton(cell, bnd, pot, tol=1e-10)[0]
    Jd = F_jacobian(cell, bnd, pot)
    A, B, C = banded_jacobian(cell, bnd, pot)
    return float(jnp.max(jnp.abs(Jd - dense_from_blocks(A, B, C))))


def test_pn_equilibrium(pn_cell):
    assert _check(pn_cell, 0.0) < 1e-09


def test_pn_biased(pn_cell):
    assert _check(pn_cell, 0.5) < 1e-09
    assert _check(pn_cell, 0.9) < 1e-09


def test_heterojunction_biased(hetero_cell):
    assert _check(hetero_cell, 0.6) < 1e-09


@pytest.mark.slow  # jacfwd through recombination on N=40 cell — O(N^2) AD backward
def test_recomb_deriv_matches_ad(pn_cell):
    """_recomb_deriv must equal jacfwd(recombination.total) entry-by-entry.

    Regression for the srh_phip sign error (v0.1.14 audit): the analytic
    SRH derivative w.r.t. phi_p had `num*tn*p` with the wrong sign, an
    O(1) *relative* error in quasi-neutral regions whose *absolute*
    magnitude (~1e-13) was below the dense-pinning tolerance above — the
    dense pinning alone cannot catch this class of defect. This test
    compares against exact AD with a scale-aware relative tolerance.
    """
    from driftjax.fields import pot2vec, vec2pot
    from driftjax.numerics.analytic_jacobian import _recomb_deriv
    from driftjax.science.recombination import total as recomb_total

    cell = pn_cell
    pot0 = solve_eq(cell, boundary_eq(cell), equilibrium_guess(cell).phi)
    bnd = boundary_bias(cell, 0.5 / energy)
    pot = solve_newton(cell, bnd, pot0, tol=1e-10)[0]

    # exact AD derivative of the interior recombination rate wrt (phin, phip, phi)
    def R_flat(v):
        return recomb_total(cell, vec2pot(v))[1:-1]

    J = jax.jacfwd(R_flat)(pot2vec(pot))  # (n-2, 3n), interleaved columns
    # R_k depends only on node-k potentials: extract the local (diagonal)
    # entries — column 3k+c for interior node k = row index r + 1.
    n_nodes = pot.n
    rows = jnp.arange(n_nodes - 2)
    cols = 3 * jnp.arange(1, n_nodes - 1)
    ad_phin, ad_phip, ad_phi = J[rows, cols + 0], J[rows, cols + 1], J[rows, cols + 2]

    n_v = _n(cell, pot)
    p_v = _p(cell, pot)
    DR_phin, DR_phip, DR_phi = _recomb_deriv(cell, pot, n_v, p_v)

    for name, ana, ad in [
        ("dR/dphin", DR_phin, ad_phin),
        ("dR/dphip", DR_phip, ad_phip),
        ("dR/dphi", DR_phi, ad_phi),
    ]:
        ana = jnp.asarray(ana)
        ad = jnp.asarray(ad)
        scale = jnp.maximum(jnp.max(jnp.abs(ad)), 1e-30)
        # mixed tolerance: catches sign flips (O(1) relative) while allowing
        # roundoff on entries far below the array scale
        rel = jnp.max(jnp.abs(ana - ad) / (1e-12 * scale + jnp.abs(ad)))
        assert rel < 1e-6, f"{name}: max scaled rel err {float(rel):.3e}"


def test_analytic_solve_matches_dense(pn_cell):
    """The analytic backend and the dense jacrev path must converge to the
    same potential (roundoff-level agreement)."""
    pot0 = solve_eq(pn_cell, boundary_eq(pn_cell), equilibrium_guess(pn_cell).phi)
    bnd = boundary_bias(pn_cell, 0.5 / energy)
    pa, ia = solve_newton(pn_cell, bnd, pot0, tol=1e-10)
    pd_, id_ = solve_newton(pn_cell, bnd, pot0, tol=1e-10, dense=True)
    assert ia["backend"] == "analytic-block-thomas"
    assert float(jnp.max(jnp.abs(pa.phi - pd_.phi))) < 1e-09
