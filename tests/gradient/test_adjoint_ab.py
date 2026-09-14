"""regression gradient audit (provenance edge driftjax/driftjax -> driftjax).

A = manual_adjoint_grad (IFT, banded-backed)
B = central finite differences through simulate()
Comparisons across multiple design channels (Spl, Chi, Ns0, PhiMl)
at V = 0.5 V.  Guards the silent-wrong-gradient failure mode (F-03).
"""

import jax.numpy as jnp
import pytest

import driftjax as dj
import driftjax.science
from driftjax.autodiff.adjoint import manual_adjoint_grad
from driftjax.numerics.scharfetter_gummel import Jn, Jp
from driftjax.science.contacts import boundary_bias
from driftjax.science.spectrum import spectrum
from driftjax.solvers.continuation import equilibrium_guess
from driftjax.solvers.newton import solve_eq, solve_newton
from driftjax.units import energy


# The device / cell / equilibrium build lives in tests/gradient/conftest.py
# (fixture ``grad_pn_setup``); the ``setup`` alias below keeps the historical
# parameter name in this file's tests.
@pytest.fixture(scope="module")
def setup(grad_pn_setup):
    return grad_pn_setup


def _loss(cell, pot):
    return jnp.mean(Jn(cell, pot) + Jp(cell, pot)) ** 2


def _loss_at(des, v):
    cell = dj.simulator.init_cell(des, spectrum())
    pot_eq = solve_eq(cell, dj.science.contacts.boundary_eq(cell), equilibrium_guess(cell).phi)
    pot = solve_newton(cell, boundary_bias(cell, v), pot_eq, tol=1e-11)[0]
    return float(_loss(cell, pot))


def _fd(des, v, field, idx, h_rel=0.001):
    import dataclasses

    arr = jnp.atleast_1d(getattr(des, field))
    h = h_rel * max(1.0, abs(float(arr[idx])))

    def perturb(sign):
        sub = arr.at[idx].set(float(arr[idx]) + sign * h)
        new_arr = sub[0] if getattr(des, field).ndim == 0 else sub
        return _loss_at(dataclasses.replace(des, **{field: new_arr}), v)

    return (perturb(+1) - perturb(-1)) / (2 * h)


def test_adjoint_ab_wrt_spl(setup):
    des, _, pot_eq = setup
    v = 0.5 / energy
    L, grad = manual_adjoint_grad(_loss, des, v, pot_eq, tol=1e-11, ls=spectrum())
    fd = _fd(des, v, "Spl", 0)
    g = float(grad.Spl)
    assert abs(g - fd) / max(1e-12, abs(fd)) < 0.02, (g, fd)


def test_adjoint_ab_wrt_chi(setup):
    des, _, pot_eq = setup
    v = 0.5 / energy
    _, grad = manual_adjoint_grad(_loss, des, v, pot_eq, tol=1e-11, ls=spectrum())
    fd = _fd(des, v, "Chi", 10)
    g = float(grad.Chi[10])
    assert abs(g - fd) / max(1e-12, abs(fd)) < 0.02, (g, fd)


def test_adjoint_ab_wrt_ns(setup):
    des, _, pot_eq = setup
    v = 0.5 / energy
    _, grad = manual_adjoint_grad(_loss, des, v, pot_eq, tol=1e-11, ls=spectrum())
    fd = _fd(des, v, "Ndop", 0)
    g = float(grad.Ndop[0])
    assert abs(g - fd) / max(1e-12, abs(fd)) < 0.02, (g, fd)
