"""Contact BCs: physical equilibrium densities, bias-0 == eq, Vbi."""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.smoke  # no solves: boundary-condition math only

import driftjax as dj
import driftjax.science
from driftjax.science.carrier_statistics import ni
from driftjax.science.contacts import boundary_bias, boundary_eq, eq_carrier_dens
from driftjax.science.spectrum import spectrum


def _cell(si_canon, n=40):
    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [si_canon, si_canon], [1e17, -1e17], strict=False)),
        n_points=n,
        Snl=10000000.0,
        Snr=10000000.0,
        Spl=10000000.0,
        Spr=10000000.0,
    ).design()
    return dj.simulator.init_cell(des, spectrum())


def test_eq_carrier_dens_satisfy_mass_action(si_canon):
    cell = _cell(si_canon)
    neq0, neqL, peq0, peqL = eq_carrier_dens(cell)
    ni0 = ni(cell)[0]
    assert abs(float(neq0 * peq0) - float(ni0**2)) / float(ni0**2) < 1e-08
    assert abs(float(neqL * peqL) - float(ni0**2)) / float(ni0**2) < 1e-08


def test_boundary_bias_zero_equals_eq(small_cell):
    eq = boundary_eq(small_cell)
    b0 = boundary_bias(small_cell, jnp.float64(0.0))
    for f in ("phi0", "phiL", "neq0", "neqL", "peq0", "peqL"):
        assert float(jnp.abs(getattr(eq, f) - getattr(b0, f))) < 1e-15, f


def test_builtin_voltage(si_canon):
    cell = _cell(si_canon)
    phi0, phiL = (boundary_eq(cell).phi0, boundary_eq(cell).phiL)
    ni2 = float(ni(cell)[0] ** 2)
    expect = float(jnp.log(jnp.abs(cell.Ndop[0] * cell.Ndop[-1]) / ni2))
    assert abs(float(phi0 - phiL) - expect) < 1e-08
