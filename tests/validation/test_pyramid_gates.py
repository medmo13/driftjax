"""Validation-pyramid gate regressions (locked after the tooling pass).

These pin the four gates that the tooling audit exposed as stale/broken:
  * L1 PCHIP MPP bounds vs the analytic parabola MPP,
  * L3 Poisson MMS at k=1 (error = (kh)²/12 exactly),
  * L5 G–R balance on the *asymmetric* blocking-contact cell (the previous
    trapezoid integral + |Jn|-only scale misfired on it),
  * L9 banded arithmetic-intensity model (W=13 ⇒ AI = 0.346).
"""
import jax.numpy as jnp

import driftjax.units as units
from driftjax.numerics.spline import calcPmax_cubic
from driftjax.runtime.performance import arithmetic_intensity
from driftjax.science.contacts import boundary_bias
from driftjax.science.spectrum import spectrum
from driftjax.solvers.continuation import equilibrium_guess
from driftjax.solvers.newton import solve_eq, solve_newton
from driftjax.validation.conservation import gr_balance_residual
from driftjax.validation.manufactured import mms_phi, poisson_mms_residual


def test_l1_pchip_mpp_on_parabolic_iv():
    v = jnp.linspace(0.0, 1.0, 21)
    jv = 0.06 - 0.05 * v ** 2
    p, vm = calcPmax_cubic(v, jv)
    assert 0.024 < float(p) < 0.027
    assert 0.6 < float(vm) < 0.7

def test_l3_mms_rate_at_k1():
    errs = [float(jnp.max(jnp.abs(poisson_mms_residual(n, k=1.0)))) for n in (40, 80, 160)]
    assert errs[0] < 0.01
    p = [float(jnp.log(errs[i] / errs[i + 1]) / jnp.log(2.0)) for i in range(2)]
    assert all(pp > 1.8 for pp in p)
    assert abs(errs[0] - (1.0 / 40.0) ** 2 / 12.0) < 1e-05

def test_l9_banded_arithmetic_intensity():
    ai = arithmetic_intensity(500, 'banded')
    assert abs(ai - 108.0 * 500 / (13 * 3 * 500 * 8)) < 1e-09
    assert 0.3 < ai < 0.4

def test_l5_gr_balance_blocking_contact_cell():
    """Asymmetric cell (Snr=0 blocks electrons): identity must still hold."""
    import driftjax as dj
    mat = dj.material(Chi=3.9, Eg=1.5, eps=9.4, Nc=8e+17, Nv=1.8e+19, mn=100, mp=100, tn=1e-08, tp=1e-08, A=10000.0)
    des = dj.Device(layers=list(zip([4e-05, 0.0001], [mat, mat], [1e+17, -1000000000000000.0])), n_points=100, Snl=10000000.0, Snr=0, Spl=0, Spr=10000000.0).design()
    cell = dj.simulator.init_cell(des, spectrum())
    pot0 = solve_eq(cell, boundary_bias(cell, 0.0), equilibrium_guess(cell).phi)
    pot, _ = solve_newton(cell, boundary_bias(cell, 0.4 / units.energy), pot0, tol=1e-09)
    mms_phi(40, k=1.0)
    assert gr_balance_residual(cell, pot) < 1e-06
