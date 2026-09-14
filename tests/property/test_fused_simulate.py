"""P3: fused kernels wired into the solve/simulate path (optional residual path).

Gate: simulate(fused=True) == simulate(fused=False) bit-exact on the IV curve,
and solve_newton(fused=True) reaches the identical solution (Δpot = 0).
"""

import jax.numpy as jnp
import pytest

import driftjax as dj
import driftjax.science
from driftjax.science.contacts import boundary_bias
from driftjax.science.spectrum import spectrum
from driftjax.solvers.continuation import equilibrium_guess
from driftjax.solvers.newton import solve_newton


@pytest.fixture(scope="module")
def fused_design(si_canon):
    # Bit-exact gate is mesh-independent: N=60/9 steps halves the work.
    return dj.Device(
        layers=list(zip([0.0001, 0.0001], [si_canon, si_canon], [1e17, -1e17], strict=False)),
        n_points=60,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()


@pytest.fixture(scope="module")
def fused_cell(fused_design):
    return dj.simulator.init_cell(
        fused_design, spectrum(normalize=False), alpha_mode="beer-lambert"
    )


def test_solve_newton_fused_identical(fused_cell):
    bound = boundary_bias(fused_cell, 0.4)
    pot0 = equilibrium_guess(fused_cell)
    p_plain, s_plain = solve_newton(fused_cell, bound, pot0, fused=False)
    p_fused, s_fused = solve_newton(fused_cell, bound, pot0, fused=True)
    assert s_fused["iters"] == s_plain["iters"]
    d = max(
        float(jnp.max(jnp.abs(p_plain.phi_n - p_fused.phi_n))),
        float(jnp.max(jnp.abs(p_plain.phi_p - p_fused.phi_p))),
        float(jnp.max(jnp.abs(p_plain.phi - p_fused.phi))),
    )
    assert d == 0.0


def test_simulate_fused_identical_iv(fused_design):
    ls = spectrum(normalize=False)
    r0 = dj.simulate(
        fused_design,
        dj.Sweep(vmax=1.1, n_steps=9, fused=False),
        optics=dj.BeerLambert(alpha_mode="beer-lambert"),
        ls=ls,
    )
    r1 = dj.simulate(
        fused_design,
        dj.Sweep(vmax=1.1, n_steps=9, fused=True),
        optics=dj.BeerLambert(alpha_mode="beer-lambert"),
        ls=ls,
    )
    assert float(jnp.max(jnp.abs(r0.current - r1.current))) == 0.0
    assert float(r0.efficiency) == float(r1.efficiency)
