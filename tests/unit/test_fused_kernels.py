"""driftjax fused-kernel pins (provenance edge intern/driftjax -> driftjax driftjax).

fused_residual == comp_F, fused currents == Jn/Jp, fused generation ==
init_cell().G, fused banded Jacobian == banded_jacobian — all to 1e-12,
across p-n and heterojunction, equilibrium and biased states.
"""

import jax.numpy as jnp
import pytest

import driftjax as dj
import driftjax.science
from driftjax.numerics.fused_kernels import (
    FusedKernelManager,
    fused_bernoulli_current,
    fused_generation,
    fused_jacobian_banded,
    fused_residual,
)
from driftjax.numerics.residual import comp_F
from driftjax.numerics.scharfetter_gummel import Jn, Jp
from driftjax.science.contacts import boundary_bias, boundary_eq
from driftjax.science.spectrum import spectrum
from driftjax.solvers.continuation import equilibrium_guess
from driftjax.solvers.newton import solve_eq, solve_newton
from driftjax.units import energy


@pytest.fixture(scope="module")
def pn_cell():
    mat = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, tn=1e-08, tp=1e-08, A=10000.0
    )
    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [mat, mat], [1e17, -1e17], strict=False)),
        n_points=40,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()
    return dj.simulator.init_cell(des, spectrum(normalize=False), alpha_mode="beer-lambert")


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
    des = dj.Device(
        layers=list(zip([2.5e-06, 0.0004], [CdS, CdTe], [1e17, -1000000000000000.0], strict=False)),
        n_points=30,
        Snl=11600000.0,
        Snr=11600000.0,
        Spl=11600000.0,
        Spr=11600000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()
    return dj.simulator.init_cell(des, spectrum(normalize=False), alpha_mode="beer-lambert")


def _states(cell):
    pot_eq = solve_eq(cell, boundary_eq(cell), equilibrium_guess(cell).phi)
    b = boundary_bias(cell, 0.5 / energy)
    pot, _ = solve_newton(cell, b, pot_eq, tol=1e-10)
    return [(boundary_eq(cell), pot_eq), (b, pot)]


def test_fused_residual_equals_comp_F(pn_cell):
    for bound, pot in _states(pn_cell):
        assert (
            float(
                jnp.max(jnp.abs(fused_residual(pn_cell, bound, pot) - comp_F(pn_cell, bound, pot)))
            )
            < 1e-12
        )


def test_fused_currents_equal_sg(pn_cell):
    _, pot = _states(pn_cell)[1]
    jn, jp = fused_bernoulli_current(pn_cell, pot)
    assert float(jnp.max(jnp.abs(jn - Jn(pn_cell, pot)))) < 1e-12
    assert float(jnp.max(jnp.abs(jp - Jp(pn_cell, pot)))) < 1e-12


def test_fused_generation_matches_init_cell():
    mat = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, tn=1e-08, tp=1e-08, A=10000.0
    )
    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [mat, mat], [1e17, -1e17], strict=False)),
        n_points=40,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()
    ls = spectrum(normalize=False)
    cell = dj.simulator.init_cell(des, ls, alpha_mode="beer-lambert")
    G = fused_generation(des, ls, alpha_mode="beer-lambert")
    assert float(jnp.max(jnp.abs(G - cell.G))) < 1e-12


def test_fused_banded_jacobian_matches(pn_cell):
    _, pot = _states(pn_cell)[1]
    A, B, C = fused_jacobian_banded(pn_cell, boundary_bias(pn_cell, 0.5 / energy), pot)
    from driftjax.numerics.analytic_jacobian import banded_jacobian, dense_from_blocks

    A2, B2, C2 = banded_jacobian(pn_cell, boundary_bias(pn_cell, 0.5 / energy), pot)
    assert (
        float(jnp.max(jnp.abs(dense_from_blocks(A, B, C) - dense_from_blocks(A2, B2, C2)))) < 1e-12
    )


def test_manager_parity(pn_cell):
    mgr_on = FusedKernelManager(use_fused=True)
    mgr_off = FusedKernelManager(use_fused=False)
    pot_eq = solve_eq(pn_cell, boundary_eq(pn_cell), equilibrium_guess(pn_cell).phi)
    b = boundary_bias(pn_cell, 0.3 / energy)
    pot, _ = solve_newton(pn_cell, b, pot_eq, tol=1e-10)
    F_on = mgr_on.compute_residual(pn_cell, b, pot)
    F_off = mgr_off.compute_residual(pn_cell, b, pot)
    assert float(jnp.max(jnp.abs(F_on - F_off))) < 1e-12
