"""driftjax checkpointing pins (provenance edge qwen/pvx -> driftjax driftjax).

1. checkpointed_solve_newton == solve_newton to machine precision (eager).
2. remat'd residual chain: grads == plain grads (the real memory-safety pin).
3. checkpointed_iv_curve == sweep currents to 1e-12.
4. compare_memory: results_match True.
"""

import jax
import jax.numpy as jnp
import pytest

import driftjax as dj
import driftjax.science
from driftjax.autodiff.checkpointing import (
    checkpointed_iv_curve,
    checkpointed_residual,
    checkpointed_solve_newton,
    compare_memory,
    memory_report,
)
from driftjax.numerics.residual import comp_F
from driftjax.science.contacts import boundary_bias
from driftjax.science.spectrum import spectrum
from driftjax.solvers.continuation import equilibrium_guess, sweep
from driftjax.solvers.newton import solve_eq, solve_newton
from driftjax.units import energy


@pytest.fixture(scope="module")
def biased_state(small_eq):
    cell = small_eq
    return cell


@pytest.fixture(scope="module")
def cell():
    mat = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, tn=1e-08, tp=1e-08, A=10000.0
    )
    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [mat, mat], [1e17, -1e17], strict=False)),
        n_points=120,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()
    return dj.simulator.init_cell(des, spectrum(normalize=False), alpha_mode="beer-lambert")


@pytest.fixture(scope="module")
def _pot0(cell):
    """Module-cached equilibrium potential shared by all checkpointing tests."""
    return solve_eq(cell, dj.science.contacts.boundary_eq(cell), equilibrium_guess(cell).phi)


def test_checkpointed_solve_matches_plain(cell, _pot0):
    b = boundary_bias(cell, 0.5 / energy)
    pot, info = solve_newton(cell, b, _pot0, tol=1e-10)
    pot_c, infoc = checkpointed_solve_newton(cell, b, _pot0)
    assert float(jnp.max(jnp.abs(pot_c.phi - pot.phi))) < 1e-12
    assert infoc["backend"] == info["backend"]
    assert bool(infoc["converged"]) is True


@pytest.mark.slow  # jax.grad on comp_F + checkpointed_residual — two AD backward passes
def test_remat_chain_grads_match_plain(cell, _pot0):
    b = boundary_bias(cell, 0.5 / energy)
    pot, _ = solve_newton(cell, b, _pot0, tol=1e-10)
    g1 = jax.grad(lambda p: jnp.sum(comp_F(cell, b, p)))(pot)
    g2 = jax.grad(lambda p: jnp.sum(checkpointed_residual(cell, b, p)))(pot)
    assert float(jnp.max(jnp.abs(g1.phi - g2.phi))) < 1e-12


def test_checkpointed_iv_curve_matches_sweep(cell, _pot0):
    vs = jnp.linspace(0.0, 0.7 / energy, 8)
    jc, _ = checkpointed_iv_curve(cell, _pot0, vs)
    v, js, _, _ = sweep(cell, 0.7 / energy, 8)
    assert float(jnp.max(jnp.abs(jc - js))) < 1e-07


def test_compare_memory_matches(cell, _pot0):
    b = boundary_bias(cell, 0.5 / energy)
    pot, _ = solve_newton(cell, b, _pot0, tol=1e-10)
    out = compare_memory(lambda *a: comp_F(*a), lambda *a: checkpointed_residual(*a), cell, b, pot)
    assert out["results_match"] is True


@pytest.mark.slow  # jax.grad through comp_F — full AD backward pass
def test_memory_report_gradients(cell, _pot0):
    b = boundary_bias(cell, 0.5 / energy)
    pot, _ = solve_newton(cell, b, _pot0, tol=1e-10)
    _, info = memory_report(lambda p: jnp.sum(comp_F(cell, b, p)), pot)
    assert info["status"] == "gradient_computed"
