"""v0.1.18b: megakernel backend tests — native GE with row equilibration
inlined into the compiled Newton step (zero LAPACK host callbacks).

The default backend ("auto"/LAPACK) is preserved for the fragile singular
stress device; the native GE backend is an opt-in megakernel path.
"""

import jax.numpy as jnp
import pytest

import driftjax as dj
from driftjax.science.contacts import boundary_eq
from driftjax.simulator import init_cell


@pytest.fixture(scope="module")
def homojunction():
    Si = dj.material(
        Eg=1.424,
        Chi=3.9,
        eps=9.4,
        Nc=8e17,
        Nv=1.8e19,
        mn=100,
        mp=100,
        A=2e4,
    )
    return dj.Device(
        n_points=100,
        layers=[(1e-4, Si, 1e17), (1e-4, Si, -1e17)],
        Snl=1e7,
        Snr=0,
        Spl=0,
        Spr=1e7,
    )


def test_native_ge_backend_matches_lapack(homojunction):
    """Newton(backend="native_ge_eq") produces identical results to default."""
    sol_default = dj.simulate(homojunction, dj.Sweep(n_steps=21))
    sol_native = dj.simulate(
        homojunction,
        dj.Sweep(n_steps=21),
        solver=dj.Newton(backend="native_ge_eq", fused=True),
    )
    diff = abs(float(sol_default.eff) - float(sol_native.eff))
    assert diff < 1e-10, f"Efficiency differs: {diff:.2e}"
    assert abs(float(sol_default.voc) - float(sol_native.voc)) < 1e-10


def test_native_ge_backend_converges(homojunction):
    """Native GE backend converges on a standard homojunction."""
    sol = dj.simulate(
        homojunction,
        dj.Sweep(n_steps=21),
        solver=dj.Newton(backend="native_ge_eq", fused=True),
    )
    assert bool(sol.converged), "Native GE backend did not converge"
    assert float(sol.eff) > 0.0, "Native GE backend produced zero efficiency"


def test_native_ge_backend_step_match(homojunction, tmp_path):
    """Single Newton step: native GE matches LAPACK step."""
    from driftjax.solvers.newton import step_newton

    cell = init_cell(
        homojunction,
        None,
        alpha_mode="table",
        statistics="boltzmann",
        fused=True,
    )
    from driftjax.solvers.continuation import equilibrium_guess

    pot_init = equilibrium_guess(cell)
    bound = boundary_eq(cell)

    pot_lapack, _, stats_lapack = step_newton(
        cell,
        bound,
        pot_init,
        fused=True,
        analytic=True,
    )
    pot_native, _, stats_native = step_newton(
        cell,
        bound,
        pot_init,
        fused=True,
        analytic=True,
        backend="native_ge_eq",
    )
    diff = float(jnp.max(jnp.abs(pot_lapack.phi - pot_native.phi)))
    assert diff < 1e-10, f"Newton step differs: {diff:.2e}"
