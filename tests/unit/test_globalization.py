"""Newton globalization policies: convergence contracts on a stiff device.

Pins, at tiny mesh sizes so this runs in seconds:
  * damped Newton (the ``auto`` first stage) recovers the equilibrium root
    from a guess displaced by several thermal voltages;
  * ``solve_ptc`` (pseudo-transient continuation) started from its standard
    staged initial guess (Poisson-only solution) reaches the same root;
  * ``solve_newton_ls`` (Armijo backtracking) agrees with damped Newton at
    forward bias.

Known limitation, deliberately NOT asserted otherwise: raw ``solve_ptc``
from strongly out-of-basin guesses can diverge because its Newton tail
applies no step damping; production entry points therefore stage through
``solve_eq`` first and route ``globalization="auto"`` through the line
search for recovery.
"""

import jax.numpy as jnp
import pytest

import driftjax as dj
from driftjax.fields import Potentials
from driftjax.science.contacts import boundary_bias, boundary_eq
from driftjax.solvers.continuation import equilibrium_guess
from driftjax.solvers.newton import solve_eq, solve_newton
from driftjax.solvers.ptc import solve_newton_ls, solve_ptc


@pytest.fixture(scope="module")
def stiff_cell(si_canon):
    """Small, heavily doped device: a stiff but cheap DDP problem."""
    des = dj.Device(
        layers=list(zip([5e-5, 5e-5], [si_canon, si_canon], [1e18, -1e18], strict=False)),
        n_points=24,
        Snl=1e7,
        Snr=1e7,
        Spl=1e7,
        Spr=1e7,
    ).design()
    return dj.simulator.init_cell(des, dj.AM15G())


def test_damped_newton_from_displaced_guess(stiff_cell):
    """Damped Newton recovers the equilibrium root from +5 Vt displacement."""
    cell = stiff_cell
    bound = boundary_eq(cell)
    poisson = solve_eq(cell, bound, equilibrium_guess(cell).phi)
    reference = solve_newton(cell, bound, Potentials(
        jnp.zeros_like(poisson.phi), jnp.zeros_like(poisson.phi), poisson.phi))[0]

    displaced = poisson.phi + 5.0
    pot, stats = solve_newton(
        cell, bound, Potentials(jnp.zeros_like(displaced), jnp.zeros_like(displaced), displaced)
    )
    assert stats.get("converged", True), stats
    err = float(jnp.max(jnp.abs(pot.phi - reference.phi)))
    scale = float(jnp.max(jnp.abs(reference.phi))) + 1.0
    assert err < 1e-6 * scale, err


def test_ptc_matches_staged_root(stiff_cell):
    """PTC from the staged Poisson-only guess reaches the coupled root."""
    cell = stiff_cell
    bound = boundary_eq(cell)
    poisson = solve_eq(cell, bound, equilibrium_guess(cell).phi)
    reference = solve_newton(cell, bound, Potentials(
        jnp.zeros_like(poisson.phi), jnp.zeros_like(poisson.phi), poisson.phi))[0]

    start = Potentials(
        jnp.zeros_like(poisson.phi), jnp.zeros_like(poisson.phi), poisson.phi
    )
    pot, stats = solve_ptc(cell, bound, start, max_steps=80)
    assert stats["converged"], stats
    err = float(jnp.max(jnp.abs(pot.phi - reference.phi)))
    scale = float(jnp.max(jnp.abs(reference.phi))) + 1.0
    assert err < 1e-6 * scale, err


def test_line_search_matches_damped_newton_at_bias(stiff_cell):
    """At forward bias the Armijo line search agrees with damped Newton."""
    cell = stiff_cell
    bound0 = boundary_eq(cell)
    eq = solve_newton(cell, bound0, equilibrium_guess(cell))[0]
    bound_v = boundary_bias(cell, 0.4 / float(dj.energy))
    pot_ref, ref_stats = solve_newton(cell, bound_v, eq)
    assert ref_stats.get("converged", True)

    import jax.numpy as jnp

    pot_ls, ls_stats = solve_newton_ls(cell, bound_v, eq, f_tol=1e-8)
    assert ls_stats["converged"], ls_stats
    err = float(jnp.max(jnp.abs(pot_ls.phi - pot_ref.phi)))
    assert err < 1e-6, err
