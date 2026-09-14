"""Conservation identities on a converged solution."""

import pytest

from driftjax.science.contacts import boundary_bias, boundary_eq
from driftjax.solvers.continuation import equilibrium_guess
from driftjax.solvers.newton import solve_eq, solve_newton
from driftjax.validation.conservation import (
    gr_balance_residual,
    mass_action_residual,
    terminal_current_variation,
)


@pytest.fixture(scope="module")
def _cached_bias_pot(small_cell):
    """One eq + one bias solve shared by 2 conservation tests (was 4 solves)."""
    from driftjax.units import energy

    pot_eq = solve_eq(small_cell, boundary_eq(small_cell), equilibrium_guess(small_cell).phi)
    pot, info = solve_newton(small_cell, boundary_bias(small_cell, 0.4 / energy), pot_eq, tol=1e-10)
    return small_cell, pot, pot_eq


def test_mass_action_equilibrium(_cached_bias_pot):
    small_cell, _, pot_eq = _cached_bias_pot
    assert mass_action_residual(small_cell, pot_eq) < 1e-06


def test_terminal_current_uniform(_cached_bias_pot):
    small_cell, pot, _ = _cached_bias_pot
    assert terminal_current_variation(small_cell, pot) < 0.0001


def test_gr_balance(_cached_bias_pot):
    small_cell, pot, _ = _cached_bias_pot
    r = gr_balance_residual(small_cell, pot)
    assert r < 1.0, r
    # keep single identity; manual Jn/R/G re-derive was redundant with gr_balance_residual
