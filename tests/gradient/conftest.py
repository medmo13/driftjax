"""Shared fixtures for tests/gradient — one canonical n=40 pn device.

``test_adjoint_ab.py`` exercises IFT adjoint vs finite differences at
V = 0.5 V on the symmetric p-n junction.  A single module-scoped fixture
builds the device / cell / equilibrium state ONCE so all gradient tests
share one solve chain and one XLA compile.
"""

import pytest

import driftjax as dj
import driftjax.science
from driftjax.science.contacts import boundary_eq
from driftjax.science.spectrum import spectrum
from driftjax.solvers.continuation import equilibrium_guess
from driftjax.solvers.newton import solve_eq


@pytest.fixture(scope="module")
def grad_pn_setup():
    """(design, cell, pot_eq) for the canonical n=40 gradient-check device."""
    mat = dj.material(
        Chi=3.9,
        Eg=1.5,
        eps=9.4,
        Nc=8e17,
        Nv=1.8e19,
        mn=100.0,
        mp=100.0,
        tn=1e-08,
        tp=1e-08,
        A=10000.0,
    )
    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [mat, mat], [1e17, -1e17], strict=False)),
        n_points=40,
        Snl=10000000.0,
        Snr=10000000.0,
        Spl=10000000.0,
        Spr=10000000.0,
    ).design()
    cell = dj.simulator.init_cell(des, spectrum())
    pot_eq = solve_eq(cell, boundary_eq(cell), equilibrium_guess(cell).phi)
    return (des, cell, pot_eq)
