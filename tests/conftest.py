"""Shared fixtures.  Importing driftjax enables x64 globally."""

import os
import sys
from pathlib import Path

# Persistent XLA compilation cache: kernels compile once and are reused by
# every later test process (pytest workers and `python -m driftjax`
# subprocesses alike) instead of recompiling float64 programs each run.
# Must be set before JAX is first imported.
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/driftjax-xla-cache")
)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.2")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import pytest

# Make ``tests/`` itself importable so every test directory can use the
# shared device builders in ``tests/helpers.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import driftjax as dj
import driftjax.science
from driftjax.io import material_to_dict
from driftjax.science.spectrum import spectrum


@pytest.fixture(scope="session")
def si_canon() -> dict:
    return material_to_dict(
        dj.material(
            Chi=3.9,
            Eg=1.5,
            eps=9.4,
            Nc=8e17,
            Nv=1.8e19,
            mn=100,
            mp=100,
            tn=1e-08,
            tp=1e-08,
            A=20000.0,
        )
    )


@pytest.fixture(scope="session")
def small_cell(si_canon):
    """n=60 canonical pn junction (fast, for CI-level tests)."""
    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [si_canon, si_canon], [1e17, -1e17], strict=False)),
        n_points=60,
        Snl=10000000.0,
        Snr=10000000.0,
        Spl=10000000.0,
        Spr=10000000.0,
    ).design()
    return dj.simulator.init_cell(des, spectrum())


@pytest.fixture(scope="session")
def small_eq(small_cell):
    from driftjax.science.contacts import boundary_eq
    from driftjax.solvers.continuation import equilibrium_guess
    from driftjax.solvers.newton import solve_eq

    return solve_eq(small_cell, boundary_eq(small_cell), equilibrium_guess(small_cell).phi)


@pytest.fixture(scope="session")
def pn80_cell(si_canon):
    """n=80 pn junction shared by test_fused_kernels & test_analytic_jacobian (was 2 compiles)."""
    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [si_canon, si_canon], [1e17, -1e17], strict=False)),
        n_points=80,
        Snl=10000000.0,
        Snr=10000000.0,
        Spl=10000000.0,
        Spr=10000000.0,
    ).design()
    return dj.simulator.init_cell(des, spectrum())


@pytest.fixture(scope="session")
def fast_cell120(si_canon):
    """n=120 fast suite cell shared by fused_simulate, checkpointing, release_validated (was 3 compiles)."""
    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [si_canon, si_canon], [1e17, -1e17], strict=False)),
        n_points=120,
        Snl=10000000.0,
        Snr=10000000.0,
        Spl=10000000.0,
        Spr=10000000.0,
    ).design()
    return dj.simulator.init_cell(des, spectrum())


@pytest.fixture(scope="session")
def shared_sweep9(si_canon):
    """One N=60 / 9-step Beer–Lambert sweep shared by every test that only
    needs *a* Solution (rendering, API surface, convention checks) instead
    of its own physics run. Saves one full sweep per consumer."""
    import driftjax as _dj

    des = _dj.Device(
        layers=list(zip([0.0001, 0.0001], [si_canon, si_canon], [1e17, -1e17], strict=False)),
        n_points=60,
        Snl=10000000.0,
        Snr=10000000.0,
        Spl=10000000.0,
        Spr=10000000.0,
    ).design()
    return _dj.simulate(
        des,
        _dj.Sweep(vmax=0.8, n_steps=9),
        optics=_dj.BeerLambert(alpha_mode="beer-lambert"),
        ls=spectrum(),
    )
