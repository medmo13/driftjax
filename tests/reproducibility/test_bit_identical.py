"""Bit-identical reruns + provenance stability."""

import jax.numpy as jnp

from driftjax.runtime.provenance import fingerprint, record
from driftjax.science.contacts import boundary_bias, boundary_eq
from driftjax.solvers.continuation import equilibrium_guess
from driftjax.solvers.newton import solve_eq, solve_newton
from driftjax.units import energy


def test_solve_bit_identical(small_cell):
    pot_eq = solve_eq(small_cell, boundary_eq(small_cell), equilibrium_guess(small_cell).phi)
    p1 = solve_newton(small_cell, boundary_bias(small_cell, 0.4 / energy), pot_eq, tol=1e-10)[0]
    p2 = solve_newton(small_cell, boundary_bias(small_cell, 0.4 / energy), pot_eq, tol=1e-10)[0]
    import jax

    version_a = jax.tree.leaves(p1)
    version_b = jax.tree.leaves(p2)
    for a, b in zip(version_a, version_b, strict=False):
        assert float(jnp.max(jnp.abs(a - b))) == 0.0


def test_fingerprint_stable():
    f1 = fingerprint()
    f2 = fingerprint()
    assert f1 == f2
    assert len(f1) == 16


def test_record_shape():
    r = record()
    for k in (
        "package",
        "version",
        "jax",
        "python",
        "platform",
        "config_fingerprint",
        "timestamp_utc",
    ):
        assert k in r
    assert r["package"] == "driftjax"
