"""Structural invariants: DOF roundtrip, Jacobian identity, banded layout."""

import jax
import jax.numpy as jnp
import pytest

from driftjax.fields import Potentials, pot2vec, vec2pot
from driftjax.numerics.banded_solve import extract_blocks
from driftjax.numerics.linalg import _banded_matvec, dense_to_banded, dense_to_csr
from driftjax.numerics.residual import F_jacobian, comp_F

pytestmark = pytest.mark.smoke


def test_pot2vec_roundtrip():
    n = 30
    p = Potentials(jnp.linspace(-1, 1, n), jnp.linspace(0, 1, n), jnp.linspace(2, 3, n))
    v = pot2vec(p)
    assert v.shape == (3 * n,)
    p2 = vec2pot(v)
    for k in ("phi_n", "phi_p", "phi"):
        assert float(jnp.max(jnp.abs(getattr(p2, k) - getattr(p, k)))) == 0.0
    assert float(v[0]) == float(p.phi_n[0])
    assert float(v[1]) == float(p.phi_p[0])
    assert float(v[2]) == float(p.phi[0])
    assert float(v[3]) == float(p.phi_n[1])


def test_jacobian_identity(small_cell):
    from driftjax.science.contacts import boundary_eq

    p0 = Potentials(
        jnp.zeros(small_cell.x.size), jnp.zeros(small_cell.x.size), jnp.zeros(small_cell.x.size)
    )
    beq = boundary_eq(small_cell)
    J = F_jacobian(small_cell, beq, p0)
    J2 = jax.jacrev(lambda v: comp_F(small_cell, beq, vec2pot(v)))(pot2vec(p0))
    # jacfwd (F_jacobian) vs jacrev agree to machine precision relative to
    # the entry scale (~1e65 from Nc*Nv products) — never bit-exact, since
    # forward/reverse mode sum in different orders.
    scale = float(jnp.max(jnp.abs(J2)))
    assert float(jnp.max(jnp.abs(J - J2))) / scale < 1e-12


def test_banded_matvec_matches_dense():
    n = 30
    A = jax.random.normal(jax.random.key(0), (3 * n, 3 * n))
    A = jnp.tril(jnp.triu(A, -6), 6)
    B = dense_to_banded(A)
    x = jax.random.normal(jax.random.PRNGKey(1), (3 * n,))
    assert float(jnp.max(jnp.abs(_banded_matvec(B, x) - A @ x))) < 1e-10


def test_extract_blocks_consistent():
    n = 20
    from driftjax.numerics.banded_solve import _legacy_solve_block_tridiagonal as solve_block_tridiagonal

    A = jax.random.normal(jax.random.PRNGKey(2), (3 * n, 3 * n))
    A = jnp.tril(jnp.triu(A, -3), 3) + 30.0 * jnp.eye(3 * n)
    D, U, Lb = extract_blocks(A)
    assert D.shape == (n, 3, 3) and U.shape == (n - 1, 3, 3) and (Lb.shape == (n - 1, 3, 3))
    b = jax.random.normal(jax.random.PRNGKey(3), (3 * n,))
    x_bt = solve_block_tridiagonal(A, b)
    x_d = jnp.linalg.solve(A, b)
    assert float(jnp.max(jnp.abs(x_bt - x_d))) < 1e-09


def test_dense_to_csr_roundtrip():
    n = 20
    A = jnp.tril(jnp.triu(jax.random.normal(jax.random.PRNGKey(5), (3 * n, 3 * n)), -6), 6)
    data, indices, indptr = dense_to_csr(A)
    x = jax.random.normal(jax.random.PRNGKey(6), (3 * n,))
    spill = jnp.zeros(3 * n)
    for r in range(3 * n):
        spill = spill.at[r].set(
            jnp.sum(data[indptr[r] : indptr[r + 1]] * x[indices[indptr[r] : indptr[r + 1]]])
        )
    assert float(jnp.max(jnp.abs(spill - A @ x))) < 1e-10
