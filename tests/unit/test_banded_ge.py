"""Tests for the band-storage pivoted GE via lax.scan (banded_ge.py).

Covers:
  - Forward solve correctness vs LAPACK (8 seeds, n=30)
  - Transpose solve correctness vs dense LAPACK (8 seeds, n=30)
  - Differentiability via FD comparison (n=8)
  - Singularity detection on the 3-layer stress device (rank 78/120)
  - Differentiability on the stress device (when non-singular)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import solve_banded

from driftjax.numerics.banded_ge import (
    _block_thomas,
    _dense_from_blocks,
)
from driftjax.numerics.banded_native import native_banded_solve, native_banded_transpose
from driftjax.numerics.banded_solve import _blocks_to_lapack_banded


def _rand_banded(n, seed):
    """Random well-conditioned block-tridiagonal system."""
    rs = np.random.RandomState(seed)
    A = jnp.asarray(rs.randn(n, 3, 3) * 0.3)
    A = A.at[:, 0, 0].set(A[:, 0, 0] + 6)
    A = A.at[:, 1, 1].set(A[:, 1, 1] + 6)
    A = A.at[:, 2, 2].set(A[:, 2, 2] + 6)
    B = jnp.asarray(rs.randn(n - 1, 3, 3) * 0.3)
    C = jnp.asarray(rs.randn(n - 1, 3, 3) * 0.3)
    b = jnp.asarray(rs.randn(n, 3))
    return A, B, C, b


@pytest.mark.parametrize("seed", range(8))
def test_ge_forward_matches_lapack(seed):
    """Block Thomas GE matches LAPACK dgbsv to machine precision."""
    A, B, C, b = _rand_banded(30, seed)
    x_ge, info = native_banded_solve(A, B, C, b)
    n = A.shape[0]
    ab, kl, ku = _blocks_to_lapack_banded(A, B, C)
    x_ref = solve_banded((kl, ku), np.array(ab), b.reshape(-1)).reshape(n, 3)
    rel = float(jnp.linalg.norm(x_ge - x_ref) / (jnp.linalg.norm(x_ref) + 1e-30))
    assert bool(info["ok"])
    assert int(info["method"]) == 0  # GE path used
    assert rel < 1e-10, rel


@pytest.mark.parametrize("seed", range(8))
def test_ge_transpose_matches_dense(seed):
    """Transpose solve (J^T lam = g) matches dense LAPACK."""
    A, B, C, b = _rand_banded(30, seed)
    x_ge, info = native_banded_transpose(A, B, C, b.reshape(30, 3))
    M = _dense_from_blocks(np.array(A), np.array(B), np.array(C))
    x_ref = np.linalg.solve(M.T, b.reshape(-1)).reshape(30, 3)
    rel = float(jnp.linalg.norm(x_ge - x_ref) / (jnp.linalg.norm(x_ref) + 1e-30))
    assert bool(info["ok"])
    assert rel < 1e-10, rel


def test_ge_differentiable():
    """jax.grad of native GE solve matches central finite differences."""
    n = 8
    A, B, C, b = _rand_banded(n, 3)
    A = jnp.array(A)
    B = jnp.array(B)
    C = jnp.array(C)
    b = jnp.array(b)

    def loss(Aflat):
        x, _ = native_banded_solve(Aflat.reshape(n, 3, 3), B, C, b)
        return jnp.sum(x**2)

    g = jax.grad(loss)(A.reshape(-1))
    A0 = A.reshape(-1)
    h = 1e-5
    for k in range(0, 9 * n, 7):
        e = jnp.zeros(9 * n)
        e = e.at[k].set(1.0)
        fd = (float(loss(A0 + h * e)) - float(loss(A0 - h * e))) / (2 * h)
        ad = float(g[k])
        rel = abs(ad - fd) / (abs(fd) + 1e-12)
        assert rel < 1e-4, f"entry {k}: ad={ad:.6e}, fd={fd:.6e}, rel={rel:.2e}"


def test_ge_singular_stress_detected():
    """3-layer n-p-n stress device (rank 78/120) detected as singular."""
    import driftjax as dj
    from driftjax.numerics.banded_solve import extract_blocks
    from driftjax.numerics.residual import F_jacobian
    from driftjax.science.contacts import boundary_bias
    from driftjax.solvers.api import Newton

    mt = dj.material(Eg=1.6, Chi=3.9, eps=20.0, Nc=1e18, Nv=1e18, mn=100.0, mp=100.0, A=2e4)
    mat = dj.material(Eg=1.4, Chi=3.0, eps=10.0, Nc=1e18, Nv=1e18, mn=130.0, mp=160.0, A=2e4)
    dev = dj.Device(
        n_points=40,
        layers=[(2e-5, mt, 1e18), (6e-5, mat, -1e18), (2e-5, mt, 1e18)],
        Snl=1e7,
        Snr=0.0,
        Spl=0.0,
        Spr=1e7,
    )
    s = dj.simulate(
        dev, dj.Sweep(vmax=1.0, n_steps=3), solver=Newton(globalization="ls", fused=False)
    )
    J = F_jacobian(s.cell, boundary_bias(s.cell, float(s.voltages[0])), s.potentials[0])
    A2, B2, C2 = extract_blocks(J)
    b2 = jnp.ones((A2.shape[0], 3))
    x_ge, info = native_banded_solve(A2, B2, C2, b2)
    assert bool(info["singular"]) is True
    assert bool(info["ok"]) is False


def test_ge_scan_backward_pass():
    """Backward pass through lax.scan back-substitution works."""
    n = 8
    A, B, C, b = _rand_banded(n, 5)
    A = jnp.array(A)
    B = jnp.array(B)
    C = jnp.array(C)
    b = jnp.array(b)

    def loss(Aflat):
        x, _ = native_banded_solve(Aflat.reshape(n, 3, 3), B, C, b)
        return jnp.sum(x)

    g = jax.grad(loss)(A.reshape(-1))
    assert jnp.all(jnp.isfinite(g))


def test_ge_block_dets_returned():
    """_block_thomas returns per-block determinant array for singularity check."""
    A, B, C, b = _rand_banded(10, 0)
    x_ge, A_mod, b_mod, dets = _block_thomas(A, B, C, b)
    assert dets.shape == (10,)
    assert jnp.all(jnp.isfinite(dets))
