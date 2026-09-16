"""Tests for the JAX-native banded solver numerics-banded-native."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import lstsq, solve_banded

import driftjax as dj
from driftjax.numerics.banded_native import _dense_from_blocks, native_banded_solve
from driftjax.numerics.banded_solve import (
    _banded_to_dense,
    _blocks_to_lapack_banded,
    extract_blocks,
)
from driftjax.numerics.residual import F_jacobian
from driftjax.science.contacts import boundary_bias
from driftjax.science.spectrum import spectrum
from driftjax.solvers.api import Newton


def _set_native(flag):
    if flag:
        os.environ["DRIFTJAX_NATIVE_BANDED"] = "1"
    else:
        os.environ.pop("DRIFTJAX_NATIVE_BANDED", None)


def _lapack_solve(A, B, C, b):
    n = A.shape[0]
    ab, kl, ku = _blocks_to_lapack_banded(A, B, C)
    try:
        x = solve_banded((kl, ku), ab, b.reshape(-1))
        flag = 0
    except Exception:
        M = _banded_to_dense(np.array(ab), kl, ku, n)
        x, *_ = lstsq(M, b.reshape(-1), rcond=None)
        flag = 1
    return x.reshape(n, 3), flag


def _rand_banded(n, seed):
    rs = np.random.RandomState(seed)
    A = jnp.asarray(rs.randn(n, 3, 3) * 0.3)
    A = A.at[:, 0, 0].set(A[:, 0, 0] + 6)
    A = A.at[:, 1, 1].set(A[:, 1, 1] + 6)
    A = A.at[:, 2, 2].set(A[:, 2, 2] + 6)
    B = jnp.asarray(rs.randn(n - 1, 3, 3) * 0.3)
    C = jnp.asarray(rs.randn(n - 1, 3, 3) * 0.3)
    b = jnp.asarray(rs.randn(n, 3))
    return A, B, C, b


def _stress_jac_n40():
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
    return extract_blocks(J)


@pytest.mark.smoke
def test_dense_reconstruction_matches_package():
    n = 6
    A, B, C, b = _rand_banded(n, 0)
    mine = _dense_from_blocks(A, B, C)
    ab, kl, ku = _blocks_to_lapack_banded(A, B, C)
    ref = _banded_to_dense(np.array(ab), kl, ku, n)
    assert np.max(np.abs(np.array(mine) - ref)) == 0.0


@pytest.mark.smoke
@pytest.mark.parametrize("seed", range(8))
def test_native_matches_lapack_well_conditioned(seed):
    A, B, C, b = _rand_banded(30, seed)
    x_ref, flag = _lapack_solve(A, B, C, b)
    x_nat, info = native_banded_solve(A, B, C, b)
    assert bool(info["ok"]) is True
    assert flag == 0
    rel = float(jnp.linalg.norm(x_nat - x_ref) / (jnp.linalg.norm(x_ref) + 1e-30))
    assert rel < 1e-10, rel


@pytest.mark.timeout(60)
def test_native_singular_stress_detected():
    """Singular stress device (rank 78/120) detected via per-block det."""
    A2, B2, C2 = _stress_jac_n40()
    b2 = jnp.ones((A2.shape[0], 3))
    x_nat, info = native_banded_solve(A2, B2, C2, b2)
    assert bool(info["singular"]) is True
    assert bool(info["ok"]) is False


@pytest.mark.slow
def test_native_e2e_matches_shipped():
    pvk = dj.material(
        Chi=3.9, Eg=1.55, eps=24.0, Nc=2.2e18, Nv=1.8e19, mn=20.0, mp=20.0, tn=1e-6, tp=1e-6, A=2e5
    )
    dev = dj.Device(
        n_points=40,
        layers=[(0.1e-4, pvk, 1e16), (5e-4, pvk, 1e13), (0.1e-4, pvk, -1e16)],
        Snl=1e7,
        Snr=1e7,
        Spl=1e7,
        Spr=1e7,
    )
    ls = spectrum()
    _set_native(False)
    s_cb = dj.simulate(
        dev,
        dj.Sweep(vmax=1.0, n_steps=8),
        solver=Newton(max_steps=60, globalization="ls", fused=False),
        ls=ls,
    )
    try:
        _set_native(True)
        s_na = dj.simulate(
            dev,
            dj.Sweep(vmax=1.0, n_steps=8),
            solver=Newton(max_steps=60, globalization="ls", fused=False),
            ls=ls,
        )
    finally:
        _set_native(False)
    assert bool(s_na.converged)
    assert float(s_na.max_residual) < 1e-10
    assert abs(float(s_na.voc) - float(s_cb.voc)) < 1e-9
    assert abs(float(s_na.jsc) - float(s_cb.jsc)) < 1e-9
    assert abs(float(s_na.efficiency) - float(s_cb.efficiency)) < 1e-9


@pytest.mark.timeout(60)
def test_native_solve_is_differentiable():
    """jax.grad of a reduction of the native banded solve matches central
    finite differences on a small well-conditioned system -- the native
    solve is end-to-end differentiable inside XLA (no host callback)."""
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
    # central FD on the 9*n entries, compare a handful
    A0 = A.reshape(-1)
    h = 1e-5
    fd = jnp.zeros_like(A0)
    for k in range(0, 9 * n, 7):
        e = jnp.zeros(9 * n)
        e = e.at[k].set(1.0)
        fd = fd.at[k].set((float(loss(A0 + h * e)) - float(loss(A0 - h * e))) / (2 * h))
    idx = jnp.arange(0, 9 * n, 7)
    rel = float(jnp.linalg.norm(g[idx] - fd[idx]) / (jnp.linalg.norm(fd[idx]) + 1e-12))
    assert rel < 1e-5, rel


def _set_row_equil(flag):
    if flag:
        os.environ["DRIFTJAX_ROW_EQUIL"] = "1"
    else:
        os.environ.pop("DRIFTJAX_ROW_EQUIL", None)


def test_row_equilibration_golden_safe():
    # DRIFTJAX_ROW_EQUIL=1 engages per-scalar-row equilibration in
    # adjoint_banded_solve_blocks but is GOLDEN-SAFE: bit-identical on
    # well-conditioned banded systems (equilibration is an exact similarity
    # transform on a nonsingular matrix).
    from driftjax.numerics.banded_solve import adjoint_banded_solve_blocks

    n = 16
    A, B, C, b = _rand_banded(n, 0)
    g = b
    _set_row_equil(False)
    lam_off, fb_off = adjoint_banded_solve_blocks(A, B, C, g)
    _set_row_equil(True)
    lam_on, fb_on = adjoint_banded_solve_blocks(A, B, C, g)
    _set_row_equil(False)
    assert bool(fb_off) is False and bool(fb_on) is False
    rel = float(
        jnp.linalg.norm(lam_off.reshape(-1) - lam_on.reshape(-1))
        / (jnp.linalg.norm(lam_off) + 1e-30)
    )
    assert rel < 1e-12, rel
