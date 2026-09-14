"""Analytic adjoint per-bias kernel (single-kernel, zero-AD backward).

Pins ``analytic_g_x`` / ``analytic_u_cell`` / ``analytic_lamF_c`` against
the AD references (``jax.grad(total_current)``, ``jax.vjp(comp_F)`` with
the ``boundary_bias(cell, vb)`` path) on p-n, hetero and intrinsic
fixtures, including the n=3 edge case.
"""

import jax
import jax.numpy as jnp

import driftjax as dj
from driftjax.fields import pot2vec, vec2pot
from driftjax.numerics.analytic_adjoint import (
    analytic_g_x,
    analytic_lamF_c,
    analytic_per_bias,
    analytic_u_cell,
)
from driftjax.numerics.residual import comp_F
from driftjax.science.contacts import boundary_bias
from driftjax.science.spectrum import spectrum
from driftjax.simulator import init_cell
from driftjax.solvers.continuation import equilibrium_guess, total_current
from driftjax.solvers.newton import solve_eq, solve_newton


def _pn_cell(n=8, **srvs):
    m = dj.material(
        Chi=3.9,
        Eg=1.5,
        eps=9.4,
        Nc=8e17,
        Nv=1.8e19,
        mn=100,
        mp=100,
        tn=1e-8,
        tp=1e-8,
        A=1e4,
    )
    kw = dict(Snl=1e7, Snr=0, Spl=0, Spr=1e7)
    kw.update(srvs)
    dev = dj.Device(n_points=n, layers=[(1e-4, m, 1e17), (1e-4, m, -1e17)], **kw)
    return init_cell(dev.design(), spectrum(normalize=False))


def _hetero_cell(n=8):
    m1 = dj.material(
        Chi=4.5,
        Eg=2.4,
        eps=10.0,
        Nc=2e18,
        Nv=1e19,
        mn=50,
        mp=20,
        tn=1e-9,
        tp=1e-9,
        A=1e4,
    )
    m2 = dj.material(
        Chi=4.3,
        Eg=1.5,
        eps=9.4,
        Nc=8e17,
        Nv=1.8e19,
        mn=100,
        mp=60,
        tn=1e-8,
        tp=1e-8,
        A=2e4,
    )
    dev = dj.Device(
        n_points=n,
        layers=[(5e-5, m1, 1e17), (2e-4, m2, -1e16)],
        Snl=1e5,
        Snr=1e6,
        Spl=1e6,
        Spr=1e5,
    )
    return init_cell(dev.design(), spectrum(normalize=False))


def _biased_pot(cell, vb):
    peq = solve_eq(cell, boundary_bias(cell, 0.0), equilibrium_guess(cell).phi)
    pot, _ = solve_newton(cell, boundary_bias(cell, vb), peq, tol=1e-10, max_steps=40)
    return pot


def _rel_diff(a, b):
    return float(jnp.max(jnp.abs(a - b))) / max(1.0, float(jnp.max(jnp.abs(a))))


def test_g_x_matches_grad():
    cell = _pn_cell()
    pot = _biased_pot(cell, 0.2)
    ref = jax.grad(lambda x: total_current(cell, vec2pot(x)))(pot2vec(pot))
    got = analytic_g_x(cell, pot)
    assert _rel_diff(ref, got) < 1e-12


def test_u_cell_matches_grad():
    cell = _hetero_cell()
    pot = _biased_pot(cell, 0.5)
    ref = jax.grad(lambda c: total_current(c, pot))(cell)
    got = analytic_u_cell(cell, pot)
    for a, b in zip(jax.tree_util.tree_leaves(ref), jax.tree_util.tree_leaves(got), strict=False):
        if a.size:
            assert _rel_diff(a, b) < 1e-9


def test_lamF_c_matches_vjp_pn():
    cell = _pn_cell()
    pot = _biased_pot(cell, 0.2)
    lam = jax.random.normal(jax.random.PRNGKey(0), (3 * pot.n,))
    _, vjp = jax.vjp(lambda c: comp_F(c, boundary_bias(c, 0.2), pot), cell)
    ref = vjp(lam)[0]
    got = analytic_lamF_c(cell, pot, jnp.asarray(0.2), lam)
    paths = jax.tree_util.tree_flatten_with_path(ref)[0]
    for (path, a), b in zip(paths, jax.tree_util.tree_leaves(got), strict=False):
        if a.size:
            name = getattr(path[0], "name", str(path[0]))
            assert _rel_diff(a, b) < 1e-9, name


def test_lamF_c_matches_vjp_hetero_n3():
    cell = _hetero_cell(n=3)
    pot = _biased_pot(cell, 0.0)
    lam = jax.random.normal(jax.random.PRNGKey(3), (3 * pot.n,))
    _, vjp = jax.vjp(lambda c: comp_F(c, boundary_bias(c, 0.0), pot), cell)
    ref = vjp(lam)[0]
    got = analytic_lamF_c(cell, pot, jnp.asarray(0.0), lam)
    for a, b in zip(jax.tree_util.tree_leaves(ref), jax.tree_util.tree_leaves(got), strict=False):
        if a.size:
            assert _rel_diff(a, b) < 1e-9


def test_per_bias_matches_reference():
    cell = _pn_cell()
    pot = _biased_pot(cell, 0.2)
    lam = jax.random.normal(jax.random.PRNGKey(5), (3 * pot.n,))
    gcb = jnp.asarray(0.7)
    u_ref = jax.grad(lambda c: total_current(c, pot))(cell)
    _, vjp = jax.vjp(lambda c: comp_F(c, boundary_bias(c, 0.2), pot), cell)
    ref = jax.tree.map(lambda a, b: (a - b) * gcb, u_ref, vjp(lam)[0])
    got = analytic_per_bias(cell, pot, jnp.asarray(0.2), lam, gcb)
    for a, b in zip(jax.tree_util.tree_leaves(ref), jax.tree_util.tree_leaves(got), strict=False):
        if a.size:
            assert _rel_diff(a, b) < 1e-9
