"""Implicit Voc/MPP roots: secant-refined Voc + stationary MPP verification."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import driftjax as dj


def test_mpp_soft_bound_and_fd(homo_dev):
    """Soft-maximum MPP: bounded bias vs hard selection, FD-tight gradient.

    0 <= soft - hard <= tau*log(#candidates); and because the soft
    objective has no segment-selection jump, its adjoint gradient must
    match central FD closely (unlike the hard objective near a switch).
    """

    from driftjax.simulator import _mpp

    sol = dj.simulate(homo_dev, dj.Sweep(n_steps=12, vmax=1.2))
    v = np.asarray(sol.voltages, dtype=float)
    j = np.asarray(sol.current, dtype=float)
    p_hard, _ = _mpp(v, j)
    tau = 1e-4
    p_soft, v_soft = _mpp(v, j, tau=tau)
    n_cand = 5 * (len(v) - 1)
    assert 0.0 <= float(p_soft - p_hard) <= tau * np.log(n_cand) + 1e-12
    assert float(v[0]) <= float(v_soft) <= float(v[-1])

    # Adjoint vs FD on the soft objective through an Eg rebuild knob.
    def _e(eg):
        m = dj.material(Eg=eg, Chi=3.0, eps=10.0, Nc=1e18, Nv=1e18, mn=130.0, mp=160.0, A=2e4)
        dev = dj.Device(
            n_points=15,
            layers=[(1e-4, m, 1e17), (1e-4, m, -1e17)],
            Snl=1e7,
            Snr=0.0,
            Spl=0.0,
            Spr=1e7,
        )
        return dj.simulate(dev, dj.Sweep(n_steps=8, vmax=1.2, mpp_tau=1e-4)).efficiency

    g_adj = float(jax.grad(_e)(1.4))
    h = 1e-4
    g_fd = (float(_e(1.4 + h)) - float(_e(1.4 - h))) / (2 * h)
    rel = abs(g_adj - g_fd) / (abs(g_fd) + 1e-30)
    assert rel < 0.05, (g_adj, g_fd, rel)


@pytest.fixture(scope="module")
def homo_dev():
    mat = dj.material(Eg=1.4, Chi=3.0, eps=10.0, Nc=1e18, Nv=1e18, mn=130.0, mp=160.0, A=2e4)
    return dj.Device(
        n_points=25,
        layers=[(1e-4, mat, 1e17), (1e-4, mat, -1e17)],
        Snl=1e7,
        Snr=0.0,
        Spl=0.0,
        Spr=1e7,
    )


def test_voc_bracketed_and_root(homo_dev):
    """Coarse sweep Voc is bisection-refined on the sweep branch: bracketed
    flag set, value inside the bracket, and an independent Newton solve
    warm-started from the NEAREST SWEEP state (continuation history stays
    on-branch; foreign guesses can collapse onto a spurious near-zero
    branch) gives |J| ~ 0 with a converged residual."""
    from driftjax.science.contacts import boundary_bias
    from driftjax.solvers.continuation import total_current
    from driftjax.solvers.newton import solve_newton
    from driftjax.units import thermal_scales

    sol = dj.simulate(homo_dev, dj.Sweep(n_steps=6, vmax=1.2))
    assert bool(sol.voc_bracketed) is True
    assert bool(jnp.isfinite(sol.voc))
    e = float(thermal_scales(float(sol.cell.T))["energy"])
    vv = [float(v) for v in sol.voltages]
    assert min(vv) <= float(sol.voc) <= max(vv)
    ia = int(min(range(len(vv)), key=lambda i: abs(vv[i] - float(sol.voc))))
    pot, st = solve_newton(
        sol.cell, boundary_bias(sol.cell, float(sol.voc) / e), sol.potentials[ia]
    )
    assert bool(st.get("converged", False)) is True
    j = abs(float(total_current(sol.cell, pot)))
    jscale = max(abs(float(x)) for x in sol.current) + 1e-30
    assert j / jscale < 1e-4, (j, jscale)


def test_voc_unbracketed_nan_and_flag(homo_dev):
    """No sign change -> Voc NaN with bracketed False (never a fake value)."""
    sol = dj.simulate(homo_dev, dj.Sweep(n_steps=5, vmax=0.5))
    assert bool(jnp.isnan(sol.voc))
    assert bool(sol.voc_bracketed) is False


def test_mpp_stationary_on_interpolant(homo_dev):
    """The reported Vmpp satisfies dP/dV ~= 0 on the PCHIP interpolant
    (continuous local root condition), or sits at a sweep endpoint."""
    from driftjax.numerics.spline import pchip_coefs
    from driftjax.simulator import _mpp

    sol = dj.simulate(homo_dev, dj.Sweep(n_steps=12, vmax=1.2))
    v = np.asarray(sol.voltages, dtype=float)
    j = np.asarray(sol.current, dtype=float)
    pmax, vmpp = _mpp(v, j)
    pmax, vmpp = float(pmax), float(vmpp)
    assert pmax >= float(np.max(v * j)) - 1e-12 * abs(pmax)
    if v[0] < vmpp < v[-1]:
        p = v * j
        a, b, c, _ = (
            np.asarray(x, dtype=float) for x in pchip_coefs(jnp.asarray(v), jnp.asarray(p))
        )
        i = int(np.searchsorted(v, vmpp) - 1)
        i = min(max(i, 0), len(v) - 2)
        dx = vmpp - v[i]
        dp = 3 * a[i] * dx**2 + 2 * b[i] * dx + c[i]
        scale = abs(pmax) / max(v[-1] - v[0], 1e-30) + 1e-30
        assert abs(dp) / scale < 1e-6, (dp, scale, vmpp)


def test_voc_implicit_gradient_vs_fd():
    """VJP through the implicit root: dVoc/dEg (adjoint) vs central FD.

    Exercises the gated implicit term (bracketed homo sweep): the old
    vmax-substitution path is stopped, so this number comes purely from
    -J_theta/J_V contracted through the existing adjoint.
    """

    def voc_of_eg(eg):
        m = dj.material(Eg=eg, Chi=3.0, eps=10.0, Nc=1e18, Nv=1e18, mn=130.0, mp=160.0, A=2e4)
        dev = dj.Device(
            n_points=15,
            layers=[(1e-4, m, 1e17), (1e-4, m, -1e17)],
            Snl=1e7,
            Snr=0.0,
            Spl=0.0,
            Spr=1e7,
        )
        return dj.simulate(dev, dj.Sweep(n_steps=8, vmax=1.2)).voc

    eg0 = 1.4
    g_adj = float(jax.grad(voc_of_eg)(eg0))
    h = 1e-4
    vp = float(voc_of_eg(eg0 + h))
    vm = float(voc_of_eg(eg0 - h))
    g_fd = (vp - vm) / (2 * h)
    assert abs(vp - vm) > 0.0  # FD actually moves (bracketed both sides)
    rel = abs(g_adj - g_fd) / (abs(g_fd) + 1e-30)
    assert rel < 0.05, (g_adj, g_fd, rel)
