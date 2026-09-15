"""Implicit Voc/MPP roots: secant-refined Voc + stationary MPP verification."""

import jax.numpy as jnp
import numpy as np
import pytest

import driftjax as dj


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
    """Coarse sweep Voc is secant-refined: bracketed flag set and an
    independent Newton solve at the reported Voc gives |J| ~ 0."""
    from driftjax.science.contacts import boundary_bias
    from driftjax.solvers.continuation import total_current
    from driftjax.solvers.newton import solve_newton
    from driftjax.units import thermal_scales

    sol = dj.simulate(homo_dev, dj.Sweep(n_steps=6, vmax=1.2))
    assert bool(sol.voc_bracketed) is True
    assert bool(jnp.isfinite(sol.voc))
    e = float(thermal_scales(float(sol.cell.T))["energy"])
    pot, st = solve_newton(
        sol.cell, boundary_bias(sol.cell, float(sol.voc) / e), sol.potentials[-1]
    )
    assert bool(st.get("converged", False)) is True
    j = abs(float(total_current(sol.cell, pot)))
    jscale = max(abs(float(x)) for x in sol.current) + 1e-30
    assert j / jscale < 1e-4, (j, jscale)
    # Refined value lies inside the sweep bracket.
    vv = [float(v) for v in sol.voltages]
    assert min(vv) <= float(sol.voc) <= max(vv)


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
