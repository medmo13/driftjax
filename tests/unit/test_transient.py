"""Pins for the transient / small-signal machinery (solvers/transient.py).

These tests make the module docstring's claims *checked* claims:

  * per-step residual of the backward-Euler solver is below tolerance,
  * a step taken from an exact DC root at fixed bias stays put (fixed-point
    consistency),
  * the AC admittance's ω→0 real limit reproduces the differential
    conductance read off the same converged sweep — the property that fails
    loudly if the perturbation RHS stops coming from the applied-bias
    channel.

Conventions that matter:
  * solver voltages are in Vt = kT/q units (V ≈ 0.36 V ⇔ v ≈ 13.9!);
  * operating points MUST lie on the sweep's own voltage grid — `at_bias`
    snaps to the nearest stored bias, and pairing that state with a bound
    at a different voltage produces O(1) residuals;
  * unit conversion: dI_raw/dv_dimless = (dJ_phys/dV_phys) · (scE / scI).
"""

import jax.numpy as jnp
import numpy as np
import pytest

import driftjax as dj
from driftjax import Equilibrium, Newton, Sweep, simulate
from driftjax.fields import pot2vec
from driftjax.numerics.scharfetter_gummel import Jn, Jp
from driftjax.science.contacts import boundary_bias
from driftjax.solvers.newton import solve_newton
from driftjax.solvers.transient import (
    ac_small_signal,
    solve_transient_step,
    transient_residual,
)
from driftjax.units import thermal_scales

N = 40


@pytest.fixture(scope="module")
def si_ops():
    Si = dj.load_material("Si")
    dev = dj.Device(n_points=N,
                    layers=[(5e-5, Si, 1e16), (5e-5, Si, -1e16)],
                    Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7)
    eq = simulate(dev, Equilibrium(), solver=Newton())
    # dense grid; we always operate on exact grid points
    sol = simulate(dev, Sweep(vmax=0.6, n_steps=16), solver=Newton())
    sc = thermal_scales(300.0)
    return eq.cell, dev, sol, float(sc["energy"]), float(sc["current"])


def _on_grid_index(sol, v_v):
    """Index whose STORED voltage equals v_v exactly (grid must contain it)."""
    v_grid = np.asarray(sol.voltages)
    k = int(np.argmin(np.abs(v_grid - v_v)))
    assert abs(v_grid[k] - v_v) < 1e-12, \
        f"v={v_v} not on sweep grid (nearest {v_grid[k]})"
    return k


def test_sweep_states_are_exact_roots(si_ops):
    """Meta-pin: stored per-bias potentials satisfy comp_F(bound(v_i)) = 0."""
    cell, _, sol, scE, _ = si_ops
    for k in (1, len(sol.voltages) // 2, len(sol.voltages) - 1):
        v_v = float(sol.voltages[k])
        b = boundary_bias(cell, jnp.asarray(v_v / scE))
        r = float(jnp.max(jnp.abs(comp_F(cell, b, sol.at_bias(v_v)))))
        assert r < 1e-10, f"bias {v_v:.2f} V: residual {r:.2e}"


from driftjax.numerics.residual import comp_F  # noqa: E402  (used above via closure)


def test_transient_step_residual_below_tolerance(si_ops):
    cell, _, sol, scE, _ = si_ops
    k = _on_grid_index(sol, 0.36)
    v_dim = float(sol.voltages[k]) / scE
    pot_dc = sol.at_bias(float(sol.voltages[k]))
    pot_new, info = solve_transient_step(cell, boundary_bias(cell, jnp.asarray(v_dim)),
                                         pot_dc, dt=1e3)
    assert info["converged"], info
    F_t = transient_residual(cell, boundary_bias(cell, jnp.asarray(v_dim)),
                             pot_new, pot_dc, dt=1e3)
    assert float(jnp.max(jnp.abs(F_t))) < 1e-6


def test_transient_fixed_point_at_dc_root(si_ops):
    """A backward-Euler step from an exact DC root at FIXED bias must stay
    there when dt is large (the C·(X−X_prev)/dt mass term vanishes)."""
    cell, _, sol, scE, _ = si_ops
    k = _on_grid_index(sol, 0.36)
    v_dim = float(sol.voltages[k]) / scE
    pot_dc = sol.at_bias(float(sol.voltages[k]))
    pot_new, info = solve_transient_step(cell, boundary_bias(cell, jnp.asarray(v_dim)),
                                         pot_dc, dt=1e12)
    assert info["converged"], info
    d_max = float(jnp.max(jnp.abs(pot2vec(pot_new) - pot2vec(pot_dc))))
    assert d_max < 1e-6, f"DC root drifted by {d_max:.2e}"


def test_ac_omega_zero_matches_didv(si_ops):
    """THE pin: Re Y(ω→0) equals dI/dV from warm-started re-solves.

    The DD Jacobian is ill-conditioned (k ~ 1e14), so "low frequency" means
    REALLY low: Re Y has only reached its DC asymptote for w <~ 1e-8 here.
    The reference is a central finite difference of terminal currents from
    Newton re-solves started AT the operating point (warm-started solves
    converge in a few steps and stay on the sweep's solution branch).
    """
    cell, dev, sol, scE, scI = si_ops
    k = _on_grid_index(sol, 0.36)
    v_v = float(sol.voltages[k])
    v_dim = v_v / scE
    pot_dc = sol.at_bias(v_v)

    omega = jnp.array([1e-8, 1e-10, 1e-12])
    Y = np.asarray(ac_small_signal(cell, jnp.asarray(v_dim), pot_dc, omega))
    g_ac = float(np.real(Y[0]))
    # DC plateau must be flat across those frequencies
    assert (np.max(np.real(Y)) - np.min(np.real(Y))) / max(abs(g_ac), 1e-30) < 1e-3

    h = 2e-3  # dimensionless voltage step
    xa = solve_newton(cell, boundary_bias(cell, jnp.asarray(v_dim + h)), pot_dc)[0]
    xb = solve_newton(cell, boundary_bias(cell, jnp.asarray(v_dim - h)), pot_dc)[0]
    i_a = float(jnp.mean(Jn(cell, xa) + Jp(cell, xa)))
    i_b = float(jnp.mean(Jn(cell, xb) + Jp(cell, xb)))
    g_ref = (i_a - i_b) / (2.0 * h)

    rel = abs(g_ac - g_ref) / max(abs(g_ref), 1e-30)
    assert rel < 5e-3, f"Re Y(w->0)={g_ac:.6e} vs dI/dv={g_ref:.6e} (rel {rel:.2e})"


def test_ac_finite(si_ops):
    cell, _, sol, scE, _ = si_ops
    omega = jnp.array([1e-2, 1e0, 1e2, 1e4])
    k = _on_grid_index(sol, 0.36)
    Y = np.asarray(ac_small_signal(cell, float(sol.voltages[k]) / scE,
                                   sol.at_bias(float(sol.voltages[k])), omega))
    assert np.all(np.isfinite(Y.view(np.float64))), "non-finite admittance"
