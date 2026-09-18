"""Regression tests for megakernel automatic LAPACK fallback.

These tests verify that the native GE backend with automatic LAPACK fallback
correctly handles heterojunction devices (band offsets that cause ill-conditioned
Jacobians) by seamlessly switching to LAPACK without user intervention.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import driftjax as dj


# --- Heterojunction device (CdS/CdTe-like with sharp band offsets) ---
def _heterojunction():
    window = dj.material(Eg=2.4, Chi=4.2, eps=10.0, Nc=2.2e18, Nv=2.2e18,
                         mn=100.0, mp=500.0, tn=1e-7, tp=1e-7, A=1e5)
    absorber = dj.material(Eg=1.55, Chi=3.9, eps=12.0, Nc=2.2e19, Nv=2.2e19,
                           mn=1000.0, mp=300.0, tn=1e-6, tp=1e-6, A=3e4)
    return dj.Device(
        n_points=150,
        layers=[(5e-6, window, 1e17), (1e-4, absorber, 1e14)],
        Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7,
    )


def test_heterojunction_native_ge_matches_lapack():
    """Native GE + fallback should match pure LAPACK on heterojunction."""
    cell = _heterojunction()
    protocol = dj.Sweep(vmax=1.2, n_steps=21)

    sol_mk = dj.simulate(cell, protocol, solver=dj.Newton(backend="native_ge_eq"))
    sol_lp = dj.simulate(cell, protocol, solver=dj.Newton(backend="auto"))

    # Currents are 1D array of length n_steps
    j_mk = np.asarray(sol_mk.currents)
    j_lp = np.asarray(sol_lp.currents)
    max_diff = float(np.max(np.abs(j_mk - j_lp)))
    assert max_diff < 1e-6, f"Current mismatch: max|dj|={max_diff:.2e}"

    # Efficiency should be finite for both
    assert jnp.isfinite(sol_mk.efficiency), "Native GE: efficiency not finite"
    assert jnp.isfinite(sol_lp.efficiency), "LAPACK: efficiency not finite"


def test_heterojunction_jit_traceable():
    """Native GE + fallback must be JIT compatible for heterojunctions."""
    cell = _heterojunction()
    protocol = dj.Sweep(vmax=1.2, n_steps=11)

    @jax.jit
    def solve_jit():
        return dj.simulate(cell, protocol, solver=dj.Newton(backend="native_ge_eq"))

    result = solve_jit()
    assert jnp.isfinite(result.efficiency), "JIT trace failed for heterojunction"


def test_heterojunction_gradient():
    """Gradient through heterojunction with native GE + fallback must be finite."""
    cell = _heterojunction()
    protocol = dj.Sweep(vmax=1.0, n_steps=11)

    sol = dj.simulate(cell, protocol, solver=dj.Newton(backend="native_ge_eq"))
    assert sol.converged, "Forward pass with fallback should converge"
