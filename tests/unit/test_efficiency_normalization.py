"""Efficiency normalization gate (referee Step 29).

efficiency must equal pmax/Pin computed from the SAME incident spectrum
used for generation. Consequence: eff * sum(P_in) / pmax is invariant
across spectrum normalizations (raw ~899.9 W/m^2 vs normalized 1000).
A truncated or inconsistent Pin would break this invariant --- the exact
figure-level failure mode documented in the supplement (20.00% with
Pin = 89.94, not 100 mW/cm^2).
"""

import jax.numpy as jnp

import driftjax as dj
from driftjax.science.spectrum import spectrum
from examples.support import ex1_device

pytestmark = __import__("pytest").mark.slow  # two small sweeps


def _k(sol):
    return float(sol.eff * jnp.sum(sol.P_in) / sol.pmax)


def test_efficiency_uses_own_pin():
    dev = ex1_device(n_points=60)
    sols = [
        dj.simulate(dev, dj.Sweep(vmax=1.1, n_steps=11), ls=spectrum(normalize=flag))
        for flag in (False, True)
    ]
    pins = [float(jnp.sum(s.P_in)) for s in sols]
    assert abs(pins[0] - 899.9) < 5.0, pins  # raw table convention
    assert abs(pins[1] - 1000.0) < 1e-9, pins
    k = [_k(s) for s in sols]
    assert abs(k[0] - k[1]) / abs(k[0]) < 1e-9, k
    for s in sols:
        assert 0.0 < float(s.eff) < 1.0
