"""Heterojunction builder-threaded gradient pin (review N3).

Two specified behaviours:

1. Post-hoc mutation of the cached ``Ns``/``Ls`` leaves is INERT:
   ``Device.design()`` returns the design built at construction, so the
   simulation (and its gradient) is bit-identical after mutation.
2. Builder-threaded scalars (here: absorber thickness scale, the same
   coordinate as the manuscript Taylor study) carry NONZERO adjoint
   gradients that agree with central FD in sign and magnitude.

Tiny mesh (N=25, 5 biases) keeps this in the gradient tier, not smoke.
"""

import equinox as eqx
import jax
import numpy as np

import driftjax as dj
from examples.support import ex2_device

N_POINTS = 25
N_STEPS = 5
BASE_ABS_W = 4e-4  # cm, matches ex2_device absorber width


def device_from_thickness(s, n_points=N_POINTS):
    CdS = dj.material(
        Nc=2.2e18,
        Nv=1.8e19,
        Eg=2.4,
        eps=10,
        Et=0,
        mn=100,
        mp=25,
        tn=1e-8,
        tp=1e-13,
        Chi=4.0,
        A=1e4,
    )
    CdTe = dj.material(
        Nc=8e17,
        Nv=1.8e19,
        Eg=1.5,
        eps=9.4,
        Et=0,
        mn=320,
        mp=40,
        tn=5e-9,
        tp=5e-9,
        Chi=3.9,
        A=1e4,
    )
    return dj.Device(
        n_points=n_points,
        layers=[(2.5e-6, CdS, 1e17), (s * BASE_ABS_W, CdTe, -1e15)],
        Snl=1.16e7,
        Snr=1.16e7,
        Spl=1.16e7,
        Spr=1.16e7,
    )


def _eff(dev):
    return float(dj.simulate(dev, dj.Sweep(vmax=0.9, n_steps=N_STEPS)).efficiency)


def test_posthoc_ns_mutation_is_inert():
    dev = ex2_device(n_points=N_POINTS)
    e0 = _eff(dev)
    mutated = eqx.tree_at(lambda d: d.Ns, dev, dev.Ns * 10.0)
    e1 = _eff(mutated)
    assert e0 == e1, f"Ns mutation changed output: {e0} vs {e1}"


def test_builder_thickness_grad_nonzero_matches_fd():
    def f(s):
        return dj.simulate(device_from_thickness(s), dj.Sweep(vmax=0.9, n_steps=N_STEPS)).efficiency

    s0 = 1.0
    g = float(jax.grad(f)(s0))
    assert g != 0.0 and np.isfinite(g), f"builder grad is zero/nonfinite: {g}"
    h = 1e-3
    fd = (float(f(s0 + h)) - float(f(s0 - h))) / (2 * h)
    assert np.sign(fd) == np.sign(g), f"sign mismatch: adj {g}, fd {fd}"
    assert abs(fd - g) / (abs(g) + 1e-30) < 0.05, f"adj {g} vs fd {fd}"
