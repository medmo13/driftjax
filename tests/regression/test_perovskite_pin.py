"""Perovskite p-i-n positive-control regression (R-audit finding).

The 3-layer n-p-n stress device (validation/solver_fallback.py) is a
singular-matrix stress test, NOT a solar cell -- it converges to ~1e-8
but Voc=nan and Jsc~0 because the n-p-n stack with electron-selective
contacts has no hole-collection path.

This test pins the *correct* counterpart: a structurally-sound perovskite
p-i-n (non-selective ohmic contacts) that converges to machine precision
and shows the textbook diode J-V.  It rules out the interpretation that
"the solver cannot handle perovskite parameters."

Archived record: docs/paper/records/perovskite_p-i-n_audit.json
"""

import warnings

import jax.numpy as jnp
import pytest

import driftjax as dj
from driftjax.science.spectrum import spectrum


@pytest.fixture(scope="module")
def _pin_result():
    pvk = dj.material(
        Eg=1.55,
        Chi=3.9,
        eps=24.0,
        Nc=2.2e18,
        Nv=1.8e19,
        mn=20.0,
        mp=20.0,
        A=2e5,
        tn=1e-6,
        tp=1e-6,
    )
    dev = dj.Device(
        n_points=120,
        layers=[(0.1e-4, pvk, 1e16), (5e-4, pvk, 1e13), (0.1e-4, pvk, -1e16)],
        Snl=1e7,
        Snr=1e7,
        Spl=1e7,
        Spr=1e7,
    )
    ls = spectrum(normalize=False)  # raw AM1.5G = 899.9168 W/m2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = dj.simulate(
            dev,
            dj.Sweep(n_steps=5, vmax=1.0),
            optics=dj.BeerLambert(),
            solver=dj.Newton(max_steps=400),
            ls=ls,
        )
    return s


@pytest.mark.slow
def test_perovskite_pin_converges(_pin_result):
    s = _pin_result
    assert bool(s.converged), f"max|F|={float(s.max_residual):.2e}"
    assert float(s.max_residual) < 1e-10, float(s.max_residual)
    assert not any(bool(x) for x in s.fallback_used)


@pytest.mark.slow
def test_perovskite_pin_physics(_pin_result):
    s = _pin_result
    assert abs(float(s.voc) - 0.9699) < 0.01, f"Voc={float(s.voc):.4f}"
    assert abs(float(s.jsc) * 1e3 - 14.78) < 0.5, f"Jsc={float(s.jsc) * 1e3:.2f} mA/cm2"
    assert abs(float(s.ff) - 0.657) < 0.02, f"FF={float(s.ff):.4f}"
    assert abs(float(s.eff) - 0.1046) < 0.005, f"PCE={float(s.eff) * 100:.2f}%"
    # Jsc in forward direction (collected carriers)
    assert float(s.jsc) > 0, "Jsc must be positive (forward collection)"
    # Voc within [0, vmax]
    assert 0.0 <= float(s.voc) <= 1.0 + 1e-6


@pytest.mark.slow
def test_perovskite_pin_iv_shape(_pin_result):
    """J(0) > 0 (forward), J crosses zero near Voc, J(Vmax) < 0 (diode quadrant)."""
    s = _pin_result
    js = jnp.array(list(s.current))  # A/cm2
    assert float(js[0]) > 0, f"J(0)={float(js[0]):.4e} should be positive"
    assert float(js[-1]) < 0, f"J(Vmax)={float(js[-1]):.4e} should be negative"
