"""Level-9 (experimental comparison) — validated against published Si data.

A real gate (never skipped): a planar Si homojunction (1-sun AM1.5G, table
optics, SRH tau = 1e-6 s) must land inside the experimental band of
well-characterised silicon cells:

    Jsc  33-40 mA/cm2   (record lab Si ~42; commercial ~35-40; SQ limit ~44)
    Voc  0.60-0.75 V    (record lab 0.744; commercial 0.65-0.70)
    FF   0.72-0.84      (record ~0.84; typical 0.75-0.82)
    eta  15-24 %        (record 26.8 %; simple SRH model ~18-20)

Metrics are derived from the IV curve itself; the convenience fields are only
reliable when the sweep crosses zero, so the J-V array is the ground truth.
Both tests share one module-scoped simulation (the sweep is not re-run).
"""

import numpy as np
import pytest
from helpers import si_homojunction_device

import driftjax as dj
from driftjax.science.spectrum import spectrum


@pytest.fixture(scope="module")
def si_result():
    """One shared planar-Si homojunction sweep plus its derived metrics."""
    r = dj.simulate(
        si_homojunction_device(200).design(),
        dj.Sweep(vmax=0.75, n_steps=31),
        optics=dj.BeerLambert(alpha_mode="table"),
        ls=spectrum(normalize=True),
    )
    v = np.asarray(r.voltages)
    j = np.asarray(r.current)
    jsc = float(j[0]) * 1000.0  # A/cm2 -> mA/cm2
    p = v * j
    k = int(np.argmax(p))
    eta = float(p[k]) * 10000.0 / 1000.0  # W/m2 / (1000 W/m2)
    idx = np.where(j > 0)[0]
    i = idx[-1]
    voc = float(v[i] - j[i] * (v[i + 1] - v[i]) / (j[i + 1] - j[i]))
    ff = float(p[k]) / (voc * jsc * 0.001)
    return {"jsc": jsc, "voc": voc, "eta": eta, "ff": ff, "v": v, "j": j}


@pytest.mark.slow  # N=200 Si homojunction sweep fixture builds ~30s on CPU
def test_si_homojunction_experimental_band(si_result):
    """Planar Si homojunction matches published Jsc/Voc/FF/eta bands."""
    assert 33.0 < si_result["jsc"] < 40.0, (
        f"Jsc={si_result['jsc']:.2f} mA/cm2 outside experimental band"
    )
    assert 0.6 < si_result["voc"] < 0.75, f"Voc={si_result['voc']:.3f} V outside experimental band"
    assert 0.72 < si_result["ff"] < 0.84, f"FF={si_result['ff']:.3f} outside experimental band"
    assert 0.15 < si_result["eta"] < 0.24, f"eta={si_result['eta']:.2%} outside experimental band"


@pytest.mark.slow
def test_si_homojunction_max_power_inside_sweep(si_result):
    """Sanity: the MPP and zero crossing lie INSIDE the swept window, so
    the derived metrics are not extrapolations."""
    v, j = si_result["v"], si_result["j"]
    k = int(np.argmax(v * j))
    assert v[k] < 0.75 * 0.95, "MPP sits on the sweep edge — raise vmax"
    assert np.any(j > 0) and (not np.all(j > 0)), "no J-V zero crossing in sweep"
