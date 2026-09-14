"""L8 literature anchor — SCAPS-1D CdS/CdTe test."""

import pytest

import driftjax as dj
import driftjax.science
from driftjax.science.spectrum import spectrum


@pytest.mark.slow  # literature anchor: 2x N=80 forward solves (Voc + trend)
def _meas(tn_tp, Br):
    mat = dj.material(
        Chi=3.9,
        Eg=1.5,
        eps=9.4,
        Nc=8e17,
        Nv=1.8e19,
        mn=100,
        mp=100,
        tn=tn_tp,
        tp=tn_tp,
        Br=Br,
        A=10000.0,
    )
    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [mat, mat], [1e17, -1e17], strict=False)),
        n_points=80,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()
    r = dj.simulate(
        des,
        dj.Sweep(vmax=1.2, n_steps=10),
        optics=dj.BeerLambert(alpha_mode="beer-lambert"),
        ls=spectrum(normalize=False),
    )
    return (
        float(r.voc),
        float(r.current[0]) * 10000.0,
        float(r.ff),
        float(r.efficiency) * 100.0,
    )


@pytest.mark.slow
def test_ideal_diode_anchor():
    voc, jsc, ff, eff = _meas(1e-08, 1e-10)
    assert abs(voc - 1.032) * 1000.0 < 30.0, f"Voc = {voc * 1000:.1f} mV"
    assert abs(ff - 0.849) < 0.06, f"FF = {ff:.3f}"
    assert abs(eff - 16.0) < 0.5, f"eff = {eff:.2f}%"
    assert abs(jsc - 163.0) < 8.0, f"Jsc = {jsc:.1f} A/m²"


@pytest.mark.slow
def test_recombination_physics_trend():
    """More radiative recombination must lower Voc (physical direction)."""
    voc_low_br = _meas(1e-06, 1e-06)[0]
    voc_high_br = _meas(1e-08, 1e-10)[0]
    assert voc_high_br > voc_low_br + 0.001
