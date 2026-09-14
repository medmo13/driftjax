"""L8 literature anchor — SCAPS-1D CdS/CdTe test."""

import pytest

import driftjax as dj
import driftjax.science
from driftjax.science.spectrum import spectrum


@pytest.fixture(scope="module")
def cdte_result():
    CdS = dj.material(
        Nc=2.2e18,
        Nv=1.8e19,
        Eg=2.4,
        eps=10,
        Et=0,
        mn=100,
        mp=25,
        tn=1e-08,
        tp=1e-13,
        Chi=4.0,
        A=10000.0,
    )
    CdTe = dj.material(
        Nc=8e17,
        Nv=1.8e19,
        Eg=1.5,
        eps=9.4,
        Et=0,
        mn=320,
        mp=40,
        tn=5e-09,
        tp=5e-09,
        Chi=3.9,
        A=10000.0,
    )
    des = dj.Device(
        layers=list(zip([2.5e-06, 0.0004], [CdS, CdTe], [1e17, -1000000000000000.0], strict=False)),
        n_points=500,
        Snl=11600000.0,
        Snr=11600000.0,
        Spl=11600000.0,
        Spr=11600000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()
    return dj.simulate(
        des,
        dj.Sweep(vmax=0.9, n_steps=19),
        optics=dj.BeerLambert(alpha_mode="beer-lambert"),
        ls=spectrum(normalize=False),
    )


@pytest.mark.slow  # N=500 CdS/CdTe heterojunction (fixture builds ~60s solve)
def test_scaps_cdte_voc(cdte_result):
    mv = float(cdte_result.voc) * 1000.0
    assert abs(mv - 891.0) < 8.0, f"Voc = {mv:.1f} mV"


@pytest.mark.slow
def test_scaps_cdte_jsc(cdte_result):
    jsc = float(cdte_result.current[0]) * 10000.0
    assert abs(jsc - 181.6) < 6.0, f"Jsc = {jsc:.1f} A/m² ({jsc / 100:.2f} mA/cm²)"


@pytest.mark.slow
def test_scaps_cdte_ff(cdte_result):
    ff = float(cdte_result.ff)
    assert abs(ff - 0.74) < 0.02, f"FF = {ff:.3f}"


@pytest.mark.slow
def test_scaps_cdte_eff(cdte_result):
    eff = float(cdte_result.efficiency) * 100.0
    assert abs(eff - 13.3) < 0.4, f"eff = {eff:.2f}%"
