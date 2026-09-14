"""Reference IV anchors: golden-file regression gates.

Three levels of IV-curve regression:
  * a symmetric Si p-n canary with pinned Jsc/Voc bands (physics anchor),
  * the N=500 golden reference file (``resources/golden_reference_iv.json``)
    reproduced to roundoff at N=500 (release gate, marked slow),
  * fast structural checks of the same geometry at N=200/120.

The golden files pin the exact float64 physics of the current release; any
change that perturbs the converged IV curve beyond tolerance fails here.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

import driftjax as dj
from driftjax.science.spectrum import spectrum

_GOLDEN = Path(__file__).parent.parent / "resources" / "golden_previous_iv.json"
JSC_REF = 174.27  # A/m2
VOC_REF = 0.494  # V


def _reference_design(n=500):
    """Thin-junction p-n reference geometry (golden file provenance)."""
    mat = dj.material(
        Chi=3.9,
        Eg=1.5,
        eps=9.4,
        Nc=8e17,
        Nv=1.8e19,
        mn=100,
        mp=100,
        Et=0,
        tn=1e-08,
        tp=1e-08,
        A=10000.0,
    )
    junction, width = (5e-06, 3e-04)
    return dj.Device(
        layers=list(
            zip(
                [junction, width - junction],
                [mat, mat],
                [1e17, -1e15],
                strict=False,
            )
        ),
        n_points=n,
        Snl=1e7,
        Snr=0,
        Spl=0,
        Spr=1e7,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()


def _canary_design(n=200):
    si = dj.load_material("Si")
    return dj.Device(
        layers=list(zip([0.0001, 0.0001], [si, si], [1e17, -1e17], strict=False)),
        n_points=n,
        Snl=1e7,
        Snr=1e7,
        Spl=1e7,
        Spr=1e7,
    ).design()


@pytest.mark.slow
def test_canary_anchor():
    """Flagship canary anchor: symmetric Si p-n design, 1000 W/m2 spectrum,
    smoothed-tauc optics -> Jsc~174 A/m2, Voc~0.49 V (re-pinned to current
    float64 / corrected-adjoint DriftJax physics; 5% relative tolerance)."""
    res = dj.simulate(_canary_design(), dj.Sweep(vmax=1.2, n_steps=25), ls=spectrum())
    jsc = float(jnp.abs(res.current[0]) * 10000.0)
    voc = float(res.voc)
    ff = float(res.ff)
    eff = float(res.efficiency)
    assert abs(jsc - JSC_REF) / JSC_REF < 0.05, jsc
    assert abs(voc - VOC_REF) / VOC_REF < 0.05, voc
    assert 0.6 <= ff <= 0.95, ff
    assert 0.0 < eff < 0.337, eff


@pytest.mark.slow
def test_golden_reference_n500_roundoff():
    """N=500 golden gate: reproduce ``golden_reference_iv.json`` currents to
    max|dj| < 1e-8 A/cm2 (the historical rtol = 1e-5 contract)."""
    data = json.loads(_GOLDEN.read_text())
    res = dj.simulate(
        _reference_design(500),
        dj.Sweep(vmax=0.95, n_steps=20),
        optics=dj.BeerLambert(alpha_mode="beer-lambert"),
        ls=spectrum(normalize=False),
    )
    v, j = res.iv_curve()
    v_ref = jnp.array(data["v"])
    j_ref = jnp.array(data["original"])
    assert jnp.allclose(v, v_ref, atol=1e-12, rtol=1e-06), (
        f"voltage grid mismatch: {float(v[0])}..{float(v[-1])}"
    )
    err = float(jnp.max(jnp.abs(j - j_ref)))
    assert err < 1e-08, f"golden reference mismatch max|dj|={err:.3e}"


def test_golden_reference_n200_fast():
    """Fast (~15 s) N=200 check of the same reference: shape, sign,
    monotonicity, |dj| < 3e-3 vs the N=500 gate above."""
    res = dj.simulate(
        _reference_design(200),
        dj.Sweep(vmax=0.95, n_steps=20),
        optics=dj.BeerLambert(alpha_mode="beer-lambert"),
        ls=spectrum(normalize=False),
    )
    v, j = res.iv_curve()
    assert v.shape == (20,)
    assert jnp.allclose(v, jnp.linspace(0.0, 0.95, 20), atol=1e-12)
    assert float(j[0]) > 0 and float(j[-1]) < 0
    assert float(jnp.min(jnp.diff(j))) < 0
    assert 0.7 < float(res.voc) < 1.0
    assert float(res.efficiency) > 0.05
    data = json.loads(_GOLDEN.read_text())
    j_500 = jnp.array(data["original"])
    err = float(jnp.max(jnp.abs(j - j_500)))
    assert err < 0.003, f"N=200 too far from the N=500 golden reference: {err:.3e}"


@pytest.mark.slow
def test_iv_tauc_optics_variant():
    """Same reference geometry under the flagship conventions (1000 W/m2
    spectrum, smoothed-tauc optics) against its own golden column."""
    des = _reference_design(500)
    res = dj.simulate(
        des,
        dj.Sweep(vmax=0.95, n_steps=20),
        optics=dj.BeerLambert(alpha_mode="tauc"),
        ls=spectrum(),
    )
    j_phys = res.current
    data = json.loads(_GOLDEN.read_text())
    j_golden = jnp.array(data["files_pad"])
    err = float(jnp.max(jnp.abs(j_phys - j_golden)))
    assert err < 0.005, f"tauc-optics golden mismatch max|dj|={err:.3e}"


def test_solution_iv_convention():
    """API convention: ``iv_curve()`` returns (volts, A/cm2) and is consistent
    with the dimensionless pair via the module unit scales."""
    # Convention-only gate (no golden values): N=60/12 steps suffices.
    res = dj.simulate(
        _reference_design(60),
        dj.Sweep(vmax=0.95, n_steps=12),
        optics=dj.BeerLambert(alpha_mode="beer-lambert"),
        ls=spectrum(normalize=False),
    )
    v, j = res.iv_curve()
    v_dim = v / dj.energy
    j_dim = j / dj.current
    assert jnp.allclose(v, v_dim * dj.energy, atol=1e-12)
    assert jnp.allclose(j, j_dim * dj.current, atol=1e-12)
    assert jnp.allclose(v, jnp.linspace(0.0, 0.95, 12), atol=1e-12)
    assert float(j[0]) > 0
