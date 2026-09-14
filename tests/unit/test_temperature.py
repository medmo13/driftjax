"""I5 — temperature + spectrum API.

Gates:
1. T = 300 path is bit-identical to the pre-I5 behaviour (default vs
   explicit T=300; design built at 300).
2. Voc(T) drops ≈ −1.4 mV/K over 290–330 K (constant-Eg / constant-τ
   diffusion-limited diode: −(Eg−qVoc)/(qT); the −2 mV/K textbook value
   additionally assumes Eg(T) shrinking and Nc(T) ∝ T^{3/2}).
3. simulate(T=…) equals rebuilding the design at T (with_temperature).
4. temperature_sweep returns per-T results with monotone Voc.
5. spectral_sensitivity: above-gap bins positive; sub-gap bins sit at the
   dilution floor −η/P_in (extra sub-gap power cannot generate, only dilutes).
"""

import numpy as np

import driftjax as dj
import driftjax.science
from driftjax.science.spectrum import spectrum

MAT = dj.material(
    Chi=3.9,
    Eg=1.12,
    eps=11.7,
    Nc=2.8e19,
    Nv=1.04e19,
    mn=1.08,
    mp=0.56,
    tn=1e-06,
    tp=1e-06,
    A=20000.0,
)
LS = spectrum(normalize=True)
# Fast-tier sizing: every gate here is comparative (bit-identity, slope,
# equality, monotonicity) so N=40/11 steps pins the same physics as N=120/21
# at ~1/6 the Newton work. The N=500 publication anchors live in validation/.
KW = dict(vmax=0.8, n_steps=11)


def _des(T=300.0):
    return dj.Device(
        layers=list(zip([0.0001, 0.0001], [MAT, MAT], [1e16, -1e16], strict=False)),
        n_points=40,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
        T=T,
    ).design()


def test_t300_bit_identical():
    d = _des()
    d2 = _des(300.0)
    for k in ("x", "dgrid", "Eg", "Ndop", "Snl", "PhiMl"):
        assert np.array_equal(np.asarray(getattr(d, k)), np.asarray(getattr(d2, k))), k
    r1 = dj.simulate(d, dj.Sweep(vmax=0.8, n_steps=11), ls=LS)
    r2 = dj.simulate(d, dj.Sweep(vmax=0.8, n_steps=11), ls=LS, T=300.0)
    assert np.array_equal(np.asarray(r1.voltages), np.asarray(r2.voltages))
    assert np.array_equal(np.asarray(r1.current), np.asarray(r2.current))


def test_voc_slope_about_minus_1_4_mv_per_k():
    """Constant-Eg / constant-τ model: dVoc/dT ≈ −(Eg−qVoc)/(qT) ≈ −1.4 mV/K.
    (The textbook −2 mV/K additionally assumes Eg(T) shrinking and
    Nc/Nv ∝ T^{3/2}, which this first-order model does not include.)"""
    # 3-point slope is sufficient: ±20 K around 300 K pins the sign + magnitude
    Ts = np.array([280.0, 300.0, 320.0])
    vocs = np.array(
        [float(dj.simulate(_des(), dj.Sweep(vmax=0.8, n_steps=11), ls=LS, T=T).voc) for T in Ts]
    )
    slope = np.polyfit(Ts, vocs, 1)[0] * 1000.0
    assert -2.5 < slope < -0.8, f"dVoc/dT = {slope:.2f} mV/K"


def test_simulate_T_equals_rebuilt_design():
    rT = dj.simulate(_des(300.0), dj.Sweep(vmax=0.8, n_steps=11), ls=LS, T=310.0)
    rB = dj.simulate(_des(310.0), dj.Sweep(vmax=0.8, n_steps=11), ls=LS)
    assert np.allclose(np.asarray(rT.current), np.asarray(rB.current), rtol=1e-08, atol=1e-08)


def test_temperature_sweep_api():
    sw = dj.simulator.temperature_sweep(_des(), LS, Ts=[300.0, 310.0, 320.0], **KW)
    assert set(sw) == {300.0, 310.0, 320.0}
    vocs = [float(sw[T].voc) for T in (300.0, 310.0, 320.0)]
    assert vocs[0] > vocs[1] > vocs[2]


def test_spectral_sensitivity_signature():
    # 4 bins suffices to check above-gap > 0 and below-gap at floor
    ss = dj.simulator.spectral_sensitivity(_des(), LS, n_bins=4, n_steps=11, vmax=0.7)
    lam, sens = (ss[:, 0], ss[:, 1])
    assert lam.min() > 200.0 and lam.max() < 5000.0
    above = sens[lam < 1107.0]
    below = sens[lam > 1107.0]
    assert above.min() > 0.0, "above-gap sensitivity must be positive"
    eta0 = float(dj.simulate(_des(), dj.Sweep(vmax=0.7, n_steps=11), ls=LS).efficiency)
    floor = -eta0 / 1000.0
    assert np.all(np.abs(below / floor - 1.0) < 0.05), (
        "sub-gap sensitivity must sit at the −η/P_in dilution floor"
    )
