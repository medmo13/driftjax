"""Coherent TMM validation test."""

import numpy as np
import pytest

import driftjax as dj
import driftjax.science
from driftjax.science.optics import _tmm_core
from driftjax.science.spectrum import spectrum

pytestmark = pytest.mark.smoke

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


def _absorptance_from_G(G_node, d_m, phi0):
    """Endpoint-corrected trapezoid: ∫G·dx = Σ G_j·d − ½(G₀+G_L)·d."""
    d = float(np.asarray(d_m)[0])
    G = np.asarray(G_node)
    return (float(np.sum(G)) - 0.5 * (G[0] + G[-1])) * d / float(phi0)


def _airy_rt(n_real, k_ext, lam_m, D_m):
    """Free-standing 1-slab Airy (power) R, T — independent analytic check."""
    n_t = complex(n_real, k_ext)
    beta = 2 * np.pi * n_t * D_m / lam_m
    r0 = (1.0 - n_t) / (1.0 + n_t)
    rA = r0 * (1 - np.exp(2j * beta)) / (1 - r0**2 * np.exp(2j * beta))
    tA = (1 - r0**2) * np.exp(1j * beta) / (1 - r0**2 * np.exp(2j * beta))
    return (abs(rA) ** 2, abs(tA) ** 2)


def test_tmm_matches_airy_one_slab():
    """Coherent R, T of a uniform slab equal the analytic Airy result."""
    lam_nm, D_m = (600.0, 1e-06)
    nint = 100
    d_m = np.full(nint, D_m / nint)
    _, R, T = _tmm_core(np.full(nint, 3.4), np.full(nint, 0.1), d_m, lam_nm * 1e-09, 1.0)
    R_a, T_a = _airy_rt(3.4, 0.1, lam_nm * 1e-09, D_m)
    assert abs(float(R) - R_a) < 1e-08, f"R TMM {float(R):.10f} vs Airy {R_a:.10f}"
    assert abs(float(T) - T_a) < 1e-08, f"T TMM {float(T):.10f} vs Airy {T_a:.10f}"


def test_tmm_energy_conservation_per_wavelength():
    """∫G·dx = Φ₀·(1−R−T) — Maxwell-consistent absorption accounting."""
    lam_nm = 600.0
    nint = 200
    D_m = 1e-06
    d_m = np.full(nint, D_m / nint)
    phi0 = 1.0
    G, R, T = _tmm_core(np.full(nint, 3.4), np.full(nint, 0.1), d_m, lam_nm * 1e-09, phi0)
    A = 1.0 - float(R) - float(T)
    A_int = _absorptance_from_G(G, d_m, phi0)
    assert abs(A_int - A) < 1e-06, f"∫G/Φ₀ = {A_int:.10f} vs A = {A:.10f}"


def test_tmm_tends_to_beer_lambert_in_thick_cell():
    """100 µm Si: coherent TMM → the incoherent thick-cell limit.

    For αL ≫ 1 interference washes out and the exact result is the
    single-interface front reflection times the Beer–Lambert transmittance:

        A(λ) → (1−R₀)·(1−e^{−α(λ)L}),   R₀ = |(1−√ε)/(1+√ε)|².

    Verified per wavelength (chain-derived R/T is stable at any thickness —
    unlike the internal-field reconstruction, which amplifies backward-wave
    roundoff for αL ≳ 25; that limitation is documented in optics.py).

    Note: the simulator's Beer–Lambert path does NOT include front-surface
    reflection and its per-interval left-Riemann sum overestimates ∫G·dx on
    coarse meshes (Δτ ≈ 1), so comparing G-fields directly is meaningless
    here — the analytic limit is the correct gate.
    """
    from driftjax.science.optics import alpha_tauc, tmm_rt

    des = dj.Device(
        layers=list(zip([0.005, 0.005], [MAT, MAT], [1e16, -1e16], strict=False)),
        n_points=200,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()
    ls = spectrum(normalize=True)
    alpha = np.asarray(alpha_tauc(des, ls))
    R, T = tmm_rt(des, ls, alpha_mode="tauc")
    A_l = 1.0 - np.asarray(R) - np.asarray(T)
    R0 = ((1.0 - np.sqrt(11.7)) / (1.0 + np.sqrt(11.7))) ** 2
    L_m = 0.0001
    lam = np.asarray(ls.Lambda)
    for wl in [450.0, 600.0, 800.0, 1000.0, 1100.0]:
        i = int(np.argmin(np.abs(lam - wl)))
        aL = float(alpha[i, 0]) * L_m
        expect = (1.0 - R0) * (1.0 - np.exp(-aL))
        assert abs(A_l[i] / expect - 1.0) < 0.02, (
            f"λ={lam[i]:.0f}nm: TMM {A_l[i]:.4f} vs limit {expect:.4f}"
        )


def test_tmm_gradient_matches_fd():
    """d(ΣG)/dE_g through the coherent chain agrees with central FD.

    Mesh-independent identity check: a coarse (N=60) device keeps the
    full-spectrum TMM trace cheap without weakening the assertion.
    """
    import jax
    import jax.numpy as jnp

    from driftjax.science.optics import tmm_generation

    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [MAT, MAT], [1e16, -1e16], strict=False)),
        n_points=30,  # was 60→40→30, saves ~60% vs 60 with same gate
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()
    ls = spectrum(normalize=True)

    def loss(d):
        return jnp.sum(tmm_generation(d, ls, alpha_mode="tauc"))

    import equinox as eqx

    g = jax.grad(loss)(des)
    j = des.Eg.shape[0] // 2  # midpoint, was hardcoded 30 for n=60
    h = 0.0001
    Eg_p = des.Eg.at[j].add(h)
    Eg_m = des.Eg.at[j].add(-h)
    fd = (
        loss(eqx.tree_at(lambda d: d.Eg, des, Eg_p)) - loss(eqx.tree_at(lambda d: d.Eg, des, Eg_m))
    ) / (2 * h)
    ratio = float(g.Eg[j]) / float(fd)
    assert abs(ratio - 1.0) < 0.01, (
        f"AD {float(g.Eg[j]):.6e} vs FD {float(fd):.6e} (ratio {ratio:.5f})"
    )


def test_tmm_multilayer_energy_conservation():
    """H1 regression: ∫G·dx must equal (1−R−T)·Φ₀ per wavelength for a
    MULTILAYER (n changes at interfaces).  The pre-driftjax chain used an inverted
    propagation phase + r = T[1,0]/T[0,0] — accidentally correct for a
    single slab (Airy 1e-16) but over-shot ∫G·dx vs 1−R−T by up to 3.5× on
    the ETL/perovskite/HTM stack below.  Fixed in driftjax (H1 audit); this gate
    keeps the fix.
    """
    from driftjax.science.optics import _refractive_index, _tmm_core
    from driftjax.units import length as _L

    etl = dj.material(
        Chi=4.0, Eg=3.2, eps=9.0, Nc=1e19, Nv=1e19, mn=20.0, mp=20.0, tn=1e-07, tp=1e-07, A=0.0
    )
    perov = dj.load_material("MAPbI3")
    htm = dj.material(
        Chi=2.2, Eg=3.0, eps=3.0, Nc=1e19, Nv=1e19, mn=1.0, mp=1.0, tn=1e-07, tp=1e-07, A=0.0
    )
    des = dj.Device(
        layers=list(
            zip([5e-06, 4e-05, 5e-06], [etl, perov, htm], [1e17, 0.0, -1e17], strict=False)
        ),
        n_points=150,
        Snl=10000000.0,
        Snr=10000000.0,
        Spl=10000000.0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()
    ls = spectrum(normalize=True)
    n_real, k = _refractive_index(des, ls, "tauc")
    d_m = np.asarray(des.dgrid) * float(_L) * 0.01
    lam = np.asarray(ls.Lambda)
    worst = 0.0
    for wl in (450.0, 500.0, 600.0, 700.0):
        i = int(np.argmin(np.abs(lam - wl)))
        G, R, T = _tmm_core(np.asarray(n_real[i]), np.asarray(k[i]), d_m, lam[i] * 1e-09, 1.0)
        d = d_m[0]
        A_int = (float(np.sum(G)) - 0.5 * (float(G[0]) + float(G[-1]))) * d
        A = 1.0 - float(R) - float(T)
        worst = max(worst, abs(A_int / A - 1.0))
    assert worst < 1e-09, f"multilayer ∫G·dx vs 1−R−T mismatch: {worst:.2e}"
