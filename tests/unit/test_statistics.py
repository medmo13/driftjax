"""Fermi-Dirac / Boltzmann statistics golden (driftjax-pinned)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import driftjax as dj
import driftjax.science
from driftjax.science.carrier_statistics import F_half, dF_half, ni, resolve_statistics
from driftjax.science.spectrum import spectrum


def test_f_half_golden():
    assert abs(float(F_half(jnp.float64(0.0))) - 0.7651) < 0.0005
    assert abs(float(F_half(jnp.float64(2.53))) - 3.657) < 0.0005
    assert abs(float(F_half(jnp.float64(20.0))) - 67.49) < 0.01


def test_f_half_boltzmann_limit():
    for eta in (-8.0, -5.0):
        ex = float(jnp.exp(eta))
        rel = abs(float(F_half(jnp.float64(eta))) - ex) / ex
        assert rel < 0.01, eta


def test_d_f_half_consistency():
    eta = jnp.array([-2.0, 0.0, 1.5, 5.0])
    g = jax.grad(lambda e: jnp.sum(F_half(e)))(eta)
    assert jnp.allclose(g, dF_half(eta), rtol=0.0001, atol=0.0001), (g, dF_half(eta))


def test_d_f_half_boltzmann():
    g = jax.grad(lambda e: jnp.sum(F_half(e, "boltzmann")))(jnp.float64(1.3))
    assert abs(float(g) - float(jnp.exp(1.3))) < 1e-12


def test_d_f_half_blend_weight_transition():
    """P0-1: dF must include the w'(S-Q) blend-weight term at eta ~= 10.

    The pre-fix formula (1-w)Q' + wS' missed it, erring by ~2.5e-4
    relative at the quadrature/Sommerfeld transition. The fixed formula
    is the exact derivative of the implemented primal: it must match both
    central differences and jax.grad through F_half over eta = -20..50
    with dense sampling around 8..12.
    """
    etas = np.concatenate(
        [np.linspace(-20, 7, 15), np.linspace(8, 12, 41), np.linspace(13, 50, 10)]
    )
    h = 1e-5
    worst_fd, argmax = 0.0, None
    for e in etas:
        ef = jnp.float64(e)
        fd = (float(F_half(jnp.float64(e + h))) - float(F_half(jnp.float64(e - h)))) / (2 * h)
        got = float(dF_half(ef))
        sc = abs(fd) + 1e-30
        r = abs(got - fd) / sc
        if r > worst_fd:
            worst_fd, argmax = r, e
    assert worst_fd < 5e-5, (worst_fd, argmax)
    eta = jnp.asarray(etas)
    g = jax.grad(lambda ee: jnp.sum(F_half(ee)))(eta)
    assert jnp.allclose(g, dF_half(eta), rtol=1e-4, atol=1e-6), (
        float(jnp.max(jnp.abs(g - dF_half(eta)))),
    )


def test_resolve_statistics():
    assert resolve_statistics("fermi-dirac") == "exact"
    assert resolve_statistics("boltzmann") == "boltzmann"
    with pytest.raises(ValueError):
        resolve_statistics("mystery")


def test_ni_physical():

    class _C:
        Nc = jnp.full(10, 8e17)
        Nv = jnp.full(10, 1.8e19)
        Eg = jnp.full(10, 1.5)

    v = ni(_C())
    assert float(v[0]) == float(jnp.sqrt(8e17 * 1.8e19) * jnp.exp(-1.5 / 2))


def test_blakemore_formula_golden_pins():
    """The Blakemore branch must reproduce the historical closed form
    F = (e^{−η} + t^{−2.7})^{−1/2.7}, t = (3√π/8)·η^{3/2} (provenance pin:
    this is the inaccurate historical approximation, NOT the exact path)."""
    eta = jnp.float64(2.0)
    t = 3.0 * jnp.sqrt(jnp.pi) / 8.0 * float(eta) ** 1.5
    golden = (float(jnp.exp(-eta)) + t ** (-2.7)) ** (-1.0 / 2.7)
    assert abs(float(F_half(eta, "blakemore")) - golden) < 1e-12


def test_blakemore_boltzmann_branch_error():
    """η ≤ 0: the driftjax fix keeps the Boltzmann branch (F → e^{η}); error vs
    exact FD must stay < 5% (pre-fix clamp made F → 0 → 100% error)."""
    for eta in (-10.0, -5.0, -2.0, -1.0, -0.5):
        ex = float(jnp.exp(eta))
        rel = abs(float(F_half(jnp.float64(eta), "blakemore")) - ex) / ex
        assert rel < 0.05, f"η={eta}: rel err {rel:.3f}"


def test_blakemore_degenerate_branch_error_envelope():
    """η ≥ 5: the degenerate power branch (2/3·η^{3/2}) dominates; the
    historical form is a coarse provenance approximation.  Measured regression
    error profile (pinned so future edits cannot silently "improve" the
    historical formula): 67 % at η=5, 21 % at η=10, 13 % at η=20 — the
    exact-FD path is the accurate one; blakemore is kept ONLY for
    cross-checking."""
    expect = {5.0: 0.7, 10.0: 0.25, 20.0: 0.15}
    for eta, env in expect.items():
        ex = float(F_half(jnp.float64(eta)))
        bl = float(F_half(jnp.float64(eta), "blakemore"))
        assert bl > 0.0 and abs(ex / bl - 1.0) < env, f"η={eta}"


def test_blakemore_device_ab():
    """regression: statistics="fd" vs "blakemore" on a non-degenerate Si
    homojunction agree to device precision (carrier densities sit on the
    Boltzmann branch where Blakemore ≡ exact; FD corrections only matter in
    degenerate regions).  Jsc bit-identical (generation path untouched),
    Voc within 1 mV."""
    mat = dj.material(
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
    ls = spectrum(normalize=True)
    # Fast-tier sizing: same-mesh A/B comparison, so N=60/11 steps pins the
    # identical gate as N=200/21 at a fraction of the Newton work.
    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [mat, mat], [1e16, -1e16], strict=False)),
        n_points=60,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()
    r_fd = dj.simulate(des, dj.Sweep(vmax=0.75, n_steps=11), statistics="fd", ls=ls)
    with pytest.warns(UserWarning, match="blakemore"):
        r_bl = dj.simulate(des, dj.Sweep(vmax=0.75, n_steps=11), statistics="blakemore", ls=ls)
    j_fd, j_bl = (float(np.asarray(r.current)[0]) for r in (r_fd, r_bl))
    assert abs(j_fd - j_bl) < 0.0001 * abs(j_fd)
    assert abs(float(r_fd.voc) - float(r_bl.voc)) < 0.001
