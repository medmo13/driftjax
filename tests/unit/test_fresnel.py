"""H2 — Fresnel double-pass optics: Beer–Lambert + front reflection + rear mirror.

Gates:
1. Exact double-pass absorptance   A = (1−R_f)·(1−e^{−τ_L})·(1 + R_b·e^{−τ_L})
   on a fine mesh (left-Riemann truncation ~ Δτ/2, verified 1st-order).
2. First-order mesh convergence of the discrete integral.
3. Front-reflection loss: R_b = 0 generation < plain Beer–Lambert.
4. End-to-end dispatch: simulate(alpha_mode="fresnel") runs.
"""

import numpy as np

import driftjax as dj
import driftjax.science
from driftjax.science.optics import (
    _refractive_index,
    alpha_tauc,
    beer_lambert_G,
    fresnel_generation,
    photonflux,
)
from driftjax.science.spectrum import spectrum
from driftjax.units import length as _L

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


def _per_lambda_with_Rb(x_m, phi0, alpha, n_tilde, Rb):
    """Same math as _fresnel_per_lambda but with tunable R_b (the module
    hardcodes 0.9); used for the analytic double-pass gate."""
    dx = np.diff(x_m)
    ai = alpha[:, :-1]
    tau = np.cumsum(ai * dx, axis=1)
    tau_prep = np.concatenate([np.zeros((len(phi0), 1)), tau], axis=1)
    tau_L = tau[:, -1:]
    R_f = np.abs((1.0 - n_tilde) / (1.0 + n_tilde)) ** 2
    phi_f = phi0[:, None] * (1.0 - R_f)[:, None]
    phi_fwd = phi_f * np.exp(-tau_prep[:, :-1])
    phi_bwd = phi_f * np.exp(-2.0 * tau_L) * np.exp(+tau_prep[:, :-1]) * Rb
    G_int = ai * (phi_fwd + phi_bwd)
    G_l = np.concatenate([G_int, G_int[:, -1:]], axis=1)
    G_r = np.concatenate([G_int[:, :1], G_int], axis=1)
    return 0.5 * (G_l + G_r)


def test_fresnel_matches_analytic_double_pass():
    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [MAT, MAT], [1e16, -1e16], strict=False)),
        n_points=2000,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()
    x_m = np.asarray(des.x) * float(_L) * 0.01
    dx = x_m[1] - x_m[0]
    alpha = np.asarray(alpha_tauc(des, LS))
    n_real, k = _refractive_index(des, LS, "tauc")
    n_tilde = n_real[:, 0] + 1j * k[:, 0]
    phi0 = np.asarray(photonflux(LS))
    tauL = np.sum(alpha[:, :-1] * dx, axis=1)
    R_f = np.abs((1.0 - n_tilde) / (1.0 + n_tilde)) ** 2
    for Rb in (0.0, 0.9):
        G = _per_lambda_with_Rb(x_m, phi0, alpha, n_tilde, Rb)
        A = (np.sum(G, axis=1) - 0.5 * (G[:, 0] + G[:, -1])) * dx / phi0
        A_an = (1.0 - R_f) * (1.0 - np.exp(-tauL)) * (1.0 + Rb * np.exp(-tauL))
        m = A_an > 1e-06
        err = float(np.max(np.abs(A[m] / A_an[m] - 1.0)))
        assert err < 0.0025, f"Rb={Rb}: max relerr {err:.2e}"


def test_fresnel_integral_first_order_convergence():
    errs = []
    for n in (100, 400):
        des = dj.Device(
            layers=list(zip([0.0001, 0.0001], [MAT, MAT], [1e16, -1e16], strict=False)),
            n_points=n,
            Snl=10000000.0,
            Snr=0,
            Spl=0,
            Spr=10000000.0,
            PhiMl=-1.0,
            PhiMr=-1.0,
        ).design()
        x_m = np.asarray(des.x) * float(_L) * 0.01
        dx = x_m[1] - x_m[0]
        alpha = np.asarray(alpha_tauc(des, LS))
        n_real, k = _refractive_index(des, LS, "tauc")
        n_tilde = n_real[:, 0] + 1j * k[:, 0]
        phi0 = np.asarray(photonflux(LS))
        G = _per_lambda_with_Rb(x_m, phi0, alpha, n_tilde, 0.9)
        A = (np.sum(G, axis=1) - 0.5 * (G[:, 0] + G[:, -1])) * dx / phi0
        tauL = np.sum(alpha[:, :-1] * dx, axis=1)
        R_f = np.abs((1.0 - n_tilde) / (1.0 + n_tilde)) ** 2
        A_an = (1.0 - R_f) * (1.0 - np.exp(-tauL)) * (1.0 + 0.9 * np.exp(-tauL))
        m = A_an > 1e-06
        errs.append(float(np.max(np.abs(A[m] / A_an[m] - 1.0))))
    assert errs[1] < 0.35 * errs[0], f"expected ~1/4 error ratio, got {errs}"


def test_fresnel_front_reflection_loss():
    """R_b = 0: generation < plain Beer–Lambert (front loss (1−R_f) < 1)."""
    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [MAT, MAT], [1e16, -1e16], strict=False)),
        n_points=400,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()
    G_bl = np.asarray(beer_lambert_G(des, LS, alpha_mode="tauc"))
    G_f0 = np.asarray(fresnel_generation(des, LS, alpha_mode="tauc", rear_reflectance=0.0))
    ratio = float(np.sum(G_f0) / np.sum(G_bl))
    assert 0.55 < ratio < 0.95, f"fresnel(Rb=0)/BL = {ratio:.3f}"


def test_fresnel_dispatch_simulates():
    # Dispatch + Jsc-range gate only: N=60/9 steps suffices (the analytic
    # accuracy gates above use fine numpy-only meshes, no solves).
    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [MAT, MAT], [1e16, -1e16], strict=False)),
        n_points=60,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()
    r = dj.simulate(des, dj.Sweep(vmax=0.7, n_steps=9), optics=dj.Fresnel(alpha_mode="tauc"), ls=LS)
    jsc = float(np.asarray(r.current)[0]) * 1000.0
    assert 0.0 < jsc < 45.0, f"fresnel Jsc {jsc:.2f} mA/cm² out of range"
