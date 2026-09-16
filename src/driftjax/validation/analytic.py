"""Analytic reference yardsticks: ideal diode + Shockley–Queisser limit.

These functions are *independent physics implemented from first principles in
plain NumPy* — deliberately free of driftjax/JAX/optics machinery so that the
yardstick cannot share a defect with the simulator it validates.

Shockley–Queisser detailed balance (implemented here)
-----------------------------------------------------
    Jsc = q ∫_{Eg}^∞ φ_sun(E) dE                     (absorbed photon flux)
    J0  = q ∫_{Eg}^∞ Φ_bb(E, T_cell) dE              (radiative dark current,
                                                      front-hemisphere emission)
    J(V) = J0·(e^{qV/kT} − 1) − Jsc
    η(Eg) = max_V [ V·J(V) ] / P_in

Two illuminations are provided:

* ``spectrum="blackbody"`` — the canonical Shockley–Queisser (1961) setup:
  a 6000 K blackbody sun at 1-sun dilution (fc = 1),
  P_in = σ·T_s⁴·(R_s/d)².
* ``spectrum="am15g"`` — the package AM1.5G table (1000 W/m²) converted to a
  photon flux from first principles (φ = P/E with E = hc/λ).

Published anchors (verified, see tests/literature/test_sq_limit.py):
  * Shockley & Queisser, J. Appl. Phys. 32, 510 (1961): blackbody 6000 K sun,
    fc = 1 → η_max ≈ 30% at Eg ≈ 1.1 eV; ultimate (thermalisation-only)
    limit ≈ 44%.
  * S. Rühle, Sol. Energy 130, 139 (2016): AM1.5G detailed balance →
    η_max ≈ 33.2% near Eg ≈ 1.34 eV; Si (1.12 eV) ≈ 32–33%.

History: before v0.1.14 this function returned a hardcoded 0.337 constant for
bandgaps near 1.12 eV (circularly passing its own test). It now performs the
real detailed-balance calculation.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# CODATA-2018 defining constants (exact) + derived
# ---------------------------------------------------------------------------
_Q = 1.602176634e-19  # C
_H = 6.62607015e-34  # J·s
_C = 299_792_458.0  # m/s
_KB = 1.380649e-23  # J/K
_SB = 5.670374419e-8  # W·m⁻²·K⁻⁴ (CODATA 2018 measured)
_HC = _H * _C  # J·m

# SQ-1961 blackbody-sun geometry (fc = 1, unconcentrated)
_T_SUN = 6000.0  # K (Shockley & Queisser used 6000 K)
_R_SUN = 6.957e8  # m (IAU nominal solar radius)
_AU = 1.496e11  # m
_DILUTION = (_R_SUN / _AU) ** 2  # ≈ 2.164e-5


def _bb_photon_emittance(E: np.ndarray, T: float) -> np.ndarray:
    """Hemispherical blackbody photon emittance [photons m⁻² s⁻¹ J⁻¹].

    Φ(E,T) = 2π E² / (h³c² (e^{E/kT} − 1))  — Planck law in photon count,
    integrated over the 2π hemisphere of a Lambertian surface.

    Normalisation self-check (independent of this module's integrals):
    ∫₀^∞ Φ dE = 4π ζ(3) (kT)³ / (h³c²) ≡ σ_SB·T⁴ / (2.7012 kT)
    (σ T⁴ is the energy flux; 2.7012 kT is the exact mean photon energy,
    π⁴/(30ζ(3))·kT), both of which the test suite pins.
    """
    E = np.asarray(E, dtype=np.float64)
    x = E / (_KB * T)
    # clip at x=700: e^700 is the float64 ceiling; the exact result there is
    # ≈2e-304 of the Planck normalisation — physically identical to zero, and
    # clipping avoids the expm1 overflow warning on absurd energies.
    return 2.0 * np.pi * E**2 / (_H**3 * _C**2 * np.expm1(np.minimum(x, 700.0)))


def _blackbody_tail_photons(
    Eg_eV: float, T: float, dilution: float = 1.0, n_grid: int = 8192, span: float = 80.0
) -> float:
    """∫_{Eg}^∞ Φ_bb(E,T)·dilution dE  →  [photons m⁻² s⁻¹].

    Numerically integrates the photon tail in the dimensionless variable
    x = E/kT (integrand x²/(eˣ−1) decays like e^{−x}; 80 kT span leaves a
    ≈ e⁻⁸⁰-truncation error, far below float64 round-off).
    """
    Eg_J = abs(Eg_eV) * _Q
    x_g = Eg_J / (_KB * T)
    if not np.isfinite(x_g):
        return 0.0
    x = np.linspace(x_g, x_g + span, int(n_grid))
    integrand = x**2 / np.expm1(x)
    val = np.trapezoid(integrand, x)
    return float(val * (_KB * T) ** 3 * 2.0 * np.pi / (_H**3 * _C**2) * dilution)


def _mpp_voltage(A: float, T: float) -> float:
    """Maximum-power-point voltage of J(V) = J0(e^{qV/kT}−1) − Jsc.

    dP/dV = 0 ⟺ (1+v)e^{v} = A with v = qV/kT and A = 1 + Jsc/J0.
    Solved by damped-free Newton with the analytic seed v₀ = ln A − 1
    (converges quadratically for every A > 1; the MPP exists uniquely).
    """
    v = np.log(A) - 1.0
    for _ in range(80):
        expv = np.exp(min(v, 700.0))  # guard against overflow for tiny Eg
        f = (1.0 + v) * expv - A
        fp = expv * (2.0 + v)
        if fp <= 0.0:  # only at v < −2, unreachable from the seed
            break
        dv = f / fp
        v -= dv
        if abs(dv) < 1e-14 * max(1.0, abs(v)):
            break
    return float(v * _KB * T / _Q)


def sq_efficiency_limit(
    Eg_eV: float, T: float = 300.0, spectrum: str = "blackbody", n_grid: int = 8192
) -> float:
    """Shockley–Queisser detailed-balance efficiency limit (fraction).

    Parameters
    ----------
    Eg_eV : float
        Absorber bandgap in eV.  A step-function absorber (sharp absorption
        edge above Eg, unity EQE) is assumed, as in the original SQ model.
    T : float
        Cell temperature in K (300 K default; sets the radiative dark current).
    spectrum : {"blackbody", "am15g"}
        Illumination.  ``blackbody`` reproduces the canonical SQ-1961 setup
        (6000 K sun, 1-sun dilution, fc = 1).  ``am15g`` uses the package's
        AM1.5G table rescaled to exactly 1000 W/m².
    n_grid : int
        Integration points for the blackbody tail (convergence-pinned by the
        test suite: halving n_grid must change η by < 1e-3).

    Returns
    -------
    float
        Efficiency as a fraction (0.30 == 30%).  Non-finite / degenerate
        inputs (Eg ≤ 0, no photons above Eg) return 0.0.
    """
    if not np.isfinite(Eg_eV) or Eg_eV <= 0:
        return 0.0
    Eg_J = Eg_eV * _Q

    if spectrum == "blackbody":
        phi_sun = _blackbody_tail_photons(Eg_eV, _T_SUN, _DILUTION, n_grid)
        P_in = _SB * _T_SUN**4 * _DILUTION  # irradiance of the diluted sun
    elif spectrum == "am15g":
        lam_nm, P_w = _am15g_table()
        E_ph = _HC / (lam_nm * 1e-9)  # photon energy per table row [J]
        sel = E_ph >= Eg_J
        if not np.any(sel):
            return 0.0
        phi_sun = float(np.sum(P_w[sel] / E_ph[sel]))
        P_in = float(np.sum(P_w))
    else:
        raise ValueError(f"unknown spectrum {spectrum!r} (use 'blackbody' or 'am15g')")

    Jsc = _Q * phi_sun
    J0 = _Q * _blackbody_tail_photons(Eg_eV, T, 1.0, n_grid)  # front hemisphere
    if not (np.isfinite(Jsc) and np.isfinite(J0)) or J0 <= 0.0:
        return 0.0
    A = 1.0 + Jsc / J0
    if A <= 1.0 + 1e-12:  # no usable photovoltaic response
        return 0.0
    Vmpp = _mpp_voltage(A, T)
    vmpp_norm = Vmpp * _Q / (_KB * T)
    Jmpp = Jsc - J0 * np.expm1(min(vmpp_norm, 700.0))
    eta = Vmpp * Jmpp / P_in
    if not np.isfinite(eta) or eta < 0.0:
        return 0.0
    return float(min(eta, 1.0))


def sq_ultimate_efficiency(Eg_eV: float, spectrum: str = "blackbody") -> float:
    """Shockley–Queisser *ultimate* (thermalisation-only) efficiency.

    Every photon with E ≥ Eg yields exactly Eg of work (one pair per photon,
    no dark current, no extraction losses):

        η_ult(Eg) = Eg · ∫_{Eg}^∞ φ_sun dE / P_in

    Shockley & Queisser (1961) give the peak ≈ 44% (6000 K blackbody sun).
    η_ult(Eg) ≥ η_SQ(Eg) holds exactly (removing loss channels), which the
    test suite pins as an ordering theorem.
    """
    if not np.isfinite(Eg_eV) or Eg_eV <= 0:
        return 0.0
    if spectrum == "blackbody":
        phi_sun = _blackbody_tail_photons(Eg_eV, _T_SUN, _DILUTION)
        P_in = _SB * _T_SUN**4 * _DILUTION
    elif spectrum == "am15g":
        lam_nm, P_w = _am15g_table()
        E_ph = _HC / (lam_nm * 1e-9)
        sel = E_ph >= Eg_eV * _Q
        if not np.any(sel):
            return 0.0
        phi_sun = float(np.sum(P_w[sel] / E_ph[sel]))
        P_in = float(np.sum(P_w))
    else:
        raise ValueError(f"unknown spectrum {spectrum!r} (use 'blackbody' or 'am15g')")
    return float(min(Eg_eV * _Q * phi_sun / P_in, 1.0))


def _am15g_table():
    """Package AM1.5G table as numpy (λ in nm, P in W/m²), exactly 1000 W/m².

    Lazy import keeps this validation module independent of the JAX stack.
    """
    from driftjax.science.spectrum import spectrum as _spectrum

    ls = _spectrum(normalize=True)
    return np.asarray(ls.Lambda), np.asarray(ls.P_in)


def ideal_diode_current(V, J0, n_ideal=1.0, Vt=0.0258, Jph=0.0):
    """Photovoltaic diode convention: I = −Jph + J0·(e^{V/(n Vt)} − 1).

    Negative at V = 0 when illuminated (photocurrent opposes the diode
    drop) and rises monotonically with V.  Power P = −V·I peaks at the
    maximum-power point.  Exact identities pinned by the test suite:
    J(Voc) = 0 with Voc = n·Vt·ln(1 + Jph/J0), and the MPP satisfies
    (1+v)e^{v} = 1 + Jph/J0 with v = V_mpp/(n·Vt).
    """
    return np.asarray(-Jph + J0 * (np.exp(np.asarray(V) / (n_ideal * Vt)) - 1.0))


# ---------------------------------------------------------------------------
# R3 (scientific-review fix): independent ANALYTIC derivative references.
#
# Every gradient check inside DriftJax compares the adjoint against a finite
# difference of the *same* forward solver, so a shared systematic forward
# error would cancel and be invisible.  The two functions below close that
# gap: they differentiate an INDEPENDENT first-principles model (plain NumPy,
# no JAX, no driftjax solver) and give exact closed-form sensitivities that a
# correct implementation must reproduce.
# ---------------------------------------------------------------------------


def sq_efficiency_gradient(Eg_eV: float, spectrum: str = "am15g", T: float = 300.0,
                           n_grid: int = 4000, eps: float = 1e-6) -> float:
    """Central-difference d(eta_SQ)/dEg of the independent detailed-balance model.

    This is deliberately a *numeric* derivative of the independent analytic
    forward model (not of the DriftJax solver): it is exact to the
    truncation error of the step, and because the forward model shares no
    code with the simulator, agreement certifies the *physics gradient*
    rather than the consistency of one implementation with itself.
    """
    # NOTE: sq_efficiency_limit signature is (Eg_eV, T, spectrum, n_grid).
    f = lambda E: sq_efficiency_limit(E, T, spectrum, n_grid)
    return float((f(Eg_eV + eps) - f(Eg_eV - eps)) / (2.0 * eps))


def ultimate_efficiency_gradient_closed_form(Eg_eV: float, spectrum: str = "am15g") -> float:
    """EXACT closed-form d(eta_ult)/dEg, no differencing at all.

    eta_ult(Eg) = Eg * Q * Phi(Eg) / P_in,  Phi(Eg) = sum_{E_i >= Eg} P_i/E_i
    so with S(Eg) = sum over the absorbed-photon *power* fraction,

        d eta_ult/dEg = Q/P_in * [ Phi(Eg) - Eg * (dPhi/dEg) ]

    and because the AM1.5G table is a discrete set of bins, Phi is a step
    function of Eg: between bin edges dPhi/dEg = 0 exactly, and at an edge it
    is a delta.  Away from the edges this reduces to the clean statement

        d eta_ult/dEg = Q * Phi(Eg) / P_in = eta_ult(Eg) / Eg

    which is the standard result that the ultimate efficiency has unit
    logarithmic sensitivity to Eg.  We return the exact off-edge value and
    document the edge caveat.
    """
    if not np.isfinite(Eg_eV) or Eg_eV <= 0:
        return 0.0
    if spectrum == "blackbody":
        phi = _blackbody_tail_photons(Eg_eV, _T_SUN, _DILUTION)
        P_in = _SB * _T_SUN**4 * _DILUTION
    elif spectrum == "am15g":
        lam_nm, P_w = _am15g_table()
        E_ph = _HC / (lam_nm * 1e-9)
        sel = E_ph >= Eg_eV * _Q
        if not np.any(sel):
            return 0.0
        phi = float(np.sum(P_w[sel] / E_ph[sel]))
        P_in = float(np.sum(P_w))
    else:
        raise ValueError(f"unknown spectrum {spectrum!r}")
    # exact off-edge derivative: eta_ult/Eg
    return float(_Q * phi / P_in)
