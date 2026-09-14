"""Literature gates: Shockley–Queisser detailed balance + ideal diode.

The yardsticks live in ``driftjax.validation.analytic`` and are implemented
from first principles (plain NumPy, no driftjax/JAX machinery) so they cannot
share a defect with the simulator.  History: until v0.1.14
``sq_efficiency_limit`` returned a hardcoded 0.337 constant — the old test
passed circularly by construction and validated nothing.

Published anchors (all verified against the cited sources):
* Shockley & Queisser, J. Appl. Phys. 32, 510 (1961): 6000 K blackbody sun,
  fc = 1 → "maximum efficiency is found to be 30% for an energy gap of
  1.1 eV"; ultimate (thermalisation-only) limit ≈ 44%.
* S. Rühle, Sol. Energy 130, 139 (2016) (AM1.5G detailed-balance tabulation):
  η_max ≈ 33.2% near Eg ≈ 1.3–1.4 eV; Si (1.12 eV) ≈ 32–33%.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from driftjax.units import energy
from driftjax.validation.analytic import (
    _C,
    _DILUTION,
    _H,
    _KB,
    _SB,
    _T_SUN,
    _bb_photon_emittance,
    ideal_diode_current,
    sq_efficiency_limit,
    sq_ultimate_efficiency,
)

# ---------------------------------------------------------------------------
# Shockley–Queisser vs published anchors
# ---------------------------------------------------------------------------


def test_sq_blackbody_matches_shockley_queisser_1961():
    """SQ-1961 abstract: η_max = 30% at Eg = 1.1 eV (6000 K blackbody, fc=1)."""
    eta_11 = sq_efficiency_limit(1.1)
    assert eta_11 == pytest.approx(0.30, abs=0.005), f"SQ-61 anchor: η(1.1)={eta_11:.4f}"


def test_sq_blackbody_shape():
    """Physical shape of the blackbody η(Eg) curve (6000 K sun)."""
    egs = np.linspace(0.6, 2.6, 41)
    etas = np.array([sq_efficiency_limit(float(e)) for e in egs])
    assert etas.max() > 0.29, f"blackbody peak too low: {etas.max():.4f}"
    assert etas.max() < 0.33, f"blackbody peak too high: {etas.max():.4f}"
    i_peak = int(np.argmax(etas))
    assert 0.9 <= egs[i_peak] <= 1.6, f"peak at {egs[i_peak]:.2f} eV, expected ~1.1–1.4"
    # tails: the 6000 K blackbody has a genuine IR tail — η decays but never
    # hits exact zero (unlike the finite am15g table)
    assert sq_efficiency_limit(3.5) < 0.10
    assert sq_efficiency_limit(0.25) < 0.15
    assert sq_efficiency_limit(6.0) < 0.02


def test_sq_am15g_matches_ruhle_2016():
    """Rühle 2016 AM1.5G tabulation: η_max ≈ 33.2%, Si (1.12 eV) ≈ 32–33%.

    Bands carry ±0.015 slack for the package AM1.5G table's discretisation
    (99 rows, smoothed — measured J_ph(<1120 nm) = 44.5 mA/cm²).
    """
    eta_si = sq_efficiency_limit(1.12, spectrum="am15g")
    assert 0.30 < eta_si < 0.35, f"Si AM1.5G: {eta_si:.4f} (published ≈ 0.32–0.33)"
    egs = np.linspace(1.0, 1.6, 25)
    etas = np.array([sq_efficiency_limit(float(e), spectrum="am15g") for e in egs])
    assert 0.31 < etas.max() < 0.36, f"AM1.5G peak: {etas.max():.4f} (published ≈ 0.332)"


def test_sq_ultimate_efficiency_matches_sq_1961():
    """SQ-1961: ultimate (thermalisation-only) limit ≈ 44% (6000 K sun)."""
    egs = np.linspace(0.5, 2.2, 35)
    ults = np.array([sq_ultimate_efficiency(float(e)) for e in egs])
    assert ults.max() == pytest.approx(0.44, abs=0.012), (
        f"ultimate peak {ults.max():.4f} at {egs[int(np.argmax(ults))]:.2f} eV"
    )


def test_sq_ordering_theorem():
    """η_ult ≥ η_SQ exactly (removing loss channels cannot lower η)."""
    for e in np.linspace(0.5, 2.8, 24):
        e = float(e)
        assert sq_ultimate_efficiency(e) + 1e-12 >= sq_efficiency_limit(e), e


def test_sq_grid_convergence():
    """Halving the integration grid must move η by < 1e-3 (convergence)."""
    e1 = sq_efficiency_limit(1.12, n_grid=4096)
    e2 = sq_efficiency_limit(1.12, n_grid=8192)
    assert abs(e1 - e2) < 1e-3, abs(e1 - e2)


def test_sq_degenerate_inputs():
    """Eg ≤ 0 / NaN / no-photons must return 0.0, not NaN."""
    for eg in (0.0, -1.0, float("nan"), float("inf")):
        v = sq_efficiency_limit(eg)
        assert v == 0.0, f"sq({eg}) = {v}"


# ---------------------------------------------------------------------------
# Independent normalisation identities of the Planck photon emittance
# ---------------------------------------------------------------------------


def test_blackbody_photon_emittance_normalisation():
    """∫₀^∞ Φ dE = σT⁴/(2.7012·kT)  (energy flux ÷ exact mean photon energy).

    Two independent closed forms: 4πζ(3)(kT)³/(h³c²) (ζ(3) = 1.2020569…)
    and σ_SB·T⁴/(2.70118·kT) with mean photon energy π⁴/(30ζ(3))·kT.
    Pins the h³c² and 2π normalisation from first principles.
    """
    T = 300.0
    x = np.linspace(1e-4, 700.0, 400_000)  # ascending, dimensionless
    E = x * _KB * T
    total = np.trapezoid(_bb_photon_emittance(E, T), E)
    closed = 4.0 * np.pi * 1.202056903159594 * (_KB * T) ** 3 / (_H**3 * _C**2)
    assert total == pytest.approx(closed, rel=2e-4)
    sb_form = _SB * T**4 / (2.70118 * _KB * T)
    assert closed == pytest.approx(sb_form, rel=2e-4)


def test_blackbody_tail_asymptotic_identity():
    """∫_{Eg}^∞ Φ dE vs the analytic Boltzmann-tail expansion.

    For x_g = Eg/kT ≫ 1: ∫ = (kT)³·e^{−x}(x²+2x+2)·2π/(h³c²).
    Independent derivation; rel. error O(e^{−x}·x⁴) ≈ 1e-6 at x = 20.
    """
    T = 300.0
    x_g = 20.0
    Eg_eV = x_g * _KB * T / 1.602176634e-19
    numeric = sq_efficiency_limit  # noqa: F841  (keep name importable)
    from driftjax.validation.analytic import _blackbody_tail_photons

    got = _blackbody_tail_photons(Eg_eV, T, 1.0, n_grid=16384)
    kT = _KB * T
    closed = np.exp(-x_g) * (x_g**2 + 2 * x_g + 2) * kT**3 * 2 * np.pi / (_H**3 * _C**2)
    assert got == pytest.approx(closed, rel=2e-5)


def test_blackbody_sun_irradiance():
    """Diluted 6000 K sun: P_in = σT⁴(R_s/d)² ≈ 1.59 kW/m² (SQ-61 setup)."""
    from driftjax.validation.analytic import _SB

    p_in = _SB * _T_SUN**4 * _DILUTION
    assert p_in == pytest.approx(1590.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Ideal diode: exact analytic identities
# ---------------------------------------------------------------------------


def test_ideal_diode_voc_identity():
    """J(Voc) = 0 with Voc = n·Vt·ln(1 + Jph/J0) — textbook exact."""
    Vt, J0, Jph = 0.02585, 1e-12, 0.02
    voc = Vt * np.log(1.0 + Jph / J0)
    j_at_voc = float(ideal_diode_current(voc, J0=J0, Vt=Vt, Jph=Jph))
    assert abs(j_at_voc) < 1e-15, j_at_voc


def test_ideal_diode_mpp_identity():
    """The analytic MPP solves (1+v)e^{v} = 1 + Jph/J0 (v = V_mpp/Vt).

    Independent of the implementation: Newton on the P = −V·J curve must
    land on the same root, and P there must exceed P at neighbours.
    """
    Vt, J0, Jph = 0.02585, 1e-12, 0.02
    A = 1.0 + Jph / J0
    v_grid = np.linspace(0.0, np.log(A) - 0.5, 4000)
    P = np.asarray(Vt * v_grid) * -ideal_diode_current(Vt * v_grid, J0=J0, Vt=Vt, Jph=Jph)
    i_mpp = int(np.argmax(P))
    # solve (1+v)e^v = A by bisection (root is unique for A > 1)
    lo, hi = 0.0, np.log(A)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (1 + mid) * np.exp(mid) < A:
            lo = mid
        else:
            hi = mid
    v_exact = 0.5 * (lo + hi)
    assert abs(v_grid[i_mpp] - v_exact) < (v_grid[1] - v_grid[0]) * 2


def test_ideal_diode_monotonic_and_saturation():
    """Monotone in V; reverse saturation → −J0; forward doubling identity.

    The exact ideal-diode property J(V + Vt·ln2) = 2·J(V) + J0 is an
    algebraic identity of the exponential law (independent of Vt fitting
    or implementation details) — the "18 mV per doubling" textbook rule.
    """
    Vt, J0, Jph = 0.02585, 1e-12, 0.02
    v = jnp.linspace(0.0, 0.6, 7)
    j = ideal_diode_current(v, J0=J0, Jph=Jph, Vt=Vt)
    assert float(j[0]) == -0.02
    assert float(jnp.min(jnp.diff(j))) > 0
    # dark reverse bias ≫ Vt saturates at −J0
    j_rev = float(ideal_diode_current(-1.0, J0=J0, Jph=0.0, Vt=Vt))
    assert j_rev == pytest.approx(-J0, rel=1e-3)
    # doubling identity (dark, forward): J(V+d) = 2 J(V) + J0, d = Vt ln 2
    V1, d = 0.4, Vt * float(np.log(2.0))
    j1 = float(ideal_diode_current(V1, J0=J0, Jph=0.0, Vt=Vt))
    j2 = float(ideal_diode_current(V1 + d, J0=J0, Jph=0.0, Vt=Vt))
    assert j2 == pytest.approx(2.0 * j1 + J0, rel=1e-10)


# ---------------------------------------------------------------------------
# Simulator-side sanity gates (unchanged from the pre-audit suite)
# ---------------------------------------------------------------------------


def test_canary_voc_below_bandgap(small_cell):
    import jax.numpy as jnp

    from driftjax.science.contacts import boundary_bias, boundary_eq
    from driftjax.solvers.continuation import equilibrium_guess, find_voc
    from driftjax.solvers.newton import solve_eq, solve_newton

    pot_eq = solve_eq(small_cell, boundary_eq(small_cell), equilibrium_guess(small_cell).phi)
    vs = jnp.linspace(0.0, 1.4 / energy, 10)
    js = []
    for vf in vs:
        p = solve_newton(small_cell, boundary_bias(small_cell, vf), pot_eq, tol=1e-08)[0]
        js.append(float(jnp.mean(_jtot(small_cell, p))))
    js = jnp.array(js)
    voc_dim = float(find_voc(vs, js))
    assert 0.0 < voc_dim * energy < 1.5, voc_dim * energy


def _jtot(cell, pot):
    from driftjax.numerics.scharfetter_gummel import Jn, Jp

    return jnp.mean(Jn(cell, pot) + Jp(cell, pot))
