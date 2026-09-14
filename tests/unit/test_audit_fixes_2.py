"""Regression tests for the v0.1.14 audit fix batch (Task 8).

Covers three audit findings:
* F-14 (HIGH, units): ``monochromatic(wavelength_m)`` stored raw metres into
  the nm-convention ``LightSource.Lambda`` — the constructed source was
  physically transparent (λ = 5e-16 m ⇒ absurd photon energy) and silently
  rendered the cell dark.
* F-15 (MEDIUM, silent misinterpretation): ``material(alpha=..., Lambda=...)``
  stored custom tables verbatim while every consumer reads alpha rows on the
  canonical 200–1400 nm grid — non-canonical tables were silently misread.
* F-16 (LOW, resource hygiene): ``simulate()`` leaked the progress reporter's
  file handle when the sweep raised.
"""

import json

import jax.numpy as jnp
import numpy as np
import pytest

import driftjax as dj
from driftjax.science.optics import photonflux
from driftjax.science.spectrum import monochromatic
from driftjax.units import hc as _hc_J_m

# ---------------------------------------------------------------------------
# F-14: monochromatic metres→nm conversion
# ---------------------------------------------------------------------------


def test_monochromatic_lambda_in_package_nm_convention():
    """API contract: metres in, nm stored (package convention)."""
    ls = monochromatic(5e-07)  # 500 nm, documented API (metres)
    assert float(ls.Lambda[0]) == pytest.approx(500.0, rel=1e-12)


def test_monochromatic_energy_conservation():
    """Independent physics: P_in = φ·E_γ must hold exactly for a mono source.

    E_γ = hc/λ with λ = 500 nm ⇒ E_γ = 3.974e-19 J; φ = 1000/E_γ photons.
    (Would fail under the pre-fix bug: λ=5e-16 m gives a 5e15 eV photon and
    φ ≈ 2.5e21 — energy balance off by 9 orders of magnitude.)
    """
    lam_m = 5e-07
    ls = monochromatic(lam_m)
    E_gamma_J = _hc_J_m / lam_m  # independent constant from units.py
    phi = float(photonflux(ls)[0])
    assert float(ls.P_in[0]) == pytest.approx(phi * E_gamma_J, rel=1e-12)


def test_monochromatic_illuminates_above_bandgap():
    """End-to-end: a 500-nm source must actually generate carriers.

    Regression signature: with the pre-fix bug the table lookup at
    λ = 5e-16 m clamped to the transparent extrapolation and Jsc was
    ~1e-11 A/m² (dark).  Physics: 1000 W/m² at 500 nm ⇒ φ·q = 403 A/m²
    hard upper bound; the thin high-S junction collects a fraction of it.
    Also checks the below-bandgap control (900 nm, hν = 1.38 eV < Eg) is
    genuinely transparent — generation must be spectrally selective.
    """
    mat = dj.material(Eg=1.5, Chi=3.9, eps=9.4, A=2e4)
    des = dj.Device(
        layers=[(1e-4, mat, 1e17), (1e-4, mat, -1e17)],
        n_points=60,
        Snl=1e7,
        Snr=1e7,
        Spl=1e7,
        Spr=1e7,
    )
    jsc_500 = float(
        jnp.abs(dj.simulate(des, dj.Sweep(vmax=0.6, n_steps=4), ls=monochromatic(5e-07)).current[0])
    )
    # hard physical bound: photocurrent ≤ incident photon flux × q
    phi_q = float(photonflux(monochromatic(5e-07))[0]) * 1.602176634e-19
    assert 1e-3 < jsc_500 < phi_q * 1.0001, (
        f"Jsc(500nm)={jsc_500:.3e} outside (1e-3, phi·q={phi_q:.1f}) — "
        "dark-cell signature below 1e-3, unphysical above phi·q"
    )
    jsc_900 = float(
        jnp.abs(dj.simulate(des, dj.Sweep(vmax=0.6, n_steps=4), ls=monochromatic(9e-07)).current[0])
    )
    assert jsc_900 < 1e-4 * jsc_500, (
        f"below-bandgap 900 nm generated Jsc={jsc_900:.3e} — not transparent"
    )


# ---------------------------------------------------------------------------
# F-15: material() canonicalises custom alpha tables
# ---------------------------------------------------------------------------


def _step_alpha_table(lam_nm):
    """Sharp absorption step at 600 nm on an arbitrary grid (50 pts)."""
    return np.where(np.asarray(lam_nm) < 600.0, 1e6, 1e2)


def test_material_custom_alpha_resampled_from_metres():
    lam_nm = np.linspace(300.0, 900.0, 50)
    m = dj.material(Eg=1.5, alpha=_step_alpha_table(lam_nm), Lambda=lam_nm * 1e-9)
    assert m.alpha.shape[0] == 200  # canonical row count
    lam_grid = np.asarray(m.Lambda)
    alpha = np.asarray(m.alpha)
    # in the measured range, away from the log-interp step-edge smoothing
    below = (lam_grid >= 300e-9) & (lam_grid <= 590e-9)
    assert below.sum() > 40
    assert np.all(alpha[below] > 9e5)
    above = (lam_grid >= 612e-9) & (lam_grid <= 900e-9)
    assert above.sum() > 40
    assert np.all(alpha[above] < 2e2)
    # outside the measured range → transparent clamp, not garbage
    assert np.all(alpha[lam_grid < 300e-9 - 1e-12] <= 1e-30 * (1 + 1e-9) + 1e-12)
    assert np.all(alpha[lam_grid > 900e-9 + 1e-12] <= 1e-30 * (1 + 1e-9) + 1e-12)


def test_material_custom_alpha_resampled_from_nm():
    lam_nm = np.linspace(300.0, 900.0, 50)
    m = dj.material(Eg=1.5, alpha=_step_alpha_table(lam_nm), Lambda=lam_nm)  # nm units
    alpha = np.asarray(m.alpha)
    lam_grid = np.asarray(m.Lambda)
    below = (lam_grid >= 300e-9) & (lam_grid <= 590e-9)
    assert np.all(alpha[below] > 9e5)
    above = (lam_grid >= 612e-9) & (lam_grid <= 900e-9)
    assert np.all(alpha[above] < 2e2)


def test_material_canonical_table_untouched():
    """DB-shaped input (200 rows, canonical grid, metres) must pass through."""
    canon_m = np.linspace(200.0, 1400.0, 200) * 1e-9
    a_in = np.full(200, 5e4)
    m = dj.material(Eg=1.5, alpha=a_in, Lambda=canon_m)
    assert np.allclose(np.asarray(m.alpha), a_in)
    assert np.allclose(np.asarray(m.Lambda), canon_m)


# ---------------------------------------------------------------------------
# F-16: simulate() closes the progress reporter on exception paths
# ---------------------------------------------------------------------------


def test_simulate_closes_progress_on_failure(tmp_path, monkeypatch):
    """A mid-sweep exception must still emit the JSONL 'done' event.

    Regression: before the fix the DebugLog handle leaked (no 'done' event,
    unflushed file) when _simulate_sweep raised.
    """
    log_path = tmp_path / "run.jsonl"
    ls = dj.material(Eg=1.5, Chi=3.9, eps=9.4, A=2e4)
    des = dj.Device(
        layers=[(1e-4, ls, 1e17)],
        n_points=20,
        Snl=1e7,
        Snr=1e7,
        Spl=1e7,
        Spr=1e7,
    )

    import importlib

    _sim_mod = importlib.import_module("driftjax.simulate")  # module, not the
    # shadowing function attribute (``driftjax.simulate`` is the function —
    # the package __init__ rebinds the name)
    sim_func = _sim_mod.simulate
    from driftjax import console

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(_sim_mod, "_simulate_sweep", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        sim_func(
            des,
            dj.Sweep(vmax=0.6, n_steps=4),
            progress=console.DebugLog(str(log_path), n_steps=4, vmax_v=0.6),
        )
    lines = [json.loads(x) for x in log_path.read_text().strip().splitlines()]
    assert lines[0]["event"] == "sweep"
    assert lines[-1]["event"] == "done"


def test_simulate_progress_close_success_contract_unchanged():
    """Success path still emits sweep → step… → done (behaviour preserved)."""
    import os
    import tempfile

    from driftjax import console

    mat = dj.material(Eg=1.5, Chi=3.9, eps=9.4, A=2e4)
    des = dj.Device(
        layers=[(1e-4, mat, 1e17)],
        n_points=20,
        Snl=1e7,
        Snr=1e7,
        Spl=1e7,
        Spr=1e7,
    )
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ok.jsonl")
        dj.simulate(
            des, dj.Sweep(vmax=0.6, n_steps=4), progress=console.DebugLog(p, n_steps=4, vmax_v=0.6)
        )
        with open(p) as f:
            lines = [json.loads(x) for x in f.read().strip().splitlines()]
        assert lines[0]["event"] == "sweep"
        assert lines[-1]["event"] == "done"
        assert sum(1 for x in lines if x["event"] == "step") == 4
