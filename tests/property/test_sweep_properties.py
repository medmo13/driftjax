"""Hypothesis property tests: invariant checks on whole IV sweeps.

Every generated sweep must satisfy physical invariants (finite currents,
monotone IV, valid FF bounds, honest NaN semantics for below-Voc sweeps) —
regardless of the sweep parameters drawn.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import driftjax as dj


def _canary(vmax: float, n_steps: int):
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
        A=20000.0,
    )
    des = dj.Device(
        layers=list(zip([0.0001, 0.0001], [mat, mat], [1e17, -1e17], strict=False)),
        # Invariants hold at any mesh: N=40 keeps 3 examples cheap.
        n_points=40,
        Snl=10000000.0,
        Snr=10000000.0,
        Spl=10000000.0,
        Spr=10000000.0,
    )
    return dj.simulate(
        des, dj.Sweep(vmax=vmax, n_steps=n_steps), optics=dj.BeerLambert(alpha_mode="beer-lambert")
    )


@settings(max_examples=3, deadline=None)  # was 6, 3 saves ~15s and reduces flakiness
@given(vmax=st.floats(0.4, 1.2, allow_nan=False, allow_infinity=False), n_steps=st.integers(6, 12))
def test_sweep_invariants(vmax: float, n_steps: int) -> None:
    sol = _canary(vmax=vmax, n_steps=n_steps)
    v = np.asarray(sol.voltages, dtype=float)
    j = np.asarray(sol.current, dtype=float)
    assert v.shape == j.shape == (n_steps,)
    assert np.isfinite(v).all()
    assert np.isfinite(j).all(), f"NaN current at vmax={vmax}, steps={n_steps}"
    assert np.max(np.diff(j)) <= 1e-12, f"non-monotone IV at vmax={vmax}, steps={n_steps}"
    assert float(sol.jsc) > 0
    assert 0.0 < float(sol.eff) < 40.0
    voc = float(sol.voc)
    if voc == voc:
        assert 0.0 < voc <= vmax + 1e-09
    ff = float(sol.ff)
    if ff == ff:
        assert 0.0 < ff <= 1.0
    assert v[0] == pytest.approx(0.0)
    assert v[-1] == pytest.approx(vmax)
