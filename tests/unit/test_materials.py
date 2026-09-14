"""Material objects: defaults, library loads, alpha-from-k (F-06)."""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.smoke  # no solves: DB loading + validation only

import driftjax as dj
import driftjax.io


def test_create_material_defaults():
    m = dj.material()
    for k in ("Chi", "Eg", "eps", "Nc", "Nv", "mn", "mp", "tn", "tp", "Et", "Br", "Cn", "Cp", "A"):
        assert hasattr(m, k), k
    assert m.alpha.size == 0


def test_load_material_si():
    si = dj.load_material("Si")
    assert si.alpha.size == 200  # canonical 200-1400 nm resampled grid (io.py)
    assert 1.0 < si.Eg < 1.3
    assert float(jnp.max(si.alpha)) > 100000.0
    assert float(jnp.min(si.alpha)) >= 0.0


def test_alpha_from_k_column():
    from driftjax.io import _load_alpha_table

    lam, alpha = _load_alpha_table("Si")
    assert lam.shape == alpha.shape
    i = int(jnp.argmin(jnp.abs(lam - 500.0)))
    assert 500000.0 < float(alpha[i]) < 5000000.0


def test_multiple_materials():
    names = dj.io.list_materials()
    assert {"Si", "GaAs", "Ge"} <= set(names)
    for nm in ("Ge", "GaAs", "AlN", "GaP", "InP", "GaSb", "InAs", "InSb", "BN", "GaN", "InN", "Si"):
        m = dj.load_material(nm)
        assert m.Eg >= 0.0
