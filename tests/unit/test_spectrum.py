"""AM1.5G spectrum rescale (F-08): exactly 1000 W/m² integrated."""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.smoke  # no solves: spectrum-table math only

from driftjax.science.optics import photonflux
from driftjax.science.spectrum import monochromatic, spectrum, white


def test_am15g_power():
    ls = spectrum()
    assert abs(float(jnp.sum(ls.P_in)) - 1000.0) < 1e-06


def test_am15g_wavelength_range():
    ls = spectrum()
    lam = jnp.asarray(ls.Lambda)
    assert float(jnp.min(lam)) >= 200.0
    assert float(jnp.max(lam)) <= 5000.0


def test_am15g_photonflux_positive():
    assert bool(jnp.all(photonflux(spectrum()) > 0))


def test_white_and_mono():
    ls = white()
    assert float(jnp.max(ls.P_in)) > 0
    m = monochromatic(5e-07)
    assert m.Lambda.size == 1
    assert abs(float(jnp.sum(m.P_in)) - 1000.0) < 1e-06
