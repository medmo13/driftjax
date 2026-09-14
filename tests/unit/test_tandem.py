"""Pins for the analytic series-connection helper (science/tandem.py).

Uses synthetic diode IVs (no DD solve) so the identities are exact:
  * two IDENTICAL sub-cells  -> tandem curve equals the single-cell curve
    evaluated at HALF the total voltage, and Voc doubles;
  * unequal Jsc              -> I(0) = min(Jsc_top, Jsc_bot);
  * differentiable in both curves' currents; jit-traceable.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from driftjax.science.tandem import series_two_terminal

pytestmark = pytest.mark.smoke

V_T = 0.0259


def diode_iv(jsc_ma, voc, v):
    """Illuminated diode in PACKAGE convention (negative under illumination),
    J = -Jph*(1 - exp((V - Voc)/vt)) in mA/cm^2."""
    return -jsc_ma * (1.0 - jnp.exp((v - voc) / V_T))


V_GRID = jnp.linspace(0.0, 2.4, 241)


# NOTE: sub-cell Voc's are deliberately OFF the 0.01-V grid nodes — an
# exact zero landing on a node makes the crossing interpolation degenerate
# (value right, gradient zero).
VOC_T, VOC_B = 1.103, 0.723


@pytest.fixture(scope="module")
def identical_pair():
    iv = (V_GRID, diode_iv(20.0, VOC_T, V_GRID))
    tandem = series_two_terminal(iv, iv)
    return iv, tandem


def test_identical_cells_double_voc(identical_pair):
    (_, single_j), t = identical_pair
    assert abs(float(t["voc"]) - 2 * VOC_T) < 5e-2


def test_identical_cells_half_voltage_identity(identical_pair):
    """I_tandem(V) must equal I_single(V/2): each junction sits at V/2.
    Compare on the finite prefix only (NaN tail past 2*Voc is by design)."""
    (v_grid, single_j), t = identical_pair
    cur = np.asarray(t["current"])
    m = np.isfinite(cur)
    assert m.any(), "entire tandem curve is NaN"
    half = np.interp(np.asarray(t["voltages"])[m] / 2.0, np.asarray(v_grid), np.asarray(single_j))
    err = float(np.max(np.abs(cur[m] - half)))
    assert err < 5e-3, f"max |dJ| = {err:.2e} mA/cm^2"


def test_current_matching_rule():
    top = (V_GRID, diode_iv(31.0, VOC_T, V_GRID))
    bot = (V_GRID, diode_iv(20.0, VOC_B, V_GRID))
    t = series_two_terminal(top, bot)
    # I(0) = min(Jsc) — limited by the BOTTOM cell here
    assert abs(float(t["isc"]) - 20.0) < 1e-6
    # Voc adds under the ideal junction (crossing-interpolation estimate)
    assert VOC_T + VOC_B - 0.05 < float(t["voc"]) < VOC_T + VOC_B + 0.05


def test_differentiable_in_both_curves():
    """Gradients flow from BOTH input curves into the combined curve.
    Bottom cell is the LIMITING one here, so its scale moves the whole
    limited plateau; the top still enters through the high-voltage tail.
    Objective is a smooth functional of the series current (the voc
    LOCATOR is discrete by design — exact value, no gradient across node
    jumps)."""

    def objective(scale_top, scale_bot):
        top = (V_GRID, diode_iv(31.0, VOC_T * scale_top, V_GRID))
        bot = (V_GRID, diode_iv(20.0, VOC_B * scale_bot, V_GRID))
        cur = series_two_terminal(top, bot)["current"]
        return jnp.nansum(jnp.where(jnp.isfinite(cur), cur, 0.0) ** 2)

    g = jax.grad(objective, argnums=(0, 1))(1.0, 1.0)
    assert np.isfinite(float(g[0])) and np.isfinite(float(g[1]))
    assert abs(float(g[0])) > 1e-6 and abs(float(g[1])) > 1e-6


def test_jit_traceable():
    def run(s):
        top = (V_GRID, diode_iv(20.0 * s, 1.10, V_GRID))
        bot = (V_GRID, diode_iv(31.0, 0.72, V_GRID))
        return series_two_terminal(top, bot)["isc"]

    compiled = jax.jit(run)(1.0)
    assert abs(float(compiled) - float(run(1.0))) < 1e-9
