"""Two-terminal series connection of independently simulated sub-cells.

A monolithic tandem needs two DD solves coupled through an internal
recombination junction — machinery DriftJax does not have.  What every
PV modelling tool (and this module) implements instead is the STANDARD
analytic series connection of two *independently simulated* sub-cells
under an IDEAL tunnel/recombination junction:

    per total voltage V, find the split v_t solving the CURRENT-
    CONTINUITY condition  |I_top|(v_t) = |I_bot|(V - v_t),
    restricted to splits where BOTH sub-cells deliver (delivery mode).

Because delivery currents rise monotonically toward their own Voc, the
difference D(v_t) = |I_top|(v_t) - |I_bot|(V-v_t) is strictly increasing,
so the crossing is unique and robust:

    * D changes sign  -> linear interpolation onto D = 0;
    * D > 0 everywhere -> bottom-limited: I = |I_bot| at the FIRST split;
    * D < 0 everywhere -> top-limited:    I = |I_top| at the LAST split.

This reproduces the textbook rules  Jsc_tandem = min(Jsc), Voc_tandem =
Voc_top + Voc_bot, and degrades gracefully past either sub-cell's own Voc.

SIGN CONVENTION: inputs follow the package convention (currents NEGATIVE
under illumination); each curve is normalized by the sign of its own
J(0), so devices whose contact ordering flips the sign work identically.
The returned curve uses the TOP cell's sign convention.

ASSUMPTIONS (documented, not hidden):
  * lossless, zero-resistance recombination junction;
  * each sub-cell IV was computed under ITS OWN illumination — optical
    filtering of the transmitted spectrum into the bottom cell is NOT
    modeled here (pass pre-filtered designs/light sources if needed);
  * input voltage grids must be ascending.  Total voltages above
    ~Voc_top + Voc_bot exceed the usable range and are returned as NaN.

Everything is pure-jnp: differentiable in both curves' currents,
`jax.jit`-able, `vmap`-safe.  Returned scalars are 0-d arrays — call
``float(...)`` outside of traces.
"""
from __future__ import annotations

import jax.numpy as jnp


def series_two_terminal(iv_top, iv_bottom, v_out=None):
    """Series-connect two simulated sub-cells into one 2-terminal IV curve.

    Parameters
    ----------
    iv_top, iv_bottom : tuple(jnp.ndarray, jnp.ndarray)
        ``(voltages, currents)``, ascending voltage, physical units.  Either
        sign convention is accepted; the curve is normalised to
        delivery-positive internally and reported in the TOP cell's sign
        convention.
    v_out : jnp.ndarray, optional
        Total-voltage grid for the returned curve.  Defaults to the top
        cell's voltage grid.

    Returns
    -------
    dict of 0-d/array jnp values: ``voltages``, ``current``, ``isc``,
    ``voc``, ``pmax``, ``vmpp``.  ``pmax`` is ``Vmpp * Jmpp`` in the units
    of the inputs (V * A/cm^2 -> W/cm^2 for driftjax sweeps); multiply by
    1e4 and divide by the input power in W/m^2 for an efficiency fraction.

    Method
    ------
    Current-parametrised series connection: with both sub-curves
    normalised delivery-positive and monotone in the first quadrant, the
    series constraint "same current through both" reads
    ``W(J) = V_top(J) + V_bottom(J)`` -- a decreasing function of the
    common current J.  The tandem operating point at total voltage V is
    the J solving ``W(J) = V``; below the current-matched knee
    (``V < min W``) the limiting sub-cell saturates and the curve is the
    flat plateau ``J = min(Jsc_top, Jsc_bot)``; above ``W(0) = Voc_top +
    Voc_bot`` there is no solution and the curve is NaN.  The maximum
    power point is taken on the parametric ``J * W(J)`` table, so it does
    not depend on the requested output grid.
    """
    vt, it = iv_top
    vb, ib = iv_bottom
    vt, it = jnp.asarray(vt), jnp.asarray(it)
    vb, ib = jnp.asarray(vb), jnp.asarray(ib)

    if v_out is None:
        v_out = vt
    v_out = jnp.asarray(v_out)

    # normalise into delivery-positive space using each curve's own J at
    # its first (lowest-voltage) sample
    s_t = jnp.where(it[0] != 0.0, jnp.sign(it[0]), 1.0)
    s_b = jnp.where(ib[0] != 0.0, jnp.sign(ib[0]), 1.0)
    pt_tab = it * s_t
    pb_tab = ib * s_b

    def v_of_j(v_grid, j_tab, j_query):
        """Cell voltage at a common current J (ascending-J interpolation)."""
        order = jnp.argsort(j_tab)
        return jnp.interp(j_query, j_tab[order], v_grid[order])

    j_max = jnp.minimum(jnp.max(pt_tab), jnp.max(pb_tab))
    # Sample the common-current grid where the data actually lives: the
    # union of both curves' own current samples (exact at their nodes,
    # dense on their knees) plus a uniform fill for coverage.  Values are
    # clipped into (0, j_max) to keep the shape static under jit.
    lo = j_max * 1e-9
    hi = j_max * (1.0 - 1e-9)
    fill = jnp.linspace(lo, hi, 256)
    from_curve = jnp.clip(jnp.concatenate([pt_tab, pb_tab]), lo, hi)
    j_grid = jnp.sort(jnp.concatenate([fill, from_curve]))
    w_grid = v_of_j(vt, pt_tab, j_grid) + v_of_j(vb, pb_tab, j_grid)

    order_w = jnp.argsort(w_grid)          # ascending total voltage
    w_asc = w_grid[order_w]
    j_asc = j_grid[order_w]

    plateau = j_asc[0]                     # current-matched saturation value
    # tandem curve: NaN above Voc_sum, parametric inside, plateau below
    # jnp.interp: left applies for v < min W (below the knee -> plateau),
    # right for v > max W (past Voc_sum -> no solution)
    current_norm = jnp.interp(v_out, w_asc, j_asc, left=plateau, right=jnp.nan)

    current = current_norm * s_t           # report in the TOP cell's convention

    # scalar quantities from the parametric table (grid-independent MPP)
    isc = plateau                          # I(0): limited sub-cell saturated
    voc = w_asc[-1]                        # J -> 0 end of the parametric curve
    p_table = j_grid * w_grid              # same units as V * J of the inputs
    impp = jnp.argmax(p_table)
    pmax = p_table[impp]
    vmpp = w_grid[impp]
    return {
        "voltages": v_out,
        "current": current,
        "isc": isc,
        "voc": voc,
        "pmax": pmax,
        "vmpp": vmpp,
    }
