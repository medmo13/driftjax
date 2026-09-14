"""Publication-style plots: IV curve, bars,
band diagram, charge.

All functions accept standard calling conventions for driftjax
plotting module (physical units: V / A cm⁻² / eV / µm / cm⁻³); designs and
potentials are passed in their dimensionless internal form and converted
here. matplotlib is imported lazily (Agg), so this module never blocks a
headless solve.

Rendering uses :mod:`driftjax.viz.style` — a SciencePlots-derived
publication style (serif STIX fonts, inward minor ticks, 300 dpi, frame-less
legends) — so every figure comes out in the same journal look.
"""

from __future__ import annotations

import jax.numpy as jnp

from driftjax.units import density, energy, length
from driftjax.viz import style

_COLORS = style.MAT_COLORS

# presentation ("gallery") palette — the classic device-physics hues used by
# the reference ∂PV figures: lightcoral for electrons/conduction band,
# cornflower blue for holes/valence band (single source of truth in style).
EC_SOFT = style.EC  # lightcoral
EV_SOFT = style.EV  # cornflowerblue
FERMI_SOFT = style.FERMI  # light gray Fermi level
BAR_PALETTE = style.BAR_PALETTE


def _best_legend(ax, avoid_lines=True, outside=False):
    """Place the legend where it does not intersect any data line/patch.

    Tries inside locations first, then (optionally) outside-right; frame-less
    by default (style). Returns the legend or None.
    """
    import numpy as _np

    cands = [
        "upper right",
        "upper left",
        "lower right",
        "lower left",
        "center right",
        "center left",
        "upper center",
        "lower center",
    ]
    if outside:
        cands.append("outside")
    for loc in cands:
        if loc == "outside":
            leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        else:
            leg = ax.legend(loc=loc, frameon=False)
        ax.figure.canvas.draw()
        bb = leg.get_window_extent()
        tr = ax.transData
        bad = False
        if avoid_lines:
            for line in ax.lines:
                xd, yd = line.get_xdata(), line.get_ydata()
                if len(xd) < 2:
                    continue
                pts = _np.column_stack([xd, _np.asarray(yd, dtype=float)])
                if any(bb.contains(*(tr.transform(p))) for p in pts[:: max(1, len(pts) // 400)]):
                    bad = True
                    break
        if not bad:
            return leg
        leg.remove()
    return leg


def _plt(mode: str = "journal"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style.apply(mode)
    return plt


def _nice_top(x, step=2.5, margin=1.15):
    """Nice axis top above the data: margin, then round up to a tick step.

    Guarantees a tick mark strictly above the data maximum so the curve
    never visually touches the frame (matplotlib's locator alone may stop
    ticks at/below the data top).
    """
    import math

    return math.ceil(float(x) * margin / step) * step


def _mpp_point(v, j):
    """MPP from the plotted curve, consistent with ``Solution`` FoM.

    Uses the same PCHIP-cubic MPP extraction as the solver
    (``calcPmax_cubic``) so the annotated corner, FF and efficiency match
    ``Solution.eff/voc/ff`` exactly; ``Jmpp`` is interpolated on the drawn
    curve so the marker sits on it. Inputs: V in volts, j in A/cm².
    Returns ``(vmax_V, jmpp_A_per_cm2, pmax_W_per_m2)``.
    """
    import numpy as _np

    from driftjax.numerics.spline import calcPmax_cubic

    vv = _np.asarray(v, float)
    jj = _np.asarray(j, float)
    pmax_wcm2, vmax = calcPmax_cubic(jnp.asarray(vv), jnp.asarray(jj))
    pmax_wcm2, vmax = float(pmax_wcm2), float(vmax)
    try:
        from scipy.interpolate import PchipInterpolator as _Pchip

        jmpp = float(_Pchip(vv, jj)(vmax))
    except Exception:
        jmpp = float(_np.interp(vmax, vv, jj))
    return vmax, jmpp, pmax_wcm2 * 1e4


def _mpp(v, j):
    """(voc, vmax, pmax_wm2, ff) from a physical (V, A/cm2) IV pair."""
    v = jnp.asarray(v)
    j = jnp.asarray(j)
    j0 = float(j[0])
    p = v * j * 1e4  # W/m2
    imax = int(jnp.argmax(p))
    pmax, vmax = float(p[imax]), float(v[imax])
    cross = jnp.where(j[:-1] * j[1:] < 0)[0]
    voc = None
    if cross.size:
        i = int(cross[0])
        vi, vip1 = float(v[i]), float(v[i + 1])
        j1, j2 = float(j[i]), float(j[i + 1])
        voc = vi - j1 * (vip1 - vi) / (j2 - j1)
    ff = pmax / (voc * j0 * 1e4) if voc else None
    return voc, vmax, j0, pmax, ff


def _clip_iv(v, j):
    """Cut the non-converged high-bias tail of an illuminated IV sweep.

    The continuation solver can diverge past the forward knee (|j| blows up
    beyond a few x the short-circuit magnitude) — a documented artifact also
    present in the solver (parity is verified for V < Voc ± margin). Plotting
    that tail would crush the physical curve to a sliver and draw annotations
    on top of the runaway data, so we truncate the curve where it leaves a
    ±40x jsc window and stop the x-axis there. Returns (v, j, v_max_plot).
    """
    v = jnp.asarray(v)
    j = jnp.asarray(j)
    n = int(j.size)
    jsc = abs(float(j[0])) if n else 0.0
    if jsc <= 0 or n < 3:
        return v, j, float(v[jnp.maximum(n - 1, 0)])
    # keep the physical branch: everything up to just past the last zero
    # crossing (the Voc/MPP knee) while |j| stays inside a small multiple of
    # the short-circuit magnitude; drop the non-converged runaway tail.
    flips = jnp.where(jnp.sign(j[:-1]) != jnp.sign(j[1:]))[0]
    cut = n
    for f in flips:
        after = jnp.where(jnp.abs(j[f + 1 :]) > 2.0 * jsc)[0]
        if after.size:
            cut = min(cut, int(f + 1 + after[0]))
    if cut <= 1:
        cut = n
    cut = max(cut, 2)
    return v[:cut], j[:cut], float(v[cut - 1])


def _smooth_iv(v, j, n: int = 800):
    """Monotone (PCHIP) smoothing of a J–V sweep for a smooth, overshoot-free
    knee. Falls back to linear interpolation if SciPy is unavailable.

    Mirrors ∂PV's quintic-spline smoothing of the IV sweep before drawing the
    curve and locating the MPP / Voc.
    """
    import numpy as _np

    v = _np.asarray(v, float)
    j = _np.asarray(j, float)
    order = _np.argsort(v)
    v, j = v[order], j[order]
    if len(v) < 3:
        return v, j
    try:
        from scipy.interpolate import PchipInterpolator as _Pchip

        f = _Pchip(v, j)
        vd = _np.linspace(v[0], v[-1], int(n))
        return vd, f(vd)
    except Exception:
        vd = _np.linspace(v[0], v[-1], int(n))
        return vd, _np.interp(vd, v, j)


def plot_iv_curve(
    v,
    j,
    path: str = "iv.png",
    title: str | None = None,
    flip: bool = False,
    mode: str = "presentation",
    p_in: float | None = None,
) -> str:
    """Physical J–V with MPP/FF annotation (standard convention).

    v in V, j in A/cm²; plotted in mA/cm²; optionally flip the sign of both
    (reverse-bias convention). ``p_in`` is the incident power density in W/m²
    used for the efficiency annotation (default 1000, i.e. 1 sun); pass the
    light source's actual ``sum(P_in)`` when using other spectra.
    """
    if mode == "presentation":
        return _iv_presentation(v, j, path=path, title=title, flip=flip, p_in=p_in)
    plt = _plt(mode)
    from matplotlib.patches import Rectangle

    v = jnp.asarray(v)
    j = jnp.asarray(j)
    if flip:
        v, j = -v, -j
    v, j, vmax_plot = _clip_iv(v, j)
    voc, vmax, j0, pmax, ff = _mpp(v, j)
    eff = pmax / float(p_in if p_in is not None else 1000.0)
    # MPP corner from the plotted curve (standard FF-rectangle convention):
    # interpolate J at Vmpp rather than reusing Jsc.
    import numpy as _np

    jmpp = float(_np.interp(vmax, _np.asarray(v, float), _np.asarray(j, float)))
    fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
    ax.plot(
        v,
        j * 1e3,
        "-o",
        ms=3.2,
        lw=1.3,
        color=style.EF,
        markeredgecolor="white",
        markeredgewidth=0.4,
        zorder=3,
    )
    if ff is not None:
        ax.add_patch(
            Rectangle(
                (0, 0),
                vmax,
                jmpp * 1e3,
                fill=True,
                facecolor="0.9",
                edgecolor="0.45",
                hatch="///",
                lw=0.5,
                zorder=0,
            )
        )
        ax.text(
            0.02,
            0.02,
            f"FF = {ff * 100:.2f} %    \u03b7 = {eff * 100:.2f} %",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            zorder=4,
        )
        if voc is not None:
            ax.axvline(voc, color=style.GRAY, ls=":", lw=0.9, zorder=2)
            ax.text(
                voc,
                j.max() * 1e3,
                f"$V_{{oc}} = {voc:.3f}$ V",
                fontsize=8,
                ha="center",
                va="bottom",
            )
    ax.set_xlim(left=0, right=_nice_top(vmax_plot, step=0.25, margin=1.0))
    ax.set_ylim(0, _nice_top(max(float(j[0]), jmpp) * 1e3))
    ax.set_xlabel("bias / V")
    ax.set_ylabel("current density / mA cm$^{-2}$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if title:
        ax.set_title(title)
    style.save(fig, path)
    return path


def plot_bars(
    design, path: str = "bars.png", title: str | None = None, mode: str = "presentation"
) -> str:
    """Material-bar band diagram from a design alone (material-bar diagram).

    Ec = −χ, Ev = −χ − Eg (eV), EF from doping (intrinsic level + band-filling
    shift), x in µm. Optional contact work-function bars when PhiMl/PhiMr > 0.
    """
    if mode == "presentation":
        return _bars_presentation(design, path=path, title=title)
    from matplotlib.patches import Rectangle

    plt = _plt(mode)
    x_um = jnp.asarray(design.x) * float(length) * 1e4
    ec = -jnp.asarray(design.Chi) * float(energy)
    ev = (-jnp.asarray(design.Chi) - jnp.asarray(design.Eg)) * float(energy)
    nc, nv = jnp.asarray(design.Nc), jnp.asarray(design.Nv)
    eg_d = jnp.asarray(design.Eg)
    ndop = jnp.asarray(design.Ndop)
    ni_d = jnp.sqrt(nc * nv) * jnp.exp(-eg_d / 2.0)
    efi = ec / float(energy) - eg_d / 2.0 + 0.5 * jnp.log(nc / nv)
    ndop_nz = jnp.where(ndop != 0.0, ndop, ni_d)
    defx = jnp.where(
        ndop_nz > 0, jnp.log(jnp.abs(ndop_nz) / ni_d), -jnp.log(jnp.abs(ndop_nz) / ni_d)
    )
    ef = (efi + defx) * float(energy)
    fig, ax1 = plt.subplots(figsize=(4.6, 3.5), constrained_layout=True)
    if float(jnp.asarray(design.PhiMl)) > 0 or float(jnp.asarray(design.PhiMr)) > 0:
        ax1.margins(x=0.2, y=0.4)
    else:
        ax1.margins(y=0.4)
        ax1.set_xlim(0, float(x_um[-1]))
    idx = jnp.concatenate([jnp.array([0]), jnp.argwhere(ec[:-1] != ec[1:]).flatten() + 1])
    uc, uv = ec[idx], ev[idx]
    startx = x_um[idx]
    starty = uv
    height = uc - uv
    width = jnp.diff(jnp.append(startx, x_um[-1]))
    for i in range(int(startx.size)):
        x, y, w, h = float(startx[i]), float(starty[i]), float(width[i]), float(height[i])
        ax1.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor=_COLORS[i % len(_COLORS)],
                edgecolor="none",
                alpha=0.22,
                zorder=0,
            )
        )
        ax1.text(x + w / 2, y + h + 0.06, f"{y + h:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax1.text(x + w / 2, y - 0.06, f"{y:.2f}", ha="center", va="top", fontsize=7.5)
    ax1.plot(x_um, ef, ls="--", color=style.EF, lw=1.1, label="$E_F$", zorder=3)
    if float(jnp.asarray(design.PhiMl)) > 0:
        phim = -float(jnp.asarray(design.PhiMl)) * float(energy)
        xstart, _ = ax1.get_xlim()
        w = -xstart
        ystart = phim - 0.1
        ax1.add_patch(
            Rectangle(
                (xstart, ystart),
                w,
                0.2,
                facecolor=style.RED,
                edgecolor="none",
                alpha=0.25,
                zorder=1,
            )
        )
        ax1.text(xstart + w / 2, ystart + 0.32, f"{phim:.2f}", ha="center", fontsize=7.5)
        ax1.text(xstart + w / 2, ystart - 0.30, "contact", ha="center", va="top", fontsize=7.5)
        ax1.axhline(y=phim, xmin=0, xmax=1 / 7, ls="--", color=style.EF, lw=0.8)
    if float(jnp.asarray(design.PhiMr)) > 0:
        phim = -float(jnp.asarray(design.PhiMr)) * float(energy)
        xstart = float(x_um[-1])
        _, xend = ax1.get_xlim()
        w = xend - xstart
        ystart = phim - 0.1
        ax1.add_patch(
            Rectangle(
                (xstart, ystart),
                w,
                0.2,
                facecolor=style.BLUE,
                edgecolor="none",
                alpha=0.25,
                zorder=1,
            )
        )
        ax1.text(xstart + w / 2, ystart + 0.32, f"{phim:.2f}", ha="center", fontsize=7.5)
        ax1.text(xstart + w / 2, ystart - 0.30, "contact", ha="center", va="top", fontsize=7.5)
        ax1.axhline(y=phim, xmin=6 / 7, xmax=1, ls="--", color=style.EF, lw=0.8)
    ax1.set_xlabel("position / µm")
    ax1.set_ylabel("energy / eV")
    if title:
        ax1.set_title(title)
    _best_legend(ax1, avoid_lines=False)
    style.save(fig, path)
    return path


def plot_band_diagram(
    cell,
    pot,
    path: str = "bands.png",
    title: str | None = None,
    eq: bool = False,
    mode: str = "presentation",
) -> str:
    """Physical band diagram: Ec/Ev and Fermi levels, x in µm, E in eV.

    ``cell`` may be a design or an init_cell (both carry dimensionless Chi/Eg/x);
    ``pot`` is a Potentials from ``simulate()``. At equilibrium (eq=True) a
    single Fermi level −φ is drawn; otherwise the quasi-Fermi levels.
    """
    if mode == "presentation":
        return _band_presentation(cell, pot, path=path, title=title, eq=eq)
    plt = _plt(mode)
    x_um = jnp.asarray(cell.x) * float(length) * 1e4
    ec = (-jnp.asarray(cell.Chi) - jnp.asarray(pot.phi)) * float(energy)
    ev = (-jnp.asarray(cell.Chi) - jnp.asarray(cell.Eg) - jnp.asarray(pot.phi)) * float(energy)
    fig, ax = plt.subplots(figsize=(5.0, 3.75), constrained_layout=True)
    # shaded band gap
    ax.fill_between(x_um, jnp.asarray(ev), jnp.asarray(ec), color="0.4", alpha=0.07, zorder=0)
    ax.plot(x_um, ec, color=style.EC, lw=1.4, label="conduction band", zorder=2)
    ax.plot(x_um, ev, color=style.EV, lw=1.4, label="valence band", zorder=2)
    if eq:
        # Equilibrium = single Fermi level referenced to 0 eV
        # (its eq pot has phi_n = phi_p = 0), and with that convention the bulk
        # offsets are exactly kT*ln(Nc/Nd) / kT*ln(Nv/Na). driftjax keeps the same
        # physical reference: draw the flat E_F = 0 line exactly like previous.
        ax.axhline(0.0, color=style.EF, ls="--", lw=1.0, label="Fermi level", zorder=1)
        ax.text(x_um[-1] * 0.985, 0.0, "$E_F$", color=style.EF, fontsize=9, va="bottom", ha="right")
    else:
        ax.plot(
            x_um,
            jnp.asarray(pot.phi_n) * float(energy),
            color=style.EC,
            ls="-.",
            lw=0.9,
            label="electron quasi-Fermi",
            zorder=1,
        )
        ax.plot(
            x_um,
            jnp.asarray(pot.phi_p) * float(energy),
            color=style.EV,
            ls="-.",
            lw=0.9,
            label="hole quasi-Fermi",
            zorder=1,
        )
    ax.set_xlabel("position / µm")
    ax.set_ylabel("energy / eV")
    if title:
        ax.set_title(title)
    leg = _best_legend(ax)
    if leg is not None:
        try:
            leg.set_ncols(2)
        except Exception:
            pass
    style.save(fig, path)
    return path


def plot_charge(
    cell, pot, path: str = "charge.png", title: str | None = None, mode: str = "presentation"
) -> str:
    """Physical carrier densities n(x), p(x) in cm⁻³, x in µm (log scale)."""
    if mode == "presentation":
        return _charge_presentation(cell, pot, path=path, title=title)
    plt = _plt(mode)
    from driftjax.science.carrier_statistics import n, p

    x_um = jnp.asarray(cell.x) * float(length) * 1e4
    # raw physical densities, NO floor: previous implementation plots n/p directly and its
    # thermal-eq minority carriers reach ~1e-6 cm^-3 (a +1.0 cm^-3 floor would
    # flatten those curves at 1 cm^-3 and visually diverge from previous).
    nd = jnp.asarray(n(cell, pot)) * float(density)
    pd = jnp.asarray(p(cell, pot)) * float(density)
    fig, ax = plt.subplots(figsize=(5.6, 3.75), constrained_layout=True)
    ax.semilogy(x_um, nd, color=style.EC, lw=1.4, label="electrons ($n$)", zorder=2)
    ax.semilogy(x_um, pd, color=style.EV, lw=1.4, label="holes ($p$)", zorder=2)
    ax.set_xlabel("position / µm")
    ax.set_ylabel("density / cm$^{-3}$")
    if title:
        ax.set_title(title)
    _best_legend(ax, outside=True)
    ax.grid(which="both", alpha=0.18)
    style.save(fig, path)
    return path


# ---------------------------------------------------------------------------
# presentation ("gallery") renderers — large CMU-Serif figures in the spirit
# of the classic ∂PV gallery (band / bars / charge / iv), improved: real-MPP
# power rectangle, subtle band-gap shading, runaway-tail clipping on J–V,
# colour-blind-safe soft hues kept from the reference palette.
# ---------------------------------------------------------------------------
def _iv_presentation(v, j, *, path, title, flip, p_in=None) -> str:
    import numpy as _np
    from matplotlib.patches import Rectangle

    plt = _plt("presentation")
    v = jnp.asarray(v)
    j = jnp.asarray(j)
    if flip:
        v, j = -v, -j
    v, j, _ = _clip_iv(v, j)
    vv = _np.asarray(v)
    jj = _np.asarray(j) * 1e3  # mA/cm2

    # monotone (PCHIP) smoothing -> smooth, overshoot-free knee
    vd, jd = _smooth_iv(vv, jj, n=800)

    # MPP corner from the same cubic extraction as Solution (exact match
    # with the annotated efficiency); Jmpp interpolated on the drawn line.
    _vv = jnp.asarray(v)
    _jj = jnp.asarray(j)
    _vmax, _jmpp, pmax = _mpp_point(_vv, _jj)
    vmax, jmax_a = _vmax, _jmpp * 1e3
    voc, _, jsc_a, _, _ = _mpp(v, j)
    ff = pmax / (voc * jsc_a * 1e4) if voc else None
    eff = pmax / float(p_in if p_in is not None else 1000.0)

    fig, ax = plt.subplots()
    ax.plot(vd, jd, color="black", lw=2, zorder=3)
    ax.plot(vv, jj, "o", color="black", ms=5, zorder=4)
    if ff is not None:
        rect = Rectangle(
            (0, 0),
            vmax,
            jmax_a,
            fill=False,
            edgecolor="0.72",
            hatch="/",
            ls="--",
            lw=1.6,
            zorder=1,
        )
        ax.add_patch(rect)
        ax.text(
            vmax / 2,
            jmax_a / 2,
            f"FF = {ff * 100:.2f}%\nMPP = {pmax:.1f} W/m$^2$\n$\\eta$ = {eff * 100:.2f}%",
            ha="center",
            va="center",
            zorder=5,
        )
        ax.plot([vmax], [jmax_a], "o", ms=7, mfc="white", mec="black", mew=1.6, zorder=5)
    ax.set_xlabel("bias / V")
    ax.set_ylabel("current density / mA/cm$^2$")
    # photovoltaic quadrant only (never negative axes), nice round top
    ax.set_xlim(
        0,
        _nice_top(
            max(float(vv[-1]), float(voc) if voc is not None else 0.0), step=0.25, margin=1.0
        ),
    )
    ax.set_ylim(0, _nice_top(max(float(jj[0]), float(jmax_a))))
    if voc is not None:
        ax.annotate(
            f"$V_{{oc}}={voc:.2f}\\,\\mathrm{{V}}$",
            (voc, 0.0),
            xytext=(4, 10),
            textcoords="offset points",
            fontsize=13,
            color="0.35",
        )
    if title:
        ax.set_title(title)
    fig.tight_layout()
    style.save(fig, path)
    return path


def _band_presentation(cell, pot, *, path, title, eq) -> str:
    plt = _plt("presentation")
    x_um = jnp.asarray(cell.x) * float(length) * 1e4
    ec = (-jnp.asarray(cell.Chi) - jnp.asarray(pot.phi)) * float(energy)
    ev = (-jnp.asarray(cell.Chi) - jnp.asarray(cell.Eg) - jnp.asarray(pot.phi)) * float(energy)

    fig, ax = plt.subplots()
    ax.fill_between(x_um, jnp.asarray(ev), jnp.asarray(ec), color="0.4", alpha=0.06, zorder=0)
    ax.plot(x_um, ec, ls="--", lw=2, color=EC_SOFT, label="conduction band", zorder=3)
    ax.plot(x_um, ev, ls="--", lw=2, color=EV_SOFT, label="valence band", zorder=3)
    if eq:
        ax.axhline(0.0, color=FERMI_SOFT, lw=2, label="Fermi level", zorder=2)
    else:
        ax.plot(
            x_um,
            jnp.asarray(pot.phi_n) * float(energy),
            color=EC_SOFT,
            lw=2,
            label="$e^-$ quasi-Fermi energy",
            zorder=2,
        )
        ax.plot(
            x_um,
            jnp.asarray(pot.phi_p) * float(energy),
            color=EV_SOFT,
            lw=2,
            label="$h^+$ quasi-Fermi energy",
            zorder=2,
        )
    ax.set_xlabel("position / µm")
    ax.set_ylabel("energy / eV")
    ax.set_xlim(0, float(x_um[-1]))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    style.save(fig, path)
    return path


def _bars_presentation(design, *, path, title) -> str:
    from matplotlib.patches import Rectangle

    plt = _plt("presentation")
    ec = -jnp.asarray(design.Chi) * float(energy)
    ev = (-jnp.asarray(design.Chi) - jnp.asarray(design.Eg)) * float(energy)
    nc, nv = jnp.asarray(design.Nc), jnp.asarray(design.Nv)
    eg_d = jnp.asarray(design.Eg)
    ndop = jnp.asarray(design.Ndop)
    ni_d = jnp.sqrt(nc * nv) * jnp.exp(-eg_d / 2.0)
    efi = ec / float(energy) - eg_d / 2.0 + 0.5 * jnp.log(nc / nv)
    ndop_nz = jnp.where(ndop != 0.0, ndop, ni_d)
    defx = jnp.where(
        ndop_nz > 0, jnp.log(jnp.abs(ndop_nz) / ni_d), -jnp.log(jnp.abs(ndop_nz) / ni_d)
    )
    ef = (efi + defx) * float(energy)
    x_um = jnp.asarray(design.x) * float(length) * 1e4

    phil = float(jnp.asarray(design.PhiMl))
    phir = float(jnp.asarray(design.PhiMr))
    fig, ax1 = plt.subplots()
    if phil > 0 and phir > 0:
        ax1.margins(x=0.2, y=0.5)
    else:
        ax1.margins(y=0.5)
        ax1.set_xlim(0, float(x_um[-1]))

    idx = jnp.concatenate([jnp.array([0]), jnp.argwhere(ec[:-1] != ec[1:]).flatten() + 1])
    uc, uv = ec[idx], ev[idx]
    startx = x_um[idx]
    width = jnp.diff(jnp.append(startx, x_um[-1]))
    for i in range(int(startx.size)):
        x, y = float(startx[i]), float(uv[i])
        w, h = float(width[i]), float(uc[i] - uv[i])
        ax1.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor=BAR_PALETTE[i % len(BAR_PALETTE)],
                linewidth=0,
                alpha=0.25,
                zorder=0,
            )
        )
        ax1.text(x + w / 2, y + h + 0.10, f"{y + h:.1f}", ha="center", va="bottom", fontsize=12)
        ax1.text(x + w / 2, y - 0.10, f"{y:.1f}", ha="center", va="top", fontsize=12)
    ax1.plot(x_um, ef, ls="--", color="black", lw=2, label="$E_F$", zorder=3)

    for phim_val, side, col in ((phil, "l", "#d62728"), (phir, "r", "#1f77b4")):
        if phim_val <= 0:
            continue
        phim = -phim_val * float(energy)
        if side == "l":
            xstart, _ = ax1.get_xlim()
            w = -xstart
        else:
            xstart = float(x_um[-1])
            _, xend = ax1.get_xlim()
            w = xend - xstart
        ystart = phim - 0.1
        ax1.add_patch(
            Rectangle((xstart, ystart), w, 0.2, facecolor=col, linewidth=0, alpha=0.25, zorder=1)
        )
        ax1.text(xstart + w / 2, ystart + 0.32, f"{phim:.1f}", ha="center", fontsize=12)
        ax1.text(xstart + w / 2, ystart - 0.30, "contact", ha="center", va="top", fontsize=12)
        if side == "l":
            ax1.axhline(y=phim, xmin=0.0, xmax=1 / 7, ls="--", color="black", lw=2)
            ax1.axvline(float(x_um[0]), color="white", lw=2)
            ax1.axvline(float(x_um[0]), color="lightgray", lw=2, ls="dashed")
        else:
            ax1.axhline(y=phim, xmin=6 / 7, xmax=1.0, ls="--", color="black", lw=2)
            ax1.axvline(float(x_um[-1]), color="white", lw=2)
            ax1.axvline(float(x_um[-1]), color="lightgray", lw=2, ls="dashed")
    pos = jnp.argwhere(ndop[:-1] != ndop[1:]).flatten()
    for i in pos:
        vmid = (float(x_um[i]) + float(x_um[i + 1])) / 2
        ax1.axvline(vmid, color="white", lw=4)
        ax1.axvline(vmid, color="lightgray", lw=2, ls="dashed")
    ax1.set_ylim(float(jnp.min(uv)) * 1.5, 0)
    ax1.set_xlabel("position / µm")
    ax1.set_ylabel("energy / eV")
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    style.save(fig, path)
    return path


def _charge_presentation(cell, pot, *, path, title) -> str:
    plt = _plt("presentation")
    from driftjax.science.carrier_statistics import n, p

    x_um = jnp.asarray(cell.x) * float(length) * 1e4
    nd = jnp.asarray(n(cell, pot)) * float(density)
    pd = jnp.asarray(p(cell, pot)) * float(density)
    fig, ax = plt.subplots()
    ax.semilogy(x_um, nd, color=EC_SOFT, lw=2, label="electron")
    ax.semilogy(x_um, pd, color=EV_SOFT, lw=2, label="hole")
    ax.set_xlabel("position / µm")
    ax.set_ylabel("density / cm$^{-3}$")
    ax.set_xlim(0, float(x_um[-1]))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    style.save(fig, path)
    return path


def _layer_spans(x_um, key):
    """Contiguous index spans where ``key`` is constant (for shading)."""
    import numpy as _np

    k = _np.asarray(key)
    change = _np.concatenate([[0], _np.argwhere(k[:-1] != k[1:]).flatten() + 1, [len(k)]])
    return [(int(change[i]), int(change[i + 1])) for i in range(len(change) - 1)]


def plot_dossier(
    source,
    pot=None,
    *,
    iv=None,
    path: str = "dossier.png",
    title: str | None = None,
    show_qfl: bool = False,
) -> str:
    """One-figure 2×2 device dossier: IV + bars + bands + charge.

    The canonical publication panel set (mirrors the ∂PV gallery layout):
    top-left J–V with MPP/FF/Voc annotation, top-right layer band-offset
    bars with equilibrium Fermi level, bottom-left band diagram (``pot``,
    default equilibrium; ``show_qfl=True`` overlays quasi-Fermi levels),
    bottom-right carrier densities (log scale). Spatial panels share the
    µm axis with light layer shading; a metrics strip reports eff/Voc/Jsc/FF.

    ``source`` may be a :class:`~driftjax.solution.Solution` (everything
    derived from it) or a cell/design object. Physical units throughout
    (V, A/cm², eV, µm, cm⁻³); internal dimensionless fields converted here.
    """
    from matplotlib.patches import Rectangle

    # deltapv-style presentation look (CMU Serif, lw 2, outward ticks):
    # the gallery dossier is the visual twin of the ∂PV figure set.
    plt = _plt("presentation")
    cell = getattr(source, "cell", source)
    user_pot = pot is not None
    if not user_pot:
        pot = getattr(source, "eq_pot", None)
    if iv is None and hasattr(source, "voltages"):
        iv = (source.voltages, source.current)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    ax_iv, ax_bars, ax_band, ax_chg = axes.flat
    fs = 11.0

    # ---- Panel A: J–V ----
    # FoM convention: the Solution's own (eff, voc, ff) are canonical
    # (P_in-aware efficiency). Geometry (Vmpp, Jmpp, curve) comes from the
    # plotted data via the standard max(V*J) MPP rule.
    _fom = None
    if hasattr(source, "eff"):
        try:
            _fom = (
                float(source.voc),
                float(source.ff),
                float(source.efficiency),
                float(source.jsc) * 1e3,
            )
        except Exception:
            _fom = None
    if iv is not None:
        import numpy as _np

        v, j = _clip_iv(*iv)[:2]
        vv = _np.asarray(v, float)
        jj = _np.asarray(j, float)
        # Draw the full clipped sweep (including the first sub-zero points)
        # so the curve visibly crosses J=0 at Voc; the ylim floor at 0
        # clips the deep tail. Masking J<0 here amputated the knee.
        voc, _, j0, _, ff_curve = _mpp(vv, jj)
        # deltapv treatment: PCHIP-smoothed black line, dots at solved
        # points, MPP corner from the same cubic extraction as Solution.
        vd, jd = _smooth_iv(vv, jj * 1e3)
        _vm, _jm, pmax_wm2 = _mpp_point(vv, jj)
        vmpp, jmpp_ma = _vm, _jm * 1e3
        if _fom is not None:
            voc, ff, eff, _ = _fom
        else:
            eff = pmax_wm2 / 1000.0
        ax_iv.plot(vd, jd, color="black", lw=2, zorder=3)
        ax_iv.plot(vv, jj * 1e3, "o", color="black", ms=4, zorder=4)
        if ff is not None and _np.isfinite(ff):
            ax_iv.add_patch(
                Rectangle(
                    (0, 0),
                    vmpp,
                    jmpp_ma,
                    fill=False,
                    edgecolor="lightgray",
                    hatch="/",
                    ls="--",
                    lw=1.6,
                    zorder=1,
                )
            )
            ax_iv.text(
                vmpp / 2,
                jmpp_ma / 2,
                f"FF = {ff * 100:.2f}%\nMPP = {pmax_wm2:.1f} W/m$^2$\n$\\eta$ = {eff * 100:.2f}%",
                ha="center",
                va="center",
                fontsize=fs,
                zorder=5,
            )
            ax_iv.plot([vmpp], [jmpp_ma], "o", ms=7, mfc="white", mec="black", mew=1.6, zorder=5)
            if voc is not None and _np.isfinite(voc):
                ax_iv.annotate(
                    f"$V_{{oc}}={voc:.2f}$ V",
                    (voc, 0.0),
                    xytext=(4, 10),
                    textcoords="offset points",
                    fontsize=fs - 1,
                    color="0.35",
                )
        ax_iv.set_xlim(
            0,
            _nice_top(
                max(float(vv[-1]), float(voc) if voc is not None else 0.0), step=0.25, margin=1.0
            ),
        )
        ax_iv.set_ylim(0, _nice_top(max(float(jj[0]) * 1e3, jmpp_ma)))
        ax_iv.set_xlabel("bias / V", fontsize=fs)
        ax_iv.set_ylabel("current density / mA cm$^{-2}$", fontsize=fs)
        ax_iv.tick_params(labelsize=fs - 1)
    ax_iv.text(
        0.02,
        0.90,
        "(a) J–V",
        transform=ax_iv.transAxes,
        fontsize=fs + 1,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # ---- shared spatial grid ----
    x_um = jnp.asarray(cell.x) * float(length) * 1e4
    x0, x1 = 0.0, float(x_um[-1])
    try:
        spans = _layer_spans(x_um, getattr(cell, "Eg", None))
        shade_key = getattr(cell, "Eg", None) is not None
    except Exception:
        spans, shade_key = [], False

    def _shade(ax):
        if not shade_key:
            return
        for k, (i0, i1) in enumerate(spans):
            if k % 2 == 1:
                ax.axvspan(
                    float(x_um[i0]),
                    float(x_um[min(i1, len(x_um) - 1)]),
                    color="0.5",
                    alpha=0.06,
                    zorder=0,
                    lw=0,
                )

    # ---- Panel B: layer bars ----
    ec_b = -jnp.asarray(cell.Chi) * float(energy)
    ev_b = (-jnp.asarray(cell.Chi) - jnp.asarray(cell.Eg)) * float(energy)
    nc, nv = jnp.asarray(cell.Nc), jnp.asarray(cell.Nv)
    ni_d = jnp.sqrt(nc * nv) * jnp.exp(-jnp.asarray(cell.Eg) / 2.0)
    efi = ec_b / float(energy) - jnp.asarray(cell.Eg) / 2.0 + 0.5 * jnp.log(nc / nv)
    ndop = getattr(cell, "Ndop", jnp.zeros_like(nc))
    ndop_nz = jnp.where(ndop != 0.0, ndop, ni_d)
    ef_b = (
        efi
        + jnp.where(
            ndop_nz > 0, jnp.log(jnp.abs(ndop_nz) / ni_d), -jnp.log(jnp.abs(ndop_nz) / ni_d)
        )
    ) * float(energy)
    idx = jnp.concatenate([jnp.array([0]), jnp.argwhere(ec_b[:-1] != ec_b[1:]).flatten() + 1])
    for i, j_ in enumerate(range(int(idx.size))):
        xs = float(x_um[int(idx[j_])])
        xe = float(x_um[-1]) if j_ == int(idx.size) - 1 else float(x_um[int(idx[j_ + 1])])
        yb, yt = float(ev_b[int(idx[j_])]), float(ec_b[int(idx[j_])])
        ax_bars.add_patch(
            Rectangle(
                (xs, yb),
                xe - xs,
                yt - yb,
                facecolor=BAR_PALETTE[i % len(BAR_PALETTE)],
                linewidth=0,
                alpha=0.25,
                zorder=0,
            )
        )
        # deltapv-style band-edge value labels per layer
        ax_bars.text(
            (xs + xe) / 2, yt + 0.05, f"{yt:.2f}", ha="center", va="bottom", fontsize=fs - 2
        )
        ax_bars.text((xs + xe) / 2, yb - 0.05, f"{yb:.2f}", ha="center", va="top", fontsize=fs - 2)
    ax_bars.plot(x_um, ef_b, ls="--", color="black", lw=2, label="$E_F$", zorder=3)
    ax_bars.legend(frameon=False, fontsize=fs - 1)
    ax_bars.set_xlim(x0, x1)
    ax_bars.set_ylim(float(jnp.min(ev_b)) * 1.5, 0)
    ax_bars.set_xlabel("position / µm", fontsize=fs)
    ax_bars.set_ylabel("energy / eV", fontsize=fs)
    ax_bars.tick_params(labelsize=fs - 1)
    ax_bars.text(
        0.02,
        0.95,
        "(b) layers",
        transform=ax_bars.transAxes,
        fontsize=fs + 1,
        fontweight="bold",
        va="top",
    )

    # ---- Panel C: band diagram ----
    if pot is not None:
        ec = (-jnp.asarray(cell.Chi) - jnp.asarray(pot.phi)) * float(energy)
        ev = (-jnp.asarray(cell.Chi) - jnp.asarray(cell.Eg) - jnp.asarray(pot.phi)) * float(energy)
        _shade(ax_band)
        ax_band.fill_between(
            x_um, jnp.asarray(ev), jnp.asarray(ec), color="0.4", alpha=0.06, zorder=0
        )
        ax_band.plot(x_um, ec, ls="--", lw=2, color=EC_SOFT, label="$E_c$", zorder=3)
        ax_band.plot(x_um, ev, ls="--", lw=2, color=EV_SOFT, label="$E_v$", zorder=3)
        if show_qfl and user_pot:
            ax_band.plot(
                x_um,
                jnp.asarray(pot.phi_n) * float(energy),
                color=EC_SOFT,
                lw=2,
                label="$E_{Fn}$",
                zorder=2,
            )
            ax_band.plot(
                x_um,
                jnp.asarray(pot.phi_p) * float(energy),
                color=EV_SOFT,
                lw=2,
                label="$E_{Fp}$",
                zorder=2,
            )
        else:
            ax_band.axhline(0.0, color=FERMI_SOFT, lw=2, label="$E_F$", zorder=2)
        ax_band.set_xlim(x0, x1)
        ax_band.set_xlabel("position / µm", fontsize=fs)
        ax_band.set_ylabel("energy / eV", fontsize=fs)
        ax_band.tick_params(labelsize=fs - 1)
        ax_band.legend(frameon=False, fontsize=fs - 1)
    ax_band.text(
        0.02,
        0.95,
        "(c) bands",
        transform=ax_band.transAxes,
        fontsize=fs + 1,
        fontweight="bold",
        va="top",
    )

    # ---- Panel D: charge ----
    if pot is not None:
        from driftjax.science.carrier_statistics import n, p

        nd = jnp.asarray(n(cell, pot)) * float(density)
        pd = jnp.asarray(p(cell, pot)) * float(density)
        _shade(ax_chg)
        ax_chg.semilogy(x_um, nd, color=EC_SOFT, lw=2, label="$n$")
        ax_chg.semilogy(x_um, pd, color=EV_SOFT, lw=2, label="$p$")
        ax_chg.set_xlim(x0, x1)
        ax_chg.set_xlabel("position / µm", fontsize=fs)
        ax_chg.set_ylabel("density / cm$^{-3}$", fontsize=fs)
        ax_chg.tick_params(labelsize=fs - 1)
        ax_chg.legend(frameon=False, fontsize=fs - 1)
    ax_chg.text(
        0.02,
        0.95,
        "(d) charge",
        transform=ax_chg.transAxes,
        fontsize=fs + 1,
        fontweight="bold",
        va="top",
    )

    if title:
        fig.suptitle(title, fontsize=fs + 2)
    style.save(fig, path)
    return path


def plot_all(
    source,
    pot=None,
    *,
    iv=None,
    outdir: str = "plots",
    mode: str = "presentation",
) -> dict:
    """Render the full device gallery (bars / band / charge / iv) to ``outdir``.

    ``source`` may be a :class:`~driftjax.solution.Solution` (everything is
    derived from it) or any cell/design object carrying ``x/Chi/Eg/Nc/Nv/
    Ndop/PhiMl/PhiMr``.  Pass ``pot`` to draw the band diagram + densities at
    that operating point (default: the equilibrium potential of a Solution,
    else equilibrium bands).  Pass ``iv=(voltages, currents)`` (physical
    units, V and A/cm2) to render the J–V panel.
    Returns the mapping panel-name -> written path.
    """
    import os

    cell = getattr(source, "cell", source)
    user_pot = pot is not None
    if not user_pot:
        pot = getattr(source, "eq_pot", None)  # equilibrium bands by default
    if iv is None and hasattr(source, "voltages"):
        iv = (source.voltages, source.current)

    os.makedirs(outdir, exist_ok=True)
    out: dict = {}
    out["bars"] = plot_bars(cell, f"{outdir}/bars.png", mode=mode)
    if pot is not None:
        out["band"] = plot_band_diagram(cell, pot, f"{outdir}/band.png", eq=not user_pot, mode=mode)
        out["charge"] = plot_charge(cell, pot, f"{outdir}/charge.png", mode=mode)
    if iv is not None:
        try:
            import numpy as _np2

            _pin = float(_np2.sum(_np2.asarray(source.P_in, float)))
        except Exception:
            _pin = None
        out["iv"] = plot_iv_curve(iv[0], iv[1], f"{outdir}/iv.png", mode=mode, p_in=_pin)
    return out
