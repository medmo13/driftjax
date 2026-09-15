"""Shared utilities for the active scientific example gallery.

Every gallery script follows the same contract:

1. parse ``--output-dir`` via :func:`example_args`,
2. build devices and run simulations,
3. render one publication-style figure with the shared :mod:`driftjax.viz.style`
   vocabulary (colours, labels, sizes),
4. persist a machine-readable JSON sidecar next to the figure, carrying the
   result payload plus execution metadata (versions, backend, platform),
5. print a single ``EXAMPLE_RESULT`` line so CI can assert on outcomes.

The helpers below exist so that no script re-implements boilerplate and all
figures come out in the same professional look.
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

import driftjax as dj
from driftjax.units import energy, length
from driftjax.viz import style

EXAMPLE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = EXAMPLE_ROOT / "outputs"

# Named figure presets (inches) so the whole gallery shares geometry.
SINGLE = dict(width=6.4, height=4.2)  # one panel
DOUBLE_WIDE = dict(width=9.0, height=3.8)  # two panels side by side


def example_args(name: str):
    """Parse common gallery controls (--output-dir).

    The default output directory is ``outputs/`` inside the calling script's
    own tier (``research/``, ``tutorial/`` or ``developer/``), so each tier
    owns its artifacts; ``--output-dir`` still overrides.
    """
    global OUTPUT_ROOT
    try:
        caller_dir = Path(inspect.currentframe().f_back.f_code.co_filename).resolve().parent
    except Exception:
        caller_dir = EXAMPLE_ROOT
    tier_default = caller_dir / "outputs"
    parser = argparse.ArgumentParser(description=f"Run the DriftJax example {name}.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for the PNG and JSON artifacts",
    )
    args = parser.parse_args()
    OUTPUT_ROOT = (args.output_dir or tier_default).expanduser().resolve()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return args


def configure_figures() -> None:
    """Apply the shared journal style."""
    style.apply("journal")


def new_figure(
    *,
    nrows: int = 1,
    ncols: int = 1,
    width: float | None = None,
    height: float | None = None,
):
    """Create a consistently sized gallery figure.

    Defaults to the SINGLE preset; pass ``width``/``height`` (or use the
    DOUBLE_WIDE values) for multi-panel figures.
    """
    configure_figures()
    w = width if width is not None else SINGLE["width"]
    h = height if height is not None else SINGLE["height"]
    return plt.subplots(nrows, ncols, figsize=(w, h), constrained_layout=True)


def save_figure(figure, name: str) -> Path:
    """Save a figure below the output directory and close it."""
    OUTPUT_ROOT.mkdir(exist_ok=True)
    path = OUTPUT_ROOT / f"{name}.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def save_json(name: str, payload: dict) -> Path:
    """Serialize JSON-safe results and execution metadata."""
    OUTPUT_ROOT.mkdir(exist_ok=True)
    path = OUTPUT_ROOT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _append_log(name: str, line: str) -> None:
    """Append one EXAMPLE_RESULT line to per-example and aggregate logs."""
    try:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_ROOT / f"{name}.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        with open(OUTPUT_ROOT / "example_results.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass  # logging must never break the example


def report(name: str, **metrics: float) -> None:
    """Print one machine-parseable summary line for CI and auto-save to file."""
    body = " ".join(f"{k}={v:.6g}" for k, v in metrics.items())
    line = f"EXAMPLE_RESULT {name} {body}"
    print(line, flush=True)
    _append_log(name, line)


def execution_metadata() -> dict:
    """Return environment information needed to reproduce a result."""
    return {
        "driftjax": dj.__version__,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "x64": bool(jax.config.x64_enabled),
        "platform": platform.platform(),
    }


def solution_metrics(solution) -> dict:
    """Extract named physical observables from a ``Solution``."""
    return {
        "efficiency_fraction": float(solution.efficiency),
        "voc_V": float(solution.voc),
        "jsc_A_per_cm2": float(solution.jsc),
        "ff": float(solution.ff),
        "pmax_dimensionless": float(solution.pmax),
    }


def full_fom(solution) -> dict:
    """Full physical figures of merit for one ``Solution``.

    Returns efficiency (%), Voc (V), Jsc (mA cm^-2), fill factor, and the
    MPP power density pmax = Voc * Jsc * FF (W cm^-2).  ``Solution.pmax`` is
    in dimensionless scaled units and is NOT reported directly; the physical
    MPP power is reconstructed from the (already physical) Voc/Jsc/FF so the
    number is unambiguous.
    """
    eff = float(solution.efficiency)
    voc = float(solution.voc)
    jsc = float(solution.jsc)
    ff = float(solution.ff)
    return {
        "eff_percent": eff * 100.0,
        "voc_V": voc,
        "jsc_mA_cm2": jsc * 1e3,
        "ff": ff,
        "pmax_W_cm2": voc * jsc * ff,
    }


def report_fom(name: str, **solutions) -> None:
    """Print one machine-parseable ``EXAMPLE_RESULT`` line with the full FoM
    (efficiency %, Voc, Jsc, FF, MPP power) for every labelled ``Solution``.

    Usage::

        report_fom("research_07", top=top_solution, bottom=best_bottom_solution)

    Emits e.g. ``research_07 top_eff=15.23 top_voc=1.10 top_jsc=17.2
    top_ff=0.81 top_pmax=0.015 ... bottom_eff=...``.  Use one label per
    simulated cell/case (e.g. ``Beer-Lambert``/``TMM``, or ``top``/``bottom``
    for a tandem) so the CI line carries the complete per-cell figure of merit.
    """
    parts = []
    for label, sol in solutions.items():
        f = full_fom(sol)
        parts += [
            f"{label}_eff={f['eff_percent']:.4f}",
            f"{label}_voc={f['voc_V']:.4f}",
            f"{label}_jsc={f['jsc_mA_cm2']:.4f}",
            f"{label}_ff={f['ff']:.4f}",
            f"{label}_pmax={f['pmax_W_cm2']:.6g}",
        ]
    line = f"EXAMPLE_RESULT {name} " + " ".join(parts)
    print(line, flush=True)
    _append_log(name, line)


def run_timed(simulator, *args, **kwargs):
    """Run one simulation and return ``(solution, wall_seconds)``."""
    start = time.perf_counter()
    solution = simulator(*args, **kwargs)
    return solution, time.perf_counter() - start


def si_material():
    """Return the database silicon material used by baseline examples."""
    return dj.load_material("Si")


def pn_device(material, *, points: int = 80, thickness: float = 2e-4, doping: float = 1e17):
    """Build a symmetric, reproducible p-n test device."""
    return dj.Device(
        layers=[(thickness / 2, material, doping), (thickness / 2, material, -doping)],
        n_points=points,
        Snl=1e7,
        Snr=1e7,
        Spl=1e7,
        Spr=1e7,
    )


# ---------------------------------------------------------------------------
# Shared scientific-annotation helpers (gallery-wide figure vocabulary)
# ---------------------------------------------------------------------------


def plot_smooth_iv(ax, v, j, *, color, label=None, n_interp=400, lw=1.6, ms=3.0):
    """Deltapv-style IV: PCHIP-smoothed line plus dots at solved points.

    v in V, j in A/cm² (plotted mA/cm²). Raw straight-segment plots look
    polygonal on coarse sweeps; the smoothed line mirrors ∂PV's
    quintic-spline presentation while the dots keep every solved point
    visible. Returns ``(vd, jd)`` in plotted units.
    """
    vv = np.asarray(v, float)
    jj = np.asarray(j, float) * 1e3
    order = np.argsort(vv)
    vv, jj = vv[order], jj[order]
    try:
        from scipy.interpolate import PchipInterpolator as _Pchip

        vd = np.linspace(vv[0], vv[-1], int(n_interp))
        jd = _Pchip(vv, jj)(vd)
    except Exception:
        vd = np.linspace(vv[0], vv[-1], int(n_interp))
        jd = np.interp(vd, vv, jj)
    ax.plot(vd, jd, color=color, lw=lw, zorder=3, label=label)
    if color is None:
        color = ax.lines[-1].get_color()
    ax.plot(vv, jj, "o", color=color, ms=ms, zorder=4)
    return vd, jd


def tag_panels(axes, labels="abcdefghijklmnop", x=-0.14, y=1.04):
    """Add ``(a)``, ``(b)`` ... tags to a grid of axes (publication convention)."""
    import matplotlib.pyplot as plt

    flat = np.atleast_1d(axes).ravel()
    for ax, lab in zip(flat, labels, strict=False):
        ax.text(
            x,
            y,
            f"({lab})",
            transform=ax.transAxes,
            fontsize=plt.rcParams["axes.titlesize"],
            fontweight="bold",
            va="bottom",
            ha="right",
        )
    return axes


def layer_spans(solution, labels=None):
    """Layer boundary positions (µm) and default layer names for a solution."""
    bounds = [0.0]
    x = np.asarray(solution.cell.x) * float(length) * 1e4  # x*length is cm -> um
    # a layer change shows up as a jump in any per-node material array
    for arr_name in ("Eg", "Chi", "eps"):
        arr = np.asarray(getattr(solution.cell, arr_name))
        jumps = np.flatnonzero(np.abs(np.diff(arr)) > 1e-9)
        if jumps.size:
            bounds = [0.0, *(float(x[j + 1]) for j in jumps), float(x[-1])]
            break
    if bounds[0] != 0.0 or len(bounds) < 2:
        bounds = [float(x[0]), float(x[-1])]
    if labels is None:
        labels = [f"layer {i + 1}" for i in range(len(bounds) - 1)]
    return bounds, labels


def shade_layers(ax, solution, labels=None, alpha=0.05, label_size=7.5):
    """Alternate-shade device layers and (optionally) name them at the top.

    Gives 1-D profile panels immediate physical context (window / absorber /
    contact) without the reader counting grid points.
    """
    bounds, labels = layer_spans(solution, labels)
    for i in range(len(bounds) - 1):
        if i % 2 == 1:
            ax.axvspan(bounds[i], bounds[i + 1], color="0.0", alpha=alpha, lw=0)
    xmax = bounds[-1]
    for i, lab in enumerate(labels):
        # skip labels of very thin layers (they would overlap neighbours)
        if lab and (bounds[i + 1] - bounds[i]) / xmax >= 0.08:
            # pure transAxes: some matplotlib builds blend data-x into
            # get_xaxis_transform(), which would silently misplace the labels
            ax.text(
                0.5 * (bounds[i] + bounds[i + 1]) / xmax,
                0.985,
                lab,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=label_size,
                color="0.25",
            )
    return ax


def mark_iv_points(ax, solution, show=("jsc", "mpp", "voc"), color="0.35", annotate=True):
    """Mark $J_{sc}$, the MPP and $V_{oc}$ on an IV axes (dashed guides + dot).

    ``annotate=False`` draws the guides only (use when the axes title already
    reports the numeric values).
    """
    import matplotlib.pyplot as plt

    v = np.asarray(solution.voltages)
    j = np.asarray(solution.current) * 1e3  # mA/cm^2
    if "jsc" in show:
        ax.axhline(j[0], color=color, linestyle=":", lw=0.7, alpha=0.7)
        if annotate:
            ax.annotate(
                f"$J_{{sc}}$={j[0]:.1f}",
                xy=(0.02, j[0]),
                xycoords=("axes fraction", "data"),
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=7.5,
                color=color,
            )
    if "voc" in show and np.isfinite(float(solution.voc)):
        voc = float(solution.voc)
        ax.axvline(voc, color=color, linestyle=":", lw=0.7, alpha=0.7)
        if annotate:
            ax.annotate(
                f"$V_{{oc}}$={voc:.2f} V",
                xy=(voc, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -12),
                textcoords="offset points",
                fontsize=7.5,
                color=color,
            )
    if "mpp" in show:
        p = v * j
        k = int(np.argmax(p))
        ax.plot(
            v[k],
            j[k],
            "o",
            ms=4.5,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
            zorder=5,
        )
        ax.plot(
            [v[k], v[k]], [ax.get_ylim()[0], j[k]], color=color, lw=0.7, linestyle="--", alpha=0.7
        )
        ax.plot([0, v[k]], [j[k], j[k]], color=color, lw=0.7, linestyle="--", alpha=0.7)
        if annotate:
            ax.annotate(
                f"MPP ({v[k]:.2f} V, {j[k]:.1f})",
                xy=(v[k], j[k]),
                xytext=(6, -10),
                textcoords="offset points",
                fontsize=7.5,
                color=color,
            )
    return ax


def slope_guide(ax, x0, y0, decades=2.0, slope=-2.0, color="0.55", label=None):
    """Reference power-law segment on log-log axes (reference slope, NOT a fit)."""
    if label is not None:
        ax.plot([], [], linestyle="--", lw=1.1, color=color, label=label)
    import math

    # move `decades` down in y along the slope: dy/dlog10(x) = slope, so going
    # DOWN `decades` decades multiplies x by 10^(-decades/slope) (>1 for slope<0).
    x1 = x0 * 10.0 ** (-decades / slope)
    y1 = y0 * 10.0 ** (-decades)
    ax.plot([x0, x1], [y0, y1], linestyle="--", lw=1.1, color=color, zorder=1)
    xm, ym = math.sqrt(x0 * x1), math.sqrt(y0 * y1)
    ax.annotate(
        f"slope {slope:g}",
        xy=(xm, ym),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7.5,
        color=color,
        rotation=0,
    )
    return ax


def iv_knee_inset(ax, solution, width="42%", height="38%", loc="center right"):
    """Zoomed inset on the $V_{oc}$ knee of an IV curve (where meshes/models differ)."""
    v = np.asarray(solution.voltages)
    j = np.asarray(solution.current) * 1e3
    voc = float(solution.voc)
    if not np.isfinite(voc):
        return None
    axins = ax.inset_axes([0.62, 0.08, 0.36, 0.42])
    axins.plot(v, j, lw=1.2, color=ax.lines[0].get_color() if ax.lines else "C0")
    dv = max(0.08 * voc, 1e-3)
    axins.set_xlim(max(voc - 6 * dv, 0.0), min(voc + 2 * dv, v[-1]))
    # Robust y-range: the raw IV tail can diverge towards Voc; exclude the
    # tail beyond 90% of Voc when scaling the axes so the knee stays in view.
    pre = np.abs(j[v < 0.9 * voc])
    j_scale = float(pre.max()) if pre.size else float(np.abs(j).max())
    axins.set_ylim(-0.25 * j_scale, 0.55 * j_scale)
    axins.axhline(0.0, color="0.6", lw=0.5)
    axins.tick_params(labelsize=6.5)
    axins.set_title("knee", fontsize=7, pad=2)
    ax.indicate_inset_zoom(axins, edgecolor="0.7", alpha=0.7)
    return axins


def fit_ylim(ax, margin=0.08, skip_reference=False):
    """Set y-limits tightly around all plotted Line2D data on ``ax``.

    Replaces hardcoded floors like ``max(jsc * 1.25, 18)`` that left huge
    empty space above low-current curves.  Log-scale axes are left alone.
    """
    lo, hi = np.inf, -np.inf
    for line in ax.lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        if y.size == 0:
            continue
        lo = min(lo, float(np.nanmin(y)))
        hi = max(hi, float(np.nanmax(y)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return ax
    if ax.get_yscale() == "log":
        return ax
    rng = (hi - lo) or max(abs(hi), 1.0)
    ax.set_ylim(lo - margin * rng, hi + margin * rng)
    return ax


def iv_metrics_box(ax, solution, loc="upper right", fontsize=7.0):
    """Annotate an IV axes with the deltapv-style figure of merit box.

    eta (PCE), V_oc, J_sc, FF and the MPP point (V, J), all in physical units.
    """
    # physical MPP from the curve itself
    v = np.asarray(solution.voltages)
    j_ma = np.asarray(solution.current) * 1e3
    k = int(np.argmax(v * j_ma))
    eta = float(solution.efficiency) * 100.0
    lines = [
        f"$\\eta$ = {eta:.2f}%",
        f"$V_{{oc}}$ = {float(solution.voc):.3f} V",
        f"$J_{{sc}}$ = {float(solution.jsc) * 1e3:.2f} mA cm$^{{-2}}$",
        f"FF = {float(solution.ff):.3f}",
        f"MPP = ({v[k]:.2f} V, {j_ma[k]:.1f} mA cm$^{{-2}}$)",
    ]
    ax.text(
        0.02,
        0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color="0.15",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.85),
    )
    return ax


def compare_metrics_box(ax, solutions, loc="upper left", fontsize=6.5):
    """Annotate an IV axes with one figure-of-merit block per optics case.

    ``solutions`` maps a display name to a ``Solution``; each entry gets a
    colour-keyed line "eta / Jsc / Voc / FF" using the shared series
    palette order, so all simulated models are visible on the figure (not
    just in the JSON sidecar).
    """
    lines = []
    for name, sol in solutions.items():
        lines.append(
            rf"{name}: $\eta$={float(sol.efficiency) * 100:.2f}%  "
            rf"$J_{{sc}}$={float(sol.jsc) * 1e3:.2f}  "
            rf"$V_{{oc}}$={float(sol.voc):.3f} V  FF={float(sol.ff):.3f}"
        )
    ax.text(
        0.02,
        0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color="0.15",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.88),
    )
    return ax


def plot_bands(ax, solution, labels=None, show_ef=True):
    """deltapv-style equilibrium band diagram on an existing axes (gallery style).

    Includes the electrostatic potential: E_c = -(chi + phi), so the built-in
    band bending of the junction is visible (a pure -chi plot is FLAT and
    physically wrong at equilibrium).
    """
    pos_um = np.asarray(solution.cell.x) * float(length) * 1e4  # cm -> um
    phi = np.asarray(solution.eq_pot.phi)  # dimensionless (Vt units)
    ec = -(np.asarray(solution.cell.Chi) + phi) * float(energy)
    ev = ec - np.asarray(solution.cell.Eg) * float(energy)
    shade_layers(ax, solution, labels=labels)
    ax.plot(pos_um, ec, color=style.EC, lw=1.6, label="$E_c$")
    ax.plot(pos_um, ev, color=style.EV, lw=1.6, label="$E_v$")
    if show_ef:
        ax.axhline(0.0, color=style.FERMI, lw=0.8, linestyle=":", label="$E_f$")
    ax.set(xlabel=style.LBL_POS, ylabel="band energy / eV")
    fit_ylim(ax)
    return ax


def plot_charges(ax, solution, labels=None):
    """deltapv-style carrier/charge densities on an existing axes (log scale).

    Shows n(x), p(x) and the net ionized dopant profile |N_A - N_D| in
    physical cm^-3 — the same content as deltapv's charge plot, in the shared
    gallery style.
    """
    from driftjax.science.carrier_statistics import n as _n
    from driftjax.science.carrier_statistics import p as _p
    from driftjax.units import density as _DENS

    pos_um = np.asarray(solution.cell.x) * float(length) * 1e4  # cm -> um
    dens = float(_DENS)  # scaled densities -> cm^-3 (unit: 1e19 cm^-3)
    n_v = np.asarray(_n(solution.cell, solution.eq_pot)) * dens
    p_v = np.asarray(_p(solution.cell, solution.eq_pot)) * dens
    dop = np.abs(np.asarray(solution.cell.Ndop)) * dens
    floor = 1e3  # cm^-3 plotting floor
    top = max(float(np.max(n_v)), float(np.max(p_v)), float(np.max(dop)))
    shade_layers(ax, solution, labels=labels)
    ax.plot(pos_um, np.maximum(n_v, floor), color=style.BLUE, lw=1.5, label="$n$")
    ax.plot(pos_um, np.maximum(p_v, floor), color=style.RED, lw=1.5, label="$p$")
    ax.plot(
        pos_um,
        np.maximum(dop, floor),
        color=style.GREEN,
        lw=1.2,
        linestyle="--",
        label="$|N_A-N_D|$",
    )
    ax.set(xlabel=style.LBL_POS, ylabel="density / cm$^{-3}$")
    ax.set_yscale("log")
    ax.set_ylim(floor, top * 100)
    ax.legend(frameon=False, fontsize=7.5, handlelength=1.3, loc="center right")
    ax.grid(alpha=0.18, linestyle="--")
    return ax


def plot_layer_bars(ax, solution, fontsize=7.0):
    """deltapv-style material-bars summary: one horizontal bar per layer,
    annotated with bandgap and doping (shared gallery style)."""
    from driftjax.units import energy as _e

    bounds, _ = layer_spans(solution)
    eg = np.asarray(solution.cell.Eg) * float(_e)
    dop = np.asarray(solution.cell.Ndop)
    for i in range(len(bounds) - 1):
        mid = slice(
            np.searchsorted(np.asarray(solution.cell.x) * float(length) * 1e4, bounds[i]),
            np.searchsorted(np.asarray(solution.cell.x) * float(length) * 1e4, bounds[i + 1]),
        )
        mid = slice(max(mid.start, 0), min(mid.stop, eg.size) or eg.size)
        i0 = mid.start if mid.stop > mid.start else 0
        color = style.SERIES[i % len(style.SERIES)]
        ax.barh(
            i,
            bounds[i + 1] - bounds[i],
            left=bounds[i],
            height=0.6,
            color=color,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.text(
            bounds[i] + 0.02 * (bounds[-1] - bounds[0]),
            i,
            f"E$_g$={eg[i0]:.2f} eV, $N$={dop[i0]:.1e} cm$^{{-3}}$",
            va="center",
            ha="left",
            fontsize=fontsize,
            color="0.1",
        )
    ax.set(ylabel="layer", xlabel="depth / $\\mu$m", yticks=range(len(bounds) - 1))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18, linestyle="--")
    return ax


def synthetic_homo_material():
    """Synthetic 1.5 eV homojunction absorber (NOT silicon: real Si Eg=1.12 eV).

    Named after the deltapv ex1_np_junction benchmark geometry whose device
    layout it reproduces; the "Si-like" label is retired to avoid implying
    a calibrated silicon model. Single source of truth shared by 01, 06,
    17 (and ex1_device below).
    """
    return dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, Et=0, tn=1e-8, tp=1e-8, A=2e4
    )


# Backwards-compatibility alias (v0.1.16 and earlier name).
ex1_material = synthetic_homo_material


def ex1_device(n_points: int = 500):
    """The deltapv ex1_np_junction benchmark device (synthetic Eg=1.5 eV
    homojunction, NOT silicon; 2 x 1 um, 1e17/-1e17, Snr = Spl = 0).
    Published benchmark: Jsc ~ 20.2 mA/cm2, Voc ~ 1.05 V, FF ~ 0.88,
    PCE ~ 20%.
    """
    mat = synthetic_homo_material()
    return dj.Device(
        n_points=n_points,
        layers=[(1e-4, mat, 1e17), (1e-4, mat, -1e17)],
        Snl=1e7,
        Snr=0,
        Spl=0,
        Spr=1e7,
    )


def ex2_device(n_points: int = 500):
    """The deltapv ex2_np_hetero benchmark device (CdS 25 nm / CdTe 4 um,
    doping 1e17/-1e15, Sn = 1.16e7).  Published benchmark: PCE ~ 13.3%.
    """
    CdS = dj.material(
        Nc=2.2e18, Nv=1.8e19, Eg=2.4, eps=10, Et=0, mn=100, mp=25, tn=1e-8, tp=1e-13, Chi=4.0, A=1e4
    )
    CdTe = dj.material(
        Nc=8e17, Nv=1.8e19, Eg=1.5, eps=9.4, Et=0, mn=320, mp=40, tn=5e-9, tp=5e-9, Chi=3.9, A=1e4
    )
    return dj.Device(
        n_points=n_points,
        layers=[(2.5e-6, CdS, 1e17), (4e-4, CdTe, -1e15)],
        Snl=1.16e7,
        Snr=1.16e7,
        Spl=1.16e7,
        Spr=1.16e7,
    )


def as_list(values):
    """Convert array-like values into JSON-compatible floats."""
    return np.asarray(values).astype(float).tolist()
