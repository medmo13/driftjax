"""Publication-quality matplotlib style for driftjax figures.

Two ready-made looks:

* ``journal``      — compact SciencePlots-derived style (serif STIX, inward
                    minor ticks, frame-less legends, 300 dpi). Ideal for inline
                    paper figures and the device-gallery (band / bars / charge /
                    iv) renderers.
* ``presentation`` — a large CMU Serif gallery look (8 x 5 in at
                    300 dpi, 2 pt lines, lightcoral conduction / cornflower
                    valence, outward ticks). Used by the example gallery.

Gallery scripts (examples/research/*.py) use journal style via examples/support.py
(6.4×4.2 / 9.0×3.8, constrained_layout, inward ticks, 300 dpi tight). The
presentation style (8×5, outward ticks, CMU Serif) is for standalone viz.plotting
paper figures and viz/figs.py helpers. Keep the split documented.

The presentation style uses a fixed 8 x 5 inch canvas, black axes, and
semantic carrier colours so figures remain comparable across examples.

Usage:
    from driftjax.viz import style
    style.apply("presentation")          # once, before any plt call
    fig, ax = style.figure(8.0, 5.0)
    ...
    style.save(fig, "fig.png")           # 300 dpi, exact inch size
"""

import os
from typing import Any

RC: dict[str, Any] = {
    # ---- fonts: serif + STIX mathtext (LaTeX-free, Times-like) ----
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Liberation Serif", "Times"],
    "font.size": 9,
    "mathtext.fontset": "stix",
    "mathtext.rm": "STIXGeneral",
    "text.usetex": False,
    "axes.formatter.use_mathtext": True,
    # ---- figure defaults ----
    "figure.figsize": (4.8, 3.6),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.transparent": False,
    # ---- axes ----
    "axes.linewidth": 0.6,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "regular",
    "axes.labelpad": 3.0,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.axisbelow": True,
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "axes.prop_cycle": "cycler('color', ['0C5DA5', '00B945', 'FF9500', "
    "'FF2C00', '845B97', '474747', '9e9e9e'])",
    "axes.grid": False,
    # ---- ticks: inward, both sides, minor on (science style) ----
    "xtick.direction": "in",
    "xtick.top": True,
    "xtick.minor.visible": True,
    "xtick.major.size": 3.5,
    "xtick.minor.size": 1.8,
    "xtick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "xtick.labelsize": 9,
    "ytick.direction": "in",
    "ytick.right": True,
    "ytick.minor.visible": True,
    "ytick.major.size": 3.5,
    "ytick.minor.size": 1.8,
    "ytick.major.width": 0.6,
    "ytick.minor.width": 0.5,
    "ytick.labelsize": 9,
    # ---- lines / markers ----
    "lines.markersize": 4,
    "lines.markeredgewidth": 0.0,
    "errorbar.capsize": 2,
    # ---- legend: frame-less (science style) ----
    "legend.fontsize": 8.5,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.3,
    "legend.loc": "best",
    # ---- grid: optional, subtle ----
    "grid.color": "0.45",
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
}

# ----------------------------------------------------------------------------
# semantic colours (shared across both modes and driftjax.viz.plotting)
# ----------------------------------------------------------------------------
BLUE = "#0C5DA5"   # SciencePlots blue
GREEN = "#00B945"
ORANGE = "#FF9500"
RED = "#FF2C00"
VIOLET = "#845B97"
GRAY = "#474747"

# Semiconductor band-diagram / carrier conventions.
EC = "#F08080"     # lightcoral  (electron / conduction band)
EV = "#6495ED"     # cornflowerblue (hole / valence band)
EF = "#111111"     # Fermi level (dark)
FERMI = "#9a9a9a"  # equilibrium Fermi level

# multi-series categorical palette (distinguishable in colour + greyscale)
SERIES = ["#0C5DA5", "#00B945", "#FF9500", "#845B97", "#FF2C00", "#474747",
          "#18857C", "#B48C2F"]
BAR_PALETTE = ["#D4A017", "#E69F00", "#009E73", "#56B4E9", "#CC79A7"]  # gold/orange/green/blue/pink (print-safe, no yellow-on-white)
# Colorblind-safe Wong/Okabe-Ito series (use when DRIFTJAX_COLORBLIND=1)
COLORBLIND_SERIES = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000", "#999999"]
# retained alias used by driftjax.viz.plotting (material bars)
MAT_COLORS = SERIES
# default sequential colormap for 2-D scans / trajectories
CMAP = "viridis"

# common axis labels
LBL_BIAS = "bias / V"
LBL_J = "current density / mA cm$^{-2}$"
LBL_J_A = "current density / A cm$^{-2}$"
LBL_POS = "position / $\\mu$m"
LBL_ENERGY = "energy / eV"
LBL_DENSITY = "density / cm$^{-3}$"

_FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")


def register_fonts() -> None:
    """Register bundled CMU Serif faces with explicit weight/style (idempotent).

    The CMU TTFs all report weight "normal" in their metadata, so matplotlib
    would otherwise fail to resolve bold/italic and fall back to DejaVu. We
    therefore register hand-built font entries that pin each file to its real
    role (roman=normal, bold-extended=bold, italic=italic).
    """
    from matplotlib import font_manager as fm

    if not os.path.isdir(_FONTS_DIR):
        return
    specs = {
        "cmunrm.ttf": dict(weight="normal", style="normal"),
        "cmunbx.ttf": dict(weight="bold", style="normal"),
        "cmunti.ttf": dict(weight="normal", style="italic"),
    }
    seen = {(fe.name, fe.weight, fe.style) for fe in fm.fontManager.ttflist}
    for fn, meta in specs.items():
        fp = os.path.join(_FONTS_DIR, fn)
        if not os.path.isfile(fp):
            continue
        fm.fontManager.addfont(fp)
        key = ("CMU Serif", meta["weight"], meta["style"])
        if key in seen:
            continue
        fm.fontManager.ttflist.append(
            fm.FontEntry(
                fname=fp, name="CMU Serif", style=meta["style"],
                variant="normal", weight=meta["weight"],
                stretch="normal", size="scalable",
            )
        )
        seen.add(key)


# Large serif gallery look (the classic device-physics "textbook" figure:
# 8x5 in @ 300 dpi, CMU Serif, 2 pt lines, outward ticks, framed legends).
# No tight bbox -> the saved PNG is exactly figsize x dpi (2400 x 1500).
PRESENTATION_RC: dict[str, Any] = {
    "font.family": ["CMU Serif", "STIXGeneral", "DejaVu Serif", "serif"],
    "font.serif": ["CMU Serif", "STIXGeneral", "DejaVu Serif", "Times"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "axes.titleweight": "regular",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.figsize": (8.0, 5.0),
    "figure.dpi": 80,
    "savefig.dpi": 300,
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.0,
    "savefig.transparent": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "lines.markeredgewidth": 0.6,
    "patch.linewidth": 2.0,
    "hatch.linewidth": 2.0,
    "axes.linewidth": 0.8,
    "axes.labelpad": 4.0,
    "axes.prop_cycle": "cycler('color', ['0C5DA5', '00B945', 'FF9500', "
    "'845B97', 'FF2C00', '474747', '18857C', 'B48C2F'])",
    "axes.grid": False,
    # outward ticks (classic textbook look)
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.major.size": 5.0,
    "xtick.minor.size": 2.5,
    "xtick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "xtick.minor.visible": True,
    "ytick.major.size": 5.0,
    "ytick.minor.size": 2.5,
    "ytick.major.width": 0.8,
    "ytick.minor.width": 0.6,
    "ytick.minor.visible": True,
    # framed, rounded legends for presentation figures
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.85",
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.5,
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.3,
    "legend.loc": "best",
}


def apply(mode: str = "journal") -> None:
    """Apply a figure style to the current matplotlib session.

    ``mode="journal"``      — compact SciencePlots-derived style (default).
    ``mode="presentation"`` — large CMU-Serif gallery style.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(RC)  # type: ignore[arg-type]
    if mode == "presentation":
        register_fonts()
        plt.rcParams.update(PRESENTATION_RC)  # type: ignore[arg-type]


# convenience alias matching the gallery renderers
def apply_presentation() -> None:
    """Apply the large presentation figure style."""
    apply("presentation")


def figure(
    w_in: float | None = None,
    h_in: float | None = None,
    ncols: int = 1,
    nrows: int = 1,
    mode: str = "journal",
    constrained: bool = True,
):
    """New styled figure+axes.

    Sizes default to the chosen mode (journal 4.8x3.6, presentation 8x5).
    """
    import matplotlib.pyplot as plt

    apply(mode)
    if w_in is None:
        w_in, h_in = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(
        ncols, nrows, figsize=(w_in, h_in), constrained_layout=constrained
    )
    return fig, ax


def save(fig, path: str, dpi: int = 300) -> str:
    """Save at publication dpi (exact inch size) and close the figure."""
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
