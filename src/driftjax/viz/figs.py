"""General-purpose scientific figure helpers (∂PV-inspired).

Thin, ergonomic layer over :mod:`driftjax.viz.style` for the example gallery.
Applies the publication style once, then gives you ``figure``/``save`` and the
shared colour / label vocabulary so every example plot comes out in the same
professional look (CMU Serif, 8 x 5 in @ 300 dpi, lightcoral conduction /
cornflower valence, outward ticks).

Example
-------
    from driftjax.viz import figs
    figs.apply()                                   # presentation style
    fig, ax = figs.figure(8.0, 5.0)
    ax.plot(v, j * 1e3, color=figs.EF, lw=2)
    ax.set_xlabel(figs.LBL_BIAS)
    ax.set_ylabel(figs.LBL_J)
    figs.save(fig, "iv.png")
"""

from driftjax.viz import style

# re-export the public surface
apply = style.apply
apply_presentation = style.apply_presentation
figure = style.figure
save = style.save

# colour / carrier vocabulary
EC = style.EC
EV = style.EV
EF = style.EF
FERMI = style.FERMI
SERIES = style.SERIES
BAR_PALETTE = style.BAR_PALETTE
CMAP = style.CMAP

# common axis labels (units follow the ∂PV convention)
LBL_BIAS = style.LBL_BIAS
LBL_J = style.LBL_J
LBL_J_A = style.LBL_J_A
LBL_POS = style.LBL_POS
LBL_ENERGY = style.LBL_ENERGY
LBL_DENSITY = style.LBL_DENSITY
