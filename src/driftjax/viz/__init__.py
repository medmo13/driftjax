"""Visualisation layer (matplotlib is imported lazily — never required)."""

from driftjax.viz import (
    figs,  # noqa: F401  (general-purpose figure helpers)
    io,  # noqa: F401
    style,  # noqa: F401
)
from driftjax.viz.io import save_results  # noqa: F401
from driftjax.viz.optim import convergence, parameters  # noqa: F401
from driftjax.viz.plotting import (  # noqa: F401
    plot_all,
    plot_band_diagram,
    plot_bars,
    plot_charge,
    plot_dossier,
    plot_iv_curve,
)
