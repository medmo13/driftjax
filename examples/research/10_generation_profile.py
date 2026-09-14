"""Research example 10: how deep does the light reach?

Question: for a fixed spectrum, how does the Beer–Lambert generation
profile $G(x)$ change with absorber thickness and band gap? Baking three
absorber thicknesses (and two band gaps) shows the exponential skin
collapsing into uniform bulk generation — the optical argument behind
thin-film vs wafer design choices. No sweep needed: generation is a
baked cell property.
"""

import numpy as np

import driftjax as dj
from driftjax.science.spectrum import spectrum
from driftjax.simulator import init_cell
from driftjax.units import length
from driftjax.viz import style
from examples.support import (
    example_args,
    execution_metadata,
    new_figure,
    report,
    save_figure,
    save_json,
    tag_panels,
)


def _generation(thickness_m, eg_ev, n_points=500):
    mat = dj.material(Chi=4.05, Eg=eg_ev, eps=11.7, Nc=2.8e19, Nv=1.04e19,
                      mn=1400, mp=450, tn=1e-6, tp=1e-6, A=1e4)
    dev = dj.Device(
        layers=[(thickness_m, mat, 1e16)],
        n_points=n_points, Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7,
    )
    cell = init_cell(dev.design(), spectrum(normalize=False))
    x_um = np.asarray(cell.x) * float(length) * 1e4
    return x_um, np.asarray(cell.G)


def main():
    example_args("10_generation_profile")
    thicknesses = [20e-6, 100e-6, 500e-6]
    gaps = [1.1, 1.5]

    fig, axes = new_figure(nrows=1, ncols=2, width=9.0, height=3.8)
    ax0, ax1 = axes.ravel()
    summary = {}
    for i, eg in enumerate(gaps):
        for th in thicknesses:
            x_um, g = _generation(th, eg)
            label = f"{th * 1e6:.0f} µm, Eg={eg} eV"
            (ax0 if i == 0 else ax1).semilogy(
                x_um, g, lw=1.6, label=label,
                color=style.SERIES[list(thicknesses).index(th) % len(style.SERIES)])
            summary[label] = {"g_mean": float(np.mean(g)),
                              "g_max": float(np.max(g)),
                              "depth_90pct_um": float(x_um[np.argmin(
                                  np.abs(np.cumsum(g) / np.sum(g) - 0.9))])}
    for ax, eg in zip([ax0, ax1], gaps, strict=True):
        ax.set(xlabel="position / µm", ylabel="generation (dimensionless)",
               title=f"Eg = {eg} eV")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.2, which="both", linestyle="--")
    tag_panels(axes.ravel())
    fp = save_figure(fig, "research_10_generation_profile")
    save_json("research_10_generation_profile", {
        "metadata": execution_metadata(),
        "summary": summary, "figure": fp.name,
    })
    thin_key = f"{thicknesses[0] * 1e6:.0f} µm, Eg=1.1 eV"
    report("research_10", n_curves=len(summary),
           gmax_thin=float(summary[thin_key]["g_max"]))


if __name__ == "__main__":
    main()
