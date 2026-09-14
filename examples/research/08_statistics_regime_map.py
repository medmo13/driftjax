"""Compare carrier-statistics assumptions and their device-level consequence.

The three modes share the same transport model. The first panel exposes where
the approximation changes; the second measures the resulting IV metrics.
"""

import numpy as np

import driftjax as dj
from driftjax.science.carrier_statistics import F_half
from driftjax.viz import style
from examples.support import (
    DOUBLE_WIDE,
    as_list,
    compare_metrics_box,
    example_args,
    execution_metadata,
    new_figure,
    pn_device,
    report_fom,
    save_figure,
    save_json,
    si_material,
    solution_metrics,
)


def main():
    example_args("08_statistics_regime_map")
    # F_half computation is cheap; the full N=500 sweep runs the carrier
    # statistics comparison + IV sweep at the production mesh.
    eta = np.linspace(-4.0, 6.0, 240)
    modes = ("boltzmann", "blakemore", "exact")
    fig, axes = new_figure(nrows=1, ncols=2, **DOUBLE_WIDE)
    for idx, mode in enumerate(modes):
        axes[0].semilogy(
            eta, np.asarray(F_half(eta, mode)), label=mode,
            color=style.SERIES[idx],
        )
    axes[0].set(xlabel=r"reduced Fermi energy $\eta$", ylabel=r"$F_{1/2}(\eta)$")
    axes[0].legend()

    material = si_material()
    points = 500
    steps = 41
    device = pn_device(material, points=points)
    rows = []
    solutions = {}
    for mode in modes:
        solution = dj.simulate(device, dj.Sweep(vmax=0.9, n_steps=steps), statistics=mode)
        solutions[mode] = solution
        rows.append({"mode": mode, **solution_metrics(solution)})
        axes[1].plot(solution.voltages, solution.currents * 1e3,
                     label=mode, color=style.SERIES[len(rows) - 1], linewidth=1.6)
    axes[1].set(xlabel=style.LBL_BIAS, ylabel=style.LBL_J)
    axes[1].set_xlim(0, 0.9)
    y_top = max(abs(float(s.jsc)*1e3) for s in solutions.values())
    axes[1].set_ylim(0, y_top*1.15)
    axes[1].axhline(0, color="black", linewidth=0.5, alpha=0.6)
    axes[1].legend(frameon=False, fontsize=8, handlelength=1.2)
    compare_metrics_box(axes[1], solutions, loc="lower left", fontsize=6.5)
    axes[1].grid(alpha=0.18, linestyle="--")
    axes[1].set_title("IV: statistics impact on device performance", fontsize=9, pad=8)
    figure = save_figure(fig, "research_08_statistics_regime_map")
    save_json("research_08_statistics_regime_map", {
        "metadata": execution_metadata(), "eta": as_list(eta), "results": rows,
        "figure": figure.name,
    })
    # Full figure of merit for EVERY statistics mode (Boltzmann / Blakemore / exact).
    report_fom("research_08", **solutions)


if __name__ == "__main__":
    main()
