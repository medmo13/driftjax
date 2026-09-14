"""Research example 16: graded vs abrupt doping profiles.

Question: does grading the junction (5-step graded doping instead of an
abrupt step) change the built-in field and the open-circuit voltage?
Comparing a 5-layer linear ramp against the abrupt p-n step on the same
total thickness shows how the depletion field spreads — the physical
argument behind graded emitters.
"""

import jax.numpy as jnp
import numpy as np

import driftjax as dj
from driftjax.viz import style
from examples.support import (
    example_args,
    execution_metadata,
    new_figure,
    plot_smooth_iv,
    report,
    report_fom,
    run_timed,
    save_figure,
    save_json,
    si_material,
    solution_metrics,
    tag_panels,
)


def main():
    example_args("16_graded_doping")
    points = 500
    steps = 21

    mat = si_material()
    abrupt = dj.Device(
        layers=[(1e-4, mat, 1e17), (1e-4, mat, -1e17)],
        n_points=points,
        Snl=1e7,
        Snr=1e7,
        Spl=1e7,
        Spr=1e7,
    )
    ramp = [1e17, 5e16, 0.0, -5e16, -1e17]
    graded = dj.Device(
        layers=[(4e-5, mat, d) for d in ramp], n_points=points, Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7
    )

    sols = {}
    for name, dev in (("abrupt", abrupt), ("graded", graded)):
        sol, wall = run_timed(dj.simulate, dev, dj.Sweep(vmax=1.1, n_steps=steps))
        sols[name] = (sol, wall)

    fig, axes = new_figure(nrows=1, ncols=2, width=9.0, height=3.8)
    ax0, ax1 = axes.ravel()
    for (name, (sol, _)), color in zip(sols.items(), style.SERIES, strict=False):
        plot_smooth_iv(
            ax0,
            np.asarray(sol.voltages),
            np.asarray(sol.current),
            color=color,
            label=f"{name} ({float(sol.efficiency) * 100:.2f}%)",
        )
        phi = np.asarray(sol.eq_pot.phi)
        fld = np.abs(np.diff(phi))
        x = np.linspace(0, 2.0, len(fld))
        ax1.semilogy(x, fld / max(float(jnp.max(fld)), 1e-30), lw=1.6, color=color, label=name)
    # photovoltaic quadrant only: the ideal-diode forward tail (no series
    # resistance in the model) would otherwise compress the PV quadrant flat
    j_top = max(float(np.asarray(sol.current)[0]) * 1e3 for sol, _ in sols.values())
    ax0.set(xlabel=style.LBL_BIAS, ylabel=style.LBL_J)
    ax0.set_xlim(0, 1.1)
    ax0.set_ylim(0, j_top * 1.15)
    ax0.axhline(0, color="black", linewidth=0.5, alpha=0.6)
    ax0.legend(frameon=False, fontsize=8)
    ax0.grid(alpha=0.2, linestyle="--")
    ax1.set(xlabel="position / µm", ylabel="normalized |field|")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.2, which="both", linestyle="--")
    tag_panels(axes.ravel())
    fp = save_figure(fig, "research_16_graded_doping")
    save_json(
        "research_16_graded_doping",
        {
            "metadata": execution_metadata(),
            "n_points": points,
            **{f"{n}_{k}": v for n, (s, _) in sols.items() for k, v in solution_metrics(s).items()},
            "figure": fp.name,
        },
    )
    report(
        "research_16",
        abrupt_eff=float(sols["abrupt"][0].efficiency),
        graded_eff=float(sols["graded"][0].efficiency),
    )
    report_fom("research_16", abrupt=sols["abrupt"][0], graded=sols["graded"][0])


if __name__ == "__main__":
    main()
