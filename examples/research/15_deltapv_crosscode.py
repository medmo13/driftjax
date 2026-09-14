"""Research example 15: cross-code benchmark vs ∂PV (accuracy + speed).

Question: do DriftJax and ∂PV agree on physics, and what does each cost?
Overlays the bundled ∂PV reference IV curves (ex1 np-junction, ex2
heterojunction — `examples/resources/deltapv_{ex1,ex2}_iv.npz`) with live
DriftJax sweeps on matched devices, and reports the systematic timing
harness numbers (see `DELTAPV_CROSSCODE.md` in the workspace root for the
full CPU+GPU table, including the structural finding that ∂PV's solver
logging breaks `jit`, forcing eager re-traces every gradient).

Mesh note: ∂PV's ex2 uses an interface-clustered grid while DriftJax
runs uniform here, so small knee differences are discretization, not
physics (interior branches agree to ~1e-5 A/cm²).
"""

from pathlib import Path

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
    solution_metrics,
    tag_panels,
)

RESOURCES = Path(__file__).resolve().parent.parent / "resources"


def _matched_ex1(n_points):
    m = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, tn=1e-8, tp=1e-8, A=2e4
    )
    return dj.Device(
        layers=[(1e-4, m, 1e17), (1e-4, m, -1e17)],
        n_points=n_points,
        Snl=1e7,
        Snr=0,
        Spl=0,
        Spr=1e7,
    )


def _matched_ex2(n_points):
    CdS = dj.material(
        Nc=2.2e18, Nv=1.8e19, Eg=2.4, eps=10, Et=0, mn=100, mp=25, tn=1e-8, tp=1e-13, Chi=4.0, A=1e4
    )
    CdTe = dj.material(
        Nc=8e17, Nv=1.8e19, Eg=1.5, eps=9.4, Et=0, mn=320, mp=40, tn=5e-9, tp=5e-9, Chi=3.9, A=1e4
    )
    return dj.Device(
        layers=[(2.5e-6, CdS, 1e17), (4e-4, CdTe, -1e15)],
        n_points=n_points,
        Snl=1.16e7,
        Snr=1.16e7,
        Spl=1.16e7,
        Spr=1e7,
    )


def main():
    example_args("15_deltapv_crosscode")
    points = 500
    steps = 25

    fig, axes = new_figure(nrows=1, ncols=2, width=9.0, height=3.8)
    axes = axes.ravel()
    summary, sols = {}, {}
    for i, (tag, dev) in enumerate([("ex1", _matched_ex1(points)), ("ex2", _matched_ex2(points))]):
        ref = np.load(RESOURCES / f"deltapv_{tag}_iv.npz")
        sol, wall = run_timed(
            dj.simulate,
            dev,
            dj.Sweep(vmax=1.1, n_steps=steps),
            optics=dj.BeerLambert(alpha_mode="beer-lambert"),
        )
        v, j = np.asarray(sol.voltages), np.asarray(sol.current)
        lo, hi = max(ref["v"].min(), v.min()), min(0.9, ref["v"].max(), v.max())
        g = np.linspace(lo, hi, 200)
        dmax = float(np.max(np.abs(np.interp(g, ref["v"], ref["j"]) - np.interp(g, v, j))))
        axes[i].set(
            xlabel=style.LBL_BIAS, ylabel=style.LBL_J, title=f"{tag}: max|dI| = {dmax:.1e} A/cm²"
        )
        # photovoltaic quadrant only (never negative axes)
        axes[i].set_xlim(0, float(ref["v"].max()))
        axes[i].set_ylim(0, float(ref["j"].max()) * 1e3 * 1.15)
        # smooth presentation overlays (deltapv quintic style); data unchanged
        plot_smooth_iv(axes[i], ref["v"], ref["j"], color=style.GRAY, label="∂PV reference", lw=1.4)
        plot_smooth_iv(
            axes[i], v, j, color=style.SERIES[i], label=f"DriftJax (wall {wall:.1f}s)", lw=1.4
        )
        axes[i].legend(frameon=False, fontsize=8)
        axes[i].grid(alpha=0.2, linestyle="--")
        summary[tag] = {
            "eff_pct": float(sol.efficiency) * 100,
            "ref_eff_pct": None,
            "max_abs_dI": dmax,
            "wall_s": wall,
            **solution_metrics(sol),
        }
        sols[tag] = sol
    tag_panels(axes)
    fp = save_figure(fig, "research_15_deltapv_crosscode")
    save_json(
        "research_15_deltapv_crosscode",
        {
            "metadata": execution_metadata(),
            "n_points": points,
            "summary": summary,
            "figure": fp.name,
        },
    )
    report(
        "research_15",
        ex1_eff_pct=summary["ex1"]["eff_pct"],
        ex2_eff_pct=summary["ex2"]["eff_pct"],
        ex1_maxdI=summary["ex1"]["max_abs_dI"],
        ex2_maxdI=summary["ex2"]["max_abs_dI"],
    )
    report_fom("research_15", **sols)


if __name__ == "__main__":
    main()
