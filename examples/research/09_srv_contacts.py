"""Research example 09: surface-recombination-velocity sensitivity.

Question: how much open-circuit voltage and fill factor do slow
(front) versus fast contacts cost? Sweeping the front SRV over eight
decades (1e1–1e9 cm/s) on the plain 2 µm Si p-n device shows Voc pinned by bulk
recombination at low SRV and collapsing once surface recombination
dominates — the standard contact-quality diagnostic.
"""

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
    save_figure,
    save_json,
    si_material,
    solution_metrics,
    tag_panels,
)


def main():
    example_args("09_srv_contacts")
    points = 500
    steps = 21
    snls = [1e1, 1e2, 1e3, 1e5, 1e7, 1e9]

    mat = si_material()
    rows, sols = [], {}
    for snl in snls:
        dev = dj.Device(
            layers=[(1e-4, mat, 1e17), (1e-4, mat, -1e17)],
            n_points=points,
            Snl=snl,
            Snr=1e7,
            Spl=1e7,
            Spr=1e7,
        )
        sol = dj.simulate(dev, dj.Sweep(vmax=1.1, n_steps=steps))
        rows.append({"snl": snl, **solution_metrics(sol)})
        sols[snl] = sol

    fig, axes = new_figure(nrows=1, ncols=2, width=9.0, height=3.8)
    ax0, ax1 = axes.ravel()
    xs = np.array([r["snl"] for r in rows])
    for key, label, color in [
        ("voc_V", "$V_{oc}$ / V", style.SERIES[0]),
        ("ff", "FF", style.SERIES[1]),
    ]:
        ax0.semilogx(xs, [r[key] for r in rows], "o-", color=color, label=label, ms=4)
    ax0.set(xlabel="front SRV $S_{nl}$ / cm s$^{-1}$", ylabel="$V_{oc}$ / V, FF")
    ax0.legend(frameon=False, fontsize=8)
    ax0.grid(alpha=0.2, which="both", linestyle="--")
    for snl in snls:
        sol = sols[snl]
        plot_smooth_iv(ax1, sol.voltages, sol.current, color=None, label=f"{snl:.0e}")
    ax1.set(xlabel=style.LBL_BIAS, ylabel=style.LBL_J)
    ax1.legend(frameon=False, fontsize=7, title="Snl")
    ax1.grid(alpha=0.2, linestyle="--")
    tag_panels(axes.ravel())
    fp = save_figure(fig, "research_09_srv_contacts")
    save_json(
        "research_09_srv_contacts",
        {
            "metadata": execution_metadata(),
            "n_points": points,
            "rows": rows,
            "figure": fp.name,
        },
    )
    last = rows[-1]
    report(
        "research_09",
        n_srvs=len(rows),
        voc_fast_V=last["voc_V"],
        eff_fast_pct=last["efficiency_fraction"] * 100,
    )
    report_fom("research_09", **{f"snl_{r['snl']:.0e}": sols[r["snl"]] for r in rows})


if __name__ == "__main__":
    main()
