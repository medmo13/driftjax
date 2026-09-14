"""Research example 2: band alignment and transport in a CdS/CdTe junction.

Question: how does the conduction-band offset at a heterointerface shape the
device response? The left panel shows the equilibrium band diagram (the
cliff at the CdS/absorber interface), the right panel the illuminated IV curve.
"""

import numpy as np

import driftjax as dj
from driftjax.units import energy, length
from driftjax.viz import style
from examples.support import (
    as_list,
    ex2_device,
    example_args,
    execution_metadata,
    iv_metrics_box,
    mark_iv_points,
    new_figure,
    plot_bands,
    plot_charges,
    report_fom,
    run_timed,
    save_figure,
    save_json,
    solution_metrics,
    tag_panels,
)


def main():
    example_args("02_heterojunction_physics")
    # deltapv ex2_np_hetero benchmark geometry (published PCE 13.31%):
    # CdS 25 nm window (1e17) / CdTe 4 um absorber (-1e15), Sn = 1.16e7 cm/s.
    # Builder lives in support.py (single source of truth shared with 17).
    device = ex2_device(n_points= 500)
    solution, wall = run_timed(
        dj.simulate, device, dj.Sweep(vmax=1.1, n_steps= 61),
        optics=dj.BeerLambert(alpha_mode="beer-lambert"),
    )
    position_um = np.asarray(solution.cell.x) * float(length) * 1e4  # x*length is cm -> um
    conduction = -(np.asarray(solution.cell.Chi) + np.asarray(solution.eq_pot.phi)) * float(energy)
    valence = conduction - np.asarray(solution.cell.Eg) * float(energy)
    figure, axes = new_figure(nrows=1, ncols=3, width=12.6, height=3.8)
    plot_bands(axes[0], solution, labels=["CdS window", "absorber"])
    plot_charges(axes[1], solution, labels=["CdS window", "absorber"])
    axes[1].set_title("Equilibrium carrier densities", fontsize=9, pad=10)
    axes[2].plot(solution.voltages, solution.current * 1e3, color=style.BLUE, linewidth=1.6)
    axes[2].set(xlabel=style.LBL_BIAS, ylabel=style.LBL_J)
    axes[2].set_xlim(0, 1.1)
    jsc = abs(float(solution.jsc)*1e3)
    axes[2].set_ylim(0, jsc*1.15)
    iv_metrics_box(axes[2], solution)
    axes[1].axhline(0, color="black", linewidth=0.5, alpha=0.6)
    axes[2].axvline(float(solution.voc), color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
    axes[2].grid(alpha=0.18, linestyle="--")
    axes[2].set_title(f"Illuminated $J$--$V$: $J_{{sc}}$={jsc:.1f} mA/cm$^2$, $V_{{oc}}$={float(solution.voc):.2f} V, PCE={float(solution.efficiency)*100:.1f}%", fontsize=8, pad=8)
    axes[0].grid(alpha=0.18, linestyle="--")
    axes[0].set_title("CdS/CdTe (deltapv ex2) band alignment", fontsize=9, pad=10)
    mark_iv_points(axes[2], solution, annotate=False)
    tag_panels(axes)
    figure_path = save_figure(figure, "research_02_heterojunction_physics")
    save_json("research_02_heterojunction_physics", {
        "metadata": execution_metadata(),
        "metrics": solution_metrics(solution),
        "wall_s": wall,
        "position_um": as_list(position_um),
        "conduction_band_eV": as_list(conduction),
        "valence_band_eV": as_list(valence),
        "figure": figure_path.name,
    })
    # Full figure of merit for the simulated heterojunction cell.
    report_fom("research_02", cell=solution)


if __name__ == "__main__":
    main()
