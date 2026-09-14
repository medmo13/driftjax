"""Research example 3: optical-model sensitivity in a thin-film cell.

Question: when does coherent interference matter? The same perovskite
p-i-n stack is simulated with incoherent Beer-Lambert absorption and with the
coherent transfer-matrix method; the IV curves and generation profiles show
that for this 400 nm absorber the two optical models bracket only a small
efficiency difference, while the generation profiles differ near the front.
"""

import numpy as np

import driftjax as dj
from driftjax.units import gratedens, length
from driftjax.viz import style
from examples.support import (
    DOUBLE_WIDE,
    compare_metrics_box,
    example_args,
    execution_metadata,
    mark_iv_points,
    new_figure,
    report_fom,
    save_figure,
    save_json,
    shade_layers,
    solution_metrics,
    tag_panels,
)


def main():
    example_args("03_optical_model_comparison")
    absorber = dj.load_material("MAPbI3")
    etl = dj.material(Eg=3.2, Chi=4.0, eps=9, Nc=1e19, Nv=1e19, mn=20, mp=20, tn=1e-7, tp=1e-7, A=0)
    htl = dj.material(Eg=3.0, Chi=2.2, eps=3, Nc=1e19, Nv=1e19, mn=20, mp=20, tn=1e-7, tp=1e-7, A=0)
    device = dj.Device(
        layers=[(5e-6, etl, 1e17), (4e-5, absorber, 0), (5e-6, htl, -1e17)],
        n_points=500,
        Snl=1e7,
        Snr=1e7,
        Spl=1e7,
        Spr=1e7,
    )
    cases = {
        "Beer-Lambert": dj.BeerLambert(alpha_mode="tauc"),
        "TMM": dj.TMM(alpha_mode="table"),
    }
    solutions = {
        name: dj.simulate(device, dj.Sweep(vmax=1.2, n_steps=61), optics=optics)
        for name, optics in cases.items()
    }
    figure, axes = new_figure(nrows=1, ncols=2, **DOUBLE_WIDE)
    for idx, (name, solution) in enumerate(solutions.items()):
        color = style.SERIES[idx]
        axes[0].plot(
            solution.voltages, solution.current * 1e3, label=name, color=color, linewidth=1.6
        )
        # cell.G is dimensionless; G[cm^-3 s^-1] = G_dim * gratedens
        axes[1].plot(
            np.asarray(solution.cell.x) * float(length) * 1e4,
            np.asarray(solution.cell.G) * float(gratedens),
            label=name,
            color=color,
            linewidth=1.5,
        )
    axes[0].set(xlabel=style.LBL_BIAS, ylabel=style.LBL_J)
    axes[0].set_xlim(0, 1.2)
    jsc_vals = [abs(float(s.jsc) * 1e3) for s in solutions.values()]
    y_top = max(jsc_vals)
    axes[0].set_ylim(0, y_top * 1.15)
    compare_metrics_box(axes[0], solutions, loc="lower left")
    axes[0].axhline(0, color="black", linewidth=0.5, alpha=0.6)
    axes[0].legend(frameon=False, fontsize=8, handlelength=1.2)
    axes[0].grid(alpha=0.18, linestyle="--")
    axes[1].set(xlabel="position / $\\mu$m", ylabel="generation $G$ (cm$^{-3}$ s$^{-1}$)")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=8, handlelength=1.2)
    axes[1].grid(which="both", alpha=0.18, linestyle="--")
    axes[1].set_title("Generation profile (log)", fontsize=9, pad=12)
    # Physical context: layer boundaries + key points on the IV curve.
    shade_layers(
        axes[1],
        next(iter(solutions.values())),
        labels=["ETL", "MAPbI$_3$ absorber", "HTL"],
        label_size=7,
    )
    mark_iv_points(axes[0], next(iter(solutions.values())), show=("mpp",), annotate=False)
    axes[0].set_title("Optics: IV (TMM vs Beer–Lambert)", fontsize=9, pad=10)
    tag_panels(axes)
    figure_path = save_figure(figure, "research_03_optical_model_comparison")
    save_json(
        "research_03_optical_model_comparison",
        {
            "metadata": execution_metadata(),
            "metrics": {name: solution_metrics(solution) for name, solution in solutions.items()},
            "figure": figure_path.name,
        },
    )
    # Full figure of merit for BOTH optical models (Beer-Lambert AND TMM),
    # so the CI line carries efficiency/Voc/Jsc/FF/pmax for each, not just efficiency.
    report_fom("research_03", **solutions)


if __name__ == "__main__":
    main()
