"""Research example 7: current matching in an ideal two-terminal tandem.

Question: how does bottom-cell thickness tune the Jsc mismatch — and the
ideal two-terminal efficiency — of a perovskite/silicon tandem?  The left
panel overlays the sub-cell IV curves; the right panel tracks the mismatch
and the ideal tandem efficiency (``series_two_terminal``, lossless
recombination junction) as the bottom absorber is thinned.  Scope note: the
database Si carries tn=tp=10 ns and the model has no ARC/texturing, so both
sub-cell and tandem efficiencies sit well below record cells; the study
shows the current-matching *mechanism*, not a record design.
"""

import numpy as np

import driftjax as dj
from driftjax.viz import style
from examples.support import (
    DOUBLE_WIDE,
    example_args,
    execution_metadata,
    mark_iv_points,
    new_figure,
    plot_smooth_iv,
    report,
    report_fom,
    save_figure,
    save_json,
    solution_metrics,
    tag_panels,
)


def main():
    example_args("07_tandem_current_matching")
    points = 500
    steps = 61
    top_material = dj.load_material("MAPbI3")
    # Realistic perovskite top cell: p-i-n with a 1.2 um intrinsic layer
    # (heavily doped 2 x 250 nm stacks cannot collect; Jsc ~ 8 mA).
    top = dj.Device(
        n_points=points,
        layers=[
            (5e-6, top_material, 1e17),
            (1.2e-4, top_material, 1e15),
            (5e-6, top_material, -1e17),
        ],
        Snl=1e7,
        Snr=1e7,
        Spl=1e7,
        Spr=1e7,
    )
    bottom_material = dj.load_material("Si")
    thicknesses = np.linspace(0.5e-4, 4e-4, 6)

    def si_cell(thickness):
        return dj.Device(
            n_points=points,
            layers=[
                (thickness / 2, bottom_material, 1e17),
                (thickness / 2, bottom_material, -1e17),
            ],
            Snl=1e7,
            Snr=1e7,
            Spl=1e7,
            Spr=1e7,
        )

    top_solution = dj.simulate(top, dj.Sweep(vmax=1.3, n_steps=steps))
    bottom_solutions = {
        float(thickness): dj.simulate(si_cell(thickness), dj.Sweep(vmax=1.3, n_steps=steps))
        for thickness in thicknesses
    }
    bottom_jsc = np.asarray([s.jsc for s in bottom_solutions.values()])
    mismatch = bottom_jsc - float(top_solution.jsc)

    total_voltage = np.linspace(0.0, 2.0, 41)
    tandem_results = [
        dj.series_two_terminal(
            (top_solution.voltages, top_solution.current),
            (bottom_solutions[float(t)].voltages, bottom_solutions[float(t)].current),
            v_out=total_voltage,
        )
        for t in thicknesses
    ]
    tandem_pmax = np.asarray([float(r["pmax"]) for r in tandem_results])  # W/cm^2
    # Use the full AM1.5G incident power (1000 W/m^2 = 100 mW/cm^2),
    # not the filtered top-cell P_in which excludes absorbed wavelengths.
    pin_wm2 = 1000.0
    # ideal two-terminal tandem efficiency (fraction of AM1.5G input power)
    tandem_eta = tandem_pmax * 1e4 / pin_wm2
    best = int(np.argmax(tandem_eta))
    t_best = float(thicknesses[best])
    best_bottom = bottom_solutions[t_best]

    figure, axes = new_figure(nrows=1, ncols=2, **DOUBLE_WIDE)
    plot_smooth_iv(
        axes[0],
        top_solution.voltages,
        top_solution.current,
        color=style.BLUE,
        label="top (perovskite)",
    )
    plot_smooth_iv(
        axes[0],
        best_bottom.voltages,
        best_bottom.current,
        color=style.RED,
        label=rf"bottom (Si, $t_{{bot}}$ = {t_best * 1e6:.1f} $\mu$m)",
    )
    axes[0].set(xlabel=style.LBL_BIAS, ylabel=style.LBL_J)
    axes[0].set_xlim(0, 1.2)
    jsc_vals = [abs(float(top_solution.jsc) * 1e3), abs(float(best_bottom.jsc) * 1e3)]
    y_top = max(jsc_vals)
    axes[0].set_ylim(0, y_top * 1.15)
    axes[0].axhline(0, color="black", linewidth=0.5, alpha=0.6)
    axes[0].legend(frameon=False, fontsize=7, ncol=1, handlelength=1.2, loc="upper right")
    axes[0].grid(alpha=0.18, linestyle="--")
    axes[0].set_title("Tandem sub-cells: $J$--$V$", fontsize=9, pad=8)
    mark_iv_points(axes[0], top_solution, show=("mpp",), annotate=False)
    # Overlay: full FoM for both top and bottom (n, Voc, Jsc, FF, MPP) + best ideal tandem.
    # NOTE: Solution.pmax is dimensionless; the physical MPP power density is
    # reconstructed as Voc * Jsc * FF (W cm^-2 -> mW cm^-2), matching
    # support.full_fom (which deliberately does not report pmax directly).
    mpp_top_mW = (
        float(top_solution.voc) * abs(float(top_solution.jsc)) * float(top_solution.ff) * 1e3
    )
    mpp_bot_mW = float(best_bottom.voc) * abs(float(best_bottom.jsc)) * float(best_bottom.ff) * 1e3
    fom_top = rf"top: n={int(points)}  $V_{{oc}}$={float(top_solution.voc):.3f}V  $J_{{sc}}$={float(top_solution.jsc) * 1e3:.1f}  FF={float(top_solution.ff):.3f}  MPP={mpp_top_mW:.2f} mW/cm$^2$  $\eta$={float(top_solution.efficiency) * 100:.2f}%"
    fom_bot = rf"bot: n={int(points)}  $V_{{oc}}$={float(best_bottom.voc):.3f}V  $J_{{sc}}$={float(best_bottom.jsc) * 1e3:.1f}  FF={float(best_bottom.ff):.3f}  MPP={mpp_bot_mW:.2f} mW/cm$^2$  $\eta$={float(best_bottom.efficiency) * 100:.2f}%"
    axes[0].text(
        0.02,
        0.02,
        fom_top + "\n" + fom_bot,
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        color="0.15",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.88),
    )
    # Original compare_metrics_box removed - replaced by detailed n/Voc/Jsc/FF/MPP overlay above
    axes[1].plot(
        thicknesses * 1e6, mismatch * 1e3, "o-", color=style.BLUE, label="J$_{sc}$ mismatch"
    )
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set(xlabel="bottom-cell thickness / $\\mu$m", ylabel="J$_{sc}$ mismatch / mA cm$^{-2}$")
    power_axis = axes[1].twinx()
    power_axis.plot(
        thicknesses * 1e6, tandem_eta * 100, "s--", color=style.RED, label=r"ideal tandem $\eta$"
    )
    power_axis.set_ylabel(r"ideal 2-terminal tandem $\eta$ / %", color=style.RED)
    axes[1].legend(loc="upper left")
    power_axis.legend(loc="lower right")
    # figure of merit: top cell alone vs the best current-matched tandem
    axes[0].text(
        0.02,
        0.98,
        rf"top cell alone: $\eta$ = {float(top_solution.efficiency) * 100:.2f}%" + "\n"
        rf"best ideal tandem: $\eta$ = {tandem_eta[best] * 100:.2f}% "
        rf"($t_{{bot}}$ = {thicknesses[best] * 1e6:.1f} $\mu$m)",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="0.15",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.88),
    )
    tag_panels(axes)
    figure_path = save_figure(figure, "research_07_tandem_current_matching")
    save_json(
        "research_07_tandem_current_matching",
        {
            "metadata": execution_metadata(),
            "top_cell": solution_metrics(top_solution),
            "bottom_thickness_m": thicknesses.tolist(),
            "bottom_jsc_A_per_cm2": bottom_jsc.tolist(),
            "bottom_metrics": {
                f"{float(t) * 1e6:.2f}um": solution_metrics(sol)
                for t, sol in bottom_solutions.items()
            },
            "mismatch_A_per_cm2": mismatch.tolist(),
            "tandem_pmax_W_per_cm2": tandem_pmax.tolist(),
            "tandem_eta_fraction": tandem_eta.tolist(),
            "tandem_voc_V": [float(r["voc"]) for r in tandem_results],
            "figure": figure_path.name,
        },
    )
    # Full figure of merit for BOTH sub-cells (top perovskite AND bottom Si),
    # not just the top cell: efficiency/Voc/Jsc/FF/pmax for each, plus the
    # ideal two-terminal tandem efficiency.
    report_fom("research_07", top=top_solution, bottom=best_bottom)
    report(
        "research_07",
        min_abs_mismatch_mA=float(np.min(np.abs(mismatch))) * 1e3,
        best_tandem_eta_percent=float(tandem_eta[best]) * 100,
    )


if __name__ == "__main__":
    main()
