"""Research example 1: mesh convergence of a silicon-like p-n device.

Question: how much spatial resolution is needed before the reported IV metrics
stop changing materially? The result is a convergence study, not just a plot:
the panel shows IV curves at five meshes plus the relative efficiency error
against the finest grid; the observed order is non-asymptotic (see manuscript).
"""

import driftjax as dj
from driftjax.viz import style
from examples.support import (
    compare_metrics_box,
    ex1_device,
    example_args,
    execution_metadata,
    iv_knee_inset,
    new_figure,
    plot_bands,
    plot_charges,
    plot_smooth_iv,
    report,
    report_fom,
    run_timed,
    save_figure,
    save_json,
    slope_guide,
    solution_metrics,
    tag_panels,
)


def main():
    args = example_args("01_device_iv")
    resolutions = (62, 125, 250, 500, 1000)
    results = []
    solutions = []
    for points in resolutions:
        solution, wall = run_timed(
            dj.simulate,
            ex1_device(points),
            dj.Sweep(vmax=1.1, n_steps=61),
            solver=dj.Newton(backend=args.backend),
        )
        results.append({"n_points": points, "wall_s": wall, **solution_metrics(solution)})
        solutions.append(solution)

    reference = results[-1]["efficiency_fraction"]
    for row in results:
        row["relative_efficiency_error"] = abs(row["efficiency_fraction"] - reference) / reference

    figure, axes = new_figure(nrows=2, ncols=2, width=9.0, height=6.4)
    axes = axes.ravel()  # 2x2 grid -> flat [IV, convergence, bands, charges]
    for points, solution in zip(resolutions, solutions, strict=True):
        plot_smooth_iv(
            axes[0],
            solution.voltages,
            solution.current,
            color=style.SERIES[resolutions.index(points) % len(style.SERIES)],
            label=f"N = {points}",
        )
    axes[0].set(xlabel=style.LBL_BIAS, ylabel=style.LBL_J)
    axes[0].set_xlim(0, 1.1)
    # Tight y-range from the plotted data (no hardcoded floors)
    j_max = max(float(s.current[0]) * 1e3 for s in solutions)
    axes[0].set_ylim(0, 1.15 * j_max)
    axes[0].axhline(0, color="black", linewidth=0.5, alpha=0.6)
    compare_metrics_box(
        axes[0],
        {f"N = {p}": sol for p, sol in zip(resolutions, solutions, strict=True)},
        loc="lower left",
    )
    axes[0].legend(frameon=False, loc="upper right", fontsize=8, handlelength=1.2)
    axes[0].grid(alpha=0.18, linestyle="--")
    axes[0].set_title("Homojunction $J$--$V$ (mesh convergence)", fontsize=9, pad=8)
    iv_knee_inset(axes[0], solutions[-1])
    # Finest mesh is the reference (error exactly 0): exclude it from the
    # log panel rather than flooring it to a misleading 1e-16 dot.
    _conv = [
        (p, r["relative_efficiency_error"])
        for p, r in zip(resolutions, results, strict=False)
        if r["relative_efficiency_error"] > 0
    ]
    axes[1].loglog(
        [p for p, _ in _conv],
        [e for _, e in _conv],
        "o-",
        color=style.BLUE,
        markersize=5,
        linewidth=1.4,
    )
    axes[1].set(xlabel="number of grid points", ylabel="relative efficiency error")
    axes[1].grid(which="both", alpha=0.18, linestyle="--")
    axes[1].set_title("Observed mesh convergence (reference slope shown)", fontsize=9, pad=8)
    # Reference slope segment: illustrative N^-2 guide only; the observed
    # coupled-solver order is non-asymptotic (see manuscript Section 4.4).
    if len(resolutions) >= 2:
        err = max(results[0]["relative_efficiency_error"], 1e-16)
        slope_guide(
            axes[1],
            resolutions[0],
            err,
            decades=2.0,
            slope=-2.0,
            color=style.GRAY,
            label="reference slope $O(N^{-2})$ --- not a fit",
        )
        # Annotate local observed order p_obs between consecutive mesh pairs
        # using Eq. S5.1: p_obs = log[e(h)/e(h/r)] / log(r)
        for i in range(len(resolutions) - 2):
            n_h = resolutions[i]
            n_r = resolutions[i + 1]
            e_h = max(results[i]["relative_efficiency_error"], 1e-16)
            e_r = max(results[i + 1]["relative_efficiency_error"], 1e-16)
            import math

            p_obs = math.log(e_h / e_r) / math.log(n_r / n_h) if e_r > 0 else float("nan")
            mid_x = (n_h * n_r) ** 0.5
            mid_y = (e_h * e_r) ** 0.5
            axes[1].annotate(
                f"$p_{{\\mathrm{{obs}}}}={p_obs:.1f}$",
                xy=(mid_x, mid_y),
                fontsize=7,
                color=style.BLUE,
                ha="center",
                va="bottom",
            )
        axes[1].legend(frameon=False, loc="lower left", fontsize=8, handlelength=1.2)
    # Dark-equilibrium band diagram and carrier densities at the finest mesh.
    # A separate dark solve (no illumination) ensures thermal equilibrium:
    # np = ni^2 everywhere, consistent with the mass-action law.
    import jax.numpy as jnp

    from driftjax.science.spectrum import LightSource

    dark_ls = LightSource(Lambda=jnp.array([0.5]), P_in=jnp.array([0.0]), kind="sun")
    dark_eq = dj.simulate(ex1_device(resolutions[-1]), dj.Equilibrium(), ls=dark_ls)
    plot_bands(axes[2], dark_eq)
    axes[2].set_title(f"Dark-equilibrium bands (N = {resolutions[-1]})", fontsize=9, pad=8)
    axes[2].legend(frameon=False, fontsize=8, handlelength=1.2, loc="center right")
    plot_charges(axes[3], dark_eq)
    axes[3].set_title("Dark-equilibrium carrier densities", fontsize=9, pad=8)
    tag_panels(axes)
    figure_path = save_figure(figure, "research_01_device_iv")
    save_json(
        "research_01_device_iv",
        {"metadata": execution_metadata(), "results": results, "figure": str(figure_path.name)},
    )
    report(
        "research_01",
        efficiency_fraction=reference,
        rel_error_second_finest=results[-2]["relative_efficiency_error"],
    )

    report_fom("research_01", finest=solutions[-1])


if __name__ == "__main__":
    main()
