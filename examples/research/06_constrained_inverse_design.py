"""Research example 6: bounded thickness/doping inverse design.

Question: how far can a derivative-free search move a two-parameter design
(absorber thickness, log-doping) inside explicit bounds? The trajectory panel
shows the design path, the right panel the efficiency history. (For
gradient-based design see examples 12/13, which use the exact IFT adjoint;
this script demonstrates the black-box baseline that gradients beat.)
"""

import numpy as np

import driftjax as dj
from driftjax.viz import style
from examples.support import (
    DOUBLE_WIDE,
    ex1_material,
    example_args,
    execution_metadata,
    new_figure,
    report,
    report_fom,
    save_figure,
    save_json,
)


def main():
    example_args("06_constrained_inverse_design")
    # The full N=500 Nelder-Mead over the design is expensive but canonical:
    # gallery full runs use the production mesh throughout.
    points = 500
    steps = 17
    # deltapv ex1 benchmark material + contact asymmetry (Snr = Spl = 0), so
    # the optimized efficiency lands at the published ~20% benchmark level
    # rather than at the level of a fully-symmetrical-recombination test cell.
    # Material comes from support.py (single source of truth shared with 01/17).
    material = ex1_material()

    def build_device(parameters):
        thickness_um, log_doping = parameters
        t = float(thickness_um) * 1e-4  # um -> cm (library convention)
        return dj.Device(
            n_points=points,
            layers=[
                (t / 2, material, 10 ** float(log_doping)),
                (t / 2, material, -(10 ** float(log_doping))),
            ],
            Snl=1e7,
            Snr=0,
            Spl=0,
            Spr=1e7,
        )

    def objective(parameters):
        return -float(
            dj.simulate(build_device(parameters), dj.Sweep(vmax=0.85, n_steps=steps)).efficiency
        )

    initial = np.array([1.0, 16.0])  # centre of bounds
    history = []  # every objective evaluation: (design, efficiency)

    def tracked(x):
        eff = -objective(x)
        history.append((np.asarray(x).copy(), eff))
        return -eff

    result = dj.optimize.nelder_mead(
        tracked,
        initial,
        bounds=((0.5, 5.0), (15.0, 18.0)),
        maxiter=18,
    )
    baseline = history[0][1]
    optimized = -float(result.fun)

    figure, axes = new_figure(nrows=1, ncols=2, **DOUBLE_WIDE)
    axes[0].plot(
        [row[0][0] for row in history],
        [row[0][1] for row in history],
        "o-",
        color=style.BLUE,
        markersize=4,
        linewidth=1.2,
        alpha=0.8,
    )
    axes[0].plot(
        initial[0], initial[1], "s", color=style.GRAY, markersize=6, label="start", zorder=3
    )
    axes[0].plot(result.x[0], result.x[1], "*", color=style.RED, ms=14, label="optimum", zorder=3)
    # Draw bounds
    axes[0].set_xlim(0.45, 5.05)
    axes[0].set_ylim(14.95, 18.05)
    axes[0].set(xlabel=r"absorber thickness / $\mu$m", ylabel=r"log$_{10}$ doping / cm$^{-3}$")
    axes[0].legend(frameon=False, fontsize=8, handlelength=1.2)
    axes[0].grid(alpha=0.18, linestyle="--")
    axes[0].set_title(
        f"Design trajectory: {len(history)} evals, {baseline * 100:.1f}% → {optimized * 100:.1f}%",
        fontsize=9,
        pad=8,
    )
    axes[1].plot(
        [row[1] * 100 for row in history], "o-", color=style.GREEN, markersize=4, linewidth=1.4
    )
    axes[1].set(xlabel="objective evaluation", ylabel="efficiency / %")
    axes[1].grid(alpha=0.18, linestyle="--")
    axes[1].set_title("Nelder-Mead convergence (derivative-free)", fontsize=9, pad=8)
    figure_path = save_figure(figure, "research_06_constrained_inverse_design")
    save_json(
        "research_06_constrained_inverse_design",
        {
            "metadata": execution_metadata(),
            "success": bool(result.success),
            "message": str(result.message),
            "initial": initial.tolist(),
            "optimized": np.asarray(result.x).tolist(),
            "baseline_efficiency_fraction": float(baseline),
            "optimized_efficiency_fraction": optimized,
            "n_evaluations": len(history),
            "figure": figure_path.name,
        },
    )
    report(
        "research_06",
        baseline_fraction=float(baseline),
        optimized_fraction=optimized,
    )
    # The design's Voc exceeds the example's plotting sweep (vmax=0.85), so a
    # wider sweep is used here purely to capture meaningful Voc/FF for the FoM.
    baseline_sol = dj.simulate(
        build_device(np.asarray(history[0][0])), dj.Sweep(vmax=1.2, n_steps=steps)
    )
    optimized_sol = dj.simulate(
        build_device(np.asarray(result.x)), dj.Sweep(vmax=1.2, n_steps=steps)
    )
    report_fom("research_06", baseline=baseline_sol, optimized=optimized_sol)


if __name__ == "__main__":
    main()
