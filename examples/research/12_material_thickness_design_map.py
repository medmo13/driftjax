"""Bounded bandgap/thickness co-design with an explicit feasibility map."""

import jax
import jax.numpy as jnp
import numpy as np

import driftjax as dj
from driftjax.viz import style
from examples.support import (
    example_args,
    execution_metadata,
    new_figure,
    report,
    report_fom,
    save_figure,
    save_json,
)

BOUNDS = ((1.20, 1.70), (0.30, 2.50))


def device_from(x, points, window):
    absorber = dj.material(Chi=3.9, Eg=x[0], eps=12.0, Nc=5e17, Nv=5e18, mn=1000., mp=300., tn=1e-6, tp=1e-6, A=3e4)
    return dj.Device(n_points=points, layers=[(5e-6, window, 1e18), (x[1] * 1e-4, absorber, -1e16)],
                     Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7, alpha_mode="tauc")


def main():
    example_args("12_material_thickness_design_map")
    # Production mesh N=500 (gallery standard).
    points = 500
    steps = 41
    # Constant electron-transport layer: identical across all probe designs.
    # Passed explicitly (no module-global mutation).
    window = dj.material(Chi=3.5, Eg=2.4, eps=10.0, Nc=2e18, Nv=2e18, mn=100., mp=100., tn=1e-7, tp=1e-7, A=1e4)
    def objective(x):
        return -dj.simulate(device_from(x, points, window), dj.Sweep(vmax=1.4, n_steps=steps),
                            optics=dj.BeerLambert("tauc")).efficiency
    eg = np.linspace(*BOUNDS[0], 11)
    thickness = np.linspace(*BOUNDS[1], 11)
    # Vectorized feasibility map: every (Eg, W) design has identical array
    # shapes. A 121-way vmap of N=500 Newton sweeps compiles one monstrous
    # XLA program (OOM on small machines); serial jitted singles compile
    # once (~1 min) then run ~1 s each instead.
    eg_g, th_g = jnp.meshgrid(jnp.asarray(eg), jnp.asarray(thickness))  # (nT, nE)
    designs = jnp.stack([eg_g.ravel(), th_g.ravel()], axis=-1)
    eff_1 = jax.jit(lambda d: dj.simulate(device_from(d, points, window),
                          dj.Sweep(vmax=1.4, n_steps=steps),
                          optics=dj.BeerLambert("tauc")).efficiency)
    values = np.asarray([eff_1(d) for d in designs]).reshape(len(thickness), len(eg)) * 100
    x0 = np.array([1.25, 0.6])
    opt_history = []  # PCE per SLSQP iteration (IFT-adjoint line search evals)
    def value_grad(x):
        value, grad = jax.value_and_grad(objective)(jnp.asarray(x))
        opt_history.append(-float(value) * 100)
        return float(value), np.asarray(grad)
    result = dj.optimize.slsqp(value_grad, x0, bounds=BOUNDS, maxiter=14)
    x_opt = np.asarray(result.x)
    pce_opt = -float(result.fun) * 100
    fig, axes = new_figure(nrows=1, ncols=3, width=12.9, height=3.8)
    image = axes[0].pcolormesh(eg, thickness, values, shading="auto", cmap=style.CMAP)
    axes[0].plot(*x0, marker="s", color="white", mec="black", ls="none", label="start")
    axes[0].plot(*x_opt, marker="*", color=style.RED, mec="black", ms=14, ls="none", label="optimized")
    axes[0].set(xlabel="absorber bandgap / eV", ylabel=r"absorber thickness / $\mu$m")
    axes[0].legend(frameon=False, fontsize=8, handlelength=1.2, loc="upper right")
    axes[0].grid(alpha=0.12, linestyle="--")
    axes[0].set_title(r"Feasibility map: $\eta$(E$_g$, $W$) with SLSQP start/end", fontsize=9, pad=8)
    cbar = fig.colorbar(image, ax=axes[0], label="efficiency / %")
    cbar.ax.tick_params(labelsize=7)
    # Zoomed bar: show gain
    eff_start = -float(objective(x0)) * 100
    axes[1].bar(["start", "optimized"], [eff_start, pce_opt], color=[style.GRAY, style.BLUE], edgecolor="black", linewidth=0.5, width=0.55)
    axes[1].set_ylim(0, max(eff_start, pce_opt) * 1.18)
    for bar, val in zip(axes[1].patches, [eff_start, pce_opt], strict=True):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4, f"{val:.1f}%", ha="center", fontsize=7)
    axes[1].set(xlabel="design", ylabel="efficiency / %")
    axes[1].grid(axis="y", alpha=0.18, linestyle="--")
    axes[1].set_title(f"Gain: {eff_start:.1f}% → {pce_opt:.1f}% (+{pce_opt-eff_start:.1f}pp)", fontsize=9, pad=8)
    # SLSQP convergence (IFT-adjoint evaluations)
    axes[2].plot(np.arange(1, len(opt_history) + 1), opt_history, "o-",
                 color=style.BLUE, markersize=4, linewidth=1.4)
    axes[2].set(xlabel="SLSQP evaluation", ylabel="PCE / %")
    axes[2].grid(alpha=0.18, linestyle="--")
    axes[2].set_title(f"Adjoint-driven optimization: {len(opt_history)} evals", fontsize=9, pad=8)
    figure = save_figure(fig, "research_12_material_thickness_design_map")
    save_json("research_12_material_thickness_design_map", {
        "metadata": execution_metadata(), "bandgaps_eV": eg.tolist(),
        "thickness_um": thickness.tolist(), "efficiency_percent": values.tolist(),
        "start": x0.tolist(), "optimized": x_opt.tolist(), "pce_optimized_percent": pce_opt,
        "success": bool(result.success), "figure": figure.name,
        "opt_history_percent": opt_history,
    })
    report(
        "research_12",
        bandgap_eV=float(x_opt[0]),
        thickness_um=float(x_opt[1]),
        pce_percent=pce_opt,
    )
    best_sol = dj.simulate(device_from(x_opt, points, window), dj.Sweep(vmax=1.4, n_steps=steps), optics=dj.BeerLambert("tauc"))
    report_fom("research_12", optimal=best_sol)


if __name__ == "__main__":
    main()
